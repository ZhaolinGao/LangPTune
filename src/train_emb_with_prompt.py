import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb
import pickle
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
)
from peft import get_peft_model, LoraConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclass
class TaskHParams:
    response_length: int = 1024
    truncate_token_id: Optional[int] = None
    emb_pad_token_id: Optional[int] = None
    max_label_count: Optional[int] = None
    penalty_reward_value: int = 0
    temperature: float = 1.0
    query_dataset: str = "GitBag/amazon_movie_tv_gemma_mxbai_v7"
    item_dataset: str = "GitBag/amazon_movie_tv_item_mxbai_v5"


@dataclass
class Args:
    # common args
    exp_name: str = "gemma_mxbai"
    """the name of this experiment"""
    seed: int = 555134
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_entity: str = "zg292"
    """the wandb's entity name"""
    wandb_project_name: str = "llm4recsys"
    """the wandb's project name"""
    cuda: bool = True
    """Whether to use cuda if available."""
    deepspeed: bool = True
    """Whether to use deepspeed to train the model"""
    offload: bool = True
    """Whether to offload ref model and reward model to CPU"""
    print_sample_output_freq: int = 100
    """How often to print sample output"""
    run_eval: bool = True
    """Whether to run evaluation"""
    prompt: str = "The user has rated (out of 5.0) and reviewed following movies and TV shows arranged chronologically from the oldest (top) to the newest (bottom). Please provide a high-level summary of the user preference in detail."
    """the name of this experiment"""

    # optimizer args
    eps: float = 1e-6
    """the epsilon value for the optimizer"""
    lr: float = 3e-7
    """the learning rate"""
    emb_lr: float = 3e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""

    local_rollout_forward_batch_size: int = 8
    """per rank no grad forward pass in the rollout phase"""
    gradient_accumulation_steps: int = 16
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: int = 1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    emb_per_device_train_batch_size: int = 64
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    per_device_eval_batch_size: int = 4
    """per rank eval batch size"""
    noptepochs: int = 4
    """data reuse schedule"""
    topk: int = 20
    """topk for evaluation"""
    score_batch_size: int = 16
    """batchsize for computing the score"""
    num_gens: int = 2
    """number of generations per session"""
    emb_batch_size: int = 16
    """batch size for embedding generation"""
    loss_temp: float = 0.1
    """temperature of the infonce loss"""
    loss_reduction: str = 'mean'
    """reduction for infonce loss"""
    reward_type: str = 'mrr'
    """[mrr, ndcg]"""
    num_lm_updates: int = 200
    """The number of updates"""
    num_emb_updates: int = 200
    """The number of updates"""
    num_iterations: int = 10
    """The number of iterations"""

    # optional args filled while running
    world_size: Optional[int] = 4
    """The number of processes (GPUs) to use"""
    batch_size: Optional[int] = 512
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_batch_size: Optional[int] = 128
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""

    # other args
    lm_model: str = "google/gemma-2b-it"
    """the name of the pretrained model to use"""
    emb_model: str = "mixedbread-ai/mxbai-embed-large-v1"
    """the name of the pretrained model to use"""
    output_dir_lm: str = "models/gemma-2b-lm"
    """Where to save the model"""
    output_dir_emb: str = "models/mxbai-embed-v1-emb"
    """Where to save the model"""
    lora_rank: int = 1
    """the rank of the lora matrix"""
    lora_alpha: int = 2
    """weight of lora"""
    lora_dropout: float = 0.0
    """dropout for lora"""
    eta: float = 1.0
    """The eta value for rebel"""
    kl_coef: float = 0.05
    """kl coef"""
    whiten_rewards: bool = True
    """whether to whiten rewards"""
    task: TaskHParams = field(default_factory=TaskHParams)


def whiten(values):
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    return whitened


# taken from https://github.com/microsoft/DeepSpeedExamples/blob/737c6740bec38b77a24a59135b6481a53d566b38/applications/DeepSpeed-Chat/training/utils/model/model_utils.py#L20C1-L26C52
def configure_dropout(model_config, dropout_layer_keys, dropout):
    if dropout is not None:
        for key in dropout_layer_keys:
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


def whiten(values, shift_mean=True):
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(args, tokenizer, responses):
    trunc_idxs = first_true_indices(responses == args.task.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [args.task.response_length]
    idxs = torch.arange(args.task.response_length, device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def forward(model, query_responses, tokenizer, ref=False):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    if ref:
        with model.disable_adapter():
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
    else:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )


def get_scores(item_embs, session_embs, bs=16):
    scores = []
    for i in range(0, len(session_embs), bs):
        mb_session_embs = session_embs[i:min(i+bs, len(session_embs))]
        mb_scores = mb_session_embs @ item_embs.T
        mb_scores = mb_scores / (torch.norm(mb_session_embs, dim=-1, keepdim=True) @ torch.norm(item_embs, dim=-1, keepdim=True).T)
        scores.append(mb_scores)
    scores = torch.cat(scores, dim=0)
    return scores


def get_embedding(emb_model, tokens, pad_token_id, device, strategy='cls'):
    inputs = {"input_ids" : tokens, "attention_mask" : tokens != pad_token_id}
    outputs = emb_model(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device)).last_hidden_state
    if strategy == 'cls':
        outputs = outputs[:, 0]
    elif strategy == 'mean':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
    else:
        raise NotImplementedError
    return outputs


def get_reward(emb_model, emb_tokenizer, item_embs, dense_out_sess, dense_in_sess, response, length, reward_type, penalty, device):

    tokenized_response = emb_tokenizer(response, padding='max_length', truncation=True, max_length=length, return_tensors="pt")['input_ids']
    emb_response = get_embedding(emb_model, tokenized_response, emb_tokenizer.pad_token_id, device)

    scores = get_scores(item_embs, emb_response)
    scores = scores - 999999 * dense_in_sess
    topks = torch.argsort(scores, dim=-1, descending=True)

    # compute lables from dense_out_sesses
    index = torch.nonzero(dense_out_sess)
    count = torch.count_nonzero(dense_out_sess, dim=-1)
    label = torch.full((len(dense_out_sess), torch.max(count)), -1).to(device)
    value = index[:, 1]
    new_index = []
    for i in count:
        new_index.append(torch.arange(i))
    new_index = torch.cat(new_index)
    label[index[:, 0], new_index] = value
        
    rewards = compute_result(topks, label, count, [dense_out_sess.shape[-1]], compute_mean=False)

    ndcg = rewards['ndcg'][0]
    mrr = rewards['mrr'][0]
    
    if reward_type == 'ndcg':
        rewards = ndcg
    elif reward_type == 'mrr':
        rewards = mrr

    contain_pad_token = torch.any(tokenized_response == emb_tokenizer.pad_token_id, dim=-1)
    rewards = torch.where(contain_pad_token.to(device), rewards, torch.full_like(rewards, penalty))

    return rewards, ndcg, mrr


def recall_precision_at_k(r, k, counts, compute_mean=True):
    right_pred = r[:, :k].sum(1)
    if compute_mean:
        recall = (right_pred / counts).mean()
        precision = right_pred.mean() / k
    else:
        recall = (right_pred / counts)
        precision = right_pred / k
    return {'recall': recall, 'precision': precision}


def mrr_at_k(r, k, compute_mean=True):
    pred_data = r[:, :k] / torch.arange(1, k+1).unsqueeze(0).to(r.device)
    if compute_mean:
        return pred_data.sum(1).mean()
    else:
        return pred_data.sum(1)


def ndcg_at_k(r, k, counts, compute_mean=True):
    dcg = r[:, :k] / torch.log2(torch.arange(2, k+2).unsqueeze(0).to(r.device))
    idcg = 1 / torch.log2(torch.arange(2, k+2)).unsqueeze(0).to(r.device)

    mask = torch.arange(k).unsqueeze(0).to(r.device)
    mask = (mask < counts.unsqueeze(1)).float()
    idcg = (idcg * mask).sum(1)

    if compute_mean:
        return (dcg.sum(1) / idcg).mean()
    else:
        return dcg.sum(1) / idcg


def compute_result(recommends, test_label, counts, topks, compute_mean=True):
    r = (recommends.unsqueeze(2) == test_label.unsqueeze(1)).float().sum(2)
    recall, pre, mrr, ndcg = [], [], [], []
    for k in topks:
        ret = recall_precision_at_k(r, k, counts, compute_mean)
        recall.append(ret['recall'])
        pre.append(ret['precision'])
        mrr.append(mrr_at_k(r, k, compute_mean))
        ndcg.append(ndcg_at_k(r, k, counts, compute_mean))
    return {'recall':recall, 'precision': pre, 'mrr': mrr, 'ndcg': ndcg}


def generate(lm_backbone, query_tokens, generation_config, pad_token_id, device):
    attention_mask = query_tokens != pad_token_id
    input_ids = torch.masked_fill(query_tokens, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    return query_tokens.to(device), output.sequences[:, query_tokens.shape[-1]:]

@dataclass
class EvalStorage:
    query: List[str] = field(default_factory=list)
    postprocessed_response: List[str] = field(default_factory=list)
    reference_response: List[str] = field(default_factory=list)


def evaluate_lm(args: Args, lm_model, emb_model, emb_tokenizer, lm_tokenizer, item_embs, dataloader, generation_config, device, sampling=True):

    # compute topks
    topks, k = [], args.topk
    while True:
        if k % 2 == 0:
            topks.append(k)
        else:
            topks.append(k)
            break
        k = k // 2
    topks = topks[::-1]

    # evaluate model
    eval_storage = EvalStorage()
    with torch.no_grad():

        all_ref_topks, all_gen_topks, all_labels, all_label_counts = [], [], [], []

        accelerator.print("evaluating gennerations")
        for data in tqdm(dataloader):
            # get reference reward
            ref_sess_embs = get_embedding(emb_model, data["ref_response_mxbai"], args.task.emb_pad_token_id, device)
            ref_scores = get_scores(item_embs, ref_sess_embs, args.score_batch_size)
            ref_scores = ref_scores - 999999 * data["dense_in_sesses"]
            _, ref_topks = torch.topk(ref_scores, k=max(topks), dim=1)
            all_ref_topks.append(ref_topks)

            # generate response
            _, gen_response = generate(
                lm_model,
                data["sess_description_gemma"],
                generation_config,
                lm_tokenizer.pad_token_id,
                device
            )
            gen_response = truncate_response(args, lm_tokenizer, gen_response)
            gen_response = lm_tokenizer.batch_decode(gen_response, skip_special_tokens=True)

            # get generated response reward
            gen_response_mxbai = emb_tokenizer(gen_response, padding='max_length', truncation=True, max_length=args.task.response_length, return_tensors="pt")['input_ids']
            gen_sess_embs = get_embedding(emb_model, gen_response_mxbai, args.task.emb_pad_token_id, device)
            gen_scores = get_scores(item_embs, gen_sess_embs, args.score_batch_size)
            gen_scores = gen_scores - 999999 * data["dense_in_sesses"]
            _, gen_topks = torch.topk(gen_scores, k=max(topks), dim=1)
            all_gen_topks.append(gen_topks)

            eval_storage.query.extend(data["sess_descriptions"])
            eval_storage.reference_response.extend(data["ref_responses"])
            eval_storage.postprocessed_response.extend(gen_response)

            # compute lables from dense_out_sesses
            index = torch.nonzero(data["dense_out_sesses"])
            count = torch.count_nonzero(data["dense_out_sesses"], dim=-1)
            label = torch.full((len(data["dense_out_sesses"]), args.max_label_count), -1).to(device)
            value = index[:, 1]
            new_index = []
            for i in count:
                new_index.append(torch.arange(i))
            new_index = torch.cat(new_index)
            label[index[:, 0], new_index] = value

            all_labels.append(label)
            all_label_counts.append(count)
            if sampling:
                break
        
        ref_result_dict = compute_result(torch.cat(all_ref_topks), torch.cat(all_labels), torch.cat(all_label_counts), topks)
        gen_result_dict = compute_result(torch.cat(all_gen_topks), torch.cat(all_labels), torch.cat(all_label_counts), topks)

    eval_df = pd.DataFrame(
        {
            "query": gather_object(eval_storage.query),
            "postprocessed_response": gather_object(eval_storage.postprocessed_response),
            "reference_responses": gather_object(eval_storage.reference_response),
        }
    )
    return eval_storage, eval_df, ref_result_dict, gen_result_dict, topks


def evaluate_emb(args: Args, lm_model, emb_model, emb_tokenizer, item_data_desp_token, dataloader, device, gen_responses, sampling=True):

    # compute topks
    topks, k = [], args.topk
    while True:
        if k % 2 == 0:
            topks.append(k)
        else:
            topks.append(k)
            break
        k = k // 2
    topks = topks[::-1]

    # evaluate model
    eval_storage = EvalStorage()
    with torch.no_grad():

        item_embs = []
        accelerator.print("generating item embeddings for evaluating embedding model")
        for i in tqdm(range(0, len(item_data_desp_token), args.emb_batch_size)):
            mb_item_token = item_data_desp_token[i : min(i+args.emb_batch_size, len(item_data_desp_token))]
            mb_item_embs = get_embedding(emb_model, mb_item_token, args.task.emb_pad_token_id, device)
            item_embs.append(mb_item_embs)
        item_embs = torch.cat(item_embs, dim=0)

        all_ref_topks, all_gen_topks, all_labels, all_label_counts = [], [], [], []

        accelerator.print("evaluating embeddings")
        for data in tqdm(dataloader):
            # get reference reward
            ref_sess_embs = get_embedding(emb_model, data["ref_response_mxbai"], args.task.emb_pad_token_id, device)
            ref_scores = get_scores(item_embs, ref_sess_embs, args.score_batch_size)
            ref_scores = ref_scores - 999999 * data["dense_in_sesses"]
            _, ref_topks = torch.topk(ref_scores, k=max(topks), dim=1)
            all_ref_topks.append(ref_topks)

            # generate response
            # print('gen_responses', gen_responses, gen_responses.shape)
            gen_response = []
            for i in data["index"]:
                gen_response.append(gen_responses[i])
            gen_response = torch.stack(gen_response, dim=0)
            # print('gen_response', gen_response, gen_response.shape)

            # get generated response reward
            gen_sess_embs = get_embedding(emb_model, gen_response, args.task.emb_pad_token_id, device)
            gen_scores = get_scores(item_embs, gen_sess_embs, args.score_batch_size)
            gen_scores = gen_scores - 999999 * data["dense_in_sesses"]
            _, gen_topks = torch.topk(gen_scores, k=max(topks), dim=1)
            all_gen_topks.append(gen_topks)

            eval_storage.query.extend(data["sess_descriptions"])
            eval_storage.reference_response.extend(data["ref_responses"])
            eval_storage.postprocessed_response.extend(emb_tokenizer.batch_decode(gen_response, skip_special_tokens=True))

            # compute lables from dense_out_sesses
            index = torch.nonzero(data["dense_out_sesses"])
            count = torch.count_nonzero(data["dense_out_sesses"], dim=-1)
            label = torch.full((len(data["dense_out_sesses"]), args.max_label_count), -1).to(device)
            value = index[:, 1]
            new_index = []
            for i in count:
                new_index.append(torch.arange(i))
            new_index = torch.cat(new_index)
            label[index[:, 0], new_index] = value

            all_labels.append(label)
            all_label_counts.append(count)
            if sampling:
                break
        
        ref_result_dict = compute_result(torch.cat(all_ref_topks), torch.cat(all_labels), torch.cat(all_label_counts), topks)
        gen_result_dict = compute_result(torch.cat(all_gen_topks), torch.cat(all_labels), torch.cat(all_label_counts), topks)

    eval_df = pd.DataFrame(
        {
            "query": gather_object(eval_storage.query),
            "postprocessed_response": gather_object(eval_storage.postprocessed_response),
            "reference_responses": gather_object(eval_storage.reference_response),
        }
    )
    return eval_storage, eval_df, ref_result_dict, gen_result_dict, topks


class ModelWrapper(nn.Module):
    def __init__(self, lm_model, emb_model) -> None:
        super().__init__()
        self.lm_model = lm_model
        self.emb_model = emb_model

    def forward(self, **kwargs):
        return self.lm_model(**kwargs), self.emb_model(**kwargs)
    

if __name__ == "__main__":

    # init
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.batch_size = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    if args.whiten_rewards:
        assert (args.local_batch_size >= 8), f"Per-rank minibatch size {args.local_batch_size} is insufficient for whitening"

    # init seed
    local_seed = args.seed + accelerator.process_index * 100003
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    # init logging
    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}__{args.output_dir_lm.split('/')[1]}"
    accelerator.print("Wandb run name: ", run_name)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z, max_bins: None
    if accelerator.is_main_process:
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                sync_tensorboard=True,
                config=asdict(args),
                name=run_name,
                save_code=True,
                entity=args.wandb_entity,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    torch.backends.cudnn.deterministic = True

    # load tokenizer
    emb_tokenizer = AutoTokenizer.from_pretrained(
        args.emb_model,
        padding_side="left",
        trust_remote_code=True,
    )
    emb_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    args.task.emb_pad_token_id = emb_tokenizer.pad_token_id

    lm_tokenizer = AutoTokenizer.from_pretrained(
        args.lm_model,
        padding_side="left",
        trust_remote_code=True,
    )
    lm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    args.task.truncate_token_id = lm_tokenizer.eos_token_id

    # load dataset
    dataset = load_dataset(args.task.query_dataset, split='train')
    dataset = dataset.with_format("torch", columns=["index", "sess_descriptions", "dense_in_sesses", "dense_out_sesses", "out_sess_item_idxs", "sess_description_gemma", "ref_response_mxbai"])
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
    emb_dataloader = DataLoader(dataset, batch_size=args.emb_per_device_train_batch_size, shuffle=True)

    validation_dataset = load_dataset(args.task.query_dataset, split="test")
    validation_dataset = validation_dataset.with_format("torch", columns=["index", "dense_in_sesses", "dense_out_sesses", "sess_descriptions", "sess_description_gemma", "ref_responses", "ref_response_mxbai"])
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size)
    args.max_label_count = int(torch.max(validation_dataset["dense_out_sesses"].sum(-1)).item())

    item_dataset = load_dataset(args.task.item_dataset, split='train')
    item_dataset = item_dataset.with_format("torch", columns=["index", "item_description_tokens"])
    item_data_desp_token = item_dataset["item_description_tokens"]

    # get all profiles
    train_profiles, train_indices = [], []
    for i in tqdm(range(0, len(dataset))):
        train_profiles.append(dataset[i]["sess_descriptions"][237:-34])
        train_indices.append(dataset[i]["index"].item())

    train_indices = np.array(train_indices)
    assert len(np.unique(train_indices)) == len(dataset)

    all_train_profiles = ['' for i in range(len(dataset))]
    for (i, profile) in zip(train_indices, train_profiles):
        all_train_profiles[i] = profile
    all_train_profiles = emb_tokenizer(all_train_profiles, padding='max_length', truncation=True, max_length=args.task.response_length, return_tensors="pt")['input_ids'][:, 512:]

    # print("all_train_profiles", all_train_profiles, all_train_profiles.shape)

    validation_profiles, validation_indices = [], []
    for i in tqdm(range(0, len(validation_dataset))):
        validation_profiles.append(validation_dataset[i]["sess_descriptions"][237:-34])
        validation_indices.append(validation_dataset[i]["index"].item())
        
    validation_indices = np.array(validation_indices)
    assert len(np.unique(validation_indices)) == len(validation_dataset)
        
    all_val_profiles = ['' for i in range(len(validation_dataset))]
    for (i, profile) in zip(validation_indices, validation_profiles):
        all_val_profiles[i] = profile
    all_val_profiles = emb_tokenizer(all_val_profiles, padding='max_length', truncation=True, max_length=args.task.response_length, return_tensors="pt")['input_ids'][:, 512:]

    # print("all_val_profiles", all_val_profiles, all_val_profiles.shape)

    # load model
    lm_model_config = AutoConfig.from_pretrained(args.lm_model)
    configure_dropout(lm_model_config, ["attention_dropout"], 0.0)
    emb_model_config = AutoConfig.from_pretrained(args.emb_model)
    configure_dropout(emb_model_config, ["hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout"], 0.0)
    if accelerator.is_main_process:
        pprint(lm_model_config)
        pprint(emb_model_config)

    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model, config=lm_model_config, trust_remote_code=True)
    emb_model = AutoModel.from_pretrained(args.emb_model, config=emb_model_config, trust_remote_code=True)

    # lora for the model
    lm_peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
    )
    lm_model = get_peft_model(lm_model, peft_config=lm_peft_config)
    accelerator.print(lm_model)
    accelerator.print(emb_model)

    # generate without eos or padding
    lm_model.generation_config.eos_token_id = None
    lm_model.generation_config.pad_token_id = None

    model = ModelWrapper(lm_model, emb_model)
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)

    # sync random states before prepare
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    emb_dataloader = accelerator.prepare(emb_dataloader)
    validation_dataloader = accelerator.prepare(validation_dataloader)

    def repeat_generator():
        while True:
            yield from dataloader
    iter_dataloader = iter(repeat_generator())

    def emb_repeat_generator():
        while True:
            yield from emb_dataloader
    emb_iter_dataloader = iter(emb_repeat_generator())

    torch.manual_seed(local_seed)

    # config for generation
    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=(args.task.temperature + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    validation_generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=(args.task.temperature + 1e-7),
        # temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    # start training
    accelerator.print("===start training===")
    global_step = 1

    for _ in range(args.num_iterations):

    # ======================= training emb ======================================

        accelerator.print("===training emb===")
        accelerator.print(global_step, args.num_emb_updates + global_step)
        model.lm_model.eval()
        model.emb_model.train()

        optimizer.param_groups[0]["lr"] = args.emb_lr

        for update in range(global_step, args.num_emb_updates + global_step):

            data = next(emb_iter_dataloader)

            with torch.no_grad():

                # evaluation
                if args.print_sample_output_freq > 0 and (update - 1) % args.print_sample_output_freq == 0: # !!! and update > 1
                    if args.run_eval:
                        eval_storage, eval_df, all_ref_result_dict, all_gen_result_dict, topks = evaluate_emb(
                            args,
                            accelerator.unwrap_model(model).lm_model,
                            accelerator.unwrap_model(model).emb_model,
                            emb_tokenizer,
                            item_data_desp_token,
                            validation_dataloader,
                            device,
                            sampling=False,
                            gen_responses=all_val_profiles
                        )
                        if accelerator.is_main_process:
                            eval_df.to_csv(f"runs/{run_name}/query_responses_{update}.csv")
                            if args.track:
                                wandb.log({f"eval/query_responses_{update}": wandb.Table(dataframe=eval_df)}, step=update)
                        
                        for i in range(len(topks)):
                            writer.add_scalar(f"objective/ref_recall@{topks[i]}", accelerator.gather(all_ref_result_dict['recall'][i]).mean().item(), update)
                            writer.add_scalar(f"objective/ref_precision@{topks[i]}", accelerator.gather(all_ref_result_dict['precision'][i]).mean().item(), update)
                            writer.add_scalar(f"objective/ref_mrr@{topks[i]}", accelerator.gather(all_ref_result_dict['mrr'][i]).mean().item(), update)
                            writer.add_scalar(f"objective/ref_ndcg@{topks[i]}", accelerator.gather(all_ref_result_dict['ndcg'][i]).mean().item(), update)
                            writer.add_scalar(f"objective/gen_recall@{topks[i]}", accelerator.gather(all_gen_result_dict['recall'][i]).mean().item(), update)
                            writer.add_scalar(f"objective/gen_precision@{topks[i]}", accelerator.gather(all_gen_result_dict['precision'][i]).mean().item(), update)
                            writer.add_scalar(f"objective/gen_mrr@{topks[i]}", accelerator.gather(all_gen_result_dict['mrr'][i]).mean().item(), update)
                            writer.add_scalar(f"objective/gen_ndcg@{topks[i]}", accelerator.gather(all_gen_result_dict['ndcg'][i]).mean().item(), update)

                    # save model
                    os.makedirs(os.path.dirname(args.output_dir_lm), exist_ok=True)
                    unwrapped: PreTrainedModel = accelerator.unwrap_model(model).lm_model
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        lm_tokenizer.save_pretrained(args.output_dir_lm)
                        unwrapped.save_pretrained(
                            args.output_dir_lm,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                            state_dict=accelerator.get_state_dict(unwrapped),
                            safe_serialization=False,
                        )
                    os.makedirs(os.path.dirname(args.output_dir_emb), exist_ok=True)
                    unwrapped: PreTrainedModel = accelerator.unwrap_model(model).emb_model
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        emb_tokenizer.save_pretrained(args.output_dir_emb)
                        unwrapped.save_pretrained(
                            args.output_dir_emb,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                            state_dict=accelerator.get_state_dict(unwrapped),
                            safe_serialization=False,
                        )
                    del eval_storage, eval_df
                    torch.cuda.empty_cache()

            # training
            mini_batch_inds = np.random.permutation(args.emb_per_device_train_batch_size)
                    
            mb_pos_item = torch.stack([item_data_desp_token[data["out_sess_item_idxs"][i.item()][0]] for i in mini_batch_inds], dim=0)
            mb_profile = torch.stack([all_train_profiles[i.item()] for i in data["index"][mini_batch_inds]], dim=0)

            mb_pos_emb = get_embedding(model.emb_model, mb_pos_item, args.task.emb_pad_token_id, device)
            mb_ses_emb = get_embedding(model.emb_model, mb_profile, args.task.emb_pad_token_id, device)

            mb_pos_emb = F.normalize(mb_pos_emb, dim=-1)
            mb_ses_emb = F.normalize(mb_ses_emb, dim=-1)

            all_scores = mb_ses_emb @ mb_pos_emb.T
            labels = torch.arange(len(all_scores), device=device)
            loss = F.cross_entropy(all_scores / args.loss_temp, labels, reduction=args.loss_reduction)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.is_main_process:
                console.print(
                    f"loss",
                    loss.item(),
                )

            # logging
            with torch.no_grad():
                writer.add_scalar("dpo/loss/policy", accelerator.gather(loss).mean().item(), update)
                writer.add_scalar("dpo/lr", optimizer.param_groups[0]["lr"], update)
                torch.cuda.empty_cache()

        global_step += args.num_emb_updates

    # final evaluation; eval with evaluate_lm since trained lm last
    if args.run_eval:
        eval_storage, eval_df, all_ref_result_dict, all_gen_result_dict, topks = evaluate_lm(
            args,
            accelerator.unwrap_model(model).lm_model,
            accelerator.unwrap_model(model).emb_model,
            emb_tokenizer,
            lm_tokenizer,
            item_embs.to(device),
            validation_dataloader,
            validation_generation_config,
            device,
            sampling=False,
        )
        if accelerator.is_main_process:
            eval_df.to_csv(f"runs/{run_name}/table.csv")
            if args.track:
                wandb.log({"eval/query_responses": wandb.Table(dataframe=eval_df)}, step=update)
        
        for i in range(len(topks)):
            writer.add_scalar(f"objective/ref_recall@{topks[i]}", accelerator.gather(all_ref_result_dict['recall'][i]).mean().item(), update)
            writer.add_scalar(f"objective/ref_precision@{topks[i]}", accelerator.gather(all_ref_result_dict['precision'][i]).mean().item(), update)
            writer.add_scalar(f"objective/ref_mrr@{topks[i]}", accelerator.gather(all_ref_result_dict['mrr'][i]).mean().item(), update)
            writer.add_scalar(f"objective/ref_ndcg@{topks[i]}", accelerator.gather(all_ref_result_dict['ndcg'][i]).mean().item(), update)
            writer.add_scalar(f"objective/gen_recall@{topks[i]}", accelerator.gather(all_gen_result_dict['recall'][i]).mean().item(), update)
            writer.add_scalar(f"objective/gen_precision@{topks[i]}", accelerator.gather(all_gen_result_dict['precision'][i]).mean().item(), update)
            writer.add_scalar(f"objective/gen_mrr@{topks[i]}", accelerator.gather(all_gen_result_dict['mrr'][i]).mean().item(), update)
            writer.add_scalar(f"objective/gen_ndcg@{topks[i]}", accelerator.gather(all_gen_result_dict['ndcg'][i]).mean().item(), update)

    # save model
    os.makedirs(os.path.dirname(args.output_dir_lm), exist_ok=True)
    unwrapped: PreTrainedModel = accelerator.unwrap_model(model).lm_model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        lm_tokenizer.save_pretrained(args.output_dir_lm)
        unwrapped.save_pretrained(
            args.output_dir_lm,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(unwrapped),
            safe_serialization=False,
        )
    os.makedirs(os.path.dirname(args.output_dir_emb), exist_ok=True)
    unwrapped: PreTrainedModel = accelerator.unwrap_model(model).emb_model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        emb_tokenizer.save_pretrained(args.output_dir_emb)
        unwrapped.save_pretrained(
            args.output_dir_emb,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(unwrapped),
            safe_serialization=False,
        )