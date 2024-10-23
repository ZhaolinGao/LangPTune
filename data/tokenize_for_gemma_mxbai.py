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
from rich.console import Console
from rich.pretty import pprint
from torch import optim
from torch.utils.data import Dataset, DataLoader
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
from datasets import Dataset as Huggingface_dataset
from datasets import DatasetDict
from huggingface_hub import create_repo


@dataclass
class DatasetHParams:
    data: str = 'amazon_books'
    data_max_length: int = 1024


@dataclass
class TaskHParams:
    response_length: int = 512


@dataclass
class Args:
    # common args
    exp_name: str = "gemma_mxbai"
    """the name of this experiment"""
    prompt: str = "The user has rated (out of 5.0) and reviewed following books arranged chronologically from the oldest (top) to the newest (bottom). Please provide a high-level summary of the user preference in detail."
    """the name of this experiment"""

    # other args
    lm_model: str = "google/gemma-2b-it"
    """the name of the pretrained model to use"""
    emb_model: str = "mixedbread-ai/mxbai-embed-large-v1"
    """the name of the pretrained model to use"""
    first_n_as_train: int = 4
    """use the first n item as train"""
    hf_item_repo: str = ""
    hf_gemma_sess_repo: str = ""
    task: TaskHParams = field(default_factory=TaskHParams)
    data: DatasetHParams = field(default_factory=DatasetHParams)


class ReviewDataset(Dataset):
    def __init__(self, args, split):

        data_dir = args.data.data
        self.args = args
        self.lm_tokenizer = lm_tokenizer
        self.prompt = args.prompt
        self.data_max_length = args.data.data_max_length
        self.response_length = args.task.response_length

        # load dataset
        with open(os.path.join('./data', data_dir, 'train_test_sessions.pkl'), "rb") as f:
            train_sess, train_rating, test_sess, test_rating = pickle.load(f)
        with open(os.path.join('./data', data_dir, 'item_data.pkl'), "rb") as f:
            item_data = pickle.load(f)

        # prep dataset
        if split == "train":
            full_sessions, full_ratings = train_sess, train_rating
        else:
            full_sessions, full_ratings = test_sess, test_rating
        self.item_des = [0] * len(item_data)
        self.num_items, self.num_sess = len(item_data), len(full_sessions)
        self.session, self.session_ratings, self.session_reviews, self.label = [], [], [], []

        for k, v in item_data.items():
            self.item_des[k] = self.get_description(v)
        for i in full_sessions:
            self.session.append(i[:args.first_n_as_train])
            self.label.append(i[args.first_n_as_train:])
        for i in full_ratings:
            self.session_ratings.append(i[0][:args.first_n_as_train])
            self.session_reviews.append(i[1][:args.first_n_as_train])

        # prep session data
        self.ref_response = []
        for i in range(self.num_sess):
            self.ref_response.append(self.item_des[self.session[i][-1]])

    def get_description(self, item):
        return f'Title: {item["Title"]}\nDescription: (average rating: {item["Average_rating"]}) {item["Description"]}\nCategory: {item["Category"]}\nPrice: {item["Price"]}'

    def __len__(self):
        return self.num_sess

    def __getitem__(self, idx):
        session_dense = np.zeros(self.num_items)
        session_dense[self.session[idx]] = 1

        session_dense_label = np.zeros(self.num_items)
        session_dense_label[self.label[idx]] = 1

        sess_des = [self.prompt]
        for i, item_idx in enumerate(self.session[idx]):
            sess_des.append(self.item_des[item_idx] + f'\nReview from the user: (rating: {self.session_ratings[idx][i]}) {self.session_reviews[idx][i]}')
        sess_des = "\n\n".join(sess_des)
        sess_des = f"<start_of_turn>user\n{sess_des}<end_of_turn>\n<start_of_turn>model"

        return self.session[idx], self.label[idx], session_dense, session_dense_label, sess_des, self.ref_response[idx]


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

    # load tokenizer
    emb_tokenizer = AutoTokenizer.from_pretrained(
        args.emb_model,
        padding_side="right",
        trust_remote_code=True,
    )
    emb_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    lm_tokenizer = AutoTokenizer.from_pretrained(
        args.lm_model,
        padding_side="left",
        trust_remote_code=True,
    )
    lm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # load dataset
    dataset = ReviewDataset(args, split="train")
    validation_dataset = ReviewDataset(args, split="test")

    # process item
    item_idx, item_deses, item_mxbai_tokens = [], [], []
    for i, item_des in tqdm(enumerate(dataset.item_des)):
        if len(emb_tokenizer(item_des)['input_ids']) > args.task.response_length:
            continue 

        item_mxbai_token = emb_tokenizer(item_des, padding='max_length', max_length=args.task.response_length)['input_ids']

        assert len(item_mxbai_token) == args.task.response_length
        assert item_mxbai_token[-1] == 0 or item_mxbai_token[-1] == 102

        item_idx.append(i)
        item_deses.append(item_des)
        item_mxbai_tokens.append(item_mxbai_token)

    item_dataset = pd.DataFrame({'index' : item_idx,
                                 'item_descriptions' : item_deses,
                                 'item_description_tokens' : item_mxbai_tokens})

    item_dataset = Huggingface_dataset.from_pandas(item_dataset)
    print('num item', len(item_dataset))
    item_dataset.push_to_hub(args.hf_item_repo)

    # process train
    in_sess_item_idxs, out_sess_item_idxs, dense_in_sesses, dense_out_sesses, sess_deses, ref_resps = [], [], [], [], [], []
    gemma_prompt_tokens, mxbai_ref_resp_tokens = [], []

    for sess in tqdm(dataset):
        in_sess_item_idx, out_sess_item_idx, dense_in_sess, dense_out_sess, sess_des, ref_resp = sess

        if len(lm_tokenizer(sess_des)['input_ids']) > args.data.data_max_length:
            continue
        
        if len(emb_tokenizer(ref_resp)['input_ids']) > args.task.response_length:
            continue

        prompt_tokens = lm_tokenizer(sess_des, padding='max_length', max_length=args.data.data_max_length)['input_ids']
        ref_resp_tokens = emb_tokenizer(ref_resp, padding='max_length', max_length=args.task.response_length)['input_ids']

        assert len(prompt_tokens) == args.data.data_max_length and len(ref_resp_tokens) == args.task.response_length
        assert prompt_tokens[0] == 256000 or prompt_tokens[0] == 2
        assert ref_resp_tokens[-1] == 0 or ref_resp_tokens[-1] == 102

        in_sess_item_idxs.append(in_sess_item_idx)
        out_sess_item_idxs.append(out_sess_item_idx)
        dense_in_sesses.append(dense_in_sess)
        dense_out_sesses.append(dense_out_sess)
        sess_deses.append(sess_des)
        ref_resps.append(ref_resp)
        gemma_prompt_tokens.append(prompt_tokens)
        mxbai_ref_resp_tokens.append(ref_resp_tokens)

    df_train = pd.DataFrame({'in_sess_item_idxs' : in_sess_item_idxs,
                             'out_sess_item_idxs' : out_sess_item_idxs,
                             'dense_in_sesses' : dense_in_sesses,
                             'dense_out_sesses' : dense_out_sesses,
                             'sess_descriptions' : sess_deses,
                             'sess_description_gemma' : gemma_prompt_tokens,
                             'ref_responses' : ref_resps,
                             'ref_response_mxbai' : mxbai_ref_resp_tokens})

    # process validation
    in_sess_item_idxs, out_sess_item_idxs, dense_in_sesses, dense_out_sesses, sess_deses, ref_resps = [], [], [], [], [], []
    gemma_prompt_tokens, mxbai_ref_resp_tokens = [], []

    for sess in tqdm(validation_dataset):
        in_sess_item_idx, out_sess_item_idx, dense_in_sess, dense_out_sess, sess_des, ref_resp = sess

        if len(lm_tokenizer(sess_des)['input_ids']) > args.data.data_max_length:
            continue
        
        if len(emb_tokenizer(ref_resp)['input_ids']) > args.task.response_length:
            continue

        prompt_tokens = lm_tokenizer(sess_des, padding='max_length', max_length=args.data.data_max_length)['input_ids']
        ref_resp_tokens = emb_tokenizer(ref_resp, padding='max_length', max_length=args.task.response_length)['input_ids']

        assert len(prompt_tokens) == args.data.data_max_length and len(ref_resp_tokens) == args.task.response_length
        assert prompt_tokens[0] == 256000 or prompt_tokens[0] == 2
        assert ref_resp_tokens[-1] == 0 or ref_resp_tokens[-1] == 102

        in_sess_item_idxs.append(in_sess_item_idx)
        out_sess_item_idxs.append(out_sess_item_idx)
        dense_in_sesses.append(dense_in_sess)
        dense_out_sesses.append(dense_out_sess)
        sess_deses.append(sess_des)
        ref_resps.append(ref_resp)
        gemma_prompt_tokens.append(prompt_tokens)
        mxbai_ref_resp_tokens.append(ref_resp_tokens)

    df_test = pd.DataFrame({'in_sess_item_idxs' : in_sess_item_idxs,
                             'out_sess_item_idxs' : out_sess_item_idxs,
                             'dense_in_sesses' : dense_in_sesses,
                             'dense_out_sesses' : dense_out_sesses,
                             'sess_descriptions' : sess_deses,
                             'sess_description_gemma' : gemma_prompt_tokens,
                             'ref_responses' : ref_resps,
                             'ref_response_mxbai' : mxbai_ref_resp_tokens})

    print('train num sess', len(df_train))
    print('test num sess', len(df_test))

    ds_dict = {'train' : Huggingface_dataset.from_pandas(df_train),
               'test' : Huggingface_dataset.from_pandas(df_test)}

    ds = DatasetDict(ds_dict)
    ds.push_to_hub(args.hf_gemma_sess_repo)