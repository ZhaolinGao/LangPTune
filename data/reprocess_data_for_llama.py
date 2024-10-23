import numpy as np
import argparse
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_gemma_sess_repo", type=str, default="")
    parser.add_argument("--hf_llama_sess_repo", type=str, default="")
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    train_dataset = load_dataset(args.hf_gemma_sess_repo, split="train")
    validation_dataset = load_dataset(args.hf_gemma_sess_repo, split="val")
    test_dataset = load_dataset(args.hf_gemma_sess_repo, split="test")

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', padding_side='left')
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def convert_to_llama(example):
        dialogue = [{'role': 'user', 'content': example['sess_descriptions'][20:-34]}]
        example["sess_description_gemma"] = tokenizer.apply_chat_template(dialogue, tokenize=True, add_generation_prompt=True, max_length=args.max_prompt_len, padding='max_length',)
        example["sess_descriptions"] = tokenizer.decode(example["sess_description_gemma"], skip_special_tokens=False)
        return example

    def filter_long_prompt(example):
        return len(example["sess_description_gemma"]) == args.max_prompt_len and (example["sess_description_gemma"][0] == 128000 or example["sess_description_gemma"][0] == 128256)

    print(train_dataset)
    print(validation_dataset)
    print(test_dataset)

    train_dataset = train_dataset.map(convert_to_llama)
    train_dataset = train_dataset.filter(lambda row: filter_long_prompt(row))
    train_dataset = train_dataset.rename_column("sess_description_gemma", "sess_description_llama")
    train_dataset = train_dataset.remove_columns(["index"])
    idx = np.arange(len(train_dataset))
    train_dataset = train_dataset.add_column("index", idx)

    validation_dataset = validation_dataset.map(convert_to_llama)
    validation_dataset = validation_dataset.filter(lambda row: filter_long_prompt(row))
    validation_dataset = validation_dataset.rename_column("sess_description_gemma", "sess_description_llama")
    validation_dataset = validation_dataset.remove_columns(["index"])
    idx = np.arange(len(validation_dataset))
    validation_dataset = validation_dataset.add_column("index", idx)

    test_dataset = test_dataset.map(convert_to_llama)
    test_dataset = test_dataset.filter(lambda row: filter_long_prompt(row))
    test_dataset = test_dataset.rename_column("sess_description_gemma", "sess_description_llama")
    test_dataset = test_dataset.remove_columns(["index"])
    idx = np.arange(len(test_dataset))
    test_dataset = test_dataset.add_column("index", idx)

    print(train_dataset)
    print(validation_dataset)
    print(test_dataset)

    ds_dict = {'train' : train_dataset,
                'val'  : validation_dataset,
                'test' : test_dataset}

    ds = DatasetDict(ds_dict)
    ds.push_to_hub(args.hf_llama_sess_repo)