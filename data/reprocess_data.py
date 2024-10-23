import argparse
import numpy as np
from datasets import load_dataset, DatasetDict

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_gemma_sess_repo", type=str, default="")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    train_dataset = load_dataset(args.hf_gemma_sess_repo, split="train")
    validation_dataset = load_dataset(args.hf_gemma_sess_repo, split="test")

    def only_first_item(example):
        i = 0
        for idx in range(len(example["out_sess_item_idxs"])):
            if example["out_sess_item_idxs"][idx] not in example["in_sess_item_idxs"]:
                i = idx
                break
        example["out_sess_item_idxs"] = [example["out_sess_item_idxs"][i]]
        example["dense_out_sesses"] = [0. for _ in range(len(example["dense_out_sesses"]))]
        example["dense_out_sesses"][example["out_sess_item_idxs"][0]] = 1.0
        return example

    train_dataset = train_dataset.map(only_first_item)
    validation_dataset = validation_dataset.map(only_first_item)

    train_dataset = train_dataset.filter(lambda row: row["out_sess_item_idxs"][0] not in row["in_sess_item_idxs"])
    validation_dataset = validation_dataset.filter(lambda row: row["out_sess_item_idxs"][0] not in row["in_sess_item_idxs"])
    validation_dataset = validation_dataset.train_test_split(test_size=0.5)

    idx = np.arange(len(train_dataset))
    train_dataset = train_dataset.add_column("index", idx)
    idx = np.arange(len(validation_dataset['train']))
    validation_dataset['train'] = validation_dataset['train'].add_column("index", idx)
    idx = np.arange(len(validation_dataset['test']))
    validation_dataset['test'] = validation_dataset['test'].add_column("index", idx)

    ds_dict = {'train' : train_dataset,
               'val'   : validation_dataset['train'],
               'test'  : validation_dataset['test']}
    ds = DatasetDict(ds_dict)
    ds.push_to_hub(args.hf_gemma_sess_repo)