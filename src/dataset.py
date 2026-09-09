from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


class PaddedPackedTokenDataset(Dataset):
    def __init__(self, path, sequence_length, pad_length, pad_token_id):
        self.pad_length = pad_length
        self.content_length = sequence_length - pad_length
        self.pad_token_id = pad_token_id
        self.tokens = np.memmap(Path(path), dtype=np.uint32, mode="r")

    def __len__(self):
        return len(self.tokens) // self.content_length

    def __getitem__(self, index):
        start = index * self.content_length
        tokens = torch.from_numpy(self.tokens[start : start + self.content_length].astype(np.int64))
        input_ids = torch.cat((torch.full((self.pad_length,), self.pad_token_id), tokens))
        labels = input_ids.clone()
        labels[: self.pad_length] = -100
        return {"input_ids": input_ids, "labels": labels}


if __name__ == "__main__":
    DATASET_NAME = "HuggingFaceFW/fineweb-edu"
    SUBSET_NAME = "sample-100BT"
    PROCESSED_DIR = Path("data/fineweb-edu-100bt-20m-20k")
    TOKENIZER_NAME = "Qwen/Qwen3-0.6B-Base"
    TRAIN_DOCUMENTS = 20_000_000
    TEST_DOCUMENTS = 20_000
    SEED = 42
    BATCH_SIZE = 16384
    BUFFERING = 16 * 1024 * 1024

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_dataset(DATASET_NAME, name=SUBSET_NAME, split="train")

    documents = documents.select_columns(["text"])
    documents = documents.shuffle(seed=SEED)
    train_documents = documents.select(range(TRAIN_DOCUMENTS))
    test_documents = documents.select(range(TRAIN_DOCUMENTS, TRAIN_DOCUMENTS + TEST_DOCUMENTS))

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    def write_token_file(dataset, output_path, batch_size):
        token_count = 0
        total_batches = (len(dataset) + batch_size - 1) // batch_size

        with output_path.open("wb", buffering=BUFFERING) as file:
            batches = dataset.iter(batch_size=batch_size, drop_last_batch=False)

            for batch in tqdm(batches, total=total_batches, desc=f"Writing {output_path.name}", unit="batch"):
                encoded = tokenizer(batch["text"], add_special_tokens=False, return_attention_mask=False)

                chunks = [np.asarray(token_ids + [tokenizer.eos_token_id], dtype=np.uint32) for token_ids in encoded["input_ids"]]

                batch_tokens = np.concatenate(chunks)
                batch_tokens.tofile(file)
                token_count += batch_tokens.size

        print(f"{output_path}: {token_count:,} tokens")

    write_token_file(train_documents, PROCESSED_DIR / "train.bin", BATCH_SIZE)
    write_token_file(test_documents, PROCESSED_DIR / "test.bin", BATCH_SIZE)
