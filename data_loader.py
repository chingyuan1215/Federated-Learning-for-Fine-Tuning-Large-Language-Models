from collections import OrderedDict
import warnings
import sys

import flwr as fl
import torch
import numpy as np

import random
from torch.utils.data import DataLoader

from datasets import load_dataset
from evaluate import load as load_metric

from transformers import AutoTokenizer, DataCollatorWithPadding

def _load_data(train_raw_dataset, test_raw_dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    tokenized_datasets = train_raw_dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    test_tokenized_datasets = test_raw_dataset.map(tokenize_function, batched=True)
    test_tokenized_datasets = test_tokenized_datasets.remove_columns(["idx", "sentence"])
    test_tokenized_datasets = test_tokenized_datasets.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        tokenized_datasets,
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        test_tokenized_datasets, batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader


def load_all_data(path, name, tokenizer_ckpt):
    train_raw_dataset = load_dataset(path, name, split="train")
    test_raw_dataset = load_dataset(path, name, split="validation")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
    return _load_data(train_raw_dataset, test_raw_dataset, tokenizer)


def load_data(path, name, rank, num_splits, tokenizer_ckpt, teacher_data_pct=None):
    if teacher_data_pct is None:
        teacher_data_pct = 100 // num_splits
    
    if rank == 0:
        split = f"0%:{teacher_data_pct}%"
    else:
        num_clients = num_splits - 1
        # just math to determine how much data the clients get
        split_start = teacher_data_pct + (rank - 1) * (100 - teacher_data_pct) // num_clients
        split_end = teacher_data_pct + (rank) * (100 - teacher_data_pct) // num_clients
        split = f"{split_start}%:{split_end}%"
    train_split = f"train[{split}]"
    train_raw_dataset = load_dataset(path, name, split=train_split)
    test_raw_dataset = load_dataset(path, name, split="validation")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)

    return _load_data(train_raw_dataset, test_raw_dataset, tokenizer)
