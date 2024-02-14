from collections import OrderedDict
import warnings

import flwr as fl
import torch
import numpy as np

import random
from torch.utils.data import DataLoader

from datasets import load_dataset
from evaluate import load as load_metric

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda")
# CHECKPOINT = "bert-base-uncased"
CHECKPOINT = "distilbert-base-uncased"
# CHECKPOINT = "roberta-large"

def load_data():

    raw_datasets = load_dataset("glue", "cola")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    # random 100 samples
    population_train = random.sample(range(len(raw_datasets["train"])), 100)
    population_test = random.sample(range(len(raw_datasets["validation"])), 100)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets["train"] = tokenized_datasets["train"].select(population_train)
    tokenized_datasets["validation"] = tokenized_datasets["validation"].select(population_test)

    tokenized_datasets = tokenized_datasets.remove_columns("sentence")
    tokenized_datasets = tokenized_datasets.remove_columns("idx")
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        tokenized_datasets["validation"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader

def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=5e-5, no_deprecation_warning=True)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'Total Parameters': total_params,
        'Trainable Parameters': trainable_params,
        'Non-Trainable Parameters': total_params - trainable_params
    }

def main():

    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)

    print(net)

    peft_config = LoraConfig(
        task_type="SEQ_CLS", 
        inference_mode=False, 
        target_modules=["q_lin", "v_lin"],
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1)

    net = get_peft_model(net, peft_config)

    print('After LoRA:')
    print(net)
    count_parameters(net)

    net.print_trainable_parameters()

    trainloader, testloader = load_data()

    train(net, trainloader, epochs=20)

    loss, accuracy = test(net, testloader)
    print(float(loss), len(testloader), {"accuracy": float(accuracy)})


if __name__ == "__main__":
    main()