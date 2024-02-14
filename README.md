# Federated Learning for Fine-Tuning LLM

## Prerequisites
1. We ran our experiments on 4 RTX 6000 GPUs.
1. Python3.9+
1. `pip install -r requirements.txt`
## Architecture
<img width="854" alt="截圖 2024-02-14 下午2 19 06" src="https://github.com/chingyuan1215/Federated-Learning-for-Fine-Tuning-Large-Language-Models/assets/65063401/23026dd9-7e6b-4e79-a13c-85f11d49767a">

## Steps to run
### Run teacher pre-training
To give the teacher some advantage, we first let the teacher pre-train on the public dataset.
```bash
PYTHONPATH=. python3 baselines/ft.py --teacher_ckpt bert-base-uncased --data_name cola --teacher_data_pct 20 --teacher_pretrain_epochs 5
```
In this case, the teacher model `bert-base-uncased` finetunes on the first 20% of the CoLA dataset for 5 epochs.

For SST2:
```bash
PYTHONPATH=. python3 baselines/ft.py --teacher_ckpt bert-base-uncased --data_name sst2 --teacher_data_pct 20 --teacher_pretrain_epochs 3
```
### Run FL scheme

For running 4 clients:
```bash
# view help
source multigpu/run-kd-lora.sh 4 -h
# run with the pretrained teacher model on the cola dataset. teacher_data_pct is used for the clients to figure out how much data they should use.
source multigpu/run-kd-lora.sh 4 --teacher_ckpt results/model/bert-base-uncased_cola_20pct_5epochs/ --data_name cola --teacher_data_pct 20
```
Results are stored in `results/*.json`.

### Baselines
```bash
# FedLoRA (FedAvg with LoRA)
source multigpu/run-lora.sh 4 --data_name cola
# FedAvg
source multigpu/run.sh 4 --data_name cola
```

## Packages used
### Huggingface
We used [Huggingface](https://huggingface.co/) libraries, all of which are written in Python.
1. [`transformers`](https://huggingface.co/docs/transformers/) (1.2M LOC): For downloading pre-trained LLMs (BERT and DistilBERT in our case) into pytorch format. [Github](https://github.com/huggingface/transformers), 1.2 million LOC.
1. [`datasets`](https://huggingface.co/docs/datasets/) (87k LOC): For loading the `SST2` and `CoLA` datasets that we used for evaluation. [Github](https://github.com/huggingface/datasets/)
1. [`peft`](https://huggingface.co/docs/peft/) (112k LOC): For transforming large models into parameter-efficient versions (LoRA). [Github](https://github.com/huggingface/peft/)

### Flower
[Flower](https://flower.dev/) (141k LOC) is a simple FL framework written in Python, but also supports FL with mobile devices (iOS/Android). It provides a standard FedAvg implementation, and we implemented FedLoRA and our FL scheme (FedKDLoRA) ourselves. [Github](https://github.com/adap/flower)

### PyTorch
We use [PyTorch](https://pytorch.org/) (220k LOC) for model training. Written in C++ with Python bindings. [Github](https://github.com/pytorch/pytorch)

## Datasets
We used datasets within the [`GLUE` benchmark](https://gluebenchmark.com/).
### [`SST2`](https://huggingface.co/datasets/glue/viewer/sst2)
[`SST2`](https://huggingface.co/datasets/glue/viewer/sst2) is a sentiment classification dataset. Each sample contains a sentence and the corresponding sentiment label. It contains 67k training samples and 1.6k testing samples.

### [`CoLA`](https://huggingface.co/datasets/glue/viewer/cola)
[`CoLA`](https://huggingface.co/datasets/glue/viewer/cola) is a test of grammatical correctness. Each sample contains a sentence and whether it is grammatically correct. It contains 9k training samples and 1k testing samples.

## Performance measurement
We evaluate all the schemes (FedAvg, FedLora, FedKDLora) via accuracy and loss with respect to the test dataset at every communication round. We also time the total runtime of each scheme.


<img width="853" alt="截圖 2024-02-14 下午2 19 24" src="https://github.com/chingyuan1215/Federated-Learning-for-Fine-Tuning-Large-Language-Models/assets/65063401/39976f37-fc20-42a5-abe9-bdcc0c2e6d89">

