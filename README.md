# Federated-Learning-for-Fine-Tuning-Large-Language-Models

Prerequisites
We ran our experiments on 4 RTX 6000 GPUs.
Python3.9+
pip install -r requirements.txt

Run teacher pre-training
To give the teacher some advantage, we first let the teacher pre-train on the public dataset.

PYTHONPATH=. python3 baselines/ft.py --teacher_ckpt bert-base-uncased --data_name cola --teacher_data_pct 20 --teacher_pretrain_epochs 5
In this case, the teacher model bert-base-uncased finetunes on the first 20% of the CoLA dataset for 5 epochs.

For SST2:

PYTHONPATH=. python3 baselines/ft.py --teacher_ckpt bert-base-uncased --data_name sst2 --teacher_data_pct 20 --teacher_pretrain_epochs 3
Run FL scheme
For running 4 clients:

# view help
source multigpu/run-kd-lora.sh 4 -h
# run with the pretrained teacher model on the cola dataset. teacher_data_pct is used for the clients to figure out how much data they should use.
source multigpu/run-kd-lora.sh 4 --teacher_ckpt results/model/bert-base-uncased_cola_20pct_5epochs/ --data_name cola --teacher_data_pct 20
Results are stored in results/*.json.

Baselines
# FedLoRA (FedAvg with LoRA)
source multigpu/run-lora.sh 4 --data_name cola
# FedAvg
source multigpu/run.sh 4 --data_name cola
Packages used
Huggingface
We used Huggingface libraries, all of which are written in Python.

transformers (1.2M LOC): For downloading pre-trained LLMs (BERT and DistilBERT in our case) into pytorch format. Github, 1.2 million LOC.
datasets (87k LOC): For loading the SST2 and CoLA datasets that we used for evaluation. Github
peft (112k LOC): For transforming large models into parameter-efficient versions (LoRA). Github
Flower
Flower (141k LOC) is a simple FL framework written in Python, but also supports FL with mobile devices (iOS/Android). It provides a standard FedAvg implementation, and we implemented FedLoRA and our FL scheme (FedKDLoRA) ourselves. Github

PyTorch
We use PyTorch (220k LOC) for model training. Written in C++ with Python bindings. Github

Datasets
We used datasets within the GLUE benchmark.

SST2
SST2 is a sentiment classification dataset. Each sample contains a sentence and the corresponding sentiment label. It contains 67k training samples and 1.6k testing samples.

CoLA
CoLA is a test of grammatical correctness. Each sample contains a sentence and whether it is grammatically correct. It contains 9k training samples and 1k testing samples.

Performance measurement
We evaluate all the schemes (FedAvg, FedLora, FedKDLora) via accuracy and loss with respect to the test dataset at every communication round. We also time the total runtime of each scheme.
