import warnings

import flwr as fl
import torch

from evaluate import load as load_metric

from peft import (
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
)

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

import transformers
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler
from arguments import parse_args
from data_loader import load_data

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

args = parse_args()
RANK = args.rank
NUM_CLIENTS = args.num_clients
NUM_SPLITS = NUM_CLIENTS + 1  # teacher also has a split

DEVICE = torch.cuda.current_device()
CHECKPOINT = args.client_ckpt

def train(net, trainloader, epochs, lr):
    optimizer = AdamW(net.parameters(), lr=lr, no_deprecation_warning=True)
    net.train()
    for i in range(epochs):
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

def main():
    logging.info(f"Started client {RANK}")
    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)

    peft_config = LoraConfig(
        task_type="SEQ_CLS", 
        inference_mode=False, 
        target_modules=["q_lin", "v_lin"],
        r=args.lora_r, 
        lora_alpha=16, 
        lora_dropout=0.1)

    net = get_peft_model(net, peft_config)

    trainloader, testloader = load_data(args.data_path, args.data_name, RANK, NUM_SPLITS, CHECKPOINT, args.teacher_data_pct)
    peft_state_dict_keys = get_peft_model_state_dict(net).keys()
    # Flower client
    class Client(fl.client.NumPyClient):
        def get_parameters(self, config=None):
            state_dict = get_peft_model_state_dict(net)
            return [val.cpu().numpy() for _, val in state_dict.items()]

        def set_parameters(self, parameters):
            params_dict = zip(peft_state_dict_keys, parameters)
            state_dict = {k: torch.Tensor(v) for k, v in params_dict}
            set_peft_model_state_dict(net, state_dict)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            logging.info(f"Client {RANK} Training Started...")
            train(net, trainloader, epochs=args.client_epochs, lr=args.client_lr)
            return self.get_parameters(), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())


if __name__ == "__main__":
    main()
