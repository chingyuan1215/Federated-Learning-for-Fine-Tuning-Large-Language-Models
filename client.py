from collections import OrderedDict
import sys

import flwr as fl
import torch

from evaluate import load as load_metric

from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler
from arguments import parse_args

from data_loader import load_data

args = parse_args()
RANK = args.rank
NUM_CLIENTS = args.num_clients
NUM_SPLITS = NUM_CLIENTS + 1
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
    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)

    trainloader, testloader = load_data(args.data_path, args.data_name, RANK, NUM_SPLITS, CHECKPOINT, args.teacher_data_pct)

    # Flower client
    class Client(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            print("Training Started...")
            train(net, trainloader, epochs=args.client_epochs, lr=args.client_lr)
            return self.get_parameters(config={}), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())


if __name__ == "__main__":
    main()