from transformers import get_scheduler
import torch
import logging
from arguments import parse_args, pretty_print_args
from data_loader import load_data
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from evaluate import load as load_metric
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
DEVICE = torch.cuda.current_device()


def train(net, trainloader, epochs, lr):
    num_training_steps = epochs * len(trainloader)
    optimizer = AdamW(net.parameters(), lr=lr, no_deprecation_warning=True)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    net.train()
    for i in range(epochs):
        print(f"epoch {i + 1} / {epochs}")
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
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

    args = parse_args()
    pretty_print_args(args)
    ckpt = args.teacher_ckpt
    num_splits = args.num_clients + 1
    train_loader, test_loader = load_data(
        args.data_path, args.data_name,
        rank=0, num_splits=num_splits, tokenizer_ckpt=ckpt, teacher_data_pct=args.teacher_data_pct
    )
    net = AutoModelForSequenceClassification.from_pretrained(
        ckpt, num_labels=2
    ).to(DEVICE)
    train(net, train_loader, epochs=args.teacher_pretrain_epochs, lr=args.client_lr)
    print(test(net, test_loader))

    filename = f"{args.teacher_ckpt}_{args.data_name}_{args.teacher_data_pct}pct_{args.teacher_pretrain_epochs}epochs"
    filename = f"results/model/{filename}"
    net.save_pretrained(filename)
    print(f"model saved to {filename}")


if __name__ == "__main__":
    main()
