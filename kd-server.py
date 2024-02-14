import time
import flwr as fl
import torch
import logging
from data_loader import load_data

from peft import (
    get_peft_model,
    LoraConfig,
)

from transformers import AutoModelForSequenceClassification, AutoConfig

from kd_strategy import CustomStrategy
from arguments import parse_args, pretty_print_args
from output_result import save_run_as_json
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

RANK = 0
args = parse_args()
pretty_print_args(args)
NUM_CLIENTS = args.num_clients
NUM_SPLITS = NUM_CLIENTS + 1
DEVICE = torch.cuda.current_device()

STUDENT_CKPT = args.client_ckpt
TEACHER_CKPT = args.teacher_ckpt


def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def create_student_model():
    student = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_CKPT,
        config=AutoConfig.from_pretrained(STUDENT_CKPT, output_attentions=True,output_hidden_states=True, num_labels=2)
    ).to(DEVICE)
    
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        target_modules=["q_lin", "v_lin"],
        r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.1)

    return get_peft_model(student, peft_config)


def main():
    student = create_student_model()
    teacher = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_CKPT,
        config=AutoConfig.from_pretrained(TEACHER_CKPT, output_attentions=True,output_hidden_states=True, num_labels=2)
    ).to(DEVICE)

    train_loader, test_loader = load_data(args.data_path, args.data_name, RANK, NUM_SPLITS, STUDENT_CKPT, args.teacher_data_pct)  # both teacher and student must use the same tokenizer

    # Define strategy
    strategy = CustomStrategy(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        test_loader=test_loader,
        kd_epochs=args.kd_epochs,
        teacher_pretrain_epochs=args.teacher_pretrain_epochs,
        distil_lr=args.distill_lr,
        min_available_clients=NUM_CLIENTS,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start server
    t1 = time.perf_counter()
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
    t2 = time.perf_counter()
    extra_data = dict(
        elapsed_time_secs=t2 - t1
    )
    save_run_as_json(args, history, extra_data=extra_data)


if __name__ == "__main__":
    main()
