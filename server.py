import flwr as fl
from arguments import parse_args, pretty_print_args
from output_result import save_run_as_json
import time
import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

args = parse_args()
pretty_print_args(args)
NUM_CLIENTS = args.num_clients
NUM_SPLITS = NUM_CLIENTS + 1

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

if __name__ == "__main__":
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_average,
        min_available_clients=args.num_clients,
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
