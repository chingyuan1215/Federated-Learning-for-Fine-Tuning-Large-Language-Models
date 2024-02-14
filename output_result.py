import json
import dataclasses
from flwr.server import History
from datetime import datetime

from arguments import Args
import logging


def save_run_as_json(config: Args, history: History, extra_data: dict=None, filename=None):
    if filename is None:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"results/{current_datetime}.json"
    config_dict = dataclasses.asdict(config)
    history_dict = vars(history)
    results = {
        "config": config_dict,
        "results": history_dict,
    }
    results.update(extra_data)
    json.dump(results, open(filename, "w"), indent=2)
    logging.info(f"Saved run to {filename}")


def create_experiment_json(name, epoch_per_round, num_client, optimizer, data, filename="experiment.json"):
    # Prepare the data
    experiment_data = {
        "name": name,
        "epochs_per_round": epoch_per_round,
        "num_client": num_client,
        "optimizer": optimizer,
        "teacher_loss": [teacher_loss for _, teacher_loss, in data['teacher_loss']],
        "teacher_accuracy": [teacher_accuracy for _, teacher_accuracy in data['teacher_accuracy']],
        "student_loss": [student_loss for _, student_loss in data['student_loss']],
        "student_accuracy": [student_accuracy for _, student_accuracy in data['student_accuracy']],
    }

    print(experiment_data)
