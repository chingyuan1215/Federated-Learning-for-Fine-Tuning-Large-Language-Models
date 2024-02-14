import flwr as fl
from flwr.common.logger import log
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from logging import WARNING

import torch
from torch import nn

from distillation import Distillation

from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from evaluate import load as load_metric

from transformers import AdamW, get_scheduler

class CustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, teacher: nn.Module, student: nn.Module, train_loader, test_loader, teacher_pretrain_epochs, kd_epochs, distil_lr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.peft_state_dict_keys = get_peft_model_state_dict(student).keys()
        self.device = next(student.parameters()).device
        self.teacher_pretrain_epochs = teacher_pretrain_epochs
        self.kd_epochs = kd_epochs
        self.distil_lr = distil_lr
    
    def evaluate_model(self, model):
        metric = load_metric("accuracy")
        model.eval()
        loss = 0
        for batch in self.test_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            loss += outputs.loss.item()
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        loss /= len(self.test_loader.dataset)
        return loss, metric.compute()["accuracy"]
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        t_loss, t_acc = self.evaluate_model(self.teacher)
        s_loss, s_acc = self.evaluate_model(self.student)
        metrics = dict(
            teacher_loss=t_loss,
            teacher_accuracy=t_acc,
            student_loss=s_loss,
            student_accuracy=s_acc,
        )
        return (t_loss + s_loss) / 2, metrics
    
    def train(self, distiller: Distillation, epochs, who=None):
        optimizer = AdamW(distiller.parameters(), lr=self.distil_lr, no_deprecation_warning=True)
        num_training_steps = epochs * len(self.train_loader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        distiller.train()
        for _ in range(epochs):
            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = distiller(**batch, who=who)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

    def set_student_parameters(self, parameters):
        params_dict = zip(self.peft_state_dict_keys, parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        set_peft_model_state_dict(self.student, state_dict)

    def get_student_parameters(self):
            state_dict = get_peft_model_state_dict(self.student)
            return ndarrays_to_parameters([val.cpu().numpy() for _, val in state_dict.items()])

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # KD
        parameters_aggregated = aggregate(weights_results)
        self.set_student_parameters(parameters_aggregated)
        distiller = Distillation(self.teacher, self.student)
        distiller.to(self.device)
        # train only teacher first on the dataset
        self.train(distiller, who="teacher", epochs=self.teacher_pretrain_epochs)
        # then train both with distillation loss
        self.train(distiller, epochs=self.kd_epochs)

        parameters_aggregated = self.get_student_parameters()
        metrics_aggregated = {}
        # Aggregate custom metrics if aggregation fn was provided
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics)
                           for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        return parameters_aggregated, metrics_aggregated
