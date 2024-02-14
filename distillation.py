import torch
from transformers.modeling_outputs import SequenceClassifierOutput

from torch import nn


class Distillation(nn.Module):
    def __init__(self, teacher_model, student_model):
        super(Distillation, self).__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.mse_loss = nn.MSELoss()
        self.eps = 1

    def forward(self, input_ids=None, attention_mask=None, labels=None, who=None):
        assert who in (None, "teacher", "student"), f"invalid argument {who=}"
        if who == "teacher":
            return self.teacher(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if who == "student":
            return self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # type: SequenceClassifierOutput
        t_outputs = self.teacher(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # type: SequenceClassifierOutput
        s_outputs = self.student(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # task loss
        t_task_loss, t_logits, t_hidden_states, t_attentions = t_outputs.to_tuple()
        s_task_loss, s_logits, s_hidden_states, s_attentions = s_outputs.to_tuple()

        # KL-divergence loss
        s_kl_loss = self.kl_div_loss(t_logits.detach(
        ), s_logits) / (s_task_loss.detach() + t_task_loss.detach() + self.eps)
        t_kl_loss = self.kl_div_loss(s_logits.detach(
        ), t_logits) / (s_task_loss.detach() + t_task_loss.detach() + self.eps)

        # attention loss
        attention_layer_ratio = len(t_attentions) // len(s_attentions)

        # TODO if the attention dimensions don't line up for some pair of teacher/student model we want to evaluate (like bert-large vs bert)
        # then we need to apply a learnable matrix to the teacher attention to line up the dimensions before applying the mse loss
        assert t_attentions[0].size() == s_attentions[0].size(
        ), f"teacher and student attention dimensions don't match, {t_attentions[0].size()=}, {s_attentions[0].size()=}"
        mse_loss = 0.
        for s_layer in range(len(s_attentions)):
            t_layer = s_layer * attention_layer_ratio
            mse_loss += self.mse_loss(s_attentions[-s_layer - 1],
                                      t_attentions[-t_layer - 1])
            # adding the hidden-layer loss blows up the training
            # self.mse_loss(s_hidden_states[-s_layer - 1], t_hidden_states[-t_layer - 1])
        mse_loss = mse_loss / (s_task_loss + t_task_loss + self.eps)
        # outputs = last_hidden_state=[batch_size, sentence_length, features/embeddings] hidden_states=7 attentions=6

        total_loss = s_task_loss + t_task_loss + mse_loss + s_kl_loss + t_kl_loss
#     total_loss = s_task_loss + t_task_loss + s_kl_loss + t_kl_loss + mse_loss
        total_logits = s_outputs.logits + t_outputs.logits
        if torch.any(total_logits.isnan()):
            raise RuntimeError("logits blew up")
        return SequenceClassifierOutput(loss=total_loss, logits=total_logits)
