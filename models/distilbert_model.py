"""
DistilBERT fine-tuned for binary SMS spam classification.

  DistilBERT is 40% smaller and 60% faster than BERT-base while
  retaining ~97% of BERT's performance on GLUE

Fine-tuning settings:
  lr = 2e-5
  epochs ≤ 5 with early stopping on validation F1

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel

DISTILBERT_NAME = "distilbert-base-uncased"


class DistilBertClassifier(nn.Module):

    def __init__(
        self,
        model_name: str = DISTILBERT_NAME,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.distilbert     = DistilBertModel.from_pretrained(model_name)
        hidden              = self.distilbert.config.hidden_size   # 768
        self.pre_classifier = nn.Linear(hidden, hidden)
        self.dropout        = nn.Dropout(dropout)
        self.classifier     = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids      : (batch, seq_len)
        attention_mask : (batch, seq_len)
        returns logits : (batch, num_classes)
        """
        out = self.distilbert(input_ids=input_ids,
                              attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]        # [CLS] token
        cls = F.relu(self.pre_classifier(cls))
        cls = self.dropout(cls)
        return self.classifier(cls)

    def configure_optimizers(self, lr: float = 2e-5, weight_decay: float = 0.01):
        """
        Configure AdamW optimizer for this model.
        AdamW separates weight decay from the gradient update, which is
        crucial for transformer generalization.
        """
        # Exclude bias and LayerNorm weights from weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    @staticmethod
    def suggest_bayesian_hyperparameters(trial):
        """
        Define the Bayesian Optimization search space (Optuna) for DistilBERT.
        Using TPE (Tree-structured Parzen Estimator).
        """
        return {
            "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
            "epochs": trial.suggest_int("epochs", 4, 8),
            "warmup": trial.suggest_float("warmup", 0.0, 0.15),
            "wd": trial.suggest_float("wd", 1e-4, 0.3, log=True),
            "bs": trial.suggest_categorical("bs", [8, 16, 32]),
            "cls_dropout": trial.suggest_float("cls_dropout", 0.1, 0.3),
        }


def build_distilbert(model_name: str = DISTILBERT_NAME,
                     dropout: float = 0.3) -> DistilBertClassifier:
    print(f"  Loading DistilBERT: {model_name}")
    return DistilBertClassifier(model_name=model_name, dropout=dropout)
