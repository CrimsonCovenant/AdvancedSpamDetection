"""
Architecture:
  RoBERTa-base (12 transformer layers, 768 hidden dim)
  pooler_output ([CLS] after linear + tanh, built into RoBERTa)
  Dropout(0.3)
  Classifier FC(768 → 2)

Fine-tuning settings:
  lr = 2e-5  (standard for transformer fine-tuning)
  epochs ≤ 5 with early stopping on validation F1
"""

import torch
import torch.nn as nn
from transformers import RobertaModel

ROBERTA_NAME = "roberta-base"


class RobertaClassifier(nn.Module):

    def __init__(
        self,
        model_name: str = ROBERTA_NAME,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.roberta    = RobertaModel.from_pretrained(model_name)
        hidden          = self.roberta.config.hidden_size   # 768
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
        returns logits: (batch, num_classes)
        """
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output  # (batch, 768)
        return self.classifier(self.dropout(pooled))

    def configure_optimizers(self, lr: float = 2e-5, weight_decay: float = 0.01):
        """
        Configure AdamW optimizer for this model.
        AdamW separates weight decay from the gradient update.
        """
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
        Define the Bayesian Optimization search space (Optuna) for RoBERTa.
        """
        return {
            "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
            "epochs": trial.suggest_int("epochs", 4, 8),
            "warmup": trial.suggest_float("warmup", 0.0, 0.15),
            "wd": trial.suggest_float("wd", 1e-4, 0.3, log=True),
            "bs": trial.suggest_categorical("bs", [8, 16, 32]),
            "cls_dropout": trial.suggest_float("cls_dropout", 0.1, 0.3),
        }


def build_roberta(model_name: str = ROBERTA_NAME,
                  dropout: float = 0.3) -> RobertaClassifier:
    print(f"  Loading RoBERTa: {model_name}")
    return RobertaClassifier(model_name=model_name, dropout=dropout)
