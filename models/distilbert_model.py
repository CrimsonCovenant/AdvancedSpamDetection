"""
models/distilbert_model.py
--------------------------
DistilBERT fine-tuned for binary SMS spam classification.  [1]

Architecture:
  DistilBERT-base-uncased (6 transformer layers, 768 hidden dim)
  → [CLS] token representation (last hidden state, position 0)
  → Pre-classifier FC(768 → 768) + ReLU
  → Dropout(0.3)
  → Classifier FC(768 → 2)

Motivation (proposal §1, §2):
  DistilBERT leverages contextual embeddings and self-attention to
  capture semantic relationships even with limited tokens [1, 2, 3].
  Its subword tokenisation (WordPiece) can improve robustness to
  misspellings and rare words [1, 2], though its effectiveness on
  highly abbreviated SMS language is an empirical question explored
  in this comparative study.

  DistilBERT is 40% smaller and 60% faster than BERT-base while
  retaining ~97% of BERT's performance on GLUE [1].

Fine-tuning settings:
  lr = 2e-5  (standard for transformer fine-tuning)
  epochs ≤ 5 with early stopping on validation F1

Reference
---------
[1] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT: A
    distilled version of BERT: Smaller, faster, cheaper and lighter,"
    arXiv:1910.01108, 2019.
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


def build_distilbert(model_name: str = DISTILBERT_NAME,
                     dropout: float = 0.3) -> DistilBertClassifier:
    print(f"  Loading DistilBERT: {model_name}")
    return DistilBertClassifier(model_name=model_name, dropout=dropout)
