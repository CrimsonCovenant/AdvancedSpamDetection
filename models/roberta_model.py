"""
models/roberta_model.py
-----------------------
RoBERTa fine-tuned for binary SMS spam classification.  [2]

Architecture:
  RoBERTa-base (12 transformer layers, 768 hidden dim)
  → pooler_output  ([CLS] after linear + tanh, built into RoBERTa)
  → Dropout(0.3)
  → Classifier FC(768 → 2)

Motivation (proposal §1, §2):
  RoBERTa represents the high-performance transformer model in this
  study. It improves on BERT [3] through longer pretraining, larger
  mini-batches, dynamic masking, and removal of the NSP objective [2].
  Its byte-level BPE tokeniser (50,000 merges) offers an alternative
  subword segmentation to DistilBERT's WordPiece, which may behave
  differently on informal SMS abbreviations — an aspect examined in
  the comparative analysis.

Fine-tuning settings:
  lr = 2e-5  (standard for transformer fine-tuning)
  epochs ≤ 5 with early stopping on validation F1

References
----------
[2] Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining
    approach," arXiv:1907.11692, 2019.
[3] J. Devlin et al., "BERT: Pre-training of deep bidirectional
    transformers for language understanding," NAACL-HLT 2019.
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
        input_ids      : (batch, seq_len)
        attention_mask : (batch, seq_len)
        returns logits : (batch, num_classes)
        """
        out    = self.roberta(input_ids=input_ids,
                              attention_mask=attention_mask)
        pooled = out.pooler_output          # (batch, 768)
        return self.classifier(self.dropout(pooled))


def build_roberta(model_name: str = ROBERTA_NAME,
                  dropout: float = 0.3) -> RobertaClassifier:
    print(f"  Loading RoBERTa: {model_name}")
    return RobertaClassifier(model_name=model_name, dropout=dropout)
