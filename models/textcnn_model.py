"""
models/textcnn_model.py
-----------------------
Text Convolutional Neural Network — Kim (2014)  [5]

Architecture:
  Embedding (vocab × 100d, optionally GloVe-initialised [4])
  → Parallel Conv1d filters: kernel_size ∈ {2, 3, 4, 5}, 128 filters each
  → ReLU + max-over-time pooling per filter size
  → Concatenate (128 × 4 = 512 dims)
  → Dropout(0.5)
  → FC(512 → 2)

Motivation (proposal §2):
  TextCNN operates at the local pattern level and captures n-gram
  features such as repeated symbols and common spam phrases.
  Note: subword tokenization in transformers may handle abbreviations
  differently; TextCNN complements this through n-gram pattern capture.

References
----------
[4] Pennington et al., GloVe, EMNLP 2014.
[5] Kim, TextCNN, EMNLP 2014.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        num_classes: int = 2,
        num_filters: int = 128,
        filter_sizes: list = None,
        dropout: float = 0.5,
        pretrained_emb: np.ndarray = None,
        freeze_emb: bool = False,
    ):
        """
        Parameters
        ----------
        vocab_size     : vocabulary size (including PAD and UNK)
        embed_dim      : embedding dimension (100 to match GloVe-100d [4])
        num_classes    : 2 (ham / spam)
        num_filters    : number of feature maps per filter size
        filter_sizes   : list of kernel widths (n-gram windows)
        dropout        : dropout probability on penultimate layer
        pretrained_emb : np.ndarray (vocab_size, embed_dim) — GloVe init [4]
        freeze_emb     : freeze embedding weights when True
        """
        super().__init__()
        if filter_sizes is None:
            filter_sizes = [2, 3, 4, 5]

        # ── Embedding layer ────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(pretrained_emb, dtype=torch.float),
                requires_grad=not freeze_emb,
            )

        # ── Parallel convolutional filters (one per n-gram width) ──────────
        # Each Conv1d(embed_dim, num_filters, k) acts as a bigram/trigram/…
        # detector sliding over the token sequence.
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        """
        x      : (batch, seq_len)  token indices
        return : (batch, num_classes)  logits
        """
        # (batch, seq_len, embed_dim) → permute → (batch, embed_dim, seq_len)
        emb = self.embedding(x).permute(0, 2, 1)

        # Per-filter-size: convolve, ReLU, max-over-time pool
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(emb))                         # (B, F, L-k+1)
            h = F.max_pool1d(h, h.size(2)).squeeze(2)     # (B, F)
            pooled.append(h)

        # Concatenate all filter outputs → dropout → classify
        cat = self.dropout(torch.cat(pooled, dim=1))      # (B, F*n_sizes)
        return self.fc(cat)


def build_textcnn(vocab_size: int, embed_dim: int = 100,
                  pretrained_emb=None, freeze_emb: bool = False) -> TextCNN:
    return TextCNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        pretrained_emb=pretrained_emb,
        freeze_emb=freeze_emb,
    )
