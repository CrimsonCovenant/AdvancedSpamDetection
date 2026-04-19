"""
models/bigru_model.py
---------------------
Bidirectional Gated Recurrent Unit (Bi-GRU) classifier
with GloVe pretrained embeddings [4] and additive attention pooling.

Architecture:
  Embedding (vocab × 100d, GloVe-100d init [4])
  → Bi-GRU (2 layers, hidden=128 per direction → 256 combined)
  → Additive attention over timesteps
  → Context vector (weighted sum)
  → Dropout(0.4)
  → FC(256 → 2)

Motivation (proposal §1):
  Bi-GRU is selected as a sequence model using pretrained GloVe
  embeddings [4], which reduce sparsity by representing words in a
  dense vector space. The bidirectional pass captures left-to-right
  and right-to-left context; attention pooling focuses on the most
  discriminative tokens regardless of position.

Note: There is no single paper that introduces Bi-GRU as a named
model. It combines the bidirectional RNN concept with GRU gating.
GloVe embeddings are cited as [4] per the proposal reference list.

Reference
---------
[4] J. Pennington, R. Socher, and C. D. Manning, "GloVe: Global
    vectors for word representation," EMNLP 2014.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGRU(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.4,
        pretrained_emb: np.ndarray = None,
        freeze_emb: bool = False,
    ):
        """
        Parameters
        ----------
        vocab_size     : vocabulary size
        embed_dim      : embedding dimension (100 to match GloVe-100d [4])
        hidden_size    : GRU hidden units per direction
        num_layers     : stacked GRU layers
        num_classes    : 2 (ham / spam)
        dropout        : dropout probability
        pretrained_emb : GloVe embedding matrix [4]
        freeze_emb     : freeze GloVe weights when True
        """
        super().__init__()

        # ── Embedding (GloVe init) ─────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(pretrained_emb, dtype=torch.float),
                requires_grad=not freeze_emb,
            )

        # ── Bidirectional GRU ──────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        gru_out_dim = hidden_size * 2       # bidirectional concatenation

        # ── Additive attention ─────────────────────────────────────────────
        # score_t = w · h_t  (scalar per timestep)
        # α_t = softmax(score)
        # context = Σ α_t · h_t
        self.attention = nn.Linear(gru_out_dim, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(gru_out_dim, num_classes)

    def forward(self, x):
        """
        x      : (batch, seq_len)  token indices
        return : (batch, num_classes)  logits
        """
        emb     = self.dropout(self.embedding(x))        # (B, L, E)
        gru_out, _ = self.gru(emb)                       # (B, L, H*2)

        # Attention
        scores  = self.attention(gru_out).squeeze(-1)    # (B, L)
        alpha   = torch.softmax(scores, dim=1)           # (B, L)
        context = torch.bmm(alpha.unsqueeze(1), gru_out).squeeze(1)  # (B, H*2)

        return self.fc(self.dropout(context))


def build_bigru(vocab_size: int, embed_dim: int = 100,
                pretrained_emb=None, freeze_emb: bool = False) -> BiGRU:
    return BiGRU(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        pretrained_emb=pretrained_emb,
        freeze_emb=freeze_emb,
    )
