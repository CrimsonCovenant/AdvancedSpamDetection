"""
utils/trainer.py
----------------
Shared PyTorch training loop used by TextCNN [5], Bi-GRU,
DistilBERT [1], and RoBERTa [2].

Design decisions aligned with the proposal:
  - Weighted cross-entropy loss to address class imbalance
  - AdamW optimiser with weight decay for regularisation
  - Gradient clipping (max_norm=1.0) for transformer stability
  - ReduceLROnPlateau scheduler monitoring validation F1
  - Early stopping on validation F1 (patience configurable)
  - Best model state restored after early stopping

The internal val_loader is drawn from the training set
(VAL_FRAC=10%) and is separate from the held-out test set.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from tqdm import tqdm


def train_pytorch_model(
    model: nn.Module,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    patience: int = 3,
    is_transformer: bool = False,
):
    """
    Train a PyTorch classifier with early stopping.

    Parameters
    ----------
    model          : nn.Module
    train_loader   : DataLoader  (training partition)
    val_loader     : DataLoader  (internal validation partition)
    class_weights  : FloatTensor (2,) — inverse-frequency weights
    device         : torch.device
    epochs         : maximum epochs
    lr             : initial learning rate
    patience       : early-stopping patience (epochs without val F1 gain)
    is_transformer : True for DistilBERT / RoBERTa batches (dict format)

    Returns
    -------
    best_state : state_dict at epoch with highest val F1
    history    : dict {'train_loss', 'val_loss', 'val_f1'}
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_f1  = -1.0
    best_state   = copy.deepcopy(model.state_dict())
    patience_ctr = 0
    history      = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, epochs + 1):

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch in tqdm(train_loader,
                          desc=f"Epoch {epoch}/{epochs} [train]",
                          leave=False):
            optimizer.zero_grad()
            logits, labels = _forward(model, batch, device, is_transformer)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        # ── Validation ────────────────────────────────────────────────────────
        val_loss, val_f1, _, _ = evaluate_pytorch_model(
            model, val_loader, criterion, device, is_transformer
        )
        scheduler.step(val_f1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(f"  Epoch {epoch:02d} | train_loss={avg_train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_F1={val_f1:.4f}")

        # ── Early stopping ────────────────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
                break

    model.load_state_dict(best_state)
    print(f"  Best val F1: {best_val_f1:.4f}")
    return best_state, history


def evaluate_pytorch_model(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    is_transformer: bool = False,
):
    """
    Run inference on a DataLoader.

    Returns
    -------
    avg_loss  : float
    f1        : float  (spam class)
    y_pred    : np.ndarray of int
    y_prob    : np.ndarray of float  (P(spam))
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            logits, labels = _forward(model, batch, device, is_transformer)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches  += 1

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    f1       = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, f1, np.array(all_preds), np.array(all_probs)


# ── Internal helper ───────────────────────────────────────────────────────────
def _forward(model, batch, device, is_transformer):
    """Unpack batch and run forward pass; return (logits, labels)."""
    if is_transformer:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        logits         = model(input_ids, attention_mask=attention_mask)
    else:
        inputs, labels = batch
        inputs  = inputs.to(device)
        labels  = labels.to(device)
        logits  = model(inputs)
    return logits, labels
