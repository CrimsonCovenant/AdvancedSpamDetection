"""
utils/evaluation.py
-------------------
Evaluation metrics and visualisation.

Metrics (per proposal):
  - Matthews Correlation Coefficient (MCC)
  - PR-AUC  (Area Under the Precision-Recall Curve)
  - F1-Score (spam class)

Plots generated:
  1. results/comparison.png       — grouped bar chart (MCC / F1 / PR-AUC)
  2. results/pr_curves.png        — Precision-Recall curves overlaid
  3. results/confusion_matrices.png
  4. results/training_curves.png  — loss + val F1 per deep model
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PALETTE = {
    "SVM":        "#4C6EF5",
    "TextCNN":    "#F76707",
    "Bi-GRU":     "#2F9E44",
    "DistilBERT": "#AE3EC9",
    "RoBERTa":    "#E03131",
}


# ── Core metrics ──────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None, model_name: str = "") -> dict:
    """
    Compute MCC, F1, and PR-AUC.

    Parameters
    ----------
    y_true     : array-like int
    y_pred     : array-like int
    y_prob     : array-like float  (P(spam)); required for PR-AUC
    model_name : optional label for the printed header

    Returns
    -------
    dict with keys: mcc, f1, pr_auc
    """
    if model_name:
        print(f"\n[{model_name} — Test Results]")

    mcc    = matthews_corrcoef(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob) if y_prob is not None else None

    print(f"  MCC    : {mcc:.4f}")
    print(f"  F1     : {f1:.4f}")
    if pr_auc is not None:
        print(f"  PR-AUC : {pr_auc:.4f}")
    print(classification_report(y_true, y_pred,
                                target_names=["ham", "spam"],
                                zero_division=0))
    return {"mcc": mcc, "f1": f1, "pr_auc": pr_auc}


# ── Persist results ───────────────────────────────────────────────────────────
def save_results(metrics_dict: dict):
    path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved → {path}")


def load_results() -> dict:
    path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(path) as f:
        return json.load(f)


# ── 1. Grouped bar chart ──────────────────────────────────────────────────────
def plot_comparison(all_results: dict):
    """Grouped bar chart: MCC / F1 / PR-AUC for each model."""
    models  = list(all_results.keys())
    x       = np.arange(len(models))
    width   = 0.25
    metrics = [("MCC", "mcc"), ("F1-Score", "f1"), ("PR-AUC", "pr_auc")]
    alphas  = [1.0, 0.70, 0.42]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    for i, ((label, key), alpha) in enumerate(zip(metrics, alphas)):
        vals = [all_results[m].get(key) or 0 for m in models]
        bars = ax.bar(
            x + (i - 1) * width, vals, width,
            label=label,
            color=[PALETTE.get(m, "#888") for m in models],
            alpha=alpha, edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7.5, color="white",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, color="white", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color="white", fontsize=11)
    ax.set_title("SMS Spam Detection — Model Comparison",
                 color="white", fontsize=14, pad=14)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.yaxis.grid(True, color="#222", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend([l for l, _ in metrics],
              facecolor="#1A1A2E", edgecolor="#444",
              labelcolor="white", fontsize=10, loc="upper left")

    _save(fig, "comparison.png")


# ── 2. Precision-Recall curves ────────────────────────────────────────────────
def plot_pr_curves(pr_data: dict):
    """
    pr_data : { model_name: (y_true, y_prob) }
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    for name, (y_true, y_prob) in pr_data.items():
        if y_prob is None:
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                color=PALETTE.get(name, "#888"), linewidth=2.2)

    spam_rate = list(pr_data.values())[0][0].mean()
    ax.axhline(spam_rate, color="gray", linestyle="--",
               linewidth=1.2, label=f"Random baseline ({spam_rate:.3f})")

    ax.set_xlabel("Recall",    color="white", fontsize=11)
    ax.set_ylabel("Precision", color="white", fontsize=11)
    ax.set_title("Precision–Recall Curves", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(facecolor="#1A1A2E", edgecolor="#444",
              labelcolor="white", fontsize=9)

    _save(fig, "pr_curves.png")


# ── 3. Confusion matrices ─────────────────────────────────────────────────────
def plot_confusion_matrices(cm_data: dict):
    """
    cm_data : { model_name: (y_true, y_pred) }
    """
    n    = len(cm_data)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows))
    fig.patch.set_facecolor("#0D1117")
    axes = np.array(axes).flatten()

    for ax, (name, (y_true, y_pred)) in zip(axes, cm_data.items()):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax, cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"],
            linewidths=0.5, linecolor="#333",
        )
        ax.set_title(f"{name}\nFP={fp}  FN={fn}", color="white", fontsize=11)
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("Actual",    color="white")
        ax.tick_params(colors="white")
        ax.set_facecolor("#0D1117")

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Confusion Matrices — Test Set",
                 color="white", fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, "confusion_matrices.png")


# ── 4. Training curves ────────────────────────────────────────────────────────
def plot_training_curves(histories: dict):
    """
    histories : { model_name: {'train_loss', 'val_loss', 'val_f1'} }
    """
    n    = len(histories)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, h) in zip(axes, histories.items()):
        ep = range(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], label="Train loss",
                color=PALETTE.get(name, "#888"), linewidth=2)
        ax.plot(ep, h["val_loss"],   label="Val loss",
                color=PALETTE.get(name, "#888"), linewidth=2,
                linestyle="--", alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(ep, h["val_f1"], label="Val F1",
                 color="#FFD43B", linewidth=1.8, linestyle=":")
        ax2.set_ylabel("Val F1", fontsize=9)
        ax2.set_ylim(0, 1.05)

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")

    plt.suptitle("Training Curves — Deep Learning Models", fontsize=13)
    plt.tight_layout()
    _save(fig, "training_curves.png")


# ── Utility ───────────────────────────────────────────────────────────────────
def _save(fig, filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")
