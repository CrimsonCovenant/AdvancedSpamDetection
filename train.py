"""
train.py
--------
Main training script for all five SMS spam detection models.

Data split (per proposal):
  - Stratified 80/20 train-test split
  - 10% of training set used as internal validation partition
    (for early stopping and model selection only)
  - Test set is never used during training

Usage
-----
  python train.py                     # train all five models
  python train.py --model svm
  python train.py --model textcnn
  python train.py --model bigru
  python train.py --model distilbert
  python train.py --model roberta
  python train.py --model textcnn --epochs 15 --batch_size 64
  python train.py --device cpu
  python train.py --no_glove          # Bi-GRU / TextCNN with random embeddings
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from utils.data_loader import (
    load_raw_dataframe,
    split_data,
    prepare_svm_data,
    prepare_dl_data,
    prepare_transformer_data,
    load_glove_embeddings,
    compute_class_weights,
)
from utils.trainer    import train_pytorch_model, evaluate_pytorch_model
from utils.evaluation import compute_metrics, save_results

from models.svm_model        import train_svm, predict_svm
from models.textcnn_model    import build_textcnn
from models.bigru_model      import build_bigru
from models.distilbert_model import build_distilbert, DISTILBERT_NAME
from models.roberta_model    import build_roberta, ROBERTA_NAME

RESULTS_DIR = "results"
GLOVE_PATH  = os.path.join("data", "glove.6B.100d.txt")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Device ────────────────────────────────────────────────────────────────────
def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model runners ─────────────────────────────────────────────────────────────

def run_svm(train_df, val_df, test_df) -> dict:
    """
    SVM [6] with TF-IDF features.
    val_df used only for reporting intermediate performance.
    """
    print("\n" + "=" * 56)
    print("  MODEL 1: SVM (LinearSVC + TF-IDF)  [6]")
    print("=" * 56)

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), _ = prepare_svm_data(
        train_df, val_df, test_df
    )
    model = train_svm(X_tr, y_tr, tune=True)

    print("\n  [Validation]")
    v_pred, v_prob = predict_svm(model, X_val)
    compute_metrics(y_val, v_pred, v_prob)

    y_pred, y_prob = predict_svm(model, X_te)
    metrics = compute_metrics(y_te, y_pred, y_prob, model_name="SVM")
    return {"metrics": metrics,
            "y_true": y_te.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist()}


def run_textcnn(train_df, val_df, test_df, args, device) -> dict:
    """TextCNN [5] with optional GloVe-100d embeddings [4]."""
    print("\n" + "=" * 56)
    print("  MODEL 2: TextCNN  [5]")
    print("=" * 56)

    tr_loader, val_loader, te_loader, vocab = prepare_dl_data(
        train_df, val_df, test_df, batch_size=args.batch_size
    )
    glove_emb = None
    if not args.no_glove:
        glove_emb = load_glove_embeddings(vocab, GLOVE_PATH)

    model  = build_textcnn(len(vocab), embed_dim=100, pretrained_emb=glove_emb)
    cw     = compute_class_weights(train_df["label_int"].values, device=str(device))
    crit   = nn.CrossEntropyLoss(weight=cw)

    train_pytorch_model(
        model, tr_loader, val_loader, cw, device,
        epochs=args.epochs, lr=args.lr or 1e-3,
        patience=args.patience, is_transformer=False,
    )
    _, _, y_pred, y_prob = evaluate_pytorch_model(
        model, te_loader, crit, device, is_transformer=False
    )
    y_te = test_df["label_int"].values
    metrics = compute_metrics(y_te, y_pred, y_prob, model_name="TextCNN")
    return {"metrics": metrics,
            "y_true": y_te.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist()}


def run_bigru(train_df, val_df, test_df, args, device) -> dict:
    """Bi-GRU with GloVe-100d embeddings [4] and attention pooling."""
    print("\n" + "=" * 56)
    print("  MODEL 3: Bi-GRU + GloVe [4] + Attention")
    print("=" * 56)

    tr_loader, val_loader, te_loader, vocab = prepare_dl_data(
        train_df, val_df, test_df, batch_size=args.batch_size
    )
    glove_emb = None
    if not args.no_glove:
        glove_emb = load_glove_embeddings(vocab, GLOVE_PATH)

    model  = build_bigru(len(vocab), embed_dim=100, pretrained_emb=glove_emb)
    cw     = compute_class_weights(train_df["label_int"].values, device=str(device))
    crit   = nn.CrossEntropyLoss(weight=cw)

    train_pytorch_model(
        model, tr_loader, val_loader, cw, device,
        epochs=args.epochs, lr=args.lr or 1e-3,
        patience=args.patience, is_transformer=False,
    )
    _, _, y_pred, y_prob = evaluate_pytorch_model(
        model, te_loader, crit, device, is_transformer=False
    )
    y_te = test_df["label_int"].values
    metrics = compute_metrics(y_te, y_pred, y_prob, model_name="Bi-GRU")
    return {"metrics": metrics,
            "y_true": y_te.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist()}


def run_distilbert(train_df, val_df, test_df, args, device) -> dict:
    """DistilBERT [1] fine-tuned for spam classification."""
    print("\n" + "=" * 56)
    print("  MODEL 4: DistilBERT  [1]")
    print("=" * 56)

    bs = min(args.batch_size, 16)
    tr_loader, val_loader, te_loader, _ = prepare_transformer_data(
        train_df, val_df, test_df, model_name=DISTILBERT_NAME, batch_size=bs
    )
    model  = build_distilbert()
    cw     = compute_class_weights(train_df["label_int"].values, device=str(device))
    crit   = nn.CrossEntropyLoss(weight=cw)

    train_pytorch_model(
        model, tr_loader, val_loader, cw, device,
        epochs=args.epochs, lr=args.lr or 2e-5,
        patience=args.patience, is_transformer=True,
    )
    _, _, y_pred, y_prob = evaluate_pytorch_model(
        model, te_loader, crit, device, is_transformer=True
    )
    y_te = test_df["label_int"].values
    metrics = compute_metrics(y_te, y_pred, y_prob, model_name="DistilBERT")
    return {"metrics": metrics,
            "y_true": y_te.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist()}


def run_roberta(train_df, val_df, test_df, args, device) -> dict:
    """RoBERTa [2] fine-tuned for spam classification."""
    print("\n" + "=" * 56)
    print("  MODEL 5: RoBERTa  [2]")
    print("=" * 56)

    bs = min(args.batch_size, 16)
    tr_loader, val_loader, te_loader, _ = prepare_transformer_data(
        train_df, val_df, test_df, model_name=ROBERTA_NAME, batch_size=bs
    )
    model  = build_roberta()
    cw     = compute_class_weights(train_df["label_int"].values, device=str(device))
    crit   = nn.CrossEntropyLoss(weight=cw)

    train_pytorch_model(
        model, tr_loader, val_loader, cw, device,
        epochs=args.epochs, lr=args.lr or 2e-5,
        patience=args.patience, is_transformer=True,
    )
    _, _, y_pred, y_prob = evaluate_pytorch_model(
        model, te_loader, crit, device, is_transformer=True
    )
    y_te = test_df["label_int"].values
    metrics = compute_metrics(y_te, y_pred, y_prob, model_name="RoBERTa")
    return {"metrics": metrics,
            "y_true": y_te.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist()}


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SMS Spam Detection — Training")
    p.add_argument("--model", default="all",
                   choices=["all", "svm", "textcnn", "bigru",
                            "distilbert", "roberta"])
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--patience",   type=int,   default=3)
    p.add_argument("--no_glove",   action="store_true")
    p.add_argument("--device",     default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device(args.device)
    print(f"Device : {device}")

    # ── Load data & produce splits ────────────────────────────────────────
    print("\nLoading dataset ...")
    df = load_raw_dataframe()

    print("\nSplitting data (stratified 80/20 + internal 10% validation) ...")
    train_df, val_df, test_df = split_data(df)

    # ── Select models to run ──────────────────────────────────────────────
    run_map = {
        "svm":        lambda: run_svm(train_df, val_df, test_df),
        "textcnn":    lambda: run_textcnn(train_df, val_df, test_df, args, device),
        "bigru":      lambda: run_bigru(train_df, val_df, test_df, args, device),
        "distilbert": lambda: run_distilbert(train_df, val_df, test_df, args, device),
        "roberta":    lambda: run_roberta(train_df, val_df, test_df, args, device),
    }
    targets = list(run_map) if args.model == "all" else [args.model]

    # ── Train ─────────────────────────────────────────────────────────────
    all_results = {}
    for name in targets:
        all_results[name] = run_map[name]()

    # ── Persist raw results ───────────────────────────────────────────────
    raw_path = os.path.join(RESULTS_DIR, "raw_results.json")
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results → {raw_path}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print(f"  {'Model':<14} {'MCC':>8} {'F1':>8} {'PR-AUC':>8}")
    print("-" * 56)
    metrics_only = {}
    for name, res in all_results.items():
        m  = res["metrics"]
        pr = m["pr_auc"] if m["pr_auc"] is not None else float("nan")
        print(f"  {name:<14} {m['mcc']:8.4f} {m['f1']:8.4f} {pr:8.4f}")
        metrics_only[name] = m
    print("=" * 56)

    save_results(metrics_only)
    print("\nDone. Run `python evaluate.py` to generate comparison plots.")


if __name__ == "__main__":
    main()
