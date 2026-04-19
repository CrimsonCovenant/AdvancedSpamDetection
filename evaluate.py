"""
evaluate.py
-----------
Load raw_results.json produced by train.py and generate:

  results/comparison.png        — grouped bar chart (MCC / F1 / PR-AUC)
  results/pr_curves.png         — Precision-Recall curves
  results/confusion_matrices.png
  results/training_curves.png   — loss + val F1 per deep model
  results/metrics.json          — clean metrics dict

Usage
-----
  python evaluate.py
  python evaluate.py --results_dir results
"""

import argparse
import json
import os
import numpy as np

from utils.evaluation import (
    plot_comparison,
    plot_pr_curves,
    plot_confusion_matrices,
    plot_training_curves,
    save_results,
)

# Display names map raw keys → formatted labels
DISPLAY = {
    "svm":        "SVM",
    "textcnn":    "TextCNN",
    "bigru":      "Bi-GRU",
    "distilbert": "DistilBERT",
    "roberta":    "RoBERTa",
}


def main():
    p = argparse.ArgumentParser(description="SMS Spam Detection — Evaluation")
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    raw_path = os.path.join(args.results_dir, "raw_results.json")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"{raw_path} not found. Run `python train.py` first."
        )
    with open(raw_path) as f:
        raw = json.load(f)

    # ── Remap to display names ─────────────────────────────────────────────
    metrics_dict, pr_data, cm_data, histories = {}, {}, {}, {}

    for key, res in raw.items():
        label = DISPLAY.get(key, key)
        metrics_dict[label] = res["metrics"]

        y_true = np.array(res["y_true"])
        y_prob = np.array(res["y_prob"]) if res.get("y_prob") else None
        y_pred = np.array(res["y_pred"])

        pr_data[label] = (y_true, y_prob)
        cm_data[label] = (y_true, y_pred)

        if "history" in res:
            histories[label] = res["history"]

    # ── Save clean metrics ─────────────────────────────────────────────────
    save_results(metrics_dict)

    # ── Generate plots ─────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_comparison(metrics_dict)
    plot_pr_curves(pr_data)
    plot_confusion_matrices(cm_data)
    if histories:
        plot_training_curves(histories)

    # ── Print final summary ────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print(f"  {'Model':<14} {'MCC':>8} {'F1':>8} {'PR-AUC':>10}")
    print("-" * 58)
    for name, m in metrics_dict.items():
        pr = m.get("pr_auc") or 0.0
        print(f"  {name:<14} {m['mcc']:8.4f} {m['f1']:8.4f} {pr:10.4f}")
    print("=" * 58)

    best = lambda k: max(metrics_dict, key=lambda n: metrics_dict[n].get(k) or 0)
    print(f"\n  Best MCC    : {best('mcc')}")
    print(f"  Best F1     : {best('f1')}")
    print(f"  Best PR-AUC : {best('pr_auc')}")
    print(f"\nAll outputs saved to ./{args.results_dir}/")


if __name__ == "__main__":
    main()
