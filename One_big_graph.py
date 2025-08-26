#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def load_data(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    p = argparse.ArgumentParser(description="Plot training/evaluation metrics from JSON.")
    p.add_argument("json_path", type=Path, help="Path to the JSON file (format shown in prompt).")
    p.add_argument("--out", type=Path, default=Path("metrics_plot.png"), help="Output image file.")
    p.add_argument("--title", type=str, default=None, help="Optional plot title.")
    args = p.parse_args()

    data = load_data(args.json_path)
    evals = data.get("all_evaluations", [])
    if not evals:
        raise SystemExit("No 'all_evaluations' found in JSON.")

    x = list(range(len(evals)))  # simple sequential index along time

    # Series
    train_q = [e["train_quartet_dist"] for e in evals]
    test_q  = [e["test_quartet_dist"]  for e in evals]
    train_rf = [e["train_rf"] for e in evals]
    test_rf  = [e["test_rf"]  for e in evals]
    loss     = [e["loss"]     for e in evals]

    # NJ baselines
    nj = (data.get("base_eval", {}) or {}).get("nj", {}) or {}
    nj_train_q = ((nj.get("train") or {}).get("quartet_dist"))
    nj_test_q  = ((nj.get("test")  or {}).get("quartet_dist"))
    nj_train_rf = ((nj.get("train") or {}).get("rf"))
    nj_test_rf  = ((nj.get("test")  or {}).get("rf"))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot primary metrics (matplotlib will auto-assign distinct colors)
    ln_train_q, = ax.plot(x, train_q, label="Train Q-Dist", linewidth=1.8)
    ln_test_q,  = ax.plot(x, test_q,  label="Test Q-Dist",  linewidth=1.8)
    ln_train_rf, = ax.plot(x, train_rf, label="Train RF", linewidth=1.8)
    ln_test_rf,  = ax.plot(x, test_rf,  label="Test RF",  linewidth=1.8)

    # Secondary axis for loss
    ax2 = ax.twinx()
    ln_loss, = ax2.plot(x, loss, label="Loss", linewidth=1.8)

    # Add NJ baselines as dotted lines in matching colors (not for loss)
    if nj_train_q is not None:
        ax.axhline(nj_train_q, linestyle="--", linewidth=1.2, color=ln_train_q.get_color(),
                   label="NJ Train Q-Dist")
    if nj_test_q is not None:
        ax.axhline(nj_test_q, linestyle="--", linewidth=1.2, color=ln_test_q.get_color(),
                   label="NJ Test Q-Dist")
    if nj_train_rf is not None:
        ax.axhline(nj_train_rf, linestyle="--", linewidth=1.2, color=ln_train_rf.get_color(),
                   label="NJ Train RF")
    if nj_test_rf is not None:
        ax.axhline(nj_test_rf, linestyle="--", linewidth=1.2, color=ln_test_rf.get_color(),
                   label="NJ Test RF")

    # Labels & legend
    ax.set_xlabel("Evaluation index (ordered by epoch/step)")
    ax.set_ylabel("RF / Q-Dist")
    ax2.set_ylabel("Loss")

    # Optional title
    title = args.title
    if not title:
        cfg = data.get("config", {})
        ds = cfg.get("dataset_name", "dataset")
        title = f"Metrics over evaluations – {ds}"
    ax.set_title(title)

    # Build a combined legend from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + [ln_loss] + h2, l1 + [ln_loss.get_label()] + l2, loc="upper right", ncol=2)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out.resolve()}")

if __name__ == "__main__":
    main()
