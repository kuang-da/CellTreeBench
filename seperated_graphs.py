#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def load(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def extract_series(evals):
    x = list(range(len(evals)))
    train_q = [e["train_quartet_dist"] for e in evals]
    test_q  = [e["test_quartet_dist"]  for e in evals]
    train_rf = [e["train_rf"] for e in evals]
    test_rf  = [e["test_rf"]  for e in evals]
    loss     = [e["loss"]     for e in evals]
    return x, train_q, test_q, train_rf, test_rf, loss

def nj_lines(base_eval):
    nj = (base_eval or {}).get("nj", {}) or {}
    return {
        "train_q": (nj.get("train") or {}).get("quartet_dist"),
        "test_q":  (nj.get("test")  or {}).get("quartet_dist"),
        "train_rf": (nj.get("train") or {}).get("rf"),
        "test_rf":  (nj.get("test")  or {}).get("rf"),
    }

def plot_loss(x, loss, title, outpath):
    fig, ax = plt.subplots(figsize=(12, 5))
    ln_loss, = ax.plot(x, loss, label="Loss", linewidth=1.8)
    ax.set_title(f"{title} – Loss")
    ax.set_xlabel("Evaluation index")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_quartet(x, train_q, test_q, nj, title, outpath):
    fig, ax = plt.subplots(figsize=(12, 5))
    ln_train_q, = ax.plot(x, train_q, label="Train Q-Dist", linewidth=1.8)
    ln_test_q,  = ax.plot(x, test_q,  label="Test Q-Dist",  linewidth=1.8)

    # NJ baselines in matching colors (dotted)
    if nj.get("train_q") is not None:
        ax.axhline(nj["train_q"], linestyle="--", linewidth=1.2,
                   color=ln_train_q.get_color(), label="NJ Train Q-Dist")
    if nj.get("test_q") is not None:
        ax.axhline(nj["test_q"], linestyle="--", linewidth=1.2,
                   color=ln_test_q.get_color(), label="NJ Test Q-Dist")

    ax.set_title(f"{title} – Quartet Distance")
    ax.set_xlabel("Evaluation index")
    ax.set_ylabel("Quartet Distance")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_rf(x, train_rf, test_rf, nj, title, outpath):
    fig, ax = plt.subplots(figsize=(12, 5))
    ln_train_rf, = ax.plot(x, train_rf, label="Train RF", linewidth=1.8)
    ln_test_rf,  = ax.plot(x, test_rf,  label="Test RF",  linewidth=1.8)

    # NJ baselines in matching colors (dotted)
    if nj.get("train_rf") is not None:
        ax.axhline(nj["train_rf"], linestyle="--", linewidth=1.2,
                   color=ln_train_rf.get_color(), label="NJ Train RF")
    if nj.get("test_rf") is not None:
        ax.axhline(nj["test_rf"], linestyle="--", linewidth=1.2,
                   color=ln_test_rf.get_color(), label="NJ Test RF")

    ax.set_title(f"{title} – RF Distance")
    ax.set_xlabel("Evaluation index")
    ax.set_ylabel("RF (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Plot split charts: Loss, Q-Dist, RF.")
    ap.add_argument("json_path", type=Path, help="Path to JSON log.")
    ap.add_argument("--out_prefix", type=Path, default=Path("metrics"),
                    help="Prefix for output images (e.g., 'metrics').")
    ap.add_argument("--title", type=str, default=None, help="Optional plot title prefix.")
    args = ap.parse_args()

    blob = load(args.json_path)
    evals = blob.get("all_evaluations", [])
    if not evals:
        raise SystemExit("No 'all_evaluations' found in JSON.")

    x, train_q, test_q, train_rf, test_rf, loss = extract_series(evals)
    cfg = blob.get("config", {})
    dataset = cfg.get("dataset_name", "dataset")
    title = args.title or f"Metrics – {dataset}"
    nj = nj_lines(blob.get("base_eval"))

    # Paths
    loss_out = args.out_prefix.with_name(f"{args.out_prefix.name}_loss.png")
    quartet_out = args.out_prefix.with_name(f"{args.out_prefix.name}_quartet.png")
    rf_out = args.out_prefix.with_name(f"{args.out_prefix.name}_rf.png")

    # Plots
    plot_loss(x, loss, title, loss_out)
    plot_quartet(x, train_q, test_q, nj, title, quartet_out)
    plot_rf(x, train_rf, test_rf, nj, title, rf_out)

    print(f"Saved:\n  {loss_out.resolve()}\n  {quartet_out.resolve()}\n  {rf_out.resolve()}")

if __name__ == "__main__":
    main()
