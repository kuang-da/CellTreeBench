#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# ---------- IO ----------

def load_any(path: Path):
    """Load dict from JSON or Pickle. Tries extension, then fallbacks."""
    try:
        if path.suffix.lower() == ".json":
            with open(path, "r") as f:
                return json.load(f)
        if path.suffix.lower() in {".pkl", ".pickle"}:
            with open(path, "rb") as f:
                return pickle.load(f)
        # Fallback attempts
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        raise SystemExit(f"Failed to load {path}: {e}")

# ---------- Extract & Ticks ----------

def extract_series(evals):
    x = list(range(len(evals)))
    labels = [f"{e.get('epoch','?')}-{e.get('step','?')}" for e in evals]
    train_q = [e["train_quartet_dist"] for e in evals]
    test_q  = [e["test_quartet_dist"]  for e in evals]
    train_rf = [e["train_rf"] for e in evals]
    test_rf  = [e["test_rf"]  for e in evals]
    loss     = [e["loss"]     for e in evals]
    return x, labels, train_q, test_q, train_rf, test_rf, loss

def apply_epoch_step_ticks(ax, x, labels, max_ticks=12):
    n = len(x)
    stride = 1 if n <= max_ticks else max(1, n // max_ticks)
    idx = list(range(0, n, stride))
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] for i in idx], rotation=45, ha="right")

def add_frame(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.2)
        s.set_color("black")

# ---------- NJ Baseline Extraction ----------

def get_nj_baselines(blob):
    """Return dict of NJ baselines (train_q, test_q, train_rf, test_rf)."""
    bea = blob.get("base_eval_avg") or {}
    nj_new = bea.get("NJ") or bea.get("nj") or {}
    if nj_new:
        return {
            "train_q": nj_new.get("quartet_dist_train"),
            "test_q":  nj_new.get("quartet_dist_test"),
            "train_rf": nj_new.get("rf_train"),
            "test_rf":  nj_new.get("rf_test"),
        }

    be = blob.get("base_eval") or {}
    nj_old = be.get("nj") or be.get("NJ") or {}
    if nj_old:
        tr = nj_old.get("train") or {}
        te = nj_old.get("test") or {}
        return {
            "train_q": tr.get("quartet_dist"),
            "test_q":  te.get("quartet_dist"),
            "train_rf": tr.get("rf"),
            "test_rf":  te.get("rf"),
        }

    return {"train_q": None, "test_q": None, "train_rf": None, "test_rf": None}

# ---------- Plotters ----------

def plot_loss(x, labels, loss, title, outpath, zero_baseline=False):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, loss, label="Loss", linewidth=1.8)
    ax.set_title(f"{title} – Loss")
    ax.set_xlabel("Epoch-Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    apply_epoch_step_ticks(ax, x, labels)
    add_frame(ax)
    if zero_baseline:
        ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_quartet(x, labels, train_q, test_q, nj, title, outpath, zero_baseline=False):
    fig, ax = plt.subplots(figsize=(12, 5))
    ln_train_q, = ax.plot(x, train_q, label="Train Q-Dist", linewidth=1.8)
    ln_test_q,  = ax.plot(x, test_q,  label="Test Q-Dist",  linewidth=1.8)

    if nj.get("train_q") is not None:
        ax.axhline(nj["train_q"], linestyle="--", linewidth=1.2,
                   color=ln_train_q.get_color(), label="NJ Train Q-Dist")
    if nj.get("test_q") is not None:
        ax.axhline(nj["test_q"], linestyle="--", linewidth=1.2,
                   color=ln_test_q.get_color(), label="NJ Test Q-Dist")

    ax.set_title(f"{title} – Quartet Distance")
    ax.set_xlabel("Epoch-Step")
    ax.set_ylabel("Quartet Distance")
    ax.grid(True, alpha=0.3)
    apply_epoch_step_ticks(ax, x, labels)
    add_frame(ax)
    if zero_baseline:
        ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_rf(x, labels, train_rf, test_rf, nj, title, outpath, zero_baseline=False):
    fig, ax = plt.subplots(figsize=(12, 5))
    ln_train_rf, = ax.plot(x, train_rf, label="Train RF", linewidth=1.8)
    ln_test_rf,  = ax.plot(x, test_rf,  label="Test RF",  linewidth=1.8)

    if nj.get("train_rf") is not None:
        ax.axhline(nj["train_rf"], linestyle="--", linewidth=1.2,
                   color=ln_train_rf.get_color(), label="NJ Train RF")
    if nj.get("test_rf") is not None:
        ax.axhline(nj["test_rf"], linestyle="--", linewidth=1.2,
                   color=ln_test_rf.get_color(), label="NJ Test RF")

    ax.set_title(f"{title} – RF Distance")
    ax.set_xlabel("Epoch-Step")
    ax.set_ylabel("RF")   # removed "(lower is better)"
    ax.grid(True, alpha=0.3)
    apply_epoch_step_ticks(ax, x, labels)
    add_frame(ax)
    if zero_baseline:
        ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Plot split charts (Loss, Q-Dist, RF) from JSON or Pickle; supports old/new NJ schemas."
    )
    ap.add_argument("path", type=Path, help="Path to results.{json|pkl}")
    ap.add_argument("--out_prefix", type=Path, default=Path("metrics"),
                    help="Prefix for output images (e.g., 'metrics').")
    ap.add_argument("--title", type=str, default=None,
                    help="Optional chart title prefix. If omitted, uses dataset name from file.")
    ap.add_argument("--zero-baseline", action="store_true",
                    help="Force y-axis to start at 0 for all charts.")
    args = ap.parse_args()

    blob = load_any(args.path)
    evals = blob.get("all_evaluations", [])
    if not evals:
        raise SystemExit("No 'all_evaluations' found in file.")

    x, labels, train_q, test_q, train_rf, test_rf, loss = extract_series(evals)
    cfg = blob.get("config", {}) or {}
    dataset = cfg.get("dataset_name", "dataset")
    title = args.title or f"Metrics – {dataset}"
    nj = get_nj_baselines(blob)

    # Output paths
    loss_out = args.out_prefix.with_name(f"{args.out_prefix.name}_loss.png")
    quartet_out = args.out_prefix.with_name(f"{args.out_prefix.name}_quartet.png")
    rf_out = args.out_prefix.with_name(f"{args.out_prefix.name}_rf.png")

    # Plots
    plot_loss(x, labels, loss, title, loss_out, args.zero_baseline)
    plot_quartet(x, labels, train_q, test_q, nj, title, quartet_out, args.zero_baseline)
    plot_rf(x, labels, train_rf, test_rf, nj, title, rf_out, args.zero_baseline)

    print("Saved:")
    print(f"  {loss_out.resolve()}")
    print(f"  {quartet_out.resolve()}")
    print(f"  {rf_out.resolve()}")

if __name__ == "__main__":
    main()
