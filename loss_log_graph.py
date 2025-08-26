#!/usr/bin/env python3
import json
import pickle
import argparse
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

# ---------- Helpers ----------

def epoch_step_labels(evals):
    x = list(range(len(evals)))
    labels = [f"{e.get('epoch','?')}-{e.get('step','?')}" for e in evals]
    loss = [e["loss"] for e in evals]
    return x, labels, loss

def add_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

def apply_epoch_step_ticks(ax, x, labels, max_ticks=12):
    n = len(x)
    stride = 1 if n <= max_ticks else max(1, n // max_ticks)
    idx = list(range(0, n, stride))
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] for i in idx], rotation=45, ha="right")

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Plot Loss on a logarithmic scale from JSON or Pickle.")
    ap.add_argument("path", type=Path, help="Path to results.{json|pkl}")
    ap.add_argument("--out", type=Path, default=Path("loss_log.png"), help="Output image file.")
    ap.add_argument("--title", type=str, default=None, help="Optional title prefix.")
    ap.add_argument("--base", type=float, default=10, help="Log base (e.g., 10, 2, e). Default: 10")
    args = ap.parse_args()

    blob = load_any(args.path)
    evals = blob.get("all_evaluations", [])
    if not evals:
        raise SystemExit("No 'all_evaluations' found in file.")

    x, labels, loss = epoch_step_labels(evals)

    # Enforce no epsilon: require strictly positive losses for log scale
    if any((l is None) or (l <= 0) for l in loss):
        bad_idxs = [i for i, l in enumerate(loss) if (l is None) or (l <= 0)]
        raise SystemExit(
            f"Non-positive loss values at indices {bad_idxs}; cannot plot on log scale without modification."
        )

    cfg = blob.get("config", {}) or {}
    dataset = cfg.get("dataset_name", "dataset")
    title = args.title or f"Metrics – {dataset}"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, loss, linewidth=1.8, label="Loss (log scale)")
    ax.set_yscale("log", base=args.base)

    ax.set_title(f"{title} – Loss (Log Scale)")
    ax.set_xlabel("Epoch-Step")
    ax.set_ylabel("Loss")

    ax.grid(True, which="both", alpha=0.3)
    apply_epoch_step_ticks(ax, x, labels)
    add_frame(ax)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out.resolve()}")

if __name__ == "__main__":
    main()
