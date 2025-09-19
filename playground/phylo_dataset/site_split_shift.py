#!/usr/bin/env python3
"""Analyse how site-based train/test splits affect pairwise distances."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from celltreebench.datasets.phylo_dataset_creator import PhyloDatasetCreator

COLUMNS_PER_SITE = 22


@dataclass
class SplitConfig:
    """Configuration for a single train/test site split."""

    proportion: float
    train_sites: int
    test_sites: int
    total_selected: int

    @property
    def label(self) -> str:
        return f"{self.train_sites}/{self.test_sites} (total {self.total_selected})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="phylogenetic/200tips",
        help="Relative dataset path under data/ (default: %(default)s)",
    )
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parents[2] / "data"),
        help="Root directory containing phylogenetic datasets.",
    )
    parser.add_argument(
        "--tree-directory",
        default="trees",
        help="Subdirectory with Newick trees (default: %(default)s)",
    )
    parser.add_argument(
        "--msa-directory",
        default="msas",
        help="Subdirectory with MSA files (default: %(default)s)",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Train-site proportions to probe (each in (0,1); complementary sites go to test).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of independent resamplings per split (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().with_suffix("").parent / "output"),
        help="Directory to store experiment artefacts.",
    )
    parser.add_argument(
        "--total-sites",
        nargs="*",
        type=int,
        default=None,
        help="Total number of sites to subsample (train/test split is 50/50 of this set).",
    )
    return parser.parse_args()


def load_full_dataset(args: argparse.Namespace):
    creator = PhyloDatasetCreator(
        dataset=args.dataset,
        dataset_names=[''],
        tree_directory=args.tree_directory,
        msa_directory=args.msa_directory,
        data_dir=args.data_root,
        autosplit=None,
        seed=args.seed,
    )
    datasets = creator.get_dataset()
    if not datasets:
        raise RuntimeError('Failed to load datasets from path: ' + args.dataset)
    dataset = next(iter(datasets.values()))
    return dataset

def columns_from_site_indices(site_idx: np.ndarray) -> np.ndarray:
    """Expand site indices into column indices for one-hot encoded MSAs."""
    site_idx = np.asarray(site_idx, dtype=np.int64)
    offsets = np.arange(COLUMNS_PER_SITE, dtype=np.int64)
    return (site_idx[:, None] * COLUMNS_PER_SITE + offsets[None, :]).reshape(-1)


def euclidean_distances(data: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances for rows in ``data``."""
    data = data.astype(np.float64, copy=False)
    sq_norms = np.sum(data ** 2, axis=1, keepdims=True)
    sq_dists = sq_norms + sq_norms.T - 2.0 * data @ data.T
    np.maximum(sq_dists, 0.0, out=sq_dists)
    return np.sqrt(sq_dists, out=sq_dists)


def summarise_distances(train_dist: np.ndarray, test_dist: np.ndarray) -> dict:
    mask = np.triu(np.ones_like(train_dist, dtype=bool), k=1)
    train_vals = train_dist[mask]
    test_vals = test_dist[mask]
    abs_diff = np.abs(train_vals - test_vals)
    denom = np.maximum.reduce([
        np.abs(train_vals),
        np.abs(test_vals),
        np.full_like(train_vals, 1e-9),
    ])
    rel_diff = abs_diff / denom
    summary = {
        "mean_abs_diff": float(abs_diff.mean()),
        "median_abs_diff": float(np.median(abs_diff)),
        "max_abs_diff": float(abs_diff.max()),
        "mean_rel_diff": float(rel_diff.mean()),
        "median_rel_diff": float(np.median(rel_diff)),
        "max_rel_diff": float(rel_diff.max()),
        "pct_rel_diff_gt_0.1": float((rel_diff > 0.1).mean()),
        "pct_rel_diff_gt_0.2": float((rel_diff > 0.2).mean()),
        "mean_train_dist": float(train_vals.mean()),
        "mean_test_dist": float(test_vals.mean()),
    }
    return summary


def build_split_configs(total_sites_available: int, proportions: Iterable[float], total_sites_list: Iterable[int] | None) -> List[SplitConfig]:
    configs: List[SplitConfig] = []
    if total_sites_list:
        for subset in total_sites_list:
            if subset is None:
                continue
            subset = int(subset)
            if subset <= 1:
                continue
            subset = min(subset, total_sites_available)
            train_sites = subset // 2
            test_sites = subset - train_sites
            if train_sites == 0 or test_sites == 0:
                continue
            prop = train_sites / subset if subset else 0.5
            configs.append(SplitConfig(prop, train_sites, test_sites, subset))
    else:
        for prop in proportions:
            if not (0.0 < prop < 1.0):
                continue
            train_sites = int(round(total_sites_available * prop))
            train_sites = min(max(train_sites, 1), total_sites_available - 1)
            test_sites = total_sites_available - train_sites
            configs.append(SplitConfig(prop, train_sites, test_sites, total_sites_available))
    unique: List[SplitConfig] = []
    seen = set()
    for cfg in configs:
        key = (cfg.train_sites, cfg.test_sites, cfg.total_selected)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cfg)
    return unique


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_full_dataset(args)
    msas = dataset.data_normalized
    total_sites = msas[0].shape[1] // COLUMNS_PER_SITE

    configs = build_split_configs(total_sites, args.splits, args.total_sites)
    if not configs:
        raise ValueError("No valid split configurations derived from --splits/--total-sites")

    records = []
    for repeat in range(args.repeats):
        rng = np.random.default_rng(args.seed + repeat)
        for cfg in configs:
            for tree_idx, msa in enumerate(msas):
                permuted_sites = rng.permutation(total_sites)
                selected_sites = permuted_sites[: cfg.total_selected]
                train_sites_idx = selected_sites[: cfg.train_sites]
                test_sites_idx = selected_sites[cfg.train_sites : cfg.train_sites + cfg.test_sites]
                train_cols = columns_from_site_indices(np.sort(train_sites_idx))
                test_cols = columns_from_site_indices(np.sort(test_sites_idx))
                train_data = msa.iloc[:, train_cols].to_numpy(dtype=np.float64)
                test_data = msa.iloc[:, test_cols].to_numpy(dtype=np.float64)

                train_dist = euclidean_distances(train_data)
                test_dist = euclidean_distances(test_data)
                summary = summarise_distances(train_dist, test_dist)
                record = {
                    "repeat": repeat,
                    "tree_idx": tree_idx,
                    "split_prop": cfg.proportion,
                    "train_sites": cfg.train_sites,
                    "test_sites": cfg.test_sites,
                    "total_selected_sites": cfg.total_selected,
                    **summary,
                }
                records.append(record)

    raw_df = pd.DataFrame.from_records(records)
    group_cols = ["total_selected_sites", "train_sites", "test_sites", "split_prop"]
    summary_df = (
        raw_df.groupby(group_cols)
        .agg({
            "mean_abs_diff": ["mean", "std"],
            "median_abs_diff": ["mean", "std"],
            "mean_rel_diff": ["mean", "std"],
            "median_rel_diff": ["mean", "std"],
            "pct_rel_diff_gt_0.1": ["mean", "std"],
            "pct_rel_diff_gt_0.2": ["mean", "std"],
            "mean_train_dist": "mean",
            "mean_test_dist": "mean",
        })
        .reset_index()
    )
    summary_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary_df.columns
    ]
    return raw_df, summary_df


def plot_results(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    summary_df = summary_df.sort_values(["total_selected_sites", "train_sites"])
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

    if summary_df["total_selected_sites"].nunique() > 1:
        x = summary_df["total_selected_sites"]
        x_label = "Total selected sites (train/test split 50/50)"
    else:
        x = summary_df["train_sites"]
        x_label = "Train sites (test uses remaining sites)"

    axes[0].errorbar(
        x,
        summary_df["mean_abs_diff_mean"],
        yerr=summary_df["mean_abs_diff_std"],
        marker="o",
        capsize=4,
        label="Mean |Δ|",
    )
    axes[0].set_ylabel("Absolute diff")
    axes[0].legend()

    axes[1].errorbar(
        x,
        summary_df["mean_rel_diff_mean"],
        yerr=summary_df["mean_rel_diff_std"],
        marker="o",
        capsize=4,
        color="tab:orange",
        label="Mean relative diff",
    )
    axes[1].set_ylabel("Relative diff")
    axes[1].legend()

    axes[2].errorbar(
        x,
        summary_df["pct_rel_diff_gt_0.1_mean"],
        yerr=summary_df["pct_rel_diff_gt_0.1_std"],
        marker="o",
        capsize=4,
        color="tab:green",
        label="Rel diff > 0.1",
    )
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("Fraction")
    axes[2].legend()

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)

    title = "Pairwise distance shift vs. site counts"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "distance_shift_vs_sites.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def main() -> None:
    args = parse_args()
    raw_df, summary_df = run_experiment(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "site_split_shift_raw.csv"
    summary_path = output_dir / "site_split_shift_summary.csv"
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_path = plot_results(summary_df, output_dir)

    print("Saved raw metrics to", raw_path)
    print("Saved summary metrics to", summary_path)
    print("Saved plot to", plot_path)


if __name__ == "__main__":
    main()
