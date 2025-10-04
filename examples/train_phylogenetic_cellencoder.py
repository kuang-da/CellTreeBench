#!/usr/bin/env python3
"""
Minimal training example for CellTreeQMAttention on C. elegans dataset.
Added ability to run on phylogenetic datasets with option "-phylo [dataset_name]"

This script demonstrates how to train a phylogenetic distance learning model
using the CellTreeBench dataset with quartet-based losses.
"""

import os
import sys
import logging
import time
import torch
import torch.optim as optim
import numpy as np
import argparse
import yaml
from dotenv import load_dotenv
from math import comb

# ==== Environment / Paths =====================================================
load_dotenv()
celltreebench_path = os.getenv("CELLTREEBENCH_PATH", "/workspaces/CellTreeBench")
sys.path.append(celltreebench_path)

# ==== Imports from project ==============================================
from utils_minimal import (
    distance_error,
    distance_error_from_dm,    
    pairwise_distances,
    reconstruct_from_dm,
    _triu_vector,
)
from loss_minimal import (
    additivity_error_quartet_tensor,
    triplet_loss_quartet_tensor_vectorized,
    quadruplet_loss_quartet_tensor_vectorized,
    compute_pairwise_distance_sums
)
from quartet_utils_minimal import (
    generate_quartets_tensor,
    get_quartet_dist,
)
from celltreeqm_attention import CellTreeQMAttention

from celltreebench.datasets.phylo_dataset_creator import PhyloDatasetCreator
from celltreebench.losses.d_loss import pairwise_l2sq_regression
from celltreebench.safety.scale_sentry import ScaleSentry, ScaleSentryConfig
from celltreebench.init.mds import classical_mds


# ==== Argparse ======================================================
def parse_cli_args(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Train CellTreeQMAttention on Phylogenetic dataset"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write outputs")
    parser.add_argument("--data-dir", type=str, required=True, help="Root data directory")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda:0, cpu)")
    return parser.parse_args(cli_args)

# ==== Defaults  ===================================================
def default_config():
    return {
        "data": {
            "dataset_name": "phylogenetic/200tips",
            "dataset_names": {"train": 0.5, "test": 0.5},
            "tree_directory": "trees",
            "msa_directory": "msas",
            "autosplit": "sites",
            "lineage_name": None,
            "base_dir": celltreebench_path,
        },
        "training": {
            "seed": 42,
            "lr": 1e-4,
            "weight_decay": 5.0e-5,
            "batch_size": 4096,          # number of quartets per step
            "num_epochs": 20,
            "eval_interval": 300,
            "warmup_steps": 500,
            "p_warmup_steps": 2000,
            "grad_clip": 1.0,
            "multiview": False,
            "pair_sampling": {
                "mode": "bucket_equal",
                "bins": 10,
                "long_pair_boost": 1.5,
            },
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        },
        "loss": {
            "metric": "euclidean",
            "metric_loss": "additivity",   # additivity | triplet | quadruplet
            "weight_D": 1.0,
            "weight_P": 6.0,
            "weight_close": 1.0,
            "weight_push": 5.0,
            "push_margin": 0.10,
            "distance_error_alpha": 0.75,
            "weight_gate": -1.0,
            # D-loss (l2sq regression)
            "D_align": "affine",
            "d_loss": "l2sq_huber",
            "huber_delta": 0.05,
            "scale_anchor": 0.05,
            # P-adaptive
            "adaptive_P": True,
            "adaptive_P_ema": 0.95,
            "adaptive_P_target_ratio": 0.4,
            "adaptive_P_cap_max": 1.0,
            "cap_by_act": True,
            "cap_by_act_floor": 0.6,
            # Consistency
            "consistency_weight": 0.3,
            "consistency_weight_target": 1.2,
        },
        "model": {
            "proj_dim": 512,
            "output_dim": 256,
            "hidden_dim": 512,
            "num_heads": 2,
            "num_layers": 2,
            "dropout_data": 0.0,
            "dropout_metric": 0.05,
            "norm_method": "pre_ln",
            "gate_type": "none",
            # encoder
            "cell_encoder_type": "site_transformer",
            "site_alphabet_size": 22,
            "site_embed_dim": 32,
            "site_encoder_heads": 4,
            "site_encoder_layers": 1,
            "site_dropout": 0.1,
            "site_pma_seeds": 8,
            "site_chunk_size": 1,
            "init_mode": "random",
        },
        "schedule": {
            "phaseA_steps": 1500,
            "phaseB_ramp_steps": 2000,
        },
        "scale_safety": {
            "sentry_enabled": True,
            "threshold": 0.25,
            "window_steps": 50,
            "grad_clip_during_sentry": 0.5,
            "p_weight_drop_on_sentry": 0.2,
        },
        "evaluation": {
            "nj_from_embedding": True,
            "report_bins": True,
            "save_plots": True,
        },
    }


def deep_update(base: dict, new: dict) -> dict:
    """Recursively merge dict `new` into `base` (in-place), keeping unspecified defaults."""
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def load_and_resolve_config(args):
    cfg = default_config()

    raw_user_config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            raw_user_config = yaml.safe_load(fh) or {}
        deep_update(cfg, raw_user_config)

    # The only arg-level override we allow: device (runtime/hardware)
    if args.device:
        cfg["training"]["device"] = args.device

    # Inject runtime paths
    cfg["runtime"] = {
        "output_dir": args.output_dir,
        "data_dir": args.data_dir,
        "config_path": args.config,
        "raw_user_config": raw_user_config,
    }

    # Basic validation & normalization
    assert cfg["data"]["dataset_name"], "data.dataset_name is required"

    return cfg

# ==== Logging =================================================================
def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

# ==== Helpers =================================================================
def check_embedding(node_tensor, node_names, model, dist_metric, device):
    """
    Check the embedding produced by the model for a single tensor.

    Args:
        node_tensor (Tensor): Normalised features with shape (1, N, F) on the target device.
        node_names (Sequence[str]): Names corresponding to the tensor rows.
        model: Neural network model.
        dist_metric (str): Distance metric to use.
        device: Device to run computation on.

    Returns:
        tuple: (embeddings, distance_matrix, node_names)
    """
    model.eval()
    with torch.no_grad():
        pts_mtx = node_tensor
        embeddings = model(pts_mtx)  # Shape: (1, N, D)
        dm = pairwise_distances(embeddings, metric=dist_metric)
        dm = dm.squeeze(0).cpu().numpy()  # Shape: (N, N)
        return embeddings, dm, node_names


def create_model(input_dim: int, device: str, model_cfg: dict):
    """Create CellTreeQMAttention strictly from the `model` section (no internal defaults)."""
    model = CellTreeQMAttention(
        input_dim=input_dim,
        hidden_dim=model_cfg["hidden_dim"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        output_dim=model_cfg["output_dim"],
        dropout_data=model_cfg["dropout_data"],
        dropout_metric=model_cfg["dropout_metric"],
        norm_method=model_cfg["norm_method"],
        proj_dim=model_cfg["proj_dim"],
        gate_mode=model_cfg["gate_type"],
        device=device,
        cell_encoder_type=model_cfg["cell_encoder_type"],
        site_alphabet_size=model_cfg["site_alphabet_size"],
        site_embed_dim=model_cfg["site_embed_dim"],
        site_encoder_heads=model_cfg["site_encoder_heads"],
        site_encoder_layers=model_cfg["site_encoder_layers"],
        site_dropout=model_cfg["site_dropout"],
        site_pma_seeds=model_cfg["site_pma_seeds"],
        site_chunk_size=model_cfg.get("site_chunk_size"),
    )
    return model.to(device)

def avg_edge_split_proportion(edges):
    """Average proportion proxy using combinatorial pair counts across common/ref edge splits."""
    lens = []
    for e in edges:
        if min(len(e[0]), len(e[1])) > 1:
            lens.append(comb(len(e[0]), 2) * comb(len(e[1]), 2))
    return sum(lens) / len(lens) if len(lens) > 0 else 0

# ==== Evaluation ==============================================================
def evaluate_model(model, dataset, cfg, device="cpu", dataset_name="train", gen=None):
    """Evaluate on a dataset; returns averaged metrics dict."""

    model.eval()
    dist_metric = cfg["loss"]["metric"]

    # ---- Multi-View（可选）配置 ----
    eval_cfg = cfg.get("evaluation", {}) or {}
    mv_on   = bool(eval_cfg.get("multiview", False))
    mv_K    = int(eval_cfg.get("views", 4))
    mv_keep = float(eval_cfg.get("keep_ratio", 0.5))
    alphabet_size = int(cfg["model"].get("site_alphabet_size", 22))

    def _subsample_sites(x, keep_ratio):
        # x: [1, N, F]，F = S * alphabet_size
        B, N, F = x.shape
        if alphabet_size <= 0 or (F % alphabet_size != 0):
            return x  # 维度不匹配时回退
        S = F // alphabet_size
        K = max(1, int(round(S * keep_ratio)))
        perm_device = x.device
        if gen is not None:
            gen_device = getattr(gen, "device", None)
            if gen_device is not None:
                gen_device = torch.device(gen_device)
                if gen_device != perm_device:
                    idx_sites = torch.randperm(S, generator=gen, device=gen_device)[:K]
                    idx_sites = idx_sites.to(perm_device, non_blocking=True)
                else:
                    idx_sites = torch.randperm(S, generator=gen, device=perm_device)[:K]
            else:
                idx_sites = torch.randperm(S, generator=gen, device=perm_device)[:K]
        else:
            idx_sites = torch.randperm(S, device=perm_device)[:K]
        base = torch.arange(alphabet_size, device=x.device)
        feat_idx = torch.cat([s * alphabet_size + base for s in idx_sites], dim=0)
        return x.index_select(2, feat_idx)
    
    with torch.no_grad():
        eval_results = {
            f"rf_{dataset_name}": [],
            f"rf_max_{dataset_name}": [],
            f"quartet_dist_{dataset_name}": [],
            f"split_prop_common_{dataset_name}": [],
            f"split_prop_ref_{dataset_name}": [],
        }

        node_tensors = dataset.get_node_tensors(device=device, add_batch_dim=True)
        node_name_lists = dataset.get_node_names()
        ref_dm_batches = dataset.get_ref_distance_matrices(device=device, add_batch_dim=True)

        for i, node_tensor in enumerate(node_tensors):
            node_names = node_name_lists[i]  # 提前统一
            
            if mv_on:
                # ---- 多视角：平均距离矩阵 ----
                dm_list = []
                for _ in range(mv_K):
                    xk = _subsample_sites(node_tensor, mv_keep)
                    zk = model(xk)
                    dk = pairwise_distances(zk, metric=dist_metric).to(device)  # [1, N, N]
                    dm_list.append(dk)
                dm = torch.stack(dm_list, dim=0).mean(0)               # [1, N, N]
                emb_dm = dm.squeeze(0).cpu().numpy()
                # 为了日志，额外做一次“全列”前向取范数均值（不影响评估用 dm）
                emb_norm_mean = model(node_tensor).squeeze(0).norm(dim=-1).mean().item()  # 仅用于日志
            else:
                    # ---- 原路径：一次性全列 ----            
                embeddings, emb_dm, _ = check_embedding(
                    node_tensor, node_name_lists[i], model, dist_metric, device
                )
                dm = pairwise_distances(embeddings, metric=dist_metric).to(device)
                emb_flat = embeddings.squeeze(0).detach().cpu()
                emb_norm_mean = emb_flat.norm(dim=-1).mean().item()
            
            # 统计/日志
            dist_vals = emb_dm[np.triu_indices_from(emb_dm, k=1)]
            if dist_vals.size > 0:
                dist_mean = float(dist_vals.mean())
                dist_std = float(dist_vals.std())
                dist_min = float(dist_vals.min())
                dist_max = float(dist_vals.max())
                logging.info(
                    f"[Eval {dataset_name} #{i}] emb_norm_mean={emb_norm_mean:.4f}, "
                    f"dist_mean={dist_mean:.4f}, dist_std={dist_std:.4f}, "
                    f"dist_min={dist_min:.4f}, dist_max={dist_max:.4f}"
                )

            # NJ 与四点评估
            emb_tree = reconstruct_from_dm(emb_dm, node_names, method="nj")
            emb_topo_res = dataset.compare_trees(emb_tree, i, ref_tree="topology_tree")
            
            eval_results[f"split_prop_common_{dataset_name}"].append(
                avg_edge_split_proportion(emb_topo_res["common_edges"])
            )
            eval_results[f"split_prop_ref_{dataset_name}"].append(
                avg_edge_split_proportion(emb_topo_res["ref_edges"])
            )
            
            dm_ref = ref_dm_batches[i]
            dm_quartets, dm_ref_quartets = generate_quartets_tensor(
                batch_size=100000,
                dm=dm,
                dm_ref=dm_ref,
                device=device,
                seed=int(torch.randint(0, 100_000_000_000, (1,), generator=gen)),
            )
            quartet_dist = get_quartet_dist(dm_quartets, dm_ref_quartets)
            eval_results[f"rf_{dataset_name}"].append(emb_topo_res["rf"])
            eval_results[f"rf_max_{dataset_name}"].append(emb_topo_res["max_rf"])
            eval_results[f"quartet_dist_{dataset_name}"].append(quartet_dist.item())
        
        # Collect average metrics
        eval_results[f"split_prop_common_{dataset_name}"] = sum(
            eval_results[f"split_prop_common_{dataset_name}"]
        ) / len(eval_results[f"split_prop_common_{dataset_name}"])
        eval_results[f"split_prop_ref_{dataset_name}"] = sum(
            eval_results[f"split_prop_ref_{dataset_name}"]
        ) / len(eval_results[f"split_prop_ref_{dataset_name}"])

        eval_results[f"rf_{dataset_name}"] = sum(
            eval_results[f"rf_{dataset_name}"]
        ) / sum(eval_results[f"rf_max_{dataset_name}"])
        eval_results[f"quartet_dist_{dataset_name}"] = sum(
            eval_results[f"quartet_dist_{dataset_name}"]
        ) / len(eval_results[f"quartet_dist_{dataset_name}"])

        return eval_results

def evaluate_base(datasets, evals=1, dist_metric="euclidean", device="cpu", gen=None, eval_type="basic NJ"):
    """
    Evaluate the base model on a dataset.

    Args:
        datasets (dict): The datasets to evaluate on. ("name": PhyloDataset)
        evals (int): Number of evaluations to perform.
        dist_metric (str): Distance metric to use.
        device (str): Device to run evaluation on.
        gen (torch.Generator): Random number generator.
        eval_type (str): Randomization strategy ("shuffle", "embedding", or "basic NJ").

    Returns:
        tuple: (List of evaluations, Dict of average metrics across evaluations)
    """
    base_eval = []
    base_eval_avg = {
        f"quartet_dist_{name}": 0 for name in datasets.keys()
    }
    base_eval_avg.update({f"rf_{name}": 0 for name in datasets.keys()})

    for eval_num in range(evals):
        dataset_eval = {}

        for dataset_name, dataset in datasets.items():
            topo_eval = {"rf": [], "rf_max": [], "quartet_dist": []}

            node_tensors = dataset.get_node_tensors(device=device, add_batch_dim=True)
            node_name_lists = dataset.get_node_names()
            ref_dm_batches = dataset.get_ref_distance_matrices(device=device, add_batch_dim=True)

            for i, node_tensor in enumerate(node_tensors):
                mtx = node_tensor.clone()

                if eval_type == "shuffle":
                    rand_idx = torch.randperm(mtx.size(1), generator=gen)
                    mtx = mtx[:, rand_idx, :]
                elif eval_type == "embedding":
                    rand_emb = torch.randn(mtx.size(2), 16, generator=gen)
                    mtx = mtx @ rand_emb.to(device)
                elif eval_type != "basic NJ":
                    raise ValueError(f"Invalid randomization strategy: {eval_type}")

                dm = pairwise_distances(mtx, metric=dist_metric)
                dm_cpu = dm.squeeze(0).cpu()
                tree = reconstruct_from_dm(dm_cpu.numpy(), node_name_lists[i], method="nj")

                topo_res = dataset.compare_trees(tree, i, ref_tree="topology_tree")
                topo_eval["rf"].append(topo_res["rf"])
                topo_eval["rf_max"].append(topo_res["max_rf"])

                dm_ref = ref_dm_batches[i]
                dm_quartets, dm_ref_quartets = generate_quartets_tensor(
                    batch_size=100000,
                    dm=dm,
                    dm_ref=dm_ref,
                    device=device,
                    seed=int(torch.randint(0, 100_000_000_000, (1,), generator=gen)),
                )
                quartet_dist = get_quartet_dist(dm_quartets, dm_ref_quartets)
                topo_eval["quartet_dist"].append(quartet_dist.item())

            q_dist = sum(topo_eval["quartet_dist"]) / len(topo_eval["quartet_dist"])
            rf_dist = sum(topo_eval["rf"]) / sum(topo_eval["rf_max"])
            dataset_eval[f"quartet_dist_{dataset_name}"] = q_dist
            dataset_eval[f"rf_{dataset_name}"] = rf_dist

            base_eval_avg[f"quartet_dist_{dataset_name}"] += q_dist
            base_eval_avg[f"rf_{dataset_name}"] += rf_dist

        dataset_eval["eval_type"] = eval_type
        dataset_eval["eval_num"] = eval_num
        base_eval.append(dataset_eval)

    for dataset_name in datasets.keys():
        base_eval_avg[f"quartet_dist_{dataset_name}"] /= evals
        base_eval_avg[f"rf_{dataset_name}"] /= evals

    base_eval_avg["eval_type"] = eval_type
    base_eval_avg["evals"] = evals

    return base_eval, base_eval_avg

# ==== Training ================================================================
def train_one_epoch(
    model, train_dataset, test_dataset, optimizer, epoch, cfg, device="cpu", gen=None
):
    """
    - Full dataset forward each step (no sample batches)
    - batch_size controls quartet sampling per step
    - evaluate every eval_every_n_steps, and also ensure at least once per epoch
    """
    model.train()

    # Cache tensors once per epoch to avoid repeated host->device transfers
    node_batches = train_dataset.get_node_tensors(device=device, add_batch_dim=True)
    ref_dm_batches = train_dataset.get_ref_distance_matrices(device=device, add_batch_dim=True)

    # Unpack config
    tr = cfg["training"]
    ls = cfg["loss"]
    md = cfg["model"]

    batch_size = int(tr["batch_size"])                 # quartets per step
    eval_interval = int(tr["eval_interval"])
    base_lr        = float(tr["lr"])
    warmup_steps   = int(tr.get("warmup_steps", 0))
    max_grad_norm  = float(tr.get("grad_clip", 1.0))    

    # Two view related
    keep_ratio      = float(tr.get("site_consistency_keep_ratio", 0.5))  # 每个视角保留列比例
    lambda_cons     = float(ls.get("consistency_weight", 0.5))           # 一致性权重
    alphabet_size   = int(md.get("site_alphabet_size", 22))
    min_sites       = int(md.get("site_pma_seeds", 1))                   # 至少保留这么多 site，防止过少
    keep_ratio      = max(0.05, min(0.95, keep_ratio))                   # 合理剪裁    

    # Loss related
    weight_D = float(ls["weight_D"])
    weight_P_base = float(ls["weight_P"])                 # 基准的（可被自适应修正）
    weight_close = float(ls["weight_close"])
    weight_push = float(ls["weight_push"])
    push_margin = float(ls["push_margin"])
    dist_metric = ls["metric"]
    metric_loss_type = ls["metric_loss"]
    distance_alpha = float(ls.get("distance_error_alpha", 0.5))
    weight_gate = float(ls.get("weight_gate", 0.0))
    distance_align = ls.get("D_align", "scale")
    lambda_scale = float(ls.get("scale_anchor", 0.05))
    phaseA_disable_P = bool(ls.get("phaseA_disable_P", False))
    alpha_target = float(ls.get("alpha_target", 1.0))
    alpha_reg = float(ls.get("alpha_reg", 0.0))
    beta_reg = float(ls.get("beta_reg", 0.0))
    huber_delta_phaseA = float(ls.get("huber_delta", 0.05))
    huber_delta_phaseB = float(ls.get("phaseB_huber_delta", huber_delta_phaseA))
    enforce_scale_clamp = bool(ls.get("enforce_scale_clamp", False))
    phaseB_r2_threshold = float(ls.get("r2_phaseB_threshold", 0.0))

    # 自适应配重相关（把 EMA 状态挂在 model 上，跨 epoch 复用）
    adaptive_P        = bool(ls.get("adaptive_P", True))
    adaptive_P_ema    = float(ls.get("adaptive_P_ema", 0.9))
    target_ratio      = float(ls.get("adaptive_P_target_ratio", 1.0))
    if adaptive_P and not hasattr(model, "_adaptive_p_state"):
        model._adaptive_p_state = {
            "ema_D": torch.tensor(1.0, device=device),
            "ema_P": torch.tensor(1.0, device=device),
            "momentum": adaptive_P_ema,
            "wP_base": weight_P_base,
            "target_ratio": target_ratio,
        }
    elif adaptive_P:
        # 若用户在 config 中改了参数，更新一下
        model._adaptive_p_state["momentum"]    = adaptive_P_ema
        model._adaptive_p_state["wP_base"]     = weight_P_base
        model._adaptive_p_state["target_ratio"] = target_ratio
    
    # -------- Helper：按“site”为单位子采样列 ----------
    def _subsample_sites(batch_x, keep_ratio_local: float):
        """
        batch_x: [B=1, N_tips, F]，其中 F=sites * alphabet_size
        返回：x_sub (同设备)、若无法整除则回退为原输入
        """
        B, N, F = batch_x.shape
        if alphabet_size <= 0 or (F % alphabet_size != 0):
            # 不能整除，放弃子采样（保证稳健）
            return batch_x
        S = F // alphabet_size
        K = max(min_sites, int(round(S * keep_ratio_local)))
        K = min(S, max(1, K))

        # 采样 site 索引
        idx_sites = torch.randperm(S, generator=gen)[:K].to(batch_x.device)
        # 将每个 site 展开为 alphabet_size 个连续特征列
        base = torch.arange(alphabet_size, device=batch_x.device)
        feat_idx = torch.cat([s * alphabet_size + base for s in idx_sites], dim=0)  # [K*alphabet]
        # 在特征维度上选取
        return batch_x.index_select(dim=2, index=feat_idx)

    # -------- Helper：其他小工具 ----------
    def _pairwise_dm(emb):
        return pairwise_distances(emb, metric=dist_metric).to(device)

    def _zscore(M):
        mu = M.mean()
        sigma = M.std().clamp_min(1e-6)
        return (M - mu) / sigma
        
    sch = cfg.get("schedule", {}) or {}
    phaseA_steps = int(sch.get("phaseA_steps", 0))
    phaseB_ramp = int(sch.get("phaseB_ramp_steps", 0))

    if not hasattr(model, "_phaseB_enabled"):
        model._phaseB_enabled = not phaseA_disable_P

    # Calculate max_step like in research codebase
    max_step = len(train_dataset) // batch_size if batch_size > 0 else 1

    # Initialize metrics for this epoch
    epoch_metrics = {
        "losses": [],
        "loss_D": [],
        "loss_P": [],
        "loss_P_close": [],
        "loss_P_push": [],
        "gate_loss": [],
        "evaluations": [],  # Store evaluation results
        "diag_mismatch_rate": [],
        "diag_active_rate": [],
        "diag_gap_mean": [],
    }
    running_loss = 0.0

    # Multiple training steps per epoch (the unconventional part!)
    for step_count in range(max_step):
        global_step = epoch * max_step + step_count + 1
        in_phaseA = global_step <= phaseA_steps

        # ---- Warmup LR（线性）----
        if warmup_steps > 0:
            warmup_factor = min(1.0, global_step / warmup_steps)
            new_lr = base_lr * warmup_factor
            for g in optimizer.param_groups:
                g["lr"] = new_lr        
        
        optimizer.zero_grad()
        # === Multi-view (optional) ===
        mv_on = bool(tr.get("multiview", False))
        x_full = node_batches[0]  # single-tree batch
        z_full = model(x_full)    # [1, N, D]
        z_centered_full = z_full - z_full.mean(dim=1, keepdim=True)
        z_centered_sq = z_centered_full.squeeze(0)
        dm_full = pairwise_distances(z_full, metric=dist_metric).to(device)

        scale_clamp_value = None
        if enforce_scale_clamp:
            v_est_full = _triu_vector(dm_full)
            v_ref_full = _triu_vector(ref_dm_batches[0])
            scale_num = (v_est_full * v_ref_full).sum()
            scale_den = (v_est_full * v_est_full).sum().clamp_min(1e-8)
            scale_factor = (scale_num / scale_den).clamp_min(1e-8).detach()
            scale_factor = torch.clamp(scale_factor, min=1e-4, max=1e4)
            scale_clamp_value = float(scale_factor.item())
            z_full = z_full * scale_factor
            z_centered_full = z_full - z_full.mean(dim=1, keepdim=True)
            z_centered_sq = z_centered_full.squeeze(0)
            dm_full = pairwise_distances(z_full, metric=dist_metric).to(device)

        if mv_on:
            x_a = _subsample_sites(x_full, keep_ratio)
            x_b = _subsample_sites(x_full, keep_ratio)
            z_a = model(x_a)
            z_b = model(x_b)
            dm_a = _pairwise_dm(z_a)
            dm_b = _pairwise_dm(z_b)
            if enforce_scale_clamp:
                z_a = z_a * scale_factor
                z_b = z_b * scale_factor
                dm_a = pairwise_distances(z_a, metric=dist_metric).to(device)
                dm_b = pairwise_distances(z_b, metric=dist_metric).to(device)
            dm_est = (dm_full + dm_a + dm_b) / 3.0
        else:
            dm_est = dm_full

        # 1) D-loss (l2sq regression to tree distance)
        ps_cfg = tr.get("pair_sampling", {}) or {}
        huber_delta_use = huber_delta_phaseA if in_phaseA else huber_delta_phaseB
        dres = pairwise_l2sq_regression(
            E=z_centered_full,
            d_true=ref_dm_batches[0],
            align=distance_align if distance_align in ("affine", "scale", "none") else "affine",
            huber_delta=huber_delta_use,
            bins=int(ps_cfg.get("bins", 10)),
            sampling_mode=("bucket_equal" if (ps_cfg.get("mode", "bucket_equal") == "bucket_equal") else "none"),
            long_pair_boost=float(ps_cfg.get("long_pair_boost", 1.5)),
            center=True,
            ensure_positive_alpha=True,
        )
        batch_D = dres["loss"]
        alpha_tensor = dres.get("alpha_raw")
        beta_tensor = dres.get("beta_raw")
        if alpha_reg > 0.0 and alpha_tensor is not None:
            batch_D = batch_D + alpha_reg * (alpha_tensor - alpha_target).pow(2).mean()
        if beta_reg > 0.0 and beta_tensor is not None:
            batch_D = batch_D + beta_reg * (beta_tensor).pow(2).mean()
        alpha_logged = dres["alpha"].mean()
        beta_logged = dres["beta"].mean()
        alpha_D = float(alpha_logged.item())
        beta_D = float(beta_logged.item())
        r2_l2sq = float(dres["r2"].item())
        # 2) P-loss：在 dm_est 上采样四元组并计算 additivity / triplet / quadruplet        
        dm_quartets, dm_ref_quartets = generate_quartets_tensor(
            batch_size=batch_size,  # Number of quartets to sample
            dm=dm_est,
            dm_ref=ref_dm_batches[0],
            device=device,
            seed=int(torch.randint(0, 100_000_000_000, (1,), generator=gen)),
        )

        batch_P      = torch.tensor(0.0, device=device)
        batch_P_close = torch.tensor(0.0, device=device)
        batch_P_push  = torch.tensor(0.0, device=device)

        if metric_loss_type == "additivity":
            warmup_P = int(tr.get("p_warmup_steps", 500))
            push_margin_eff = push_margin * (min(1.0, global_step / float(warmup_P)) if warmup_P > 0 else 1.0)            
            batch_P, batch_P_close, batch_P_push, _ = additivity_error_quartet_tensor(
                dm_quartets=dm_quartets,
                dm_ref_quartets=dm_ref_quartets,
                weight_close=weight_close,
                weight_push=weight_push,
                push_margin=push_margin_eff,
                matching_mode = "all",
                # matching_mode = "all" if global_step < max(1000, 0.5*max_step) else "mismatched",
                device=device,
            )
            # compute quartet active ratio for adaptive cap logic
            with torch.no_grad():
                from loss_minimal import compute_pairwise_distance_sums
                ds_ref = compute_pairwise_distance_sums(dm_ref_quartets)
                ds_est = compute_pairwise_distance_sums(dm_quartets)
                _, top2_ref = torch.topk(ds_ref, 2, dim=1)
                top2_vals = ds_est.gather(1, top2_ref)
                S1, S2 = top2_vals[:, 0], top2_vals[:, 1]
                lowest_idx = 3 - top2_ref.sum(dim=1)
                S3 = ds_est[torch.arange(ds_est.size(0), device=ds_est.device), lowest_idx]
                den = (S1 + S2).clamp_min(1e-8)
                gap_ratio = (S1 + S2 - 2.0 * S3) / den
                active_rate_cur = (push_margin_eff - gap_ratio > 0).float().mean().item()
        
        
        elif metric_loss_type == "triplet":
            batch_P = triplet_loss_quartet_tensor_vectorized(
                dm_quartets=dm_quartets,
                dm_ref_quartets=dm_ref_quartets,
                margin=push_margin,
                device=device,
            )
            batch_P_close = torch.tensor(0.0, device=device)
            batch_P_push = torch.tensor(0.0, device=device)
            active_rate_cur = float("nan")

        elif metric_loss_type == "quadruplet":
            batch_P, _, _, _, _ = quadruplet_loss_quartet_tensor_vectorized(
                dm_quartets=dm_quartets,
                dm_ref_quartets=dm_ref_quartets,
                alpha=0.5,
                beta=0.5,
                device=device,
            )
            batch_P_close = torch.tensor(0.0, device=device)
            batch_P_push = torch.tensor(0.0, device=device)
            active_rate_cur = float("nan")
        else:
            raise ValueError(f"Invalid metric loss: {metric_loss_type}")
        
        # 3) Two-View 一致性损失：距离矩阵在换列子集时应稳定
        if lambda_cons > 0.0 and mv_on:
            # L_cons = torch.nn.functional.mse_loss(_zscore(dm_a), _zscore(dm_b))
            va = _triu_vector(dm_a)  # (N*(N-1)/2,)
            vb = _triu_vector(dm_b)
            va = (va - va.mean()) / va.std().clamp_min(1e-8)
            vb = (vb - vb.mean()) / vb.std().clamp_min(1e-8)
            L_cons = torch.mean((va - vb) ** 2)            
        else:
            L_cons = torch.tensor(0.0, device=device)

        # 4. Feature Gate Loss (if applicable)
        gate_loss = torch.tensor(0.0, device=device)
        if hasattr(model, "feature_gate") and model.feature_gate is not None:
            gate_loss = model.feature_gate.compute_penalty(
                penalty_type="sparsity",
                lambda_penalty=weight_gate,
            )

        # ---- 自适应配重 + Phase A/B 调度 ----
        # baseline consistency weight ramp
        cons_w0 = float(ls.get("consistency_weight", 0.0))
        cons_wT = float(ls.get("consistency_weight_target", cons_w0))
        if not in_phaseA and phaseB_ramp > 0:
            ramp = min(1.0, (global_step - phaseA_steps) / float(max(1, phaseB_ramp)))
            lambda_cons = cons_w0 + (cons_wT - cons_w0) * ramp
        else:
            lambda_cons = cons_w0 if in_phaseA else cons_wT

        # Adaptive P logic
        if adaptive_P:
            st = model._adaptive_p_state
            with torch.no_grad():
                st["ema_D"] = st["momentum"] * st["ema_D"] + (1.0 - st["momentum"]) * (batch_D.detach().abs() + 1e-8)
                st["ema_P"] = st["momentum"] * st["ema_P"] + (1.0 - st["momentum"]) * (batch_P.detach().abs() + 1e-8)
                raw_scale = st["target_ratio"] * (st["ema_D"] / (st["ema_P"] + 1e-8))
                cap_max = float(ls.get("adaptive_P_cap_max", 1.0))
                cap_by_act = bool(ls.get("cap_by_act", False))
                cap_floor = float(ls.get("cap_by_act_floor", 0.6))
                cap_dyn = cap_max
                if cap_by_act and isinstance(active_rate_cur, float) and not (active_rate_cur != active_rate_cur):
                    if active_rate_cur < 0.3:
                        cap_dyn = cap_max
                    elif active_rate_cur < 0.6:
                        cap_dyn = max(cap_floor, 0.8 * cap_max)
                    else:
                        cap_dyn = max(cap_floor, 0.7 * cap_max)
                weight_P_eff = st["wP_base"] * raw_scale.clamp(0.0, cap_dyn)
        else:
            weight_P_eff = torch.as_tensor(weight_P_base, device=device)

        phaseB_enabled = getattr(model, "_phaseB_enabled", not phaseA_disable_P)
        if phaseA_disable_P and (not phaseB_enabled) and (global_step > phaseA_steps):
            if phaseB_r2_threshold <= 0.0 or r2_l2sq >= phaseB_r2_threshold:
                phaseB_enabled = True
                model._phaseB_enabled = True
        elif not phaseA_disable_P and (not phaseB_enabled) and (global_step > phaseA_steps):
            phaseB_enabled = True
            model._phaseB_enabled = True

        if not phaseB_enabled:
            weight_P_eff = torch.zeros_like(weight_P_eff)
        else:
            if phaseB_ramp > 0:
                ramp = min(1.0, max(0.0, (global_step - phaseA_steps) / float(max(1, phaseB_ramp))))
                weight_P_eff = weight_P_eff * ramp

        # MDS anchor (Phase-aware pull towards classical MDS solution)
        mds_target = getattr(model, "_mds_target", None)
        mds_anchor_weight = float(getattr(model, "_mds_anchor_weight", 0.0))
        mds_anchor_decay = int(getattr(model, "_mds_anchor_decay", phaseB_ramp))
        mds_anchor_mode = getattr(model, "_mds_anchor_mode", "embedding")
        mds_anchor_weight_eff = 0.0
        L_mds = torch.tensor(0.0, device=device)
        if mds_target is not None and mds_anchor_weight > 0.0:
            if in_phaseA:
                anchor_scale = 1.0
            elif mds_anchor_decay and mds_anchor_decay > 0:
                progress = min(1.0, (global_step - phaseA_steps) / float(mds_anchor_decay))
                anchor_scale = max(0.0, 1.0 - progress)
            else:
                anchor_scale = 0.0
            if anchor_scale > 0.0:
                if mds_anchor_mode == "pairwise":
                    target_dm = getattr(model, "_mds_target_dm", None)
                    if target_dm is not None:
                        target_dm = target_dm.to(z_full.device, dtype=z_full.dtype)
                        dm_centered_sq = pairwise_distances(z_centered_full, metric=dist_metric).squeeze(0).pow(2)
                        L_mds = torch.mean((dm_centered_sq - target_dm) ** 2)
                        mds_anchor_weight_eff = mds_anchor_weight * anchor_scale
                else:
                    anchor_tgt = mds_target.to(z_full.device, dtype=z_full.dtype)
                    if anchor_tgt.shape[0] == z_centered_sq.shape[0]:
                        L_mds = torch.mean((z_centered_sq - anchor_tgt) ** 2)
                        mds_anchor_weight_eff = mds_anchor_weight * anchor_scale
                    else:
                        logging.warning(
                            f"MDS anchor size mismatch (target={anchor_tgt.shape[0]}, embedding={z_centered_sq.shape[0]})."
                        )

        # NEW: Scale anchor（不要放在 no_grad 里）
        if lambda_scale > 0.0:
            # v_est = _triu_vector(dm_est)               # 上三角展开（与 diag 中一致）
            # v_ref = _triu_vector(ref_dm_batches[0])    # 参考距离（不参与梯度）
            # s_num = (v_est * v_ref).sum()
            # s_den = (v_est * v_est).sum().clamp_min(1e-8)
            # scale_s = s_num / s_den                    # 与 D_align='scale' 同源的最优缩放
            # scale_penalty = (scale_s - 1.0).pow(2)     # 约束尺度靠近 1
            q_est = _triu_vector(dm_est.pow(2))     # q_est = d_est^2
            d_ref = _triu_vector(ref_dm_batches[0]) # 仍然是 d_T
            t_num = (q_est * d_ref).sum()
            t_den = (q_est * q_est).sum().clamp_min(1e-8)
            t = t_num / t_den                        # 在 q_est→d_T 的最小二乘斜率
            scale_penalty = (t - 1.0).pow(2)            
        else:
            scale_penalty = torch.tensor(0.0, device=device)

        # === 统一计算“距离域”的 scale_ref（s），用于 sentry 与诊断 ===
        with torch.no_grad():
            v_est_cur = _triu_vector(dm_est)
            v_ref_cur = _triu_vector(ref_dm_batches[0])
            s_num_cur = (v_est_cur * v_ref_cur).sum()
            s_den_cur = (v_est_cur * v_est_cur).sum().clamp_min(1e-8)
            scale_ref_s = (s_num_cur / s_den_cur).item()  # float
            
        # Scale Sentry adjustments
        ss_cfg = cfg.get("scale_safety", {}) or {}
        if bool(ss_cfg.get("sentry_enabled", True)):
            if not hasattr(model, "_scale_sentry"):
                model._scale_sentry = ScaleSentry(
                    ScaleSentryConfig(
                        threshold=float(ss_cfg.get("threshold", 0.25)),
                        window_steps=int(ss_cfg.get("window_steps", 50)),
                        grad_clip_during_sentry=float(ss_cfg.get("grad_clip_during_sentry", 0.5)),
                        p_weight_drop_on_sentry=float(ss_cfg.get("p_weight_drop_on_sentry", 0.2)),
                    )
                )
                model._sentry_hits = 0
                
            # 使用统一计算的 s（距离域 scale_ref）
            adj = model._scale_sentry.update(global_step, alpha=alpha_D, scale_ref=scale_ref_s)
            if adj.get("active", 0.0) > 0:
                model._sentry_hits = getattr(model, "_sentry_hits", 0) + 1
                max_grad_norm = float(adj.get("grad_clip", max_grad_norm))
                weight_P_eff = weight_P_eff * float(adj.get("p_weight_scale", 1.0))
                if adj.get("just_triggered", 0.0) > 0:   # 只在进入时打一次
                    logging.warning(
                        f"[sentry] step={global_step} | alpha={alpha_D:.4f} | "
                        f"scale_ref_s={scale_ref_s:.4f} | dev_a={adj.get('dev_alpha'):0.3f} "
                        f"| dev_s={adj.get('dev_scale'):0.3f} | grad_clip->{max_grad_norm:.3f} "
                        f"| wP_eff x {float(adj.get('p_weight_scale', 1.0)):.3f}"
                    )           
            # scale_val = float(scale_s) if 'scale_s' in locals() else None
            # adj = model._scale_sentry.update(global_step, alpha=alpha_D, scale_ref=scale_val)
            # if adj.get("active", 0.0) > 0:
            #     model._sentry_hits = getattr(model, "_sentry_hits", 0) + 1
            #     max_grad_norm = float(adj.get("grad_clip", max_grad_norm))
            #     weight_P_eff = weight_P_eff * float(adj.get("p_weight_scale", 1.0))

        # Total loss（加入尺度锚）
        total_loss = (
            weight_D * batch_D
            + weight_P_eff * batch_P
            + lambda_cons * L_cons
            + gate_loss
            + lambda_scale * scale_penalty            # NEW
            + mds_anchor_weight_eff * L_mds
        )

        # Total loss
        # total_loss = weight_D * batch_D + weight_P_eff * batch_P + lambda_cons * L_cons + gate_loss
        running_loss += total_loss.item()

        # Backward pass
        total_loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        # ---- 诊断打印（每 100 step 一次，轻量）----
        if (step_count % 100) == 0:
            with torch.no_grad():
                # 基于“当步采样到的四点”做极轻量的统计
                Bq = dm_quartets.size(0)
                if Bq > 0:
                    ds_ref = compute_pairwise_distance_sums(dm_ref_quartets)  # (Bq, 3)
                    ds_est = compute_pairwise_distance_sums(dm_quartets)      # (Bq, 3)

                    # 参考/估计各自 top-2 索引
                    _, top2_ref = torch.topk(ds_ref, 2, dim=1)
                    _, top2_est = torch.topk(ds_est, 2, dim=1)

                    # 1) 四点不一致率
                    mismatched = (top2_ref != top2_est).any(dim=1)           # (Bq,)
                    mismatch_rate = mismatched.float().mean().item()

                    # 2) hinge 激活率（基于参考的 top-2 计算 gap）
                    top2_vals = ds_est.gather(1, top2_ref)                   # (Bq, 2)
                    S1, S2 = top2_vals[:, 0], top2_vals[:, 1]
                    lowest_idx = 3 - top2_ref.sum(dim=1)
                    S3 = ds_est[torch.arange(Bq, device=ds_est.device), lowest_idx]
                    den = (S1 + S2).clamp_min(1e-8)
                    gap_ratio = (S1 + S2 - 2.0 * S3) / den                   # ∈ (-∞, 1]
                    active_rate = (push_margin_eff - gap_ratio > 0).float().mean().item()
                    gap_mean = gap_ratio.mean().item()
                else:
                    mismatch_rate = float("nan"); active_rate = float("nan"); gap_mean = float("nan")
            
                v_est = _triu_vector(dm_est)
                v_ref = _triu_vector(ref_dm_batches[0])
                s_num = (v_est * v_ref).sum()
                s_den = (v_est * v_est).sum().clamp_min(1e-8)
                scale_s = (s_num / s_den).item()

                # 诊断：平方域斜率 t（若可用）与 1/alpha 的对比
                if 't' in locals() and isinstance(t, torch.Tensor):
                    t_val = float(t.detach())
                else:
                    t_val = float('nan')
                t_from_alpha = (1.0 / max(alpha_D, 1e-8))  # 当 beta≈0 时，理论上 t ≈ 1/alpha            
                
                lr_now = optimizer.param_groups[0]["lr"]
                cap_max_log = float(ls.get("adaptive_P_cap_max", 1.0))
                # logging.info(
                #     f"[diag] gstep={global_step} | lr={lr_now:.2e} | "
                #     f"wP_eff={float(weight_P_eff):.3f} (base={weight_P_base:.3f}, cap_max={cap_max_log:.2f}) | "
                #     f"D={batch_D.item():.3f} | P={batch_P.item():.3f} | cons={L_cons.item():.3f} | "
                #     f"P_close={batch_P_close.item():.3f} | P_push={batch_P_push.item():.3f} | "
                #     f"mm={mismatch_rate:.3f} | act={active_rate:.3f} | gap={gap_mean:.4f} | "
                #     f"margin={push_margin_eff:.3f} | scale_ref={scale_s:.4f} | "
                #     f"alpha={alpha_D:.4f} | beta={beta_D:.4f} | R2_l2sq={r2_l2sq:.4f} | "
                #     f"mds_w={mds_anchor_weight_eff:.3f} | L_mds={L_mds.item():.4f} | "
                #     f"sanch={float(scale_penalty):.4e}"
                # )
                logging.info(
                    f"[diag] gstep={global_step} | lr={lr_now:.2e} | "
                    f"wP_eff={float(weight_P_eff):.3f} (base={weight_P_base:.3f}, cap_max={cap_max_log:.2f}) | "
                    f"D={batch_D.item():.3f} | P={batch_P.item():.3f} | cons={L_cons.item():.3f} | "
                    f"P_close={batch_P_close.item():.3f} | P_push={batch_P_push.item():.3f} | "
                    f"mm={mismatch_rate:.3f} | act={active_rate:.3f} | gap={gap_mean:.4f} | "
                    f"margin={push_margin_eff:.3f} | "
                    f"scale_ref_s={scale_s:.4f} | t_sq={t_val:.4f} | 1/alpha={t_from_alpha:.4f} | "
                    f"alpha={alpha_D:.4f} | beta={beta_D:.4f} | R2_l2sq={r2_l2sq:.4f} | "
                    f"mds_w={mds_anchor_weight_eff:.3f} | L_mds={L_mds.item():.4f} | "
                    f"sanch={float(scale_penalty):.4e}"
                )                
                    
        # Store step metrics
        epoch_metrics["losses"].append(total_loss.item())
        epoch_metrics["loss_D"].append(batch_D.item())
        epoch_metrics["loss_P"].append(batch_P.item())
        epoch_metrics["loss_P_close"].append(batch_P_close.item())
        epoch_metrics["loss_P_push"].append(batch_P_push.item())
        epoch_metrics["gate_loss"].append(gate_loss.item())
        epoch_metrics["diag_mismatch_rate"].append(mismatch_rate)
        epoch_metrics["diag_active_rate"].append(active_rate)
        epoch_metrics["diag_gap_mean"].append(gap_mean)
        # Periodic Evaluation (every eval_interval STEPS, not epochs!)
        if step_count % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                logging.info(
                    f"[Epoch {epoch+1}, Step {step_count}/{max_step}] Evaluating..."
                )

                train_metrics = evaluate_model(
                    model, train_dataset, cfg, device, "train", gen=gen
                )
                test_metrics = evaluate_model(
                    model, test_dataset, cfg, device, "test", gen=gen
                )

                eval_result = {
                    "epoch": epoch + 1,
                    "step": step_count,
                    "train_rf": train_metrics["rf_train"],
                    "test_rf": test_metrics["rf_test"],
                    "train_quartet_dist": train_metrics["quartet_dist_train"],
                    "test_quartet_dist": test_metrics["quartet_dist_test"],
                    "loss": total_loss.item(),
                    "loss_D": batch_D.item(),
                    "loss_P": batch_P.item(),
                }
                epoch_metrics["evaluations"].append(eval_result)

                logging.info(
                    f"[Epoch {epoch+1}, Step {step_count}/{max_step}] "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Train RF: {train_metrics['rf_train']:.4f} | "
                    f"Test RF: {test_metrics['rf_test']:.4f} | "
                    f"Train Q-Dist: {train_metrics['quartet_dist_train']:.4f} | "
                    f"Test Q-Dist: {test_metrics['quartet_dist_test']:.4f}"
                )

                logging.info(
                    f"  Loss breakdown: D={batch_D.item():.4f}, "
                    f"P={batch_P.item():.4f}, "
                    f"P_close={batch_P_close.item():.4f}, "
                    f"P_push={batch_P_push.item():.4f}, "
                    f"Gate={gate_loss.item():.4f}"
                )

                logging.info(
                    f"  Edge split stats: Common={train_metrics['split_prop_common_train']:.4f}, "
                    f"Ref={train_metrics['split_prop_ref_train']:.4f}"
                )

            model.train()

    avg_epoch_loss = running_loss / max_step if max_step > 0 else running_loss

    return {
        "avg_loss": avg_epoch_loss,
        "max_step": max_step,
        "step_metrics": epoch_metrics,
    }



def main(cli_args=None):
    """Main training function."""
    setup_logging()
    logging.info("Starting CellTreeQMAttention training on Phylogenetic dataset")

    args = parse_cli_args(cli_args)
    cfg = load_and_resolve_config(args)
    
    out_dir = cfg["runtime"]["output_dir"]
    data_dir = cfg["runtime"]["data_dir"]
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Data directory: {data_dir}")
    logging.info(f"Output directory: {out_dir}")
    
    # Seed / device
    device = torch.device(cfg["training"]["device"])
    seed = int(cfg["training"]["seed"])
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    logging.info(f"Using device: {device} | seed={seed}")
    
    logging.info(f"Config: {cfg}")
    # Load datasets
    ds_name = cfg["data"]["dataset_name"]
    logging.info(f"Loading dataset: {ds_name}")
    datasets = PhyloDatasetCreator(
        ds_name,
        dataset_names=cfg["data"]["dataset_names"],
        data_dir=data_dir,
        tree_directory=cfg["data"]["tree_directory"],
        msa_directory=cfg["data"]["msa_directory"],
        autosplit=cfg["data"]["autosplit"],
        seed=torch.randint(0, 100_000_000_000, (1,), generator=gen).item(),
    )
    datasets_dict = datasets.get_dataset(datasets=["train", "test"])
    train_dataset = datasets_dict["train"]
    test_dataset = datasets_dict["test"]
    logging.info(f"Train shape: {train_dataset.data_normalized[0].shape}")
    logging.info(f"Test shape: {test_dataset.data_normalized[0].shape}")    
    
    # Quick data summary
    results = {
        "epoch_avg_loss": [],
        "all_evaluations": [],
        "base_eval_avg": {},
        "config": {"resolved": cfg},
    }
    results["data"] = {
        name: [
            {
                "leaves": node_mtx["node_mtx"].shape[0],
                "amino_acids": node_mtx["node_mtx"].shape[1] / 22,
            }
            for node_mtx in dataset.get_node_mtx()
        ]
        for name, dataset in datasets_dict.items()
    }

    # Baselines: NJ / shuffled / random embedding
    _, avg_NJ_evals = evaluate_base(
        datasets_dict, evals=1, dist_metric=cfg["loss"]["metric"], device=device, gen=gen, eval_type="basic NJ"
    )
    logging.info(
        f"Base evaluations (NJ): "
        f"Train RF: {avg_NJ_evals['rf_train']:.4f} | "
        f"Test RF: {avg_NJ_evals['rf_test']:.4f} | "
        f"Train Q-Dist: {avg_NJ_evals['quartet_dist_train']:.4f} | "
        f"Test Q-Dist: {avg_NJ_evals['quartet_dist_test']:.4f}"
    )
    results["base_eval_avg"]["NJ"] = avg_NJ_evals

    # Evaluate the datasets on random shuffle and embedding
    _, avg_random_shuffled_evals = evaluate_base(datasets_dict, evals = 3, dist_metric=cfg["loss"]["metric"], device=device, gen=gen, eval_type="shuffle")
    _, avg_random_embedding_evals = evaluate_base(datasets_dict, evals = 3, dist_metric=cfg["loss"]["metric"], device=device, gen=gen, eval_type="embedding")

    # Log averages of random shuffle and random embedding evaluations
    logging.info(
                f"Random embedding evaluations: "
                f"Train RF: {avg_random_embedding_evals['rf_train']:.4f} | "
                f"Test RF: {avg_random_embedding_evals['rf_test']:.4f} | "
                f"Train Q-Dist: {avg_random_embedding_evals['quartet_dist_train']:.4f} | "
                f"Test Q-Dist: {avg_random_embedding_evals['quartet_dist_test']:.4f}"
            )
    logging.info(
                f"Random shuffle evaluations: "
                f"Train RF: {avg_random_shuffled_evals['rf_train']:.4f} | "
                f"Test RF: {avg_random_shuffled_evals['rf_test']:.4f} | "
                f"Train Q-Dist: {avg_random_shuffled_evals['quartet_dist_train']:.4f} | "
                f"Test Q-Dist: {avg_random_shuffled_evals['quartet_dist_test']:.4f}"
            )
    
    # Save averages of random shuffle and random embedding evaluations
    results["base_eval_avg"]["random_embedding"] = avg_random_embedding_evals
    results["base_eval_avg"]["random_shuffle"] = avg_random_shuffled_evals
        
    # Get input dimension
    assert len({df.shape[1] for df in train_dataset.data_normalized + test_dataset.data_normalized}) == 1, "Inconsistent input dimensions in data" # Ensure all data has same number of dimensions
    input_dim = train_dataset.data_normalized[0].shape[1]

    logging.info(f"Input dimension: {input_dim}")

    # Log memory usage and adjust batch size if needed
    # if "cuda" in str(device) and input_dim > 10000:
    #     logging.warning(f"High-dimensional data ({input_dim} features) detected on GPU")
    #     logging.warning(f"Original batch size: {config['batch_size']}")
    #     # Further reduce batch size for very high-dimensional data
    #     if input_dim > 10000:
    #         config["batch_size"] = min(config["batch_size"], 256)
    #         logging.warning(
    #             f"Reduced batch size to {config['batch_size']} for memory efficiency"
    #         )

    # Create model
    logging.info("Creating CellTreeQMAttention model...")
    model = create_model(input_dim, str(device), cfg["model"])
    logging.info(f"Model: {model}")
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")

    # Optional MDS warm-start / anchor
    init_mode = (cfg.get("model", {}).get("init_mode") or "random").lower()
    if init_mode in {"mds_l2sq", "mds_l2"}:
        logging.info(f"Running classical MDS initialisation (mode={init_mode})...")
        dm_batches_for_init = train_dataset.get_ref_distance_matrices(device=device, add_batch_dim=True)
        if len(dm_batches_for_init) == 0:
            logging.warning("No reference distance matrices found for MDS init; skipping.")
        else:
            dm_ref = dm_batches_for_init[0].squeeze(0).detach().cpu()
            treat_as_squared = init_mode == "mds_l2sq"
            try:
                target_dim = int(cfg["model"]["output_dim"])
                mds_embedding, _ = classical_mds(dm_ref, out_dim=target_dim, treat_as_squared=treat_as_squared)
                if mds_embedding.shape[1] < target_dim:
                    pad = target_dim - mds_embedding.shape[1]
                    pad_tensor = torch.zeros(mds_embedding.shape[0], pad, dtype=mds_embedding.dtype)
                    mds_embedding = torch.cat([mds_embedding, pad_tensor], dim=1)
                elif mds_embedding.shape[1] > target_dim:
                    mds_embedding = mds_embedding[:, :target_dim]
                mds_embedding = mds_embedding - mds_embedding.mean(dim=0, keepdim=True)
                loss_cfg = cfg.get("loss", {}) or {}
                anchor_mode = (loss_cfg.get("mds_anchor_mode", "embedding") or "embedding").lower()
                model._mds_anchor_mode = anchor_mode
                mds_embed_device = mds_embedding.to(device)
                model.register_buffer("_mds_target", mds_embed_device)
                if anchor_mode == "pairwise":
                    dm_target = pairwise_distances(mds_embed_device.unsqueeze(0), metric=loss_cfg.get("metric", "euclidean")).squeeze(0).pow(2)
                    model.register_buffer("_mds_target_dm", dm_target)
                else:
                    model.register_buffer("_mds_target_dm", torch.zeros(1, device=device))
                model._mds_anchor_weight = float(loss_cfg.get("mds_anchor_weight", 0.0))
                model._mds_anchor_decay = int(loss_cfg.get("mds_anchor_decay_steps", cfg.get("schedule", {}).get("phaseB_ramp_steps", 0)))
                logging.info("MDS embedding cached for Phase-A anchoring.")
            except Exception as exc:
                logging.warning(f"MDS init failed ({exc}); continuing with random init.")

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )


    # Evaluate the model pre-traininng
    train_metrics = evaluate_model(
                    model, train_dataset, cfg, device, "train", gen=gen
                )
    test_metrics = evaluate_model(
                    model, test_dataset, cfg, device, "test", gen=gen
                )
    
    logging.info(
                    f"Pre-training evaluations: "
                    f"Train RF: {train_metrics['rf_train']:.4f} | "
                    f"Test RF: {test_metrics['rf_test']:.4f} | "
                    f"Train Q-Dist: {train_metrics['quartet_dist_train']:.4f} | "
                    f"Test Q-Dist: {test_metrics['quartet_dist_test']:.4f}"
                )
    results["base_eval_avg"]["pre-training"] = train_metrics | test_metrics


    # Training loop
    logging.info("Starting training...")
    best_rf = float("inf")
    best_epoch = 0
    best_step = 0

    start_time = time.time()

    for epoch in range(cfg["training"]["num_epochs"]):
        epoch_start = time.time()

        # Train one epoch (handles multiple steps and evaluations internally)
        epoch_results = train_one_epoch(
            model, train_dataset, test_dataset, optimizer, epoch, cfg, str(device), gen=gen
        )

        # Store epoch-level metrics
        results["epoch_avg_loss"].append(epoch_results["avg_loss"])

        # Store all evaluation results from this epoch
        for eval_result in epoch_results["step_metrics"]["evaluations"]:
            results["all_evaluations"].append(eval_result)

            # Check if this is the best model based on test RF
            current_test_rf = eval_result["test_rf"]
            if current_test_rf < best_rf:
                best_rf = current_test_rf
                best_epoch = eval_result["epoch"]
                best_step = eval_result["step"]
                # Save best model
                model_path = os.path.join(out_dir, "best_model.pth")
                torch.save(model.state_dict(), model_path)
                logging.info(
                    f"New best model saved: RF={best_rf:.4f} at Epoch {best_epoch}, Step {best_step}"
                )

        epoch_time = time.time() - epoch_start
        logging.info(
            f"Epoch {epoch + 1}/{cfg['training']['num_epochs']} completed | "
            f"Avg Loss: {epoch_results['avg_loss']:.4f} | "
            f"Steps: {epoch_results['max_step']} | "
            f"Evaluations: {len(epoch_results['step_metrics']['evaluations'])} | "
            f"Time: {epoch_time:.2f}s"
        )

    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f}s")
    logging.info(
        f"Best test RF distance: {best_rf:.4f} at Epoch {best_epoch}, Step {best_step}"
    )

    # Save final results
    import pickle
    from datetime import datetime

    results["total_time"] = total_time # save how long the model took to run

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    results_path = os.path.join(out_dir, f"training_results_{now}.pkl") # filename based on date/time
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    logging.info(f"Results saved to {results_path}")

    # Get final metrics from last evaluation
    final_eval = results["all_evaluations"][-1] if results["all_evaluations"] else None

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Dataset: {cfg['data']['dataset_name']} ({cfg['data']['lineage_name']})")
    print(f"Model: CellTreeQMAttention")
    print(f"Input dimensions: {input_dim}")
    print(f"Total parameters: {total_params:,}")
    print(f"Training epochs: {cfg['training']['num_epochs']}")
    print(f"Batch size (quartets): {cfg['training']['batch_size']}")
    print(f"Total evaluations: {len(results['all_evaluations'])}")
    print(f"Best test RF distance: {best_rf:.4f} at Epoch {best_epoch}, Step {best_step}")
    if final_eval:
        print(f"Final train RF: {final_eval['train_rf']:.4f}")
        print(f"Final test RF: {final_eval['test_rf']:.4f}")
        print(f"Final train Q-dist: {final_eval['train_quartet_dist']:.4f}")
        print(f"Final test Q-dist: {final_eval['test_quartet_dist']:.4f}")
    print(f"Training time: {total_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
