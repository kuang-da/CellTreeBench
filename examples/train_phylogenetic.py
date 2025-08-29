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

# Add the CellTreeBench to path for imports
sys.path.append("/workspaces/CellTreeBench")

# Import our minimal utilities
from utils_minimal import (
    distance_error,
    pairwise_distances,
    train_reconstruct_eval,
    test_reconstruct_eval,
    reconstruct_from_dm,
)
from loss_minimal import (
    additivity_error_quartet_tensor,
    triplet_loss_quartet_tensor_vectorized,
    quadruplet_loss_quartet_tensor_vectorized,
)
from quartet_utils_minimal import (
    generate_quartets_tensor,
    get_quartet_dist,
)
from celltreeqm_attention import CellTreeQMAttention

# Import CellTreeBench dataset loader
from celltreebench.datasets.celegans import load_celegans_supervised_split
from celltreebench.datasets.phylo_dataset_creator import PhyloDatasetCreator
from math import comb


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

def check_embedding(get_node_mtx, model, dist_metric, device):
    """
    Copied from utils_minimal.py
    Check the embedding produced by the model for a given dataset.
 
    Args:
        dataset: Dataset object containing the data.
        model: Neural network model.
        dist_metric (str): Distance metric to use.
        device: Device to run computation on.

    Returns:
        tuple: (embeddings, distance_matrix, node_names)
    """
    model.eval()
    with torch.no_grad():
        # Get the node matrix data
        node_mtx_dict = get_node_mtx
        pts_mtx = (
            torch.tensor(node_mtx_dict["node_mtx"], dtype=torch.float)
            .unsqueeze(0)
            .to(device)
        )
        node_names = node_mtx_dict["node_names"]

        # Get embeddings
        embeddings = model(pts_mtx)  # Shape: (1, N, D)

        # Compute distance matrix
        dm = pairwise_distances(embeddings, metric=dist_metric)
        dm = dm.squeeze(0).cpu().numpy()  # Shape: (N, N)

        return embeddings, dm, node_names
    

def create_model(input_dim, device="cpu"):
    """
    Create the CellTreeQMAttention model with config parameters.

    Args:
        input_dim (int): Input feature dimension from the dataset.
        device (str): Device to use for computation.

    Returns:
        CellTreeQMAttention: The initialized model.
    """
    # Model configuration from the config file
    config = {
        "proj_dim": 1024,
        "output_dim": 128,
        "hidden_dim": 1024,
        "num_heads": 2,
        "num_layers": 8,
        "dropout_data": 0.1,
        "dropout_metric": 0.1,
        "norm_method": None,
        "gate_type": "none",  # No gating for this example
    }

    model = CellTreeQMAttention(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        output_dim=config["output_dim"],
        dropout_data=config["dropout_data"],
        dropout_metric=config["dropout_metric"],
        norm_method=config["norm_method"],
        proj_dim=config["proj_dim"],
        gate_mode=config["gate_type"],
        device=device,
    )

    return model.to(device)


def train_one_epoch(
    model, train_dataset, test_dataset, optimizer, epoch, config, device="cpu", gen=None
):
    """
    Train the model for one epoch using the research codebase's unconventional design:
    - No actual data batching - always process the full dataset
    - batch_size controls quartet sampling, not data batch size
    - Multiple training steps per epoch with different quartet samples
    - Evaluation every eval_interval steps (not epochs)

    Args:
        model: The neural network model.
        train_dataset: Training dataset.
        test_dataset: Test dataset.
        optimizer: Optimizer for training.
        epoch (int): Current epoch number.
        config (dict): Training configuration.
        device (str): Device for computation.
        gen (torch.Generator): Random number generator for reproducibility.

    Returns:
        dict: Dictionary containing metrics for this epoch.
    """
    model.train()

    # Setup data for fast reference (ONCE, outside step loop)
    node_mtx_list = train_dataset.get_node_mtx()
    pts_mtx = [(
        torch.tensor(mtx["node_mtx"], dtype=torch.float)
        .unsqueeze(0)  # Add batch dimension
        .to(device)
    ) for mtx in node_mtx_list]

    dm_ref = [ref_dm.unsqueeze(0).to(device) for ref_dm in train_dataset.ref_dm]

    # Training configuration
    batch_size = config["batch_size"]  # Number of quartets to sample per step
    eval_interval = config["eval_interval"]
    weight_D = config["weight_D"]
    weight_P = config["weight_P"]
    weight_close = config["weight_close"]
    weight_push = config["weight_push"]
    push_margin = config["push_margin"]
    dist_metric = config["metric"]
    metric_loss_type = config["metric_loss"]

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
    }

    running_loss = 0.0

    # Multiple training steps per epoch (the unconventional part!)
    for step_count in range(max_step):
        optimizer.zero_grad()

        # Generate quartets (this samples random quartets each time!)

        proportions = train_dataset.get_proportions()
        proportions = torch.tensor(proportions, dtype=torch.float)
        samples = torch.multinomial(proportions, batch_size, replacement=True, generator=gen)  # draw N samples with replacement according to proportions
        counts = samples.bincount(minlength=len(proportions)) # count how many times each group was chosen

        batch_D = 0
        batch_P = 0
        batch_P_close = 0
        batch_P_push = 0

        # Process each tree's quartet
        for i, count in enumerate(counts):
            # Always process the FULL dataset (same pts_mtx every step) (for each tree)
            trans_pts_mtx = model(pts_mtx[i])

              # 1. Distance Error Loss (L_D) - on full dataset for each tree
            batch_D += distance_error(
                pts_mtx[i],
                trans_pts_mtx,
                diff_norm=1,
                dist_metric=dist_metric,
                )

            # 2. Quartet/Metric Loss (L_P) - sample DIFFERENT quartets each step
            # Compute distance matrix from embedding
            dm = pairwise_distances(trans_pts_mtx, metric=dist_metric).to(device)
            if count > 0:
                dm_quartets, dm_ref_quartets = generate_quartets_tensor(
                    batch_size=count,  # Number of quartets to sample
                    dm=dm,
                    dm_ref=dm_ref[i],
                    device=device,
                    seed=int(torch.randint(0, 100_000_000_000, (1,), generator=gen))
                    )
                # Compute quartet loss based on metric_loss_type
                if metric_loss_type == "additivity":
                    batch_P_temp, batch_P_close_temp, batch_P_push_temp, _ = additivity_error_quartet_tensor(
                        dm_quartets=dm_quartets,
                        dm_ref_quartets=dm_ref_quartets,
                        weight_close=weight_close,
                        weight_push=weight_push,
                        push_margin=push_margin,
                        matching_mode="mismatched",
                        device=device,
                    )
                    batch_P += count * batch_P_temp
                    batch_P_close += count * batch_P_close_temp
                    batch_P_push += count * batch_P_push_temp
                elif metric_loss_type == "triplet":
                    batch_P += count * triplet_loss_quartet_tensor_vectorized(
                        dm_quartets=dm_quartets,
                        dm_ref_quartets=dm_ref_quartets,
                        margin=push_margin,
                        device=device,
                    ) 
                    batch_P_close += count * torch.tensor(0.0, device=device)
                    batch_P_push += count * torch.tensor(0.0, device=device)
                elif metric_loss_type == "quadruplet":
                    batch_P_temp, _, _, _, _ = quadruplet_loss_quartet_tensor_vectorized(
                        dm_quartets=dm_quartets,
                        dm_ref_quartets=dm_ref_quartets,
                        alpha=0.5,
                        beta=0.5,
                        device=device,
                    )

                    batch_P += count * batch_P_temp
                    batch_P_close += count * torch.tensor(0.0, device=device)
                    batch_P_push += count * torch.tensor(0.0, device=device)
                else:
                    raise ValueError(f"Invalid metric loss: {metric_loss_type}")
        
        # multiplied by count inside of loop to ensure that each metric is represennted based on how manny quartets from that tree are used then we average here:
        batch_P /=sum(counts)
        batch_P_close /= sum(counts)
        batch_P_push /= sum(counts)

        # 3. Feature Gate Loss (if applicable)
        gate_loss = torch.tensor(0.0, device=device)
        if hasattr(model, "feature_gate") and model.feature_gate is not None:
            gate_loss = model.feature_gate.compute_penalty(
                penalty_type="sparsity",
                lambda_penalty=config.get("weight_gate", 0.0),
            )

        # Total loss
        total_loss = weight_D * batch_D + weight_P * batch_P + gate_loss
        running_loss += total_loss.item()

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Store step metrics
        epoch_metrics["losses"].append(total_loss.item())
        epoch_metrics["loss_D"].append(batch_D.item())
        epoch_metrics["loss_P"].append(batch_P.item())
        epoch_metrics["loss_P_close"].append(batch_P_close.item())
        epoch_metrics["loss_P_push"].append(batch_P_push.item())
        epoch_metrics["gate_loss"].append(gate_loss.item())

        # Periodic Evaluation (every eval_interval STEPS, not epochs!)
        if step_count % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                logging.info(
                    f"[Epoch {epoch+1}, Step {step_count}/{max_step}] Evaluating..."
                )

                # Evaluate on train set
                train_metrics = evaluate_model(
                    model, train_dataset, config, device, "train", gen=gen
                )

                # Evaluate on test set
                test_metrics = evaluate_model(
                    model, test_dataset, config, device, "test", gen=gen
                )

                # Store evaluation results
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

                # Log progress
                logging.info(
                    f"[Epoch {epoch+1}, Step {step_count}/{max_step}] "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Train RF: {train_metrics['rf_train']:.4f} | "
                    f"Test RF: {test_metrics['rf_test']:.4f} | "
                    f"Train Q-Dist: {train_metrics['quartet_dist_train']:.4f} | "
                    f"Test Q-Dist: {test_metrics['quartet_dist_test']:.4f}"
                )

                # Detailed loss breakdown
                logging.info(
                    f"  Loss breakdown: D={batch_D.item():.4f}, "
                    f"P={batch_P.item():.4f}, "
                    f"P_close={batch_P_close.item():.4f}, "
                    f"P_push={batch_P_push.item():.4f}, "
                    f"Gate={gate_loss.item():.4f}"
                )
                # Edge split stats
                logging.info(f"  Edge split stats: Common={train_metrics['split_prop_common_train']:.4f}, "
                             f"Ref={train_metrics['split_prop_ref_train']:.4f}")

            model.train()  # Return to train mode

    # Calculate average loss  the epoch
    avg_epoch_loss = running_loss / max_step if max_step > 0 else running_loss

    return {
        "avg_loss": avg_epoch_loss,
        "max_step": max_step,
        "step_metrics": epoch_metrics,
    }

def avg_edge_split_proportion(edges):
    """
    Calculate the average proportion of edge splits for a list of edge pairs.
    """
    lens = []
    for e in edges:
        if min(len(e[0]), len(e[1])) > 1:
            lens.append(comb(len(e[0]), 2) * comb(len(e[1]), 2))

    return sum(lens)/len(lens) if len(lens) > 0 else 0

def evaluate_model(model, dataset, config, device="cpu", dataset_name="train", gen=None):
    """
    Evaluate the model on a dataset.

    Args:
        model: The neural network model.
        dataset: Dataset to evaluate on.
        config (dict): Configuration dictionary.
        device (str): Device for computation.
        dataset_name (str): Name of the dataset for logging.
        gen (torch.Generator): Random number generator.

    Returns:
        list: List of
            dict: Dictionaries containing evaluation metrics.
    """
    model.eval()

    with torch.no_grad():
        eval_results = {
            f"rf_{dataset_name}": [],
            f"rf_max_{dataset_name}": [],
            f"quartet_dist_{dataset_name}": [],
            f"split_prop_common_{dataset_name}": [],
            f"split_prop_ref_{dataset_name}": [],
            }
        for i, get_node_mtx in enumerate(dataset.get_node_mtx()):
            # Get embeddings and distance matrix
            _, emb_dm, node_names = check_embedding(
                get_node_mtx, model, config["metric"], device
            )

                # Reconstruct tree from embedding
            emb_tree = reconstruct_from_dm(emb_dm, node_names, method="nj")


            # Compare with reference tree
            emb_topo_res = dataset.compare_trees(emb_tree, i, ref_tree="topology_tree")
            logging.info(f"split prop common edge {avg_edge_split_proportion(emb_topo_res['common_edges'])}")
            logging.info(f"split prop ref edge {avg_edge_split_proportion(emb_topo_res['ref_edges'])}")

            # Compute quartet distance
            pts_mtx = (
                torch.tensor(get_node_mtx["node_mtx"], dtype=torch.float)
                .unsqueeze(0)
                .to(device)
                )
            trans_pts_mtx = model(pts_mtx)
            dm = pairwise_distances(trans_pts_mtx, metric=config["metric"]).to(device)
            dm_ref = dataset.ref_dm[i].unsqueeze(0).to(device)

            # Generate quartets for evaluation
            dm_quartets, dm_ref_quartets = generate_quartets_tensor(
                batch_size=100000,  # Use many quartets for evaluation
                dm=dm,
                dm_ref=dm_ref,
                device=device,
                seed=int(torch.randint(0, 100_000_000_000, (1,), generator=gen))
            )
            quartet_dist = get_quartet_dist(dm_quartets, dm_ref_quartets)
            eval_results[f"rf_{dataset_name}"].append(emb_topo_res["rf"])
            eval_results[f"rf_max_{dataset_name}"].append(emb_topo_res["max_rf"])
            eval_results[f"quartet_dist_{dataset_name}"].append(quartet_dist.item())
        
        eval_results[f"split_prop_common_{dataset_name}"] = sum(eval_results[f"split_prop_common_{dataset_name}"])/len(eval_results[f"split_prop_common_{dataset_name}"]) # average between trees
        eval_results[f"split_prop_ref_{dataset_name}"] = sum(eval_results[f"split_prop_ref_{dataset_name}"])/len(eval_results[f"split_prop_ref_{dataset_name}"]) # average between trees
        
        eval_results[f"rf_{dataset_name}"] = sum(eval_results[f"rf_{dataset_name}"])/sum(eval_results[f"rf_max_{dataset_name}"]) # average between trees
        eval_results[f"quartet_dist_{dataset_name}"] = sum(eval_results[f"quartet_dist_{dataset_name}"])/len(eval_results[f"quartet_dist_{dataset_name}"]) # average between trees
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
        eval_type (str): Randomization strategy ("shuffle" or "embedding" or "basic NJ").

    Returns:
        tuple: (List of evaluations, Dictionary containing average evaluation metrics among evals)
    """
    base_eval = []
    base_eval_avg = {}
    for dataset_name in datasets.keys(): # create average dict
        base_eval_avg |= {
            f"quartet_dist_{dataset_name}": 0,
            f"rf_{dataset_name}": 0
            }

    for eval_num in range(evals):
        
        dataset_eval = {}
        
        for dataset_name, dataset in datasets.items():

            topo_eval = {
                    "rf": [],
                    "rf_max": [],
                    "quartet_dist": []
                    }
            for i, node_mtx_dict in enumerate(dataset.get_node_mtx()): # for each tree
                # Reconstruct tree (to get base RF)
                mtx = (torch.tensor(node_mtx_dict["node_mtx"], dtype=torch.float)
                    .unsqueeze(0)  # Add batch dimension
                    .to(device))
                
                if eval_type == "shuffle":
                    rand_idx = torch.randperm(mtx.size(1), generator=gen)
                    mtx = mtx[:, rand_idx, :]
                elif eval_type == "embedding":
                    rand_emb = torch.randn(mtx.size(2), 16, generator=gen)  # Random embedding
                    mtx = mtx @ rand_emb.to(device)
                elif eval_type != "basic NJ":
                    raise ValueError(f"Invalid randomization strategy: {eval_type}")
                
                dm = pairwise_distances(mtx, metric=dist_metric)
                dm = dm.squeeze(0).cpu()  # Shape: (N, N)
                tree = reconstruct_from_dm(dm.numpy(), node_mtx_dict["node_names"], method="nj")

                # Compare with reference tree
                topo_res = dataset.compare_trees(tree, i, ref_tree="topology_tree")
                topo_eval[f"rf"].append(topo_res["rf"])
                topo_eval[f"rf_max"].append(topo_res["max_rf"])
                
                
                # Generate quartets for evaluation
                dm_ref = dataset.ref_dm[i].unsqueeze(0).to(device)
                dm_quartets, dm_ref_quartets = generate_quartets_tensor(
                    batch_size=100000,  # Use many quartets for evaluation
                    dm=dm,
                    dm_ref=dm_ref,
                    device=device,
                    seed=int(torch.randint(0, 100_000_000_000, (1,), generator=gen))
                    )
                quartet_dist = get_quartet_dist(dm_quartets, dm_ref_quartets)
                topo_eval[f"quartet_dist"].append(quartet_dist.item())

            q_dist = sum(topo_eval["quartet_dist"])/len(topo_eval["quartet_dist"]) # Average of Q-dist among trees
            rf_dist = sum(topo_eval["rf"])/sum(topo_eval["rf_max"]) # Average of RF dist among trees
            dataset_eval[f"quartet_dist_{dataset_name}"] = q_dist
            dataset_eval[f"rf_{dataset_name}"] = rf_dist

            base_eval_avg[f"quartet_dist_{dataset_name}"] += q_dist
            base_eval_avg[f"rf_{dataset_name}"] += rf_dist

        dataset_eval["eval_type"] = eval_type # "shuffle" or "embedding" or "basic NJ"
        dataset_eval["eval_num"] = eval_num # num of times evaluated
        base_eval.append(dataset_eval)
    
    for dataset_name in datasets.keys():
        base_eval_avg[f"quartet_dist_{dataset_name}"] /= evals
        base_eval_avg[f"rf_{dataset_name}"] /= evals

    base_eval_avg["eval_type"] = eval_type
    base_eval_avg["evals"] = evals
    

    return base_eval, base_eval_avg

def main():
    """Main training function."""
    setup_logging()
    logging.info("Starting CellTreeQMAttention training on Phylogenetic dataset")
    # Configuration
    config = {
        # Random Seed
        "seed": 42,
        # Dataset
        "dataset_name": "phylogenetic/cellreg_100tips", # where to find data (parent directory of both train and test if using pre-split data)
        "dataset_names": {"train":0.5, "test":0.5}, #datasets to generate. Also can include split proportions if autosplit is enabled (but be careful because uneven splits may cause zero padding)
        "tree_directory": "trees", # where to find trees (within dataset directory)
        "msa_directory": "msas", # where to find MSAs (within dataset directory)
        "autosplit": True, # Autosplit the data into datasets (if disabled each dataset name will be assumed to be a directory)
        "lineage_name": None,
        "base_dir": "/workspaces/CellTreeBench",
        # Training
        "lr": 0.0001,
        "weight_decay": 0.001,
        "weight_D": 6.0,
        "weight_P": 4.0,
        "weight_close": 0.5,
        "weight_push": 1.0,
        "push_margin": 0.2,
        "batch_size": 1024,  # Reduced from 2048 for high-dimensional data
        "num_epochs": 60,
        "eval_interval": 100,
        "weight_gate": -1.0,  # Disabled
        # Model
        "metric": "euclidean",
        "metric_loss": "additivity",  # Can be "additivity", "triplet", or "quadruplet"
        # Device
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        

    }

    results = {
        "epoch_avg_loss": [],
        "all_evaluations": [],  # Store all evaluation results from all epochs
        "base_eval_avg": {}, # store average base evaluations (NOTE: formatted differently than all_evaluations)
        "config": config # store config for later reference
    }

    device = torch.device(config["device"])
    logging.info(f"Using device: {device}")
    # gen = torch.Generator().manual_seed(config["seed"])  # Set seed for reproducibility
    # torch.manual_seed(config["seed"])  # Set global seed for reproducibility
    gen = None

    # Load dataset
    logging.info(f"Loading {config['dataset_name']} dataset...")
    data_dir = f"{config['base_dir']}/data/"
    out_dir = f"{config['base_dir']}/examples/out/phylogenetic_{config['dataset_name']}"

    os.makedirs(out_dir, exist_ok=True)

    dataset_name=config["dataset_name"]

    datasets = PhyloDatasetCreator(
        dataset_name,
        dataset_names=config["dataset_names"],
        data_dir=data_dir,
        tree_directory=config["tree_directory"],
        msa_directory=config["msa_directory"],
        autosplit=config["autosplit"],
        seed=torch.randint(0, 100_000_000_000, (1,), generator=gen).item()
    )
    datasets_dict = datasets.get_dataset(datasets=["train", "test"])
    train_dataset = datasets_dict["train"]
    test_dataset = datasets_dict["test"]

    # Get amino acid and leaf count of each tree for each dataset
    results["data"] = {name: [{"leaves": node_mtx["node_mtx"].shape[0], "amino_acids": node_mtx["node_mtx"].shape[1]/22} for node_mtx in dataset.get_node_mtx()] for name, dataset in datasets_dict.items()}
    
    # Evaluate the datasets on basic NJ
    _, avg_NJ_evals = evaluate_base(datasets_dict, evals=1, dist_metric=config["metric"], device=device, gen=gen, eval_type="basic NJ")

    logging.info(
                    f"Base evaluations: "
                    f"Train RF: {avg_NJ_evals['rf_train']:.4f} | "
                    f"Test RF: {avg_NJ_evals['rf_test']:.4f} | "
                    f"Train Q-Dist: {avg_NJ_evals['quartet_dist_train']:.4f} | "
                    f"Test Q-Dist: {avg_NJ_evals['quartet_dist_test']:.4f}"
                )
    
    results["base_eval_avg"]["NJ"]= avg_NJ_evals # append basic NJ evaluation to final results dict

    logging.info(f"Train shape: {train_dataset.data_normalized[0].shape}")
    logging.info(f"Test shape: {test_dataset.data_normalized[0].shape}")
    

    # Evaluate the datasets on random shuffle and embedding
    _, avg_random_shuffled_evals = evaluate_base(datasets_dict, evals = 3, dist_metric=config["metric"], device=device, gen=gen, eval_type="shuffle")
    _, avg_random_embedding_evals = evaluate_base(datasets_dict, evals = 3, dist_metric=config["metric"], device=device, gen=gen, eval_type="embedding")

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
    model = create_model(input_dim, str(device))

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )


    # Evaluate the model pre-traininng
    train_metrics = evaluate_model(
                    model, train_dataset, config, device, "train", gen=gen
                )
    test_metrics = evaluate_model(
                    model, test_dataset, config, device, "test", gen=gen
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

    
    for epoch in range(config["num_epochs"]):
        epoch_start = time.time()

        # Train one epoch (handles multiple steps and evaluations internally)
        epoch_results = train_one_epoch(
            model, train_dataset, test_dataset, optimizer, epoch, config, str(device), gen=gen
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
            f"Epoch {epoch + 1}/{config['num_epochs']} completed | "
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
    print(f"Dataset: {config['dataset_name']} ({config['lineage_name']})")
    print(f"Model: CellTreeQMAttention")
    print(f"Input dimensions: {input_dim}")
    print(f"Total parameters: {total_params:,}")
    print(f"Training epochs: {config['num_epochs']}")
    print(f"Batch size (quartets): {config['batch_size']}")
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
    parser = argparse.ArgumentParser(description="Train CellTreeQMAttention model")
    # parser.add_argument("--phylo", type=str, help="Path to phylogenetic dataset", default=None)
    # args = parser.parse_args()
    # main(phylo_dataset=args.phylo)
    main()
