# CellTreeQMAttention - Minimal Training Example

This directory contains a minimal, self-contained example for training the CellTreeQMAttention model on C. elegans lineage data using quartet-based phylogenetic distance learning.

## Overview

This example demonstrates how to:
1. Load C. elegans developmental lineage data from CellTreeBench
2. Train a transformer-based attention model (CellTreeQMAttention) to learn phylogenetic distances
3. Use three types of loss functions: distance error, quartet additivity loss, and feature gating penalties
4. Evaluate the model using Robinson-Foulds (RF) distance and quartet distance metrics
5. Reconstruct phylogenetic trees from learned embeddings using Neighbor Joining

## Files

### Core Implementation
- `celltreeqm_attention.py` - The main attention-based model
- `utils_minimal.py` - Essential utility functions (distance calculations, tree reconstruction)
- `loss_minimal.py` - Quartet-based loss functions (additivity, triplet, quadruplet)
- `feature_gates_minimal.py` - Feature gating module (optional)
- `quartet_utils_minimal.py` - Quartet generation and distance calculation utilities

### Training Script
- `train_minimal_example.py` - Complete training script with evaluation

### Dataset Example
- `dataset_example.py` - Shows how to load different C. elegans datasets

## Usage

### Prerequisites

Make sure you have the following dependencies installed:
```bash
pip install torch numpy ete3 tqdist scipy biopython
```

You only need access to:
- CellTreeBench dataset and utilities

**Note**: This example is completely self-contained! All tree reconstruction functions have been included in `utils_minimal.py`, so no external simulation pipeline dependencies are required.

### Running the Training

```bash
cd /workspaces/CellTreeBench/examples
python train_minimal_example.py
```

### Configuration

The training script uses configuration from the original research (`celegans_dev-0/config.yaml`):

```python
config = {
    # Dataset
    "dataset_name": "celegans_small",
    "lineage_name": "P0",
    
    # Training hyperparameters
    "lr": 0.0001,
    "weight_decay": 0.01,
    "weight_D": 0.1,      # Distance error weight
    "weight_P": 20.0,     # Quartet loss weight  
    "weight_close": 1.0,  # Close pair weight
    "weight_push": 30.0,  # Push margin weight
    "push_margin": 0.1,
    "batch_size": 2048,
    "num_epochs": 20,
    
    # Model architecture
    "proj_dim": 1024,
    "output_dim": 128,
    "hidden_dim": 1024,
    "num_heads": 2,
    "num_layers": 8,
    "dropout_data": 0.1,
    "dropout_metric": 0.1,
    "norm_method": "batch_norm",
    "gate_type": "none",  # No feature gating
    
    # Loss function
    "metric": "euclidean",
    "metric_loss": "additivity",  # or "triplet", "quadruplet"
}
```

## Model Architecture

The `CellTreeQMAttention` model consists of:

1. **Optional Feature Gating** - Learns which input features are most important
2. **Linear Projection** - Projects input features to transformer dimension
3. **Transformer Encoder** - Multi-head self-attention layers with residual connections
4. **Output Projection** - Maps to final embedding dimension
5. **Normalization & Dropout** - For regularization

## Loss Functions

The model is trained with three types of losses:

### 1. Distance Error Loss (L_D)
Preserves pairwise distances between the original and embedded spaces:
```
L_D = ||D_orig - D_emb||_F
```

### 2. Quartet Loss (L_P)
Ensures quartet topologies are preserved. Three variants:

**Additivity Loss**: Enforces the quartet additivity property
**Triplet Loss**: Metric learning with anchor-positive-negative triplets  
**Quadruplet Loss**: Extension of triplet loss with additional constraints

### 3. Feature Gate Loss (L_G)
Optional sparsity penalty for feature selection (disabled by default)

## Evaluation Metrics

### Robinson-Foulds (RF) Distance
Measures topological differences between reconstructed and reference trees:
- Lower values indicate better tree reconstruction
- Normalized by maximum possible RF distance

### Quartet Distance  
Measures how often quartet topologies differ between embeddings and reference:
- Fraction of quartets with incorrect topology
- Lower values indicate better phylogenetic structure preservation

## Expected Results

For the C. elegans small dataset, you should expect:
- **Training time**: ~2-5 minutes on GPU, ~10-15 minutes on CPU
- **Final test RF distance**: ~0.1-0.3 (lower is better)
- **Final quartet distance**: ~0.05-0.15 (lower is better)

The model typically converges within 15-20 epochs.

## Output Files

The training script saves:
- `best_model.pth` - Best model weights (lowest test RF)
- `training_results.pkl` - Training curves and metrics
- Detailed logs with per-epoch progress

## Customization

### Trying Different Loss Functions
Change the `metric_loss` parameter:
```python
config["metric_loss"] = "triplet"      # or "quadruplet"
```

### Using Feature Gating
Enable feature selection:
```python
config["gate_type"] = "gumbel"         # or "sigmoid", "softmax"
config["weight_gate"] = 1.0           # Enable gate penalty
```

### Different Datasets
Change dataset size:
```python
config["dataset_name"] = "celegans_mid"    # or "celegans_large"
```

## Key Differences from Full Research Codebase

This minimal example:
- **Completely self-contained** - No external simulation pipeline dependencies
- Uses only essential dependencies (no complex dataset handlers)
- Focuses on fully supervised learning only
- Simplified evaluation (NJ reconstruction only)
- Hardcoded configuration instead of YAML files
- Single training script instead of modular pipeline
- Includes built-in tree reconstruction functions (NJ, UPGMA, hierarchical clustering)

The core model architecture and loss functions are identical to the research implementation. 