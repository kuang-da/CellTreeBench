#!/bin/bash
timestamp=$(date +"%Y%m%d_%H%M%S")
python /workspaces/CellTreeBench-Phylo/examples/train_phylogenetic_cellencoder.py > /workspaces/CellTreeBench-Phylo/examples/logs/exp_${timestamp}.log 2>&1