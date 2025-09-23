#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: run_cellencoder_single.sh <exp_id> <device> <config_dir> <data_dir> <out_dir_base> [extra args...]" >&2
  exit 1
fi

EXP_ID="$1"
DEVICE="$2"
CONFIG_DIR="$3"
DATA_DIR="$4"
OUT_BASE="$5"
shift 5
EXTRA_ARGS=("$@")

REPO_ROOT=${REPO_ROOT:-"/workspaces/CellTreeBench-Phylo"}
EXAMPLES_DIR="$REPO_ROOT/examples"
TRAIN_SCRIPT="$EXAMPLES_DIR/train_phylogenetic_cellencoder.py"

CONFIG_FILE="$CONFIG_DIR/$EXP_ID/config.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

EXP_OUT_DIR="$OUT_BASE/$EXP_ID"
mkdir -p "$EXP_OUT_DIR"
cp "$CONFIG_FILE" "$EXP_OUT_DIR/config.yaml"

cd "$REPO_ROOT"

CLI_ARGS=("--config" "$CONFIG_FILE" "--output-dir" "$EXP_OUT_DIR" "--data-dir" "$DATA_DIR")
if [[ "$DEVICE" != "auto" && -n "$DEVICE" ]]; then
  CLI_ARGS+=("--device" "$DEVICE")
fi
CLI_ARGS+=("${EXTRA_ARGS[@]}")

python "$TRAIN_SCRIPT" "${CLI_ARGS[@]}"
