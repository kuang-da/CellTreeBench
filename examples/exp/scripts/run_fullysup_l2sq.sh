#!/usr/bin/env bash
set -euo pipefail

# Matrix runner for single-tree fully-supervised l2sq experiments
# Usage: examples/exp/scripts/run_fullysup_l2sq.sh <device> <data_dir> <out_root>

DEVICE=${1:-cuda:0}
DATA_DIR=${2:-/workspaces/CellTreeBench-Phylo/data}
OUT_ROOT=${3:-/workspaces/CellTreeBench-Phylo/examples/outs/single_tree_fullysup}

BASE_CFG="examples/exp/configs/single_tree_fullysup/exp-single-tree-fullysup-l2sq.yaml"
TMP_DIR="examples/exp/tmp_configs"
mkdir -p "$TMP_DIR" "$OUT_ROOT"

# Factors
OUTPUT_DIMS=(128 256)
NUM_LAYERS=(2 4)
P_CAP_MAX=(0.8 1.0)
CAP_BY_ACT=(false true)
PAIR_SAMPLING=(uniform bucket_equal)
INIT_MODES=(random mds_l2sq)

IDX=0
for OD in "${OUTPUT_DIMS[@]}"; do
  for NL in "${NUM_LAYERS[@]}"; do
    for CAPMAX in "${P_CAP_MAX[@]}"; do
      for CBA in "${CAP_BY_ACT[@]}"; do
        for PS in "${PAIR_SAMPLING[@]}"; do
          for IM in "${INIT_MODES[@]}"; do
            IDX=$((IDX+1))
            TAG="od${OD}_nl${NL}_cap${CAPMAX}_cba${CBA}_ps${PS}_im${IM}"
            CFG_OUT="$TMP_DIR/cfg_${TAG}.yaml"
            OUT_DIR="$OUT_ROOT/$TAG"
            mkdir -p "$OUT_DIR"
            echo "[run $IDX] $TAG"
            # Build override config via python (merge base + overrides)
            python - "$BASE_CFG" "$CFG_OUT" <<'PY'
import sys, yaml
base_path, out_path = sys.argv[1], sys.argv[2]
base = yaml.safe_load(open(base_path))
od    = int("${OD}")
nl    = int("${NL}")
capmx = float("${CAPMAX}")
cba   = True if "${CBA}".lower()=="true" else False
ps    = "${PS}"
im    = "${IM}"
if ps == "uniform":
    base.setdefault("training", {}).setdefault("pair_sampling", {})["mode"] = "none"
else:
    base.setdefault("training", {}).setdefault("pair_sampling", {})["mode"] = "bucket_equal"
base.setdefault("model", {})["output_dim"] = od
base.setdefault("model", {})["num_layers"] = nl
base.setdefault("loss", {})["adaptive_P_cap_max"] = capmx
base.setdefault("loss", {})["cap_by_act"] = cba
base.setdefault("model", {})["init_mode"] = im
yaml.safe_dump(base, open(out_path, 'w'))
print(out_path)
PY
            # Launch training
            python examples/train_phylogenetic_cellencoder.py \
              --config "$CFG_OUT" \
              --output-dir "$OUT_DIR" \
              --data-dir "$DATA_DIR" \
              --device "$DEVICE"
          done
        done
      done
    done
  done
done

echo "All runs completed. Outputs in $OUT_ROOT"

