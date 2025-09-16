#!/usr/bin/env bash
set -euo pipefail

# Package the minimal celegans_small dataset required by CI.
# Contents included in the tarball:
#   data/celegans_small/P0/tree_df-P0.csv
#   data/celegans_small/P0/exprs_df_cache.pkl
#   data/celegans_small/raw/metadata.csv
# Artifacts and other derived files are intentionally excluded.

BENCH_ROOT=${BENCH_ROOT:-/workspaces/CellTreeQM/CellTreeBench}
DATASET=${DATASET:-celegans_small}
LINEAGE=${LINEAGE:-P0}
OUT_TGZ=${OUT_TGZ:-${BENCH_ROOT}/data/${DATASET}_mini.tgz}

REQ_TREE="${BENCH_ROOT}/data/${DATASET}/${LINEAGE}/tree_df-${LINEAGE}.csv"
REQ_EXPR="${BENCH_ROOT}/data/${DATASET}/${LINEAGE}/exprs_df_cache.pkl"
REQ_META="${BENCH_ROOT}/data/${DATASET}/raw/metadata.csv"

for f in "$REQ_TREE" "$REQ_EXPR" "$REQ_META"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Required file not found: $f" >&2
    exit 1
  fi
done

echo "[INFO] Packaging minimal dataset from: $BENCH_ROOT"
echo "[INFO] Output tarball: $OUT_TGZ"

# Ensure output directory exists
mkdir -p "$(dirname "$OUT_TGZ")"

pushd "$BENCH_ROOT" >/dev/null
tar -czf "$OUT_TGZ" \
  "data/${DATASET}/${LINEAGE}/tree_df-${LINEAGE}.csv" \
  "data/${DATASET}/${LINEAGE}/exprs_df_cache.pkl" \
  "data/${DATASET}/raw/metadata.csv"
popd >/dev/null

# Optional: write checksum alongside
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$OUT_TGZ" > "${OUT_TGZ}.sha256"
  echo "[INFO] Wrote checksum: ${OUT_TGZ}.sha256"
fi

echo "[INFO] Done. Contents:" 
tar -tzf "$OUT_TGZ"


