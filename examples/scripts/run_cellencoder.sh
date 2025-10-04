#!/usr/bin/env bash
set -euo pipefail

# Experiment set configuration
REPO_ROOT=${REPO_ROOT:-"/workspaces/CellTreeBench-Phylo"}
EXAMPLES_DIR="$REPO_ROOT/examples"
EXP_SET_NAME=${EXP_SET_NAME:-"test_cellencoder"}
RUN_FILE="$EXAMPLES_DIR/scripts/run_cellencoder_single.sh"
OUT_DIR=${OUT_DIR:-"$EXAMPLES_DIR/outs/$EXP_SET_NAME"}
CONFIG_DIR=${CONFIG_DIR:-"$EXAMPLES_DIR/exp/configs/$EXP_SET_NAME"}
DATA_DIR=${DATA_DIR:-"$REPO_ROOT/data"}
LOG_DIR=${LOG_DIR:-"$EXAMPLES_DIR/logs/$EXP_SET_NAME"}

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "[ERROR] Config directory not found: $CONFIG_DIR" >&2
    exit 1
fi

if [[ ! -x "$RUN_FILE" ]]; then
    echo "[ERROR] Run file not executable: $RUN_FILE" >&2
    exit 1
fi

# Batch size configuration
BATCH_SIZE=${BATCH_SIZE:-2}

# Define experiments as "<exp_id> <device> [extra args...]"
experiments=(
  # "exp-1 cuda:0"
#   "exp-2 cuda:1"
#   "exp-3 cuda:1"
#   "exp-4 cuda:1"
#   "exp-5 cuda:0"
#   "exp-6 cuda:1"
#   "exp-7 cuda:0"
  "exp-9 cuda:1"
)

# Batch execution logic
TOTAL_EXPERIMENTS=${#experiments[@]}
TOTAL_BATCHES=$(( (TOTAL_EXPERIMENTS + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Total number of batches to run: $TOTAL_BATCHES"
for ((i=0; i<TOTAL_EXPERIMENTS; i+=BATCH_SIZE)); do
    echo "Starting batch $((i / BATCH_SIZE + 1))..."
    for ((j=i; j<i+BATCH_SIZE && j<TOTAL_EXPERIMENTS; j++)); do
        IFS=' ' read -ra ARGS <<<"${experiments[j]}"
        EXP_ID="${ARGS[0]}"
        DEVICE="${ARGS[1]:-auto}"
        EXTRA_ARGS=()
        if (( ${#ARGS[@]} > 2 )); then
            EXTRA_ARGS=("${ARGS[@]:2}")
        fi

        LOG_FILE="$LOG_DIR/${EXP_ID}.log"
        echo "Initiating experiment $EXP_ID on device $DEVICE"

        {
        echo "Current time: $(date)"
        echo "Running experiment $EXP_ID on device $DEVICE"
        if (( ${#EXTRA_ARGS[@]} > 0 )); then
            time bash "$RUN_FILE" "$EXP_ID" "$DEVICE" "$CONFIG_DIR" "$DATA_DIR" "$OUT_DIR" "${EXTRA_ARGS[@]}"
        else
            time bash "$RUN_FILE" "$EXP_ID" "$DEVICE" "$CONFIG_DIR" "$DATA_DIR" "$OUT_DIR"
        fi
        echo "Experiment $EXP_ID on device $DEVICE completed."
        } > "$LOG_FILE" 2>&1 &
    done

    wait
    echo "Batch $((i / BATCH_SIZE + 1)) completed."
done

echo "All experiments completed."

echo "Checking log files for errors..."
ERROR_LOGS=$(grep -l "Error" "$LOG_DIR"/*.log 2>/dev/null || true)

if [[ -n "$ERROR_LOGS" ]]; then
    echo "The following log files contain errors:"
    echo "$ERROR_LOGS"
else
    echo "No errors found in the log files."
fi
