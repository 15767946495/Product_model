#!/usr/bin/env bash
set -euo pipefail

if (($# != 0)); then printf 'error: positional arguments are not supported\n' >&2; exit 2; fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/GNN-RNN"
PYTHON="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
DATASET_JSONL="${DATASET_JSONL:-$PROJECT_ROOT/train_dataset/dataset.jsonl}"
DATA_ROOT="${DATA_ROOT:-/data/raid0/hqx/DataSrc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MODEL_DIR/output_dir}"
EPOCHS="${EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
AGGREGATOR="${AGGREGATOR:-mean}"
GPU_ID="${GPU_ID:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

for name in EPOCHS NUM_WORKERS; do
  value="${!name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf 'error: %s must be a positive integer\n' "$name" >&2
    exit 2
  fi
done
case "$DRY_RUN" in 0|1) ;; *) printf 'error: DRY_RUN must be 0 or 1\n' >&2; exit 2 ;; esac
case "$AGGREGATOR" in mean|pool) ;; *) printf 'error: AGGREGATOR must be mean or pool\n' >&2; exit 2 ;; esac
if [[ -n "${BATCH_SIZE:-}" && ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  printf 'error: BATCH_SIZE must be a positive integer\n' >&2
  exit 2
fi

LOG_DIR="$OUTPUT_ROOT/logs/$RUN_ID"
LOG_FILE="$LOG_DIR/run.log"
run() {
  local formatted
  formatted="$(printf '%q ' "$@")"
  printf 'COMMAND=%s\n' "$formatted"
  if [[ "$DRY_RUN" == 1 ]]; then return 0; fi
  printf 'COMMAND=%s\n' "$formatted" >>"$LOG_FILE"
  "$@" >>"$LOG_FILE" 2>&1
}
if [[ "$DRY_RUN" != 1 ]]; then mkdir -p "$LOG_DIR"; fi

train=("$PYTHON" "$MODEL_DIR/train.py" --dataset-jsonl "$DATASET_JSONL" --data-root "$DATA_ROOT"
  --output_root "$OUTPUT_ROOT" --run_id "$RUN_ID" --device cuda:0 --epochs "$EPOCHS"
  --num_workers "$NUM_WORKERS" --aggregator "$AGGREGATOR")
if [[ -n "${BATCH_SIZE:-}" ]]; then train+=(--batch_size "$BATCH_SIZE"); fi
checkpoint="$OUTPUT_ROOT/gnn_rnn/$RUN_ID/checkpoint-final.pth"
infer=("$PYTHON" "$MODEL_DIR/infer.py" --checkpoint "$checkpoint" --dataset-jsonl "$DATASET_JSONL"
  --data-root "$DATA_ROOT" --device cuda:0)
run "${train[@]}"
run "${infer[@]}"
