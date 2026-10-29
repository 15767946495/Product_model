#!/usr/bin/env bash
set -euo pipefail

if (($# != 0)); then printf 'error: positional arguments are not supported\n' >&2; exit 2; fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/UNet-ConvLSTM"
PYTHON="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
DATASET_JSONL="${DATASET_JSONL:-$PROJECT_ROOT/train_dataset/dataset.jsonl}"
DATA_ROOT="${DATA_ROOT:-/data/raid0/hqx/DataSrc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MODEL_DIR/output_dir}"
EPOCHS="${EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
COUNTY_BATCH_SIZE="${COUNTY_BATCH_SIZE:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
GRID_CHUNK="${GRID_CHUNK:-128}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

for name in EPOCHS NUM_WORKERS COUNTY_BATCH_SIZE IMAGE_SIZE GRID_CHUNK; do
  value="${!name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf 'error: %s must be a positive integer\n' "$name" >&2
    exit 2
  fi
done
if [[ ! "$WARMUP_EPOCHS" =~ ^[0-9]+$ ]]; then
  printf 'error: WARMUP_EPOCHS must be a non-negative integer\n' >&2
  exit 2
fi
case "$DRY_RUN" in 0|1) ;; *) printf 'error: DRY_RUN must be 0 or 1\n' >&2; exit 2 ;; esac

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
  --num_workers "$NUM_WORKERS" --county_batch_size "$COUNTY_BATCH_SIZE" --image_size "$IMAGE_SIZE"
  --grid_chunk "$GRID_CHUNK" --warmup_epochs "$WARMUP_EPOCHS")
checkpoint="$OUTPUT_ROOT/unet_convlstm/$RUN_ID/checkpoint-final.pth"
infer=("$PYTHON" "$MODEL_DIR/infer.py" --checkpoint "$checkpoint" --dataset-jsonl "$DATASET_JSONL"
  --data-root "$DATA_ROOT" --device cuda:0 --num_workers "$NUM_WORKERS")
run "${train[@]}"
run "${infer[@]}"
