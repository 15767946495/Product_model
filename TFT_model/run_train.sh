#!/usr/bin/env bash
set -euo pipefail

if (($# != 0)); then printf 'error: positional arguments are not supported\n' >&2; exit 2; fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/TFT_model"
PYTHON="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
DATASET_JSONL="${DATASET_JSONL:-$PROJECT_ROOT/train_dataset/dataset.jsonl}"
DATA_ROOT="${DATA_ROOT:-/data/raid0/hqx/DataSrc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MODEL_DIR/output_dir}"
PVT_PRETRAINED_PATH="${PVT_PRETRAINED_PATH:-/data/raid0/hqx/pretrained/pvt_tiny.pth}"
CUTOFFS="${CUTOFFS:-04-28,05-28,06-28,07-28,08-28,09-28}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
NUM_HEADS="${NUM_HEADS:-4}"
NUM_LSTM_LAYERS="${NUM_LSTM_LAYERS:-1}"
DROPOUT="${DROPOUT:-0.1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
USE_CONSTRUCTED="${USE_CONSTRUCTED:-1}"
VIT_PRETRAINED="${VIT_PRETRAINED:-1}"
PVT_FREEZE_BACKBONE="${PVT_FREEZE_BACKBONE:-0}"
REMOTE_CONSTANT="${REMOTE_CONSTANT:-0}"
PVT_LR_SCALE="${PVT_LR_SCALE:-1.0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
GPU_ID=4
export CUDA_VISIBLE_DEVICES="$GPU_ID"

for name in EPOCHS NUM_WORKERS BATCH_SIZE HIDDEN_SIZE NUM_HEADS NUM_LSTM_LAYERS; do
  value="${!name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf 'error: %s must be a positive integer\n' "$name" >&2
    exit 2
  fi
done
if [[ ! "$DROPOUT" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
  printf 'error: DROPOUT must be a number between 0 and 1\n' >&2
  exit 2
fi
if [[ ! "$WARMUP_EPOCHS" =~ ^[0-9]+$ ]]; then
  printf 'error: WARMUP_EPOCHS must be a non-negative integer\n' >&2
  exit 2
fi
for name in DRY_RUN USE_CONSTRUCTED VIT_PRETRAINED PVT_FREEZE_BACKBONE REMOTE_CONSTANT; do
  value="${!name}"
  case "$value" in 0|1) ;; *) printf 'error: %s must be 0 or 1\n' "$name" >&2; exit 2 ;; esac
done

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
  --num_workers "$NUM_WORKERS" --batch_size "$BATCH_SIZE" --hidden_size "$HIDDEN_SIZE"
  --num_heads "$NUM_HEADS" --num_lstm_layers "$NUM_LSTM_LAYERS" --dropout "$DROPOUT"
  --warmup_epochs "$WARMUP_EPOCHS" --pvt-lr-scale "$PVT_LR_SCALE")
if [[ "$USE_CONSTRUCTED" == 1 ]]; then train+=(--use-constructed); else train+=(--no-use-constructed); fi
if [[ "$VIT_PRETRAINED" == 1 ]]; then train+=(--vit-pretrained --pvt_pretrained_path "$PVT_PRETRAINED_PATH"); fi
if [[ "$PVT_FREEZE_BACKBONE" == 1 ]]; then train+=(--vit-freeze-backbone); fi
if [[ "$REMOTE_CONSTANT" == 1 ]]; then train+=(--remote-constant); fi
checkpoint="$OUTPUT_ROOT/tft/$RUN_ID/checkpoint-final.pth"
infer=("$PYTHON" "$MODEL_DIR/infer.py" --checkpoint "$checkpoint" --dataset-jsonl "$DATASET_JSONL"
  --data-root "$DATA_ROOT" --device cuda:0 --num_workers "$NUM_WORKERS" --cutoffs "$CUTOFFS")
run "${train[@]}"
run "${infer[@]}"
