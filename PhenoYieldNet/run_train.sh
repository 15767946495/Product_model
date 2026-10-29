#!/usr/bin/env bash
set -euo pipefail

if (($# != 0)); then printf 'error: positional arguments are not supported\n' >&2; exit 2; fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/PhenoYieldNet"
PYTHON="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
DATASET_JSONL="${DATASET_JSONL:-$PROJECT_ROOT/train_dataset/dataset.jsonl}"
DATA_ROOT="${DATA_ROOT:-/data/raid0/hqx/DataSrc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MODEL_DIR/output_dir}"
EPOCHS="${EPOCHS:-10}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
NO_PRETRAIN="${NO_PRETRAIN:-0}"
GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-0}"
FROZEN_FM="${FROZEN_FM:-0}"
REUSE_STAGE1_CHECKPOINT="${REUSE_STAGE1_CHECKPOINT:-}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

for name in EPOCHS STAGE1_EPOCHS STAGE2_EPOCHS NUM_WORKERS; do
  value="${!name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf 'error: %s must be a positive integer\n' "$name" >&2
    exit 2
  fi
done
for name in DRY_RUN NO_PRETRAIN GRAD_CHECKPOINTING FROZEN_FM; do
  value="${!name:-0}"
  case "$value" in 0|1) ;; *) printf 'error: %s must be 0 or 1\n' "$name" >&2; exit 2 ;; esac
done
if [[ -n "${STAGE1_BATCH_SIZE:-}" && ! "$STAGE1_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  printf 'error: STAGE1_BATCH_SIZE must be a positive integer\n' >&2
  exit 2
fi

run() {
  local logfile="$1"
  shift
  local formatted
  formatted="$(printf '%q ' "$@")"
  printf 'COMMAND=%s\n' "$formatted"
  if [[ "$DRY_RUN" == 1 ]]; then return 0; fi
  printf 'COMMAND=%s\n' "$formatted" >>"$logfile"
  "$@" >>"$logfile" 2>&1
}
LOG_DIR="$OUTPUT_ROOT/logs/$RUN_ID"
if [[ "$DRY_RUN" != 1 ]]; then mkdir -p "$LOG_DIR"; fi

if [[ -n "$REUSE_STAGE1_CHECKPOINT" ]]; then
  printf 'REUSE_STAGE1_CHECKPOINT=%s\n' "$REUSE_STAGE1_CHECKPOINT"
  stage1_checkpoint="$REUSE_STAGE1_CHECKPOINT"
else
  stage1_id="${RUN_ID}-stage1"
  stage1=("$PYTHON" "$MODEL_DIR/train_da.py" --dataset-jsonl "$DATASET_JSONL" --data-root "$DATA_ROOT"
    --output_root "$OUTPUT_ROOT" --run_id "$stage1_id" --device cuda:0 --epochs "$STAGE1_EPOCHS"
    --num_workers "$NUM_WORKERS")
  if [[ -n "${STAGE1_BATCH_SIZE:-}" ]]; then stage1+=(--batch_size "$STAGE1_BATCH_SIZE"); fi
  if [[ "$NO_PRETRAIN" == 1 ]]; then stage1+=(--no_pretrain); fi
  stage1_checkpoint="$OUTPUT_ROOT/phenoyieldnet_stage1/$stage1_id/checkpoint-final.pth"
  run "$LOG_DIR/stage1.log" "${stage1[@]}"
fi

stage2=("$PYTHON" "$MODEL_DIR/train_ft.py" --dataset-jsonl "$DATASET_JSONL" --data-root "$DATA_ROOT"
  --output_root "$OUTPUT_ROOT" --run_id "$RUN_ID" --device cuda:0 --epochs "$STAGE2_EPOCHS"
  --num_workers "$NUM_WORKERS" --pretrain_model "$stage1_checkpoint")
if [[ "$GRAD_CHECKPOINTING" == 1 ]]; then stage2+=(--grad_checkpointing); fi
if [[ "$FROZEN_FM" == 1 ]]; then stage2+=(--frozen_fm); fi
checkpoint="$OUTPUT_ROOT/phenoyieldnet/$RUN_ID/checkpoint-final.pth"
infer=("$PYTHON" "$MODEL_DIR/infer.py" --checkpoint "$checkpoint" --dataset-jsonl "$DATASET_JSONL"
  --data-root "$DATA_ROOT" --device cuda:0 --num_workers "$NUM_WORKERS")

if [[ "$DRY_RUN" != 1 && ! -f "$stage1_checkpoint" ]]; then
  printf 'error: missing Stage 1 checkpoint: %s\n' "$stage1_checkpoint" >&2
  exit 1
fi
run "$LOG_DIR/stage2.log" "${stage2[@]}"
run "$LOG_DIR/infer.log" "${infer[@]}"
