#!/usr/bin/env bash
# PVT-SimCLR 预训练：统一 CropNet JSONL，固定 2017--2020。
# 输出 output_dir/pvt_simclr/$RUN_ID/（含 checkpoint-final.pth）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

cd "$ROOT"

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  main_pretrain_mmst_vit.py \
  --dataset-jsonl ../train_dataset/dataset.jsonl \
  --data-root /data/raid0/hqx/DataSrc \
  --output_root ./output_dir \
  --run_id "$RUN_ID" \
  --device cuda \
  "$@"
