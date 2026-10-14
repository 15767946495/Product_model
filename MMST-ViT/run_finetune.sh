#!/usr/bin/env bash
# MMST-ViT 微调（只用 2021）并对 checkpoint-final.pth 做 2022 最终测试。
# PRETRAIN_RUN_ID 指向 output_dir/pvt_simclr/<id>/checkpoint-final.pth。
# 输出 output_dir/mmst_vit/$RUN_ID/。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
: "${PRETRAIN_RUN_ID:?set PRETRAIN_RUN_ID to the pvt_simclr run id}"

cd "$ROOT"

exec "$PYTHON" main_finetune_mmst_vit.py \
  --dataset-jsonl ../train_dataset/dataset.jsonl \
  --data-root /data/raid0/hqx/DataSrc \
  --pvt_simclr "./output_dir/pvt_simclr/$PRETRAIN_RUN_ID/checkpoint-final.pth" \
  --output_root ./output_dir \
  --run_id "$RUN_ID" \
  --device cuda:0 \
  --epochs 100 \
  --lr 1e-3 \
  --warmup_epochs 5 \
  "$@"
