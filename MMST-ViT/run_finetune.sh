#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/product/bin/python}"

cd "$ROOT"

exec "$PYTHON" main_finetune_mmst_vit.py \
  --root_dir /data/raid0/hqx/Product_model_runtime/DataSrc \
  --data_file_train ./data/corn_train.json \
  --data_file_val ./data/corn_val.json \
  --pvt_simclr ./output_dir/pvt_simclr/checkpoint-latest.pth \
  --device cuda:0 \
  --epochs 100 \
  --lr 1e-3 \
  --warmup_epochs 5 \
  --output_dir ./output_dir/mmst_vit \
  --log_dir ./output_dir/mmst_vit \
  "$@"
