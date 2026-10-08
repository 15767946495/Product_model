#!/usr/bin/env bash
# 固定 2021 Corn MMST-TFT 正式训练，单机单卡，不构建验证集。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PYTHON="${PYTHON:-/root/miniconda3/envs/product/bin/python}"

exec "${PYTHON}" TFT_model/train_mmst_frontend.py \
  "$@" \
  --root-dir /data/raid0/hqx/Product_model_runtime/DataSrc \
  --train-config MMST-ViT/data/corn_train.json \
  --soil-file /data/raid0/hqx/Product_model_runtime/DataSrc/soil/county_soil.json \
  --pretrained MMST-ViT/output_dir/pvt_simclr/checkpoint-latest.pth \
  --epochs 100 \
  --lr 1e-3 \
  --outer-batch-size 1 \
  --hidden-size 512 \
  --heads 4 \
  --dropout 0.0 \
  --weight-decay 0.05 \
  --warmup-epochs 5 \
  --seed 42 \
  --train-year 2021 \
  --device cuda:0
