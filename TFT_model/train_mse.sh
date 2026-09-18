#!/bin/bash
# cropnet TFT 训练 — MSE 损失 + 遥感模块 (DINOv2 ViT-S)
# 默认五州（illinois/iowa/louisiana/mississippi），AG 数据从 OSS 自动下载
# 使用方法: conda activate product && bash train_mse.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python train.py \
  --epochs 500 \
  --lr 1e-4 \
  --batch_size 4 \
  --hidden_size 128 \
  --num_lstm_layers 2 \
  --num_heads 2 \
  --dropout 0.1 \
  --weight_decay 5e-4 \
  --val_year 2022 \
  --seed 42 \
  --use_remote_sensing \
  --use_constructed \
  --keep_ag_cache
