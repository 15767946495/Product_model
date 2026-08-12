#!/bin/bash
# cropnet TFT 训练 — MSE 损失
# 单样本：masked MSE（>=8月时间步平均），batch：均值聚合

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python train.py \
  --epochs 500 \
  --lr 5e-4 \
  --batch_size 8 \
  --hidden_size 32 \
  --num_heads 2 \
  --dropout 0.1 \
  --weight_decay 5e-4 \
  --val_year 2021 \
  --states minnesota,wisconsin,michigan,iowa,illinois,indiana,ohio,missouri,kentucky \
  --seed 42
