#!/bin/bash
# cropnet TFT 训练 — CRUCIAL 损失
# 单样本：masked MSE（>=8月时间步平均），batch：CRUCIAL 聚合

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python train.py \
  --epochs 500 \
  --lr 5e-4 \
  --batch_size 8 \
  --hidden_size 18 \
  --num_heads 3 \
  --dropout 0.2 \
  --weight_decay 5e-4 \
  --val_year 2021 \
  --states minnesota,wisconsin,michigan,iowa,illinois,indiana,ohio,missouri,kentucky \
  --seed 42 \
  --use_crucial
