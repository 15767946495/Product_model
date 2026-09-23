#!/bin/bash
# cropnet TFT 训练 — MSE 损失 + 遥感模块 (DINOv2 ViT-S)
# 全美 27 玉米主产州，4卡数据并行
# 使用方法: conda activate product && bash train_mse.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py \
  --epochs 500 \
  --lr 1e-5 \
  --batch_size 8 \
  --hidden_size 128 \
  --num_lstm_layers 2 \
  --num_heads 2 \
  --dropout 0.1 \
  --weight_decay 5e-4 \
  --val_year 2022 \
  --seed 42 \
  --use_constructed \
  --device cuda
