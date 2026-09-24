#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MMST_BATCH_SIZE="${MMST_BATCH_SIZE:-512}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"

cd "$ROOT"

"$PYTHON" -c "import torch, torchvision, timm, einops, h5py, pandas, sklearn" || {
  echo "缺少 MMST-ViT 运行依赖，请执行: $PYTHON -m pip install -r requirements.txt"
  exit 1
}

"$PYTHON" current_data.py \
  --manifest data/pretrain_2017_2020.json \
  --years 2017 2018 2019 2020 \
  --no-long-term \
  --all-states

"$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" pretrain_current.py \
  --manifest data/pretrain_2017_2020.json \
  --output-dir output_current/pretrain \
  --batch-size "$MMST_BATCH_SIZE" \
  "$@"
