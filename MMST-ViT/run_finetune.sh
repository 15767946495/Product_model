#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

cd "$ROOT"

"$PYTHON" -c "import torch, torchvision, timm, einops, h5py, pandas, sklearn" || {
  echo "缺少 MMST-ViT 运行依赖，请执行: $PYTHON -m pip install -r requirements.txt"
  exit 1
}

if [[ ! -f data/finetune_2021.json ]]; then
  "$PYTHON" current_data.py --manifest data/finetune_2021.json --years 2021
fi
if [[ ! -f data/eval_2022.json ]]; then
  "$PYTHON" current_data.py --manifest data/eval_2022.json --years 2022
fi

"$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" finetune_current.py \
  --train-manifest data/finetune_2021.json \
  --eval-manifest data/eval_2022.json \
  --pretrained output_current/pretrain/checkpoint-latest.pth \
  --output-dir output_current/finetune \
  "$@"
