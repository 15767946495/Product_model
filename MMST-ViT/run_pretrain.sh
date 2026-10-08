#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

cd "$ROOT"

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  main_pretrain_mmst_vit.py \
  --root_dir /data/raid0/hqx/Product_model_runtime/DataSrc \
  --data_file ./data/corn_pretrain_2017_2021.json \
  --device cuda \
  --output_dir ./output_dir/pvt_simclr \
  --log_dir ./output_dir/pvt_simclr \
  "$@"
