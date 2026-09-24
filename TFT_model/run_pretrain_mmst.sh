#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
MMST_ROOT="${ROOT}/../MMST-ViT"

cd "$ROOT"
"$PYTHON" pretrain_mmst.py \
  --manifest "${MMST_ROOT}/data/pretrain_2017_2020.json" \
  --output "${MMST_ROOT}/output_current/tft_pretrain.pth" \
  "$@"
