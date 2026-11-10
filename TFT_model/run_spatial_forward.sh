#!/usr/bin/env bash
# 前向滚动起点 × 按县五折空间交叉验证（PA-TFT）。
#
# 对每个测试年 y：训练县只用 y 之前的年份；留出县只在 y 年上测试；
# 全局标准化、县均值偏移、遥感县内中心都只由 y 之前的县年估计。
# 与旧 run_spatial.sh 的区别：旧脚本用 --all-years，统计量会看到测试年及未来年份；
# 这里不传 --all-years，train.py 的 split_years 已改为严格前向。
#
#   bash TFT_model/run_spatial_forward.sh
#   YEARS="2022 2021" FOLDS="0 1" bash TFT_model/run_spatial_forward.sh
#   DRY_RUN=1 bash TFT_model/run_spatial_forward.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
OUT="${OUTPUT_ROOT:-/data/raid0/hqx/TFT_out}"
BASE="${SPATIAL_ROOT:-/data/raid0/hqx/train_dataset/spatial_folds}"
DATA_ROOT="${DATA_ROOT:-/data/raid0/hqx/DataSrc}"
RS="${RS_FEATURES:-/data/raid0/hqx/train_dataset/rs_grid_features_31states.npz}"
FOLDS="${FOLDS:-0 1 2 3 4}"
# 2022 最重要，先发；2018 只有 2017 一年前史，最后发。
YEARS="${YEARS:-2022 2021 2020 2019 2018}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a FOLD_LIST <<< "$FOLDS"
read -r -a YEAR_LIST <<< "$YEARS"

THREADS="${THREADS:-4}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
WORKERS="${WORKERS:-4}"

run_patft() {   # gpu fold year
  local gpu=$1 fold=$2 year=$3
  local id="spatialfwd-improved-fold${fold}-y${year}-s0"
  if [[ -f "$OUT/tft/$id/test_metrics.json" ]]; then echo "skip $id (done)"; return 0; fi
  mkdir -p "$OUT/logs/$id"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/TFT_model/train.py" \
    --dataset-jsonl "$BASE/fold$fold/train.jsonl" --test-jsonl "$BASE/fold$fold/test.jsonl" \
    --offset-jsonl "$ROOT/train_dataset/dataset.jsonl" \
    --data-root "$DATA_ROOT" --output_root "$OUT" --run_id "$id" --device cuda:0 --seed 0 \
    --epochs 20 --num_workers "$WORKERS" --batch_size 4 --hidden_size 512 --num_heads 4 --num_lstm_layers 1 \
    --dropout 0.1 --weight_decay 0.05 --warmup_epochs 5 --use-constructed \
    --rs-features "$RS" --county-offset --demean-rs --grad-clip 1.0 --lr 1e-4 \
    --test-year "$year" --spatial-fold "$fold" >>"$OUT/logs/$id/run.log" 2>&1 \
    || { echo "FAILED $id"; return 0; }
  [[ -f "$OUT/tft/$id/test_metrics.json" ]] && rm -f "$OUT/tft/$id/checkpoint-latest.pth"
}

jobs=()
for year in "${YEAR_LIST[@]}"; do
  for fold in "${FOLD_LIST[@]}"; do
    jobs+=("$fold $year")
  done
done

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  for job in "${jobs[@]}"; do read -r fold year <<< "$job"; echo "fold $fold test=$year"; done
  exit 0
fi

worker() {
  local gpu=$1 index=$2
  for ((i = index; i < ${#jobs[@]}; i += ${#GPUS[@]})); do
    read -r fold year <<< "${jobs[$i]}"
    run_patft "$gpu" "$fold" "$year"
  done
}
for k in "${!GPUS[@]}"; do worker "${GPUS[$k]}" "$k" & done
wait
echo "all forward spatial jobs finished"
