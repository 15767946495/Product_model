#!/usr/bin/env bash
# 空间五折交叉验证：把 741 个县按州内分层分成 5 折，每折留出约 1/5 的县，
# 这些县 2017—2022 年的全部县年都不参与训练；训练集是其余 4/5 县 2017—2022 年的全部县年，
# 测试集是留出县 2017—2022 年的全部县年。
# 标准化统计量在每折内部只用该折训练县池化估计（fit_stats 的全局口径）。
#   bash TFT_model/run_spatial.sh              # 5 折并行，每折一块 GPU
#   DRY_RUN=1 bash TFT_model/run_spatial.sh    # 只打印命令
# 划分文件由 train_dataset/make_spatial_folds.py 生成（州内分层，种子 0）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
OUT="${OUTPUT_ROOT:-/data/raid0/hqx/TFT_out}"
DATA_ROOT="${DATA_ROOT:-/data/raid0/hqx/DataSrc}"
FOLDS="${SPATIAL_FOLDS:-/data/raid0/hqx/train_dataset/spatial_folds}"
RS="${RS_FEATURES:-/data/raid0/hqx/train_dataset/rs_grid_features_31states.npz}"
TAG="${TAG:-spatial}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4}"
K="${K:-5}"
SEED="${SEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

COMMON=(--data-root "$DATA_ROOT" --output_root "$OUT" --device cuda:0 --epochs 20
        --num_workers "${NUM_WORKERS:-4}" --batch_size 4 --hidden_size 512 --num_heads 4
        --num_lstm_layers 1 --dropout 0.1 --warmup_epochs 5 --use-constructed
        --rs-features "$RS" --grad-clip 1.0 --lr 1e-4
        --no-rs-within-norm --no-weather-within-norm --seed "$SEED" --all-years)

run_fold() {   # gpu fold
  local gpu=$1 fold=$2
  local id="${TAG}-main-f${fold}-allyears-s${SEED}"
  if [[ -f "$OUT/tft/$id/test_metrics.json" ]]; then echo "skip $id (done)"; return 0; fi
  mkdir -p "$OUT/logs/$id"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/TFT_model/train.py" "${COMMON[@]}" \
    --dataset-jsonl "$FOLDS/fold$fold/train.jsonl" \
    --test-jsonl "$FOLDS/fold$fold/test.jsonl" \
    --spatial-fold "$fold" --run_id "$id" >>"$OUT/logs/$id/run.log" 2>&1 || echo "FAILED $id"
  if [[ -f "$OUT/tft/$id/test_metrics.json" ]]; then rm -f "$OUT/tft/$id/checkpoint-latest.pth"; fi
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  for ((fold = 0; fold < K; fold++)); do
    echo "fold $fold: ${COMMON[*]} --dataset-jsonl $FOLDS/fold$fold/train.jsonl --test-jsonl $FOLDS/fold$fold/test.jsonl --spatial-fold $fold"
  done
  exit 0
fi

for ((fold = 0; fold < K; fold++)); do
  run_fold "${GPUS[$((fold % ${#GPUS[@]}))]}" "$fold" &
done
wait
echo "all spatial folds finished"
