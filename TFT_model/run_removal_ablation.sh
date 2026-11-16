#!/usr/bin/env bash
# 日粒度主模型下的消融与单组件消融重跑（2022 折、种子 0）。
# 主模型见 run_loyo.sh 的 main；本脚本只跑三个对照配置，每个配置一块 GPU。
#   bash TFT_model/run_removal_ablation.sh
#   DRY_RUN=1 bash TFT_model/run_removal_ablation.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
OUT="${OUTPUT_ROOT:-/data/raid0/hqx/TFT_out}"
DATA_ROOT="${DATA_ROOT:-/data/raid0/hqx/DataSrc}"
RS="${RS_FEATURES:-/data/raid0/hqx/train_dataset/rs_grid_features_31states.npz}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# 统一配方（第5.1节）：学习率 1e-5、权重衰减 0.05、10 轮 warmup + 余弦、最多 100 轮；
# 与主模型一样，从 2017—2021 训练县年中随机留出 10% 做早停。
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-5}"
WD="${WD:-0.05}"
VALID_FRAC="${VALID_FRAC:-0.1}"
COMMON=(--dataset-jsonl "$ROOT/train_dataset/dataset.jsonl" --data-root "$DATA_ROOT"
        --output_root "$OUT" --device cuda:0 --epochs "$EPOCHS" --num_workers "${NUM_WORKERS:-4}"
        --batch_size 4 --hidden_size 512 --num_heads 4 --num_lstm_layers 1 --dropout 0.1
        --warmup_epochs 10 --seq-level day --test-year 2022 --seed 0 --rs-features "$RS"
        --grad-clip 1.0 --lr "$LR" --weight_decay "$WD" --no-rs-within-norm --no-weather-within-norm
        --valid-frac "$VALID_FRAC" --early-stop-patience 10 --checkpoint-policy best)
# 主模型为 12 个动态变量（11 个原始气象 + CumGDD，默认）；
# USE_CONSTRUCTED=1 加入全部四个构造变量，=none 一个都不加（去掉 CumGDD 的对照）。
USE_CONSTRUCTED="${USE_CONSTRUCTED:-0}"
case "$USE_CONSTRUCTED" in
  1) COMMON+=(--use-constructed) ;;
  0) COMMON+=(--no-use-constructed) ;;
  none) COMMON+=(--no-constructed-weather) ;;
  *) echo "USE_CONSTRUCTED 只取 1/0/none" >&2; exit 2 ;;
esac

declare -A CFG=(
  [ablate-spatialmean-y2022-s0]="--spatial-pool mean"
  [ablate-norsbranch-y2022-s0]="--no-rs-branch"
  [ablate-nosoilbranch-y2022-s0]="--no-soil-branch"
)
ORDER=(ablate-spatialmean-y2022-s0 ablate-norsbranch-y2022-s0 ablate-nosoilbranch-y2022-s0)

read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3}"

run_job() {   # gpu id
  local gpu=$1 id=$2
  if [[ -f "$OUT/tft/$id/test_metrics.json" ]]; then echo "skip $id (done)"; return 0; fi
  mkdir -p "$OUT/logs/$id"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/TFT_model/train.py" "${COMMON[@]}" \
    --run_id "$id" ${CFG[$id]} >>"$OUT/logs/$id/run.log" 2>&1 || echo "FAILED $id"
  [[ -f "$OUT/tft/$id/test_metrics.json" ]] && rm -f "$OUT/tft/$id/checkpoint-latest.pth"
  return 0
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  for i in "${!ORDER[@]}"; do echo "gpu ${GPUS[$((i % ${#GPUS[@]}))]}: ${ORDER[$i]} ${CFG[${ORDER[$i]}]}"; done
  exit 0
fi

for i in "${!ORDER[@]}"; do
  run_job "${GPUS[$((i % ${#GPUS[@]}))]}" "${ORDER[$i]}" &
done
wait
echo "all ablation jobs finished"
