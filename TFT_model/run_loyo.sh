#!/usr/bin/env bash
# 论文的批量训练入口：配置 × 测试年份 × 种子的笛卡尔积，任务按 GPU 排队并行。
# 论文只报告 2022 这一折、种子 0 的单次运行（第5.1节），所以默认 YEARS=2022、SEEDS=0：
#   bash TFT_model/run_loyo.sh                       # PA-TFT + 三个组件消融，2022 折，种子 0
#   CONFIGS=original SEEDS="42 1 2" bash TFT_model/run_loyo.sh   # 对照配置 PVT-TFT 的三个种子
#   DRY_RUN=1 bash TFT_model/run_loyo.sh             # 只打印命令
# 输出根目录务必外置（默认 /data/raid0/hqx/TFT_out），run 子目录名为 ${TAG}-${配置}-y${年份}-s${种子}。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
OUT="${OUTPUT_ROOT:-/data/raid0/hqx/TFT_out}"
DATA_ROOT="${DATA_ROOT:-/data/raid0/hqx/DataSrc}"
RS="${RS_FEATURES:-/data/raid0/hqx/train_dataset/rs_grid_features_31states.npz}"
RS_ZERO="${RS_ZERO_FEATURES:-/data/raid0/hqx/train_dataset/rs_grid_features_zero_31states.npz}"
PVT="${PVT_PRETRAINED_PATH:-/data/raid0/hqx/pretrained/pvt_tiny.pth}"
TAG="${TAG:-loyo31}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a CONFIGS <<< "${CONFIGS:-improved noffset nodemean nors}"
read -r -a YEARS <<< "${YEARS:-2022}"
read -r -a SEEDS <<< "${SEEDS:-0}"

COMMON=(--dataset-jsonl "$ROOT/train_dataset/dataset.jsonl" --data-root "$DATA_ROOT" --output_root "$OUT"
        --device cuda:0 --epochs 20 --num_workers "${NUM_WORKERS:-4}" --batch_size 4 --hidden_size 512 --num_heads 4
        --num_lstm_layers 1 --dropout 0.1 --warmup_epochs 5 --use-constructed)
# 配置名 -> 额外参数。学习率按配置给出：PA-TFT 及其消融用 1e-4（第4.11节），
# 对照配置 PVT-TFT 沿用 TFT 原文的 1e-3。
declare -A CFG=(
  [improved]="--rs-features $RS --county-offset --demean-rs --grad-clip 1.0 --lr 1e-4"
  [original]="--vit-pretrained --pvt_pretrained_path $PVT --lr 1e-3"
  # 消融：每次去掉改进模型的一个组件，其余与 improved 相同
  [noffset]="--rs-features $RS --demean-rs --grad-clip 1.0 --lr 1e-4"
  [nodemean]="--rs-features $RS --county-offset --grad-clip 1.0 --lr 1e-4"
  # 消融：遥感输入全部置零（模型结构不变，遥感分支只剩常数偏置，不含任何影像信息）
  [nors]="--rs-features $RS_ZERO --county-offset --demean-rs --grad-clip 1.0 --lr 1e-4"
)

jobs=()
for cfg in "${CONFIGS[@]}"; do
  for year in "${YEARS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      jobs+=("$cfg $year $seed")
    done
  done
done

run_job() {   # gpu cfg year seed
  local gpu=$1 cfg=$2 year=$3 seed=$4
  local id="${TAG}-${cfg}-y${year}-s${seed}"
  if [[ -f "$OUT/tft/$id/test_metrics.json" ]]; then echo "skip $id (done)"; return 0; fi
  mkdir -p "$OUT/logs/$id"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/TFT_model/train.py" "${COMMON[@]}" --run_id "$id" --seed "$seed" \
    --test-year "$year" ${CFG[$cfg]} >>"$OUT/logs/$id/run.log" 2>&1 || echo "FAILED $id"
  # 训练完成后 checkpoint-latest.pth 与 checkpoint-final.pth 内容相同，删除重复副本以节省磁盘
  if [[ -f "$OUT/tft/$id/test_metrics.json" ]]; then rm -f "$OUT/tft/$id/checkpoint-latest.pth"; fi
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  for job in "${jobs[@]}"; do read -r cfg year seed <<< "$job"; echo "$cfg test=$year seed=$seed: ${CFG[$cfg]}"; done
  exit 0
fi

# 每块 GPU 一个工作进程，按顺序领取任务
worker() {
  local gpu=$1 index=$2
  for ((i = index; i < ${#jobs[@]}; i += ${#GPUS[@]})); do
    read -r cfg year seed <<< "${jobs[$i]}"
    run_job "$gpu" "$cfg" "$year" "$seed"
  done
}
for k in "${!GPUS[@]}"; do worker "${GPUS[$k]}" "$k" & done
wait
echo "all LOYO jobs finished"
