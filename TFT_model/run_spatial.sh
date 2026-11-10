#!/usr/bin/env bash
# 按县五折空间交叉验证的批量训练入口：PA-TFT、UNet-ConvLSTM、GNN-RNN、PhenoYieldNet。
# 每折留出 1/5 的县作测试（全部年份），其余县的同一批年份作训练。
#   bash TFT_model/run_spatial.sh              # 5 折 × 4 个模型，按 GPU 排队
#   FOLDS="0 1" bash TFT_model/run_spatial.sh  # 只跑指定折
#   DRY_RUN=1 bash TFT_model/run_spatial.sh    # 只打印任务清单
# 输出（务必外置）：$OUTPUT_ROOT/tft、$OUTPUT_ROOT/spatial_baselines，日志在 $OUTPUT_ROOT/logs。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/root/miniconda3/envs/product/bin/python}"
OUT="${OUTPUT_ROOT:-/data/raid0/hqx/TFT_out}"
BASE="${SPATIAL_ROOT:-/data/raid0/hqx/train_dataset/spatial_folds}"
DATA_ROOT="${DATA_ROOT:-/data/raid0/hqx/DataSrc}"
RS="${RS_FEATURES:-/data/raid0/hqx/train_dataset/rs_grid_features_31states.npz}"
FOLDS="${FOLDS:-0 1 2 3 4}"
read -r -a GPUS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a FOLD_LIST <<< "$FOLDS"

# 多卡并行时限制每个进程的 CPU 线程数：默认每个进程吃满 224 核会把整机拖成
# 超订状态（load average 三位数），数据加载与 HDF5 读取反而更慢。
THREADS="${THREADS:-4}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"

# PhenoYieldNet 每步只处理一个县年、要读 6 年 224×224 影像，是这批任务里最慢的一个
# （约 26 分钟/epoch）。实测把 worker 从 4 提到 12 对每轮耗时没有影响，限速不在
# worker 数，因此提高 worker 数只用于安全余量，不要指望它加速。
WORKERS="${WORKERS:-4}"
PHENO_WORKERS="${PHENO_WORKERS:-12}"

ALL_YEARS=2017,2018,2019,2020,2021,2022

run_patft() {   # gpu fold
  local gpu=$1 fold=$2 id="spatial-improved-fold$2-s0"
  if [[ -f "$OUT/tft/$id/test_metrics.json" ]]; then echo "skip $id"; return 0; fi
  mkdir -p "$OUT/logs/$id"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/TFT_model/train.py" \
    --dataset-jsonl "$BASE/fold$fold/train.jsonl" --test-jsonl "$BASE/fold$fold/test.jsonl" \
    --offset-jsonl "$ROOT/train_dataset/dataset.jsonl" \
    --data-root "$DATA_ROOT" --output_root "$OUT" --run_id "$id" --device cuda:0 --seed 0 \
    --epochs 20 --num_workers "$WORKERS" --batch_size 4 --hidden_size 512 --num_heads 4 --num_lstm_layers 1 \
    --dropout 0.1 --weight_decay 0.05 --warmup_epochs 5 --use-constructed \
    --rs-features "$RS" --county-offset --demean-rs --grad-clip 1.0 --lr 1e-4 \
    --all-years --spatial-fold "$fold" >>"$OUT/logs/$id/run.log" 2>&1 || echo "FAILED $id"
  [[ -f "$OUT/tft/$id/test_metrics.json" ]] && rm -f "$OUT/tft/$id/checkpoint-latest.pth"
}

run_patft_noffset() {   # gpu fold
  # 冷启动复核：与 run_patft 相同的空间折与超参数，但不给县均值偏移（消融中的 noffset）。
  # 模型必须直接由特征预测绝对水平，留出县的标签与历史单产都不参与；遥感仍按县内去均值
  # （只用到该县自己的影像历史，不含单产标签）。
  local gpu=$1 fold=$2 id="spatial-noffset-fold$2-s0"
  if [[ -f "$OUT/tft/$id/test_metrics.json" ]]; then echo "skip $id"; return 0; fi
  mkdir -p "$OUT/logs/$id"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/TFT_model/train.py" \
    --dataset-jsonl "$BASE/fold$fold/train.jsonl" --test-jsonl "$BASE/fold$fold/test.jsonl" \
    --offset-jsonl "$ROOT/train_dataset/dataset.jsonl" \
    --data-root "$DATA_ROOT" --output_root "$OUT" --run_id "$id" --device cuda:0 --seed 0 \
    --epochs 20 --num_workers "$WORKERS" --batch_size 4 --hidden_size 512 --num_heads 4 --num_lstm_layers 1 \
    --dropout 0.1 --weight_decay 0.05 --warmup_epochs 5 --use-constructed \
    --rs-features "$RS" --demean-rs --grad-clip 1.0 --lr 1e-4 \
    --all-years --spatial-fold "$fold" >>"$OUT/logs/$id/run.log" 2>&1 || echo "FAILED $id"
  [[ -f "$OUT/tft/$id/test_metrics.json" ]] && rm -f "$OUT/tft/$id/checkpoint-latest.pth"
}

run_unet() {    # gpu fold
  local gpu=$1 fold=$2 id="spatial-unet-fold$2"
  if [[ -f "$OUT/spatial_baselines/unet_convlstm/$id/test_metrics.json" ]]; then echo "skip $id"; return 0; fi
  mkdir -p "$OUT/logs/$id"
  CUDA_VISIBLE_DEVICES=$gpu UNET_TRAIN_YEARS=$ALL_YEARS UNET_TEST_YEARS=$ALL_YEARS \
    "$PY" "$ROOT/UNet-ConvLSTM/train.py" \
    --dataset-jsonl "$BASE/fold$fold/train.jsonl" --eval_jsonl "$BASE/fold$fold/test.jsonl" \
    --data-root "$DATA_ROOT" --output_root "$OUT/spatial_baselines" --run_id "$id" --device cuda:0 \
    --epochs 50 --num_workers "$WORKERS" --county_batch_size 1 --image_size 128 --grid_chunk 128 --warmup_epochs 2 \
    >>"$OUT/logs/$id/run.log" 2>&1 || echo "FAILED $id"
  [[ -f "$OUT/spatial_baselines/unet_convlstm/$id/test_metrics.json" ]] && \
    rm -f "$OUT/spatial_baselines/unet_convlstm/$id/checkpoint-latest.pth"
}

run_gnn() {     # gpu fold
  local gpu=$1 fold=$2 id="spatial-gnn-fold$2"
  if [[ -f "$OUT/spatial_baselines/gnn_rnn/$id/test_metrics.json" ]]; then echo "skip $id"; return 0; fi
  mkdir -p "$OUT/logs/$id"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/GNN-RNN/train.py" \
    --dataset-jsonl "$ROOT/train_dataset/dataset.jsonl" --data-root "$DATA_ROOT" \
    --output_root "$OUT/spatial_baselines" --run_id "$id" --device cuda:0 --epochs 50 --num_workers "$WORKERS" \
    --aggregator mean --fold-manifest "$BASE/folds.json" --county-fold "$fold" \
    >>"$OUT/logs/$id/run.log" 2>&1 || echo "FAILED $id"
}

run_pheno() {   # gpu fold
  local gpu=$1 fold=$2 id="spatial-pheno-fold$2"
  local out="$OUT/spatial_baselines"
  if [[ -f "$out/phenoyieldnet/$id/test_metrics.json" ]]; then echo "skip $id"; return 0; fi
  mkdir -p "$OUT/logs/$id"
  # Stage 1 无标签表示适配（2017--2020）、Stage 2 监督微调（2021）、测试 2022：
  # 只把县换成"训练县"，年份协议与第5.1节完全相同。
  if [[ ! -f "$out/phenoyieldnet_stage1/$id-stage1/checkpoint-final.pth" ]]; then
    CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/PhenoYieldNet/train_da.py" \
      --dataset-jsonl "$BASE/fold$fold/train.jsonl" --data-root "$DATA_ROOT" --output_root "$out" \
      --run_id "$id-stage1" --device cuda:0 --epochs 10 --num_workers "$PHENO_WORKERS" \
      >>"$OUT/logs/$id/stage1.log" 2>&1 || { echo "FAILED $id stage1"; return 0; }
  fi
  CUDA_VISIBLE_DEVICES=$gpu PHENO_TEST_YEARS=2022 "$PY" "$ROOT/PhenoYieldNet/train_ft.py" \
    --dataset-jsonl "$BASE/fold$fold/train.jsonl" --eval_jsonl "$BASE/fold$fold/test.jsonl" \
    --data-root "$DATA_ROOT" --output_root "$out" \
    --run_id "$id" --device cuda:0 --epochs 20 --num_workers "$PHENO_WORKERS" \
    --pretrain_model "$out/phenoyieldnet_stage1/$id-stage1/checkpoint-final.pth" \
    >>"$OUT/logs/$id/stage2.log" 2>&1 || { echo "FAILED $id stage2"; return 0; }
  [[ -f "$out/phenoyieldnet/$id/test_metrics.json" ]] && rm -f "$out/phenoyieldnet/$id/checkpoint-latest.pth"
  CUDA_VISIBLE_DEVICES=$gpu PHENO_TEST_YEARS=2022 "$PY" "$ROOT/PhenoYieldNet/infer.py" \
    --checkpoint "$out/phenoyieldnet/$id/checkpoint-final.pth" --dataset-jsonl "$BASE/fold$fold/train.jsonl" \
    --eval_jsonl "$BASE/fold$fold/test.jsonl" --data-root "$DATA_ROOT" --device cuda:0 --num_workers "$PHENO_WORKERS" \
    >>"$OUT/logs/$id/infer.log" 2>&1 || echo "FAILED $id infer"
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  for fold in "${FOLD_LIST[@]}"; do
    echo "fold $fold: PA-TFT / UNet-ConvLSTM / GNN-RNN / PhenoYieldNet"
  done
  exit 0
fi

# 长任务先发：UNet（约5--6小时）> Pheno > PA-TFT > GNN
jobs=()
run_kinds() {   # 按给定顺序追加任务；KINDS 可只跑其中几类
  local kind
  for kind in "$@"; do
    case " ${KINDS:-unet pheno patft patft_noffset gnn} " in
      *" $kind "*) for fold in "${FOLD_LIST[@]}"; do jobs+=("$kind $fold"); done ;;
    esac
  done
}
run_kinds unet pheno patft patft_noffset gnn

worker() {
  local gpu=$1 index=$2
  for ((i = index; i < ${#jobs[@]}; i += ${#GPUS[@]})); do
    read -r kind fold <<< "${jobs[$i]}"
    case "$kind" in
      patft) run_patft "$gpu" "$fold" ;;
      patft_noffset) run_patft_noffset "$gpu" "$fold" ;;
      unet)  run_unet  "$gpu" "$fold" ;;
      gnn)   run_gnn   "$gpu" "$fold" ;;
      pheno) run_pheno "$gpu" "$fold" ;;
    esac
  done
}
for k in "${!GPUS[@]}"; do worker "${GPUS[$k]}" "$k" & done
wait
echo "all spatial CV jobs finished"
