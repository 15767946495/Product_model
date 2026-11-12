# TFT

TFT 统一 CropNet 训练入口使用 `./TFT_model/run_train.sh`，固定 GPU 4，输出到 `TFT_model/output_dir`。

论文的批量训练入口是 `./TFT_model/run_loyo.sh`：默认只跑 2022 折、种子 0，训练 PA-TFT 主模型（`main`）与遥感消融（`nors` 去掉遥感输入）。县分组五折空间交叉验证见 `./TFT_model/run_spatial.sh`，汇总用 `spatial_summary.py`。输出根目录默认外置为 `/data/raid0/hqx/TFT_out`，run 子目录名为 `${TAG}-${配置}-y${年份}-s${种子}`，汇总脚本按这个命名读取；空间五折的 run 子目录名为 `${TAG}-main-f${折}-allyears-s${种子}`，每折的标准化统计量只用该折训练县的 JSONL 估计。主模型直接预测对数单产水平，气象、土壤与遥感都按训练年份的全局统计量标准化，学习率 1e-4，不使用县均值偏移。

默认训练配置为 `EPOCHS=20`、`BATCH_SIZE=4`。可通过 `EPOCHS`、`BATCH_SIZE`、`HIDDEN_SIZE`、`NUM_HEADS`、`NUM_LSTM_LAYERS`、`WARMUP_EPOCHS`、`CUTOFFS` 和 `DRY_RUN` 覆盖。

遥感编码只对 `ag_grid_mask` 与 `grid_mask` 同时有效的月份-网格图像运行 PVT，并将编码结果散射回 `(B, G, 6, H)`；无效位置保持严格为零。

预测头不使用 Dropout，避免训练态与评估态之间产生预测漂移；其余 TFT 模块默认使用 `dropout=0.1`。修复前训练的 checkpoint 仍需重新训练。
