# TFT

TFT 统一 CropNet 训练入口使用 `./TFT_model/run_train.sh`，固定 GPU 4，输出到 `TFT_model/output_dir`。

论文的批量训练入口是 `./TFT_model/run_loyo.sh`：默认只跑 2022 折、种子 0，同时训练 PA-TFT（`improved`）与三个组件消融（`noffset`、`nodemean`、`nors`）；`CONFIGS=original SEEDS="42 1 2"` 则跑对照配置 PVT-TFT。输出根目录默认外置为 `/data/raid0/hqx/TFT_out`，run 子目录名为 `${TAG}-${配置}-y${年份}-s${种子}`，汇总脚本按这个命名读取。两种配置的差异见脚本内 `CFG`：PA-TFT 走数值植被指数、县均值偏移与县内去均值、学习率 1e-4；PVT-TFT 走端到端 PVT 编码器、学习率 1e-3。

默认训练配置为 `EPOCHS=20`、`BATCH_SIZE=4`。可通过 `EPOCHS`、`BATCH_SIZE`、`HIDDEN_SIZE`、`NUM_HEADS`、`NUM_LSTM_LAYERS`、`WARMUP_EPOCHS`、`CUTOFFS` 和 `DRY_RUN` 覆盖。

遥感编码只对 `ag_grid_mask` 与 `grid_mask` 同时有效的月份-网格图像运行 PVT，并将编码结果散射回 `(B, G, 6, H)`；无效位置保持严格为零。

预测头不使用 Dropout，避免训练态与评估态之间产生预测漂移；其余 TFT 模块默认使用 `dropout=0.1`。修复前训练的 checkpoint 仍需重新训练。
