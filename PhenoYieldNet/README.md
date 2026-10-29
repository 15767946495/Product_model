# PhenoYieldNet-unified

本目录把官方 PhenoYieldNet 接入统一 CropNet 数据，只做 corn（PhenoYieldNet-SC）。来源、许可证（CC BY-NC 4.0）、本地修改和 SpectralGPT+ 加载报告见 [UPSTREAM.md](UPSTREAM.md)。

## 协议

| 阶段 | 年份 | 标签 | 入口 |
| --- | --- | --- | --- |
| Stage 1 对比适配 | 2017–2020 | 不使用 | `train_da.py`，用 SpectralGPT+ 初始化 |
| Stage 2 监督微调 | 2021 | 2021 玉米单产 | `train_ft.py`，加载 Stage 1 的 `checkpoint-final.pth` |
| 最终测试 | 2022 | 只用于指标 | `train_ft.py` 训练结束时自动执行，或单独跑 `infer.py` |

不使用长期气象，不设验证集，训练轮数固定，只评估 `checkpoint-final.pth`。

## 运行

```bash
./PhenoYieldNet/run_train.sh
```

脚本默认 `EPOCHS=50`、`GPU 0`，输出根为 `PhenoYieldNet/output_dir`，并自动执行 Stage 1、Stage 2 和 2022 最终测试。Stage 1 的 `checkpoint-final.pth` 自动作为 Stage 2 输入。真实执行时，三个阶段日志分别写入 `PhenoYieldNet/output_dir/logs/<run_id>/stage1.log`、`stage2.log` 和 `infer.log`；`DRY_RUN=1` 只打印命令且不创建日志目录。常用覆盖为 `EPOCHS`、`STAGE1_EPOCHS`、`STAGE2_EPOCHS`、`RUN_ID`、`OUTPUT_ROOT`、`NUM_WORKERS`、`GRAD_CHECKPOINTING`、`FROZEN_FM`、`DRY_RUN`；例如 `DRY_RUN=1 EPOCHS=1 RUN_ID=smoke ./PhenoYieldNet/run_train.sh`。

SpectralGPT+ 默认路径为 `/data/raid0/hqx/pretrained/SpectralGPT+.pth`，加载前会校验 SHA-256。Stage 1 只输出训练文件，没有测试文件；Stage 2 输出到 `PhenoYieldNet/output_dir/phenoyieldnet/<run_id>/`。
