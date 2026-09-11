# 五卡消融并行设计

## 目标

让 `TFT_model/ablation_rope.py --constructed` 的五组消融实验分别占用一张 GPU 并行运行：`mean`、`r0t0`、`r0t1`、`r1t0`、`r1t1`。

## 方案

新增 `--parallel` 参数。主进程保留组合构造、断点跳过、训练后推理和最终汇总职责；每个待运行组合通过一个独立 Python 子进程执行，子进程继承当前环境但使用独立的 `CUDA_VISIBLE_DEVICES`。传入的可见 GPU 按顺序映射到组合列表，例如：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6 \
python ablation_rope.py --constructed --parallel
```

映射为 `mean->2`、`r0t0->3`、`r0t1->4`、`r1t0->5`、`r1t1->6`。

## 行为约束

- `--parallel` 仅接受 `--constructed`，避免旧的 8 组合误用五卡映射。
- 至少需要与有效组合数相同的可见 GPU；不足时立即报错，不启动部分任务。
- 每组继续使用现有独立输出目录、`train.log` 和 `infer.log`，不会共享模型或日志文件。
- 已存在 checkpoint 的组合沿用现有跳过逻辑，不占用 GPU；并行映射仅针对实际待运行组合。
- 子进程训练成功后立即执行该组合推理；某组失败只记录该组失败状态，其余组合继续运行。
- 汇总文件 `ablation_results.json` 只由主进程在全部任务结束后写入，避免并发写冲突。
- 不带 `--parallel` 的现有串行行为保持不变。

## 验证

- 使用 `--help` 验证新参数暴露。
- 使用无效模式和不足 GPU 的模拟环境验证参数校验，不启动训练。
- 使用 `python -m py_compile TFT_model/ablation_rope.py` 验证语法。
- 使用 product 环境确认 `torch==2.11.0+cu128` 且 CUDA 可用。
