# Task 3 实现报告

## 状态

DONE

TFT、TFT 推理、`TFT_model/ablation_rope.py` 和 `ablation/ablation.py` 已接入 version-4 八州/4--9 月/每月前 28 天/168 步协议。未修改模型核心结构，未训练模型，未删除或修改运行时 JSONL/cache。

## 提交哈希

- `cf837ada8702e008cf6b66d9521c04137a588a62`：`feat: align TFT and ablations with cropnet short-season protocol`
- `abc23a2e69f72641a2fbc9eab7a17339473198ce`：`fix: close task 3 protocol validation gaps`

## 修改文件

- `TFT_model/data.py`
- `TFT_model/train.py`
- `TFT_model/infer.py`
- `TFT_model/ablation_rope.py`
- `ablation/ablation.py`
- `tests/test_cropnet_protocol.py`

## 实现内容

- `train.py`、`infer.py` 和 `ablation/ablation.py` 复用项目级八州 allowlist；默认州集合严格为 Minnesota、Wisconsin、Michigan、Illinois、Indiana、Ohio、Missouri、Kentucky 的小写值。
- 训练、验证和推理均按每条序列最后一个有效步取值，不再使用旧的月份阈值选择逻辑。
- `hargreaves_pet()` 的序列 DOY 默认起点改为 4 月 1 日（DOY 91）；GDD/KDD/CumPRCP/CumDeficit 公式未改变。
- 推理默认 cutoff 改为 4--9 月协议内的 12 个节点，并依据缓存中的 `month/day` 字段定位节点。
- `infer.py` 的自定义 cutoff 使用共享协议常量校验，拒绝 4/1--9/28 之外的日期，并对格式错误返回清晰的 `ValueError`。
- `cropnet_protocol.py` 提供低耦合的 grid entry 校验；TFT `load_grid_cache()` 对每个 entry 强制检查 168 步、`l_enc`、month/day 一维形状、长度和值域。
- 网格 Dataset/collate 传递 `day` 字段，以支持协议日期定位；动态序列处理和 `TFTEncoderForYieldPrediction` 接口保持不变。
- `ablation_rope.py` 显式向训练/推理子进程传递协议八州、JSONL 和 grid cache 路径；另一消融入口使用相同八州过滤和已有 version-4 数据默认路径。
- 清理 `TFT_model/train.py` 中旧的“9 州” batch size 帮助文案。
- 没有训练模型，没有删除运行产物；既有未跟踪目录 `DataSrc`、`BaseLine_Model/output` 未纳入提交。

## TDD 测试闭环

先增加以下行为测试并确认旧实现失败：

- TFT/推理/消融入口使用精确八州集合。
- Hargreaves 默认 DOY 起点为 91。
- 训练和推理最后有效序列步索引一致。
- cutoff 按 4 月起始序列的 `month/day` 字段解析。

旧实现结果：`4 failed, 45 passed`，失败均为预期的缺失行为或旧默认值。

实现后测试结果：`60 passed, 1 warning`。

## 测试命令及结果

使用指定解释器 `/root/miniconda3/envs/hqx/bin/python`。

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_cropnet_protocol.py -q
```

结果：`60 passed, 1 warning in 5.52s`。

新增覆盖：`03-31`、`10-01`、`09-29` cutoff 被拒绝，`09-28` 被接受；损坏的 feats 时间维、`l_enc`、month/day 形状长度和值域均被 TFT cache loader 拒绝。

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest -q
```

结果：`60 passed, 1 warning in 8.34s`。

```bash
/root/miniconda3/envs/hqx/bin/python -m py_compile \
  TFT_model/data.py TFT_model/train.py TFT_model/infer.py \
  TFT_model/ablation_rope.py ablation/ablation.py \
  tests/test_cropnet_protocol.py
```

结果：退出码 `0`，无输出。

```bash
git diff --check
```

结果：退出码 `0`，无输出。

## 运行时数据解析审计

既有运行时文件：

- `/data/raid0/hqx/Product_model_runtime/train_dataset/dataset.jsonl`
- `/data/raid0/hqx/Product_model_runtime/train_dataset/grid_cache.pt`

审计命令：

```bash
/root/miniconda3/envs/hqx/bin/python -c 'import json,sys; sys.path.insert(0,"train_dataset"); import prepare_grid; result=prepare_grid.audit_artifacts("/data/raid0/hqx/Product_model_runtime/train_dataset/dataset.jsonl","/data/raid0/hqx/Product_model_runtime/train_dataset/grid_cache.pt"); print(json.dumps({"rows":result["rows"],"states":result["states"],"assertions":result["assertions"],"error_count":result["error_count"]},ensure_ascii=False)); assert result["error_count"] == 0; assert all(result["assertions"].values())'
```

结果：

- `rows=3247`。
- 解析出的州只有八州：`illinois=532`、`indiana=391`、`kentucky=443`、`michigan=329`、`minnesota=410`、`missouri=328`、`ohio=463`、`wisconsin=351`。
- `error_count=0`。
- `states_allowed`、`calendar_valid`、`cache_version`、`cache_time_window`、`cache_max_steps`、`no_none_entries`、`entry_calendar_valid`、`entries_equal_rows`、`entry_l_enc_equal_row`、`entry_shapes_valid` 共 10 项断言全部为 `true`。

## Concerns

- 测试仍出现 PyTorch 的 `pynvml` 弃用警告；不影响测试结果。
- 本任务按要求未训练模型，因此未执行真实训练/推理性能回归。
- `TFT_model/ablation_rope.py` 的 launcher 只做参数接入验证，未启动消融训练。
- 消融配对过滤增加了异常州回归测试，确保先保持 JSONL/cache 对齐再过滤。
- 工作区仍存在任务前已有的未跟踪运行产物 `DataSrc` 和 `BaseLine_Model/output`，未修改、未提交。
