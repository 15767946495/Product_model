# Task 9 报告：Task 4 data-only 审查修复

## 状态

- 状态：通过。
- 五州 `protocol.json` 现由 `tools/audit_tft_ag_data.py` 自动完整生成，不再依赖手工追加字段。
- 测试和真实 AG 读取流程不导入或构造 TFT 模型，不调用 TFT `forward`。
- 未修改生产模型、训练入口或推理入口；未训练、未推理。
- 未修改 Sentinel 原始 HDF5 或八州 `task8-active` bundle。

## 协议与真实覆盖

五州协议范围：

- Illinois
- Iowa
- Louisiana
- Mississippi
- New York

当前共享输入经五州过滤后的真实覆盖：

| 州 | 输入 | 有效 | 无效 |
| --- | ---: | ---: | ---: |
| Illinois | 532 | 526 | 6 |
| Iowa | 0 | 0 | 0 |
| Louisiana | 0 | 0 | 0 |
| Mississippi | 0 | 0 | 0 |
| New York | 0 | 0 | 0 |

协议范围是五州，但当前真实样本覆盖仅 Illinois；其余四州明确为 0，没有伪造或从八州 runtime 补齐样本。

固定年份 split 及有效 manifest：

- train，2017--2020：359 输入，355 有效，4 无效。
- val，2021：79 输入，78 有效，1 无效。
- test，2022：94 输入，93 有效，1 无效。
- 总计：532 输入，526 有效，6 无效。
- 无效原因：4 个样本缺少目标日期组，2 个样本缺少 FIPS HDF5 group。

## Data-only 边界

- `tests/test_tft_ag_data.py` 不再顶层导入 `models`。
- 数据集独立性测试向 `sys.modules` 注入构造和 `forward` 必抛错的 `TFTEncoderForYieldPrediction` 哨兵，同时阻断任何 `models` 导入。
- 真实 AG 读取测试删除已有 `models` 缓存并阻断模型模块导入，读取完成后确认 `models` 仍未导入。
- `TFTEncoderForYieldPrediction.forward` 的参数签名通过 AST 静态读取 `TFT_model/models.py` 验证，不执行该模块。
- `TFT_model/models.py`、`TFT_model/train.py`、`TFT_model/infer.py` 本次无 diff。

## 真实样本

代表样本来自重新生成的 train manifest 首行：

- FIPS：`17001`
- Year：`2017`
- State：`illinois`
- County：`adams`
- 12 日期：`04-01, 04-15, 05-01, 05-15, 06-01, 06-15, 07-01, 07-15, 08-01, 08-15, 09-01, 09-15`
- tensor shape：`(12, 24, 3, 224, 224)`
- dtype：`torch.float32`
- grid_count：`24`
- seed：`0`
- transform：eval `CenterCrop(224)` 加 `Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])`
- metadata：`FIPS`、`Year`、`State`、`County`、`grid_count`、`ag_dates`
- 排除输出：`weather`、`label`

测试显式锁定当前真实样本完整 shape、dtype 和 grid_count，同时断言 shape 的网格维与 `item["grid_count"]` 一致。

## 自动审计产物

已通过 CLI 重新生成：

`/data/raid0/hqx/Product_model_runtime/cropnet-five-state/audit/protocol.json`

生成命令：

```bash
/root/miniconda3/envs/hqx/bin/python tools/audit_tft_ag_data.py \
  --shared-jsonl /data/raid0/hqx/Product_model_runtime/task8-active/train_dataset/dataset.jsonl \
  --ag-root "/data/raid0/hqx/Product_model_runtime/DataSrc/mmst_vit/download/Sentinel-2 Imagery" \
  --output-dir /data/raid0/hqx/Product_model_runtime/cropnet-five-state
```

自动生成字段包括：

- `model_boundary`
- `training_run=false`
- `inference_run=false`
- `tft_model_imported=false`
- `tft_forward_called=false`
- `source_sha256`
- 五州逐州 `input_count/valid_count/invalid_count`
- 代表样本 12 日期、完整 shape、dtype、grid_count、seed 和 transform
- split 与 manifest 统计

`source_sha256` 共 16 项，包括 12 个实际审计 AG 季度 HDF5 和 `TFT_model/models.py`、`train.py`、`infer.py`、`data.py`。源文件均只读。

## 验证

- hqx 专项：`/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_tft_ag_data.py -q`
- 结果：`64 passed`，1 条环境层 `pynvml` 弃用警告。
- 全量：`/root/miniconda3/envs/hqx/bin/python -m pytest -q`
- 结果：`167 passed`，1 条环境层 `pynvml` 弃用警告。
- 编译：`/root/miniconda3/envs/hqx/bin/python -m py_compile cropnet_protocol.py TFT_model/data.py tools/audit_tft_ag_data.py mmst_vit/config.py mmst_vit/manifest.py tests/test_tft_ag_data.py`
- 结果：通过。

## Concerns

- 当前数据不能证明 Iowa、Louisiana、Mississippi、New York 的真实样本可用性，因为共享输入对这四州的覆盖均为 0。
- `protocol.json` 中的未训练、未推理和文件未改标记描述本次受控审计流程；代码测试另外验证数据读取不导入模型模块。
- 工作树原有 `_txt/*` 删除以及未跟踪的 `BaseLine_Model/output`、`DataSrc` 未被修改或纳入本任务提交。
