# Task 3 报告：五州 AG-only manifest 与真实审计

## 状态

- 已实现 `tools/audit_tft_ag_data.py` CLI、AG-only manifest 生成和真实 HDF5 审计。
- 已增加 `mmst_vit.config.tft_ag_quarter_paths()`，只返回两个 Agriculture 季度路径。
- 已加强 `TFT_model/data.py` 的五州 FIPS/州 ANSI 校验，并兼容 fixture 的 `MM-DD` 与真实数据的 `YYYY-MM-DD` 日期组命名。
- 未修改 `TFT_model/models.py`、训练/推理入口、Sentinel 原始文件或八州 `task8-active` bundle。

## 产物

输出目录：`/data/raid0/hqx/Product_model_runtime/cropnet-five-state/`

- `manifests/train.jsonl`: 355 行
- `manifests/val.jsonl`: 78 行
- `manifests/test.jsonl`: 93 行
- `audit/ag_integrity.json`: 526 个有效样本、6 个无效样本
- `audit/protocol.json`: 五州、2017--2020/2021/2022 split、AG-only、12 日期和两季度协议

所有 manifest 行均包含恰好两个 AG 路径，未包含 NDVI 或 Vegetation 路径。

## 真实审计统计

输入为 task8-active 的共享 `dataset.jsonl`，经五州过滤后共 532 个 Illinois 样本；共享输入中没有 Iowa、Louisiana、Mississippi、New York 行，因此没有伪造或补齐这些州。

- train：355 有效，5 无效
- val：78 有效，1 无效
- test：93 有效，1 无效
- 无效原因：4 个样本缺少目标日期组，2 个样本缺少 FIPS group
- 已对 12 个实际涉及的季度 HDF5 文件记录 SHA256
- AG 日期：`04-01, 04-15, 05-01, 05-15, 06-01, 06-15, 07-01, 07-15, 08-01, 08-15, 09-01, 09-15`

## 验证

- `/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_tft_ag_data.py -q`: 43 passed
- `/root/miniconda3/envs/hqx/bin/python -m pytest -q`: 146 passed
- `/root/miniconda3/envs/hqx/bin/python -m py_compile tools/audit_tft_ag_data.py mmst_vit/config.py mmst_vit/manifest.py TFT_model/data.py tests/test_tft_ag_data.py`: 通过
- 真实 CLI 扫描完成并生成上述 manifest/audit/protocol 文件

## Concerns

- 当前 task8-active 共享数据本身是八州包，五州交集只有 Illinois；本任务严格保留该输入事实，没有从外部构造其他四州样本。
- 真实数据的部分县年缺日期或 FIPS group，已进入独立审计报告并从 manifest 排除。
- runtime 产物不纳入 git；源代码和测试单独提交。
