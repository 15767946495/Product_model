# Task 5 报告

## 状态

已完成。MMST-ViT 清单已按项目八州协议重建，排除 Iowa；官方清单现在可通过 CLI 从三份 `valid-no-ia.*.jsonl` 重建；未训练模型，未修改 Sentinel 原始数据，未删除已有运行产物。

## 变更

- `mmst_vit/manifest.py`
  - 复用项目级八州 allowlist。
  - USDA 样本使用解析后的州值过滤，仅保留 Illinois、Indiana、Kentucky、Michigan、Minnesota、Missouri、Ohio、Wisconsin。
  - CLI 强制输出 `valid-no-ia.jsonl`、`valid-no-ia.train.jsonl`、`valid-no-ia.val.jsonl`、`valid-no-ia.test.jsonl`。
- `mmst_vit/config.py`
  - 短期 HRRR 固定为 4--9 月，共 6 个文件。
  - 长期 HRRR 保持官方形状：一个 60 文件上下文，包含 2017--2021 五年月度文件。
  - Sentinel 固定生成 AG 和 NDVI 两类、4--6 月与 7--9 月两个季度，共 4 个路径；NDVI 文件名按运行目录实际格式使用 `Vegetation_*`。
- `mmst_vit/official.py`
  - 增加逐记录路径收集和写入前文件存在性验证。
  - `write_official_manifests()` 自身校验共享八州 allowlist，非法州抛出 `ValueError`。
  - 增加 CLI：从指定 manifest 目录读取三份 `valid-no-ia.{train,val,test}.jsonl`，生成 `train/val/test.official.no-ia.json`。
  - 三个 split 的所有路径在任何 official JSON 写入前统一验证，避免生成部分结果。
- `mmst_vit/config.py`
  - `official_sample_record()` 严格拒绝未知或不允许州名。
- `tests/test_mmst_manifest_protocol.py`
  - 增加州值过滤、严格州名拒绝、6 个短期 HRRR、60 个长期月度上下文、4 个 Sentinel 精确路径集合、写入前路径验证和 CLI 端到端测试。

## 运行产物

运行目录：`/data/raid0/hqx/Product_model_runtime/DataSrc/mmst_vit/manifests`

- `valid-no-ia.jsonl`：3393 条
- `valid-no-ia.train.jsonl`：2227 条
- `valid-no-ia.val.jsonl`：569 条
- `valid-no-ia.test.jsonl`：597 条
- `train.official.no-ia.json`：2227 条
- `val.official.no-ia.json`：569 条
- `test.official.no-ia.json`：597 条

## 验证

使用解释器 `/root/miniconda3/envs/hqx/bin/python`：

- TDD RED：初次测试因 `ALLOWED_STATES` 和官方写入函数尚不存在而失败；随后修正 fixture 与路径断言，并实现最小功能。
- `python -m pytest tests/test_mmst_manifest_protocol.py -q`：`7 passed`。
- `python -m pytest -q`：`70 passed`，仅有已有的 `pynvml` 弃用警告。
- `python -m py_compile mmst_vit/manifest.py mmst_vit/config.py mmst_vit/official.py`：通过，退出码 0。
- CLI 端到端测试仅使用临时目录和最小 fixture，不读取大运行数据。
- 实际 JSONL 构建：3393 条，其中 train/val/test 为 2227/569/597。
- 实际官方 JSON 构建：train/val/test 为 2227/569/597。
- 最终验收：所有 JSONL 解析后州值均属于八州；所有官方记录均有 6 个短期 HRRR、1 个 60 月文件的长期上下文、4 个 Sentinel 路径；逐条路径存在性验证通过。

## 提交

提交信息：`fix: close MMST-ViT manifest review gaps`

## Concerns

- 运行时旧的 `valid*.jsonl`、`valid-hqx*.jsonl` 和旧官方 JSON 未删除，符合“不删除运行产物”要求。
- 实际生成扫描 HRRR CSV 首次超过默认 120 秒，使用更长超时完成；没有训练模型。
- 运行时 NDVI 目标季度文件的实际前缀为 `Vegetation`，代码按数据实际命名生成，数据类型仍为 NDVI。
- 运行目录中的产物位于外部路径，不纳入 Git 提交；提交仅包含代码、测试和本报告。
