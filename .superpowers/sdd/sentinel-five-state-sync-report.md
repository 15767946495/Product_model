# Sentinel 五州清理/同步报告

## 状态

- 实现状态：已完成代码、测试和 dry-run 流程。
- 执行模式：仅 dry-run。
- 明确未执行：`--execute`。
- 目标模式：`--sync-cropnet-five-state`。
- 协议：Illinois、Iowa、Louisiana、Mississippi、New York；ANSI `17/19/22/28/36`；年份 2017--2022；模态 AG-only。

## Dry-Run 命令

```bash
python -m mmst_vit.sentinel \
  --sync-cropnet-five-state \
  --usda-dir DataSrc/cropnet_dataset/data/usda_corn \
  --weather-dir DataSrc/cropnet_dataset/data/weather \
  --url-manifest DataSrc/mmst_vit/manifests.task8-legacy/sentinel_urls.jsonl \
  --destination DataSrc/mmst_vit/download \
  --extract-root DataSrc/mmst_vit/county \
  --sync-run-root /tmp/opencode/sentinel-five-state-run \
  --tree-json /tmp/opencode/empty-tree.json
```

使用空的本地 HF tree fixture 进行离线 dry-run，避免在验证阶段下载或上传数据。正式运行时去掉 `--tree-json` 会通过现有 HF API 枚举 AG 源文件；仍然必须先检查计划，再由用户显式添加 `--execute`。

## Dry-Run 输出

运行目录：`/tmp/opencode/sentinel-five-state-run/20260914T075642Z`

- 目标 FIPS 数量：323
- 待删除 weather 州目录：185
- 旧 URL manifest 待删记录：364
- OSS 待删除对象：364
- 待下载 AG 源文件：0
- 待上传县文件：0
- 旧 URL manifest 保留记录：800

USDA 每年保留/删除行数：

| 年份 | 保留 | 删除 |
| --- | ---: | ---: |
| 2017 | 291 | 1176 |
| 2018 | 254 | 1077 |
| 2019 | 247 | 997 |
| 2020 | 302 | 1345 |
| 2021 | 264 | 1192 |
| 2022 | 290 | 1209 |

计划和状态文件：

- `/tmp/opencode/sentinel-five-state-run/20260914T075642Z/plan.json`
- `/tmp/opencode/sentinel-five-state-run/20260914T075642Z/status.json`
- 捕获的标准输出：`/tmp/opencode/sentinel-five-state-dry-run-final.json`

## 实现内容

- 新增 ANSI/FIPS allowlist 和规范化州值解析，不依赖源码路径子串。
- 新增 USDA 行过滤、旧 URL manifest 分类和同步计划纯函数。
- 新增时间戳运行目录、原子文本写入、USDA 备份 SHA256/行数报告。
- 新增 `bucket.delete_object` 删除封装，失败会汇总并抛错，不静默继续。
- 新增 AG-only 五州同步编排：AG 下载/提取/上传先于清理步骤。
- USDA 和 URL manifest 使用临时文件加 `os.replace` 重写。
- 普通 sentinel 下载/上传 CLI 分支保持原有参数和行为。
- 新增纯函数及 dry-run 不删除测试。

## 验证

- `python -m py_compile mmst_vit/sentinel.py tests/test_sentinel_five_state_sync.py`：通过。
- 手工纯函数契约检查：通过，覆盖州过滤、USDA 过滤、URL 分类、OSS 删除失败传播和 dry-run 不修改。
- `git diff --check`：通过。
- `pytest` 专项/全量测试：未能运行，环境中没有 `pytest` 模块。
- `hqx` 专项/全量测试：未能运行，环境中没有 `hqx` 可执行文件（`command -v hqx` 无输出）。

## Concerns

- 本次 dry-run 使用空 HF tree fixture，因此 AG 待下载和县文件待上传统计为 0；这是离线验证结果，不代表线上 HF API 没有 AG 文件。
- 本地 weather 数据约 28G，计划生成采用五州目录存在性与 USDA 样本 FIPS 交集，避免 dry-run 逐行扫描所有州的大文件；execute 前仍应人工审阅 `plan.json`。
- 由于 `pytest`、`hqx` 和本地 h5py 的静态环境不可用，完整测试和真实 HDF5/OSS 执行链未在本环境执行。
- 未执行任何 OSS 删除、weather 删除、USDA 重写、manifest 重写或 AG 下载/上传操作。
