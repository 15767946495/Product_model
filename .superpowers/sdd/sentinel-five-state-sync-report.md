# Sentinel 五州清理/同步报告

## 状态

- 修复状态：已完成代码、mock/临时目录测试和 dry-run 验证。
- 执行模式：仅 dry-run 和测试隔离的 mock execute 流程。
- 明确未执行：CLI `--execute`；未访问真实 OSS；未修改真实 USDA、weather、Sentinel 原始文件或真实 manifest。
- 目标模式：`--sync-cropnet-five-state`。
- 协议：Illinois、Iowa、Louisiana、Mississippi、New York；ANSI `17/19/22/28/36`；年份 2017--2022；模态 AG-only。

## 修复内容

- USDA 只保留 `commodity_desc=CORN`、`reference_period_desc=YEAR`、五州有效县级 FIPS，且执行重写进一步限制为 USDA 与完整 weather 协议日期交集。
- USDA 重写改用 `csv.DictReader`/`csv.DictWriter`，保留带逗号字段和引号；写入临时内容后验证字段名和行数，再通过 `os.replace` 原子替换。
- USDA 重写前继续创建带 SHA256 和行数的备份。
- 目标 FIPS 改为 USDA 与实际 weather CSV 内容交集；只统计 `Daily` 行，并要求每个 `(FIPS, 年份)` 包含 4 月 1 日至 9 月 28 日的完整 168 个协议日期。
- AG 计划只列出本地源文件不存在或文件大小不匹配的条目。
- OSS 删除失败使用 `OssDeletionError` 携带成功/失败明细；执行 status 先记录失败和未执行的 weather、USDA、URL 清理步骤，再抛出非零错误。
- AG 下载、县提取、上传顺序和原始 Sentinel 源文件保护加入 mock 流程测试。
- CLI 默认 dry-run 加入临时目录回归测试；dry-run 不创建 OSS bucket、不删除 weather、不重写 USDA 或 manifest。

## 测试覆盖

- USDA 五州/CORN/YEAR/FIPS 过滤和错误测试 oracle 修正。
- weather CSV 的 `Daily`、FIPS、月份/日期解析和完整 168 日期校验。
- 已存在且大小正确的 AG 源文件不列入待下载；大小不匹配时列入待下载。
- CSV roundtrip 验证包含带逗号字段和带引号县名。
- USDA 备份、SHA256/行数报告和原子重写。
- AG 下载 -> 县提取 -> OSS 上传顺序。
- execute mock 流程中不删除 Sentinel 原始源文件。
- OSS 删除部分成功/失败传播、status 持久化和后续清理不执行。
- 输出 manifest 只包含 AG/五州记录。
- CLI 新模式默认 dry-run。

## 验证

使用指定 hqx Python 环境：

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest -q tests/test_sentinel_five_state_sync.py
```

结果：`11 passed`。

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest -q
```

结果：`178 passed`，1 条既有环境层面的 `pynvml` 弃用警告。

```bash
/root/miniconda3/envs/hqx/bin/python -m py_compile \
  mmst_vit/sentinel.py tests/test_sentinel_five_state_sync.py
```

结果：通过。

```bash
git diff --check
```

结果：通过。

CLI dry-run 测试使用 pytest 临时目录、临时 USDA/weather/JSONL/tree fixture；execute 流程测试直接调用同步函数并注入 mock bucket/下载/提取/上传函数。没有运行 CLI `--execute`，没有使用真实 OSS 凭证，也没有对真实数据执行删除或重写。

## 已知边界

- 本次没有重新扫描原报告中的约 28G 真实 weather 数据，因此不更新原报告中的真实目标 FIPS 和删除数量；真实运行前仍须在不带 `--execute` 的模式下生成并人工审阅新的 `plan.json`。
- 当前测试验证的是完整协议日期 `04-01` 至 `09-28`，与项目 `cropnet_protocol.py` 的 168 日历一致。
- 测试中的 execute 仅使用临时目录和 mock，不代表真实 HDF5、网络下载或 OSS 服务执行成功。
