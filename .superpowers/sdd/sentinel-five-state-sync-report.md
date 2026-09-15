# Sentinel 五州清理/同步报告

## 状态

- 修复状态：两个阻断已修复，专项/全量测试和编译验证完成；已提交源代码、测试和本报告。
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
- 错误大小或缺失的源 HDF5 先下载并重新校验，随后才提取和上传；正确源文件才允许复用提取结果。
- USDA/weather 交集按年份保存；weather 目录年份和 CSV 行年份不一致的记录跳过并写入计划审计明细。
- 每个 execute 步骤成功或失败后立即原子持久化 status；失败后续步骤标记 `not-run` 并停止破坏性操作。
- 旧 manifest 保留同时限制为 2017--2022、五州、AG；目标 FIPS 非空但没有 HF AG entries 时 dry-run 标记危险，execute 直接停止。
- HF 州级条目按条目年份使用对应的 `target_fips_by_year`，不会把其他年份有效县扩展到当前年份；dry-run 上传计划也按条目年份过滤。
- 当所有年份的目标 FIPS 都为空时，dry-run 写入 `danger` 失败状态，execute 在任何 OSS、weather、USDA 或 manifest 操作前写状态并停止。
- AG 同步返回失败时，status 原样持久化完整同步结果（包括下载、上传、跳过和失败明细），并将后续清理标记为 `not-run`。
- AG 每个 state/year entry 的 expected FIPS 与实际 HDF5 可提取 FIPS 严格相等；缺源、空提取、部分提取均为失败，禁止跳过、上传及后续 OSS/weather/USDA/manifest 清理。
- OSS 删除在任何调用前严格校验规范化 `sentinel/` prefix；拒绝空、绝对路径、`..` 遍历、其他 prefix 和 prefix 本身，并在 status 中保存逐项失败原因。

## 测试覆盖

- USDA 五州/CORN/YEAR/FIPS 过滤和错误测试 oracle 修正。
- weather CSV 的 `Daily`、FIPS、月份/日期解析和完整 168 日期校验。
- 已存在且大小正确的 AG 源文件不列入待下载；大小不匹配时列入待下载。
- CSV roundtrip 验证包含带逗号字段和带引号县名。
- USDA 备份、SHA256/行数报告和原子重写。
- AG 下载 -> 县提取 -> OSS 上传顺序。
- execute mock 流程中不删除 Sentinel 原始源文件。
- OSS 删除部分成功/失败传播、status 持久化和后续清理不执行。
- AG 缺 group/空提取/部分提取不会上传且返回详细失败集合；OSS 非法 key 整体预检失败且不会调用 bucket。
- 输出 manifest 只包含 AG/五州记录。
- CLI 新模式默认 dry-run。

## 验证

使用指定 hqx Python 环境：

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest -q tests/test_sentinel_five_state_sync.py
```

结果：`30 passed`；与既有 manifest 协议测试合计 `38 passed`。

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest -q
```

结果：`197 passed`，1 条既有环境层面的 `pynvml` 弃用警告。

```bash
/root/miniconda3/envs/hqx/bin/python -m py_compile \
  mmst_vit/sentinel.py tests/test_sentinel_five_state_sync.py
```

结果：通过。

本轮专项与协议回归命令：

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest -q tests/test_sentinel_five_state_sync.py tests/test_mmst_manifest_protocol.py
```

结果：`38 passed`。

本轮专项 `py_compile`、全量 pytest 和 `git diff --check` 均通过；验证仅使用临时目录、测试 fixture 和 mock。CLI dry-run 回归测试以临时 USDA/weather/JSONL/tree fixture 执行 `python -m mmst_vit.sentinel --sync-cropnet-five-state`，检查 `dry_run=true`、空目标集的 danger 状态以及原始文件未改变；本轮未对真实 runtime 数据执行 CLI dry-run。未运行 CLI `--execute`，未创建真实 OSS bucket，未删除或重写真实 weather、USDA、Sentinel 源文件及真实 manifest。

```bash
git diff --check
```

结果：通过。

CLI dry-run 测试使用 pytest 临时目录、临时 USDA/weather/JSONL/tree fixture；execute 流程测试直接调用同步函数并注入 mock bucket/下载/提取/上传函数。没有运行 CLI `--execute`，没有使用真实 OSS 凭证，也没有对真实数据执行删除或重写。

## 已知边界

- 本次没有重新扫描原报告中的约 28G 真实 weather 数据，因此不更新原报告中的真实目标 FIPS 和删除数量；真实运行前仍须在不带 `--execute` 的模式下生成并人工审阅新的 `plan.json`。
- 当前测试验证的是完整协议日期 `04-01` 至 `09-28`，与项目 `cropnet_protocol.py` 的 168 日历一致。
- 测试中的 execute 仅使用临时目录和 mock，不代表真实 HDF5、网络下载或 OSS 服务执行成功。
- 本轮新增回归覆盖缺 group/空提取/部分提取、缺失源文件、非法 OSS key，以及零 OSS 调用的整体预检。
- target_fips_by_year 空集保护和 AG 失败状态保持不变，并由既有回归测试覆盖。
