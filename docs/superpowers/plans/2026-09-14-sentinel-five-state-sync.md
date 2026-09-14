# Sentinel Five-State Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dry-run-first, recoverable five-state AG-only cleanup/synchronization workflow to `mmst_vit/sentinel.py` while preserving ordinary download/upload behavior.

**Architecture:** Keep existing HF enumeration, resumable download, HDF5 county extraction, and OSS upload helpers. Add small pure planning/classification helpers and one orchestration function that writes a timestamped plan, performs AG synchronization before destructive cleanup, and atomically rewrites USDA and URL manifests. Use FIPS/ANSI allowlists for identity and inject the bucket only at execution time.

**Tech Stack:** Python 3, argparse, csv, pathlib, hashlib, JSONL, h5py, oss2, pytest.

## Global Constraints

- 五州固定为 Illinois、Iowa、Louisiana、Mississippi、New York。
- 州过滤使用解析州值和 ANSI/FIPS allowlist，不依赖源码子串。
- 年份固定为 2017--2022，模态固定为 AG。
- 新增模式默认 dry-run，只有同时传 `--execute` 才执行破坏性操作。
- 不删除 Sentinel 原始下载文件，直到 AG 补下载完成。
- OSS 删除使用 `bucket.delete_object`，凭证只从现有环境变量读取。
- 不改模型、不训练，普通下载/上传行为保持兼容。
- 必须运行专项/全量测试和 `py_compile`；不执行 `--execute`。

---

### Task 1: Add Pure Planning and Filtering Contracts

**Files:**
- Modify: `mmst_vit/sentinel.py`
- Create: `tests/test_sentinel_five_state_sync.py`

**Interfaces:**
- Produces `FIVE_STATE_NAMES`, `FIVE_STATE_ANSI`, `FIVE_STATE_YEARS`, `normalize_state_name`, `is_five_state_fips`, `filter_usda_rows`, `classify_url_rows`, and `build_sync_plan`.

- [ ] **Step 1: Write failing tests**

Add tests for parsed state values, USDA row retention, URL deletion classification, and a dry-run executor that does not call deletion methods.

- [ ] **Step 2: Run the focused tests and verify the expected missing-symbol failures**

Run: `pytest -q tests/test_sentinel_five_state_sync.py`

Expected: collection or assertion failures because the new planning helpers are not yet defined.

- [ ] **Step 3: Implement the minimal pure helpers**

Represent the five states as names and ANSI codes, normalize state names by whitespace/case, derive state identity from `state_ansi` or FIPS fields, retain only valid CORN/YEAR rows whose parsed FIPS belongs to the five-state allowlist, and classify old URL rows as keep/delete for non-five-state or NDVI records without inspecting arbitrary source substrings.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `pytest -q tests/test_sentinel_five_state_sync.py`

- [ ] **Step 5: Commit the pure contracts**

Run: `git add mmst_vit/sentinel.py tests/test_sentinel_five_state_sync.py && git commit -m "test: define five-state sentinel sync contracts"`

### Task 2: Implement Atomic Reports and Execution Orchestration

**Files:**
- Modify: `mmst_vit/sentinel.py`
- Modify: `tests/test_sentinel_five_state_sync.py`

**Interfaces:**
- Produces `atomic_write_text`, `backup_usda_files`, `delete_oss_objects`, `sync_cropnet_five_state`, and an execution result containing counts, failures, and report paths.

- [ ] **Step 1: Add failing tests for atomic write/backup and OSS failure propagation**

Assert USDA backups include SHA256 and line counts, temporary replacement is used for writes, and an OSS deletion exception is recorded and causes the sync to raise/fail.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest -q tests/test_sentinel_five_state_sync.py`

Expected: failures for the missing orchestration functions.

- [ ] **Step 3: Implement execution helpers**

Create timestamped run directories, serialize plan and status reports atomically, back up each USDA file before modification, delete OSS keys through `bucket.delete_object`, collect every failure, and raise after failures. Ensure execute order is AG download/extract/upload first, then OSS cleanup, weather directory cleanup, USDA replacement, and URL manifest replacement.

- [ ] **Step 4: Run focused and existing sentinel-related tests**

Run: `pytest -q tests/test_sentinel_five_state_sync.py tests/test_mmst_manifest_protocol.py`

- [ ] **Step 5: Commit orchestration**

Run: `git add mmst_vit/sentinel.py tests/test_sentinel_five_state_sync.py && git commit -m "feat: add recoverable five-state sentinel sync"`

### Task 3: Add Explicit CLI Mode and Verification Report

**Files:**
- Modify: `mmst_vit/sentinel.py`
- Modify: `tests/test_sentinel_five_state_sync.py`
- Create: `.superpowers/sdd/sentinel-five-state-sync-report.md`

**Interfaces:**
- Adds `--sync-cropnet-five-state`, `--execute`, source USDA/weather/run-root options, and `sentinel_urls_cropnet5_ag.jsonl` output behavior while leaving the existing CLI branch unchanged.

- [ ] **Step 1: Add failing CLI dry-run test**

Run the module with temporary USDA/weather/manifest fixtures and assert the JSON plan includes target FIPS, USDA keep/delete counts, weather deletions, URL deletions, AG download/upload lists, and no filesystem deletion occurs without `--execute`.

- [ ] **Step 2: Run the test and verify failure**

Run: `pytest -q tests/test_sentinel_five_state_sync.py`

- [ ] **Step 3: Implement CLI parsing and dry-run output**

Force years and `image_types={"AG"}` in the new mode, compute target FIPS from original USDA/HRRR data, print the plan summary, save plan/status JSON in the run directory, and never instantiate an OSS bucket or delete anything during dry-run.

- [ ] **Step 4: Run focused, full, compile, and dry-run verification**

Run: `pytest -q tests/test_sentinel_five_state_sync.py tests/test_mmst_manifest_protocol.py`; `pytest -q`; `python -m py_compile mmst_vit/sentinel.py`; run the new CLI mode against the available runtime data without `--execute`, saving stdout and reports under the requested run directory.

- [ ] **Step 5: Write the final report with observed counts and concerns**

Record the exact dry-run command, output paths, counts, test results, compile result, missing `hqx` executable, and explicit statement that `--execute` was not run.

- [ ] **Step 6: Commit code, tests, plan, and report**

Run: `git status --short`, inspect `git diff` and recent log, then stage only this plan, `mmst_vit/sentinel.py`, the new tests, and the requested report; commit with `git commit -m "feat: add five-state sentinel cleanup sync"`.
