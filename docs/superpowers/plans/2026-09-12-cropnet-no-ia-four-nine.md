# CropNet 八州四月至九月统一协议实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 MMST-ViT、TFT、四类基线和消融实验统一到八州、4--9 月每月前 28 天的 168 步 CropNet 协议，并清理旧协议权重和缓存，不启动模型训练。

**Architecture:** 在 `train_dataset` 层统一生成八州、4--9 月、每月 1--28 日的县级 JSONL 和网格缓存；TFT/基线/消融只消费这份共享数据。MMST-ViT 使用独立的 `valid-no-ia.*.jsonl` 和 `*.official.no-ia.json` 清单，但共享相同八州范围和 4--9 月短期气象协议。所有旧 checkpoint 和会被默认入口误用的旧缓存在重建前清理。

**Tech Stack:** Python 3.10+、PyTorch、NumPy、Pandas、h5py、现有 `hqx` conda 环境、JSONL/JSON、PyTorch `.pt` 缓存。

## Global Constraints

- 只保留 Minnesota、Wisconsin、Michigan、Illinois、Indiana、Ohio、Missouri、Kentucky 八州；所有生成的样本、缓存条目和 MMST-ViT manifest 都必须通过八州允许列表校验。
- 所有短期气象统一为 4 月 1 日至 9 月 28 日，每月前 28 天，共 168 个最大时间步。
- MMST-ViT 遥感只引用 4--6 月和 7--9 月两个季度的 AG/NDVI 文件。
- MMST-ViT 清单文件名必须是 `valid-no-ia.jsonl`、`valid-no-ia.train.jsonl`、`valid-no-ia.val.jsonl`、`valid-no-ia.test.jsonl`、`train.official.no-ia.json`、`val.official.no-ia.json`、`test.official.no-ia.json`。
- 不修改模型核心网络结构，不启动训练，不删除 Sentinel 原始下载文件。
- 所有命令使用 `/root/miniconda3/envs/hqx/bin/python` 或 `conda run -n hqx`。
- 所有手工编辑使用 `apply_patch`；运行时生成的数据和日志不加入 Git。

---

### Task 1: 建立统一数据协议常量和测试

**Files:**
- Create: `tests/test_cropnet_protocol.py`
- Modify: `train_dataset/prepare_jsonl.py`
- Modify: `train_dataset/prepare_grid.py`
- Modify: `TFT_model/data.py`
- Modify: `BaseLine_Model/common/data.py`

**Interfaces:**
- Produces a single source of truth for allowed states, start/end month, days per month, and maximum time steps.
- Existing scripts continue to expose their current command-line entry points.

- [ ] **Step 1: Write failing protocol tests**

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train_dataset"))
sys.path.insert(0, str(ROOT / "TFT_model"))
sys.path.insert(0, str(ROOT / "BaseLine_Model"))

from prepare_jsonl import ALLOWED_STATES, START_MONTH, END_MONTH, DAYS_PER_MONTH
from common import data as baseline_data
from data import PROTOCOL_MAX_STEPS


def test_shared_protocol_values():
    assert ALLOWED_STATES == {
        "minnesota", "wisconsin", "michigan", "illinois",
        "indiana", "ohio", "missouri", "kentucky",
    }
    assert (START_MONTH, END_MONTH, DAYS_PER_MONTH) == (4, 9, 28)
    assert PROTOCOL_MAX_STEPS == 168
    assert baseline_data.N_STEPS == 168


def test_protocol_has_exactly_eight_allowed_states():
    assert len(ALLOWED_STATES) == 8
    assert all(len(name) > 2 for name in ALLOWED_STATES)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_cropnet_protocol.py -q
```

Expected: FAIL because the shared constants and the old 275-step/old state definitions still exist.

- [ ] **Step 3: Add shared constants with exact values**

In `train_dataset/prepare_jsonl.py`, define near the path configuration:

```python
ALLOWED_STATES = {
    "minnesota", "wisconsin", "michigan", "illinois",
    "indiana", "ohio", "missouri", "kentucky",
}
START_MONTH = 4
END_MONTH = 9
DAYS_PER_MONTH = 28
PROTOCOL_MAX_STEPS = (END_MONTH - START_MONTH + 1) * DAYS_PER_MONTH
```

Filter weather rows by:

```python
df = df[
    df["Month"].between(START_MONTH, END_MONTH)
    & df["Day"].between(1, DAYS_PER_MONTH)
].copy()
```

Filter USDA groups using the same explicit allowed-state set before weather processing. The validation target is the extracted state values, not source-code substring inspection.

Make `_county_daily_series()` preserve both calendar fields:

```python
result["month"] = daily_avg["date"].dt.month.tolist()
result["day"] = daily_avg["date"].dt.day.tolist()
```

In `TFT_model/data.py`, define:

```python
PROTOCOL_START_MONTH = 4
PROTOCOL_END_MONTH = 9
PROTOCOL_DAYS_PER_MONTH = 28
PROTOCOL_MAX_STEPS = 168
```

In `BaseLine_Model/common/data.py`, use the same eight-state set and:

```python
N_STEPS = 168
```

- [ ] **Step 4: Update cache contract checks**

Change the grid-cache loader to require the new cache version and protocol metadata:

```python
if payload.get("version") != 4:
    raise ValueError("grid_cache must use version 4")
if payload.get("time_window") != "04-01--09-28":
    raise ValueError("grid_cache time window mismatch")
if payload.get("days_per_month") != 28:
    raise ValueError("grid_cache days_per_month mismatch")
```

Update `prepare_grid.py` to preserve `month` and `day` tensors and save `version=4`, `time_window="04-01--09-28"`, `days_per_month=28`, and `max_steps=168`.

- [ ] **Step 5: Run the protocol tests**

Run:

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_cropnet_protocol.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the protocol layer**

```bash
git add tests/test_cropnet_protocol.py train_dataset/prepare_jsonl.py train_dataset/prepare_grid.py TFT_model/data.py BaseLine_Model/common/data.py
git commit -m "refactor: define unified eight-state cropnet protocol"
```

### Task 2: Rebuild county JSONL and grid cache

**Files:**
- Modify: `train_dataset/prepare_jsonl.py`
- Modify: `train_dataset/prepare_grid.py`
- Runtime output: `/data/raid0/hqx/Product_model_runtime/train_dataset/dataset.jsonl`
- Runtime output: `/data/raid0/hqx/Product_model_runtime/train_dataset/grid_cache.pt`
- Runtime output: `/data/raid0/hqx/Product_model_runtime/train_dataset/grid_cache_meta.json`

**Interfaces:**
- Consumes the existing USDA/WRF-HRRR data under `/data/raid0/hqx/Product_model_runtime/DataSrc/cropnet_dataset/data`.
- Produces JSONL and cache rows with identical ordering and no excluded state.

- [ ] **Step 1: Add a pre-generation dry-run validator**

Add a function in `prepare_jsonl.py` that reports, without writing files:

```python
{
    "allowed_states": sorted(ALLOWED_STATES),
    "start_month": 4,
    "end_month": 9,
    "days_per_month": 28,
    "max_steps": 168,
}
```

Expose it with `--dry-run` so the real data file is not changed during the first check.

- [ ] **Step 2: Run the dry run in `hqx`**

```bash
/root/miniconda3/envs/hqx/bin/python train_dataset/prepare_jsonl.py --dry-run
```

Expected: the printed protocol shows eight allowed states and `max_steps=168`.

- [ ] **Step 3: Generate the new JSONL to a temporary output**

Add `--output` support if needed, then run:

```bash
/root/miniconda3/envs/hqx/bin/python train_dataset/prepare_jsonl.py --output /tmp/cropnet-dataset-04-09.jsonl
```

Do not replace the runtime dataset until the validation script passes.

- [ ] **Step 4: Validate temporary JSONL content**

Run a `hqx` one-off validator that asserts every row satisfies:

```python
assert row["State"].lower() in ALLOWED_STATES
assert set(row["month"]).issubset(set(range(4, 10)))
assert len(row["day"]) == row["l_enc"]
assert all(1 <= day <= 28 for day in row["day"])
assert row["l_enc"] <= 168
```

Also report per-year and per-state counts.

- [ ] **Step 5: Replace the runtime JSONL after validation**

Copy the validated temporary output to:

```text
/data/raid0/hqx/Product_model_runtime/train_dataset/dataset.jsonl
```

Keep a timestamped validation report beside it.

- [ ] **Step 6: Rebuild the grid cache from the new JSONL**

Run:

```bash
/root/miniconda3/envs/hqx/bin/python train_dataset/prepare_grid.py
```

The rebuilt cache must have:

```text
version = 4
time_window = 04-01--09-28
max_steps = 168
```

- [ ] **Step 7: Validate cache alignment**

Run a `hqx` validator that checks:

```python
assert len(cache["entries"]) == len(jsonl_rows)
assert cache["version"] == 4
assert cache["max_steps"] == 168
for row, entry in zip(jsonl_rows, cache["entries"]):
    if entry is not None:
        assert row["State"].lower() in ALLOWED_STATES
        assert entry["feats"].shape[1] <= 168
        assert entry["month"].shape[0] == entry["feats"].shape[1]
        assert entry["day"].shape[0] == entry["feats"].shape[1]
```

- [ ] **Step 8: Commit data-pipeline code only**

```bash
git add train_dataset/prepare_jsonl.py train_dataset/prepare_grid.py
git commit -m "feat: rebuild cropnet data for eight states and 168 steps"
```

Do not stage generated data under `/data/raid0/hqx`.

### Task 3: Update TFT and ablation time semantics

**Files:**
- Modify: `TFT_model/train.py`
- Modify: `TFT_model/infer.py`
- Modify: `TFT_model/data.py`
- Modify: `TFT_model/ablation_rope.py`
- Modify: `ablation/ablation.py`

**Interfaces:**
- Consumes the rebuilt version-4 grid cache.
- Keeps the existing `TFTEncoderForYieldPrediction` network interface unchanged.

- [ ] **Step 1: Replace TFT default state lists with the eight-state allowlist**

Import or reuse one allowlist containing exactly the eight permitted full state names. Filter loaded metadata with this allowlist in `train.py`, `infer.py`, and `ablation/ablation.py`. Do not add a state-deny list.

- [ ] **Step 2: Make feature construction start at April**

Change the cumulative feature functions in `TFT_model/data.py` so their day-of-year origin is April 1 rather than March 1. Keep GDD/KDD/CumPRCP/CumDeficit formulas unchanged apart from the new sequence origin.

- [ ] **Step 3: Remove hard-coded 11-month inference anchoring**

In `infer.py`:

```python
end_d = _date(yr, 9, 28)
```

Change the default cutoff string to valid 4--9 month points, for example:

```text
06-01,06-15,06-28,07-01,07-15,07-28,08-01,08-15,08-28,09-01,09-15,09-28
```

Use the last valid sequence step for final prediction rather than filtering with `month_ids >= 8`.

- [ ] **Step 4: Update TFT cache contract and shape documentation**

Replace all remaining TFT comments/docstrings describing 275 steps or months 3--11 with 168 steps and months 4--9. Preserve the model’s dynamic sequence handling.

- [ ] **Step 5: Update ablation launchers**

Ensure both ablation launchers pass the same rebuilt data defaults and do not contain old state names, old 275-step text, or old 11-month cutoff text.

- [ ] **Step 6: Run static verification**

```bash
/root/miniconda3/envs/hqx/bin/python -m py_compile TFT_model/data.py TFT_model/train.py TFT_model/infer.py TFT_model/ablation_rope.py ablation/ablation.py
```

- [ ] **Step 7: Commit TFT changes**

```bash
git add TFT_model/data.py TFT_model/train.py TFT_model/infer.py TFT_model/ablation_rope.py ablation/ablation.py
git commit -m "feat: align TFT and ablations with cropnet short-season protocol"
```

### Task 4: Update shared baselines and DeepCropNet

**Files:**
- Modify: `BaseLine_Model/common/data.py`
- Modify: `BaseLine_Model/CNNRNN/cnn_rnn.py`
- Modify: `BaseLine_Model/ConvLSTM/convlstm.py`
- Modify: `BaseLine_Model/GNNRNN/gnn_rnn.py`
- Modify: `BaseLine_Model/deepcropnet/deepcropnet.py`

**Interfaces:**
- All baselines consume `common.data.prepare()` output built from the rebuilt 168-step cache.
- Model forward signatures remain unchanged.

- [ ] **Step 1: Update the shared baseline constants**

Set the eight-state allowlist and `N_STEPS = 168` in `common/data.py`. Update padding, tensor shape, and comments accordingly.

- [ ] **Step 2: Update CNN-RNN shape assumptions**

Change docstrings and pooling shape comments from 275 to 168. Keep the two pooling layers and forward signature unchanged.

- [ ] **Step 3: Update ConvLSTM shape assumptions**

Change input documentation to `(B,G,168,11)` and verify `grid_batch()` uses the shared `N_STEPS` without another hard-coded length.

- [ ] **Step 4: Update GNN-RNN shape assumptions**

Change input documentation to `(N,168,11)` and ensure graph construction remains based on the rebuilt sample set.

- [ ] **Step 5: Update DeepCropNet input origin**

Set the weekly accumulation start index to `0` because the rebuilt daily sequence begins on April 1. Preserve 20 weekly bins, which consume the first 140 protocol days and remain fully contained in the 168-step input.

- [ ] **Step 6: Run baseline compile and dry-run checks**

```bash
/root/miniconda3/envs/hqx/bin/python -m py_compile BaseLine_Model/common/data.py BaseLine_Model/common/train.py BaseLine_Model/CNNRNN/cnn_rnn.py BaseLine_Model/ConvLSTM/convlstm.py BaseLine_Model/GNNRNN/gnn_rnn.py BaseLine_Model/deepcropnet/deepcropnet.py
/root/miniconda3/envs/hqx/bin/python BaseLine_Model/CNNRNN/cnn_rnn.py --help
/root/miniconda3/envs/hqx/bin/python BaseLine_Model/ConvLSTM/convlstm.py --help
/root/miniconda3/envs/hqx/bin/python BaseLine_Model/GNNRNN/gnn_rnn.py --help
/root/miniconda3/envs/hqx/bin/python BaseLine_Model/deepcropnet/deepcropnet.py --help
```

- [ ] **Step 7: Commit baseline changes**

```bash
git add BaseLine_Model/common/data.py BaseLine_Model/common/train.py BaseLine_Model/CNNRNN/cnn_rnn.py BaseLine_Model/ConvLSTM/convlstm.py BaseLine_Model/GNNRNN/gnn_rnn.py BaseLine_Model/deepcropnet/deepcropnet.py
git commit -m "feat: align baselines with 168-step cropnet protocol"
```

### Task 5: Rebuild MMST-ViT manifests without the excluded state

**Files:**
- Modify: `mmst_vit/manifest.py`
- Modify: `mmst_vit/config.py`
- Modify: `mmst_vit/official.py`
- Test: `tests/test_mmst_manifest_protocol.py`
- Runtime output: `/data/raid0/hqx/Product_model_runtime/DataSrc/mmst_vit/manifests/valid-no-ia*.jsonl`
- Runtime output: `/data/raid0/hqx/Product_model_runtime/DataSrc/mmst_vit/manifests/*.official.no-ia.json`

**Interfaces:**
- Manifest generation returns only the eight allowed states.
- `official_sample_record()` creates paths for 4--9 month short-term HRRR and the two Sentinel quarters.

- [ ] **Step 1: Write manifest tests**

Test that manifest construction preserves only the eight allowed states and `official_sample_record()` produces exactly six short-term HRRR month paths plus four Sentinel paths. The test must inspect parsed state values and paths, not source-code substrings.

- [ ] **Step 2: Run the manifest tests and verify failure**

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_mmst_manifest_protocol.py -q
```

Expected: FAIL against the current all-state logic and old output names.

- [ ] **Step 3: Add the eight-state filter and exact output filenames**

Update `manifest.py` to filter input samples with the explicit eight-state allowlist and write:

```text
valid-no-ia.jsonl
valid-no-ia.train.jsonl
valid-no-ia.val.jsonl
valid-no-ia.test.jsonl
```

- [ ] **Step 4: Restrict official short-term paths**

In `config.py`, generate short-term HRRR paths only for months `4..9`; preserve long-term monthly lists in the official five-year shape. Keep Sentinel paths exactly as:

```text
AG 04-01--06-30
AG 07-01--09-30
NDVI 04-01--06-30
NDVI 07-01--09-30
```

- [ ] **Step 5: Write official JSON arrays with required names**

Generate:

```text
train.official.no-ia.json
val.official.no-ia.json
test.official.no-ia.json
```

All paths must be validated before writing the final files.

- [ ] **Step 6: Run manifest tests and compile checks**

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_mmst_manifest_protocol.py -q
/root/miniconda3/envs/hqx/bin/python -m py_compile mmst_vit/manifest.py mmst_vit/config.py mmst_vit/official.py
```

- [ ] **Step 7: Commit MMST-ViT manifest changes**

```bash
git add mmst_vit/manifest.py mmst_vit/config.py mmst_vit/official.py tests/test_mmst_manifest_protocol.py
git commit -m "feat: rebuild MMST-ViT manifests for unified eight-state protocol"
```

### Task 6: Remove old weights and protocol caches

**Files:**
- No repository source files.
- Delete runtime artifacts under `/data/raid0/hqx/Product_model_runtime` only after code and data validation.

**Interfaces:**
- No model training is run.
- Sentinel raw downloads remain untouched.

- [ ] **Step 1: Print the deletion candidate list**

```bash
find /data/raid0/hqx/Product_model_runtime -type f \( -name '*.pth' -o -name 'baselines_data*.pt' -o -name 'train_dcn_data*.pt' -o -name 'val_dcn_data*.pt' -o -name 'test_dcn_data*.pt' \) -print
```

- [ ] **Step 2: Delete old checkpoint files**

Delete only old model weights under baseline output directories and old TFT/ablation training output directories. Do not delete raw Sentinel files, USDA files, weather CSVs, soil files, or newly generated manifests.

- [ ] **Step 3: Delete old shared caches before rebuilding**

Delete:

```text
/data/raid0/hqx/Product_model_runtime/train_dataset/dataset.jsonl
/data/raid0/hqx/Product_model_runtime/train_dataset/grid_cache.pt
/data/raid0/hqx/Product_model_runtime/train_dataset/grid_cache_meta.json
/data/raid0/hqx/Product_model_runtime/BaseLine_Model/output/baselines_data.pt
/data/raid0/hqx/Product_model_runtime/BaseLine_Model/output/baselines_data_val*_test*.pt
```

Preserve historical JSON/PNG/log files unless they are inside a checkpoint-only directory explicitly identified in the deletion list.

- [ ] **Step 4: Verify no old checkpoint or cache remains in active paths**

```bash
find /data/raid0/hqx/Product_model_runtime -type f \( -name '*.pth' -o -name 'baselines_data*.pt' -o -name '*dcn_data*.pt' \) -print
```

Expected: no old active weights or protocol caches.

### Task 7: Final protocol audit without training

**Files:**
- Create: `tests/test_cropnet_runtime_protocol.py`
- Modify: none unless audit finds a mismatch.

**Interfaces:**
- Consumes rebuilt JSONL, grid cache, and MMST-ViT official manifests.
- Produces a machine-readable audit report; does not train or infer.

- [ ] **Step 1: Implement runtime audit checks**

The audit must assert:

```python
assert all(row["State"].lower() in ALLOWED_STATES for row in jsonl_rows)
assert all(len(row["day"]) == row["l_enc"] for row in jsonl_rows)
assert all(all(1 <= day <= 28 for day in row["day"]) for row in jsonl_rows)
assert cache["version"] == 4
assert cache["max_steps"] == 168
assert cache["time_window"] == "04-01--09-28"
assert len(cache["entries"]) == len(jsonl_rows)
```

For each official JSON, assert parsed state values and paths:

```python
assert all(row["state"].lower() in ALLOWED_STATE_ABBRS for row in rows)
assert len(row["data"]["HRRR"]["short_term"]) == 6
assert len(row["data"]["sentinel"]) == 4
```

The audit must validate parsed state values against the eight-state allowlist; it must not scan source code or filenames for state substrings.

- [ ] **Step 2: Run the full static and runtime audit**

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_cropnet_protocol.py tests/test_mmst_manifest_protocol.py tests/test_cropnet_runtime_protocol.py -q
/root/miniconda3/envs/hqx/bin/python -m py_compile train_dataset/prepare_jsonl.py train_dataset/prepare_grid.py TFT_model/data.py TFT_model/train.py TFT_model/infer.py BaseLine_Model/common/data.py BaseLine_Model/CNNRNN/cnn_rnn.py BaseLine_Model/ConvLSTM/convlstm.py BaseLine_Model/GNNRNN/gnn_rnn.py BaseLine_Model/deepcropnet/deepcropnet.py mmst_vit/manifest.py mmst_vit/config.py
```

Expected: all tests pass; no training process starts.

- [ ] **Step 3: Inspect final Git state**

```bash
git status --short
git diff --check
git log --oneline -10
```

- [ ] **Step 4: Commit final audit tests**

```bash
git add tests/test_cropnet_runtime_protocol.py
git commit -m "test: audit unified cropnet runtime protocol"
```

## Self-Review Checklist

- Spec requirement for exact MMST-ViT manifest filenames is covered in Task 5.
- Spec requirement for eight-state extraction is covered in Tasks 1, 2, 3, 4, 5, and 7.
- Spec requirement for 4--9 months and 28 days per month is covered in Tasks 1--5.
- Spec requirement for 168-step cache metadata and old-cache rejection is covered in Tasks 1, 2, and 7.
- Spec requirement for old weights/cache cleanup without Sentinel deletion is covered in Task 6.
- No task starts model training; all runtime checks are static or data-only.
- No task changes the core model forward signatures.
