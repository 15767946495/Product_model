# TFT 五州 AG 遥感数据管线实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 TFT 模型结构和训练逻辑的前提下，为五州 CropNet 数据建立 AG-only、12 时相遥感加载管线，并完成 2017--2020/2021/2022 年份划分和真实数据审计。

**Architecture:** 保留现有气象网格数据接口，新增独立的 AG 遥感 Dataset 和五州样本划分函数。AG Dataset 读取两个季度 HDF5 文件，每月保留 1 日和 15 日共 12 个时间点，应用与 MMST-ViT 一致的图像变换，但不把 AG 接入 `TFTEncoderForYieldPrediction`。五州运行清单和审计产物放在独立 runtime 目录，不覆盖现有八州 bundle。

**Tech Stack:** Python 3.10+、PyTorch、torchvision、NumPy、h5py、Pandas、JSONL、现有 `hqx` conda 环境。

## Global Constraints

- 只保留 Illinois、Iowa、Louisiana、Mississippi、New York 五州；最终清单中的州值必须属于五州允许集合。
- 训练集为 2017--2020，验证集为 2021，测试集为 2022。
- 短期气象仍为 4--9 月每月 1--28 日，共 168 个日时间步。
- 遥感只使用 AG，不读取 NDVI 或 Vegetation 文件。
- TFT AG Dataset 每月读取 1 日和 15 日影像，共 12 个时间点，输出 `(12, G, 3, 224, 224)`。
- 图像处理必须与 MMST-ViT 官方 DataWrapper 一致：训练随机增强，验证/测试 CenterCrop 加官方 Normalize。
- 随机种子默认使用 0；数据层不得修改全局随机状态。
- 不修改 `TFT_model/models.py`、模型 forward 签名、训练优化器或损失；不训练、不推理、不删除或修改原始 Sentinel 数据。
- 五州运行产物写入 `/data/raid0/hqx/Product_model_runtime/cropnet-five-state/`，不覆盖当前八州 `task8-active`。
- 所有验证命令使用 `/root/miniconda3/envs/hqx/bin/python`。

---

### Task 1: 建立五州协议和年份划分接口

**Files:**
- Create: `tests/test_tft_ag_data.py`
- Modify: `cropnet_protocol.py`
- Modify: `TFT_model/data.py`

**Interfaces:**
- `cropnet_protocol.py` exports `CROPNET_FIVE_STATES` as a set of normalized full state names.
- `TFT_model/data.py` exports `split_samples_by_year(samples, train_years=(2017, 2018, 2019, 2020), val_years=(2021,), test_years=(2022,)) -> dict[str, list[dict]]`.
- `TFT_model/data.py` exports `validate_five_state_sample(sample) -> None` and rejects unsupported state or year.

- [ ] **Step 1: Write failing tests for five-state filtering and split**

```python
def test_split_samples_by_year_uses_five_state_protocol():
    samples = [
        {"State": "illinois", "Year": 2017, "FIPS": "17001"},
        {"State": "iowa", "Year": 2020, "FIPS": "19001"},
        {"State": "new york", "Year": 2021, "FIPS": "36001"},
        {"State": "mississippi", "Year": 2022, "FIPS": "28001"},
    ]
    splits = split_samples_by_year(samples)
    assert [row["Year"] for row in splits["train"]] == [2017, 2020]
    assert [row["Year"] for row in splits["val"]] == [2021]
    assert [row["Year"] for row in splits["test"]] == [2022]


def test_split_rejects_state_outside_five_state_protocol():
    with pytest.raises(ValueError):
        split_samples_by_year([{"State": "texas", "Year": 2017, "FIPS": "48001"}])


def test_split_rejects_year_outside_protocol():
    with pytest.raises(ValueError):
        split_samples_by_year([{"State": "illinois", "Year": 2023, "FIPS": "17001"}])
```

- [ ] **Step 2: Run tests and verify failure**

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_tft_ag_data.py -q
```

Expected: FAIL because the five-state constant and split function do not exist.

- [ ] **Step 3: Implement the shared five-state constant and split function**

Add the normalized five-state set to `cropnet_protocol.py`. In `TFT_model/data.py`, import it and implement strict parsing:

```python
def split_samples_by_year(samples, train_years=(2017, 2018, 2019, 2020),
                          val_years=(2021,), test_years=(2022,)):
    splits = {"train": [], "val": [], "test": []}
    year_to_split = {
        **{int(year): "train" for year in train_years},
        **{int(year): "val" for year in val_years},
        **{int(year): "test" for year in test_years},
    }
    for index, sample in enumerate(samples):
        state = str(sample.get("State", "")).strip().lower()
        year = int(sample.get("Year", -1))
        if state not in CROPNET_FIVE_STATES:
            raise ValueError(f"sample {index} has unsupported state: {sample.get('State')!r}")
        if year not in year_to_split:
            raise ValueError(f"sample {index} has unsupported year: {year}")
        splits[year_to_split[year]].append(dict(sample))
    return splits
```

- [ ] **Step 4: Run tests and compile**

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_tft_ag_data.py -q
/root/miniconda3/envs/hqx/bin/python -m py_compile cropnet_protocol.py TFT_model/data.py tests/test_tft_ag_data.py
```

- [ ] **Step 5: Commit**

```bash
```

### Task 2: Implement AG-only HDF5 path and Dataset

**Files:**
- Modify: `TFT_model/data.py`
- Modify: `tests/test_tft_ag_data.py`

**Interfaces:**
- `build_ag_paths(sample, ag_root: str | Path) -> list[Path]` returns exactly two AG quarterly paths in chronological order.
- `select_ag_dates(group) -> list[str]` returns exactly `04-01, 04-15, ..., 09-01, 09-15`.
- `AgricultureImageDataset(samples, ag_root, train, image_size=224, seed=0)` returns a dictionary with `ag_images`, `ag_dates`, `FIPS`, `Year`, `State`, `County`, and `grid_count`.

- [ ] **Step 1: Create temporary HDF5 fixture tests**

The fixture must create two quarterly HDF5 files with one FIPS group and six date groups per quarter. Each `data` dataset has shape `(G, 224, 224, 3)` and dtype `uint8`. Tests must assert:

```python
item = AgricultureImageDataset(
    [sample], ag_root=tmp_path, train=False, seed=0
)[0]
assert item["ag_images"].shape == (12, 2, 3, 224, 224)
assert item["ag_dates"] == [
    "04-01", "04-15", "05-01", "05-15", "06-01", "06-15",
    "07-01", "07-15", "08-01", "08-15", "09-01", "09-15",
]
assert item["FIPS"] == "17001"
assert item["grid_count"] == 2
```

Add tests that assert:

- `build_ag_paths()` returns two paths and both contain `Agriculture`;
- a sample with an NDVI path is rejected;
- missing file, FIPS group, date group, or `data` dataset raises `ValueError` with sample context;
- inconsistent grid counts across dates raises `ValueError`.

- [ ] **Step 2: Run tests and verify failure**

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_tft_ag_data.py -q
```

Expected: FAIL because the AG path and Dataset interfaces do not exist.

- [ ] **Step 3: Implement strict AG path construction**

Construct exactly:

```text
<ag_root>/data/AG/<year>/<state_abbr>/Agriculture_<state_ansi>_<state_abbr>_<year>-04-01_<year>-06-30.h5
<ag_root>/data/AG/<year>/<state_abbr>/Agriculture_<state_ansi>_<state_abbr>_<year>-07-01_<year>-09-30.h5
```

Accept the actual runtime root layout used by the downloaded files, but reject any path whose filename or parent modality is not AG. Do not implement NDVI/Vegetation fallback.

- [ ] **Step 4: Implement date selection and HDF5 loading**

Use h5py with a context manager. Read only the target FIPS group, sort date names, require the exact 12 expected date strings, read `group[date]["data"]`, convert `G,H,W,C` to `G,C,H,W`, and stack the time dimension to `(12,G,3,224,224)`.

- [ ] **Step 5: Implement MMST-ViT-compatible transforms**

Use `torchvision.transforms`:

```python
Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
```

For training, apply `RandomResizedCrop(224)`, `RandomHorizontalFlip()`, `RandomApply(ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8)`, `RandomGrayscale(p=0.2)`, `GaussianBlur(kernel_size=9)`, then Normalize. For validation/test, apply `CenterCrop(224)`, then Normalize.

Use a local generator or per-sample deterministic seed derived from `seed`, sample index, and time index; do not call `torch.manual_seed()` or `np.random.seed()` inside Dataset construction.

- [ ] **Step 6: Run tests and compile**

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_tft_ag_data.py -q
/root/miniconda3/envs/hqx/bin/python -m py_compile TFT_model/data.py tests/test_tft_ag_data.py
```

- [ ] **Step 7: Commit**

```bash
```

### Task 3: Add five-state manifest generation and real AG audit

**Files:**
- Create: `tools/audit_tft_ag_data.py`
- Modify: `mmst_vit/config.py`
- Modify: `mmst_vit/manifest.py`
- Modify: `tests/test_tft_ag_data.py`
- Runtime output: `/data/raid0/hqx/Product_model_runtime/cropnet-five-state/`

**Interfaces:**
- `build_tft_ag_manifest(shared_rows, ag_root, output_dir) -> dict[str, int]` writes `train.jsonl`, `val.jsonl`, `test.jsonl` using the fixed year split and only AG paths.
- `audit_ag_samples(samples, ag_root) -> dict` returns valid/invalid counts, invalid reasons, state/year counts, and SHA256 for checked files.
- CLI `tools/audit_tft_ag_data.py --shared-jsonl ... --ag-root ... --output-dir ...` writes manifests and `ag_integrity.json` without training.

- [ ] **Step 1: Write failing manifest and audit tests**

Test that a valid five-state sample produces one of the three split files, that each row has exactly two AG paths, and that an invalid file is reported rather than silently skipped. Test that generated split counts sum to the input count and that no output path contains NDVI/Vegetation modality.

- [ ] **Step 2: Run tests and verify failure**

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_tft_ag_data.py -q
```

- [ ] **Step 3: Implement AG-only manifest generation**

Use `split_samples_by_year()` and `build_ag_paths()`. Before writing a row, validate both files and the target FIPS/date/data structure using the same logic as `AgricultureImageDataset`. Write only valid rows to `train.jsonl`, `val.jsonl`, and `test.jsonl`; write every invalid sample with its FIPS, year, paths, and error to `ag_integrity.json`.

- [ ] **Step 4: Update MMST config helpers to AG-only where used by TFT preparation**

Do not alter the official MMST-ViT six-time-point loader contract. Add a clearly named AG-only helper for TFT preparation that returns the two Agriculture quarterly paths and never adds NDVI/Vegetation paths.

- [ ] **Step 5: Run the real five-state AG audit**

```bash
/root/miniconda3/envs/hqx/bin/python tools/audit_tft_ag_data.py \
  --shared-jsonl /data/raid0/hqx/Product_model_runtime/task8-active/train_dataset/dataset.jsonl \
  --ag-root "/data/raid0/hqx/Product_model_runtime/DataSrc/mmst_vit/download/Sentinel-2 Imagery" \
  --output-dir /data/raid0/hqx/Product_model_runtime/cropnet-five-state
```

If the shared JSONL is still the previous eight-state file, first construct the five-state input by parsing and filtering through `split_samples_by_year()`; do not modify the eight-state active bundle. Record valid and invalid sample counts by split, state, and year.

- [ ] **Step 6: Verify real manifest and audit output**

```bash
/root/miniconda3/envs/hqx/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/data/raid0/hqx/Product_model_runtime/cropnet-five-state')
audit = json.loads((root / 'audit/ag_integrity.json').read_text())
assert set(audit['states']) <= {'illinois', 'iowa', 'louisiana', 'mississippi', 'new york'}
for name in ('train', 'val', 'test'):
    assert (root / 'manifests' / f'{name}.jsonl').exists()
PY
```

- [ ] **Step 7: Run tests and commit**

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_tft_ag_data.py -q
/root/miniconda3/envs/hqx/bin/python -m py_compile tools/audit_tft_ag_data.py mmst_vit/config.py mmst_vit/manifest.py
git add tools/audit_tft_ag_data.py mmst_vit/config.py mmst_vit/manifest.py tests/test_tft_ag_data.py
git commit -m "feat: generate five-state AG manifests and audit"
```

### Task 4: Final data-only integration verification

**Files:**
- Modify: `tests/test_tft_ag_data.py`
- Create: `.superpowers/sdd/task-9-report.md`
- Runtime output: `/data/raid0/hqx/Product_model_runtime/cropnet-five-state/audit/protocol.json`

**Interfaces:**
- Does not change production model or training code.
- Produces a final report proving data interfaces and real AG samples are ready for a later TFT integration task.

- [ ] **Step 1: Add no-model-integration tests**

Assert that importing `TFTEncoderForYieldPrediction` still exposes the existing forward signature and that `AgricultureImageDataset` is independent of model construction. Assert that the AG Dataset output contains metadata but no weather or label loading side effects.

- [ ] **Step 2: Run data loader smoke tests against one real valid sample**

Use the first valid real manifest row and run:

```python
item = AgricultureImageDataset([row], ag_root=AG_ROOT, train=False, seed=0)[0]
assert item["ag_images"].shape[0] == 12
assert item["ag_images"].shape[2:] == (3, 224, 224)
```

Do not instantiate or train TFT.

- [ ] **Step 3: Run complete verification**

```bash
/root/miniconda3/envs/hqx/bin/python -m pytest -q
/root/miniconda3/envs/hqx/bin/python -m py_compile cropnet_protocol.py TFT_model/data.py tools/audit_tft_ag_data.py mmst_vit/config.py mmst_vit/manifest.py tests/test_tft_ag_data.py
```

- [ ] **Step 4: Write the final report**

Record:

- five-state counts by split;
- valid/invalid AG sample counts and reasons;
- exact 12 dates;
- representative AG tensor shape;
- transform and seed configuration;
- source file SHA256 summary;
- confirmation that `TFT_model/models.py` and training entrypoints were not changed;
- confirmation that no training/inference was run;
- confirmation that current eight-state `task8-active` and original Sentinel files were untouched.

- [ ] **Step 5: Commit tests only**

```bash
```

## Self-Review Checklist

- Five-state scope and fixed year split are covered by Tasks 1 and 3.
- AG-only two-quarter paths and 12 dates are covered by Task 2.
- MMST-ViT transforms and seed 0 are covered by Task 2.
- Missing/corrupt HDF5 handling is covered by Tasks 2 and 3.
- Real five-state AG audit is covered by Task 3.
- No TFT model/fusion/training changes are included in any task.
- Existing eight-state runtime data is isolated from the new five-state output.
- No task runs model training or inference.
