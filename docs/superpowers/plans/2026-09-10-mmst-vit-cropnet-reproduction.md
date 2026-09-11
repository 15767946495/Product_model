# MMST-ViT CropNet 复现实装计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改动现有 TFT 模型的前提下，建立可重复生成有效样本、筛选/下载 Sentinel-2 数据、对齐 MMST-ViT 输入并执行固定年份训练评估的运行工具链。

**Architecture:** 新增 `mmst_vit/` 复现工具包。`manifest.py` 负责 USDA+气象有效样本和固定年份划分；`sentinel.py` 负责 Hugging Face 远程树解析、下载清单、断点下载和 SHA256 校验；`alignment.py` 负责本地三源数据对齐；`run.py` 负责配置、官方源码 checkout、训练前检查和结果目录初始化。官方 MMST-ViT 作为独立 source 目录运行，适配代码通过配置和薄 loader 层连接，不复制模型结构。

**Tech Stack:** Python 3.10+, stdlib (`argparse`, `json`, `urllib`, `hashlib`, `pathlib`, `subprocess`), 现有 pandas/numpy/pytest，官方 MMST-ViT 独立环境。

## Global Constraints

- 训练集固定为 2017--2020 年，验证集固定为 2021 年，测试集固定为 2022 年。
- 不使用 Tiny CropNet，不下载全部 CropNet 县域数据。
- 目标县来自同时具有有效 USDA 产量标签和 WRF-HRRR 气象序列的 `(FIPS, Year)` 样本。
- 所有标准化统计量只能由训练年份计算。
- 下载前必须输出文件数量、总大小和目标 FIPS 统计，并保存清单。
- 下载支持断点续传、已存在文件跳过、下载后校验；认证或链接失效时保留清单并报告缺失文件。
- 2022 年样本只能用于最终测试，不得参与训练、早停、超参数或结构选择。
- 结果输出 RMSE（`bu/ac`）、R2、Pearson `Corr` 和 2022 年逐县预测。
- 不修改官方 MMST-ViT 模型计算图；兼容性修改必须记录在运行配置中。

---

### Task 1: 有效样本清单与固定划分

**Files:**
- Create: `mmst_vit/__init__.py`
- Create: `mmst_vit/manifest.py`
- Create: `tests/test_mmst_manifest.py`

**Interfaces:**
- `build_valid_samples(usda_dir: Path, weather_dir: Path, years: Sequence[int]) -> list[dict]`
- `split_samples(samples: Sequence[dict]) -> dict[str, list[dict]]`
- `write_jsonl(path: Path, rows: Iterable[dict]) -> None`

- [ ] **Step 1: Write failing tests** for intersecting USDA/HRRR `(FIPS, Year)`, five-digit FIPS normalization, exact year split, and rejection of 2022 in train/validation.
- [ ] **Step 2: Run `pytest tests/test_mmst_manifest.py -q` and verify the new imports/behavior fail.**
- [ ] **Step 3: Implement the smallest stdlib/pandas implementation.** USDA rows must require numeric yield and matching `USDA_Corn_County_<year>.csv`; weather availability is derived from `weather/<year>/<state>/*.csv` and FIPS column inspection, without materializing weather sequences in the manifest.
- [ ] **Step 4: Run the focused test and then validate against the repository data with `python -m mmst_vit.manifest --output ... --years 2017 2018 2019 2020 2021 2022`.**
- [ ] **Step 5: Commit with `git add mmst_vit tests/test_mmst_manifest.py && git commit -m "feat: add MMST valid sample manifest"`.**

### Task 2: Sentinel-2 清单、下载与校验

**Files:**
- Create: `mmst_vit/sentinel.py`
- Create: `tests/test_mmst_sentinel.py`

**Interfaces:**
- `parse_hf_tree(payload: list[dict], target_fips: set[str], years: set[int], image_types: set[str]) -> list[dict]`
- `summarize_manifest(entries: Sequence[dict]) -> dict`
- `download_manifest(entries: Sequence[dict], destination: Path, resume: bool = True, dry_run: bool = False) -> dict`

- [ ] **Step 1: Write failing tests** for AG/NDVI and year filtering, FIPS extraction from file/path metadata, state-level fallback entries, size summary, `.part` resume, existing-file skip, and SHA256 verification.
- [ ] **Step 2: Run `pytest tests/test_mmst_sentinel.py -q` and verify failure for missing behavior.**
- [ ] **Step 3: Implement HF tree parsing without assuming independent county files.** Entries must preserve URL, relative path, size, optional checksum, image type, year, state, and matched FIPS; unmatched files are excluded.
- [ ] **Step 4: Implement HTTP range resume using `urllib.request`, atomic rename after verification, manifest JSONL output, and explicit missing/error records.** Do not make network requests during unit tests.
- [ ] **Step 5: Run focused tests and a dry-run against the HF API, printing count/bytes/FIPS summary before any download.**
- [ ] **Step 6: Commit with `git add mmst_vit/sentinel.py tests/test_mmst_sentinel.py && git commit -m "feat: add scoped Sentinel manifest downloader"`.**

### Task 3: 三源对齐与运行配置

**Files:**
- Create: `mmst_vit/alignment.py`
- Create: `mmst_vit/config.py`
- Create: `mmst_vit/run.py`
- Create: `tests/test_mmst_alignment.py`

**Interfaces:**
- `validate_alignment(samples: Sequence[dict], sentinel_index: Mapping, weather_index: Mapping, usda_index: Mapping) -> dict`
- `make_config(output_dir: Path, data_root: Path, source_dir: Path, ...) -> dict`
- CLI subcommands: `validate`, `prepare-config`, `train`, `evaluate`

- [ ] **Step 1: Write failing tests** proving missing AG/NDVI/weather/USDA produces a structured error and that a complete `(FIPS, Year)` row passes.
- [ ] **Step 2: Run `pytest tests/test_mmst_alignment.py -q` and verify failure.**
- [ ] **Step 3: Implement deterministic indexes keyed by `(FIPS, Year)` and a report containing counts, missing keys, train/validation/test counts, and the exact target FIPS list.**
- [ ] **Step 4: Implement JSON configuration containing split years, source commit, data paths, feature normalization scope, manifest paths, and output paths.**
- [ ] **Step 5: Implement `prepare-config` and `validate` CLI commands; `train`/`evaluate` must fail clearly when the official source, dependencies, or Sentinel files are unavailable rather than silently substituting another model.**
- [ ] **Step 6: Run focused tests plus a repository-data validation.**
- [ ] **Step 7: Commit with `git add mmst_vit tests/test_mmst_alignment.py && git commit -m "feat: add MMST alignment and run configuration"`.**

### Task 4: 官方 MMST-ViT 源码与最小运行接入

**Files:**
- Modify: `mmst_vit/run.py`
- Create: `mmst_vit/official.py`
- Create: `tests/test_mmst_official.py`
- Create: `docs/mmst_vit_reproduction.md`

**Interfaces:**
- `ensure_official_source(source_dir: Path, repository: str, revision: str) -> str`
- `build_official_command(config_path: Path, mode: str) -> list[str]`
- `write_run_metadata(output_dir: Path, config: dict, source_commit: str) -> None`

- [ ] **Step 1: Write failing tests** for pinned repository/revision metadata, command construction, and refusal to run with 2022 in the training input.
- [ ] **Step 2: Run `pytest tests/test_mmst_official.py -q` and verify failure.**
- [ ] **Step 3: Implement clone/fetch with a pinned revision, source metadata, and command construction around the official entrypoint.** The adapter must not import or reimplement official model classes.
- [ ] **Step 4: Add documented commands for manifest generation, HF dry-run/download, alignment, one epoch/one validation cycle, and final 2022 evaluation.** Record unavailable credentials/files as an explicit blocker.
- [ ] **Step 5: Run all unit tests and, if local Sentinel data is complete, execute the smallest official smoke run; otherwise run the preflight and report the exact missing files.**
- [ ] **Step 6: Commit with `git add mmst_vit docs/mmst_vit_reproduction.md tests/test_mmst_official.py && git commit -m "feat: wire official MMST-ViT reproduction runner"`.**

### Task 5: 全量验证与交付检查

**Files:**
- Modify: `docs/mmst_vit_reproduction.md`
- Create: `.superpowers/sdd/progress.md` (ignored progress ledger)

- [ ] **Step 1: Run `pytest -q`.**
- [ ] **Step 2: Run manifest generation and alignment validation on the configured runtime data root.**
- [ ] **Step 3: Run Sentinel dry-run and record remote file count, bytes, and FIPS count; do not start a large download without a successful preflight.**
- [ ] **Step 4: Run official one-epoch/one-validation-cycle smoke test if prerequisites exist.**
- [ ] **Step 5: Confirm output schema includes checkpoint/config/manifests/validation metrics/test metrics/per-county predictions and confirm no 2022 key appears in training or early-stopping inputs.**
- [ ] **Step 6: Record evidence and residual blockers in the reproduction guide.**
