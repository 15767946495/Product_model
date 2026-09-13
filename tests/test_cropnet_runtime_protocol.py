"""Read-only audit of the current CropNet and MMST-ViT runtime artifacts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cropnet_protocol import ALLOWED_STATES, DAYS_PER_MONTH, PROTOCOL_MAX_STEPS, TIME_WINDOW  # noqa: E402


# Official manifests store the parsed state as a full lowercase state value.
ALLOWED_STATE_ABBRS = ALLOWED_STATES
RUNTIME_ROOT = Path(os.environ.get("CROPNET_RUNTIME_ROOT", "/data/raid0/hqx/Product_model_runtime"))
DATASET_ROOT = Path(os.environ.get("CROPNET_RUNTIME_DATASET_ROOT", RUNTIME_ROOT / "train_dataset"))
MMST_ROOT = Path(os.environ.get("MMST_RUNTIME_ROOT", RUNTIME_ROOT / "DataSrc" / "mmst_vit"))
MANIFEST_DIR = Path(os.environ.get("MMST_MANIFEST_DIR", MMST_ROOT / "manifests"))
DATA_ROOT = Path(os.environ.get("MMST_DATA_ROOT", RUNTIME_ROOT / "DataSrc"))
REPORT_PATH = Path(
    os.environ.get(
        "CROPNET_RUNTIME_AUDIT_REPORT",
        ROOT / ".superpowers" / "sdd" / "task-7-report.md",
    )
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _path_status(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _audit_jsonl(rows: list[dict]) -> dict:
    violations = []
    for index, row in enumerate(rows):
        state = str(row.get("State", "")).strip().lower()
        if state not in ALLOWED_STATES:
            violations.append(f"dataset row {index}: parsed State={state!r} is outside the eight-state allowlist")
        day = row.get("day")
        l_enc = row.get("l_enc")
        if not isinstance(day, list) or len(day) != l_enc:
            violations.append(f"dataset row {index}: len(day)={len(day) if isinstance(day, list) else None} != l_enc={l_enc}")
        elif any(not isinstance(value, int) or not 1 <= value <= DAYS_PER_MONTH for value in day):
            violations.append(f"dataset row {index}: parsed day values are outside 1..28")
        month = row.get("month")
        if not isinstance(month, list) or len(month) != l_enc:
            violations.append(f"dataset row {index}: len(month)={len(month) if isinstance(month, list) else None} != l_enc={l_enc}")
        elif any(not isinstance(value, int) or not 4 <= value <= 9 for value in month):
            violations.append(f"dataset row {index}: parsed month values are outside 4..9")
        if l_enc != PROTOCOL_MAX_STEPS:
            violations.append(f"dataset row {index}: l_enc={l_enc}, expected {PROTOCOL_MAX_STEPS}")
    return {"row_count": len(rows), "violations": violations}


def _audit_cache(path: Path, row_count: int) -> dict:
    result = {"path": str(path), "status": _path_status(path), "violations": []}
    if not path.is_file():
        result["status"] = "missing_shared_cache_after_task6_cleanup"
        return result

    import torch

    cache = torch.load(path, map_location="cpu", weights_only=False)
    for field, expected in (("version", 4), ("max_steps", 168), ("time_window", TIME_WINDOW)):
        if cache.get(field) != expected:
            result["violations"].append(f"cache {field}={cache.get(field)!r}, expected {expected!r}")
    entries = cache.get("entries")
    if not isinstance(entries, list):
        result["violations"].append("cache entries is not a list")
    elif len(entries) != row_count:
        result["violations"].append(f"cache entries={len(entries)}, dataset rows={row_count}")
    else:
        for index, entry in enumerate(entries):
            if entry.get("l_enc") != PROTOCOL_MAX_STEPS:
                result["violations"].append(f"cache entry {index}: l_enc is not 168")
            for field, low, high in (("month", 4, 9), ("day", 1, 28)):
                values = entry.get(field)
                values = values.tolist() if hasattr(values, "tolist") else values
                if not isinstance(values, list) or len(values) != PROTOCOL_MAX_STEPS:
                    result["violations"].append(f"cache entry {index}: {field} is not 168 values")
                elif any(not low <= value <= high for value in values):
                    result["violations"].append(f"cache entry {index}: {field} outside {low}..{high}")
    return result


def _record_paths(record: dict) -> list[str]:
    weather = record["data"]["HRRR"]
    return [
        record["data"]["USDA"],
        *weather["short_term"],
        *(path for context in weather["long_term"] for path in context),
        *record["data"]["sentinel"],
    ]


def _audit_official(split: str, records: list[dict], valid_rows: list[dict]) -> dict:
    violations = []
    state_values = set()
    invalid_state_count = 0
    valid_state_values = set()
    invalid_valid_state_count = 0
    for index, row in enumerate(valid_rows):
        state = str(row.get("State", "")).strip().lower()
        valid_state_values.add(state)
        if state not in ALLOWED_STATES:
            invalid_valid_state_count += 1
    missing_paths = []
    if len(records) != len(valid_rows):
        violations.append(f"{split}: official records={len(records)}, valid-no-ia rows={len(valid_rows)}")
    for index, record in enumerate(records):
        state = str(record.get("state", "")).strip().lower()
        state_values.add(state)
        if state not in ALLOWED_STATE_ABBRS:
            invalid_state_count += 1
        try:
            short_term = record["data"]["HRRR"]["short_term"]
            long_term = record["data"]["HRRR"]["long_term"]
            sentinel = record["data"]["sentinel"]
        except (KeyError, TypeError) as error:
            violations.append(f"{split} record {index}: missing protocol field: {error}")
            continue
        if len(short_term) != 6:
            violations.append(f"{split} record {index}: short_term length={len(short_term)}, expected 6")
        if len(long_term) != 1 or len(long_term[0]) != 60:
            violations.append(f"{split} record {index}: long_term shape is not [60]")
        if len(sentinel) != 4:
            violations.append(f"{split} record {index}: sentinel length={len(sentinel)}, expected 4")
        missing_paths.extend(
            str(DATA_ROOT / relative)
            for relative in _record_paths(record)
            if not (DATA_ROOT / relative).is_file()
        )
    return {
        "record_count": len(records),
        "valid_no_ia_count": len(valid_rows),
        "violations": violations,
        "missing_paths": sorted(set(missing_paths)),
        "parsed_state_values": sorted(state_values),
        "invalid_state_count": invalid_state_count,
        "valid_parsed_state_values": sorted(valid_state_values),
        "invalid_valid_state_count": invalid_valid_state_count,
    }


def _write_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Task 7 Runtime Protocol Audit",
        "",
        "本报告由只读审计测试生成；未调用数据生成、缓存生成、训练或推理入口。",
        "",
        "## 输入与状态",
        f"- runtime root: `{RUNTIME_ROOT}`",
        f"- dataset JSONL: `{report['dataset']['path']}` ({report['dataset']['status']})",
        f"- grid cache: `{report['cache']['path']}` ({report['cache']['status']})",
        f"- valid-no-ia / official manifest directory: `{MANIFEST_DIR}`",
        f"- shared cache conclusion: **{report['shared_cache_conclusion']}**",
        "",
        "## 协议审计",
        f"- dataset rows: `{report['dataset'].get('row_count', 0)}`",
        f"- dataset protocol violations: `{len(report['dataset'].get('violations', []))}`",
        f"- cache protocol violations: `{len(report['cache'].get('violations', []))}`",
        f"- official manifest protocol violations: `{report['official_violation_count']}`",
        f"- missing official data/Sentinel paths: `{report['official_missing_path_count']}`",
        "",
        "## 结论分类",
        "- **协议违规**：仅表示已存在并成功解析的产物违反八州、日历、步数、cache 对齐、manifest 结构或路径存在性约束。",
        "- **共享缓存已清理**：dataset/grid cache 缺失单独记录，不为审计重新生成，也不将缺失缓存误报为协议违规。",
        "",
        "## 详细问题",
    ]
    details = report["dataset"].get("violations", []) + report["cache"].get("violations", [])
    for split, result in report["official"].items():
        details.extend(result["violations"])
        if result["invalid_state_count"]:
            details.append(
                f"{split}: {result['invalid_state_count']} parsed state values are outside the abbreviation allowlist; "
                f"values={result['parsed_state_values']}"
            )
        if result["invalid_valid_state_count"]:
            details.append(
                f"{split}: {result['invalid_valid_state_count']} valid-no-ia parsed State values are outside the full-name allowlist; "
                f"values={result['valid_parsed_state_values']}"
            )
        details.extend(f"{split}: missing path `{path}`" for path in result["missing_paths"])
    lines.extend(f"- {item}" for item in details) if details else lines.append("- 无")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_current_runtime_protocol_is_audited_read_only():
    dataset_path = DATASET_ROOT / "dataset.jsonl"
    cache_path = DATASET_ROOT / "grid_cache.pt"
    dataset = {"path": str(dataset_path), "status": _path_status(dataset_path), "violations": []}
    rows = []
    if dataset_path.is_file():
        rows = _read_jsonl(dataset_path)
        dataset.update(_audit_jsonl(rows))
    else:
        dataset["status"] = "missing_shared_cache_after_task6_cleanup"

    cache = _audit_cache(cache_path, len(rows))
    official = {}
    official_violation_count = 0
    official_missing_path_count = 0
    for split in ("train", "val", "test"):
        valid_path = MANIFEST_DIR / f"valid-no-ia.{split}.jsonl"
        official_path = MANIFEST_DIR / f"{split}.official.no-ia.json"
        valid_rows = _read_jsonl(valid_path) if valid_path.is_file() else []
        records = _read_json(official_path) if official_path.is_file() else []
        result = _audit_official(split, records, valid_rows)
        result.update({"valid_path": str(valid_path), "official_path": str(official_path)})
        if not valid_path.is_file():
            result["violations"].append(f"{split}: valid-no-ia manifest missing")
        if not official_path.is_file():
            result["violations"].append(f"{split}: official.no-ia manifest missing")
        official[split] = result
        official_violation_count += (
            len(result["violations"])
            + result["invalid_state_count"]
            + result["invalid_valid_state_count"]
        )
        official_missing_path_count += len(result["missing_paths"])

    report = {
        "dataset": dataset,
        "cache": cache,
        "official": official,
        "official_violation_count": official_violation_count,
        "official_missing_path_count": official_missing_path_count,
        "shared_cache_conclusion": (
            "旧共享 dataset/grid cache 已清理；当前运行时缺失属于产物存在性状态，不是由审计生成或修复"
            if not dataset_path.is_file() or not cache_path.is_file()
            else "共享 dataset/grid cache 存在并已审计"
        ),
    }
    _write_report(report)

    assert not dataset["violations"], dataset["violations"]
    assert not cache["violations"], cache["violations"]
    assert official_violation_count == 0, official
    assert official_missing_path_count == 0, official
