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

    try:
        cache = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        result["status"] = "cache_audit_unavailable"
        result["violations"].append(f"cache could not be safely loaded with weights_only=True: {error}")
        return result
    if not isinstance(cache, dict):
        result["violations"].append("cache payload is not a dict")
        return result
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


def _record_paths(record: dict) -> tuple[list[str] | None, str | None]:
    try:
        data = record["data"]
        weather = data["HRRR"]
        paths = [
            data["USDA"],
            *weather["short_term"],
            *(path for context in weather["long_term"] for path in context),
            *data["sentinel"],
        ]
        if not all(isinstance(path, str) for path in paths):
            raise TypeError("all protocol paths must be strings")
        return paths, None
    except (KeyError, TypeError, ValueError) as error:
        return None, f"malformed record paths: {error}"


def _sample_key(row: dict, *, official: bool) -> tuple[str, int, str, str]:
    if official:
        return (
            str(row["FIPS"]),
            int(row["year"]),
            str(row["state"]).strip().lower(),
            str(row["county"]).strip(),
        )
    return (
        str(row["FIPS"]),
        int(row["Year"]),
        str(row["State"]).strip().lower(),
        str(row["County"]).strip(),
    )


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
    valid_keys = set()
    official_keys = set()
    for index, row in enumerate(valid_rows):
        try:
            valid_keys.add(_sample_key(row, official=False))
        except (KeyError, TypeError, ValueError) as error:
            violations.append(f"{split} valid-no-ia row {index}: malformed identity: {error}")
    if len(valid_keys) != len(valid_rows):
        violations.append(f"{split}: valid-no-ia contains duplicate or collapsed identity keys")
    for index, record in enumerate(records):
        try:
            state = str(record["state"]).strip().lower()
            state_values.add(state)
            if state not in ALLOWED_STATE_ABBRS:
                invalid_state_count += 1
            official_keys.add(_sample_key(record, official=True))
        except (KeyError, TypeError, ValueError) as error:
            violations.append(f"{split} record {index}: malformed identity: {error}")
            continue
        try:
            short_term = record["data"]["HRRR"]["short_term"]
            long_term = record["data"]["HRRR"]["long_term"]
            sentinel = record["data"]["sentinel"]
        except (KeyError, TypeError, ValueError) as error:
            violations.append(f"{split} record {index}: missing protocol field: {error}")
            continue
        try:
            short_term_length = len(short_term)
            long_term_length = len(long_term)
            long_term_context_length = len(long_term[0]) if long_term_length else None
            sentinel_length = len(sentinel)
        except (TypeError, IndexError):
            violations.append(f"{split} record {index}: malformed protocol collection")
            continue
        if short_term_length != 6:
            violations.append(f"{split} record {index}: short_term length={len(short_term)}, expected 6")
        if long_term_length != 1 or long_term_context_length != 60:
            violations.append(f"{split} record {index}: long_term shape is not [60]")
        if sentinel_length != 4:
            violations.append(f"{split} record {index}: sentinel length={sentinel_length}, expected 4")
        paths, path_error = _record_paths(record)
        if path_error:
            violations.append(f"{split} record {index}: {path_error}")
        elif paths is not None:
            missing_paths.extend(
                str(DATA_ROOT / relative)
                for relative in paths
                if not (DATA_ROOT / relative).is_file()
            )
    if valid_keys != official_keys:
        violations.append(
            f"{split}: valid-no-ia and official identity key sets differ; "
            f"only-valid={len(valid_keys - official_keys)}, only-official={len(official_keys - valid_keys)}"
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
        "valid_identity_key_count": len(valid_keys),
        "official_identity_key_count": len(official_keys),
        "identity_keys_equal": valid_keys == official_keys,
    }


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

    assert not dataset["violations"], dataset["violations"]
    assert not cache["violations"], cache["violations"]
    assert official_violation_count == 0, official
    assert official_missing_path_count == 0, official


def test_official_audit_reports_malformed_records_without_raising():
    result = _audit_official(
        "train",
        [{"FIPS": "17001", "year": 2020, "state": "illinois", "county": "ADAMS", "data": None}],
        [{"FIPS": "17001", "Year": 2020, "State": "illinois", "County": "ADAMS"}],
    )

    assert result["identity_keys_equal"] is True
    assert any("missing protocol field" in violation for violation in result["violations"])


def test_official_audit_reports_identity_key_mismatch():
    result = _audit_official(
        "train",
        [{"FIPS": "17003", "year": 2020, "state": "illinois", "county": "ADAMS", "data": {}}],
        [{"FIPS": "17001", "Year": 2020, "State": "illinois", "County": "ADAMS"}],
    )

    assert result["identity_keys_equal"] is False
    assert any("identity key sets differ" in violation for violation in result["violations"])
