"""Task 8 staging, audit, read-only smoke test, and atomic runtime install."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "train_dataset"))
sys.path.insert(0, str(ROOT / "TFT_model"))
sys.path.insert(0, str(ROOT / "BaseLine_Model"))
sys.path.insert(0, str(ROOT))

import prepare_grid
import prepare_jsonl
from cropnet_protocol import ALLOWED_STATES, EXPECTED_CALENDAR, TIME_WINDOW, PROTOCOL_MAX_STEPS
from mmst_vit.manifest import SPLIT_YEARS, build_samples_from_shared_jsonl, split_samples, write_jsonl
from mmst_vit.official import write_official_manifests
from TFT_model import data as tft_data
from BaseLine_Model.common import data as baseline_data
from BaseLine_Model.deepcropnet.deepcropnet import prepare_dcn


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keys(rows, official=False):
    if official:
        return {(str(r["FIPS"]), int(r["year"])) for r in rows}
    return {(str(r["FIPS"]), int(r["Year"])) for r in rows}


def _validate_official_record(record: object, data_root: Path, index: int) -> list[str]:
    violations = []
    if not isinstance(record, dict):
        return [f"official record {index}: not an object"]
    try:
        if str(record["state"]).strip().lower() not in ALLOWED_STATES:
            violations.append(f"official record {index}: state outside allowlist")
        data = record["data"]
        short_term = data["HRRR"]["short_term"]
        long_term = data["HRRR"]["long_term"]
        sentinel = data["sentinel"]
        if len(short_term) != 6:
            violations.append(f"official record {index}: short_term length is not 6")
        if len(long_term) != 1 or len(long_term[0]) != 60:
            violations.append(f"official record {index}: long_term shape is not [60]")
        if len(sentinel) != 4:
            violations.append(f"official record {index}: sentinel length is not 4")
        paths = [data["USDA"], *short_term, *(p for context in long_term for p in context), *sentinel]
        for relative in paths:
            if not isinstance(relative, str):
                violations.append(f"official record {index}: missing path {relative!r}")
                continue
            candidate = (data_root / relative).resolve()
            try:
                candidate.relative_to(data_root.resolve())
            except ValueError:
                violations.append(f"official record {index}: path escapes data_root {relative!r}")
                continue
            if not candidate.is_file():
                violations.append(f"official record {index}: missing path {relative!r}")
    except (KeyError, TypeError, IndexError) as error:
        violations.append(f"official record {index}: malformed structure: {error}")
    return violations


def audit_staging(staging: Path, data_root: Path, manifest_dir: Path) -> dict:
    dataset = staging / "dataset.jsonl"
    cache_path = staging / "grid_cache.pt"
    meta_path = staging / "grid_cache_meta.json"
    rows = tft_data.load_jsonl(str(dataset))
    cache = tft_data.load_grid_cache(str(cache_path))
    expected = list(EXPECTED_CALENDAR)
    violations = []
    for index, row in enumerate(rows):
        if str(row.get("State", "")).lower() not in ALLOWED_STATES:
            violations.append(f"dataset row {index}: state outside allowlist")
        l_enc = row.get("l_enc")
        if isinstance(l_enc, bool) or not isinstance(l_enc, int) or l_enc != PROTOCOL_MAX_STEPS:
            violations.append(f"dataset row {index}: l_enc is not 168")
        if list(zip(row.get("month", []), row.get("day", []))) != expected:
            violations.append(f"dataset row {index}: calendar mismatch")
    if len(cache["entries"]) != len(rows):
        violations.append("cache and JSONL row counts differ")
    for index, entry in enumerate(cache["entries"]):
        if list(zip(entry["month"].tolist(), entry["day"].tolist())) != expected:
            violations.append(f"cache entry {index}: calendar mismatch")
    split_rows = split_samples(rows)
    root_valid_path = manifest_dir / "valid-no-ia.jsonl"
    root_valid = tft_data.load_jsonl(str(root_valid_path))
    root_keys = _keys(root_valid)
    shared_keys = _keys(rows)
    if len(root_valid) != len(rows):
        violations.append("root valid-no-ia row count differs from shared JSONL")
    if len(root_keys) != len(root_valid):
        violations.append("root valid-no-ia contains duplicate identity keys")
    if root_keys != shared_keys:
        violations.append("root valid-no-ia differs from shared JSONL")
    official = {}
    for name in SPLIT_YEARS:
        valid_path = manifest_dir / f"valid-no-ia.{name}.jsonl"
        official_path = manifest_dir / f"{name}.official.no-ia.json"
        valid = tft_data.load_jsonl(str(valid_path))
        records = json.loads(official_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            violations.append(f"{name}: official payload is not a list")
            records = []
        for record_index, record in enumerate(records):
            violations.extend(_validate_official_record(record, data_root, record_index))
        if _keys(valid) != _keys(split_rows[name]):
            violations.append(f"{name}: valid-no-ia differs from shared JSONL split")
        if _keys(valid) != _keys(records, official=True):
            violations.append(f"{name}: official differs from valid-no-ia")
        official[name] = {
            "valid_no_ia_rows": len(valid),
            "official_rows": len(records),
            "identity_keys_equal": _keys(valid) == _keys(records, official=True),
            "states": dict(Counter(str(row["State"]).lower() for row in valid)),
            "years": dict(Counter(int(row["Year"]) for row in valid)),
        }
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for field, expected_value in (("version", 4), ("time_window", TIME_WINDOW),
                                  ("days_per_month", 28), ("max_steps", 168)):
        if meta.get(field) != expected_value:
            violations.append(f"meta {field} mismatch")
    return {
        "violations": violations,
        "sha256": {
            "dataset.jsonl": sha256(dataset),
            "grid_cache.pt": sha256(cache_path),
            "grid_cache_meta.json": sha256(meta_path),
            **{
                path.name: sha256(path)
                for path in sorted(manifest_dir.iterdir())
                if path.is_file()
            },
        },
        "rows": len(rows),
        "states": dict(Counter(str(row["State"]).lower() for row in rows)),
        "years": dict(Counter(int(row["Year"]) for row in rows)),
        "split_keys": {name: sorted(_keys(rows_)) for name, rows_ in split_rows.items()},
        "splits": official,
        "cache_entries": len(cache["entries"]),
        "root_valid_no_ia_rows": len(root_valid),
    }


def smoke(staging: Path, soil_path: Path, smoke_dir: Path) -> dict:
    meta = tft_data.load_jsonl(str(staging / "dataset.jsonl"))
    cache = tft_data.load_grid_cache(str(staging / "grid_cache.pt"))
    soil = tft_data.load_county_soil(str(soil_path))
    pairs = list(zip(meta, cache["entries"]))
    grid_samples = tft_data.build_grid_samples(pairs[: min(8, len(pairs))], soil)
    baseline = baseline_data.prepare(
        out_dir=smoke_dir / "baseline", force=True,
        jsonl_path=str(staging / "dataset.jsonl"),
        grid_cache_path=str(staging / "grid_cache.pt"), soil_path=str(soil_path),
    )
    dcn = prepare_dcn(
        out_dir=smoke_dir / "dcn", force=True,
        jsonl_path=str(staging / "dataset.jsonl"),
        grid_cache_path=str(staging / "grid_cache.pt"),
    )
    dcn_reload = prepare_dcn(out_dir=smoke_dir / "dcn", force=False)
    for generated, reloaded in zip(dcn, dcn_reload):
        if generated.get("val_year") != reloaded.get("val_year") or generated.get("test_year") != reloaded.get("test_year"):
            raise ValueError("DeepCropNet cache reload metadata mismatch")
    return {"jsonl_rows": len(meta), "grid_samples": len(grid_samples),
            "baseline_splits": {name: len(baseline[name]["y"]) for name in ("train", "val", "test")},
            "deepcropnet_splits": [len(part["y_raw"]) for part in dcn],
            "deepcropnet_reload_splits": [len(part["y_raw"]) for part in dcn_reload],
            "deepcropnet_reload_metadata": [
                {"val_year": part["val_year"], "test_year": part["test_year"]}
                for part in dcn_reload
            ]}


def atomic_exchange_directories(staging: Path, active: Path) -> None:
    """Atomically exchange two directories on Linux without an absent/mixed active state."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(staging), -100, os.fsencode(active), 2) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(active))


def install_runtime_bundle(bundle: Path, active: Path) -> None:
    """Exchange one complete bundle and preserve the previous active directory."""
    active.parent.mkdir(parents=True, exist_ok=True)
    old_exists = active.exists()
    if old_exists:
        atomic_exchange_directories(bundle, active)
        backup = active.with_name("task8-backup")
        if backup.exists():
            suffix = next(index for index in range(1, 10000) if not backup.with_name(f"task8-backup.{index}").exists())
            backup = backup.with_name(f"task8-backup.{suffix}")
        try:
            os.replace(bundle, backup)
        except Exception:
            atomic_exchange_directories(bundle, active)
            raise
    else:
        try:
            os.replace(bundle, active)
        except Exception:
            if active.exists() and not bundle.exists():
                os.replace(active, bundle)
            raise


def ensure_runtime_entrypoints(runtime_root: Path, active: Path) -> None:
    """Create compatibility links once; both links resolve through one stable active directory."""
    endpoints = {
        runtime_root / "train_dataset": active / "train_dataset",
        runtime_root / "DataSrc" / "mmst_vit" / "manifests": active / "manifests",
    }
    for endpoint, target in endpoints.items():
        endpoint.parent.mkdir(parents=True, exist_ok=True)
        if endpoint.exists() or endpoint.is_symlink():
            if not endpoint.is_symlink() or endpoint.resolve() != target.resolve():
                raise RuntimeError(f"runtime entrypoint is not the stable task8 link: {endpoint}")
            continue
        os.symlink(target, endpoint, target_is_directory=True)


def _entrypoint_paths(runtime_root: Path, active: Path) -> dict[Path, Path]:
    return {
        runtime_root / "train_dataset": active / "train_dataset",
        runtime_root / "DataSrc" / "mmst_vit" / "manifests": active / "manifests",
    }


def _validate_entrypoint_state(runtime_root: Path, active: Path) -> None:
    for endpoint, target in _entrypoint_paths(runtime_root, active).items():
        if endpoint.exists() or endpoint.is_symlink():
            if not endpoint.is_symlink() or endpoint.resolve() != target.resolve():
                raise RuntimeError(f"runtime entrypoint is not the stable task8 link: {endpoint}")


def _create_missing_entrypoints(runtime_root: Path, active: Path) -> list[Path]:
    created = []
    try:
        for endpoint, target in _entrypoint_paths(runtime_root, active).items():
            endpoint.parent.mkdir(parents=True, exist_ok=True)
            if endpoint.exists() or endpoint.is_symlink():
                continue
            os.symlink(target, endpoint, target_is_directory=True)
            created.append(endpoint)
    except Exception:
        for endpoint in reversed(created):
            endpoint.unlink(missing_ok=True)
        raise
    return created


def install_runtime_bundle_with_entrypoints(bundle: Path, runtime_root: Path) -> None:
    """Install a complete bundle with rollback for post-exchange failures."""
    active = runtime_root / "task8-active"
    _validate_entrypoint_state(runtime_root, active)
    old_exists = active.exists()
    if old_exists:
        atomic_exchange_directories(bundle, active)
    else:
        os.replace(bundle, active)
    created = []
    try:
        created = _create_missing_entrypoints(runtime_root, active)
        if old_exists:
            backup = active.with_name("task8-backup")
            if backup.exists():
                suffix = next(index for index in range(1, 10000) if not backup.with_name(f"task8-backup.{index}").exists())
                backup = backup.with_name(f"task8-backup.{suffix}")
            os.replace(bundle, backup)
    except Exception:
        for endpoint in reversed(created):
            endpoint.unlink(missing_ok=True)
        if old_exists:
            atomic_exchange_directories(bundle, active)
        else:
            os.replace(active, bundle)
        raise


def audit_active_bundle(runtime_root: Path, data_root: Path) -> dict:
    active = runtime_root / "task8-active"
    return audit_staging(active / "train_dataset", data_root, active / "manifests")


def generate_and_install(runtime_root: Path, data_root: Path) -> dict:
    active = runtime_root / "task8-active"
    active_dataset = active / "train_dataset"
    active_manifest = active / "manifests"
    staging = Path(tempfile.mkdtemp(prefix="task8-", dir=str(runtime_root)))
    smoke_dir = Path(tempfile.mkdtemp(prefix="task8-smoke-", dir=str(runtime_root)))
    try:
        dataset = staging / "dataset.jsonl"
        cache = staging / "grid_cache.pt"
        meta = staging / "grid_cache_meta.json"
        prepare_jsonl.process_all(output_path=str(dataset), data_dir=str(data_root / "cropnet_dataset" / "data"))
        prepare_grid.process(str(dataset), str(cache), str(meta), str(data_root / "cropnet_dataset" / "data"))
        rows = build_samples_from_shared_jsonl(dataset)
        splits = split_samples(rows)
        manifest_staging = staging / "manifests"
        write_jsonl(manifest_staging / "valid-no-ia.jsonl", rows)
        for name, split in splits.items():
            write_jsonl(manifest_staging / f"valid-no-ia.{name}.jsonl", split)
        write_official_manifests(rows, manifest_staging, data_root)
        audit = audit_staging(staging, data_root, manifest_staging)
        if audit["violations"]:
            raise RuntimeError("staging audit failed: " + "; ".join(audit["violations"][:10]))
        smoke_result = smoke(staging, data_root / "soil_dataset" / "county_soil.json", smoke_dir)
        audit["smoke"] = smoke_result
        install_dir = staging / "bundle"
        install_dataset = install_dir / "train_dataset"
        install_manifest = install_dir / "manifests"
        install_dataset.mkdir(parents=True)
        install_manifest.mkdir(parents=True)
        for name in ("dataset.jsonl", "grid_cache.pt", "grid_cache_meta.json"):
            os.replace(staging / name, install_dataset / name)
        for name in ("valid-no-ia.jsonl", "valid-no-ia.train.jsonl", "valid-no-ia.val.jsonl", "valid-no-ia.test.jsonl",
                     "train.official.no-ia.json", "val.official.no-ia.json", "test.official.no-ia.json"):
            os.replace(staging / "manifests" / name, install_manifest / name)
        install_runtime_bundle_with_entrypoints(install_dir, runtime_root)
        audit["installed"] = True
        audit["source_roots"] = {
            "sentinel": str(data_root / "mmst_vit" / "download"),
            "usda": str(data_root / "cropnet_dataset" / "data" / "usda_corn"),
            "weather": str(data_root / "cropnet_dataset" / "data" / "weather"),
            "soil": str(data_root / "soil_dataset"),
            "mutation_policy": "read-only inputs; no source files modified or deleted",
        }
        return audit
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(smoke_dir, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=Path("/data/raid0/hqx/Product_model_runtime"))
    parser.add_argument("--data-root", type=Path, default=Path("/data/raid0/hqx/Product_model_runtime/DataSrc"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / ".superpowers" / "sdd" / "task-8-report.json")
    args = parser.parse_args(argv)
    if args.audit_only:
        result = audit_active_bundle(args.runtime_root, args.data_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = generate_and_install(args.runtime_root, args.data_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
