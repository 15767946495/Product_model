"""下载并按官方文件清单只读验证 Tiny CropNet 快照。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, cast

import h5py
from huggingface_hub import HfApi, snapshot_download


DEFAULT_REPO_ID = "fudong03/Tiny-CropNet"
DEFAULT_REVISION = "main"
DEFAULT_LOCAL_DIR = Path("/data/raid0/hqx/Tiny-CropNet.download")
DEFAULT_REPORT = Path("/data/raid0/hqx/Tiny-CropNet.download.validation.json")
DEFAULT_INVENTORY = Path("/data/raid0/hqx/Tiny-CropNet.download.inventory.json")
OFFICIAL_FILE_COUNT = 409
OFFICIAL_TOTAL_BYTES = 358054151828
OFFICIAL_CATEGORY_COUNTS = {
    "hdf5": 96,
    "wrf_csv": 288,
    "usda_csv": 24,
    "root_metadata": 1,
}
TOP_LEVEL_DIRECTORIES = {
    "Sentinel-2 Imagery",
    "WRF-HRRR Computed Dataset",
    "USDA Crop Dataset",
}
WRF_COLUMNS = {
    "Year", "Month", "Day", "Daily/Monthly", "State", "County", "FIPS Code",
    "Grid Index", "Lat (llcrnr)", "Lon (llcrnr)", "Lat (urcrnr)", "Lon (urcrnr)",
    "Avg Temperature (K)", "Max Temperature (K)", "Min Temperature (K)",
    "Precipitation (kg m**-2)", "Relative Humidity (%)", "Wind Gust (m s**-1)",
    "Wind Speed (m s**-1)", "U Component of Wind (m s**-1)",
    "V Component of Wind (m s**-1)", "Downward Shortwave Radiation Flux (W m**-2)",
    "Vapor Pressure Deficit (kPa)",
}
USDA_COMMON_COLUMNS = {
    "commodity_desc", "reference_period_desc", "year", "state_ansi", "state_name",
    "county_ansi", "county_name", "asd_code", "asd_desc", "domain_desc",
    "source_desc", "agg_level_desc",
}


def _category(path: str) -> str:
    if path.startswith("Sentinel-2 Imagery/data/") and path.endswith(".h5"):
        return "hdf5"
    if path.startswith("WRF-HRRR Computed Dataset/data/") and path.endswith(".csv"):
        return "wrf_csv"
    if path.startswith("USDA Crop Dataset/data/") and path.endswith(".csv"):
        return "usda_csv"
    return "root_metadata" if "/" not in path else "other"


def _category_counts(paths: Iterable[str]) -> dict[str, int]:
    counts = {"hdf5": 0, "wrf_csv": 0, "usda_csv": 0, "root_metadata": 0, "other": 0}
    for path in paths:
        counts[_category(path)] += 1
    return {name: count for name, count in counts.items() if count}


def _parent_directories(paths: Iterable[str]) -> list[str]:
    directories = set()
    for value in paths:
        parent = Path(value).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return sorted(directories)


def _inventory_errors(inventory: dict) -> list[str]:
    errors = []
    files = inventory.get("files")
    if not isinstance(files, list):
        return ["files must be a list"]
    raw_paths = [item.get("path") for item in files if isinstance(item, dict)]
    if len(raw_paths) != len(files) or any(not isinstance(path, str) for path in raw_paths):
        return ["every file requires a string path"]
    paths = cast(list[str], raw_paths)
    sizes = [item.get("size") for item in files]
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in sizes):
        return ["every file requires a non-negative integer size"]
    unique_paths = set(paths)
    computed_bytes = sum(sizes)
    computed_counts = _category_counts(unique_paths)
    computed_directories = _parent_directories(unique_paths)
    expected_counts = dict(OFFICIAL_CATEGORY_COUNTS)
    if len(files) != OFFICIAL_FILE_COUNT or len(unique_paths) != OFFICIAL_FILE_COUNT:
        errors.append(f"file count must be {OFFICIAL_FILE_COUNT}")
    if computed_bytes != OFFICIAL_TOTAL_BYTES:
        errors.append(f"total bytes must be {OFFICIAL_TOTAL_BYTES}")
    if computed_counts != expected_counts:
        errors.append(f"category counts must be {expected_counts}")
    root_paths = sorted(path for path in unique_paths if "/" not in path)
    if root_paths != [".gitattributes"]:
        errors.append("root metadata must be exactly .gitattributes")
    declared = {
        "expected_file_count": len(files),
        "expected_total_bytes": computed_bytes,
        "expected_category_counts": computed_counts,
        "expected_directories": computed_directories,
    }
    for name, value in declared.items():
        if inventory.get(name) != value:
            errors.append(f"{name} does not match files")
    return errors


def resolve_official_inventory(
    repo_id: str,
    requested_revision: str,
    *,
    api: HfApi | None = None,
) -> dict:
    """解析 revision 为 commit SHA，并取得带大小和 blob oid 的完整文件清单。"""
    info = (api or HfApi()).dataset_info(
        repo_id,
        revision=requested_revision,
        files_metadata=True,
    )
    if not info.sha:
        raise ValueError(f"无法解析 revision: {requested_revision}")
    siblings = list(info.siblings or [])
    missing_sizes = [sibling.rfilename for sibling in siblings if sibling.size is None]
    if missing_sizes:
        raise ValueError(f"官方 inventory 文件缺少 size metadata: {missing_sizes}")
    files = sorted(
        (
            {
                "path": sibling.rfilename,
                "size": cast(int, sibling.size),
                "blob_oid": sibling.blob_id,
            }
            for sibling in siblings
        ),
        key=lambda item: item["path"],
    )
    inventory = {
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": info.sha,
        "expected_file_count": len(files),
        "expected_total_bytes": sum(item["size"] for item in files),
        "expected_category_counts": _category_counts(item["path"] for item in files),
        "expected_directories": _parent_directories(item["path"] for item in files),
        "files": files,
    }
    errors = _inventory_errors(inventory)
    if errors:
        raise ValueError("官方 inventory 不符合固定清单: " + "; ".join(errors))
    return inventory


def download_snapshot(repo_id: str, resolved_revision: str, local_dir: Path) -> Path:
    """使用 HF local_dir cache 在同一目录继续已中断的快照下载。"""
    return Path(snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        local_dir=str(local_dir),
    ))


def _is_hf_local_cache(relative: Path) -> bool:
    return relative.parts[:2] == (".cache", "huggingface")


def _actual_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    result = {}
    for path in root.rglob("*"):
        if path.is_file():
            relative = path.relative_to(root)
            if not _is_hf_local_cache(relative):
                result[relative.as_posix()] = path
    return result


def _actual_directories(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    directories = set()
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        relative = path.relative_to(root)
        if (
            relative.parts == (".cache",)
            and (root / ".cache" / "huggingface").is_dir()
        ) or _is_hf_local_cache(relative):
            continue
        directories.add(relative.as_posix())
    return directories


def _csv_columns(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return set(next(csv.reader(handle)))


def _usda_columns(relative: str) -> set[str]:
    crop = relative.split("/")[2]
    if crop == "Cotton":
        measures = {
            "PRODUCTION, MEASURED IN 480 LB BALES",
            "YIELD, MEASURED IN LB / ACRE",
        }
    else:
        measures = {
            "PRODUCTION, MEASURED IN BU",
            "YIELD, MEASURED IN BU / ACRE",
        }
    return USDA_COMMON_COLUMNS | measures


def _check_csv(path: Path, required: set[str]) -> str | None:
    try:
        missing = sorted(required - _csv_columns(path))
        return f"missing columns: {missing}" if missing else None
    except (OSError, UnicodeError, csv.Error, StopIteration) as error:
        return f"{type(error).__name__}: {error}"


def _check_hdf5(path: Path) -> str | None:
    try:
        with h5py.File(path, "r") as handle:
            counties = list(handle.keys())
            if not counties:
                return "no county groups"
            dates = list(handle[counties[0]].keys())
            if not dates:
                return "no date groups"
            missing = sorted({"data", "coordinates"} - set(handle[counties[0]][dates[0]].keys()))
            return f"missing datasets: {missing}" if missing else None
    except (OSError, KeyError, TypeError) as error:
        return f"{type(error).__name__}: {error}"


def validate_snapshot(root: Path, inventory: dict) -> dict:
    """依据不可变官方 inventory 对本地快照执行只读精确验证。"""
    root = Path(root)
    inventory_errors = _inventory_errors(inventory)
    raw_files = inventory.get("files")
    safe_files = raw_files if isinstance(raw_files, list) else []
    expected_entries = {
        item["path"]: item
        for item in safe_files
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and isinstance(item.get("size"), int)
        and not isinstance(item.get("size"), bool)
    }
    actual_files = _actual_files(root)
    expected_paths = set(expected_entries)
    actual_paths = set(actual_files)
    expected_directories = set(inventory.get("expected_directories", []))
    actual_directories = _actual_directories(root)
    missing_directories = sorted(expected_directories - actual_directories)
    extra_directories = sorted(actual_directories - expected_directories)
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    size_mismatches = []
    for relative in sorted(expected_paths & actual_paths):
        expected_size = int(expected_entries[relative]["size"])
        actual_size = actual_files[relative].stat().st_size
        if actual_size != expected_size:
            size_mismatches.append({"path": relative, "expected": expected_size, "actual": actual_size})

    expected_counts = _category_counts(expected_paths)
    actual_counts = _category_counts(actual_paths)
    category_ok = {
        name: expected_counts.get(name, 0) == count and actual_counts.get(name, 0) == count
        for name, count in OFFICIAL_CATEGORY_COUNTS.items()
    }
    actual_top_levels = {
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and not (path.name == ".cache" and (path / "huggingface").is_dir())
    } if root.is_dir() else set()
    missing_top_levels = sorted(TOP_LEVEL_DIRECTORIES - actual_top_levels)
    extra_top_levels = sorted(actual_top_levels - TOP_LEVEL_DIRECTORIES)
    schema_errors = {}
    checked = {"hdf5": 0, "wrf_csv": 0, "usda_csv": 0}
    for relative in sorted(expected_paths & actual_paths):
        category = _category(relative)
        error = None
        if category == "hdf5":
            checked[category] += 1
            error = _check_hdf5(actual_files[relative])
        elif category == "wrf_csv":
            checked[category] += 1
            error = _check_csv(actual_files[relative], WRF_COLUMNS)
        elif category == "usda_csv":
            checked[category] += 1
            error = _check_csv(actual_files[relative], _usda_columns(relative))
        if error:
            schema_errors[relative] = error

    expected_bytes = sum(int(item["size"]) for item in expected_entries.values())
    actual_bytes = sum(path.stat().st_size for path in actual_files.values())
    checks = [
        {
            "name": "inventory_self_consistency",
            "ok": not inventory_errors,
            "errors": inventory_errors,
        },
        {
            "name": "exact_directory_inventory",
            "ok": not missing_directories and not extra_directories,
            "missing": missing_directories,
            "extra": extra_directories,
        },
        {
            "name": "top_level_directories",
            "ok": not missing_top_levels and not extra_top_levels,
            "missing": missing_top_levels,
            "extra": extra_top_levels,
        },
        {"name": "exact_file_inventory", "ok": not missing and not extra},
        {"name": "file_sizes", "ok": not size_mismatches},
        {"name": "official_category_counts", "ok": all(category_ok.values()), "categories": category_ok},
        {"name": "all_file_schemas", "ok": not schema_errors},
    ]
    return {
        "repo_id": inventory.get("repo_id"),
        "requested_revision": inventory.get("requested_revision"),
        "resolved_revision": inventory.get("resolved_revision"),
        "local_dir": str(root),
        "expected_file_count": len(expected_entries),
        "actual_file_count": len(actual_files),
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "missing": missing,
        "extra": extra,
        "missing_directories": missing_directories,
        "extra_directories": extra_directories,
        "size_mismatches": size_mismatches,
        "expected_category_counts": expected_counts,
        "actual_category_counts": actual_counts,
        "checked_files": checked,
        "schema_errors": schema_errors,
        "checks": checks,
        "valid": all(item["ok"] for item in checks),
    }


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _outside_local_dir(path: Path, local_dir: Path, name: str) -> None:
    try:
        path.resolve().relative_to(local_dir.resolve())
    except ValueError:
        return
    raise ValueError(f"{name} 必须位于 local_dir 外部: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "validate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
        subparser.add_argument("--revision", default=DEFAULT_REVISION)
        subparser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
        subparser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
        subparser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _outside_local_dir(args.inventory, args.local_dir, "inventory")
    _outside_local_dir(args.report, args.local_dir, "report")
    if args.command == "download":
        inventory = resolve_official_inventory(args.repo_id, args.revision)
        write_json(args.inventory, inventory)
        local_dir = download_snapshot(args.repo_id, inventory["resolved_revision"], args.local_dir)
    else:
        if args.inventory.is_file():
            inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        else:
            inventory = resolve_official_inventory(args.repo_id, args.revision)
        local_dir = args.local_dir
    report = validate_snapshot(local_dir, inventory)
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
