"""迁移验收与旧运行时目录删除闸门：只报告，不删除任何文件。

  python train_dataset/migration_check.py

仅当全部门槛通过时输出 READY_TO_DELETE_OLD_RUNTIME=true；删除需用户另行明确批准后人工执行。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_dataset.dataset import DATA_ROOT, DATASET_PATH, YEARS, check_dataset_file  # noqa: E402

OLD_RUNTIME = Path("/data/raid0/hqx/Product_model_runtime")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_LINK = PROJECT_ROOT / "train_dataset" / "dataset.jsonl"
MODALITIES = {
    "USDA Crop Dataset": "usda", "WRF-HRRR Computed Dataset": "hrrr",
    "Sentinel-2 Imagery": "sentinel", "soil": "soil", "census": "census",
}
TEST_MODULES = ("tests.test_cropnet_model_interfaces", "tests.test_cropnet_real_smoke")
AUDIT_NAME = "download_audit.jsonl"


def count_files(root: Path) -> dict:
    """按模态统计文件数和字节数；不跟随符号链接。"""
    out = {"total": {"files": 0, "bytes": 0}}
    for dirpath, _, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root).parts
        # 旧目录多一层 DataSrc/，按路径中任一层目录名识别模态
        modality = next((MODALITIES[p] for p in rel if p in MODALITIES), "other")
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            size = path.stat().st_size
            for key in (modality, "total"):
                bucket = out.setdefault(key, {"files": 0, "bytes": 0})
                bucket["files"] += 1
                bucket["bytes"] += size
    return out


def run_interface_tests() -> dict:
    result = subprocess.run([sys.executable, "-m", "unittest", *TEST_MODULES],
                            cwd=PROJECT_ROOT, capture_output=True, text=True)
    return {"command": f"python -m unittest {' '.join(TEST_MODULES)}",
            "returncode": result.returncode, "tail": result.stderr.strip().splitlines()[-3:]}


def compare_year_county_sets(jsonl_path: Path, *, years=YEARS) -> dict:
    """独立扫描每年的 FIPS 集合，并报告相对首年的双向差集。"""
    years = tuple(years)
    by_year = {year: set() for year in years}
    with open(jsonl_path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                year, fips = row["year"], row["fips"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{jsonl_path}:{lineno}: cannot read year/FIPS: {exc}") from None
            if year in by_year:
                by_year[year].add(fips)
    reference_year = years[0]
    reference = by_year[reference_year]
    report = {}
    for year in years:
        current = by_year[year]
        report[str(year)] = {
            "count": len(current),
            "missing_from_year": sorted(reference - current),
            "extra_in_year": sorted(current - reference),
        }
    return {
        "reference_year": reference_year,
        "all_equal": all(by_year[year] == reference for year in years),
        "years": report,
    }


def summarize_download_audit(path: Path) -> dict:
    candidates, retained, reasons = set(), set(), Counter()
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                fips, status, reason = row["fips"], row["status"], row["reason"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid audit row: {exc}") from None
            candidates.add(fips)
            if status == "retained":
                retained.add(fips)
            elif status == "excluded":
                reasons.update(
                    part.strip().split(":")[0]
                    for part in reason.split(";")
                    if part.strip() and not part.strip().startswith("county ")
                )
    return {
        "path": str(path),
        "candidates": len(candidates),
        "retained_counties": len(retained),
        "excluded_reasons": dict(reasons.most_common()),
    }


def sample_hdf5_references(jsonl_path: Path, *, years=YEARS) -> dict:
    """每年抽取第一条样本，记录其遥感月引用数量；全量校验另行执行。"""
    selected = {}
    with open(jsonl_path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                year = row["year"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{jsonl_path}:{lineno}: invalid sample for HDF5 sampling: {exc}") from None
            if year in years and year not in selected:
                selected[year] = row
    samples = []
    for year in years:
        row = selected.get(year)
        if row is None:
            samples.append({"year": year, "sample_id": None, "remote_month_references": 0, "ok": False})
            continue
        grids = row.get("grids", [])
        month_counts = [len(grid.get("remote_sensing", [])) for grid in grids]
        samples.append({
            "year": year,
            "sample_id": row.get("sample_id"),
            "remote_month_references": sum(month_counts),
            "grid_count": len(grids),
            "ok": bool(grids) and len(set(month_counts)) == 1 and month_counts[0] > 0,
        })
    return {
        "sample_count": len(samples),
        "years": [sample["year"] for sample in samples],
        "samples": samples,
        "ok": all(sample["ok"] for sample in samples),
    }


def migration_report(old_root: Path, new_root: Path, jsonl_path: Path, *, link: Path = PROJECT_LINK,
                     expected_target: Path = DATASET_PATH, run_tests=run_interface_tests) -> dict:
    old_root, new_root, jsonl_path = Path(old_root), Path(new_root), Path(jsonl_path)
    checks, report = {}, {"old_root": str(old_root), "new_root": str(new_root), "jsonl": str(jsonl_path)}

    checks["new_root_exists"] = new_root.is_dir()
    checks["jsonl_exists"] = jsonl_path.is_file()
    checks["symlink_ok"] = Path(link).is_symlink() and Path(os.readlink(link)) == Path(expected_target)
    report["symlink"] = {"link": str(link), "target": os.readlink(link) if Path(link).is_symlink() else None}
    report["old_root_status"] = "present" if old_root.is_dir() else "missing_unable_to_record_file_counts"

    audit_path = new_root / AUDIT_NAME
    if audit_path.is_file():
        report["download_audit"] = summarize_download_audit(audit_path)
        checks["download_audit_recorded"] = True
    else:
        report["download_audit"] = {"path": str(audit_path), "status": "missing"}
        checks["download_audit_recorded"] = False

    if checks["jsonl_exists"]:
        digest = hashlib.sha256()
        with open(jsonl_path, "rb") as handle:
            while chunk := handle.read(16 << 20):
                digest.update(chunk)
        report["jsonl_sha256"] = digest.hexdigest()
        report["jsonl_bytes"] = jsonl_path.stat().st_size
        try:
            report["dataset"] = check_dataset_file(jsonl_path, check_files=True, data_root=new_root)
            report["year_county_sets"] = compare_year_county_sets(jsonl_path)
            report["hdf5_sampling"] = sample_hdf5_references(jsonl_path)
            checks["schema_ok"] = checks["hdf5_refs_ok"] = True
            checks["year_county_sets_equal"] = report["year_county_sets"]["all_equal"]
            checks["hdf5_sampling_ok"] = report["hdf5_sampling"]["ok"]
        except ValueError as exc:
            report["dataset_error"] = str(exc)
            checks["schema_ok"] = checks["hdf5_refs_ok"] = False
            checks["year_county_sets_equal"] = False
            checks["hdf5_sampling_ok"] = False
    else:
        checks["schema_ok"] = checks["hdf5_refs_ok"] = checks["year_county_sets_equal"] = False
        checks["hdf5_sampling_ok"] = False

    tests = run_tests()
    report["tests"] = tests
    checks["interface_tests_ok"] = bool(tests.get("returncode") == 0)

    counts = {}
    for name, root in (("old", old_root), ("new", new_root)):
        counts[name] = count_files(root) if root.is_dir() else None
    report["file_counts"] = counts
    checks["file_counts_recorded"] = all(v is not None for v in counts.values())

    report["checks"] = checks
    report["ready_to_delete"] = all(checks.values())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-root", type=Path, default=OLD_RUNTIME)
    parser.add_argument("--new-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--jsonl", type=Path, default=PROJECT_LINK)
    args = parser.parse_args(argv)
    report = migration_report(args.old_root, args.new_root, args.jsonl)
    text = json.dumps(report, indent=1, ensure_ascii=False)
    print(text)
    if args.new_root.is_dir():
        out = args.new_root / "migration_report.json"
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, out)
    print(f"READY_TO_DELETE_OLD_RUNTIME={'true' if report['ready_to_delete'] else 'false'}")
    return 0 if report["ready_to_delete"] else 1


if __name__ == "__main__":
    sys.exit(main())
