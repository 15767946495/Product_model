"""Scoped Hugging Face Sentinel-2 manifest and resumable downloader."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote

HF_BASE = "https://huggingface.co/datasets/CropNet/CropNet/resolve/main/"
HF_API_BASE = "https://huggingface.co/api/datasets/CropNet/CropNet/tree/main/"
FIPS_RE = re.compile(r"(?:^|[_/])(?P<fips>\d{5})(?:[_./]|$)")
YEAR_RE = re.compile(r"(?:^|[/_-])(?P<year>20(?:17|18|19|20|21|22))(?:[/_.-]|$)")

DEFAULT_RUNTIME_ROOT = Path("/data/raid0/hqx/Product_model_runtime/DataSrc/mmst_vit")
DEFAULT_TARGET_FIPS = DEFAULT_RUNTIME_ROOT / "manifests/target_fips.txt"
DEFAULT_REMOTE_MANIFEST = DEFAULT_RUNTIME_ROOT / "manifests/sentinel.jsonl"
DEFAULT_URL_MANIFEST = DEFAULT_RUNTIME_ROOT / "manifests/sentinel_urls.jsonl"
DEFAULT_DOWNLOAD_ROOT = DEFAULT_RUNTIME_ROOT / "download"
DEFAULT_EXTRACT_ROOT = DEFAULT_RUNTIME_ROOT / "county"
DEFAULT_OSS_ENDPOINT = "https://oss-ap-southeast-1.aliyuncs.com"
DEFAULT_OSS_BUCKET = "alisg-tec-chi-pai-oss-prod-01"
DEFAULT_OSS_ACCESS_KEY_ID = ""
DEFAULT_OSS_ACCESS_KEY_SECRET = ""

FIVE_STATE_NAMES = frozenset({"illinois", "iowa", "louisiana", "mississippi", "new york"})
FIVE_STATE_ANSI = {"17", "19", "22", "28", "36"}
FIVE_STATE_YEARS = tuple(range(2017, 2023))
STATE_NAME_TO_ANSI = {
    "illinois": "17", "iowa": "19", "louisiana": "22", "mississippi": "28", "new york": "36",
}
STATE_ABBR_TO_ANSI = {"IL": "17", "IA": "19", "LA": "22", "MS": "28", "NY": "36"}


def _matched_fips(path: str, target_fips: set[str]) -> set[str]:
    found = {match.group("fips") for match in FIPS_RE.finditer(path)}
    return found & target_fips


def normalize_state_name(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _parsed_state_ansi(value: object) -> str:
    text = normalize_state_name(value)
    if text in STATE_NAME_TO_ANSI:
        return STATE_NAME_TO_ANSI[text]
    return STATE_ABBR_TO_ANSI.get(str(value or "").strip().upper(), str(value or "").strip().zfill(2) if str(value or "").strip().isdigit() else "") or ""


def is_five_state_fips(fips: object, county_ansi: object = "", state: object = "") -> bool:
    text = str(fips or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit() and len(text) == 5:
        return text.zfill(5)[:2] in FIVE_STATE_ANSI
    state_ansi = _parsed_state_ansi(state if state else (text if len(text) <= 2 else ""))
    if state_ansi in FIVE_STATE_ANSI and county_ansi != "":
        county = str(county_ansi).strip()
        if county.endswith(".0"):
            county = county[:-2]
        return county.isdigit()
    return state_ansi in FIVE_STATE_ANSI


def _row_fips(row: dict) -> str | None:
    for key in ("FIPS", "fips", "FIPS Code"):
        value = row.get(key)
        if value not in (None, ""):
            text = str(value).strip()
            if text.endswith(".0"):
                text = text[:-2]
            if text.isdigit():
                return text.zfill(5)
    state = _parsed_state_ansi(row.get("state_ansi", row.get("State ANSI", row.get("state"))))
    county = str(row.get("county_ansi", row.get("County ANSI", row.get("county"))) or "").strip()
    if county.endswith(".0"):
        county = county[:-2]
    if state.isdigit() and county.isdigit():
        return state.zfill(2) + county.zfill(3)
    return None


def filter_usda_rows(rows: Iterable[dict]) -> list[dict]:
    kept = []
    for row in rows:
        fips = _row_fips(row)
        if fips and is_five_state_fips(fips):
            kept.append(row)
    return kept


def _usda_sample_rows(rows: Iterable[dict]) -> list[dict]:
    return [
        row for row in rows
        if str(row.get("commodity_desc", "")).strip().upper() == "CORN"
        and str(row.get("reference_period_desc", "")).strip().upper() == "YEAR"
        and _row_fips(row)
        and is_five_state_fips(_row_fips(row))
    ]


def _row_is_five_state(row: dict) -> bool:
    fips = _row_fips(row)
    if fips:
        return is_five_state_fips(fips)
    return is_five_state_fips("", state=row.get("state", row.get("state_name", row.get("state_ansi"))))


def classify_url_rows(rows: Sequence[dict]) -> dict[str, list[dict]]:
    keep, delete = [], []
    for row in rows:
        image_type = str(row.get("image_type", "")).strip().upper()
        if image_type == "AG" and _row_is_five_state(row):
            keep.append(row)
        else:
            delete.append(row)
    return {"keep": keep, "delete": delete}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _weather_fips(weather_dir: Path, years: Iterable[int]) -> set[str]:
    result = set()
    for year in years:
        for path in sorted((weather_dir / str(year)).glob("**/*.csv")):
            state_dir = next((parent for parent in path.parents if parent.parent == weather_dir / str(year)), None)
            if state_dir is not None and _parsed_state_ansi(state_dir.name) not in FIVE_STATE_ANSI:
                continue
            try:
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    for row in csv.DictReader(handle):
                        value = _row_fips(row)
                        if value:
                            result.add(value)
            except (OSError, UnicodeError):
                continue
    return result


def _usda_rows_by_year(usda_dir: Path, years: Iterable[int]) -> dict[int, tuple[list[str], list[dict]]]:
    result = {}
    for year in sorted(years):
        path = usda_dir / f"USDA_Corn_County_{year}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            result[year] = (reader.fieldnames or [], list(reader))
    return result


def _valid_target_fips(usda_dir: Path, weather_dir: Path, years: Iterable[int]) -> set[str]:
    usda = set()
    for rows in _usda_rows_by_year(usda_dir, years).values():
        usda.update(fips for row in _usda_sample_rows(rows[1]) if (fips := _row_fips(row)))
    weather_states = set()
    for year in years:
        year_dir = weather_dir / str(year)
        if year_dir.exists():
            weather_states.update(
                _parsed_state_ansi(path.name)
                for path in year_dir.iterdir()
                if path.is_dir() and _parsed_state_ansi(path.name) in FIVE_STATE_ANSI
            )
    return {fips for fips in usda if fips[:2] in weather_states}


def _weather_directories_to_delete(weather_dir: Path, years: Iterable[int]) -> list[str]:
    deleted = []
    for year in sorted(years):
        year_dir = weather_dir / str(year)
        if not year_dir.exists():
            continue
        for state_dir in sorted(path for path in year_dir.iterdir() if path.is_dir()):
            state_ansi = _parsed_state_ansi(state_dir.name)
            if state_ansi not in FIVE_STATE_ANSI:
                deleted.append(str(state_dir))
    return deleted


def build_sync_plan(
    usda_dir: Path,
    weather_dir: Path,
    url_manifest: Path,
    years: Iterable[int] = FIVE_STATE_YEARS,
    source_entries: Sequence[dict] = (),
    extract_root: Path | None = None,
) -> dict:
    years = tuple(sorted({int(year) for year in years}))
    if set(years) != set(FIVE_STATE_YEARS):
        raise ValueError("five-state sync requires years 2017--2022")
    target_fips = sorted(_valid_target_fips(usda_dir, weather_dir, years))
    usda_plan = {}
    for year, (fieldnames, rows) in _usda_rows_by_year(usda_dir, years).items():
        kept = [row for row in rows if _row_is_five_state(row)]
        usda_plan[str(year)] = {"keep": len(kept), "delete": len(rows) - len(kept), "path": str(usda_dir / f"USDA_Corn_County_{year}.csv")}
    url_classification = classify_url_rows(_read_jsonl(url_manifest))
    entries = [dict(entry) for entry in source_entries if str(entry.get("image_type", "AG")).upper() == "AG"]
    county_paths = []
    if extract_root:
        county_paths = [str(extract_root / "AG" / str(entry["year"]) / fips / Path(entry["path"]).name) for entry in entries for fips in entry.get("fips", []) if fips in target_fips]
    return {
        "protocol": {"states": sorted(FIVE_STATE_NAMES), "ansi": sorted(FIVE_STATE_ANSI), "years": list(years), "image_types": ["AG"]},
        "target_fips": target_fips,
        "usda": usda_plan,
        "weather_state_directories_to_delete": _weather_directories_to_delete(weather_dir, years),
        "old_url_manifest_to_delete": [{"oss_key": row.get("oss_key"), "path": row.get("path"), "reason": "non-five-state-or-NDVI"} for row in url_classification["delete"]],
        "old_url_manifest_to_keep": len(url_classification["keep"]),
        "oss_objects_to_delete": [row.get("oss_key") for row in url_classification["delete"] if row.get("oss_key")],
        "ag_source_files_to_download": [entry.get("path") for entry in entries],
        "county_files_to_upload": county_paths,
    }


def backup_usda_files(usda_dir: Path, years: Iterable[int], backup_root: Path) -> list[dict]:
    report = []
    backup_root.mkdir(parents=True, exist_ok=True)
    for year in sorted(years):
        source = usda_dir / f"USDA_Corn_County_{year}.csv"
        if not source.exists():
            continue
        target = backup_root / source.name
        shutil.copy2(source, target)
        report.append({"source": str(source), "backup": str(target), "sha256": sha256(source), "lines": sum(1 for _ in source.open(encoding="utf-8"))})
    return report


def delete_oss_objects(rows: Sequence[dict], bucket) -> dict:
    deleted, failed = [], []
    for row in rows:
        key = row.get("oss_key") if isinstance(row, dict) else str(row)
        if not key:
            continue
        try:
            bucket.delete_object(key)
            deleted.append(key)
        except Exception as exc:
            failed.append({"oss_key": key, "error": str(exc)})
    if failed:
        raise RuntimeError(f"OSS deletion failed: {json.dumps(failed, sort_keys=True)}")
    return {"deleted": deleted, "failed": failed}


def _rewrite_usda_files(usda_dir: Path, years: Iterable[int]) -> dict:
    result = {}
    for year in sorted(years):
        path = usda_dir / f"USDA_Corn_County_{year}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
        kept = [row for row in rows if _row_is_five_state(row)]
        text_lines = ["\ufeff" + ",".join(fieldnames)]
        text_lines.extend(",".join(row.get(field, "") for field in fieldnames) for row in kept)
        atomic_write_text(path, "\n".join(text_lines) + "\n")
        result[str(year)] = {"keep": len(kept), "delete": len(rows) - len(kept)}
    return result


def sync_cropnet_five_state(
    usda_dir: Path,
    weather_dir: Path,
    years: Iterable[int] = FIVE_STATE_YEARS,
    url_manifest: Path = DEFAULT_URL_MANIFEST,
    run_root: Path = DEFAULT_RUNTIME_ROOT / "sync-runs",
    download_root: Path = DEFAULT_DOWNLOAD_ROOT,
    extract_root: Path = DEFAULT_EXTRACT_ROOT,
    url_output: Path | None = None,
    bucket=None,
    oss_prefix: str = "sentinel",
    execute: bool = False,
    tree_payload: Sequence[dict] | None = None,
) -> dict:
    years = tuple(sorted({int(year) for year in years}))
    target_fips = _valid_target_fips(usda_dir, weather_dir, years)
    entries = parse_hf_tree(tree_payload, target_fips, set(years), {"AG"}) if tree_payload is not None else (fetch_hf_files(target_fips, set(years), {"AG"}) if target_fips else [])
    plan = build_sync_plan(usda_dir, weather_dir, url_manifest, years, entries, extract_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(run_dir / "plan.json", json.dumps(plan, indent=2, sort_keys=True) + "\n")
    result = {"dry_run": not execute, "run_dir": str(run_dir), "plan": plan, "status": []}
    if not execute:
        atomic_write_text(run_dir / "status.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
    if bucket is None:
        raise ValueError("execute requires an authenticated OSS bucket")
    backup = backup_usda_files(usda_dir, years, run_dir / "usda-backup")
    result["status"].append({"step": "backup-usda", "report": backup})
    download_result = sync_counties_to_oss(entries, download_root, extract_root, bucket, run_dir / "sentinel_urls_cropnet5_ag.jsonl", oss_prefix)
    if download_result["failed"]:
        result["status"].append({"step": "ag-sync", "result": download_result})
        atomic_write_text(run_dir / "status.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise RuntimeError("AG synchronization failed")
    result["status"].append({"step": "ag-sync", "result": download_result})
    delete_result = delete_oss_objects([{"oss_key": key} for key in plan["oss_objects_to_delete"]], bucket)
    result["status"].append({"step": "delete-oss", "result": delete_result})
    for directory in plan["weather_state_directories_to_delete"]:
        shutil.rmtree(directory)
    result["status"].append({"step": "delete-weather", "count": len(plan["weather_state_directories_to_delete"])})
    result["status"].append({"step": "rewrite-usda", "result": _rewrite_usda_files(usda_dir, years)})
    new_rows = [*classify_url_rows(_read_jsonl(url_manifest))["keep"], *_read_jsonl(run_dir / "sentinel_urls_cropnet5_ag.jsonl")]
    new_rows = [row for row in new_rows if str(row.get("image_type", "")).upper() == "AG" and _row_is_five_state(row)]
    url_output = url_output or url_manifest.with_name("sentinel_urls_cropnet5_ag.jsonl")
    atomic_write_text(url_output, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in new_rows))
    result["status"].append({"step": "rewrite-url-manifest", "count": len(new_rows), "path": str(url_output)})
    atomic_write_text(run_dir / "status.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def parse_hf_tree(payload: Sequence[dict], target_fips: set[str], years: set[int], image_types: set[str]) -> list[dict]:
    """Filter HF tree entries, retaining state-level files when FIPS is implicit."""
    target_fips = {str(value).zfill(5) for value in target_fips}
    image_types = {value.upper() for value in image_types}
    entries = []
    for item in payload:
        path = str(item.get("path", ""))
        if item.get("type", "file") != "file" or not path.lower().endswith((".h5", ".hdf5")):
            continue
        match_type = re.search(r"(?:^|/)data/(?P<kind>AG|NDVI)/", path, re.IGNORECASE)
        year_match = YEAR_RE.search(path)
        if not match_type or not year_match:
            continue
        image_type = match_type.group("kind").upper()
        year = int(year_match.group("year"))
        if image_type not in image_types or year not in years:
            continue
        fips = _matched_fips(path, target_fips)
        state_match = re.search(rf"/data/{image_type}/{year}/([^/]+)/", path, re.IGNORECASE)
        state = state_match.group(1) if state_match else None
        state_fips = {fips_code for fips_code in target_fips if fips_code[:2] == _state_ansi(state)} if state else set()
        if not fips and not state_fips:
            continue
        entries.append({
            "path": path,
            "url": HF_BASE + quote(path, safe="/"),
            "size": int(item.get("size", item.get("lfs", {}).get("size", 0)) or 0),
            "oid": item.get("oid") or item.get("lfs", {}).get("oid"),
            "image_type": image_type,
            "year": year,
            "state": state,
            "fips": sorted(fips or state_fips),
            "scope": "county" if fips else "state",
        })
    return sorted(entries, key=lambda item: item["path"])


def fetch_hf_files(target_fips: set[str], years: set[int], image_types: set[str]) -> list[dict]:
    """Enumerate only target state/year directories through the HF tree API."""
    states = sorted({_state_abbr(fips[:2]) for fips in target_fips})
    payload = []
    for image_type in sorted({value.upper() for value in image_types}):
        for year in sorted(years):
            for state in states:
                url = f"{HF_API_BASE}Sentinel-2%20Imagery/data/{image_type}/{year}/{state}?expand=true"
                with urllib.request.urlopen(url) as response:
                    payload.extend(json.load(response))
    return parse_hf_tree(payload, target_fips, years, image_types)


def _state_ansi(state: str | None) -> str:
    if not state:
        return ""
    # State directories use USPS abbreviations. The caller only needs this mapping for filtering.
    states = {"AL":"01","AR":"05","CA":"06","CO":"08","DE":"10","GA":"13","IA":"19","ID":"16","IL":"17","IN":"18","KS":"20","KY":"21","LA":"22","MD":"24","MI":"26","MN":"27","MO":"29","MS":"28","MT":"30","NC":"37","ND":"38","NE":"31","NJ":"34","NM":"35","NY":"36","OH":"39","OK":"40","PA":"42","SC":"45","SD":"46","TN":"47","TX":"48","VA":"51","WA":"53","WI":"55","WV":"54","WY":"56"}
    return states.get(state.upper(), "")


def _state_abbr(ansi: str) -> str:
    reverse = {"01":"AL","05":"AR","06":"CA","08":"CO","10":"DE","13":"GA","19":"IA","16":"ID","17":"IL","18":"IN","20":"KS","21":"KY","22":"LA","24":"MD","26":"MI","27":"MN","29":"MO","28":"MS","30":"MT","37":"NC","38":"ND","31":"NE","34":"NJ","35":"NM","36":"NY","39":"OH","40":"OK","42":"PA","45":"SC","46":"SD","47":"TN","48":"TX","51":"VA","53":"WA","55":"WI","54":"WV","56":"WY"}
    return reverse.get(ansi, "")


def summarize_manifest(entries: Sequence[dict]) -> dict:
    return {
        "files": len(entries),
        "bytes": sum(int(item.get("size", 0)) for item in entries),
        "fips": len({fips for item in entries for fips in item.get("fips", [])}),
        "by_year": dict(sorted(Counter(int(item["year"]) for item in entries).items())),
        "by_image_type": dict(sorted(Counter(item["image_type"] for item in entries).items())),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(entry: dict, destination: Path, resume: bool = True,
                 min_delay: float = 0.3, max_retries: int = 6) -> str:
    if destination.exists() and entry.get("size") and destination.stat().st_size == int(entry["size"]):
        return "skipped"
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(max_retries + 1):
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            offset = partial.stat().st_size if resume and partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            request = urllib.request.Request(entry["url"], headers=headers)
            response = urllib.request.urlopen(request)
            if offset and getattr(response, "status", None) != 206:
                offset = 0
                mode = "wb"
            else:
                mode = "ab" if offset else "wb"
            with response, partial.open(mode) as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
            if entry.get("size") and partial.stat().st_size != int(entry["size"]):
                raise IOError(f"size mismatch for {entry['path']}: {partial.stat().st_size} != {entry['size']}")
            partial.replace(destination)
            return "downloaded"
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries:
                wait = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt < max_retries:
                time.sleep(random.uniform(1, 3))
                continue
            raise
    raise RuntimeError(f"download retries exhausted for {entry.get('path')}")


def download_url_manifest(url_manifest: Path, cache_root: Path, resume: bool = True) -> dict:
    """Download public OSS URLs into a reusable local cache.

    The URL manifest is the only runtime input required by a training job.
    Files are verified against the recorded size and SHA256 when available.
    """
    result = {"downloaded": 0, "skipped": 0, "failed": []}
    for line in url_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("status") not in {"uploaded", "available"}:
            continue
        relative = entry.get("cache_path") or entry.get("oss_key", entry["path"])
        target = cache_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            size_ok = not entry.get("size") or target.stat().st_size == int(entry["size"])
            hash_ok = not entry.get("sha256") or sha256(target) == entry["sha256"]
            if size_ok and hash_ok:
                result["skipped"] += 1
                continue
            target.unlink()
        partial = target.with_name(target.name + ".part")
        try:
            offset = partial.stat().st_size if resume and partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            request = urllib.request.Request(entry["url"], headers=headers)
            mode = "ab" if offset else "wb"
            with urllib.request.urlopen(request) as response, partial.open(mode) as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
            if entry.get("size") and partial.stat().st_size != int(entry["size"]):
                raise IOError(f"size mismatch: {partial.stat().st_size} != {entry['size']}")
            if entry.get("sha256") and sha256(partial) != entry["sha256"]:
                raise IOError("sha256 mismatch")
            partial.replace(target)
            result["downloaded"] += 1
        except Exception as exc:
            result["failed"].append({"url": entry.get("url"), "error": str(exc)})
    return result


def download_manifest(entries: Sequence[dict], destination: Path, resume: bool = True, dry_run: bool = False,
                     min_delay: float = 0.3) -> dict:
    result = {"downloaded": 0, "skipped": 0, "failed": []}
    for entry in entries:
        target = destination / entry["path"]
        if dry_run:
            continue
        try:
            result[download_one(entry, target, resume)] += 1
            time.sleep(min_delay)
        except Exception as exc:  # preserve all failures for resumable reruns
            result["failed"].append({"path": entry["path"], "error": str(exc)})
    return result


def upload_manifest_to_oss(
    entries: Sequence[dict],
    local_root: Path,
    bucket,
    url_output: Path,
    prefix: str = "sentinel",
) -> dict:
    """Upload downloaded files as public-read objects and write a local URL index.

    Credentials stay outside the repository. ``bucket`` is an already
    authenticated ``oss2.Bucket`` instance, which also makes this function
    straightforward to test without contacting OSS.
    """
    import oss2

    uploaded = 0
    skipped = 0
    failed = []
    existing_rows = []
    if url_output.exists():
        existing_rows = [json.loads(line) for line in url_output.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows_by_key = {row["oss_key"]: row for row in existing_rows if row.get("oss_key")}
    endpoint = str(bucket.endpoint).rstrip("/")
    bucket_name = getattr(bucket, "bucket_name", "")
    for entry in entries:
        local_path = local_root / entry["path"]
        oss_key = "/".join(part for part in (prefix.strip("/"), entry["path"]) if part)
        row = {
            "path": entry["path"],
            "local_path": str(local_path),
            "oss_key": oss_key,
            "image_type": entry.get("image_type"),
            "year": entry.get("year"),
            "state": entry.get("state"),
            "fips": entry.get("fips", []),
            "size": entry.get("size", 0),
        }
        try:
            if rows_by_key.get(oss_key, {}).get("status") == "uploaded":
                skipped += 1
                continue
            if not local_path.exists():
                raise FileNotFoundError(local_path)
            expected_size = int(entry.get("size", 0) or 0)
            if expected_size and local_path.stat().st_size != expected_size:
                raise IOError(f"size mismatch: {local_path.stat().st_size} != {expected_size}")
            bucket.put_object_from_file(oss_key, str(local_path))
            bucket.put_object_acl(oss_key, oss2.OBJECT_ACL_PUBLIC_READ)
            row["sha256"] = sha256(local_path)
            host = endpoint
            if bucket_name and host.startswith("https://") and not host.split("//", 1)[1].startswith(f"{bucket_name}."):
                host = f"https://{bucket_name}.{host.split('//', 1)[1]}"
            row["url"] = f"{host}/{quote(oss_key, safe='/')}"
            row["bucket"] = bucket_name
            row["status"] = "uploaded"
            uploaded += 1
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            failed.append({"path": entry["path"], "error": str(exc)})
        rows_by_key[oss_key] = row

    url_output.parent.mkdir(parents=True, exist_ok=True)
    with url_output.open("w", encoding="utf-8") as handle:
        for row in sorted(rows_by_key.values(), key=lambda item: item["oss_key"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"uploaded": uploaded, "skipped": skipped, "failed": failed, "url_manifest": str(url_output)}


def extract_target_counties(entries: Sequence[dict], source_root: Path, output_root: Path) -> list[dict]:
    """Extract only target FIPS groups from state-level HDF5 files.

    The HF files are state/quarter bundles. The resulting files are county
    scoped and can be uploaded independently, avoiding public exposure of
    unrelated counties and making training downloads small.
    """
    import h5py

    extracted = []
    for entry in entries:
        source = source_root / entry["path"]
        if not source.exists():
            continue
        with h5py.File(source, "r") as source_file:
            for fips in entry.get("fips", []):
                if fips not in source_file:
                    continue
                name = Path(entry["path"]).name
                target = output_root / entry["image_type"] / str(entry["year"]) / fips / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(target, "w") as target_file:
                    source_file.copy(fips, target_file)
                derived = dict(entry)
                derived["path"] = str(target.relative_to(output_root))
                derived["source_path"] = entry["path"]
                derived["fips"] = [fips]
                derived["scope"] = "county"
                derived["size"] = target.stat().st_size
                extracted.append(derived)
    return extracted


def sync_counties_to_oss(
    entries: Sequence[dict],
    download_root: Path,
    extract_root: Path,
    bucket,
    url_manifest: Path,
    prefix: str = "sentinel",
) -> dict:
    """逐个下载州级文件、提取目标县并上传，避免本地堆积全部源文件。"""
    result = {"source_files": len(entries), "processed": 0, "uploaded": 0, "skipped": 0, "failed": []}
    existing_rows = []
    if url_manifest.exists():
        existing_rows = [json.loads(line) for line in url_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    uploaded_keys = {row["oss_key"] for row in existing_rows if row.get("status") == "uploaded"}

    for index, entry in enumerate(entries, 1):
        source_path = download_root / entry["path"]
        try:
            source_exists = source_path.exists()
            county_entries = extract_target_counties([entry], download_root, extract_root)
            pending = []
            for ce in county_entries:
                oss_key = "/".join(part for part in (prefix.strip("/"), ce["path"]) if part)
                if oss_key not in uploaded_keys:
                    pending.append(ce)
            if source_exists and not pending:
                result["skipped"] += 1
                print(f"[{index}/{len(entries)}] all-uploaded: {entry['path']} -> {len(county_entries)} counties", flush=True)
                time.sleep(random.uniform(0.3, 0.8))
                continue

            download_status = download_one(entry, source_path, resume=True)
            if not county_entries:
                county_entries = extract_target_counties([entry], download_root, extract_root)
                pending = [ce for ce in county_entries
                           if "/".join(part for part in (prefix.strip("/"), ce["path"]) if part) not in uploaded_keys]
            upload_result = upload_manifest_to_oss(pending, extract_root, bucket, url_manifest, prefix)
            result["uploaded"] += upload_result["uploaded"]
            result["skipped"] += upload_result["skipped"]
            if upload_result["failed"]:
                raise RuntimeError(f"{len(upload_result['failed'])} county uploads failed")
            for county_entry in county_entries:
                (extract_root / county_entry["path"]).unlink(missing_ok=True)
            result["processed"] += 1
            print(f"[{index}/{len(entries)}] {download_status}: {entry['path']} -> {len(county_entries)} counties", flush=True)
            time.sleep(random.uniform(0.5, 1.5))
        except Exception as exc:
            result["failed"].append({"path": entry["path"], "error": str(exc)})
            print(f"[{index}/{len(entries)}] failed: {entry['path']}: {exc}", flush=True)
            time.sleep(random.uniform(2, 5))
    return result


def create_oss_bucket(endpoint: str, bucket_name: str, access_key_id: str, access_key_secret: str):
    """Create an authenticated OSS bucket without persisting credentials."""
    import oss2

    auth = oss2.Auth(access_key_id, access_key_secret)
    return oss2.Bucket(auth, endpoint, bucket_name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-json", type=Path)
    parser.add_argument("--target-fips", type=Path, default=DEFAULT_TARGET_FIPS)
    parser.add_argument("--use-manifest", type=Path,
                        help="Skip HF API, load entries from an existing JSONL manifest")
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_REMOTE_MANIFEST)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument("--extract-root", type=Path, default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--years", type=int, nargs="+", default=[2017, 2018, 2019, 2020, 2021, 2022])
    parser.add_argument("--image-types", nargs="+", default=["AG", "NDVI"])
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upload-oss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--url-manifest", type=Path, default=DEFAULT_URL_MANIFEST)
    parser.add_argument("--cropnet5-url-manifest", type=Path, default=None)
    parser.add_argument("--oss-prefix", default="sentinel")
    parser.add_argument("--oss-endpoint", default=DEFAULT_OSS_ENDPOINT)
    parser.add_argument("--oss-bucket", default=DEFAULT_OSS_BUCKET)
    parser.add_argument("--sync-cropnet-five-state", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Execute destructive five-state cleanup; otherwise dry-run")
    parser.add_argument("--usda-dir", type=Path, default=Path("/data/raid0/hqx/Product_model_runtime/train_dataset/cropnet_dataset/data/usda_corn"))
    parser.add_argument("--weather-dir", type=Path, default=Path("/data/raid0/hqx/Product_model_runtime/train_dataset/cropnet_dataset/data/weather"))
    parser.add_argument("--sync-run-root", type=Path, default=DEFAULT_RUNTIME_ROOT / "sync-runs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.sync_cropnet_five_state:
        bucket = None
        if args.execute:
            required = {
                "OSS_ENDPOINT": os.environ.get("OSS_ENDPOINT", args.oss_endpoint),
                "OSS_BUCKET": os.environ.get("OSS_BUCKET", args.oss_bucket),
                "OSS_ACCESS_KEY_ID": os.environ.get("OSS_ACCESS_KEY_ID", DEFAULT_OSS_ACCESS_KEY_ID),
                "OSS_ACCESS_KEY_SECRET": os.environ.get("OSS_ACCESS_KEY_SECRET", DEFAULT_OSS_ACCESS_KEY_SECRET),
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise SystemExit("missing OSS environment variables: " + ", ".join(missing))
            bucket = create_oss_bucket(required["OSS_ENDPOINT"], required["OSS_BUCKET"], required["OSS_ACCESS_KEY_ID"], required["OSS_ACCESS_KEY_SECRET"])
        result = sync_cropnet_five_state(
            usda_dir=args.usda_dir,
            weather_dir=args.weather_dir,
            years=FIVE_STATE_YEARS,
            url_manifest=args.url_manifest,
            run_root=args.sync_run_root,
            download_root=args.destination,
            extract_root=args.extract_root,
            url_output=args.cropnet5_url_manifest,
            bucket=bucket,
            oss_prefix=args.oss_prefix,
            execute=args.execute,
            tree_payload=json.loads(args.tree_json.read_text(encoding="utf-8")) if args.tree_json else None,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.use_manifest:
        entries = [json.loads(line) for line in args.use_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        fips = {line.strip().zfill(5) for line in args.target_fips.read_text(encoding="utf-8").splitlines() if line.strip()}
        if args.tree_json:
            payload = json.loads(args.tree_json.read_text(encoding="utf-8"))
            entries = parse_hf_tree(payload, fips, set(args.years), set(args.image_types))
        else:
            entries = fetch_hf_files(fips, set(args.years), set(args.image_types))
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text("\n".join(json.dumps(item, sort_keys=True) for item in entries) + "\n", encoding="utf-8")
    print(json.dumps(summarize_manifest(entries), indent=2))
    if args.download and args.upload_oss:
        required = {
            "OSS_ENDPOINT": args.oss_endpoint,
            "OSS_BUCKET": args.oss_bucket,
            "OSS_ACCESS_KEY_ID": os.environ.get("OSS_ACCESS_KEY_ID", DEFAULT_OSS_ACCESS_KEY_ID),
            "OSS_ACCESS_KEY_SECRET": os.environ.get("OSS_ACCESS_KEY_SECRET", DEFAULT_OSS_ACCESS_KEY_SECRET),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise SystemExit("missing OSS environment variables: " + ", ".join(missing))
        bucket = create_oss_bucket(required["OSS_ENDPOINT"], required["OSS_BUCKET"], required["OSS_ACCESS_KEY_ID"], required["OSS_ACCESS_KEY_SECRET"])
        print(json.dumps(sync_counties_to_oss(entries, args.destination, args.extract_root, bucket, args.url_manifest, args.oss_prefix), indent=2))
    elif args.download:
        print(json.dumps(download_manifest(entries, args.destination), indent=2))
    elif args.upload_oss:
        raise SystemExit("--upload-oss requires downloads; omit --no-download")


if __name__ == "__main__":
    main()
