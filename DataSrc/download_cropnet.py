"""CropNet 源数据唯一下载与审计入口。

模式：
  --audit-only    只列远端清单并与本地对照，不下载任何文件；
  --download      增量下载 USDA、HRRR、Sentinel-2 AG、县界邻接和 gSSURGO 土壤；
  --verify-local  只读扫描本地源数据，写出 download_audit.jsonl。

所有源数据固定保存到 /data/raid0/hqx/DataSrc；--data-root 只用于临时测试。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_dataset.dataset import (  # noqa: E402
    ABBR_TO_ANSI, ADJACENCY_RELPATH, DATA_ROOT, REMOTE_MONTHS, SOIL_FEATURES,
    SOIL_RELPATH, STATES, WEATHER_DAYS, YEARS, hrrr_relpath, sentinel_relpath,
    usda_relpath,
)

REPO_ID = "CropNet/CropNet"
REPO_TYPE = "dataset"
REVISION = "1dcfbe99d49910bd108b27ec0ca9b2b309a2be7e"
ADJACENCY_URL = "https://raw.githubusercontent.com/gomes-lab/GNN-RNN/main/map/county_adjacency.txt"
SDA_URL = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"
TIGERWEB_URL = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/query"
STATE_ABBRS = tuple(abbr for _, abbr in STATES.values())
MANIFEST_NAME = "source_manifest.json"
AUDIT_NAME = "download_audit.jsonl"
_USDA_RE = re.compile(r"USDA Crop Dataset/Corn/(\d{4})/USDA_Corn_County_\1\.csv")
_HRRR_RE = re.compile(r"WRF-HRRR Computed Dataset/data/(\d{4})/([A-Z]{2})/HRRR_(\d{2})_\2_\1-(\d{2})\.csv")
_AG_RE = re.compile(
    r"Sentinel-2 Imagery/data/AG/(\d{4})/([A-Z]{2})/Agriculture_(\d{2})_\2_\1-(04-01_\1-06-30|07-01_\1-09-30)\.h5"
)


# ---------------------------------------------------------------------------
# 远端清单
# ---------------------------------------------------------------------------

@dataclass
class SourceInventory:
    usda: dict = field(default_factory=dict)       # year -> remote path
    hrrr: dict = field(default_factory=dict)       # (year, state, month) -> remote path
    sentinel: dict = field(default_factory=dict)   # (year, state, quarter) -> remote path

    def paths(self) -> list[str]:
        return sorted([*self.usda.values(), *self.hrrr.values(), *self.sentinel.values()])

    def missing(self) -> list[str]:
        out = [f"usda {y}" for y in YEARS if y not in self.usda]
        for y in YEARS:
            for s in STATE_ABBRS:
                out += [f"hrrr {y} {s} {m:02d}" for m in range(1, 13) if (y, s, m) not in self.hrrr]
                out += [f"sentinel {y} {s} {q}" for q in ("Q2", "Q3") if (y, s, q) not in self.sentinel]
        return out


def parse_hf_inventory(paths: Iterable[str]) -> SourceInventory:
    """只接受 Corn、2017--2022、研究区域、Sentinel AG Q2/Q3 和 HRRR 十二个月。"""
    inventory = SourceInventory()
    for path in paths:
        if m := _USDA_RE.fullmatch(path):
            if int(m[1]) in YEARS:
                inventory.usda[int(m[1])] = path
        elif m := _HRRR_RE.fullmatch(path):
            year, state, month = int(m[1]), m[2], int(m[4])
            if year in YEARS and state in STATE_ABBRS and ABBR_TO_ANSI[state] == m[3] and 1 <= month <= 12:
                inventory.hrrr[(year, state, month)] = path
        elif m := _AG_RE.fullmatch(path):
            year, state = int(m[1]), m[2]
            if year in YEARS and state in STATE_ABBRS and ABBR_TO_ANSI[state] == m[3]:
                inventory.sentinel[(year, state, "Q2" if m[4].startswith("04") else "Q3")] = path
    return inventory


def local_relpath(remote_path: str) -> str:
    """远端路径 -> DATA_ROOT 下相对路径（与 train_dataset.dataset 的布局函数一致）。"""
    if m := _USDA_RE.fullmatch(remote_path):
        return usda_relpath(int(m[1]))
    if m := _HRRR_RE.fullmatch(remote_path):
        return hrrr_relpath(int(m[1]), m[2], int(m[4]))
    if m := _AG_RE.fullmatch(remote_path):
        return sentinel_relpath(int(m[1]), m[2], 4 if m[4].startswith("04") else 7)
    raise ValueError(f"not a target source path: {remote_path}")
@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str | None
    blob_id: str

    @property
    def algorithm(self) -> str:
        return "sha256" if self.sha256 else "git-sha1"

    @property
    def digest(self) -> str:
        return self.sha256 or self.blob_id


def list_remote(revision: str = REVISION) -> dict[str, RemoteFile]:
    from huggingface_hub import HfApi

    entries = HfApi().list_repo_tree(REPO_ID, repo_type=REPO_TYPE, recursive=True, revision=revision)
    return {
        e.path: RemoteFile(e.path, e.size, e.lfs.sha256 if e.lfs else None, e.blob_id)
        for e in entries if hasattr(e, "size")
    }


# ---------------------------------------------------------------------------
# 完整性校验与增量下载
# ---------------------------------------------------------------------------

def file_digest(path: Path, algorithm: str) -> str:
    """sha256 用于 LFS 文件；git-sha1 与 Hugging Face 非 LFS blob_id 一致。"""
    if algorithm == "sha256":
        h = hashlib.sha256()
    elif algorithm == "git-sha1":
        h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    else:
        raise ValueError(f"unknown digest algorithm {algorithm}")
    with open(path, "rb") as handle:
        while chunk := handle.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


class Manifest:
    """记录已校验文件的 size/mtime/digest，重跑时跳过未变化的文件。"""

    def __init__(self, root: Path):
        self.path = root / MANIFEST_NAME
        self.lock = threading.Lock()
        self.entries = json.loads(self.path.read_text()) if self.path.is_file() else {}

    def is_verified(self, relpath: str, local: Path, digest: str) -> bool:
        entry = self.entries.get(relpath)
        if not entry or entry.get("digest") != digest or not local.is_file():
            return False
        stat = local.stat()
        return entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns

    def record(self, relpath: str, local: Path, **info) -> None:
        stat = local.stat()
        with self.lock:
            self.entries[relpath] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, **info}
            write_atomic(self.path, json.dumps(self.entries, indent=1, sort_keys=True).encode())


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def hf_download(remote_path: str, staging: Path, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, remote_path, repo_type=REPO_TYPE,
                                revision=revision, local_dir=str(staging)))


def ensure_file(remote: RemoteFile, root: Path, manifest: Manifest, *,
                revision: str = REVISION, seed_root: Path | None = None) -> str:
    """返回 verified / seeded / downloaded。校验失败的已有文件不删除，只在通过后原子替换。"""
    relpath = local_relpath(remote.path)
    local = root / relpath
    info = {"algorithm": remote.algorithm, "digest": remote.digest,
            "remote_path": remote.path, "revision": revision}
    if manifest.is_verified(relpath, local, remote.digest):
        return "verified"
    if local.is_file() and local.stat().st_size == remote.size \
            and file_digest(local, remote.algorithm) == remote.digest:
        manifest.record(relpath, local, source="local", **info)
        return "verified"

    staging = root / ".staging" / hashlib.md5(relpath.encode()).hexdigest()
    staging.mkdir(parents=True, exist_ok=True)
    try:
        seed = seed_root / relpath if seed_root else None
        if seed and seed.is_file() and seed.stat().st_size == remote.size \
                and file_digest(seed, remote.algorithm) == remote.digest:
            staged = staging / local.name
            try:
                os.link(seed, staged)
            except OSError:
                shutil.copy2(seed, staged)
            action = "seeded"
        else:
            staged = hf_download(remote.path, staging, revision)
            if staged.stat().st_size != remote.size or file_digest(staged, remote.algorithm) != remote.digest:
                raise ValueError(f"digest mismatch after download: {remote.path}")
            action = "downloaded"
        local.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, local)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    manifest.record(relpath, local, source=action, **info)
    return action


def download_sources(remote: dict[str, RemoteFile], root: Path, *, revision: str = REVISION,
                     seed_root: Path | None = None, workers: int = 8) -> Counter:
    inventory = parse_hf_inventory(remote)
    missing = inventory.missing()
    if missing:
        raise ValueError(f"remote inventory incomplete ({len(missing)}): {missing[:10]}")
    targets = [remote[p] for p in inventory.paths()]
    manifest = Manifest(root)
    pending = [t for t in targets if not manifest.is_verified(local_relpath(t.path), root / local_relpath(t.path), t.digest)]
    need = sum(t.size for t in pending)
    free = shutil.disk_usage(root).free
    if need > free:
        raise OSError(f"need {need / 1e9:.1f} GB but only {free / 1e9:.1f} GB free under {root}")
    print(f"[download] {len(targets)} targets, {len(pending)} pending, up to {need / 1e9:.1f} GB", flush=True)

    counts, failures = Counter(), []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ensure_file, t, root, manifest, revision=revision, seed_root=seed_root): t
                   for t in pending}
        for i, future in enumerate(as_completed(futures), 1):
            target = futures[future]
            try:
                counts[future.result()] += 1
            except Exception as exc:  # 汇总后统一报错，不静默忽略
                failures.append(f"{target.path}: {exc!r}")
                counts["failed"] += 1
            if i % 20 == 0 or i == len(pending):
                print(f"[download] {i}/{len(pending)} {dict(counts)}", flush=True)
    counts["verified"] += len(targets) - len(pending)
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed:\n" + "\n".join(failures[:20]))
    return counts
# ---------------------------------------------------------------------------
# 县界邻接
# ---------------------------------------------------------------------------

def parse_county_adjacency(text: str, counties: set[str] | None = None) -> dict[str, set[str]]:
    """解析 Census/GNN-RNN 格式邻接表：对称、去自环、五位 FIPS，可限制到给定县集合。"""
    graph: dict[str, set[str]] = {}
    current = None
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"adjacency line {lineno}: expected 4 tab-separated fields")
        if fields[1].strip():
            current = fields[1].strip()
        neighbor = fields[3].strip()
        if current is None or not re.fullmatch(r"\d{5}", current) or not re.fullmatch(r"\d{5}", neighbor):
            raise ValueError(f"adjacency line {lineno}: invalid FIPS")
        graph.setdefault(current, set())
        graph.setdefault(neighbor, set())
        if neighbor != current:
            graph[current].add(neighbor)
            graph[neighbor].add(current)
    if counties is not None:
        graph = {c: graph.get(c, set()) & counties for c in counties}
    return graph


def http_bytes(url: str, data: dict | None = None, *, timeout: int = 180, retries: int = 4) -> bytes:
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    headers = {"User-Agent": "CropNet-unified/1.0"}
    if body is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 5)
    raise AssertionError("unreachable")


def fetch_adjacency(root: Path) -> str:
    data = http_bytes(ADJACENCY_URL)
    parse_county_adjacency(data.decode("latin-1"))  # 先解析验证，再落盘
    target = root / ADJACENCY_RELPATH
    digest = hashlib.sha256(data).hexdigest()
    if not (target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest):
        write_atomic(target, data)
    Manifest(root).record(ADJACENCY_RELPATH, target, source=ADJACENCY_URL, algorithm="sha256", digest=digest,
                          fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    return digest


# ---------------------------------------------------------------------------
# gSSURGO 县级土壤（USDA NRCS Soil Data Access）
# ---------------------------------------------------------------------------

_SOIL_COLUMNS = ("muacres", "comppct_r", "hzdept_r", "hzdepb_r", "claytotal_r", "sandtotal_r",
                 "silttotal_r", "om_r", "ph1to1h2o_r", "dbthirdbar_r", "awc_r")
_SOIL_MAP = {"clay_pct": "claytotal_r", "sand_pct": "sandtotal_r", "silt_pct": "silttotal_r",
             "om_pct": "om_r", "ph": "ph1to1h2o_r", "bulk_density": "dbthirdbar_r", "awc": "awc_r"}
_SOIL_SELECT = ("SELECT mu.muacres, co.comppct_r, ch.hzdept_r, ch.hzdepb_r, ch.claytotal_r, ch.sandtotal_r, "
                "ch.silttotal_r, ch.om_r, ch.ph1to1h2o_r, ch.dbthirdbar_r, ch.awc_r "
                "FROM mapunit mu JOIN component co ON co.mukey = mu.mukey "
                "JOIN chorizon ch ON ch.cokey = co.cokey ")
_SOIL_DEPTH = " AND ch.hzdept_r < 30 AND ch.hzdepb_r > 0"


def sda_rows(sql: str, *, timeout: int = 180, retries: int = 4) -> list[dict]:
    payload = json.loads(http_bytes(SDA_URL, {"query": sql, "format": "JSON+COLUMNNAME"},
                                    timeout=timeout, retries=retries).decode() or "{}")
    table = payload.get("Table", []) if isinstance(payload, dict) else []
    if not table:
        return []
    header, *rows = table
    return [dict(zip(header, row)) for row in rows]


def soil_rows_by_areasymbol(fips: str) -> list[dict]:
    areasymbol = STATES[fips[:2]][1] + fips[2:]
    if not re.fullmatch(r"[A-Z]{2}\d{3}", areasymbol):
        raise ValueError(f"bad areasymbol {areasymbol}")
    sql = (_SOIL_SELECT + "JOIN legend l ON l.lkey = mu.lkey "
           f"WHERE l.areasymbol = '{areasymbol}'" + _SOIL_DEPTH)
    return sda_rows(sql)


_SOIL_TILE_DEG = 0.2


def _tiger_polygons(fips: str) -> list:
    """TIGERweb 县界（经纬度）；每个多边形是若干环，每个环是 (lon, lat) 点列。"""
    payload = json.loads(http_bytes(TIGERWEB_URL + "?" + urllib.parse.urlencode({
        "where": f"GEOID='{fips}'", "outFields": "GEOID", "returnGeometry": "true",
        "outSR": "4326", "maxAllowableOffset": "0.001", "f": "geojson",
    })).decode())
    if not payload.get("features"):
        raise ValueError(f"no TIGERweb geometry for {fips}")
    geometry = payload["features"][0]["geometry"]
    return [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]


def _wkt(polygons) -> str:
    ring = lambda pts: "(" + ",".join(f"{x} {y}" for x, y in pts) + ")"  # noqa: E731
    return "MULTIPOLYGON(" + ",".join("(" + ",".join(ring(r) for r in p) + ")" for p in polygons) + ")"


def _mukeys_in(wkt: str, **kwargs) -> list[str]:
    rows = sda_rows("SELECT DISTINCT mukey FROM "
                    f"SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt}')", **kwargs)
    return sorted({str(r["mukey"]) for r in rows}, key=int)


def _soil_rows_for_mukeys(mukeys: list[str]) -> list[dict]:
    rows = []
    for i in range(0, len(mukeys), 200):
        chunk = ",".join(mukeys[i:i + 200])
        rows.extend(sda_rows(_SOIL_SELECT + f"WHERE mu.mukey IN ({chunk})" + _SOIL_DEPTH))
    return rows


def soil_rows_by_geometry(fips: str) -> tuple[str, list[dict]]:
    """调查区不与县一一对应时的回退：用 TIGERweb 县界 WKT 与 SSURGO 图斑求交。

    部分县的整县边界求交会被 SDA 拒绝（约 90 秒后返回 HTTP 500，Server
    configuration error），此时按 _SOIL_TILE_DEG 度网格分块求交、合并 mukey 后
    重查。分块与整县求交采用同一语义（取与边界相交的图斑，不做面积裁剪），
    两者结果一致。整县这一步只试一次：失败重试同样要等 90 秒，没有收益。
    """
    polygons = _tiger_polygons(fips)
    try:
        whole = _mukeys_in(_wkt(polygons), retries=1)
        if whole:
            return "wkt_intersection", _soil_rows_for_mukeys(whole)
    except OSError:
        pass
    points = [pt for poly in polygons for r in poly for pt in r]
    west, south = min(x for x, _ in points), min(y for _, y in points)
    east, north = max(x for x, _ in points), max(y for _, y in points)
    mukeys: set[str] = set()
    for i in range(int((east - west) / _SOIL_TILE_DEG) + 1):
        for j in range(int((north - south) / _SOIL_TILE_DEG) + 1):
            w, e = west + i * _SOIL_TILE_DEG, min(west + (i + 1) * _SOIL_TILE_DEG, east)
            s, n = south + j * _SOIL_TILE_DEG, min(south + (j + 1) * _SOIL_TILE_DEG, north)
            mukeys.update(_mukeys_in(_wkt([[[(w, s), (e, s), (e, n), (w, n), (w, s)]]])))
    return "wkt_tiled", _soil_rows_for_mukeys(sorted(mukeys, key=int))


def aggregate_soil(rows: list[dict]) -> dict[str, float]:
    """0--30 cm 表层按 图斑面积 x 组分比例 x 土层厚度 加权平均；任一属性无有效权重即失败。"""
    def num(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return math.nan
        return value

    weighted = []
    for row in rows:
        acres, pct = num(row.get("muacres")), num(row.get("comppct_r"))
        top, bottom = num(row.get("hzdept_r")), num(row.get("hzdepb_r"))
        thickness = min(bottom, 30.0) - max(top, 0.0)
        weight = acres * pct * thickness
        if math.isfinite(weight) and weight > 0:
            weighted.append((weight, row))
    result = {}
    for name, column in _SOIL_MAP.items():
        pairs = [(w, num(r.get(column))) for w, r in weighted]
        pairs = [(w, v) for w, v in pairs if math.isfinite(v)]
        total = sum(w for w, _ in pairs)
        if total <= 0:
            raise ValueError(f"no weighted values for {name}")
        result[name] = sum(w * v for w, v in pairs) / total
    return result


def candidate_fips(root: Path) -> list[str]:
    """STATES 内 2017--2022 任一年出现在 USDA 玉米县级记录中的县。"""
    out = set()
    for year in YEARS:
        with open(root / usda_relpath(year), newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                state, county = row["state_ansi"].strip().zfill(2), row["county_ansi"].strip()
                if row["commodity_desc"] == "CORN" and state in STATES and county.isdigit():
                    out.add(state + county.zfill(3))
    return sorted(out)


def fetch_soil(root: Path, fips_list: Iterable[str], *, workers: int = 4) -> dict:
    raw_dir = root / "soil" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch(fips):
        target = raw_dir / f"{fips}.json"
        if target.is_file():
            return "cached"
        method, rows = "areasymbol", soil_rows_by_areasymbol(fips)
        if not rows:
            method, rows = soil_rows_by_geometry(fips)
        write_atomic(target, json.dumps({
            "fips": fips, "method": method, "source": SDA_URL,
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "rows": rows,
        }).encode())
        return method

    fips_list = sorted(set(fips_list))
    failures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, f): f for f in fips_list}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                failures[futures[future]] = f"query failed: {exc!r}"
            if i % 100 == 0 or i == len(fips_list):
                print(f"[soil] {i}/{len(fips_list)} failures={len(failures)}", flush=True)

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["FIPS", *SOIL_FEATURES])
    for fips in fips_list:
        if fips in failures:
            continue
        try:
            values = aggregate_soil(json.loads((raw_dir / f"{fips}.json").read_text())["rows"])
        except ValueError as exc:
            failures[fips] = f"incomplete: {exc}"
            continue
        writer.writerow([fips, *(repr(values[name]) for name in SOIL_FEATURES)])
    write_atomic(root / SOIL_RELPATH, buffer.getvalue().encode())
    write_atomic(root / "soil" / "soil_failures.json", json.dumps(failures, indent=1, sort_keys=True).encode())
    return failures
# ---------------------------------------------------------------------------
# 县年审计
# ---------------------------------------------------------------------------

AUDIT_FIELDS = (
    "fips", "year", "state", "county", "status", "reason", "usda_status",
    "weather_days", "weather_grids", "remote_months", "remote_grids",
    "soil_status", "grid_match_status", "long_term_status", "grid_stability_status",
)


def row_ok(row: dict) -> bool:
    """单个县年是否满足全部模态完整性条件（不依赖其他年份）。"""
    return (
        row["usda_status"] == "ok"
        and row["weather_days"] == WEATHER_DAYS
        and row["weather_grids"] > 0
        and row["remote_months"] == len(REMOTE_MONTHS)
        and row["remote_grids"] == row["weather_grids"]
        and row["soil_status"] == "ok"
        and row["grid_match_status"] == "ok"
        and row["long_term_status"] in ("ok", "not_required")
        and row["grid_stability_status"] == "ok"
    )


def complete_counties(rows: Iterable[dict], years: tuple[int, ...] = YEARS) -> set[str]:
    """只保留指定年份全部存在且全部完整的县。"""
    by_county: dict[str, dict[int, dict]] = {}
    for row in rows:
        years_seen = by_county.setdefault(row["fips"], {})
        if row["year"] in years_seen:
            raise ValueError(f"duplicate audit row {row['fips']}_{row['year']}")
        years_seen[row["year"]] = row
    return {
        fips for fips, seen in by_county.items()
        if all(y in seen and row_ok(seen[y]) for y in years)
    }


def finalize_audit(rows: Iterable[dict], years: tuple[int, ...] = YEARS) -> list[dict]:
    """写入 retained/excluded；本身完整但因其他年份不完整而被排除的县年给出明确原因。"""
    rows = [dict(r) for r in rows]
    retained = complete_counties(rows, years)
    bad_years: dict[str, list[int]] = {}
    for row in rows:
        if not row_ok(row):
            bad_years.setdefault(row["fips"], []).append(row["year"])
    for row in rows:
        row["status"] = "retained" if row["fips"] in retained else "excluded"
        if row["status"] == "retained":
            row["reason"] = ""
        elif row_ok(row) or not row.get("reason"):
            others = sorted(bad_years.get(row["fips"], []))
            row["reason"] = row.get("reason") or (
                f"county incomplete in years {others}" if others else "county missing required years")
    return rows


def write_audit(rows: Iterable[dict], path: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["fips"], r["year"]))
    for row in rows:
        missing = [name for name in AUDIT_FIELDS if name not in row]
        if missing:
            raise ValueError(f"audit row {row.get('fips')}_{row.get('year')} missing {missing}")
    text = "".join(json.dumps({k: row[k] for k in AUDIT_FIELDS}, ensure_ascii=False) + "\n" for row in rows)
    write_atomic(path, text.encode())


def verify_local(root: Path) -> int:
    """只读扫描源数据；只写 download_audit.jsonl，不下载、不删除、不写正式 JSONL。"""
    from train_dataset.build_dataset import audit_sources

    rows = finalize_audit(audit_sources(root))
    write_audit(rows, root / AUDIT_NAME)
    retained = sorted({r["fips"] for r in rows if r["status"] == "retained"})
    by_state = Counter(STATES[f[:2]][1] for f in retained)
    # 按县年自身失败的模态计数；"county incomplete" 只是继承原因，不单列
    reasons = Counter(part.strip().split(":")[0] for r in rows if r["status"] == "excluded"
                      for part in r["reason"].split(";")
                      if part.strip() and not part.strip().startswith("county "))
    files = {
        "usda": len(list(root.glob("USDA Crop Dataset/data/Corn/*/*.csv"))),
        "hrrr": len(list(root.glob("WRF-HRRR Computed Dataset/data/*/*/*.csv"))),
        "sentinel": len(list(root.glob("Sentinel-2 Imagery/data/*/*/*.h5"))),
        "soil": int((root / SOIL_RELPATH).is_file()),
        "adjacency": int((root / ADJACENCY_RELPATH).is_file()),
    }
    print(json.dumps({
        "candidates": len({r["fips"] for r in rows}), "retained_counties": len(retained),
        "retained_by_state": dict(sorted(by_state.items())), "files": files,
        "excluded_reasons": dict(reasons.most_common()), "audit": str(root / AUDIT_NAME),
    }, indent=1, ensure_ascii=False))
    return 0 if retained else 1


def audit_only(root: Path, revision: str = REVISION) -> int:
    """只列远端清单并与本地文件大小对照；不调用任何下载函数。"""
    remote = list_remote(revision)
    inventory = parse_hf_inventory(remote)
    targets = [remote[p] for p in inventory.paths()]
    local_ok = [t for t in targets
                if (root / local_relpath(t.path)).is_file()
                and (root / local_relpath(t.path)).stat().st_size == t.size]
    pending = [t for t in targets if t not in local_ok]
    print(json.dumps({
        "revision": revision, "targets": len(targets), "remote_missing": inventory.missing(),
        "total_gb": round(sum(t.size for t in targets) / 1e9, 2),
        "local_present": len(local_ok), "pending": len(pending),
        "pending_gb": round(sum(t.size for t in pending) / 1e9, 2),
    }, indent=1))
    return 1 if inventory.missing() else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--verify-local", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="仅用于临时测试")
    parser.add_argument("--seed-root", type=Path, default=None,
                        help="只读复用已有本地副本：digest 与远端一致才硬链接，否则重新下载")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-soil", action="store_true")
    args = parser.parse_args(argv)
    root = args.data_root

    if args.audit_only:
        return audit_only(root, args.revision)
    if args.verify_local:
        return verify_local(root)

    if args.seed_root and args.seed_root.resolve() == root.resolve():
        parser.error("--seed-root must differ from --data-root")
    root.mkdir(parents=True, exist_ok=True)
    counts = download_sources(list_remote(args.revision), root, revision=args.revision,
                              seed_root=args.seed_root, workers=args.workers)
    print(f"[download] done {dict(counts)}", flush=True)
    print(f"[adjacency] sha256 {fetch_adjacency(root)}", flush=True)
    if not args.skip_soil:
        failures = fetch_soil(root, candidate_fips(root))
        print(f"[soil] counties without complete soil: {len(failures)}", flush=True)
    print("[next] run --verify-local to write download_audit.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
