import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmst_vit import sentinel


YEARS = tuple(range(2017, 2023))
USDA_HEADER = [
    "commodity_desc",
    "reference_period_desc",
    "state_ansi",
    "county_ansi",
    "county_name",
    "state_name",
    "YIELD, MEASURED IN BU / ACRE",
]


def _write_usda(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=USDA_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _usda_row(state="17", county="001", commodity="CORN", period="YEAR", yield_value="1,234"):
    return {
        "commodity_desc": commodity,
        "reference_period_desc": period,
        "state_ansi": state,
        "county_ansi": county,
        "county_name": 'St. Clair, "North"',
        "state_name": "Illinois",
        "YIELD, MEASURED IN BU / ACRE": yield_value,
    }


def _write_complete_weather(path, fips="17001", missing=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = missing or set()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["FIPS Code", "Daily/Monthly", "Year", "Month", "Day"])
        writer.writeheader()
        for month, day in sentinel.WEATHER_PROTOCOL_DATES:
            if (month, day) not in missing:
                writer.writerow({"FIPS Code": fips, "Daily/Monthly": "Daily", "Year": 2020, "Month": month, "Day": day})
        writer.writerow({"FIPS Code": fips, "Daily/Monthly": "Monthly", "Year": 2020, "Month": 4, "Day": 1})


def _basic_tree_entry(size=4):
    return {
        "type": "file",
        "path": "Sentinel-2 Imagery/data/AG/2020/IL/Agriculture_17_IL_2020-04-01_2020-06-30.h5",
        "size": size,
    }


def _make_sync_fixture(tmp_path):
    usda = tmp_path / "usda"
    weather = tmp_path / "weather"
    _write_usda(
        usda / "USDA_Corn_County_2020.csv",
        [_usda_row(), _usda_row(state="38"), _usda_row(state="19", commodity="SOYBEANS")],
    )
    _write_complete_weather(weather / "2020" / "IL" / "weather.csv")
    _write_complete_weather(weather / "2020" / "ND" / "weather.csv", fips="38001")
    manifest = tmp_path / "sentinel_urls.jsonl"
    rows = [
        {"oss_key": "sentinel/old-il", "image_type": "AG", "state_ansi": "17", "year": 2020},
        {"oss_key": "sentinel/old-nd", "image_type": "AG", "state_ansi": "38", "year": 2020},
        {"oss_key": "sentinel/old-ndvi", "image_type": "NDVI", "state_ansi": "17", "year": 2020},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return usda, weather, manifest


def test_state_allowlist_uses_parsed_values_and_ansi_codes():
    assert sentinel.normalize_state_name(" new   york ") == "new york"
    assert sentinel.is_five_state_fips("17001")
    assert sentinel.is_five_state_fips("36", county_ansi="001")
    assert not sentinel.is_five_state_fips("38001")
    assert sentinel.is_five_state_fips("", state="Iowa")


def test_filter_usda_rows_keeps_only_target_five_state_corn_year_rows():
    rows = [
        _usda_row(),
        _usda_row(state="38"),
        _usda_row(state="19", commodity="SOYBEANS"),
        _usda_row(state="22", period="WEEKLY"),
        _usda_row(state="36", county="unknown"),
    ]

    assert sentinel.filter_usda_rows(rows, {"17001", "19001"}) == [rows[0]]


def test_weather_fips_requires_complete_daily_protocol_calendar(tmp_path):
    weather = tmp_path / "weather"
    _write_complete_weather(weather / "2020" / "IL" / "complete.csv", "17001")
    _write_complete_weather(weather / "2020" / "IA" / "incomplete.csv", "19001", {(9, 28)})
    _write_complete_weather(weather / "2020" / "ND" / "outside.csv", "38001")

    assert sentinel._weather_fips(weather, [2020]) == {"17001"}


def test_classify_url_rows_deletes_non_five_state_or_ndvi():
    rows = [
        {"oss_key": "sentinel/ag-il", "image_type": "AG", "state_ansi": "17", "year": 2020},
        {"oss_key": "sentinel/ag-nd", "image_type": "AG", "state_ansi": "38", "year": 2020},
        {"oss_key": "sentinel/vegetation-il", "image_type": "NDVI", "state_ansi": "17", "year": 2020},
    ]
    result = sentinel.classify_url_rows(rows)
    assert result["keep"] == [rows[0]]
    assert result["delete"] == [rows[1], rows[2]]


def test_plan_lists_only_missing_or_wrong_size_ag_sources(tmp_path):
    usda, weather, manifest = _make_sync_fixture(tmp_path)
    download_root = tmp_path / "download"
    entry = sentinel.parse_hf_tree([_basic_tree_entry()], {"17001"}, {2020}, {"AG"})[0]
    local = download_root / entry["path"]
    local.parent.mkdir(parents=True)
    local.write_bytes(b"good")

    plan = sentinel.build_sync_plan(usda, weather, manifest, YEARS, [entry], tmp_path / "county", download_root)
    assert plan["ag_source_files_to_download"] == []

    local.write_bytes(b"bad-size")
    plan = sentinel.build_sync_plan(usda, weather, manifest, YEARS, [entry], tmp_path / "county", download_root)
    assert plan["ag_source_files_to_download"] == [entry["path"]]


def test_dry_run_does_not_delete_rewrite_or_create_bucket(tmp_path, monkeypatch):
    usda, weather, manifest = _make_sync_fixture(tmp_path)
    source = usda / "USDA_Corn_County_2020.csv"
    original = source.read_bytes()
    monkeypatch.setattr(sentinel, "fetch_hf_files", lambda *args: [])
    monkeypatch.setattr(sentinel, "create_oss_bucket", lambda *args: pytest.fail("dry-run created OSS bucket"))

    result = sentinel.sync_cropnet_five_state(
        usda_dir=usda,
        weather_dir=weather,
        years=YEARS,
        url_manifest=manifest,
        run_root=tmp_path / "runs",
        execute=False,
    )

    assert result["dry_run"] is True
    assert result["plan"]["target_fips"] == ["17001"]
    assert result["plan"]["usda"]["2020"] == {
        "keep": 1,
        "delete": 2,
        "path": str(source),
    }
    assert (weather / "2020" / "ND").exists()
    assert source.read_bytes() == original


def test_sync_counties_downloads_extracts_then_uploads_and_keeps_source(tmp_path, monkeypatch):
    entry = dict(_basic_tree_entry(), image_type="AG", year=2020, fips=["17001"])
    download_root = tmp_path / "download"
    extract_root = tmp_path / "county"
    source = download_root / entry["path"]
    events = []

    def download_one(item, destination, resume=True):
        events.append("download")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"good")
        return "downloaded"

    def extract(entries, source_root, output_root):
        if not source.exists():
            return []
        events.append("extract")
        target = output_root / "AG" / "2020" / "17001" / "county.h5"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"county")
        return [dict(entry, path=str(target.relative_to(output_root)), size=6)]

    def upload(entries, local_root, bucket, url_output, prefix):
        events.append("upload")
        row = dict(entries[0], oss_key="sentinel/county", status="uploaded")
        url_output.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return {"uploaded": 1, "skipped": 0, "failed": [], "url_manifest": str(url_output)}

    monkeypatch.setattr(sentinel, "download_one", download_one)
    monkeypatch.setattr(sentinel, "extract_target_counties", extract)
    monkeypatch.setattr(sentinel, "upload_manifest_to_oss", upload)
    monkeypatch.setattr(sentinel.time, "sleep", lambda _: None)

    result = sentinel.sync_counties_to_oss([entry], download_root, extract_root, object(), tmp_path / "new.jsonl")

    assert result["failed"] == []
    assert events == ["download", "extract", "upload"]
    assert source.read_bytes() == b"good"
    assert not (extract_root / "AG" / "2020" / "17001" / "county.h5").exists()


def test_execute_mock_backs_up_roundtrips_usda_and_writes_ag_only_manifest(tmp_path, monkeypatch):
    usda, weather, manifest = _make_sync_fixture(tmp_path)
    download_root = tmp_path / "download"
    source = download_root / _basic_tree_entry()["path"]
    source.parent.mkdir(parents=True)
    source.write_bytes(b"raw-sentinel")
    events = []

    def sync_ag(entries, download, extract, bucket, output, prefix):
        events.append("ag-sync")
        row = {
            "oss_key": "sentinel/new-il",
            "path": "AG/2020/17001/county.h5",
            "image_type": "AG",
            "state": "IL",
            "fips": ["17001"],
            "year": 2020,
            "status": "uploaded",
        }
        output.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return {"source_files": 1, "processed": 1, "uploaded": 1, "skipped": 0, "failed": []}

    class Bucket:
        def delete_object(self, key):
            events.append(f"delete:{key}")

    monkeypatch.setattr(sentinel, "sync_counties_to_oss", sync_ag)
    result = sentinel.sync_cropnet_five_state(
        usda_dir=usda,
        weather_dir=weather,
        years=YEARS,
        url_manifest=manifest,
        run_root=tmp_path / "runs",
        download_root=download_root,
        extract_root=tmp_path / "county",
        bucket=Bucket(),
        execute=True,
        tree_payload=[_basic_tree_entry(size=len(b"raw-sentinel"))],
    )

    assert events[:1] == ["ag-sync"]
    assert source.read_bytes() == b"raw-sentinel"
    backup = Path(result["run_dir"]) / "usda-backup" / "USDA_Corn_County_2020.csv"
    assert backup.exists()
    with (usda / backup.name).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == USDA_HEADER
    assert rows == [_usda_row()]
    assert rows[0]["county_name"] == 'St. Clair, "North"'
    output_rows = [json.loads(line) for line in (tmp_path / "sentinel_urls_cropnet5_ag.jsonl").read_text().splitlines()]
    assert all(row["image_type"] == "AG" and sentinel._row_is_five_state(row) for row in output_rows)
    assert {row["oss_key"] for row in output_rows} == {"sentinel/old-il", "sentinel/new-il"}
    assert not (weather / "2020" / "ND").exists()


def test_oss_delete_failure_writes_status_and_skips_remaining_cleanup(tmp_path, monkeypatch):
    usda, weather, manifest = _make_sync_fixture(tmp_path)
    monkeypatch.setattr(
        sentinel,
        "sync_counties_to_oss",
        lambda entries, download, extract, bucket, output, prefix: (
            output.write_text("", encoding="utf-8")
            or {"source_files": 0, "processed": 0, "uploaded": 0, "skipped": 0, "failed": []}
        ),
    )

    class Bucket:
        def delete_object(self, key):
            if key.endswith("ndvi"):
                raise RuntimeError("denied")

    with pytest.raises(RuntimeError, match="OSS deletion failed"):
        sentinel.sync_cropnet_five_state(
            usda_dir=usda,
            weather_dir=weather,
            years=YEARS,
            url_manifest=manifest,
            run_root=tmp_path / "runs",
            bucket=Bucket(),
            execute=True,
            tree_payload=[],
        )

    run_dir = next((tmp_path / "runs").iterdir())
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    delete_status = next(item for item in status["status"] if item["step"] == "delete-oss")
    assert delete_status["outcome"] == "failed"
    assert delete_status["result"]["deleted"] == ["sentinel/old-nd"]
    assert delete_status["result"]["failed"][0]["oss_key"] == "sentinel/old-ndvi"
    assert {item["step"] for item in status["status"] if item.get("outcome") == "not-run"} == {
        "delete-weather",
        "rewrite-usda",
        "rewrite-url-manifest",
    }
    assert (weather / "2020" / "ND").exists()


def test_cli_sync_mode_defaults_to_dry_run(tmp_path):
    usda, weather, manifest = _make_sync_fixture(tmp_path)
    tree = tmp_path / "tree.json"
    tree.write_text("[]\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mmst_vit.sentinel",
            "--sync-cropnet-five-state",
            "--usda-dir",
            str(usda),
            "--weather-dir",
            str(weather),
            "--url-manifest",
            str(manifest),
            "--sync-run-root",
            str(tmp_path / "runs"),
            "--tree-json",
            str(tree),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["dry_run"] is True
    assert (weather / "2020" / "ND").exists()


def test_delete_oss_objects_reports_successes_and_failures():
    class Bucket:
        def delete_object(self, key):
            if key == "sentinel/denied":
                raise RuntimeError("denied")

    with pytest.raises(sentinel.OssDeletionError) as error:
        sentinel.delete_oss_objects(
            [{"oss_key": "sentinel/deleted"}, {"oss_key": "sentinel/denied"}],
            Bucket(),
        )

    assert error.value.result["deleted"] == ["sentinel/deleted"]
    assert error.value.result["failed"] == [{"oss_key": "sentinel/denied", "error": "denied"}]
