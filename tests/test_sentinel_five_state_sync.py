import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmst_vit import sentinel


def test_state_allowlist_uses_parsed_values_and_ansi_codes():
    assert sentinel.normalize_state_name(" new   york ") == "new york"
    assert sentinel.is_five_state_fips("17001")
    assert sentinel.is_five_state_fips("36", county_ansi="001")
    assert not sentinel.is_five_state_fips("38001")
    assert sentinel.is_five_state_fips("", state="Iowa")


def test_filter_usda_rows_keeps_only_five_state_corn_year_rows():
    rows = [
        {"commodity_desc": "CORN", "reference_period_desc": "YEAR", "state_ansi": "17", "county_ansi": "001"},
        {"commodity_desc": "CORN", "reference_period_desc": "YEAR", "state_ansi": "38", "county_ansi": "001"},
        {"commodity_desc": "SOYBEANS", "reference_period_desc": "YEAR", "state_ansi": "19", "county_ansi": "001"},
    ]
    kept = sentinel.filter_usda_rows(rows)
    assert [row["state_ansi"] for row in kept] == ["17", "38"]


def test_classify_url_rows_deletes_non_five_state_or_ndvi():
    rows = [
        {"oss_key": "sentinel/ag-il", "image_type": "AG", "state_ansi": "17", "year": 2020},
        {"oss_key": "sentinel/ag-nd", "image_type": "AG", "state_ansi": "38", "year": 2020},
        {"oss_key": "sentinel/vegetation-il", "image_type": "NDVI", "state_ansi": "17", "year": 2020},
    ]
    result = sentinel.classify_url_rows(rows)
    assert result["keep"] == [rows[0]]
    assert result["delete"] == [rows[1], rows[2]]


def test_dry_run_does_not_delete_or_rewrite(tmp_path, monkeypatch):
    usda = tmp_path / "usda"
    weather = tmp_path / "weather"
    usda.mkdir()
    (weather / "2020" / "IL").mkdir(parents=True)
    (weather / "2020" / "ND").mkdir(parents=True)
    source = usda / "USDA_Corn_County_2020.csv"
    source.write_text(
        "commodity_desc,reference_period_desc,state_ansi,county_ansi\n"
        "CORN,YEAR,17,001\nCORN,YEAR,38,001\n", encoding="utf-8"
    )
    (weather / "2020" / "IL" / "weather.csv").write_text("FIPS Code\n17001\n", encoding="utf-8")
    (weather / "2020" / "ND" / "weather.csv").write_text("FIPS Code\n38001\n", encoding="utf-8")
    url_manifest = tmp_path / "sentinel_urls.jsonl"
    url_manifest.write_text(json.dumps({"oss_key": "old", "image_type": "NDVI", "state_ansi": "17"}) + "\n", encoding="utf-8")

    monkeypatch.setattr(sentinel, "fetch_hf_files", lambda *args: [])
    result = sentinel.sync_cropnet_five_state(
        usda_dir=usda,
        weather_dir=weather,
        years=set(range(2017, 2023)),
        url_manifest=url_manifest,
        run_root=tmp_path / "runs",
        execute=False,
    )

    assert result["dry_run"] is True
    assert result["plan"]["target_fips"] == ["17001"]
    assert result["plan"]["usda"]["2020"]["keep"] == 1
    assert result["plan"]["usda"]["2020"]["delete"] == 1
    assert (weather / "2020" / "ND").exists()
    assert source.read_text(encoding="utf-8").count("CORN") == 2


def test_delete_oss_objects_reports_failure():
    class Bucket:
        def delete_object(self, key):
            raise RuntimeError("denied")

    with pytest.raises(RuntimeError, match="OSS deletion failed"):
        sentinel.delete_oss_objects([{"oss_key": "sentinel/old"}], Bucket())
