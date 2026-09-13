import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmst_vit.config import official_sample_record
from mmst_vit.manifest import ALLOWED_STATES, build_valid_samples
from mmst_vit.official import write_official_split


def test_manifest_filters_parsed_state_values(tmp_path):
    usda = tmp_path / "usda"
    weather = tmp_path / "weather"
    usda.mkdir()
    (weather / "2020" / "Iowa").mkdir(parents=True)
    (weather / "2020" / "Illinois").mkdir(parents=True)
    header = "commodity_desc,reference_period_desc,state_ansi,county_ansi,county_name,state_name,\"YIELD, MEASURED IN BU / ACRE\"\n"
    rows = (
        "CORN,YEAR,19,001,Excluded,Iowa,100\n"
        "CORN,YEAR,17,001,Allowed,Illinois,120\n"
    )
    (usda / "USDA_Corn_County_2020.csv").write_text(header + rows, encoding="utf-8")
    weather_header = "FIPS Code,Daily/Monthly\n"
    (weather / "2020" / "Iowa" / "weather.csv").write_text(weather_header + "19001,Daily\n", encoding="utf-8")
    (weather / "2020" / "Illinois" / "weather.csv").write_text(weather_header + "17001,Daily\n", encoding="utf-8")

    samples = build_valid_samples(usda, weather, [2020])

    assert {sample["State"].lower() for sample in samples} <= ALLOWED_STATES
    assert [sample["FIPS"] for sample in samples] == ["17001"]


def test_official_sample_record_has_six_short_term_months_and_four_sentinel_paths(tmp_path):
    record = official_sample_record("17001", 2020, "Illinois", "Allowed", tmp_path)
    short_term = record["data"]["HRRR"]["short_term"]
    long_term = record["data"]["HRRR"]["long_term"]
    sentinel = record["data"]["sentinel"]

    assert len(short_term) == 6
    assert [path.rsplit("-", 1)[-1].removesuffix(".csv") for path in short_term] == ["04", "05", "06", "07", "08", "09"]
    assert len(long_term) == 1
    assert len(long_term) == 1
    assert len(long_term[0]) == 60
    assert len(sentinel) == 4
    assert any("Agriculture_17_IL" in path for path in sentinel)
    assert any("Vegetation_17_IL" in path for path in sentinel)
    assert {"2020-04-01_2020-06-30", "2020-07-01_2020-09-30"} <= {
        path.rsplit("/", 1)[-1].split("_", 3)[-1].removesuffix(".h5") for path in sentinel
    }


def test_official_split_validates_every_path_before_writing(tmp_path):
    record = official_sample_record("17001", 2020, "Illinois", "Allowed", tmp_path)
    output = tmp_path / "train.official.no-ia.json"

    with pytest.raises(FileNotFoundError):
        write_official_split(output, [record], tmp_path)
    assert not output.exists()


def test_official_split_writes_json_array_after_all_paths_exist(tmp_path):
    record = official_sample_record("17001", 2020, "Illinois", "Allowed", tmp_path)
    paths = [
        record["data"]["USDA"],
        *record["data"]["HRRR"]["short_term"],
        *(path for context in record["data"]["HRRR"]["long_term"] for path in context),
        *record["data"]["sentinel"],
    ]
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    output = tmp_path / "train.official.no-ia.json"

    write_official_split(output, [record], tmp_path)

    assert json.loads(output.read_text(encoding="utf-8")) == [record]
