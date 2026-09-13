import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmst_vit.config import official_sample_record
from mmst_vit.manifest import ALLOWED_STATES, build_valid_samples
from mmst_vit.official import write_official_manifests, write_official_split


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
    assert len(long_term[0]) == 60
    assert len(sentinel) == 4
    assert set(sentinel) == {
        "mmst_vit/download/Sentinel-2 Imagery/data/AG/2020/IL/"
        "Agriculture_17_IL_2020-04-01_2020-06-30.h5",
        "mmst_vit/download/Sentinel-2 Imagery/data/AG/2020/IL/"
        "Agriculture_17_IL_2020-07-01_2020-09-30.h5",
        "mmst_vit/download/Sentinel-2 Imagery/data/NDVI/2020/IL/"
        "Vegetation_17_IL_2020-04-01_2020-06-30.h5",
        "mmst_vit/download/Sentinel-2 Imagery/data/NDVI/2020/IL/"
        "Vegetation_17_IL_2020-07-01_2020-09-30.h5",
    }


def test_official_sample_record_rejects_unknown_state(tmp_path):
    with pytest.raises(ValueError, match="unsupported state.*Atlantis"):
        official_sample_record("99001", 2020, "Atlantis", "Unknown", tmp_path)


def test_write_official_manifests_rejects_state_outside_protocol(tmp_path):
    sample = {
        "FIPS": "19001",
        "Year": 2020,
        "State": "Iowa",
        "County": "Excluded",
    }

    with pytest.raises(ValueError, match="sample 0.*Iowa"):
        write_official_manifests([sample], tmp_path / "output", tmp_path)


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


def test_official_cli_builds_all_split_json_arrays_from_small_manifests(tmp_path):
    manifest_dir = tmp_path / "manifests"
    output_dir = tmp_path / "official"
    data_root = tmp_path / "data"
    manifest_dir.mkdir()
    sample_by_year = {
        "train": (2020, "train"),
        "val": (2021, "val"),
        "test": (2022, "test"),
    }
    for split, (year, _) in sample_by_year.items():
        sample = {
            "FIPS": "17001",
            "Year": year,
            "State": "Illinois",
            "County": split,
        }
        (manifest_dir / f"valid-no-ia.{split}.jsonl").write_text(
            json.dumps(sample) + "\n", encoding="utf-8"
        )
        record = official_sample_record(
            sample["FIPS"], sample["Year"], sample["State"], sample["County"], data_root
        )
        paths = [
            record["data"]["USDA"],
            *record["data"]["HRRR"]["short_term"],
            *(path for context in record["data"]["HRRR"]["long_term"] for path in context),
            *record["data"]["sentinel"],
        ]
        for relative in paths:
            path = data_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mmst_vit.official",
            "--manifest-dir",
            str(manifest_dir),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {"test": 1, "train": 1, "val": 1}
    for split in sample_by_year:
        output = output_dir / f"{split}.official.no-ia.json"
        assert len(json.loads(output.read_text(encoding="utf-8"))) == 1
