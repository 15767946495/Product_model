from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TFT_model"))

from cropnet_protocol import CROPNET_FIVE_STATES  # noqa: E402
from data import (  # noqa: E402
    AgricultureImageDataset,
    build_ag_paths,
    resolve_ag_date_names,
    select_ag_dates,
    split_samples_by_year,
    validate_five_state_sample,
)
import data as data_module  # noqa: E402
from tools.audit_tft_ag_data import audit_ag_samples, build_tft_ag_manifest  # noqa: E402
from mmst_vit.config import tft_ag_quarter_paths  # noqa: E402


AG_DATES = [
    "04-01", "04-15", "05-01", "05-15", "06-01", "06-15",
    "07-01", "07-15", "08-01", "08-15", "09-01", "09-15",
]


@pytest.fixture
def ag_fixture(tmp_path):
    sample = {
        "FIPS": "17001",
        "Year": 2020,
        "State": "illinois",
        "County": "Adams",
    }
    paths = build_ag_paths(sample, tmp_path)
    for path, dates in zip(paths, (AG_DATES[:6], AG_DATES[6:])):
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            county = handle.create_group("17001")
            for time_index, date in enumerate(dates):
                county.create_group(date).create_dataset(
                    "data",
                    data=np.full((2, 224, 224, 3), time_index, dtype=np.uint8),
                )
    return tmp_path, sample


def test_ag_dataset_loads_two_quarters_and_returns_mmst_shape(ag_fixture):
    root, sample = ag_fixture

    item = AgricultureImageDataset([sample], ag_root=root, train=False, seed=0)[0]

    assert item["ag_images"].shape == (12, 2, 3, 224, 224)
    assert item["ag_dates"] == AG_DATES
    assert item["FIPS"] == "17001"
    assert item["grid_count"] == 2
    assert item["Year"] == 2020
    assert item["State"] == "illinois"
    assert item["County"] == "Adams"
    assert item["ag_images"].dtype == torch.float32


def test_ag_dataset_accepts_real_year_prefixed_date_groups(ag_fixture):
    root, sample = ag_fixture
    for path in build_ag_paths(sample, root):
        with h5py.File(path, "a") as handle:
            county = handle[sample["FIPS"]]
            for date in list(county.keys()):
                county.move(date, f"{sample['Year']}-{date}")

    item = AgricultureImageDataset([sample], ag_root=root, train=False, seed=0)[0]

    assert item["ag_images"].shape == (12, 2, 3, 224, 224)


def test_ag_date_parser_rejects_duplicate_suffix_and_wrong_year(ag_fixture):
    root, sample = ag_fixture
    path = build_ag_paths(sample, root)[0]
    with h5py.File(path, "a") as handle:
        county = handle[sample["FIPS"]]
        county.copy("04-01", "2020-04-01")
        with pytest.raises(ValueError, match="duplicate"):
            resolve_ag_date_names(county, AG_DATES[:6], sample["Year"])

        del county["2020-04-01"]
        county.move("04-01", "2019-04-01")
        with pytest.raises(ValueError, match="year"):
            resolve_ag_date_names(county, AG_DATES[:6], sample["Year"])


def test_ag_date_parser_rejects_extra_date_group(ag_fixture):
    root, sample = ag_fixture
    path = build_ag_paths(sample, root)[0]
    with h5py.File(path, "a") as handle:
        county = handle[sample["FIPS"]]
        county.create_group("06-30")
        with pytest.raises(ValueError, match="unexpected"):
            resolve_ag_date_names(county, AG_DATES[:6], sample["Year"])


def test_build_ag_paths_returns_two_ag_quarter_paths(ag_fixture):
    root, sample = ag_fixture

    paths = build_ag_paths(sample, root)

    assert len(paths) == 2
    assert all(path.name.startswith("Agriculture_") for path in paths)
    assert all("Agriculture" in str(path) for path in paths)
    assert paths[0].name.endswith("2020-04-01_2020-06-30.h5")
    assert paths[1].name.endswith("2020-07-01_2020-09-30.h5")


@pytest.mark.parametrize("year", [2016, 2023, "2020", 2020.0, True])
def test_ag_path_helpers_reject_non_protocol_years(ag_fixture, year):
    root, sample = ag_fixture
    bad_sample = dict(sample, Year=year)
    with pytest.raises(ValueError, match="year"):
        build_ag_paths(bad_sample, root)
    with pytest.raises(ValueError, match="year"):
        tft_ag_quarter_paths(sample["FIPS"], year, sample["State"], root)


def test_tft_ag_quarter_paths_returns_only_two_ag_paths(ag_fixture):
    root, sample = ag_fixture
    paths = tft_ag_quarter_paths(sample["FIPS"], sample["Year"], sample["State"], root)
    assert len(paths) == 2
    assert all("/AG/" in str(path) and "NDVI" not in str(path) for path in paths)


def test_select_ag_dates_requires_the_fixed_twelve_dates():
    assert select_ag_dates({date: object() for date in AG_DATES}) == AG_DATES


def test_ag_dataset_rejects_ndvi_path(tmp_path):
    sample = {
        "FIPS": "17001", "Year": 2020, "State": "illinois",
        "sentinel": ["data/NDVI/2020/IL/Vegetation_17001.h5"],
    }

    with pytest.raises(ValueError, match="sample 0.*NDVI|Vegetation"):
        AgricultureImageDataset([sample], ag_root=tmp_path, train=False)


@pytest.mark.parametrize(
    "sentinel",
    [
        ["data/Other/2020/IL/Other_17_IL_2020.h5"],
        {"nested": ["data/AG/2020/IL/not_agriculture.h5"]},
    ],
)
def test_ag_dataset_rejects_any_non_ag_sentinel_path(tmp_path, sentinel):
    sample = {
        "FIPS": "17001", "Year": 2020, "State": "illinois",
        "data": {"sentinel": sentinel},
    }

    with pytest.raises(ValueError, match="sample 0.*AG|Agriculture"):
        AgricultureImageDataset([sample], ag_root=tmp_path, train=False)


def test_ag_dataset_rejects_data_group(ag_fixture):
    root, sample = ag_fixture
    path = build_ag_paths(sample, root)[0]
    with h5py.File(path, "a") as handle:
        del handle["17001"]["04-01"]["data"]
        handle["17001"]["04-01"].create_group("data")

    with pytest.raises(ValueError, match="sample 0.*data"):
        AgricultureImageDataset([sample], ag_root=root, train=False)


@pytest.mark.parametrize(
    "data_kwargs",
    [
        {"shape": (2, 224, 224, 4), "dtype": np.uint8},
        {"shape": (2, 224, 224, 3), "dtype": np.float32},
    ],
)
def test_ag_dataset_rejects_invalid_data_shape_or_dtype(ag_fixture, data_kwargs):
    root, sample = ag_fixture
    path = build_ag_paths(sample, root)[0]
    with h5py.File(path, "a") as handle:
        del handle["17001"]["04-01"]["data"]
        handle["17001"]["04-01"].create_dataset(
            "data", data=np.zeros(data_kwargs["shape"], dtype=data_kwargs["dtype"])
        )

    with pytest.raises(ValueError, match="sample 0.*shape or dtype"):
        AgricultureImageDataset([sample], ag_root=root, train=False)


@pytest.mark.parametrize("missing", ["file", "FIPS", "date", "data"])
def test_ag_dataset_rejects_missing_hdf5_structure(ag_fixture, missing):
    root, sample = ag_fixture
    path = build_ag_paths(sample, root)[0]
    if missing == "file":
        path.unlink()
    else:
        with h5py.File(path, "a") as handle:
            if missing == "FIPS":
                del handle["17001"]
            elif missing == "date":
                del handle["17001"]["04-01"]
            else:
                del handle["17001"]["04-01"]["data"]

    with pytest.raises(ValueError, match="sample 0"):
        AgricultureImageDataset([sample], ag_root=root, train=False)


def test_ag_dataset_rejects_inconsistent_grid_counts(ag_fixture):
    root, sample = ag_fixture
    path = build_ag_paths(sample, root)[0]
    with h5py.File(path, "a") as handle:
        del handle["17001"]["04-15"]["data"]
        handle["17001"]["04-15"].create_dataset(
            "data", data=np.zeros((3, 224, 224, 3), dtype=np.uint8)
        )

    with pytest.raises(ValueError, match="sample 0.*grid"):
        AgricultureImageDataset([sample], ag_root=root, train=False)


def test_ag_dataset_constructor_does_not_change_global_rng_state(ag_fixture):
    root, sample = ag_fixture
    torch.manual_seed(123)
    before = torch.random.get_rng_state()

    AgricultureImageDataset([sample], ag_root=root, train=True, seed=0)

    assert torch.equal(torch.random.get_rng_state(), before)


def test_ag_dataset_calls_select_ag_dates_for_loading(ag_fixture, monkeypatch):
    root, sample = ag_fixture
    calls = []
    original = data_module.select_ag_dates

    def wrapped(group):
        calls.append(tuple(sorted(group.keys())))
        return original(group)

    monkeypatch.setattr(data_module, "select_ag_dates", wrapped)
    AgricultureImageDataset([sample], ag_root=root, train=False)[0]

    assert calls
    assert set(calls[0]) == set(AG_DATES)


def test_ag_dataset_seed_zero_is_deterministic_across_instances(ag_fixture):
    root, sample = ag_fixture
    first = AgricultureImageDataset([sample], ag_root=root, train=True, seed=0)[0]
    second = AgricultureImageDataset([sample], ag_root=root, train=True, seed=0)[0]

    assert torch.equal(first["ag_images"], second["ag_images"])


def test_ag_dataset_getitem_does_not_change_cpu_rng_state(ag_fixture):
    root, sample = ag_fixture
    dataset = AgricultureImageDataset([sample], ag_root=root, train=True, seed=0)
    torch.manual_seed(123)
    before = torch.random.get_rng_state()

    dataset[0]

    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_ag_dataset_getitem_does_not_change_cuda_rng_state(ag_fixture):
    root, sample = ag_fixture
    dataset = AgricultureImageDataset([sample], ag_root=root, train=True, seed=0)
    torch.cuda.manual_seed_all(123)
    before = [torch.cuda.get_rng_state(device) for device in range(torch.cuda.device_count())]

    dataset[0]

    after = [torch.cuda.get_rng_state(device) for device in range(torch.cuda.device_count())]
    assert all(torch.equal(before_state, after_state) for before_state, after_state in zip(before, after))


def test_split_samples_by_year_uses_five_state_protocol():
    samples = [
        {"State": "illinois", "Year": 2017, "FIPS": "17001"},
        {"State": "iowa", "Year": 2018, "FIPS": "19001"},
        {"State": "louisiana", "Year": 2019, "FIPS": "22001"},
        {"State": "iowa", "Year": 2020, "FIPS": "19001"},
        {"State": "new york", "Year": 2021, "FIPS": "36001"},
        {"State": "mississippi", "Year": 2022, "FIPS": "28001"},
    ]

    splits = split_samples_by_year(samples)

    assert [row["Year"] for row in splits["train"]] == [2017, 2018, 2019, 2020]
    assert [row["Year"] for row in splits["val"]] == [2021]
    assert [row["Year"] for row in splits["test"]] == [2022]


def test_split_normalizes_state_and_copies_samples():
    sample = {"State": "  Iowa ", "Year": 2017, "FIPS": "19001"}

    splits = split_samples_by_year([sample])

    assert splits["train"] == [sample]
    assert splits["train"][0] is not sample


def test_split_rejects_state_outside_five_state_protocol():
    with pytest.raises(ValueError, match="unsupported state"):
        split_samples_by_year([{"State": "texas", "Year": 2017, "FIPS": "48001"}])


def test_split_rejects_year_outside_protocol():
    with pytest.raises(ValueError, match="unsupported year"):
        split_samples_by_year([{"State": "illinois", "Year": 2023, "FIPS": "17001"}])


def test_split_rejects_protocol_external_custom_year_group():
    with pytest.raises(ValueError, match="unsupported year"):
        split_samples_by_year(
            [],
            train_years=(2023,),
            val_years=(),
            test_years=(),
        )


def test_validate_five_state_sample_accepts_normalized_state_and_protocol_year():
    validate_five_state_sample({"State": " Louisiana ", "Year": 2021})


@pytest.mark.parametrize(
    "sample, message",
    [
        ({"State": "texas", "Year": 2021}, "unsupported state"),
        ({"State": "illinois", "Year": 2023}, "unsupported year"),
    ],
)
def test_validate_five_state_sample_rejects_protocol_violations(sample, message):
    with pytest.raises(ValueError, match=message):
        validate_five_state_sample(sample)


@pytest.mark.parametrize("year", [2017.9, 2022.1, "2021", True, None])
def test_validate_five_state_sample_rejects_non_integer_years(year):
    with pytest.raises(ValueError, match="invalid year"):
        validate_five_state_sample({"State": "illinois", "Year": year})


@pytest.mark.parametrize("year", [2017.9, 2022.1, "2021", True])
def test_split_rejects_non_integer_sample_years(year):
    with pytest.raises(ValueError, match="unsupported year"):
        split_samples_by_year([{"State": "illinois", "Year": year}])


@pytest.mark.parametrize("years", [(2017.9,), ("2021",), (True,)])
def test_split_rejects_non_integer_custom_year_groups(years):
    with pytest.raises(ValueError, match="invalid year"):
        split_samples_by_year([], train_years=years)


def test_split_rejects_overlapping_year_groups():
    with pytest.raises(ValueError, match="overlap"):
        split_samples_by_year([], train_years=(2017,), val_years=(2017,))


def test_five_state_constant_contains_only_required_states():
    assert CROPNET_FIVE_STATES == {
        "illinois",
        "iowa",
        "louisiana",
        "mississippi",
        "new york",
    }


def test_build_tft_ag_manifest_writes_only_valid_ag_rows(ag_fixture, tmp_path):
    root, sample = ag_fixture
    invalid = dict(sample, FIPS="17003")

    result = build_tft_ag_manifest([sample, invalid], root, tmp_path / "runtime")

    assert sum(result.values()) == 1
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "runtime" / "manifests" / "train.jsonl").read_text().splitlines()
        if line
    ]
    assert len(train_rows) == 1
    assert len(train_rows[0]["ag_paths"]) == 2
    assert all("NDVI" not in path and "Vegetation" not in path for path in train_rows[0]["ag_paths"])

    audit = json.loads((tmp_path / "runtime" / "audit" / "ag_integrity.json").read_text())
    assert audit["valid_count"] == 1
    assert audit["invalid_count"] == 1
    assert audit["invalid_samples"][0]["FIPS"] == "17003"
    assert audit["invalid_samples"][0]["paths"]


def test_audit_ag_samples_reports_invalid_hdf5_structure(ag_fixture):
    root, sample = ag_fixture
    path = build_ag_paths(sample, root)[0]
    with h5py.File(path, "a") as handle:
        del handle["17001"]["04-01"]["data"]

    audit = audit_ag_samples([sample], root)

    assert audit["valid_count"] == 0
    assert audit["invalid_count"] == 1
    assert "data" in audit["invalid_samples"][0]["error"]


def test_audit_ag_samples_records_non_group_fips_and_continues(ag_fixture):
    root, sample = ag_fixture
    path = build_ag_paths(sample, root)[0]
    with h5py.File(path, "a") as handle:
        del handle[sample["FIPS"]]
        handle.create_dataset(sample["FIPS"], data=np.zeros((1,), dtype=np.uint8))

    valid = dict(sample, Year=2021)
    for valid_path, dates in zip(build_ag_paths(valid, root), (AG_DATES[:6], AG_DATES[6:])):
        valid_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(valid_path, "w") as handle:
            county = handle.create_group(valid["FIPS"])
            for date in dates:
                county.create_group(date).create_dataset(
                    "data", data=np.zeros((2, 224, 224, 3), dtype=np.uint8)
                )

    audit = audit_ag_samples([sample, valid], root)
    assert audit["valid_count"] == 1
    assert audit["invalid_count"] == 1
    assert "HDF5 group" in audit["invalid_samples"][0]["error"]


def test_audit_ag_samples_records_invalid_metadata_and_continues(ag_fixture):
    root, sample = ag_fixture
    invalid = dict(sample, State="texas")
    audit = audit_ag_samples([invalid, sample], root)
    assert audit["valid_count"] == 1
    assert audit["invalid_count"] == 1
    assert audit["invalid_samples"][0]["FIPS"] == sample["FIPS"]
    assert "unsupported AG state" in audit["invalid_samples"][0]["error"]


@pytest.mark.parametrize("sample", [None, {"State": "illinois"}, {"State": "illinois", "Year": 2023}])
def test_audit_ag_samples_records_invalid_input_rows_and_continues(ag_fixture, sample):
    root, valid = ag_fixture
    audit = audit_ag_samples([sample, valid], root)
    assert audit["valid_count"] == 1
    assert audit["invalid_count"] == 1
    assert len(audit["invalid_samples"]) == 1


def test_manifest_generation_uses_one_audit_result_per_input_sample(ag_fixture, tmp_path, monkeypatch):
    root, sample = ag_fixture
    import tools.audit_tft_ag_data as audit_module

    calls = []
    original = audit_module._audit_one

    def wrapped(sample, ag_root, index, cache):
        calls.append(index)
        return original(sample, ag_root, index, cache)

    monkeypatch.setattr(audit_module, "_audit_one", wrapped)
    build_tft_ag_manifest([sample], root, tmp_path / "runtime")
    assert calls == [0]


def test_manifest_records_non_dict_shared_rows_and_reports_split_input_counts(ag_fixture, tmp_path):
    root, sample = ag_fixture
    output = tmp_path / "runtime"
    result = build_tft_ag_manifest([None, ["bad"], sample], root, output)

    assert result == {"train": 1, "val": 0, "test": 0}
    train_rows = (output / "manifests" / "train.jsonl").read_text().splitlines()
    assert len(train_rows) == 1
    audit = json.loads((output / "audit" / "ag_integrity.json").read_text())
    assert audit["valid_count"] == 1
    assert audit["invalid_count"] == 2
    assert audit["split_counts"] == {
        "train": {"input_count": 1, "valid_manifest_count": 1, "invalid_count": 0},
        "val": {"input_count": 0, "valid_manifest_count": 0, "invalid_count": 0},
        "test": {"input_count": 0, "valid_manifest_count": 0, "invalid_count": 0},
    }
    assert len(audit["invalid_samples"]) == 2
    assert audit["invalid_split_count"] == 2
