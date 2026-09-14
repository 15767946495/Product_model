from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TFT_model"))

from cropnet_protocol import CROPNET_FIVE_STATES  # noqa: E402
from data import (  # noqa: E402
    AgricultureImageDataset,
    build_ag_paths,
    select_ag_dates,
    split_samples_by_year,
    validate_five_state_sample,
)


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


def test_build_ag_paths_returns_two_ag_quarter_paths(ag_fixture):
    root, sample = ag_fixture

    paths = build_ag_paths(sample, root)

    assert len(paths) == 2
    assert all(path.name.startswith("Agriculture_") for path in paths)
    assert all("Agriculture" in str(path) for path in paths)
    assert paths[0].name.endswith("2020-04-01_2020-06-30.h5")
    assert paths[1].name.endswith("2020-07-01_2020-09-30.h5")


def test_select_ag_dates_requires_the_fixed_twelve_dates():
    assert select_ag_dates({date: object() for date in AG_DATES}) == AG_DATES


def test_ag_dataset_rejects_ndvi_path(tmp_path):
    sample = {
        "FIPS": "17001", "Year": 2020, "State": "illinois",
        "sentinel": ["data/NDVI/2020/IL/Vegetation_17001.h5"],
    }

    with pytest.raises(ValueError, match="sample 0.*NDVI|Vegetation"):
        AgricultureImageDataset([sample], ag_root=tmp_path, train=False)


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
