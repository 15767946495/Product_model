from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TFT_model"))

from cropnet_protocol import CROPNET_FIVE_STATES  # noqa: E402
from data import split_samples_by_year, validate_five_state_sample  # noqa: E402


def test_split_samples_by_year_uses_five_state_protocol():
    samples = [
        {"State": "illinois", "Year": 2017, "FIPS": "17001"},
        {"State": "iowa", "Year": 2020, "FIPS": "19001"},
        {"State": "new york", "Year": 2021, "FIPS": "36001"},
        {"State": "mississippi", "Year": 2022, "FIPS": "28001"},
    ]

    splits = split_samples_by_year(samples)

    assert [row["Year"] for row in splits["train"]] == [2017, 2020]
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


def test_split_uses_custom_year_groups():
    splits = split_samples_by_year(
        [{"State": "illinois", "Year": 2023, "FIPS": "17001"}],
        train_years=(2023,),
        val_years=(),
        test_years=(),
    )

    assert splits["train"][0]["Year"] == 2023


def test_validate_five_state_sample_accepts_normalized_state_and_protocol_year():
    validate_five_state_sample({"State": " Louisiana ", "Year": "2021"})


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


def test_five_state_constant_contains_only_required_states():
    assert CROPNET_FIVE_STATES == {
        "illinois",
        "iowa",
        "louisiana",
        "mississippi",
        "new york",
    }
