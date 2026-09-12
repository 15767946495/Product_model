from pathlib import Path
import sys

import pytest
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train_dataset"))
sys.path.insert(0, str(ROOT / "TFT_model"))
sys.path.insert(0, str(ROOT / "BaseLine_Model"))
sys.path.insert(0, str(ROOT))

from cropnet_protocol import (  # noqa: E402
    ALLOWED_STATES,
    START_MONTH,
    END_MONTH,
    DAYS_PER_MONTH,
    PROTOCOL_MAX_STEPS,
    TIME_WINDOW,
)
import prepare_jsonl  # noqa: E402
import prepare_grid  # noqa: E402
from common import data as baseline_data  # noqa: E402
from data import load_grid_cache  # noqa: E402


def test_shared_protocol_values():
    assert ALLOWED_STATES == {
        "minnesota", "wisconsin", "michigan", "illinois",
        "indiana", "ohio", "missouri", "kentucky",
    }
    assert (START_MONTH, END_MONTH, DAYS_PER_MONTH) == (4, 9, 28)
    assert PROTOCOL_MAX_STEPS == 168
    assert baseline_data.N_STEPS == 168
    assert prepare_jsonl.ALLOWED_STATES is ALLOWED_STATES
    assert prepare_jsonl.PROTOCOL_MAX_STEPS == PROTOCOL_MAX_STEPS
    assert prepare_grid.PROTOCOL_MAX_STEPS == PROTOCOL_MAX_STEPS
    assert baseline_data.STATES is ALLOWED_STATES


def test_protocol_has_exactly_eight_allowed_states():
    assert len(ALLOWED_STATES) == 8
    assert all(len(name) > 2 for name in ALLOWED_STATES)


def test_county_series_preserves_month_and_day():
    dates = pd.to_datetime(["2020-04-01", "2020-09-28"])
    frame = pd.DataFrame({"date": dates, "feature": [1.0, 2.0]})

    result = prepare_jsonl._county_daily_series(frame, ["feature"])

    assert result["month"] == [4, 9]
    assert result["day"] == [1, 28]
    assert len(result["month"]) == len(result["day"]) == 2


def test_weather_filter_keeps_only_protocol_window():
    frame = pd.DataFrame(
        {
            "Month": [3, 4, 4, 9, 9, 10],
            "Day": [31, 1, 28, 28, 29, 1],
        }
    )

    filtered = prepare_jsonl.filter_weather_rows(frame)

    assert filtered[["Month", "Day"]].values.tolist() == [[4, 1], [4, 28], [9, 28]]


def test_calendar_validation_requires_aligned_fields_and_168_step_limit():
    prepare_jsonl.validate_calendar_fields([4, 9], [1, 28], 2)

    with pytest.raises(ValueError, match="at most 168"):
        prepare_jsonl.validate_calendar_fields([4] * 169, [1] * 169, 169)

    with pytest.raises(ValueError, match="same length"):
        prepare_jsonl.validate_calendar_fields([4], [1, 2], 1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 3, "version 4"),
        ("version", None, "version 4"),
        ("time_window", "03-01--11-30", "time window mismatch"),
        ("time_window", None, "time window mismatch"),
        ("days_per_month", 30, "days_per_month mismatch"),
        ("days_per_month", None, "days_per_month mismatch"),
        ("max_steps", 167, "max_steps mismatch"),
        ("max_steps", None, "max_steps mismatch"),
    ],
)
def test_grid_cache_rejects_wrong_protocol_metadata(tmp_path, field, value, message):
    cache_path = tmp_path / "grid_cache.pt"
    payload = {
        "version": 4,
        "coord_type": "grid_center",
        "time_window": TIME_WINDOW,
        "days_per_month": 28,
        "max_steps": 168,
        "entries": [],
    }
    payload[field] = value
    torch.save(payload, cache_path)

    with pytest.raises(ValueError, match=message):
        load_grid_cache(str(cache_path))


def test_grid_cache_requires_protocol_metadata(tmp_path):
    cache_path = tmp_path / "grid_cache.pt"
    torch.save({"version": 3, "coord_type": "grid_center"}, cache_path)

    with pytest.raises(ValueError, match="version 4"):
        load_grid_cache(str(cache_path))

    torch.save(
        {
            "version": 4,
            "coord_type": "grid_center",
            "time_window": TIME_WINDOW,
            "days_per_month": 28,
            "max_steps": 168,
            "entries": [],
        },
        cache_path,
    )
    assert load_grid_cache(str(cache_path))["version"] == 4


def test_grid_input_rejects_state_outside_allowlist():
    rows = [{"State": "iowa", "Year": 2020, "FIPS": "19001", "l_enc": 1}]

    with pytest.raises(ValueError, match="allowed states"):
        prepare_grid.validate_jsonl_states(rows)
