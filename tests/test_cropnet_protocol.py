from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train_dataset"))
sys.path.insert(0, str(ROOT / "TFT_model"))
sys.path.insert(0, str(ROOT / "BaseLine_Model"))

from prepare_jsonl import (  # noqa: E402
    ALLOWED_STATES,
    START_MONTH,
    END_MONTH,
    DAYS_PER_MONTH,
)
import prepare_jsonl  # noqa: E402
from common import data as baseline_data  # noqa: E402
from data import PROTOCOL_MAX_STEPS, load_grid_cache  # noqa: E402


def test_shared_protocol_values():
    assert ALLOWED_STATES == {
        "minnesota", "wisconsin", "michigan", "illinois",
        "indiana", "ohio", "missouri", "kentucky",
    }
    assert (START_MONTH, END_MONTH, DAYS_PER_MONTH) == (4, 9, 28)
    assert PROTOCOL_MAX_STEPS == 168
    assert baseline_data.N_STEPS == 168


def test_protocol_has_exactly_eight_allowed_states():
    assert len(ALLOWED_STATES) == 8
    assert all(len(name) > 2 for name in ALLOWED_STATES)


def test_county_series_preserves_month_and_day():
    import pandas as pd

    dates = pd.to_datetime(["2020-04-01", "2020-09-28"])
    frame = pd.DataFrame({"date": dates, "feature": [1.0, 2.0]})

    result = prepare_jsonl._county_daily_series(frame, ["feature"])

    assert result["month"] == [4, 9]
    assert result["day"] == [1, 28]


def test_grid_cache_requires_protocol_metadata(tmp_path):
    cache_path = tmp_path / "grid_cache.pt"
    torch.save({"version": 3, "coord_type": "grid_center"}, cache_path)

    with pytest.raises(ValueError, match="version 4"):
        load_grid_cache(str(cache_path))

    torch.save(
        {
            "version": 4,
            "coord_type": "grid_center",
            "time_window": "04-01--09-28",
            "days_per_month": 28,
            "max_steps": 168,
            "entries": [],
        },
        cache_path,
    )
    assert load_grid_cache(str(cache_path))["version"] == 4
