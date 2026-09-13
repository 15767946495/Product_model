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

import cropnet_protocol as shared_protocol  # noqa: E402
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
import data as tft_data  # noqa: E402
from data import load_grid_cache  # noqa: E402


def test_jsonl_dry_run_reports_protocol_without_writing(tmp_path, capsys, monkeypatch):
    output = tmp_path / "dataset.jsonl"
    monkeypatch.setattr(prepare_jsonl, "OUTPUT_PATH", str(output))

    prepare_jsonl.main(["--dry-run"])

    captured = capsys.readouterr().out
    assert '"allowed_states": [' in captured
    assert '"start_month": 4' in captured
    assert '"end_month": 9' in captured
    assert '"days_per_month": 28' in captured
    assert '"max_steps": 168' in captured
    assert not output.exists()


def test_jsonl_process_all_accepts_output_path(tmp_path, monkeypatch):
    output = tmp_path / "dataset.jsonl"
    monkeypatch.setattr(prepare_jsonl, "DATA_DIR", str(tmp_path / "data"))

    with pytest.raises(SystemExit, match="1"):
        prepare_jsonl.process_all(output_path=str(output))

    assert not output.exists()


def test_soil_loader_skips_non_state_mapping(tmp_path, capsys):
    soil_path = tmp_path / "county_soil.csv"
    soil_path.write_text("FIPS,ph\n17001,6.5\n", encoding="utf-8")

    assert prepare_jsonl._load_soil_map(str(soil_path)) == {}
    assert "州级土壤映射字段不完整" in capsys.readouterr().out


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


def test_tft_protocol_constants_match_shared_protocol():
    assert tft_data.PROTOCOL_START_MONTH == shared_protocol.START_MONTH
    assert tft_data.PROTOCOL_END_MONTH == shared_protocol.END_MONTH
    assert tft_data.PROTOCOL_DAYS_PER_MONTH == shared_protocol.DAYS_PER_MONTH
    assert tft_data.PROTOCOL_MAX_STEPS == shared_protocol.PROTOCOL_MAX_STEPS
    assert tft_data.TIME_WINDOW == shared_protocol.TIME_WINDOW


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
    ("month", "day"),
    [(3, 1), (10, 1), (4, 0), (4, 29)],
)
def test_calendar_validation_rejects_invalid_month_and_day(month, day):
    with pytest.raises(ValueError, match="protocol window"):
        prepare_jsonl.validate_calendar_fields([month], [day], 1)


def test_calendar_validation_accepts_exact_168_steps_and_empty_input():
    month = [4] * 168
    day = [1] * 168

    prepare_jsonl.validate_calendar_fields(month, day, 168)
    prepare_jsonl.validate_calendar_fields([], [], 0)


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


def test_grid_input_accepts_multiple_allowed_states():
    rows = [
        {"State": "Minnesota"},
        {"State": "wisconsin"},
        {"State": "OHIO"},
    ]

    prepare_grid.validate_jsonl_states(rows)


def test_build_entry_aligns_calendar_fields_with_features_and_caps_at_168():
    dates = pd.date_range("2020-04-01", periods=168, freq="D")
    frame = pd.DataFrame(
        {
            "Grid Index": [1] * len(dates),
            "date": dates,
            "Lat (llcrnr)": [40.0] * len(dates),
            "Lat (urcrnr)": [41.0] * len(dates),
            "Lon (llcrnr)": [-90.0] * len(dates),
            "Lon (urcrnr)": [-89.0] * len(dates),
            **{feature: [float(index)] * len(dates)
               for index, feature in enumerate(prepare_grid.WRF_COLS)},
        }
    )

    entry = prepare_grid.build_entry(frame, prepare_grid.WRF_COLS)

    assert entry["feats"].shape == (1, 168, len(prepare_grid.WRF_COLS))
    assert entry["l_enc"] == entry["feats"].shape[1]
    assert len(entry["month"]) == len(entry["day"]) == entry["l_enc"]
    assert entry["l_enc"] <= PROTOCOL_MAX_STEPS


def test_build_entry_returns_none_for_empty_input():
    empty = pd.DataFrame(
        columns=[
            "Grid Index", "date", "Lat (llcrnr)", "Lat (urcrnr)",
            "Lon (llcrnr)", "Lon (urcrnr)", *prepare_grid.WRF_COLS,
        ]
    )

    assert prepare_grid.build_entry(empty, prepare_grid.WRF_COLS) is None
