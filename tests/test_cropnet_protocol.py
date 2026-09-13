from pathlib import Path
import sys
import hashlib
import json

import numpy as np
import pytest
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train_dataset"))
sys.path.insert(0, str(ROOT / "TFT_model"))
sys.path.insert(0, str(ROOT / "BaseLine_Model"))
sys.path.insert(0, str(ROOT / "ablation"))
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
import infer as tft_infer  # noqa: E402
import train as tft_train  # noqa: E402
import ablation as ablation_launcher  # noqa: E402
from CNNRNN.cnn_rnn import CNNRNN  # noqa: E402
from ConvLSTM.convlstm import ConvLSTM  # noqa: E402
from GNNRNN.gnn_rnn import GNNRNN  # noqa: E402
from deepcropnet import deepcropnet  # noqa: E402


def test_jsonl_dry_run_reports_protocol_without_writing(tmp_path, capsys, monkeypatch):
    output = tmp_path / "dataset.jsonl"
    monkeypatch.setattr(prepare_jsonl, "OUTPUT_PATH", str(output))

    prepare_jsonl.main(["--dry-run"])

    protocol = json.loads(capsys.readouterr().out)
    assert set(protocol["allowed_states"]) == ALLOWED_STATES
    assert protocol["start_month"] == 4
    assert protocol["end_month"] == 9
    assert protocol["days_per_month"] == 28
    assert protocol["max_steps"] == 168
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


@pytest.mark.parametrize("length", [0, 167, 169])
def test_jsonl_calendar_requires_exactly_168_steps(length):
    with pytest.raises(ValueError, match="exactly 168|at most 168"):
        prepare_jsonl.validate_sample_calendar(
            {"month": [4] * length, "day": [1] * length, "l_enc": length}
        )


@pytest.mark.parametrize("entries,n_ok,mismatch", [([None], 0, 0), ([{}], 0, 0), ([{}], 1, 1)])
def test_grid_validation_rejects_incomplete_cache_before_saving(entries, n_ok, mismatch):
    with pytest.raises(ValueError, match="incomplete|aligned"):
        prepare_grid.validate_entries([{"l_enc": 168}], entries, n_ok, mismatch)


def _valid_grid_entry():
    return {
        "feats": torch.zeros(1, 168, len(prepare_grid.WRF_COLS)),
        "month": torch.tensor([4] * 168),
        "day": torch.tensor([1] * 168),
        "l_enc": 168,
    }


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda entry: entry.pop("month"), "month"),
        (lambda entry: entry.update(month=torch.zeros(167, dtype=torch.long)), "month (shape|length)"),
        (lambda entry: entry.update(day=torch.zeros(167, dtype=torch.long)), "day (shape|length)"),
        (lambda entry: entry.update(month=torch.tensor([3] + [4] * 167)), "month values"),
        (lambda entry: entry.update(day=torch.tensor([0] + [1] * 167)), "day values"),
    ],
)
def test_grid_validation_rejects_corrupt_calendar_fields(mutate, message):
    entry = _valid_grid_entry()
    mutate(entry)

    with pytest.raises(ValueError, match=message):
        prepare_grid.validate_entries([{"l_enc": 168}], [entry], 1, 0)


@pytest.mark.parametrize("entry", ["not-an-entry", 3, {"feats": torch.zeros(1, 168, 1)}])
def test_grid_validation_rejects_bad_entry_type_or_fields(entry):
    with pytest.raises(ValueError, match="entry 0|entry type|month"):
        prepare_grid.validate_entries([{"l_enc": 168}], [entry], 1, 0)


def test_grid_audit_reports_corrupt_entry_calendar(tmp_path):
    jsonl = tmp_path / "dataset.jsonl"
    cache = tmp_path / "cache.pt"
    row = {"State": "minnesota", "Year": 2020, "FIPS": "27001", "month": [4] * 168, "day": [1] * 168, "l_enc": 168}
    jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
    entry = _valid_grid_entry()
    entry["month"][0] = 3
    torch.save({"version": 4, "time_window": TIME_WINDOW, "max_steps": 168, "entries": [entry]}, cache)

    result = prepare_grid.audit_artifacts(str(jsonl), str(cache))

    assert result["error_count"] > 0
    assert result["assertions"]["entry_calendar_valid"] is False
    assert any("entry 0" in error for error in result["errors"])


def test_grid_build_entry_rejects_non_168_steps():
    dates = pd.date_range("2020-04-01", periods=167, freq="D")
    frame = pd.DataFrame(
        {
            "Grid Index": [1] * len(dates), "date": dates,
            "Lat (llcrnr)": [40.0] * len(dates), "Lat (urcrnr)": [41.0] * len(dates),
            "Lon (llcrnr)": [-90.0] * len(dates), "Lon (urcrnr)": [-89.0] * len(dates),
            **{feature: [1.0] * len(dates) for feature in prepare_grid.WRF_COLS},
        }
    )
    with pytest.raises(ValueError, match="exactly 168"):
        prepare_grid.build_entry(frame, prepare_grid.WRF_COLS)


def test_runtime_artifacts_regression_if_present():
    root = Path(__import__("os").environ.get(
        "CROPNET_RUNTIME_TRAIN_DATASET", "/data/raid0/hqx/Product_model_runtime/train_dataset"
    ))
    required = [root / "dataset.jsonl", root / "grid_cache.pt", root / "grid_cache_meta.json"]
    if not all(path.exists() for path in required):
        pytest.skip("运行时产物不存在")
    result = prepare_grid.audit_artifacts(str(required[0]), str(required[1]))
    assert result["error_count"] == 0
    assert result["errors"] == []
    assert result["n_none"] == 0
    assert result["alignment"]["entries_equal_rows"] is True
    assert result["sha256"]["jsonl"] == hashlib.sha256(required[0].read_bytes()).hexdigest()
    assert result["sha256"]["cache"] == hashlib.sha256(required[1].read_bytes()).hexdigest()


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


def test_grid_input_rejects_non_168_length():
    row = {"State": "minnesota", "month": [4] * 167, "day": [1] * 167, "l_enc": 167}

    with pytest.raises(ValueError, match="exactly 168"):
        prepare_grid.validate_jsonl_rows([row])


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


def test_tft_entry_defaults_use_exact_protocol_states():
    assert set(tft_train.ALLOWED_STATES) == ALLOWED_STATES
    assert set(tft_infer.ALLOWED_STATES) == ALLOWED_STATES
    assert ablation_launcher.STATES is ALLOWED_STATES
    assert set(tft_train.DEFAULT_DYNAMIC_FEATURE_NAMES) == set(tft_infer.DEFAULT_DYNAMIC_FEATURE_NAMES)


def test_constructed_features_use_april_one_as_doy_origin():
    assert tft_data.hargreaves_pet.__defaults__ == (91,)


def test_tft_final_index_is_last_valid_sequence_step():
    assert tft_train.last_valid_index(torch.tensor([1, 3])).tolist() == [0, 2]
    assert tft_infer.last_valid_index(torch.tensor([1, 3])).tolist() == [0, 2]


def test_infer_cutoff_uses_calendar_fields_from_april_sequence():
    month = torch.tensor([4] * 28 + [5] * 28)
    day = torch.tensor(list(range(1, 29)) * 2)
    assert tft_infer.cutoff_index(month, day, 56, (4, 15)) == 14
    assert tft_infer.cutoff_index(month, day, 56, (5, 28)) == 55


@pytest.mark.parametrize("cutoffs", ["03-31", "10-01", "09-29"])
def test_infer_rejects_cutoff_outside_protocol_window(cutoffs):
    with pytest.raises(ValueError, match=r"protocol window.*04-01.*09-28"):
        tft_infer.parse_cutoffs(cutoffs)


def test_infer_accepts_protocol_end_cutoff():
    assert tft_infer.parse_cutoffs("09-28") == [(9, 28)]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda entry: entry.update(feats=torch.zeros(1, 167, len(prepare_grid.WRF_COLS))), "168"),
        (lambda entry: entry.update(l_enc=167), "l_enc"),
        (lambda entry: entry.update(month=torch.zeros(168, 1, dtype=torch.long)), "month shape"),
        (lambda entry: entry.update(day=torch.zeros(167, dtype=torch.long)), "day (shape|length)"),
        (lambda entry: entry.update(month=torch.tensor([3] + [4] * 167)), "month values"),
        (lambda entry: entry.update(day=torch.tensor([29] + [1] * 167)), "day values"),
    ],
)
def test_tft_loader_rejects_corrupt_grid_entry(tmp_path, mutate, message):
    cache_path = tmp_path / "grid_cache.pt"
    entry = {
        "feats": torch.zeros(1, 168, len(prepare_grid.WRF_COLS)),
        "coords": torch.zeros(1, 2),
        "month": torch.tensor([4] * 168),
        "day": torch.tensor([1] * 168),
        "l_enc": 168,
    }
    mutate(entry)
    torch.save(
        {
            "version": 4,
            "coord_type": "grid_center",
            "time_window": TIME_WINDOW,
            "days_per_month": 28,
            "max_steps": 168,
            "entries": [entry],
        },
        cache_path,
    )

    with pytest.raises(ValueError, match=message):
        tft_data.load_grid_cache(str(cache_path))


def test_ablation_filters_aligned_pairs_without_shifting_cache_entries():
    pairs = [({"State": "iowa", "id": 1}, {"id": "iowa"}),
             ({"State": "ohio", "id": 2}, {"id": "ohio"})]
    assert ablation_launcher.filter_allowed_pairs(pairs) == [pairs[1]]


def test_shared_baseline_entry_filters_parsed_states_and_pads_to_protocol_length():
    entries = [
        {
            "feats": torch.ones(1, 160, baseline_data.N_FEATS),
            "coords": torch.zeros(1, 2),
        },
        {
            "feats": torch.ones(1, 168, baseline_data.N_FEATS),
            "coords": torch.zeros(1, 2),
        },
    ]
    metadata = [
        {"State": "ohio", "FIPS": "39001", "Year": 2020, "yield_per_acre": 1},
        {"State": "iowa", "FIPS": "19001", "Year": 2020, "yield_per_acre": 1},
    ]
    soil = {"39001": {feature: 1 for feature in baseline_data.SOIL_FEATURES}}

    samples = baseline_data.build_dataset(entries, metadata, soil)

    assert [sample["state"] for sample in samples] == ["ohio"]
    assert samples[0]["weather"].shape == (baseline_data.N_STEPS, baseline_data.N_FEATS)
    assert samples[0]["grid_weather"].shape == (
        1, baseline_data.N_STEPS, baseline_data.N_FEATS
    )


def test_baseline_models_consume_the_rebuilt_168_step_shapes():
    weather = torch.zeros(2, PROTOCOL_MAX_STEPS, baseline_data.N_FEATS)
    soil = torch.zeros(2, baseline_data.SOIL_DIM)
    assert CNNRNN()(weather, soil).shape == (2, 1)
    assert GNNRNN()(weather, torch.eye(2), soil).shape == (2, 1)

    grid_weather = torch.zeros(2, 3, PROTOCOL_MAX_STEPS, baseline_data.N_FEATS)
    grid_mask = torch.ones(2, 3)
    assert ConvLSTM()(grid_weather, grid_mask, soil).shape == (2, 1)


def test_deepcropnet_weekly_features_start_at_protocol_day_zero():
    daily = np.arange(168, dtype=np.float64)

    result = deepcropnet.weekly_accumulate(daily, deepcropnet.START_DAY, deepcropnet.N_WEEKS)

    assert deepcropnet.START_DAY == 0
    assert result.shape == (20,)
    assert result[0] == sum(range(7))
    assert result[-1] == sum(range(133, 140))
    assert set(deepcropnet.REGIONS) == ALLOWED_STATES
