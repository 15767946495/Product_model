"""Configuration helpers documenting the official preprocessing contract."""

from __future__ import annotations

import json
from pathlib import Path

from cropnet_protocol import ALLOWED_STATES

SHORT_TERM_MONTHS = tuple(range(4, 10))
LONG_TERM_YEARS = tuple(range(2017, 2022))
SENTINEL_QUARTERS = (("04-01", "06-30"), ("07-01", "09-30"))
STATE_NAMES = {
    "IL": "Illinois", "IN": "Indiana", "KY": "Kentucky", "MI": "Michigan",
    "MN": "Minnesota", "MO": "Missouri", "OH": "Ohio", "WI": "Wisconsin",
}
STATE_ABBR = {name.lower(): abbr for abbr, name in STATE_NAMES.items()}


def make_config(output_dir: Path, data_root: Path, source_dir: Path, train_file: Path, val_file: Path, test_file: Path, target_fips: Path) -> dict:
    return {
        "data_root": str(data_root),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "train_file": str(train_file),
        "val_file": str(val_file),
        "test_file": str(test_file),
        "target_fips": str(target_fips),
        "split": {"train_years": [2017, 2018, 2019, 2020], "validation_years": [2021], "test_years": [2022]},
        "weather_normalization": {"mode": "none", "source": "official dataset/hrrr_loader.py casts selected columns to float32; ScalarNorm is not used by fine-tuning"},
        "sentinel_normalization": {"mean": [0.466, 0.471, 0.380], "std": [0.195, 0.194, 0.192]},
        "yield_transform": "natural_log_then_exp_for_metrics",
        "official_revision": "615666c8d9fcd704acb662c13065703cbf2eab70",
    }


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def official_sample_record(
    fips: str,
    year: int,
    state: str,
    county: str,
    data_root: Path,
    sentinel_types: tuple[str, ...] = ("AG", "NDVI"),
) -> dict:
    """Create the JSON shape consumed by the official three Dataset classes.

    The official loader expects relative paths rooted at ``data_root`` and a
    single JSON array per split. This adapter keeps the model/source untouched.
    """
    fips = str(fips).zfill(5)
    state_ansi, county_ansi = fips[:2], fips[2:]
    state_name = str(state).strip().lower()
    if state_name not in ALLOWED_STATES:
        raise ValueError(f"unsupported state for MMST-ViT manifest: {state!r}")
    state_abbr = STATE_ABBR[state_name]
    weather_files = [
        f"cropnet_dataset/data/weather/{weather_year}/{state_abbr}/HRRR_{state_ansi}_{state_abbr}_{weather_year}-{month:02d}.csv"
        for weather_year in LONG_TERM_YEARS
        for month in range(1, 13)
    ]
    sentinel = []
    for image_type in sentinel_types:
        prefix = "Agriculture" if image_type == "AG" else "Vegetation"
        for start, end in SENTINEL_QUARTERS:
            sentinel.append(
                f"mmst_vit/download/Sentinel-2 Imagery/data/{image_type}/{year}/{state_abbr}/"
                f"{prefix}_{state_ansi}_{state_abbr}_{year}-{start}_{year}-{end}.h5"
            )
    return {
        "FIPS": fips,
        "year": int(year),
        "county": county,
        "state": state_name,
        "county_ansi": county_ansi,
        "state_ansi": state_ansi,
        "data": {
            "HRRR": {
                "short_term": [
                    f"cropnet_dataset/data/weather/{year}/{state_abbr}/HRRR_{state_ansi}_{state_abbr}_{year}-{month:02d}.csv"
                    for month in SHORT_TERM_MONTHS
                ],
                "long_term": [weather_files],
            },
            "USDA": f"cropnet_dataset/data/usda_corn/USDA_Corn_County_{year}.csv",
            "sentinel": sentinel,
        },
    }
