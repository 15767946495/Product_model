"""Build fixed-year MMST-ViT sample manifests from local CropNet data."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

TRAIN_YEARS = tuple(range(2017, 2021))
VAL_YEAR = 2021
TEST_YEAR = 2022
SPLIT_YEARS = {"train": TRAIN_YEARS, "val": (VAL_YEAR,), "test": (TEST_YEAR,)}

STATE_ABBR = {
    "ALABAMA": "AL", "ARKANSAS": "AR", "CALIFORNIA": "CA", "COLORADO": "CO",
    "DELAWARE": "DE", "GEORGIA": "GA", "IOWA": "IA", "IDAHO": "ID", "ILLINOIS": "IL",
    "INDIANA": "IN", "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", "MARYLAND": "MD",
    "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSOURI": "MO", "MISSISSIPPI": "MS", "MONTANA": "MT",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "NEBRASKA": "NE", "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM", "NEW YORK": "NY", "OHIO": "OH", "OKLAHOMA": "OK", "PENNSYLVANIA": "PA",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX",
    "VIRGINIA": "VA", "WASHINGTON": "WA", "WISCONSIN": "WI", "WEST VIRGINIA": "WV",
    "WYOMING": "WY",
}


def normalize_fips(value: object) -> str:
    """Return a five-digit FIPS string, rejecting missing/non-numeric values."""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"invalid FIPS: {value!r}")
    return text.zfill(5)


def _state_from_fips(fips: str) -> str:
    return fips[:2]


def _usda_rows(usda_dir: Path, years: Sequence[int]) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    for year in years:
        path = usda_dir / f"USDA_Corn_County_{year}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("commodity_desc", "").strip().upper() != "CORN":
                    continue
                if row.get("reference_period_desc", "").strip().upper() != "YEAR":
                    continue
                try:
                    state_ansi = normalize_fips(row.get("state_ansi", ""))[-2:]
                    county_ansi = normalize_fips(row.get("county_ansi", ""))[-3:]
                    fips = state_ansi + county_ansi
                    yield_value = float(str(row.get("YIELD, MEASURED IN BU / ACRE", "")).replace(",", ""))
                except (TypeError, ValueError):
                    continue
                rows[(fips, year)] = {
                    "FIPS": fips,
                    "Year": year,
                    "County": row.get("county_name", "").strip(),
                    "yield_per_acre": yield_value,
                    "State": STATE_ABBR.get(row.get("state_name", "").strip().upper(), row.get("state_name", "").strip().upper()),
                    "usda_file": path.name,
                }
    return rows


def _weather_keys(
    weather_dir: Path,
    years: Sequence[int],
    candidate_fips: set[str] | None = None,
) -> set[tuple[str, int]]:
    """Scan CSV headers and rows to find FIPS/year pairs with weather data."""
    keys: set[tuple[str, int]] = set()
    for year in years:
        year_dir = weather_dir / str(year)
        if not year_dir.exists():
            continue
        for path in sorted(year_dir.glob("*/*.csv")):
            try:
                columns = ["FIPS Code", "Daily/Monthly"]
                for frame in pd.read_csv(path, usecols=tuple(columns), chunksize=250_000):
                    frame = frame[frame["Daily/Monthly"].astype(str).str.strip().str.lower() == "daily"]
                    if candidate_fips is not None:
                        fips_series = frame["FIPS Code"].astype(str).str.split(".").str[0].str.zfill(5)
                        frame = frame[fips_series.isin(candidate_fips)].copy()
                        fips_series = frame["FIPS Code"].astype(str).str.split(".").str[0].str.zfill(5)
                    else:
                        fips_series = frame["FIPS Code"].astype(str).str.split(".").str[0].str.zfill(5)
                    keys.update((fips, year) for fips in fips_series.unique())
            except (OSError, UnicodeDecodeError):
                continue
    return keys


def build_valid_samples(usda_dir: Path, weather_dir: Path, years: Sequence[int]) -> list[dict]:
    """Return USDA/weather intersections, sorted by year then FIPS."""
    years = tuple(sorted({int(year) for year in years}))
    usda = _usda_rows(usda_dir, years)
    weather = _weather_keys(weather_dir, years, {fips for fips, _ in usda})
    samples = []
    for key in sorted(set(usda) & weather, key=lambda item: (item[1], item[0])):
        row = dict(usda[key])
        row.update({"StateFIPS": _state_from_fips(row["FIPS"]), "weather_year": key[1]})
        samples.append(row)
    return samples


def split_samples(samples: Sequence[dict]) -> dict[str, list[dict]]:
    """Partition samples into the fixed train/validation/test protocol."""
    splits = {name: [] for name in SPLIT_YEARS}
    for sample in samples:
        year = int(sample["Year"])
        destination = next((name for name, allowed in SPLIT_YEARS.items() if year in allowed), None)
        if destination is not None:
            splits[destination].append(dict(sample))
    return splits


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usda-dir", type=Path, required=True)
    parser.add_argument("--weather-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=sorted({*TRAIN_YEARS, VAL_YEAR, TEST_YEAR}))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    samples = build_valid_samples(args.usda_dir, args.weather_dir, args.years)
    splits = split_samples(samples)
    write_jsonl(args.output, samples)
    for name, rows in splits.items():
        write_jsonl(args.output.with_name(f"{args.output.stem}.{name}.jsonl"), rows)
    print(json.dumps({"samples": len(samples), **{name: len(rows) for name, rows in splits.items()}}, indent=2))


if __name__ == "__main__":
    main()
