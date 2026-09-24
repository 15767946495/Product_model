"""CropNet adapter for the official MMST-ViT data contract.

The adapter keeps the official tensor semantics:
  short-term weather: (6, G, 28, 9)
  long-term weather: (N_history, 12, 9)
  Sentinel-2 images: (6, G, 3, 224, 224)
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data/raid0/hqx/Product_model_runtime/DataSrc")
LABEL_JSONL = DATA_ROOT / "label" / "dataset.jsonl"
WEATHER_ROOT = DATA_ROOT / "weather"
AG_ROOT = DATA_ROOT / "sentinel" / "source" / "AG"

SHORT_MONTHS = tuple(range(4, 10))
LONG_TERM_YEARS = tuple(range(2017, 2022))
TARGET_REMOTE_DATES = tuple(f"{month:02d}-01" for month in SHORT_MONTHS)
FIVE_STATES = frozenset({
    "illinois", "iowa", "louisiana", "mississippi", "new_york",
})
WEATHER_COLUMNS = (
    "Avg Temperature (K)",
    "Max Temperature (K)",
    "Min Temperature (K)",
    "Precipitation (kg m**-2)",
    "Relative Humidity (%)",
    "Wind Gust (m s**-1)",
    "Wind Speed (m s**-1)",
    "Downward Shortwave Radiation Flux (W m**-2)",
    "Vapor Pressure Deficit (kPa)",
)


def load_metadata(path: Path = LABEL_JSONL) -> List[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def target_year(meta: dict) -> int:
    return int(meta["Year"] if "Year" in meta else meta["year"])


def state_name(meta: dict) -> str:
    return str(meta["State"] if "State" in meta else meta["state"])


def _state_abbr(state: str) -> str:
    mapping = {
        "alabama": "AL", "arizona": "AZ", "arkansas": "AR",
        "california": "CA", "colorado": "CO", "delaware": "DE",
        "georgia": "GA", "idaho": "ID", "illinois": "IL",
        "indiana": "IN", "iowa": "IA", "kansas": "KS",
        "kentucky": "KY", "louisiana": "LA", "maryland": "MD",
        "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
        "missouri": "MO", "nebraska": "NE", "new_jersey": "NJ",
        "new_mexico": "NM", "new_york": "NY", "north_carolina": "NC",
        "north_dakota": "ND", "ohio": "OH", "oklahoma": "OK",
        "oregon": "OR", "pennsylvania": "PA", "south_carolina": "SC",
        "south_dakota": "SD", "tennessee": "TN", "texas": "TX",
        "utah": "UT", "virginia": "VA", "washington": "WA",
        "west_virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    }
    try:
        return mapping[state.strip().lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported state for AG data: {state!r}") from error


def ag_paths(meta: dict) -> Tuple[Path, Path]:
    year = target_year(meta)
    state = _state_abbr(state_name(meta))
    state_prefix = str(meta["FIPS"]).zfill(5)[:2]
    return (
        AG_ROOT / str(year) / state
        / f"Agriculture_{state_prefix}_{state}_{year}-04-01_{year}-06-30.h5",
        AG_ROOT / str(year) / state
        / f"Agriculture_{state_prefix}_{state}_{year}-07-01_{year}-09-30.h5",
    )


def weather_path(year: int, meta: dict, month: int) -> Path:
    state = _state_abbr(state_name(meta))
    prefix = str(meta["FIPS"]).zfill(5)[:2]
    return WEATHER_ROOT / str(year) / state / f"HRRR_{prefix}_{state}_{year}-{month:02d}.csv"


def historical_years(target_year: int) -> List[int]:
    # Official MMST-ViT uses a fixed five-year long-term context.
    return list(LONG_TERM_YEARS)


def has_complete_inputs(meta: dict, require_long_term: bool = True) -> bool:
    year = target_year(meta)
    if any(not weather_path(year, meta, month).is_file() for month in SHORT_MONTHS):
        return False
    if require_long_term:
        for history_year in historical_years(year):
            if any(not weather_path(history_year, meta, month).is_file() for month in range(1, 13)):
                return False
    fips = str(meta["FIPS"]).zfill(5)
    for path in ag_paths(meta):
        if not path.is_file():
            return False
        try:
            with h5py.File(path, "r") as handle:
                if fips not in handle:
                    return False
        except OSError:
            return False
    return True


def build_manifest(
    years: Iterable[int],
    output_path: Path,
    source_path: Path = LABEL_JSONL,
    require_long_term: bool = True,
    all_states: bool = False,
) -> List[dict]:
    wanted = {int(year) for year in years}
    records = []
    for meta in load_metadata(source_path):
        year = int(meta["Year"])
        if year not in wanted:
            continue
        if not all_states and state_name(meta).strip().lower() not in FIVE_STATES:
            continue
        if not has_complete_inputs(meta, require_long_term=require_long_term):
            continue
        short_paths = [str(weather_path(year, meta, month)) for month in SHORT_MONTHS]
        long_paths = [
            [str(weather_path(history_year, meta, month)) for month in range(1, 13)]
            for history_year in historical_years(year)
        ]
        ag_first, ag_second = ag_paths(meta)
        records.append({
            "FIPS": str(meta["FIPS"]).zfill(5),
            "year": year,
            "county": meta.get("County", ""),
            "state": meta["State"],
            "state_ansi": str(meta["FIPS"]).zfill(5)[:2],
            "county_ansi": str(meta["FIPS"]).zfill(5)[2:],
            "yield_per_acre": float(meta["yield_per_acre"]),
            "data": {
                "HRRR": {"short_term": short_paths, "long_term": long_paths},
                "sentinel": [str(ag_first), str(ag_second)],
            },
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def _read_weather_csv(paths: Sequence[Path], fips: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()
        frame["FIPS Code"] = frame["FIPS Code"].astype(str).str.zfill(5)
        frames.append(frame[frame["FIPS Code"] == fips])
    if not frames:
        raise ValueError(f"No weather files for FIPS {fips}")
    return pd.concat(frames, ignore_index=True)


def load_short_term(meta: dict) -> torch.Tensor:
    fips = str(meta["FIPS"]).zfill(5)
    year = target_year(meta)
    months = []
    for month in SHORT_MONTHS:
        frame = _read_weather_csv([weather_path(year, meta, month)], fips)
        frame = frame[frame["Daily/Monthly"] == "Daily"]
        grid_values = []
        for _, grid in frame.groupby("Grid Index"):
            grid = grid.sort_values("Day")
            grid = grid[grid["Day"].between(1, 28)]
            grid_values.append(torch.tensor(grid.loc[:, WEATHER_COLUMNS].to_numpy(), dtype=torch.float32))
        if not grid_values or any(value.shape[0] != 28 for value in grid_values):
            raise ValueError(f"Invalid short-term weather shape for {fips}, {year}, month {month}")
        months.append(torch.stack(grid_values))
    result = torch.stack(months)
    return result


def load_long_term(meta: dict) -> torch.Tensor:
    fips = str(meta["FIPS"]).zfill(5)
    year = target_year(meta)
    year_values = []
    for history_year in historical_years(year):
        months = []
        for month in range(1, 13):
            frame = _read_weather_csv([weather_path(history_year, meta, month)], fips)
            frame = frame[frame["Daily/Monthly"] == "Monthly"]
            if frame.empty:
                raise ValueError(f"Missing monthly weather for {fips}, {history_year}-{month:02d}")
            values = frame.loc[:, WEATHER_COLUMNS].to_numpy(dtype=np.float32)
            months.append(torch.tensor(values, dtype=torch.float32).mean(dim=0))
        year_values.append(torch.stack(months))
    if not year_values:
        raise ValueError(f"No historical years before target year {year}")
    return torch.stack(year_values)


class CurrentMMSTDataset(Dataset):
    """Load one official-style MMST-ViT county-year sample at a time."""

    def __init__(self, records: Sequence[dict], train: bool = False):
        self.records = list(records)
        self.train = bool(train)
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        meta = self.records[index]
        fips = str(meta["FIPS"]).zfill(5)
        images = []
        coords = None
        for path in meta["data"]["sentinel"]:
            with h5py.File(path, "r") as handle:
                if fips not in handle:
                    raise ValueError(f"FIPS {fips} is missing from Sentinel file {path}")
                group = handle[fips]
                for name in group.keys():
                    if str(name)[-5:] not in TARGET_REMOTE_DATES:
                        continue
                    data = torch.from_numpy(group[name]["data"][...]).permute(0, 3, 1, 2).float() / 255.0
                    images.append((str(name)[-5:], torch.stack([self.transform(image) for image in data])))
                    if coords is None:
                        coords = torch.from_numpy(group[name]["coordinates"][...]).float().mean(dim=1)
        images.sort(key=lambda item: TARGET_REMOTE_DATES.index(item[0]))
        image_tensor = torch.stack([value for _, value in images])
        short = load_short_term(meta)
        long = load_long_term(meta)
        label = torch.tensor([np.log(max(float(meta["yield_per_acre"]), 1e-6))], dtype=torch.float32)
        return image_tensor, short, long, label, meta, coords


class CurrentMMSTPretrainDataset(Dataset):
    """Official pretraining view: Sentinel-2 plus short-term weather only."""

    def __init__(self, records: Sequence[dict], train: bool = True):
        self.records = list(records)
        self.train = bool(train)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        meta = self.records[index]
        fips = str(meta["FIPS"]).zfill(5)
        images = []
        for path in meta["data"]["sentinel"]:
            with h5py.File(path, "r") as handle:
                group = handle[fips]
                for name in group.keys():
                    if str(name)[-5:] not in TARGET_REMOTE_DATES:
                        continue
                    # Keep raw HWC uint8 images; official sentinel_wrapper
                    # applies the SimCLR transforms and normalization.
                    data = torch.from_numpy(group[name]["data"][...])
                    images.append((str(name)[-5:], data))
        images.sort(key=lambda item: TARGET_REMOTE_DATES.index(item[0]))
        return torch.stack([value for _, value in images]), load_short_term(meta)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--years", required=True, nargs="+", type=int)
    parser.add_argument("--no-long-term", action="store_true")
    parser.add_argument("--all-states", action="store_true")
    args = parser.parse_args()
    records = build_manifest(
        args.years,
        args.manifest,
        require_long_term=not args.no_long_term,
        all_states=args.all_states,
    )
    print(f"wrote {len(records)} records to {args.manifest}")
