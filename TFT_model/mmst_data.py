"""官方 MMST-ViT Corn 数据集的严格适配器。"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


SOIL_FEATURES = (
    "clay_pct",
    "sand_pct",
    "silt_pct",
    "om_pct",
    "ph",
    "bulk_density",
    "awc",
)

_MMST_ROOT = Path(__file__).resolve().parents[1] / "MMST-ViT"


def _load_official_module(name: str, relative_path: str):
    path = _MMST_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载官方 MMST-ViT 模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


_OFFICIAL_SENTINEL_MODULE = _load_official_module(
    "_tft_mmst_official_sentinel_loader",
    "dataset/sentinel_loader.py",
)
_OFFICIAL_HRRR_MODULE = _load_official_module(
    "_tft_mmst_official_hrrr_loader",
    "dataset/hrrr_loader.py",
)
_OFFICIAL_USDA_MODULE = _load_official_module(
    "_tft_mmst_official_usda_loader",
    "dataset/usda_loader.py",
)

Sentinel_Dataset = _OFFICIAL_SENTINEL_MODULE.Sentinel_Dataset
HRRR_Dataset = _OFFICIAL_HRRR_MODULE.HRRR_Dataset
USDA_Dataset = _OFFICIAL_USDA_MODULE.USDA_Dataset


class MMSTCornDataset(Dataset):
    """按相同 index 对齐官方影像、短期气象、Corn 标签和县级土壤。"""

    def __init__(self, root_dir, config_file, soil_file):
        self.root_dir = Path(root_dir)
        self.config_file = Path(config_file)
        self.soil_file = Path(soil_file)
        self._closed = False

        with self.config_file.open(encoding="utf-8") as handle:
            records = json.load(handle)
        self._validate_corn_records(records)
        self.fips_codes = [record["FIPS"] for record in records]
        self.years = [record["year"] for record in records]

        self.image_dataset = Sentinel_Dataset(root_dir, config_file)
        self.weather_dataset = HRRR_Dataset(root_dir, config_file)
        try:
            self.target_dataset = USDA_Dataset(root_dir, config_file, crop_type="Corn")
            self.validate_alignment()
            self.soil_by_fips = self._load_and_validate_soil()
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return len(self.fips_codes)

    @staticmethod
    def _validate_corn_records(records):
        if not isinstance(records, list):
            raise ValueError("MMST Corn config must contain a list of records")
        for index, record in enumerate(records):
            try:
                usda_path = Path(record["data"]["USDA"])
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"MMST Corn config record {index} has no data.USDA path"
                ) from error
            if "corn" not in {part.casefold() for part in usda_path.parts}:
                raise ValueError(
                    f"MMST Corn config record {index} must reference a Corn USDA directory; "
                    f"got {record['data']['USDA']!r}"
                )

    def close(self):
        if self._closed:
            return
        self._closed = True
        weather_dataset = getattr(self, "weather_dataset", None)
        executor = getattr(weather_dataset, "executor", None)
        if executor is not None:
            executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def validate_alignment(self):
        datasets = {
            "config": (self.fips_codes, self.years),
            "Sentinel": (self.image_dataset.fips_codes, self.image_dataset.years),
            "HRRR": (self.weather_dataset.fips_codes, self.weather_dataset.years),
            "USDA": (self.target_dataset.fips_codes, self.target_dataset.years),
        }
        expected_length = len(self.fips_codes)
        for name, (fips_codes, years) in datasets.items():
            if len(fips_codes) != expected_length or len(years) != expected_length:
                raise ValueError(
                    f"MMST dataset alignment length mismatch: {name} has "
                    f"{len(fips_codes)} FIPS and {len(years)} years; expected {expected_length}"
                )
        for index, expected in enumerate(zip(self.fips_codes, self.years)):
            for name, (fips_codes, years) in datasets.items():
                actual = (fips_codes[index], years[index])
                if actual != expected:
                    raise ValueError(
                        f"MMST dataset alignment mismatch at index {index}: "
                        f"config={expected}, {name}={actual}"
                    )

    def _load_and_validate_soil(self):
        with self.soil_file.open(encoding="utf-8") as handle:
            soil_by_fips = json.load(handle)

        validated = {}
        for fips in self.fips_codes:
            row = soil_by_fips.get(fips)
            if not isinstance(row, dict):
                raise ValueError(f"soil missing for FIPS {fips}")
            values = []
            for feature in SOIL_FEATURES:
                try:
                    value = float(row[feature])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"soil incomplete for FIPS {fips}: invalid {feature}"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"soil incomplete for FIPS {fips}: non-finite {feature}"
                    )
                values.append(value)
            validated[fips] = torch.tensor(values, dtype=torch.float32)
        return validated

    def __getitem__(self, index):
        images, image_fips, image_year = self.image_dataset[index]

        weather_fips = self.weather_dataset.fips_codes[index]
        weather_year = self.weather_dataset.years[index]
        short_weather = self.weather_dataset.get_short_term_val(
            weather_fips,
            self.weather_dataset.short_term_file_path[index],
        ).to(torch.float32)

        target, target_fips, target_year = self.target_dataset[index]
        expected = (self.fips_codes[index], self.years[index])
        actual = {
            "Sentinel": (image_fips, image_year),
            "HRRR": (weather_fips, weather_year),
            "USDA": (target_fips, target_year),
        }
        for name, key in actual.items():
            if key != expected:
                raise ValueError(
                    f"MMST dataset alignment mismatch at index {index}: "
                    f"config={expected}, {name}={key}"
                )
        if target.numel() != 2:
            raise ValueError(
                f"official USDA target at index {index} must contain two values; "
                f"got shape {tuple(target.shape)}"
            )

        fips, year = expected
        return images, short_weather, target[-1], self.soil_by_fips[fips], fips, year


class MMSTCountyLoader(DataLoader):
    """县级外层 loader；可变网格数要求 batch size 固定为一。"""

    def __init__(self, dataset, batch_size=1, **kwargs):
        if batch_size != 1:
            raise ValueError("MMSTCountyLoader requires batch_size=1 for variable grid counts")
        super().__init__(dataset, batch_size=1, **kwargs)


__all__ = [
    "HRRR_Dataset",
    "MMSTCornDataset",
    "MMSTCountyLoader",
    "SOIL_FEATURES",
    "Sentinel_Dataset",
    "USDA_Dataset",
]
