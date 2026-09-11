"""Validate MMST-ViT's weather, USDA, and Sentinel sample alignment."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence


def sample_key(row: Mapping) -> tuple[str, int]:
    return str(row["FIPS"]).zfill(5), int(row["Year"])


def validate_alignment(samples: Sequence[Mapping], sentinel_index: Mapping, weather_index: Mapping, usda_index: Mapping) -> dict:
    report = {"total": len(samples), "valid": 0, "missing": [], "by_year": defaultdict(int)}
    for sample in samples:
        key = sample_key(sample)
        missing = [name for name, index in (("sentinel", sentinel_index), ("weather", weather_index), ("usda", usda_index)) if key not in index]
        if missing:
            report["missing"].append({"FIPS": key[0], "Year": key[1], "missing": missing})
        else:
            report["valid"] += 1
            report["by_year"][str(key[1])] += 1
    report["by_year"] = dict(sorted(report["by_year"].items()))
    return report
