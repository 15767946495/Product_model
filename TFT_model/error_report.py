"""Metrics for county-level and state-level prediction error reports."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def prediction_records(
    states: Iterable[str],
    years: Iterable[int],
    fips: Iterable[str],
    counties: Iterable[str],
    predictions: Iterable[float],
    labels: Iterable[float],
) -> list[dict]:
    """Build stable, JSON-serializable records for one prediction per county."""
    values = [
        list(states), list(years), list(fips), list(counties),
        list(predictions), list(labels),
    ]
    if len({len(items) for items in values}) != 1:
        raise ValueError("prediction record fields must have equal lengths")
    records = []
    for state, year, fip, county, prediction, label in zip(*values):
        prediction = float(prediction)
        label = float(label)
        error = prediction - label
        records.append({
            "state": str(state),
            "year": int(year),
            "fips": str(fip),
            "county": str(county),
            "prediction": prediction,
            "label": label,
            "error": error,
            "abs_error": abs(error),
        })
    return records


def _metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    residuals = predictions - labels
    n = int(labels.size)
    mse = float(np.mean(residuals ** 2))
    label_centered = labels - labels.mean()
    pred_centered = predictions - predictions.mean()
    ss_tot = float(np.sum(label_centered ** 2))
    pred_norm = float(np.sqrt(np.sum(pred_centered ** 2)))
    label_norm = float(np.sqrt(np.sum(label_centered ** 2)))
    corr = None if pred_norm == 0.0 or label_norm == 0.0 else float(
        np.sum(pred_centered * label_centered) / (pred_norm * label_norm)
    )
    return {
        "n": n,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residuals))),
        "bias": float(np.mean(residuals)),
        "error_std": float(np.std(residuals)),
        "r2": None if ss_tot == 0.0 else float(1.0 - np.sum(residuals ** 2) / ss_tot),
        "corr": corr,
    }


def metrics_by_group(
    groups: Iterable[str], predictions: Iterable[float], labels: Iterable[float]
) -> dict[str, dict]:
    """Return prediction metrics for every group.

    The three inputs must describe the same samples. Groups are sorted by their
    string representation to make JSON output deterministic.
    """
    group_values = np.asarray(list(groups))
    prediction_values = np.asarray(list(predictions), dtype=np.float64)
    label_values = np.asarray(list(labels), dtype=np.float64)
    if not (len(group_values) == len(prediction_values) == len(label_values)):
        raise ValueError("groups, predictions, and labels must have equal lengths")
    if len(label_values) == 0:
        return {}

    grouped: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    for group, prediction, label in zip(group_values, prediction_values, label_values):
        grouped[str(group)].append((float(prediction), float(label)))

    report = {}
    for group in sorted(grouped):
        values = np.asarray(grouped[group], dtype=np.float64)
        report[group] = _metrics(values[:, 0], values[:, 1])
    return report
