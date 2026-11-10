"""GNN-RNN 唯一数据适配边界：统一 CropNetDataset -> 每县一个连续年窗口。

默认协议（GNN_WINDOWS）：训练窗口 2017--2021 预测 2021；测试窗口 2018--2022 预测 2022。
统一训练协议（``--valid-year 2021``）下改为等长 4 年窗口：训练 2017--2020 预测 2020、
验证 2018--2021 预测 2021、测试 2019--2022 预测 2022；三窗口长度一致，2021 标签只用于早停，
2022 标签只作最终指标的 y_true。标准化统计量只来自训练窗口。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from graph import check_same_graph, validate_graph  # noqa: E402
from train_dataset.collate import collate_gnn_window  # noqa: E402
from train_dataset.dataset import DATA_ROOT, GNN_WINDOWS, YEARS, CropNetDataset, build_gnn_windows  # noqa: E402

TRAIN_YEARS = GNN_WINDOWS["train"]
TEST_YEARS = GNN_WINDOWS["test"]
TARGET_TRAIN_YEAR = TRAIN_YEARS[-1]
TARGET_TEST_YEAR = TEST_YEARS[-1]
EPS = 1e-6


def protocol_windows(valid_year: int, test_year: int, length: int = 4) -> dict:
    """2017--2020 训练 / 2021 验证 / 2022 测试的等长窗口。"""
    return {
        "train": tuple(range(int(valid_year) - length, int(valid_year))),
        "valid": tuple(range(int(valid_year) - length + 1, int(valid_year) + 1)),
        "test": tuple(range(int(test_year) - length + 1, int(test_year) + 1)),
    }


def load_windows(jsonl_path, data_root=DATA_ROOT, windows=None) -> dict:
    """返回 {split: batch}；各窗口县集合、节点顺序、边已校验一致且只来自 Census 接壤。"""
    windows = GNN_WINDOWS if windows is None else windows
    years = sorted({y for ys in windows.values() for y in ys})
    dataset = CropNetDataset(Path(jsonl_path), tuple(years), load_images=False, data_root=Path(data_root))
    items = [dataset[i] for i in range(len(dataset))]
    built = build_gnn_windows(items, windows)
    batches = {split: collate_gnn_window(built[split]) for split in windows}
    reference = next(iter(batches.values()))
    for split, batch in batches.items():
        if split != next(iter(batches)):
            check_same_graph(reference, batch)
    neighbors = {w[0]["fips"]: w[0]["neighbor_fips"] for w in built["train"]}
    for batch in batches.values():
        validate_graph(batch["fips"], batch["edge_index"], neighbors)
    for split, split_years in windows.items():
        if batches[split]["years"].tolist() != list(split_years):
            raise ValueError(f"{split} window years {batches[split]['years'].tolist()} != {list(split_years)}")
    # 输出顺序与统一 Dataset 中目标年份样本顺序一致（按 fips 排序）
    for split, batch in batches.items():
        batch["target_sample_ids"] = [ids[-1] for ids in batch["sample_id"]]
    return batches


def fit_stats(train: dict, mask=None) -> dict:
    """只在训练窗口上计算标准化统计量；``mask`` 进一步限定到训练县（空间交叉验证）。"""
    weather = train["weather"]  # (N,5,168,11)
    soil = train["soil"]        # (N,7)
    targets = train["targets"]  # (N,) 2021
    if mask is not None:
        weather, soil, targets = weather[mask], soil[mask], targets[mask]
    return {
        "weather_mean": weather.mean(dim=(0, 1, 2)), "weather_std": weather.std(dim=(0, 1, 2)).clamp_min(EPS),
        "soil_mean": soil.mean(dim=0), "soil_std": soil.std(dim=0).clamp_min(EPS),
        "y_mean": targets.mean(), "y_std": targets.std().clamp_min(EPS),
    }


def model_inputs(batch: dict, stats: dict) -> dict:
    """标准化后的模型输入；不含任何标签。"""
    return {
        "weather": (batch["weather"] - stats["weather_mean"]) / stats["weather_std"],
        "soil": (batch["soil"] - stats["soil_mean"]) / stats["soil_std"],
        "edge_index": batch["edge_index"],
    }


def stats_to_json(stats: dict) -> dict:
    return {k: v.tolist() for k, v in stats.items()}


def stats_from_json(payload: dict) -> dict:
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in payload.items()}
