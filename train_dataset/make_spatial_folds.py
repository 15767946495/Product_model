"""按县分组的五折空间交叉验证划分：划分清单 + 每折的训练/测试 JSONL。

空间外推的评价口径：把 741 个县分成 5 份，每折留出 1 份县作测试；训练用其余
4 份县的全部年份，测试用留出县的全部年份。留出县在训练中完全不出现（既不是
训练标签，也不进入标准化统计量），只有它自己的历史单产作为县均值偏移/县固定
效应基线在预测时可用，用来回答"模型能否把年际信号外推到没见过的县"。

分州分层：同一州内的县先在州内随机打乱，再按位置轮流分到 5 折，使每折都含有
29 个州的县（小州如特拉华、俄克拉何马只有 3 个县，不这样做可能整州落在一折）。

用法：
  python -I train_dataset/make_spatial_folds.py                      # 写清单与 5 折 JSONL
  python -I train_dataset/make_spatial_folds.py --dry-run            # 只打印每折县数
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/data/raid0/hqx/train_dataset/dataset.jsonl")
DEFAULT_OUT = Path("/data/raid0/hqx/train_dataset/spatial_folds")

FIPS = re.compile(r'"fips":"(\d+)"')
STATE = re.compile(r'"state_abbr":"([A-Z]{2})"')


def load_counties(path: Path) -> dict[str, str]:
    """fips -> 州缩写；只读每行开头，不解析整行 JSON。"""
    counties: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fips = FIPS.search(line[:400]).group(1)
            counties.setdefault(fips, STATE.search(line[:600]).group(1))
    return counties


def assign_folds(counties: dict[str, str], k: int, seed: int) -> dict[str, int]:
    """州内随机排序后按位置轮流分配，返回 fips -> fold。"""
    rng = np.random.default_rng(seed)
    by_state: dict[str, list[str]] = {}
    for fips, state in counties.items():
        by_state.setdefault(state, []).append(fips)
    assignment: dict[str, int] = {}
    for state in sorted(by_state):
        members = sorted(by_state[state])
        order = rng.permutation(len(members))
        for position, index in enumerate(order):
            assignment[members[int(index)]] = position % k
    return assignment


def write_split(source: Path, destination: Path, keep: set[str]) -> int:
    """按 fips 过滤原始 JSONL，并把 neighbor_fips 裁剪到该子集内的县。

    统一 Dataset 要求邻接只在固定县集合内（build_gnn_windows 与 validate_graph 依赖
    这一点），因此子集文件的邻接必须改成诱导子图。
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source.open("r", encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for line in reader:
            if FIPS.search(line[:400]).group(1) not in keep:
                continue
            record = json.loads(line)
            record["neighbor_fips"] = [f for f in record["neighbor_fips"] if f in keep]
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def main(args) -> None:
    source = Path(args.dataset_jsonl).resolve()
    out = Path(args.output).resolve()
    counties = load_counties(source)
    assignment = assign_folds(counties, args.k, args.seed)
    folds = [sorted(f for f, fold in assignment.items() if fold == index) for index in range(args.k)]

    for index, members in enumerate(folds):
        states = len({counties[f] for f in members})
        print(f"fold {index}: {len(members):3d} counties, {states} states")
    if args.dry_run:
        return

    manifest = {
        "k": args.k,
        "seed": args.seed,
        "dataset_jsonl": str(source),
        "n_counties": len(counties),
        "assignments": dict(sorted(assignment.items())),
        "folds": [{"fold": index, "counties": members} for index, members in enumerate(folds)],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "folds.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    all_counties = set(assignment)
    for index, members in enumerate(folds):
        test = set(members)
        train = all_counties - test
        train_rows = write_split(source, out / f"fold{index}" / "train.jsonl", train)
        test_rows = write_split(source, out / f"fold{index}" / "test.jsonl", test)
        print(f"fold {index}: train {train_rows} county-years, test {test_rows} county-years")
    print(f"manifest: {out / 'folds.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("county-grouped 5-fold spatial CV split")
    parser.add_argument("--dataset-jsonl", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
