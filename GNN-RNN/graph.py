"""固定县 induced subgraph：边只来自 Census 接壤关系（neighbor_fips）+ 自环，六年不变。"""

from __future__ import annotations

import torch


def census_edge_set(fips: list[str], neighbor_fips: dict[str, list[str]]) -> set[tuple[int, int]]:
    """由 neighbor_fips 构造的期望边集合（含自环、双向），只保留固定县集合内的县。"""
    position = {f: i for i, f in enumerate(fips)}
    edges = {(i, i) for i in range(len(fips))}
    for f, neighbors in neighbor_fips.items():
        for n in neighbors:
            if n in position:
                edges.add((position[f], position[n]))
                edges.add((position[n], position[f]))
    return edges


def validate_graph(fips: list[str], edge_index: torch.Tensor, neighbor_fips: dict[str, list[str]]) -> None:
    """边集合必须恰好等于 Census 接壤 + 自环；不允许任何相似度或产量边。"""
    actual = {tuple(e) for e in edge_index.t().tolist()}
    expected = census_edge_set(fips, neighbor_fips)
    if actual != expected:
        extra, missing = sorted(actual - expected)[:5], sorted(expected - actual)[:5]
        raise ValueError(f"graph edges differ from Census adjacency: extra {extra}, missing {missing}")


def check_same_graph(train: dict, test: dict) -> None:
    """训练窗口与测试窗口：县集合、节点顺序、边必须完全一致。"""
    if train["fips"] != test["fips"]:
        raise ValueError("train/test GNN windows have different county order")
    if not torch.equal(train["edge_index"], test["edge_index"]):
        raise ValueError("train/test GNN windows have different graphs")


def in_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device).index_add_(
        0, edge_index[1], torch.ones(edge_index.shape[1], device=edge_index.device))
