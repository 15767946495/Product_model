"""GNN-RNN（gomes-lab/GNN-RNN 的 SAGE_RNN 结构，纯 PyTorch 实现，不依赖 DGL）。

数据流（规范 10.4）：
    每年县级天气 (N,168,11) -> 单年 1D CNN；县级土壤 (N,7) -> 线性编码
    -> 县年环境表示 -> 同年县图 GraphSAGE（n_layers 层）-> 县年空间表示
    五个年度空间表示 -> LSTM -> 回归头 -> 末年单产（标准化尺度）
上游每层 SAGEConv 后都做 ReLU + dropout（含最后一层），此处保持一致；上游 forward 中的
y_pad 未进入网络，本实现完全不接收标签输入。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import in_degree


class WeatherSoilEncoder(nn.Module):
    """对应上游 CNN：天气 1D CNN -> 80 维，土壤 -> 40 维，拼接为县年环境表示。

    上游日尺度 CNN 针对 365 天、stride 2；CropNet 每年只有 168 天，因此使用 stride 1
    的卷积与 AvgPool，并以自适应池化收尾，通道宽度（64/128/256/512）与输出维度保持上游设置。
    统一土壤只有 7 个县级标量（无分层深度），因此用线性层替代上游按深度的 1D 卷积。
    """

    def __init__(self, num_weather_vars: int = 11, num_soil_vars: int = 7):
        super().__init__()
        self.wm_conv = nn.Sequential(
            nn.Conv1d(num_weather_vars, 64, kernel_size=9, stride=1), nn.ReLU(), nn.AvgPool1d(2, 2),
            nn.Conv1d(64, 128, 3, 1), nn.ReLU(), nn.AvgPool1d(2, 2),
            nn.Conv1d(128, 256, 3, 1), nn.ReLU(), nn.AvgPool1d(2, 2),
            nn.Conv1d(256, 512, 3, 1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.wm_fc = nn.Sequential(nn.Linear(512, 80), nn.ReLU())
        self.s_fc = nn.Sequential(nn.Linear(num_soil_vars, 40), nn.ReLU())
        self.out_dim = 80 + 40

    def forward(self, weather: torch.Tensor, soil: torch.Tensor) -> torch.Tensor:
        """weather (N,T,F)、soil (N,S) -> (N,120)。"""
        w = self.wm_fc(self.wm_conv(weather.transpose(1, 2)).squeeze(-1))
        return torch.cat([w, self.s_fc(soil)], dim=1)


class SAGEConv(nn.Module):
    """与 dgl.nn.SAGEConv 相同的 mean / pool 聚合：rst = fc_self(h) + fc_neigh(agg(neighbors))。"""

    def __init__(self, in_dim: int, out_dim: int, aggregator: str):
        super().__init__()
        if aggregator not in ("mean", "pool"):
            raise ValueError(f"aggregator must be mean or pool, got {aggregator}")
        self.aggregator = aggregator
        if aggregator == "pool":
            self.fc_pool = nn.Linear(in_dim, in_dim)
        self.fc_self = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        n = h.shape[0]
        if self.aggregator == "mean":
            summed = h.new_zeros(h.shape).index_add_(0, dst, h[src])
            neigh = summed / in_degree(edge_index, n).clamp_min(1).unsqueeze(-1).to(h.dtype)
        else:
            pooled = F.relu(self.fc_pool(h))
            index = dst.unsqueeze(-1).expand(-1, pooled.shape[1])
            neigh = pooled.new_zeros(pooled.shape).scatter_reduce(0, index, pooled[src], "amax",
                                                                  include_self=False)
        return self.fc_self(h) + self.fc_neigh(neigh) + self.bias


class GNNRNN(nn.Module):
    def __init__(self, *, aggregator: str = "mean", n_layers: int = 4, z_dim: int = 64, dropout: float = 0.0,
                 num_weather_vars: int = 11, num_soil_vars: int = 7):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.encoder = WeatherSoilEncoder(num_weather_vars, num_soil_vars)
        dims = [self.encoder.out_dim] + [z_dim] * n_layers
        self.layers = nn.ModuleList(SAGEConv(dims[i], dims[i + 1], aggregator) for i in range(n_layers))
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=z_dim, hidden_size=z_dim, num_layers=1, batch_first=True)
        self.regressor = nn.Sequential(nn.Linear(z_dim, z_dim // 2), nn.ReLU(), nn.Linear(z_dim // 2, 1))

    def forward(self, weather: torch.Tensor, soil: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """weather (N,Y,T,F) 已标准化、soil (N,S)、edge_index (2,E) -> 每年预测 (N,Y)（标准化尺度）。

        图在全部 N 个县上运行（固定县集合很小），不做邻居采样。
        """
        n, years = weather.shape[:2]
        hs = []
        for y in range(years):
            h = self.encoder(weather[:, y], soil)
            for layer in self.layers:
                h = self.dropout(F.relu(layer(h, edge_index)))
            hs.append(h)
        out, _ = self.lstm(torch.stack(hs, dim=1))  # (N,Y,z)
        return self.regressor(out).squeeze(-1)
