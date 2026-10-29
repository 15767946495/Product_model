"""UNet-ConvLSTM（CropNet 适配）。

数据流（规范 10.5）：
    每月每网格 Sentinel-2 (3,S,S) -> 共享 U-Net encoder（只用编码器，多尺度特征池化到最深层分辨率后拼接）
    同月每网格 28 天 x 11 维天气 -> 天气编码器 -> 广播到特征图
    视觉与天气 1x1 卷积融合 -> 六个月特征图序列 -> ConvLSTM（逐网格，跨月）
    -> 末月隐状态空间平均 -> 有效网格 masked pooling -> 县级当年单产（标准化尺度）
不使用长期气象、其他年份隐藏状态或历史产量。只对有效网格运行编码器，padding 网格不进入任何计算。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def masked_grid_pool(features: torch.Tensor, grid_mask: torch.Tensor) -> torch.Tensor:
    """features (B,G,D)、grid_mask (B,G) -> (B,D)；padding 网格不计入平均。"""
    weights = grid_mask.to(features.dtype).unsqueeze(-1)
    return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def _double_conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNetEncoder(nn.Module):
    """U-Net 收缩路径：4 级 double conv，每级之间 2x2 max pool；返回融合后的多尺度特征图。"""

    def __init__(self, in_channels: int = 3, base: int = 32, out_channels: int = 128):
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8]
        self.stages = nn.ModuleList()
        cin = in_channels
        for w in widths:
            self.stages.append(_double_conv(cin, w))
            cin = w
        self.fuse = nn.Sequential(nn.Conv2d(sum(widths), out_channels, 1), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for i, stage in enumerate(self.stages):
            if i:
                x = F.max_pool2d(x, 2)
            x = stage(x)
            feats.append(x)
        size = feats[-1].shape[-2:]
        multi = [F.adaptive_avg_pool2d(f, size) for f in feats[:-1]] + [feats[-1]]
        return self.fuse(torch.cat(multi, dim=1))


class WeatherEncoder(nn.Module):
    """(N,28,11) 已标准化逐日天气 -> (N,D)。"""

    def __init__(self, num_vars: int = 11, dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_vars, dim, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 5, padding=2), nn.ReLU(inplace=True),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return self.conv(w.transpose(1, 2)).mean(dim=-1)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden = hidden_channels
        self.gates = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size,
                               padding=kernel_size // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class UNetConvLSTM(nn.Module):
    def __init__(self, *, base_channels: int = 32, weather_dim: int = 64, hidden_channels: int = 64,
                 dropout: float = 0.1, num_weather_vars: int = 11, grid_chunk: int = 128):
        super().__init__()
        feat = base_channels * 4
        self.grid_chunk = grid_chunk
        self.encoder = UNetEncoder(3, base_channels, feat)
        self.weather = WeatherEncoder(num_weather_vars, weather_dim)
        self.fusion = nn.Sequential(nn.Conv2d(feat + weather_dim, feat, 1), nn.ReLU(inplace=True))
        self.convlstm = ConvLSTMCell(feat, hidden_channels)
        self.head = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Dropout(dropout),
                                  nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_channels, 1))

    def _encode(self, images: torch.Tensor, weather: torch.Tensor) -> torch.Tensor:
        """images (N,3,S,S)、weather (N,28,F) -> 融合特征图 (N,C,h,w)，按 grid_chunk 分块。"""
        out = []
        for start in range(0, images.shape[0], self.grid_chunk):
            x = self.encoder(images[start:start + self.grid_chunk])
            w = self.weather(weather[start:start + self.grid_chunk])
            w = w[:, :, None, None].expand(-1, -1, *x.shape[-2:])
            out.append(self.fusion(torch.cat([x, w], dim=1)))
        return torch.cat(out, dim=0)

    def forward(self, images: torch.Tensor, weather: torch.Tensor, grid_mask: torch.Tensor,
                image_mask: torch.Tensor | None = None) -> torch.Tensor:
        """images (B,6,G,3,S,S)、weather (B,6,G,28,F) 已标准化、grid_mask (B,G) -> (B,)。

        image_mask (B,6,G)：无图像的 (月,网格) 输入特征置零（统一数据保留县月度遥感完整，正常全 True）。
        """
        b, t, g = images.shape[:3]
        if not grid_mask.any(dim=1).all():
            raise ValueError("every sample needs at least one valid grid")
        valid = rearrange(grid_mask, 'b g -> (b g)')
        imgs = rearrange(images, 'b t g c h w -> (b g) t c h w')[valid]
        wx = rearrange(weather, 'b t g n f -> (b g) t n f')[valid]
        n = imgs.shape[0]
        feats = self._encode(rearrange(imgs, 'n t c h w -> (n t) c h w'), rearrange(wx, 'n t d f -> (n t) d f'))
        feats = rearrange(feats, '(n t) c h w -> n t c h w', n=n, t=t)
        if image_mask is not None:
            month_valid = rearrange(image_mask, 'b t g -> (b g) t')[valid].to(feats.dtype)
            feats = feats * month_valid[:, :, None, None, None]
        h = feats.new_zeros((n, self.convlstm.hidden, *feats.shape[-2:]))
        c = torch.zeros_like(h)
        for month in range(t):
            h, c = self.convlstm(feats[:, month], (h, c))
        grid_vec = h.mean(dim=(-2, -1))  # (n, hidden)
        per_grid = grid_vec.new_zeros((b * g, grid_vec.shape[1]))
        per_grid[valid] = grid_vec
        pooled = masked_grid_pool(rearrange(per_grid, '(b g) d -> b g d', b=b, g=g), grid_mask)
        return self.head(pooled).squeeze(-1)
