import os
os.environ.setdefault("HF_HOME", "/data/raid0/hqx/.cache/huggingface")

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math
from typing import Any, Dict, List, Tuple, Optional


AG_DATE_PAIRS = [
    (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1),
]


def _relative_day_index(month_ids: torch.Tensor, day_ids: torch.Tensor) -> torch.Tensor:
    """将日历日期转换为相对 4 月 1 日的索引，4 月 1 日为 1。"""
    month_ids = month_ids.to(dtype=torch.long)
    day_ids = day_ids.to(device=month_ids.device, dtype=torch.long)
    month_starts = torch.tensor(
        [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
        device=month_ids.device,
        dtype=torch.long,
    )
    safe_month = month_ids.clamp(1, 12)
    result = month_starts[safe_month - 1] + day_ids - month_starts[3]
    return torch.where((month_ids >= 1) & (day_ids >= 1), result, torch.zeros_like(result))


def _broadcast_rs_forward(
    rs_encoded: torch.Tensor,
    month_ids: torch.Tensor,
    day_ids: torch.Tensor,
    ag_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward-fill the latest available RS observation onto the weather dates."""
    B, G, N_rs, H = rs_encoded.shape
    if ag_mask.shape != (B, N_rs):
        raise ValueError(f"ag_mask shape {ag_mask.shape} != {(B, N_rs)}")
    if N_rs != len(AG_DATE_PAIRS):
        raise ValueError(f"expected {len(AG_DATE_PAIRS)} RS dates, got {N_rs}")
    T = month_ids.shape[1]
    rs_days = torch.tensor(
        [month * 32 + day for month, day in AG_DATE_PAIRS],
        device=rs_encoded.device,
        dtype=torch.long,
    )
    weather_days = month_ids.to(rs_encoded.device).long() * 32 + day_ids.to(
        rs_encoded.device
    ).long()
    eligible = (
        ag_mask.to(device=rs_encoded.device, dtype=torch.bool)[:, None, :]
        & (rs_days[None, None, :] <= weather_days[:, :, None])
    )
    rs_indices = torch.arange(N_rs, device=rs_encoded.device).view(1, 1, N_rs)
    latest = torch.where(eligible, rs_indices, torch.zeros_like(rs_indices))
    latest = latest.max(dim=-1).values
    has_value = eligible.any(dim=-1)
    gathered = torch.gather(
        rs_encoded,
        dim=2,
        index=latest[:, None, :, None].expand(B, G, T, H),
    )
    rs_by_time = gathered * has_value[:, None, :, None].to(rs_encoded.dtype)
    rs_mask = has_value[:, :, None].expand(B, T, G)
    return rs_by_time, rs_mask


def _build_interleaved_layout(
    month_ids: torch.Tensor,
    day_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    grid_mask: torch.Tensor,
    ag_mask: Optional[torch.Tensor],
    ag_dates: List[Tuple[int, int]],
) -> Dict[str, torch.Tensor]:
    """构造按日期分组的天气/遥感 token 布局，并在 batch 维度补齐。"""
    B, T = month_ids.shape
    G = grid_mask.shape[1]
    device = month_ids.device
    relative_days = _relative_day_index(month_ids, day_ids)
    seq_lens = seq_lens.to(device=device, dtype=torch.long).clamp(0, T)
    grid_mask = grid_mask.to(device=device, dtype=torch.bool)
    if ag_mask is None:
        ag_mask = torch.ones(B, len(ag_dates), device=device, dtype=torch.bool)
    else:
        ag_mask = ag_mask.to(device=device, dtype=torch.bool)

    rows = []
    max_tokens = 0
    for b in range(B):
        row = []
        valid_t = int(seq_lens[b].item())
        valid_grids = torch.nonzero(grid_mask[b], as_tuple=False).flatten().tolist()
        available_rs = {
            date: i for i, date in enumerate(ag_dates)
            if i < ag_mask.shape[1] and bool(ag_mask[b, i])
        }
        for t in range(valid_t):
            date = (int(month_ids[b, t].item()), int(day_ids[b, t].item()))
            rs_i = available_rs.get(date)
            for g in valid_grids:
                row.append((t, g, -1))
                if rs_i is not None:
                    row.append((t, g, rs_i))
        rows.append(row)
        max_tokens = max(max_tokens, len(row))

    weather_positions = torch.full((B, T, G), -1, device=device, dtype=torch.long)
    pad_mask = torch.zeros(B, max_tokens, device=device, dtype=torch.bool)
    block_boundaries = torch.zeros(B, max_tokens, device=device, dtype=torch.bool)
    token_time_indices = torch.zeros(B, max_tokens, device=device, dtype=torch.long)
    weather_time_indices = torch.full((B, max_tokens), -1, device=device, dtype=torch.long)
    rs_grid_indices = torch.full((B, max_tokens), -1, device=device, dtype=torch.long)
    rs_date_indices = torch.full((B, max_tokens), -1, device=device, dtype=torch.long)

    for b, row in enumerate(rows):
        previous_date = None
        for pos, (t, grid, rs_i) in enumerate(row):
            date = (int(month_ids[b, t].item()), int(day_ids[b, t].item()))
            pad_mask[b, pos] = True
            block_boundaries[b, pos] = date != previous_date
            previous_date = date
            token_time_indices[b, pos] = relative_days[b, t]
            rs_grid_indices[b, pos] = grid
            rs_date_indices[b, pos] = rs_i
            if rs_i < 0:
                weather_positions[b, t, grid] = pos
                weather_time_indices[b, pos] = t

    return {
        "pad_mask": pad_mask,
        "block_boundaries": block_boundaries,
        "weather_positions": weather_positions,
        "token_time_indices": token_time_indices,
        "weather_time_indices": weather_time_indices,
        "rs_grid_indices": rs_grid_indices,
        "rs_date_indices": rs_date_indices,
        "relative_days": relative_days,
    }


class GatedLinearUnit(nn.Module):
    """门控线性单元"""
    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x


class GateAddNorm(nn.Module):
    """主路 x 先经 GLU，再与 skip 残差相加后 LayerNorm（非单纯 x+skip）。"""
    def __init__(
        self,
        input_size: int,
        skip_size: int = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.skip_size = skip_size or input_size

        if self.input_size != self.skip_size:
            self.resample = nn.Linear(self.skip_size, self.input_size)
        self.glu = GatedLinearUnit(
            input_size=self.input_size,
            hidden_size=self.input_size,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        gated = self.glu(x)
        if self.input_size != self.skip_size:
            skip = self.resample(skip)
        return self.norm(gated + skip)


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        residual: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        # 主路径
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()
        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        self.glu = GatedLinearUnit(
            input_size=self.output_size,
            hidden_size=self.output_size,
            dropout=dropout
        )
        # 层归一化前置，稳定梯度
        self.norm = nn.LayerNorm(self.output_size)
        # 维度适配
        if self.input_size != self.output_size:
            self.skip_proj = nn.Linear(input_size, output_size)
        else:
            self.skip_proj = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                nn.init.zeros_(p)
            elif "fc" in n or "context" in n:
                nn.init.xavier_uniform_(p, gain=1.0)

    def forward(self, x, context=None):
        # 残差分支
        skip = self.skip_proj(x)
        # 主路径
        x = self.fc1(x)
        if context is not None:
            x = x + self.context(context)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.glu(x)
        x = self.norm(x + skip)
        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.dropout = dropout
        self.context_size = context_size
        self.num_inputs = len(input_sizes)
        self.var_names = list(input_sizes.keys())

        self.total_input_size = sum(input_sizes.values())
        self.flattened_grn = GatedResidualNetwork(
            self.total_input_size, self.hidden_size,
            self.num_inputs, self.dropout, self.context_size
        )

        self.softmax = nn.Softmax(dim=-1)

        self.single_var_grns = nn.ModuleDict()
        for name, size in input_sizes.items():
            self.single_var_grns[name] = GatedResidualNetwork(
                size, self.hidden_size, self.hidden_size, self.dropout
            )

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        seq_lens: torch.Tensor,
        context: torch.Tensor = None,
    ):
        var_outputs = []
        weight_inputs = []
        for name in self.var_names:
            tensor = x[name]
            encoded = self.single_var_grns[name](tensor)
            var_outputs.append(encoded)
            weight_inputs.append(tensor)
        var_outputs = torch.stack(var_outputs, dim=-1)
        flat_embedding = torch.cat(weight_inputs, dim=-1)
        sparse_weights_logits = self.flattened_grn(flat_embedding, context)
        sparse_weights = self.softmax(sparse_weights_logits)
        max_seq_len = sparse_weights.size(1)
        positions = torch.arange(0, max_seq_len, device=sparse_weights.device).unsqueeze(0)
        time_step_mask = (positions < seq_lens.unsqueeze(1)).float()
        sparse_weights = sparse_weights * time_step_mask.unsqueeze(-1)
        sparse_weights = sparse_weights.unsqueeze(-2)
        outputs = (var_outputs * sparse_weights).sum(dim=-1)
        return outputs, sparse_weights


class CausalScaledDotProductAttention(nn.Module):
    """多头因果缩放点积注意力：Q/K 多头；共享 V 经 W_v 映射到 head_dim，与 (B,T,T) 平均权重相乘得 head_dim 上下文，W_o 再映回 hidden_size。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 2,
        dropout: float = 0.1,
        mask_bias: float = -1e6,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) 须能被 num_heads ({self.num_heads}) 整除"
            )
        self.head_dim = self.hidden_size // self.num_heads
        self.mask_bias = mask_bias
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_v = nn.Linear(self.hidden_size, self.head_dim)
        self.W_o = nn.Linear(self.head_dim, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self._init_weights()

    def _init_weights(self):
        for m in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        """(t, t) bool，True 表示该 (i,j) 应被屏蔽（不可 attend）。"""
        j_idx = torch.arange(t, device=device).unsqueeze(0)
        i_idx = torch.arange(t, device=device).unsqueeze(1)
        return j_idx > i_idx

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, hidden) -> (B, num_heads, T, head_dim)
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        block_boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        v = self.W_v(x)
        q = self._split_heads(self.W_q(x))
        k = self._split_heads(self.W_k(x))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            float(self.head_dim)
        )

        if block_boundaries is not None:
            # block-causal mask: 同组内全可见，前面的组可见
            # block_boundaries[i]=1 表示位置 i 是新 group 的开始
            group_id = torch.cumsum(block_boundaries, dim=1)  # (B, T)
            same_group = group_id.unsqueeze(-1) == group_id.unsqueeze(-2)  # (B, T, T)
            earlier_group = group_id.unsqueeze(-1) >= group_id.unsqueeze(-2)
            allowed = same_group | earlier_group
            attn_scores = attn_scores.masked_fill(
                ~allowed.view(B, 1, T, T), self.mask_bias
            )
        else:
            causal = self._causal_mask(T, attn_scores.device)
            attn_scores = attn_scores.masked_fill(
                causal.view(1, 1, T, T), self.mask_bias
            )

        if pad_mask is not None:
            valid = pad_mask.to(device=attn_scores.device, dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(
                ~valid.view(B, 1, T, 1), self.mask_bias
            )
            attn_scores = attn_scores.masked_fill(
                ~valid.view(B, 1, 1, T), self.mask_bias
            )
            attn_weights = self.softmax(attn_scores)
            attn_weights = attn_weights * valid.view(B, 1, T, 1).float()
        elif mask is not None:
            positions = torch.arange(0, T, device=mask.device, dtype=torch.int32).unsqueeze(0)
            seq_mask = positions < mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(
                ~seq_mask.view(B, 1, 1, T), self.mask_bias
            )
            attn_weights = self.softmax(attn_scores)
            attn_weights = attn_weights * seq_mask.view(B, 1, T, 1).float()
        else:
            attn_weights = self.softmax(attn_scores)

        attn_weights = self.dropout(attn_weights)
        # (B,H,T,T) 在头维上平均 -> (B,T,T)，再与共享 v 相乘
        attn_avg = attn_weights.mean(dim=1)
        ctx = torch.matmul(attn_avg, v)  # (B, T, head_dim)
        out = self.W_o(ctx)
        return out, attn_avg


class SoilStaticEncoder(nn.Module):
    """县级连续土壤特征(7 维,不分桶)→ Linear 映射 → 四路 GRN 上下文(c_s,c_e,c_c,c_h)。

    土壤为静态特征,仅作为上下文注入时序 VSN / LSTM / 注意力,
    不参与网格注意力计算。
    """

    def __init__(
        self,
        soil_dim: int,
        hidden_size: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(soil_dim, hidden_size)
        self.grn_cs = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        self.grn_ce = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        self.grn_cc = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )
        self.grn_ch = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )

    def forward(
        self, soil_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # soil_feats: (B, soil_dim) 已 z-score 标准化
        h = F.elu(self.linear(soil_feats))               # (B, H)
        c_s = self.grn_cs(h)
        c_e = self.grn_ce(h)
        c_c = self.grn_cc(h)
        c_h = self.grn_ch(h)
        return c_s, c_e, c_c, c_h


class LSTMEncoder(nn.Module):
    """TFT的LSTM编码器（适配变长时序输入）"""
    def __init__(
        self,
        input_size: int,       # 动态时序特征维度（d_dynamic）
        hidden_size: int,      # LSTM隐藏层维度
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
        have_context: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.have_context = have_context
        self.num_directions = 2 if bidirectional else 1

        # LSTM层（支持变长序列）
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        # LSTM 输出 + 原始输入：GateAddNorm 内已对主路做 GLU 再残差
        self.gate_add_norm = GateAddNorm(
            input_size=hidden_size * self.num_directions,
            skip_size=input_size,
            dropout=dropout,
        )

        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        c_c: torch.Tensor = None,
        c_h: torch.Tensor = None,
    ):
        """
        Args:
            x: (batch_size, max_seq_len, input_size) 动态时序特征（补零后的变长序列）
            seq_lens: (batch_size,) 每个样本的实际时序长度（必传）
        Returns:
            lstm_feat: (batch_size, max_seq_len, hidden_size * num_directions) LSTM编码特征
            last_hidden: (batch_size, hidden_size * num_directions) 最后时间步特征（经 GateAddNorm）
            enc_h_last: (batch_size, hidden_size) GateAddNorm 之前、末层 LSTM 输出在「有效序列最后一步」的 hidden（供解码器初态）
            enc_c_last: (batch_size, hidden_size) 同上时刻的 cell；逐步 c 未展开时 pack 路径用 c_n[-1] 反序（与末个有效步一致）
        """
        B = x.shape[0]
        h0 = c_h.unsqueeze(0).repeat(self.num_layers * self.num_directions, 1, 1)
        c0 = c_c.unsqueeze(0).repeat(self.num_layers * self.num_directions, 1, 1)

        seq_lens_cpu = seq_lens.cpu().tolist()
        seq_lens_sorted, idx = torch.sort(
            torch.tensor(seq_lens_cpu, device=x.device), descending=True
        )
        idx = idx.long()
        x_sorted = x[idx]
        h0 = h0[:, idx, :]
        c0 = c0[:, idx, :]

        x_packed = nn.utils.rnn.pack_padded_sequence(
            x_sorted,
            seq_lens_sorted.cpu().tolist(),
            batch_first=True,
            enforce_sorted=True,
        )
        lstm_out_packed, (h_n, c_n) = self.lstm(x_packed, (h0, c0))
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out_packed, batch_first=True, total_length=x.size(1)
        )

        idx_rev = torch.argsort(idx)
        lstm_out_orig = lstm_out[idx_rev]
        dev_e = lstm_out_orig.device
        batch_idx_e = torch.arange(B, device=dev_e)
        last_idx_e = (seq_lens.to(device=dev_e) - 1).clamp(min=0).long()
        enc_h_last = lstm_out_orig[batch_idx_e, last_idx_e, :]
        enc_c_last = c_n[-1][idx_rev]

        x_orig = x_sorted[idx_rev]
        lstm_feat = self.gate_add_norm(lstm_out_orig, x_orig)
        Bsz, Tlen, _ = lstm_feat.shape
        t_ar = torch.arange(
            Tlen, device=lstm_feat.device, dtype=torch.long
        ).unsqueeze(0).expand(Bsz, Tlen)
        sl_orig = seq_lens.to(device=lstm_feat.device).unsqueeze(1)
        ok_t = t_ar < sl_orig
        lstm_feat = lstm_feat * ok_t.unsqueeze(-1).to(dtype=lstm_feat.dtype)

        batch_idx = torch.arange(B, device=lstm_feat.device)
        last_idx = (seq_lens.to(device=lstm_feat.device) - 1).clamp(min=0).long()
        last_hidden = lstm_feat[batch_idx, last_idx, :]

        return lstm_feat, last_hidden, enc_h_last, enc_c_last


class SpatialAttentionAggregator(nn.Module):
    """WeatherFormer pooling with a county CLS query and positional Q/K/V inputs."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        if hidden_size % 4 != 0:
            raise ValueError("WeatherFormer position encoding requires hidden_size divisible by 4")
        self.cls_token = nn.Parameter(torch.zeros(self.hidden_size))
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.scale = math.sqrt(float(self.hidden_size))
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        for m in (self.W_q, self.W_k, self.W_v):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _st_pe(self, coords: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """WeatherFormer four-slot encoding, shape (B,G,T,H)."""
        d = self.hidden_size
        nf = d // 4
        i = torch.arange(nf, device=coords.device, dtype=coords.dtype)
        freq = (10000.0 ** (-4.0 * i / d)).view(1, 1, 1, nf)
        t_idx = t_idx.to(device=coords.device, dtype=coords.dtype)
        if t_idx.ndim == 1:
            t = t_idx.view(1, 1, -1, 1)
        else:
            t = t_idx[:, None, :, None]
        lat = (coords[..., 0:1] * (math.pi / 180.0))[:, :, None, :]
        lon = (coords[..., 1:2] * (math.pi / 180.0))[:, :, None, :]
        pe = torch.zeros(
            coords.shape[0], coords.shape[1], t_idx.shape[-1], d,
            device=coords.device, dtype=coords.dtype,
        )
        pe[..., 0::4] = torch.sin(t * freq)
        pe[..., 1::4] = torch.cos(t * freq)
        pe[..., 2::4] = torch.sin(lat * freq)
        pe[..., 3::4] = torch.cos(lon * freq)
        return pe

    @staticmethod
    def _cls_coords(coords: torch.Tensor, grid_mask: torch.Tensor) -> torch.Tensor:
        """Return the masked mean grid center for each county, shape (B,2)."""
        valid = grid_mask.to(dtype=coords.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (coords * valid).sum(dim=1) / denom

    def forward(
        self,
        tokens: torch.Tensor,
        coords: torch.Tensor,
        grid_mask: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = self.forward_weights(tokens, coords, grid_mask)
        return out

    def forward_weights(
        self,
        tokens: torch.Tensor,
        coords: torch.Tensor,
        grid_mask: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        time_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool flattened per-time weather/RS tokens with a county CLS query."""
        # tokens: (B, T, K, H), coords: (B, K, 2)
        B, T, K, H = tokens.shape
        if coords.shape != (B, K, 2):
            raise ValueError(f"coords shape {coords.shape} != {(B, K, 2)}")
        if token_mask is None:
            key_valid = grid_mask.unsqueeze(1).expand(B, T, -1)
            if key_valid.shape[-1] != K:
                raise ValueError(f"grid_mask must have {K} entries when token_mask is omitted")
            cls_coords_input = coords
            cls_grid_mask = grid_mask
        else:
            key_valid = token_mask.to(device=tokens.device, dtype=torch.bool)
            if key_valid.shape != (B, T, K):
                raise ValueError(f"token_mask shape {key_valid.shape} != {(B, T, K)}")
            if grid_mask.shape[1] == K:
                cls_coords_input = coords
                cls_grid_mask = grid_mask
            else:
                cls_coords_input = coords[:, :grid_mask.shape[1]]
                cls_grid_mask = grid_mask
        t_idx = torch.arange(T, device=tokens.device, dtype=torch.long)
        cls_coords = self._cls_coords(cls_coords_input, cls_grid_mask)
        if time_indices is None:
            time_indices = t_idx.unsqueeze(0).expand(B, -1)
        pe_grid = self._st_pe(coords, time_indices).transpose(1, 2)  # (B,T,K,H)
        pe_cls = self._st_pe(cls_coords.unsqueeze(1), time_indices).squeeze(1)  # (B,T,H)
        cls = self.cls_token.view(1, 1, H).expand(B, T, H)
        q = self.W_q(cls + pe_cls).unsqueeze(2)          # (B,T,1,H)
        x = tokens                                             # (B,T,K,H)
        x_with_pe = x + pe_grid
        k = self.W_k(x_with_pe)
        v = self.W_v(x_with_pe)
        scores = torch.matmul(q, k.transpose(-2, -1)).squeeze(2) / self.scale
        scores = scores.masked_fill(~key_valid, float("-inf"))
        w = torch.softmax(scores, dim=-1)                         # (B,T,G)
        w = self.dropout(w)
        w = w * key_valid.to(w.dtype)
        cls_out = torch.matmul(w.unsqueeze(2), v).squeeze(2)
        return self.norm(cls_out), w


class WeatherRemoteCrossAttention(nn.Module):
    """Daily weather-grid queries over all remote-sensing grids of that month."""

    def __init__(self, hidden_size: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.head_dim = self.hidden_size // self.num_heads
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.fusion_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            context_size=hidden_size,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.distance_scale = nn.Parameter(torch.tensor(1.0))
        self.scale = math.sqrt(float(self.head_dim))
        for layer in (self.W_q, self.W_k, self.W_v):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        weather: torch.Tensor,
        remote: torch.Tensor,
        coords: torch.Tensor,
        grid_mask: torch.Tensor,
        month_ids: torch.Tensor,
        ag_mask: torch.Tensor,
        soil_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, G, T, H = weather.shape
        if remote.shape[:3] != (B, G, len(AG_DATE_PAIRS)):
            raise ValueError(
                f"remote shape {remote.shape} must start with {(B, G, len(AG_DATE_PAIRS))}"
            )
        month_to_index = torch.full((13,), -1, device=weather.device, dtype=torch.long)
        for index, (month, _) in enumerate(AG_DATE_PAIRS):
            month_to_index[month] = index
        rs_index = month_to_index[month_ids.to(weather.device).long().clamp(0, 12)]
        valid_month = rs_index >= 0
        safe_index = rs_index.clamp_min(0)
        remote_by_time = torch.gather(
            remote,
            dim=2,
            index=safe_index[:, None, :, None].expand(B, G, T, H),
        ).transpose(1, 2)  # (B,T,G,H)

        weather_t = weather.transpose(1, 2)  # (B,T,G,H)
        q = self.W_q(weather_t).view(B, T, G, self.num_heads, self.head_dim)
        k = self.W_k(remote_by_time).view(B, T, G, self.num_heads, self.head_dim)
        v = self.W_v(remote_by_time).view(B, T, G, self.num_heads, self.head_dim)
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        distance = torch.cdist(coords.float(), coords.float())
        distance = distance / distance.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        scores = scores - F.softplus(self.distance_scale) * distance[:, None, None]

        query_valid = grid_mask[:, None, None, :, None]
        remote_valid = grid_mask[:, None, None, None, :]
        month_available = torch.gather(
            ag_mask.to(weather.device, dtype=torch.bool), 1, safe_index
        ) & valid_month
        valid = query_valid & remote_valid & month_available[:, :, None, None, None]
        scores = scores.masked_fill(~valid, -1e9)
        weights = torch.softmax(scores, dim=-1) * valid.to(scores.dtype)
        weights = self.dropout(weights)
        remote_context = torch.matmul(weights, v)
        remote_context = remote_context.permute(0, 1, 3, 2, 4).reshape(B, T, G, H)

        if soil_context is None:
            soil_context = torch.zeros(B, H, device=weather.device, dtype=weather.dtype)
        context = soil_context[:, None, None, :].expand(B, T, G, H)
        fused = self.fusion_grn(weather_t + remote_context, context=context)
        fused = fused * grid_mask[:, None, :, None].to(fused.dtype)
        return fused.transpose(1, 2), weights.mean(dim=2)


class WeatherRemotePatchPretrain(nn.Module):
    """MMST-style pretraining: weather tokens query remote image patches."""

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout))
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size)
        )
        self.scale = math.sqrt(float(self.head_dim))

    def forward(self, weather_tokens: torch.Tensor, remote_patches: torch.Tensor):
        B, Nw, H = weather_tokens.shape
        _, Np, _ = remote_patches.shape
        q = self.q(weather_tokens).view(B, Nw, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(remote_patches).view(B, Np, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(remote_patches).view(B, Np, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        fused = torch.matmul(weights, v).transpose(1, 2).reshape(B, Nw, H)
        fused = self.out(fused)
        embedding = self.projection(fused.mean(dim=1))
        return embedding, weights.mean(dim=1)


# ============================================================
# ============================================================
# 遥感特征编码模块
# ============================================================

AG_DATES = [f"{month:02d}-{day:02d}" for month, day in AG_DATE_PAIRS]
AG_DAY_INDICES = [0, 30, 61, 91, 122, 153]


class PretrainedViTEncoder(nn.Module):
    """PVT-Tiny 风格编码器，直接输出与天气 token 同维的网格级 token。"""

    def __init__(self, out_dim: int = 32, freeze_backbone: bool = False, dropout: float = 0.1):
        super().__init__()
        from pvt import PVTTinyEncoder
        self.backbone = PVTTinyEncoder(out_dim=out_dim, drop=dropout)
        self.embed_dim = out_dim
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TFTEncoderForYieldPrediction(nn.Module):
    """TFT 编码器 + 产量预测头：无解码器，直接 LSTM → 注意力 → 预测头。"""
    def __init__(
        self,
        soil_dim: int,
        dynamic_feature_names: List[str],
        hidden_size: int,
        num_lstm_layers: int = 1,
        dropout: float = 0.3,
        output_size = 1,
        num_heads: int = 3,
        vit_freeze_backbone: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dynamic_feature_names = list(dynamic_feature_names)

        # 1. 静态：县级连续土壤(Linear 映射,不分桶)+ 上下文 GRN
        self.soil_static_encoder = SoilStaticEncoder(
            soil_dim=soil_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        # 2. 每列 1 维动态特征 -> hidden，供网格内 VSN 选择
        self.per_feature_linear = nn.ModuleDict(
            {name: nn.Linear(1, hidden_size) for name in self.dynamic_feature_names}
        )

        vsn_inputs = {name: hidden_size for name in self.dynamic_feature_names}
        self.grid_vsn = VariableSelectionNetwork(
            input_sizes=vsn_inputs,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size,
        )

        # 4. LSTM 编码器
        self.lstm_encoder = LSTMEncoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout,
            bidirectional=False,
            have_context=True,
        )

        # 土壤条件化的天气/遥感投影与跨模态注意力
        self.weather_context_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            context_size=hidden_size,
            dropout=dropout,
        )
        self.remote_context_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            context_size=hidden_size,
            dropout=dropout,
        )
        self.weather_remote_attn = WeatherRemoteCrossAttention(
            hidden_size, num_heads=num_heads, dropout=dropout
        )

        # LSTM 输出经 GRN 准备 → 因果注意力
        self.cat_attn_prep_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            context_size=hidden_size,
            dropout=dropout,
        )

        # 5. 多头纯因果自注意力
        self.attention = CausalScaledDotProductAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.spatial_agg = SpatialAttentionAggregator(hidden_size, dropout)

        # 6. 产量预测头
        self.pred_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            context_size=None,
            dropout=dropout,
        )
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )
        self.variance_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

        # 遥感模块是当前模型的必需输入分支。
        self.vit_encoder = PretrainedViTEncoder(
            out_dim=hidden_size, freeze_backbone=vit_freeze_backbone, dropout=dropout,
        )

    def forward(
        self,
        grid_feats: torch.Tensor,
        grid_coords: torch.Tensor,
        grid_mask: torch.Tensor,
        soil_feats: torch.Tensor,
        seq_lens: torch.Tensor,
        month_ids: Optional[torch.Tensor] = None,
        day_ids: Optional[torch.Tensor] = None,
        ag_images: Optional[torch.Tensor] = None,
        ag_mask: Optional[torch.Tensor] = None,
        diagnose_mode: Optional[Dict[str, bool]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            grid_feats: (B, G, T, F) 每县 G 个 9×9km 网格的标准化气象
            grid_coords: (B, G, 2) 每网格 [lat, lon]
            grid_mask: (B, G) bool 有效网格
            ...
            diagnose_mode: 可选诊断开关，支持:
            soil_feats: (B, soil_dim) 县级连续土壤静态特征(已标准化,不进网格注意力)
            seq_lens: (batch_size,) 有效时序长度
            ag_images: (B, 6, G, 3, 224, 224) 遥感图像

        Returns:
            pred_all: (B, 1) 县级最终产量预测
            attn_weights_out: (B, T_new, T_new) 联合序列注意力权重
            aux_dict: 含 grad_tensors、pred_all
        """
        B, _, T, _ = grid_feats.shape
        device = grid_feats.device

        # 诊断模式开关
        diag = diagnose_mode or {}

        c_s, c_e, c_c, c_h = self.soil_static_encoder(soil_feats)
        G = grid_feats.shape[1]
        projected = {
            name: self.per_feature_linear[name](grid_feats[..., j:j + 1])
            for j, name in enumerate(self.dynamic_feature_names)
        }
        grid_inputs = {
            name: tensor.reshape(B * G, T, self.hidden_size)
            for name, tensor in projected.items()
        }
        grid_seq_lens = seq_lens.repeat_interleave(G)
        grid_context = c_s.unsqueeze(1).expand(B, G, self.hidden_size)
        grid_context = grid_context.reshape(B * G, self.hidden_size)
        grid_context = grid_context.unsqueeze(1).expand(B * G, T, self.hidden_size)
        grid_token, grid_vsn_weights = self.grid_vsn(
            grid_inputs, grid_seq_lens, context=grid_context
        )
        grid_token = grid_token.reshape(B, G, T, self.hidden_size)
        grid_token = grid_token * grid_mask[:, :, None, None].to(grid_token.dtype)

        weather_feat = self.weather_context_grn(
            grid_token,
            context=c_s[:, None, None, :].expand_as(grid_token),
        )
        weather_feat = weather_feat * grid_mask[:, :, None, None].to(weather_feat.dtype)

        if ag_images is None or ag_mask is None:
            raise ValueError("当前模型必须传入 ag_images 和 ag_mask")
        rs_encoded = self._forward_remote_sensing(
            ag_images, grid_coords, grid_mask, device,
        )  # (B, G, N_rs, H)
        rs_encoded_pre_grn = rs_encoded
        rs_context = c_s[:, None, None, :].expand_as(rs_encoded)
        rs_encoded = self.remote_context_grn(rs_encoded, context=rs_context)

        # ========== 5. 同时相空间融合 -> 时间 TFT 注意力 ==========
        Te = int(T)
        pad_mask = torch.arange(Te, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
        if month_ids is None or day_ids is None:
            raise ValueError("必须传入 month_ids 和 day_ids")
        time_indices = _relative_day_index(month_ids[:, :Te], day_ids[:, :Te])
        weather_fused, cross_modal_weights = self.weather_remote_attn(
            weather_feat,
            rs_encoded,
            grid_coords,
            grid_mask,
            month_ids[:, :Te],
            ag_mask,
            soil_context=c_s,
        )
        temporal_feat, spatial_weights = self.spatial_agg.forward_weights(
            weather_fused.transpose(1, 2),
            grid_coords,
            grid_mask,
            time_indices=time_indices,
        )
        # 空间融合后才进入时间编码，避免每个网格分别保存一套 LSTM 激活。
        lstm_feat_raw, _, _, _ = self.lstm_encoder(
            temporal_feat, seq_lens=seq_lens, c_c=c_c, c_h=c_h
        )
        temporal_context = c_e[:, None, :].expand(B, Te, self.hidden_size)
        temporal_feat = self.cat_attn_prep_grn(
            lstm_feat_raw, context=temporal_context
        )
        temporal_feat = temporal_feat * pad_mask.unsqueeze(-1).to(temporal_feat.dtype)
        attn_feat, attn_weights_out = self.attention(
            x=temporal_feat + self._compute_st_pe(
                grid_coords.new_zeros(B, Te, 1),
                grid_coords.new_zeros(B, Te, 1),
                time_indices,
            ).to(temporal_feat.dtype),
            pad_mask=pad_mask,
        )
        pred_features = self.pred_grn(attn_feat)
        pred_dist_all = torch.cat(
            [self.mean_head(pred_features), self.variance_head(pred_features)], dim=-1
        )  # (B,T,2)
        last_token_idx = pad_mask.sum(dim=1).clamp_min(1) - 1
        batch_idx = torch.arange(B, device=device)
        last_token = attn_feat[batch_idx, last_token_idx]

        # 最终预测取最后有效时间步；所有时间步分布保留给一致性损失。
        pred_all = pred_dist_all[batch_idx, last_token_idx]  # (B, 2)

        grad_tensors = {
            "static_feat": c_s,
            "grid_vsn_weights": grid_vsn_weights,
            "county_vsn_weights": None,
            "temporal_feat": temporal_feat,
            "cat_feat": temporal_feat,
            "lstm_feat_raw": lstm_feat_raw,
            "last_token": last_token,
            "pred_features": pred_features,
            "pred_dist_all": pred_dist_all,
            "cross_modal_weights": cross_modal_weights,
            "weather_fused": weather_fused,
            "pred_all": pred_all,
        }
        if rs_encoded_pre_grn is not None:
            grad_tensors["rs_encoded_pre_grn"] = rs_encoded_pre_grn
            grad_tensors["rs_encoded"] = rs_encoded
        for tensor in grad_tensors.values():
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.retain_grad()

        aux_dict: Dict[str, Any] = {
            "grad_tensors": grad_tensors,
            "pred_all": pred_all,
            "grid_vsn_weights": grid_vsn_weights,
            "county_vsn_weights": None,
            "spatial_weights": spatial_weights,
            "cross_modal_weights": cross_modal_weights,
        }

        return pred_all, attn_weights_out, aux_dict

    def _compute_st_pe(
        self, lat: torch.Tensor, lon: torch.Tensor, t_idx: torch.Tensor
    ) -> torch.Tensor:
        """WeatherFormer 四槽加性时空位置编码。
        lat, lon: (B, T, 1) rad; t_idx: (B, T) int; 返回 (B, T, H)."""
        d = self.hidden_size
        nf = d // 4
        device = lat.device
        i = torch.arange(nf, device=device, dtype=lat.dtype)
        freq = 10000.0 ** (-4.0 * i / d)
        freq = freq.view(1, 1, nf)  # (1, 1, nf)

        t = t_idx.unsqueeze(-1).float()  # (B, T, 1)
        s_lat = lat  # (B, T, 1)
        s_lon = lon

        pe = torch.zeros(*lat.shape[:-1], d, device=device, dtype=lat.dtype)
        pe[..., 0::4] = torch.sin(t * freq)
        pe[..., 1::4] = torch.cos(t * freq)
        pe[..., 2::4] = torch.sin(s_lat * freq)
        pe[..., 3::4] = torch.cos(s_lon * freq)
        return pe

    def _forward_remote_sensing(
        self,
        ag_images: torch.Tensor,
        grid_coords: torch.Tensor,
        grid_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """遥感编码: ViT → 网格级 RS token（不聚合）。

        返回 rs_encoded (B, G, 12, H)，每网格每时相一个 token。"""
        B, N_rs, G, C, H_img, W_img = ag_images.shape
        H = self.hidden_size

        images = ag_images.permute(0, 2, 1, 3, 4, 5).reshape(B * G, N_rs, C, H_img, W_img)
        images = images.reshape(B * G * N_rs, C, H_img, W_img)
        # PVT is fully trainable; process image tokens in chunks to avoid
        # holding all B*G*N_rs backbone activations at once.
        chunk_size = 16
        encoded_chunks = []
        for start in range(0, images.shape[0], chunk_size):
            encoded_chunks.append(self.vit_encoder(images[start:start + chunk_size]))
        rs_encoded = torch.cat(encoded_chunks, dim=0)
        rs_encoded = rs_encoded.reshape(B, G, N_rs, H)
        rs_encoded = rs_encoded * grid_mask[:, :, None, None].to(rs_encoded.dtype)
        return rs_encoded
