import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math
from typing import Any, Dict, List, Tuple, Optional


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
        rope: Optional[nn.Module] = None,
        rope_t: Optional[torch.Tensor] = None,
        rope_lat: Optional[torch.Tensor] = None,
        rope_lon: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        v = self.W_v(x)
        q = self._split_heads(self.W_q(x))
        k = self._split_heads(self.W_k(x))

        if rope is not None:
            q, k = rope(q, k, rope_t, rope_lat, rope_lon)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            float(self.head_dim)
        )

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
        freq = 10000.0 ** (-4.0 * i / d)
        t = t_idx.to(device=coords.device, dtype=coords.dtype).view(1, 1, -1, 1)
        lat = (coords[..., 0:1] * (math.pi / 180.0)).unsqueeze(2)
        lon = (coords[..., 1:2] * (math.pi / 180.0)).unsqueeze(2)
        pe = torch.zeros(
            *coords.shape[:-1], t_idx.numel(), d,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return county features and CLS-to-grid weights shaped (B,T,G)."""
        # tokens: (B, G, T, H) 该特征每网格的 d 维内容投影
        # coords: (B, G, 2), grid_mask: (B, G) bool
        B, _, T, H = tokens.shape
        t_idx = torch.arange(T, device=tokens.device, dtype=torch.long)
        cls_coords = self._cls_coords(coords, grid_mask)
        pe_grid = self._st_pe(coords, t_idx).transpose(1, 2)  # (B,T,G,H)
        pe_cls = self._st_pe(cls_coords.unsqueeze(1), t_idx).squeeze(1)  # (B,T,H)
        cls = self.cls_token.view(1, 1, H).expand(B, T, H)
        q = self.W_q(cls + pe_cls).unsqueeze(2)          # (B,T,1,H)
        x = tokens.transpose(1, 2)                       # (B,T,G,H)
        x_with_pe = x + pe_grid
        k = self.W_k(x_with_pe)
        v = self.W_v(x_with_pe)
        scores = torch.matmul(q, k.transpose(-2, -1)).squeeze(2) / self.scale
        key_valid = grid_mask.unsqueeze(1)                       # (B,1,G)
        scores = scores.masked_fill(~key_valid, float("-inf"))
        w = torch.softmax(scores, dim=-1)                         # (B,T,G)
        w = self.dropout(w)
        w = w * key_valid.to(w.dtype)
        cls_out = torch.matmul(w.unsqueeze(2), v).squeeze(2)
        return self.norm(cls_out), w


# ============================================================
# 3D Rotary Position Embedding（多模态时空对齐）
# ============================================================

class RotaryEmbedding3D(nn.Module):
    """3D RoPE：将 head_dim 三等分，分别做 (time, lat, lon) 旋转位置编码。

    每部分 dim_p = head_dim // 3（向下取偶），对应部分对相邻维度对旋转：
      part_a[0:dim_p]:   angle = t * freq_i
      part_b[dim_p:2*dim_p]: angle = lat * freq_i
      part_c[2*dim_p:3*dim_p]: angle = lon * freq_i
    尾部余数不做旋转，直通。
    频率 freq_i = 10000^{-2i / dim_p}。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        raw_p = dim // 3
        self.dim_p = max(2, (raw_p // 2) * 2)  # 向下取偶，≥2
        self.rot_dim = 3 * self.dim_p
        self.unused_dim = dim - self.rot_dim
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.dim_p, 2).float() / self.dim_p))
        self.register_buffer("inv_freq", inv_freq)

    def _rotate(self, x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        x_rot[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
        return x_rot

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        t_indices: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = self.inv_freq.to(q.device)  # (dim_p/2,)
        freqs = freqs.view(1, 1, 1, -1)  # (1,1,1, dim_p/2)

        t = t_indices.unsqueeze(-1).float().unsqueeze(1)  # (B, 1, T, 1)
        s_lat = lat.unsqueeze(1)
        s_lon = lon.unsqueeze(1)

        d = self.dim_p
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)

        q_out[..., 0:d]       = self._rotate(q[..., 0:d],       t * freqs)
        k_out[..., 0:d]       = self._rotate(k[..., 0:d],       t * freqs)
        q_out[..., d:2*d]     = self._rotate(q[..., d:2*d],     s_lat * freqs)
        k_out[..., d:2*d]     = self._rotate(k[..., d:2*d],     s_lat * freqs)
        q_out[..., 2*d:3*d]   = self._rotate(q[..., 2*d:3*d],   s_lon * freqs)
        k_out[..., 2*d:3*d]   = self._rotate(k[..., 2*d:3*d],   s_lon * freqs)

        if self.unused_dim > 0:
            q_out[..., 3*d:] = q[..., 3*d:]
            k_out[..., 3*d:] = k[..., 3*d:]

        return q_out, k_out


# ============================================================
# 遥感特征编码模块
# ============================================================

AG_DATES = [
    "04-01", "04-15", "05-01", "05-15", "06-01", "06-15",
    "07-01", "07-15", "08-01", "08-15", "09-01", "09-15",
]
AG_DAY_INDICES = [0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154]


class PretrainedViTEncoder(nn.Module):
    """DINOv2 ViT-S（facebook/dinov2-small）遥感图像编码器。

    DINOv2 在 1.42 亿张多样化图像上自监督预训练，
    遥感领域广泛使用（Panopticon CVPR2025 / SoftCon IEEE TGRS 等）。
    冻结 backbone，只训练投影头 (384 → H*2 → H)。"""

    def __init__(self, out_dim: int = 32, freeze_backbone: bool = True, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-small")
        self.embed_dim = self.backbone.config.hidden_size
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(self.embed_dim, out_dim * 2), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(out_dim * 2, out_dim)
            )
        self._init_weights()

    def _init_weights(self):
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(
            not all(not p.requires_grad for p in self.backbone.parameters())
        ):
            out = self.backbone(pixel_values=x)
        return self.proj(out.last_hidden_state[:, 0, :])


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
        spatial_mode: str = "attention",
        variable_selection_stage: str = "grid",
        use_remote_sensing: bool = False,
        vit_freeze_backbone: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dynamic_feature_names = list(dynamic_feature_names)
        self.spatial_mode = spatial_mode
        self.variable_selection_stage = variable_selection_stage
        self.use_remote_sensing = use_remote_sensing

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
        if variable_selection_stage == "grid":
            self.grid_vsn = VariableSelectionNetwork(
                input_sizes=vsn_inputs,
                hidden_size=hidden_size,
                dropout=dropout,
                context_size=hidden_size,
            )
            self.county_vsn = None
        elif variable_selection_stage == "county":
            self.grid_vsn = None
            self.county_vsn = VariableSelectionNetwork(
                input_sizes=vsn_inputs,
                hidden_size=hidden_size,
                dropout=dropout,
                context_size=hidden_size,
            )
        else:
            raise ValueError("variable_selection_stage must be 'grid' or 'county'")

        # 4. LSTM 编码器（仅编码器，无解码器）
        self.lstm_encoder = LSTMEncoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout,
            bidirectional=False,
            have_context=True,
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

        # 6. 产量预测头
        self.pred_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            context_size=None,
            dropout=dropout,
        )
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

        # 网格级:逐特征 WeatherFormer 注意力(Q/K/V 均由内容+位置编码生成)
        # spatial_mode="mean" 为消融对照:退化为掩码加权平均(直接网格均值),不创建 spatial_agg
        if spatial_mode == "attention":
            self.spatial_agg = SpatialAttentionAggregator(hidden_size, dropout)
        elif spatial_mode == "mean":
            self.spatial_agg = None
        else:
            raise ValueError(f"未知 spatial_mode: {spatial_mode}，可选 'attention' / 'mean'")

        # ========== 遥感模块 ==========
        if use_remote_sensing:
            self.vit_encoder = PretrainedViTEncoder(
                out_dim=hidden_size, freeze_backbone=vit_freeze_backbone, dropout=dropout,
            )
            self.rs_rope = RotaryEmbedding3D(hidden_size // num_heads)
        else:
            self.vit_encoder = None
            self.rs_rope = None

    def forward(
        self,
        grid_feats: torch.Tensor,
        grid_coords: torch.Tensor,
        grid_mask: torch.Tensor,
        soil_feats: torch.Tensor,
        seq_lens: torch.Tensor,
        ag_images: Optional[torch.Tensor] = None,
        diagnose_mode: Optional[Dict[str, bool]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            grid_feats: (B, G, T, F) 每县 G 个 9×9km 网格的标准化气象
            grid_coords: (B, G, 2) 每网格 [lat, lon]
            grid_mask: (B, G) bool 有效网格
            ...
            diagnose_mode: 可选诊断开关，支持:
                disable_rs: 强制跳过遥感分支
                force_mean_spatial: 强制将空间注意力替换为掩码均值
            soil_feats: (B, soil_dim) 县级连续土壤静态特征(已标准化,不进网格注意力)
            seq_lens: (batch_size,) 有效时序长度
            ag_images: (B, 12, G, 3, 224, 224) 可选遥感图像, 仅 use_remote_sensing=True 生效

        Returns:
            pred_all: (B, T, 1) 逐时间步产量预测
            attn_weights_out: (B, T, T) 注意力权重
            aux_dict: 含 grad_tensors、pred_all
        """
        B, _, T, _ = grid_feats.shape
        max_seq_len = int(T)
        device = grid_feats.device

        # 诊断模式开关
        diag = diagnose_mode or {}
        _disable_rs = bool(diag.get("disable_rs", False))
        _force_mean = bool(diag.get("force_mean_spatial", False))
        _eff_spatial_mode = "mean" if _force_mean else self.spatial_mode

        c_s, c_e, c_c, c_h = self.soil_static_encoder(soil_feats)
        c_s_expanded = c_s.unsqueeze(1).repeat(1, max_seq_len, 1)

        G = grid_feats.shape[1]
        projected = {
            name: self.per_feature_linear[name](grid_feats[..., j:j + 1])
            for j, name in enumerate(self.dynamic_feature_names)
        }
        if self.variable_selection_stage == "grid":
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
            if _eff_spatial_mode == "attention":
                temporal_feat, spatial_weights = self.spatial_agg.forward_weights(
                    grid_token, grid_coords, grid_mask
                )
            else:
                denom = grid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                temporal_feat = grid_token.sum(dim=1) / denom.unsqueeze(-1)
                spatial_weights = None
            county_vsn_weights = None
        else:
            county_inputs = {}
            feature_spatial_weights = []
            for name, tensor in projected.items():
                tensor = tensor * grid_mask[:, :, None, None].to(tensor.dtype)
                if _eff_spatial_mode == "attention":
                    pooled, weights = self.spatial_agg.forward_weights(
                        tensor, grid_coords, grid_mask
                    )
                    feature_spatial_weights.append(weights)
                else:
                    denom = grid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                    pooled = tensor.sum(dim=1) / denom.unsqueeze(-1)
                county_inputs[name] = pooled
            temporal_feat, county_vsn_weights = self.county_vsn(
                county_inputs, seq_lens, context=c_s_expanded
            )
            grid_vsn_weights = None
            spatial_weights = (
                torch.stack(feature_spatial_weights, dim=-1)
                if feature_spatial_weights else None
            )

        # ========== 遥感: ViT → 网格级 RS token（不聚合）==========
        rs_encoded = None
        if self.use_remote_sensing and ag_images is not None and not _disable_rs:
            rs_encoded = self._forward_remote_sensing(
                ag_images, grid_coords, grid_mask, device,
            )  # (B, G, 12, H)

        # ========== 4. LSTM 编码器 ==========
        lstm_feat_raw, _, _, _ = self.lstm_encoder(
            temporal_feat, seq_lens=seq_lens, c_c=c_c, c_h=c_h
        )

        # ========== 5. GRN 准备 + 时空交错 + 因果注意力 ==========
        Te = int(lstm_feat_raw.size(1))
        pad_mask = torch.arange(Te, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
        pm_f = pad_mask.unsqueeze(-1).to(dtype=lstm_feat_raw.dtype)
        c_e_expanded = c_e.unsqueeze(1).repeat(1, Te, 1) * pm_f
        cat_feat = self.cat_attn_prep_grn(lstm_feat_raw, context=c_e_expanded)
        cat_feat = cat_feat * pm_f

        if self.use_remote_sensing and rs_encoded is not None:
            _, G, N_rs, _ = rs_encoded.shape  # N_rs=12

            # 网格经纬度 → rad
            grid_lat = grid_coords[:, :, 0:1] * (math.pi / 180.0)  # (B, G, 1)
            grid_lon = grid_coords[:, :, 1:2] * (math.pi / 180.0)

            rs_days = torch.tensor(AG_DAY_INDICES, device=device, dtype=torch.long)

            # 每个 RS 时相插入 G 个网格 token，序列总长 = Te + G*N_rs
            T_new = Te + G * N_rs

            # 构建交错序列和坐标，每样本 G 相同（batch 内 G 已固定）
            # 布局: [w0, rs0_g0..rs0_g{G-1}, w1, ..., w13, rs1_g0.., w14, ...]
            interleaved = torch.zeros(B, T_new, self.hidden_size, device=device, dtype=cat_feat.dtype)
            t_indices = torch.zeros(B, T_new, device=device, dtype=torch.long)
            token_lat = torch.zeros(B, T_new, device=device)
            token_lon = torch.zeros(B, T_new, device=device)

            # 填充天气 token
            w_positions = []
            offset = 0
            # 县中心坐标 (所有天气 token 共用)
            valid = grid_mask.to(dtype=grid_coords.dtype).unsqueeze(-1)
            county_center = (grid_coords * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            county_lat = county_center[:, 0] * (math.pi / 180.0)  # (B,)
            county_lon = county_center[:, 1] * (math.pi / 180.0)

            for t in range(Te):
                w_positions.append(offset)
                interleaved[:, offset, :] = cat_feat[:, t, :]
                t_indices[:, offset] = t
                token_lat[:, offset] = county_lat
                token_lon[:, offset] = county_lon
                offset += 1
                # 在 RS 日期插入 G 个网格 RS token（用各自网格坐标）
                if t in rs_days:
                    rs_i = (rs_days == t).nonzero(as_tuple=False)[0].item()
                    for g in range(G):
                        interleaved[:, offset, :] = rs_encoded[:, g, rs_i, :]
                        t_indices[:, offset] = t
                        token_lat[:, offset] = grid_lat[:, g, 0]
                        token_lon[:, offset] = grid_lon[:, g, 0]
                        offset += 1

            w_positions = torch.tensor(w_positions, device=device)
            interleave_pad = torch.ones(B, T_new, dtype=torch.bool, device=device)

            attn_feat_interleaved, attn_weights_out = self.attention(
                x=interleaved, pad_mask=interleave_pad,
                rope=self.rs_rope, rope_t=t_indices,
                rope_lat=token_lat.unsqueeze(-1), rope_lon=token_lon.unsqueeze(-1),
            )
            attn_feat = attn_feat_interleaved[:, w_positions, :]
        else:
            attn_feat, attn_weights_out = self.attention(
                x=cat_feat, pad_mask=pad_mask,
            )

        # ========== 6. 产量预测头（逐时间步）==========
        pred_all = self.pred_head(self.pred_grn(attn_feat))  # (B, T, 1)

        grad_tensors = {
            "static_feat": c_s,
            "grid_vsn_weights": grid_vsn_weights,
            "county_vsn_weights": county_vsn_weights,
            "temporal_feat": temporal_feat,
            "cat_feat": cat_feat,
            "lstm_feat_raw": lstm_feat_raw,
            "pred_all": pred_all,
        }
        for tensor in grad_tensors.values():
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.retain_grad()

        aux_dict: Dict[str, Any] = {
            "grad_tensors": grad_tensors,
            "pred_all": pred_all,
            "grid_vsn_weights": grid_vsn_weights,
            "county_vsn_weights": county_vsn_weights,
            "spatial_weights": spatial_weights,
        }

        return pred_all, attn_weights_out, aux_dict

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
        rs_encoded = self.vit_encoder(images)
        rs_encoded = rs_encoded.reshape(B, G, N_rs, H)
        rs_encoded = rs_encoded * grid_mask[:, :, None, None].to(rs_encoded.dtype)
        return rs_encoded
