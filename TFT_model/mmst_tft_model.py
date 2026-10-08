"""Soil-conditioned TFT temporal model over exact MMST spatial tokens."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .mmst_frontend import MMSTSpatialFrontend
from .models import (
    CausalScaledDotProductAttention,
    GateAddNorm,
    GatedResidualNetwork,
    LSTMEncoder,
    SoilStaticEncoder,
)


def last_true_index(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回每行最后一个 True 的位置；全 False 行返回 index 0 和 valid=False。"""
    if mask.ndim != 2:
        raise ValueError("mask must have shape (B, T)")
    mask = mask.to(dtype=torch.bool)
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(~mask, -1).max(dim=1).values
    sample_valid = indices >= 0
    return indices.clamp_min(0), sample_valid


class MMSTTFTYieldModel(nn.Module):
    """Combine the official MMST spatial front-end with a TFT temporal back-end."""

    def __init__(
        self,
        checkpoint_path,
        soil_dim: int = 7,
        hidden_size: int = 512,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.frontend = MMSTSpatialFrontend(checkpoint_path, dim=512)
        self.frontend_projection = (
            nn.Identity()
            if self.hidden_size == 512
            else nn.Linear(512, self.hidden_size)
        )
        self.soil_static_encoder = SoilStaticEncoder(
            soil_dim=soil_dim,
            hidden_size=self.hidden_size,
            dropout=dropout,
        )
        # The MMST frontend is one logical dynamic variable; soil conditions
        # that variable instead of selecting its 512 embedding dimensions.
        self.remote_variable_grn = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=dropout,
            context_size=self.hidden_size,
        )
        self.lstm_encoder = LSTMEncoder(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout=dropout,
            bidirectional=False,
            have_context=True,
        )
        self.temporal_grn = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=dropout,
            context_size=self.hidden_size,
        )
        self.attention = CausalScaledDotProductAttention(
            hidden_size=self.hidden_size,
            num_heads=heads,
            dropout=dropout,
        )
        self.attention_gate = GateAddNorm(
            input_size=self.hidden_size,
            skip_size=self.hidden_size,
            dropout=dropout,
        )
        self.pred_grn = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=dropout,
        )
        self.yield_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(
        self,
        images: torch.Tensor,
        short_weather: torch.Tensor,
        soil: torch.Tensor,
        month_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        frontend_tokens = self.frontend(images, short_weather)
        if frontend_tokens.shape[1:] != (6, 512):
            raise ValueError("MMST front-end must return shape (B, 6, 512)")
        batch = frontend_tokens.shape[0]
        if soil.shape != (batch, self.soil_static_encoder.soil_dim):
            raise ValueError(
                "soil must have shape "
                f"({batch}, {self.soil_static_encoder.soil_dim})"
            )
        if month_mask is None:
            month_valid = torch.ones(
                batch, 6, device=frontend_tokens.device, dtype=torch.bool,
            )
        else:
            if month_mask.shape != (batch, 6):
                raise ValueError(f"month_mask shape {month_mask.shape} != {(batch, 6)}")
            month_valid = month_mask.to(
                device=frontend_tokens.device, dtype=torch.bool,
            )

        c_s, c_e, c_c, c_h = self.soil_static_encoder(soil)
        temporal_input = self.frontend_projection(frontend_tokens)
        temporal_context = c_s[:, None, :].expand(-1, 6, -1)
        temporal_input = self.remote_variable_grn(
            temporal_input,
            context=temporal_context,
        )
        temporal_input = temporal_input * month_valid.unsqueeze(-1).to(
            temporal_input.dtype
        )
        lstm_features, _, _, _ = self.lstm_encoder(
            temporal_input,
            seq_lens=month_valid.sum(dim=1),
            c_c=c_c,
            c_h=c_h,
            valid_mask=month_valid,
        )
        temporal_context = c_e[:, None, :].expand(-1, 6, -1)
        temporal_features = self.temporal_grn(
            lstm_features, context=temporal_context,
        )
        temporal_features = temporal_features * month_valid.unsqueeze(-1).to(
            temporal_features.dtype
        )
        attention_output, temporal_weights = self.attention(
            temporal_features, pad_mask=month_valid,
        )
        attention_features = self.attention_gate(
            attention_output,
            temporal_features,
        )
        pred_features = self.pred_grn(attention_features)
        pred_dist_all = self.yield_head(pred_features)
        pred_dist_all = pred_dist_all * month_valid.unsqueeze(-1).to(
            pred_dist_all.dtype
        )

        last_valid, sample_valid = last_true_index(month_valid)
        batch_indices = torch.arange(batch, device=frontend_tokens.device)
        final_dist = pred_dist_all[batch_indices, last_valid]
        final_dist = final_dist * sample_valid.unsqueeze(-1).to(final_dist.dtype)

        aux: Dict[str, Any] = {
            "frontend_tokens": frontend_tokens,
            "month_valid": month_valid,
            "pred_dist_all": pred_dist_all,
        }
        return final_dist, temporal_weights, aux
