# PhenoYieldNet: crop-aware phenological temporal decoder for multi-crop yield prediction.
#
# Derived from the multi-crop model of MMST-ViT
# (https://github.com/fudong03/MMST-ViT). See README.md (Acknowledgements).
#
# Upstream: roroyo/PhenoYieldNet@5a14ed7f7882ef6e2d5d4680c373e12da0110147 (CC BY-NC 4.0).
# Local change (marked [unified CropNet]): optional crop_growth_cycles override.

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat

from attention import SpatialTransformer, MultiCropTrendAwareTemporalTransformer


class ViT_multimodal_multicrop(nn.Module):
    """Multi-crop yield regressor.

    A SpectralGPT-initialised ViT backbone encodes each (month, grid) Sentinel-2
    patch jointly with its HRRR weather context. Grid tokens are pooled by a
    spatial transformer, then the month sequence is decoded by a crop-aware,
    trend/seasonality-biased temporal transformer restricted to the crop's own
    phenological window.
    """

    def __init__(self, num_short_term_seq, out_dim=2, num_grid=64,
                 backbone=None, context_dim=9, dim=192, batch_size=64, depth=4,
                 heads=3, pool='mean', dim_head=64, dropout=0., emb_dropout=0.,
                 scale_dim=4, num_crops=4, grad_checkpointing=False,
                 freeze_backbone=False, crop_growth_cycles=None):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.batch_size = batch_size
        self.backbone = backbone
        self.context_dim = context_dim
        self.grad_checkpointing = grad_checkpointing

        # Use the Stage-1 backbone as a fixed feature extractor. With only ~626
        # training counties against 82M parameters, fine-tuning all of it
        # overfits well before the validation metrics converge.
        self.freeze_backbone = freeze_backbone
        if freeze_backbone and backbone is not None:
            for prm in backbone.parameters():
                prm.requires_grad = False

        # Spatial branch: one token per 9x9 km grid cell, plus a CLS token.
        # NOTE: num_grid caps the grid cells per county (see README, "Known
        # limitations") - counties with more than num_grid - 1 grids will fail.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_short_term_seq, num_grid, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = SpatialTransformer(dim, depth, heads, dim_head,
                                                    mult=scale_dim, dropout=dropout)

        # Temporal branch: crop-aware trend-aware decoder.
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.multicrop_trend_aware_temporal_transformer = MultiCropTrendAwareTemporalTransformer(
            dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout)

        # Month indices (0-based) spanning each crop's growing season.
        self.crop_growth_cycles = {
            0: list(range(3, 9)),     # soybean:     Apr-Sep
            1: list(range(3, 12)),    # corn:        Apr-Dec
            2: list(range(3, 12)),    # cotton:      Apr-Dec
            3: list(range(0, 12)),    # winterwheat: Jan-Dec
        }
        # [unified CropNet] 时序轴只有 4--9 月（索引 0 = 4 月），由调用方给出各作物在该轴上的窗口。
        if crop_growth_cycles is not None:
            self.crop_growth_cycles = {k: list(v) for k, v in crop_growth_cycles.items()}

        # Learnable per-crop query that drives the temporal cross-attention.
        self.num_crops = num_crops
        self.crop_query_bank = nn.Parameter(torch.randn(num_crops, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        # Crop-specific regression heads. The four crops are reported in
        # different units and span very different magnitudes (bu/acre for
        # soybean/corn/wheat, lb/acre for cotton), so a single shared head has
        # to absorb that scale gap on top of the actual yield signal. One head
        # per crop decouples them; the shared trunk still carries all the
        # cross-crop representation learning.
        self.mlp_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))
            for _ in range(num_crops)
        ])

    def train(self, mode=True):
        # nn.Module.train() recurses into every child, which would put a frozen
        # backbone back into training mode (dropout / drop-path active). Keep it
        # in eval so the extracted features stay deterministic.
        super().train(mode)
        if self.freeze_backbone and self.backbone is not None:
            self.backbone.eval()
        return self

    def forward_features(self, x, ys):
        x = rearrange(x, 'b t g c h w -> (b t g) c h w')
        ys = rearrange(ys, 'b t g n d -> (b t g) n d')

        # A county-month step encodes (12 months x #grids) patches at once, and
        # every chunk's activations stay alive until backward, so chunking alone
        # does not bound memory. With grad_checkpointing the backbone activations
        # are recomputed in the backward pass instead of stored: mathematically
        # identical, roughly 30% slower, and it is what makes large counties fit
        # on a ~30 GB GPU.
        B = x.shape[0]
        n = B // self.batch_size if B % self.batch_size == 0 else B // self.batch_size + 1

        chunks = []
        for i in range(n):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            x_i, ys_i = x[start:end], ys[start:end]
            if self.grad_checkpointing and self.training:
                x_hat_tmp = checkpoint(self.backbone, x_i, ys_i, use_reentrant=False)
            else:
                x_hat_tmp = self.backbone(x_i, ys_i)
            chunks.append(x_hat_tmp)

        return torch.cat(chunks, dim=0)

    def forward(self, x, ys=None, crop_idx=None, time_steps=None):
        b, t, g, _, _, _ = x.shape

        x = self.forward_features(x, ys)
        x = rearrange(x, '(b t g) d -> b t g d', b=b, t=t, g=g)

        cls_space_tokens = repeat(self.space_token, '() g d -> b t g d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(g + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t g d -> (b t) g d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        # Crop-aware temporal decoding over the crop's phenological window.
        # Every sample in a batch is the same crop (the loader emits one
        # county-year-crop per step), so a single index selects both the window
        # and the head.
        crop = crop_idx.item() if crop_idx.numel() == 1 else int(crop_idx.view(-1)[0])
        crop_query = self.crop_query_bank[crop_idx].unsqueeze(0)
        valid_months = self.crop_growth_cycles[crop]
        x_crop = x[:, valid_months, :]

        cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b t d', b=b)
        x_crop = torch.cat((cls_temporal_tokens, x_crop), dim=1)

        x_crop = self.multicrop_trend_aware_temporal_transformer(x_crop, crop_query)
        x_crop = x_crop.mean(dim=1) if self.pool == 'mean' else x_crop[:, 0]
        return self.mlp_heads[crop](x_crop)
