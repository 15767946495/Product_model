import torch
from torch import nn
from einops import rearrange, repeat

from attention import SpatialTransformer, TemporalTransformer

from models_pvt_simclr import PVTSimCLR


def masked_grid_pool(features, grid_mask):
    """features (B,G,D)、grid_mask (B,G) -> (B,D)；padding 网格不计入平均。"""
    weights = grid_mask.to(features.dtype).unsqueeze(-1)
    return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


class MMST_ViT(nn.Module):
    def __init__(self, out_dim=1, num_grid=80, num_short_term_seq=6, num_long_term_seq=12, num_year=4,
                 pvt_backbone=None, context_dim=9, dim=192, batch_size=64, depth=4, heads=3, pool='cls', dim_head=64,
                 dropout=0., emb_dropout=0., scale_dim=4, ):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.batch_size = batch_size
        self.num_grid = num_grid
        self.pvt_backbone = pvt_backbone

        self.proj_context = nn.Linear(num_year * num_long_term_seq * context_dim, num_short_term_seq * dim)
        # self.proj_context = nn.Linear(num_year * num_long_term_seq * context_dim, dim)

        # +1 为空间 cls token；80 网格位置编码与旧 64 网格 checkpoint 不兼容，微调时重新初始化
        self.pos_embedding = nn.Parameter(torch.randn(1, num_short_term_seq, num_grid + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = SpatialTransformer(dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = TemporalTransformer(dim, depth, heads, dim_head, mult=scale_dim, dropout=dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.norm1 = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )

    def forward_features(self, x, ys, grid_mask=None):
        b, t, g = x.shape[:3]
        x = rearrange(x, 'b t g c h w -> (b t g) c h w')
        ys = rearrange(ys, 'b t g n d -> (b t g) n d')
        if grid_mask is not None:
            # 只对真实网格运行 PVT，padding 网格特征置零
            valid = repeat(grid_mask, 'b g -> (b t g)', t=t)
            out = self.forward_features(x[valid][:, None, None], ys[valid][:, None, None])
            features = out.new_zeros((valid.shape[0], out.shape[-1]))
            features[valid] = out
            return features

        # prevent the number of grids from being too large to cause out of memory
        B = x.shape[0]
        n = B // self.batch_size if B % self.batch_size == 0 else B // self.batch_size + 1

        x_hat = torch.empty(0).to(x.device)
        for i in range(n):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            x_tmp = x[start:end]
            ys_tmp = ys[start:end]

            x_hat_tmp = self.pvt_backbone(x_tmp, context=ys_tmp)
            x_hat = torch.cat([x_hat, x_hat_tmp], dim=0)

        return x_hat

    def forward(self, x, ys=None, yl=None, grid_mask=None):
        b, t, g, _, _, _ = x.shape
        if g > self.num_grid:
            raise ValueError(f"{g} grids exceeds MMST num_grid {self.num_grid}")
        if yl is None:
            raise ValueError("MMST_ViT requires long-term weather (yl); 2017--2020 samples have none")
        if grid_mask is not None and not grid_mask.any(dim=1).all():
            raise ValueError("every sample needs at least one valid grid")
        x = self.forward_features(x, ys, grid_mask)
        x = rearrange(x, '(b t g) d -> b t g d', b=b, t=t, g=g)

        cls_space_tokens = repeat(self.space_token, '() g d -> b t g d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(g + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t g d -> (b t) g d')
        space_mask = None
        if grid_mask is not None:
            # cls token 恒有效；padding 网格不作为 key 参与空间注意力
            space_mask = torch.cat([grid_mask.new_ones((b, 1)), grid_mask], dim=1)
            space_mask = repeat(space_mask, 'b g -> (b t) g', t=t)
        x = self.space_transformer(x, mask=space_mask)
        if self.pool == 'mean':
            mask = space_mask[:, 1:] if space_mask is not None else x.new_ones(x.shape[:2], dtype=torch.bool)[:, 1:]
            x = masked_grid_pool(x[:, 1:], mask)
        else:
            x = x[:, 0]
        x = rearrange(x, '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b t d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        # concatenate parameters in different months
        yl = rearrange(yl, 'b y m d -> b (y m d)')
        yl = self.proj_context(yl)
        yl = rearrange(yl, 'b (t d) -> b t d', t=t)
        # yl = repeat(yl, '() d -> b t d', b=b, t=t)

        yl = torch.cat((cls_temporal_tokens, yl), dim=1)
        yl = self.norm1(yl)

        x = self.temporal_transformer(x, yl)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)


if __name__ == "__main__":
    # x.shape = B, T, G, C, H, W
    x = torch.randn((1, 6, 10, 3, 224, 224))
    # ys.shape = B, T, G, N1, d
    ys = torch.randn((1, 6, 10, 28, 9))
    # yl.shape = B, T, N2, d
    yl = torch.randn((1, 4, 12, 9))

    pvt = PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9)
    model = MMST_ViT(out_dim=4, pvt_backbone=pvt, dim=512)

    # print(model)

    z = model(x, ys=ys, yl=yl)
    print(z)
    print(z.shape)
