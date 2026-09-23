import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, qkv_bias=True, drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("PVT dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, height, width):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        source = x
        if self.sr_ratio > 1:
            source = x.transpose(1, 2).reshape(b, c, height, width)
            source = self.sr(source).flatten(2).transpose(1, 2)
            source = self.norm(source)
        kv = self.kv(source).reshape(
            b, -1, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, mlp_ratio, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, sr_ratio, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, height, width):
        x = x + self.attn(self.norm1(x), height, width)
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, height, width


class PVTTinyEncoder(nn.Module):
    """MMST-ViT 官方 models_pvt.py 中 pvt_tiny 的等价实现。

    官方 pvt_tiny(pretrained=True) 只构造模型，不提供权重下载；因此这里
    使用官方结构随机初始化，供下游产量任务端到端训练。
    """

    def __init__(self, out_dim=384, image_size=224, in_chans=3, drop=0.0):
        super().__init__()
        self.embed_dims = [16, 32, 64, out_dim]
        self.stages = nn.ModuleList([
            PatchEmbed(in_chans, 16, 4),
            PatchEmbed(16, 32, 2),
            PatchEmbed(32, 64, 2),
            PatchEmbed(64, out_dim, 2),
        ])
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 56 * 56, 16)),
            nn.Parameter(torch.zeros(1, 28 * 28, 32)),
            nn.Parameter(torch.zeros(1, 14 * 14, 64)),
            nn.Parameter(torch.zeros(1, 1 + 7 * 7, out_dim)),
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_dim))
        self.blocks = nn.ModuleList([
            nn.ModuleList([Block(16, 1, 8, 2, drop)]),
            nn.ModuleList([Block(32, 2, 4, 2, drop)]),
            nn.ModuleList([Block(64, 4, 2, 2, drop)]),
            nn.ModuleList([Block(out_dim, 4, 1, 2, drop)]),
        ])
        self.norm = nn.LayerNorm(out_dim, eps=1e-6)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for pos in self.pos_embeds:
            nn.init.trunc_normal_(pos, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        for stage_idx, (patch, blocks, pos) in enumerate(
            zip(self.stages, self.blocks, self.pos_embeds)
        ):
            tokens, height, width = patch(x)
            if stage_idx == 3:
                cls = self.cls_token.expand(tokens.size(0), -1, -1)
                tokens = torch.cat([cls, tokens], dim=1)
                tokens = tokens + pos
            else:
                tokens = tokens + pos
            for block in blocks:
                if stage_idx == 3:
                    # CLS 必须和 patch 一起参与 self-attention，才能读取图像内容。
                    tokens = block(tokens, height, width)
                else:
                    tokens = block(tokens, height, width)
            if stage_idx < 3:
                x = tokens.transpose(1, 2).reshape(
                    tokens.size(0), self.embed_dims[stage_idx], height, width
                )
        return self.norm(tokens[:, 0])
