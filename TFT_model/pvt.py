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
    """PVT-Tiny 编码器，与 MMST-ViT 官方 pvt_tiny 配置完全一致。

    embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
    mlp_ratios=[8, 8, 4, 4], sr_ratios=[8, 4, 2, 1], depths=[2, 2, 2, 2].
    """

    def __init__(self, out_dim=512, image_size=224, in_chans=3, drop=0.0):
        super().__init__()
        embed_dims = [64, 128, 320, out_dim]
        num_heads = [1, 2, 5, 8]
        mlp_ratios = [8, 8, 4, 4]
        sr_ratios = [8, 4, 2, 1]
        depths = [2, 2, 2, 2]
        patch_sizes = [4, 2, 2, 2]

        self.embed_dims = embed_dims

        self.stages = nn.ModuleList()
        for i in range(4):
            in_c = in_chans if i == 0 else embed_dims[i - 1]
            self.stages.append(PatchEmbed(in_c, embed_dims[i], patch_sizes[i]))

        num_patches = []
        h, w = image_size // patch_sizes[0], image_size // patch_sizes[0]
        num_patches.append(h * w)
        for i in range(1, 4):
            h, w = h // patch_sizes[i], w // patch_sizes[i]
            n = h * w
            if i == 3:
                n += 1  # cls_token slot
            num_patches.append(n)

        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_patches[i], embed_dims[i])) for i in range(4)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_dim))
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                Block(embed_dims[i], num_heads[i], sr_ratios[i], mlp_ratios[i], drop)
                for _ in range(depths[i])
            ]) for i in range(4)
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

    def forward_tokens(self, x, include_cls=True):
        for stage_idx, (patch, blocks, pos) in enumerate(
            zip(self.stages, self.blocks, self.pos_embeds)
        ):
            tokens, height, width = patch(x)
            if stage_idx == 3:
                cls = self.cls_token.expand(tokens.size(0), -1, -1)
                tokens = torch.cat([cls, tokens], dim=1)
            tokens = tokens + pos
            for block in blocks:
                tokens = block(tokens, height, width)
            if stage_idx < 3:
                x = tokens.transpose(1, 2).reshape(
                    tokens.size(0), self.embed_dims[stage_idx], height, width
                )
        tokens = self.norm(tokens)
        return tokens if include_cls else tokens[:, 1:]

    def forward(self, x):
        return self.forward_tokens(x, include_cls=True)[:, 0]