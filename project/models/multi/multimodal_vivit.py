import torch
from torch import nn
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from typing import Dict


def exists(val):
    return val is not None


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViViTMultiModal(nn.Module):
    def __init__(
        self,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels_dict,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        pool='cls'
    ):
        super().__init__()

        self.channels_dict = channels_dict
        total_channels = sum(channels_dict.values())

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0
        assert frames % frame_patch_size == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = frames // frame_patch_size
        patch_dim = total_channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)',
                      p1=patch_height, p2=patch_width, pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) if pool == 'cls' else None
        self.global_average_pool = pool == 'mean'

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        x = torch.cat([inputs[k] for k in self.channels_dict.keys()], dim=1)
        x = self.to_patch_embedding(x)
        b, f, n, d = x.shape

        x += self.pos_embedding[:, :f, :n]

        if self.cls_token is not None:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b f 1 d', b=b, f=f)
            x = torch.cat((cls_tokens, x), dim=2)

        x = self.dropout(x)
        x = rearrange(x, 'b f n d -> (b f) n d')
        x = self.transformer(x)
        x = rearrange(x, '(b f) n d -> b f n d', b=b)

        x = x[:, :, 0] if self.cls_token is not None else reduce(x, 'b f n d -> b f d', 'mean')
        x = reduce(x, 'b f d -> b d', 'mean')

        return self.mlp_head(x)


if __name__ == "__main__":
    channels_dict = {"rgb": 3, "flow": 2, "mask": 1, "kpt": 1}

    model = ViViTMultiModal(
        image_size=128,
        image_patch_size=16,
        frames=16,
        frame_patch_size=2,
        num_classes=3,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        channels_dict=channels_dict,
        pool="cls"
    )

    inputs = {
        "rgb": torch.randn(2, 3, 16, 128, 128),
        "flow": torch.randn(2, 2, 16, 128, 128),
        "mask": torch.randn(2, 1, 16, 128, 128),
        "kpt": torch.randn(2, 1, 16, 128, 128),
    }

    out = model(inputs)
    print(out.shape)  # Expected: (2, 10)