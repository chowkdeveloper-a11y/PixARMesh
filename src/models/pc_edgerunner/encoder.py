import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
import numpy as np
from .attention import CrossAttention


class PointEmbed(nn.Module):
    def __init__(self, dim=512, freq_embed_dim=48):
        super().__init__()

        # frequency embedding
        assert freq_embed_dim % 6 == 0
        self.freq_embed_dim = freq_embed_dim
        e = torch.pow(2, torch.arange(self.freq_embed_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [
                        e,
                        torch.zeros(self.freq_embed_dim // 6),
                        torch.zeros(self.freq_embed_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.freq_embed_dim // 6),
                        e,
                        torch.zeros(self.freq_embed_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.freq_embed_dim // 6),
                        torch.zeros(self.freq_embed_dim // 6),
                        e,
                    ]
                ),
            ]
        )
        self.register_buffer("basis", e)  # [3, 48]

        self.mlp = nn.Linear(self.freq_embed_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis.to(input.dtype))
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.embed(input, self.basis)  # B x N x C
        embed = torch.cat([embed, input], dim=2).to(input.dtype)
        embed = self.mlp(embed)  # B x N x C
        return embed


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class ResCrossAttBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.att = CrossAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim)

    def forward(self, x, c):
        x = x + self.att(self.ln1(x), c)
        x = x + self.mlp(self.ln2(x))
        return x


class EdgeRunnerPointEncoderConfig(PretrainedConfig):
    model_type = "EdgeRunnerPointEncoder"

    def __init__(
        self,
        hidden_dim=1024,
        num_heads=16,
        latent_size=2048,
        latent_dim=64,
        with_extra_feat=False,
        extra_feat_dim=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.latent_size = latent_size
        self.latent_dim = latent_dim
        self.with_extra_feat = with_extra_feat
        self.extra_feat_dim = extra_feat_dim


class EdgeRunnerPointEncoder(PreTrainedModel):
    # Original name: PointEncoderEmbed
    config_class = EdgeRunnerPointEncoderConfig

    def __init__(self, config: EdgeRunnerPointEncoderConfig):
        super().__init__(config)

        self.latent_size = config.latent_size
        self.query_embed = nn.Parameter(
            torch.randn(1, config.latent_size, config.hidden_dim)
            / config.hidden_dim**0.5
        )

        self.point_embed = PointEmbed(dim=config.hidden_dim)
        if config.with_extra_feat:
            self.extra_feat_proj = nn.Linear(config.extra_feat_dim, config.hidden_dim)
            # # Zero init
            # self.extra_feat_proj.weight.data.zero_()
            # self.extra_feat_proj.bias.data.zero_()

        self.ln = nn.LayerNorm(config.hidden_dim)

        self.cross_att = ResCrossAttBlock(config.hidden_dim, config.num_heads)

        self.linear = nn.Linear(config.hidden_dim, config.latent_dim)

        self.post_init()

    def forward(self, x, extra_feat=None, extra_feat_mask=None):
        # x: [B, N, 3]
        # return: latent [B, L, D]
        B, N, C = x.shape
        # embed
        point_emb = self.point_embed(x)
        if self.config.with_extra_feat and extra_feat is not None:
            extra_feat = self.extra_feat_proj(extra_feat)
            if extra_feat_mask is not None:
                extra_feat.masked_fill_(extra_feat_mask[:, None, None], 0.0)
            point_emb = point_emb + extra_feat
        x = self.ln(point_emb)  # [B, N, D], condition (kv)
        # downsample x to q
        q = self.query_embed.repeat(B, 1, 1)  # query
        # att
        l = self.cross_att(q, x)
        # out
        l = self.linear(l)  # [B, L, D]
        return l

    @property
    def output_dim(self):
        return self.config.latent_dim
