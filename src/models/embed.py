import torch
import torch.nn as nn
import numpy as np


class CoordEmbed(nn.Module):
    def __init__(self, num_points, dim, freq_embed_dim=48):
        super().__init__()
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
        self.num_points = num_points
        self.register_buffer("basis", e)  # [3, 48]
        self.mlp = nn.Linear(num_points * (self.freq_embed_dim + 3), dim)
        self.ln = nn.LayerNorm(dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis.to(input.dtype))
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.embed(input, self.basis)  # B x N x C
        embed = torch.cat([embed, input], dim=2).to(input.dtype)
        embed = embed.flatten(1)  # B x (N * C)
        embed = self.mlp(embed)  # B x D
        embed = self.ln(embed)
        return embed
