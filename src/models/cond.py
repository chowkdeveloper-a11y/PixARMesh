import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class MicheProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_head_proj = nn.Linear(self.in_dim, self.out_dim)
        self.cond_proj = nn.Linear(self.in_dim * 2, self.out_dim)

    def forward(self, conds):
        shape_embed, latents, decoded_latents = conds
        feat_1 = self.cond_head_proj(shape_embed)
        feat_2 = self.cond_proj(torch.cat([latents, decoded_latents], dim=-1))
        return torch.cat([feat_1, feat_2], dim=1)


class MicheProjectorBPT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_head_proj = nn.Linear(self.in_dim, self.out_dim)
        self.cond_proj = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, conds):
        shape_embed, latents, _ = conds
        feat_1 = self.cond_head_proj(shape_embed)
        feat_2 = self.cond_proj(latents)
        return torch.cat([feat_1, feat_2], dim=1)


class EdgeRunnerProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_proj = nn.Linear(self.in_dim, self.out_dim)
        self.cond_norm = nn.LayerNorm(self.out_dim)

    def forward(self, conds):
        return self.cond_norm(self.cond_proj(conds))


class ConditionEncoder(nn.Module):
    def __init__(self, encoder, freeze=True):
        super().__init__()
        self.encoder = encoder
        # Freeze the encoder
        if freeze:
            self.encoder.requires_grad_(False)

        self.has_extra_feat_proj = False
        self.extra_feat_proj_keys = []
        for name, param in self.encoder.named_parameters():
            if "extra_feat_proj" in name:
                param.requires_grad = True
                self.has_extra_feat_proj = True
                self.extra_feat_proj_keys.append("encoder." + name)
        self.freeze = freeze

    @property
    def output_dim(self):
        return self.encoder.output_dim

    def forward(self, x, *args, **kwargs):
        if self.freeze and not self.has_extra_feat_proj:
            with torch.no_grad():
                return self.encoder(x, *args, **kwargs)
        else:
            return self.encoder(x, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        states = super().state_dict(*args, **kwargs)
        if self.freeze:
            filtered_states = {
                k: v for k, v in states.items() if k in self.extra_feat_proj_keys
            }
            return filtered_states
        return states

    def load_state_dict(self, *args, **kwargs):
        if not self.freeze or self.has_extra_feat_proj:
            return super().load_state_dict(*args, **kwargs)


class ContextAggregator(nn.Module):
    def __init__(self, ctx_dim, num_heads=8):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.cross_attn = nn.MultiheadAttention(
            ctx_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, query, context):
        # query: (B, Lq, D)
        # context: (B, Lc, D)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            attn_output, _ = self.cross_attn(query, context, context)
            query = query + attn_output
            return query
