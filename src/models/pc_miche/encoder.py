import torch
import torch.nn as nn
import math
from einops import repeat
from transformers import PreTrainedModel, PretrainedConfig
from .embedding import FourierEmbedder
from .utils import DiagonalGaussianDistribution
from .attention import ResidualCrossAttentionBlock, Transformer


class CrossAttentionEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_latents: int,
        fourier_embedder,
        point_feats: int,
        width: int,
        heads: int,
        layers: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        flash: bool = False,
        use_ln_post: bool = False,
        with_extra_feat: bool = False,
        extra_feat_dim: int = 0,
    ):

        super().__init__()

        self.num_latents = num_latents

        self.query = nn.Parameter(torch.randn((num_latents, width)) * 0.02)

        self.fourier_embedder = fourier_embedder
        self.input_proj = nn.Linear(
            self.fourier_embedder.out_dim + point_feats,
            width,
        )
        if with_extra_feat:
            self.extra_feat_proj = nn.Linear(
                extra_feat_dim,
                width,
            )
            # # Zero init
            # self.extra_feat_proj.weight.data.zero_()
            # self.extra_feat_proj.bias.data.zero_()

        self.with_extra_feat = with_extra_feat
        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )

        self.self_attn = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width)
        else:
            self.ln_post = None

    def forward(self, pc, feats, extra_feat=None, extra_feat_mask=None):
        bs = pc.shape[0]

        data = self.fourier_embedder(pc)
        if feats is not None:
            data = torch.cat([data, feats], dim=-1)
        data = self.input_proj(data)

        if self.with_extra_feat and extra_feat is not None:
            extra_feat = self.extra_feat_proj(extra_feat)
            if extra_feat_mask is not None:
                extra_feat.masked_fill_(extra_feat_mask[:, None, None], 0.0)
            data = data + extra_feat

        query = repeat(self.query, "m c -> b m c", b=bs)
        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc


class PointCloudEncoderConfig(PretrainedConfig):
    model_type = "pointcloud-encoder"

    def __init__(
        self,
        num_latents=256,
        embed_dim=64,
        point_feats=3,
        num_freqs=8,
        include_pi=False,
        heads=12,
        width=768,
        num_encoder_layers=8,
        num_decoder_layers=16,
        use_ln_post=True,
        init_scale=0.25,
        qkv_bias=False,
        flash=False,
        no_decoder=False,
        with_extra_feat=False,
        extra_feat_dim=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_latents = num_latents
        self.embed_dim = embed_dim
        self.point_feats = point_feats
        self.num_freqs = num_freqs
        self.include_pi = include_pi
        self.heads = heads
        self.width = width
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.use_ln_post = use_ln_post
        self.init_scale = init_scale
        self.qkv_bias = qkv_bias
        self.flash = flash
        self.no_decoder = no_decoder
        self.with_extra_feat = with_extra_feat
        self.extra_feat_dim = extra_feat_dim


class PointCloudEncoder(PreTrainedModel):
    config_class = PointCloudEncoderConfig

    def __init__(self, config):
        super().__init__(config)

        self.num_latents = config.num_latents + 1
        self.fourier_embedder = FourierEmbedder(
            num_freqs=config.num_freqs, include_pi=config.include_pi
        )

        init_scale = config.init_scale * math.sqrt(1.0 / config.width)
        self.encoder = CrossAttentionEncoder(
            fourier_embedder=self.fourier_embedder,
            num_latents=self.num_latents,
            point_feats=config.point_feats,
            width=config.width,
            heads=config.heads,
            layers=config.num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=config.qkv_bias,
            flash=config.flash,
            use_ln_post=config.use_ln_post,
            with_extra_feat=config.with_extra_feat,
            extra_feat_dim=config.extra_feat_dim,
        )

        self.no_decoder = config.no_decoder

        if not config.no_decoder:
            self.embed_dim = config.embed_dim
            if self.embed_dim > 0:
                # VAE embed
                self.pre_kl = nn.Linear(config.width, self.embed_dim * 2)
                self.post_kl = nn.Linear(self.embed_dim, config.width)
                self.latent_shape = (self.num_latents, self.embed_dim)
            else:
                self.latent_shape = (self.num_latents, config.width)

            self.transformer = Transformer(
                n_ctx=self.num_latents,
                width=config.width,
                layers=config.num_decoder_layers,
                heads=config.heads,
                init_scale=init_scale,
                qkv_bias=config.qkv_bias,
                flash=config.flash,
            )

        self.post_init()

    def forward(self, pc_normal, extra_feat=None, extra_feat_mask=None):
        pc = pc_normal[..., :3]
        feats = pc_normal[..., 3:]

        x, _ = self.encoder(
            pc, feats, extra_feat=extra_feat, extra_feat_mask=extra_feat_mask
        )
        shape_embed = x[:, 0]
        latents = x[:, 1:]
        shape_embed = shape_embed.unsqueeze(1)

        if self.no_decoder:
            return shape_embed, latents, None

        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            kl_embed = posterior.mode()
        else:
            kl_embed = latents

        kl_embed = self.post_kl(kl_embed)
        decoded_latents = self.transformer(kl_embed)

        return shape_embed, latents, decoded_latents

    @property
    def output_dim(self):
        return self.config.width
