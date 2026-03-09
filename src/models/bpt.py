import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from einops import rearrange, repeat, pack
from transformers import PretrainedConfig, PreTrainedModel
from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    top_k,
)
from tqdm import tqdm
from src.data.typing import TokenType
from .x_transformers_patch import Decoder
from .cond import MicheProjectorBPT, ContextAggregator
from .loss import causal_lm_loss_with_token_types, CustomCausalLMOutputWithTokenTypes


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


class BPTConfig(PretrainedConfig):
    model_type = "bpt"

    def __init__(
        self,
        vocab_size=5121,
        hidden_size=1024,
        num_hidden_layers=24,
        ffn_dim=4096,
        max_position_embeddings=10_000,
        do_layer_norm_before=True,
        dropout=0.0,
        attention_dropout=0.0,
        num_attention_heads=16,
        layerdrop=0.0,
        init_std=0.02,
        use_cache=True,
        cross_attn_num_mem_kv=4,
        pad_token_id=-1,
        bos_token_id=2,
        eos_token_id=2,
        block_size=8,
        offset_size=16,
        ff_glu=True,
        num_mem_kv=4,
        indicator_token_id=-50,
        sep_token_id=None,
        with_ctx_pc=False,
        img_cond_drop_prob=0.0,
        loss_layout_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )
        self.tie_word_embeddings = False
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.ffn_dim = ffn_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.init_std = init_std
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.do_layer_norm_before = do_layer_norm_before
        self.cross_attn_num_mem_kv = cross_attn_num_mem_kv
        self.block_size = block_size
        self.offset_size = offset_size
        self.ff_glu = ff_glu
        self.num_mem_kv = num_mem_kv
        self.indicator_token_id = indicator_token_id
        self.with_ctx_pc = with_ctx_pc
        self.img_cond_drop_prob = img_cond_drop_prob
        self.loss_layout_scale = loss_layout_scale


class BPTModel(PreTrainedModel):
    config_class = BPTConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(
        self,
        config: BPTConfig,
        cond_encoder=None,
        cond_encoder_img=None,
        is_scene=False,
    ):
        super().__init__(config)

        self.sp_block_embed = nn.Parameter(torch.randn(1, config.hidden_size))
        self.sos_token = nn.Parameter(torch.randn(config.hidden_size))
        self.eos_token_id = config.eos_token_id
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)

        self.block_size = config.block_size
        self.offset_size = config.offset_size
        self.abs_pos_emb = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.max_seq_len = config.max_position_embeddings

        self.block_embed = nn.Parameter(torch.randn(1, config.hidden_size))
        self.offset_embed = nn.Parameter(torch.randn(1, config.hidden_size))

        cross_attn = cond_encoder is not None

        if cond_encoder is not None:
            self.projector = MicheProjectorBPT(
                cond_encoder.output_dim,
                config.hidden_size,
            )
            self.projector.apply(self._init_weights)

        self.cond_encoder = cond_encoder
        self.cond_encoder_img = cond_encoder_img

        attn_dim_head = config.hidden_size // config.num_attention_heads
        flash_attn = config._attn_implementation in (
            "flash_attention_2",
            "flash_attention_3",
        )

        self.decoder = Decoder(
            dim=config.hidden_size,
            depth=config.num_hidden_layers,
            dim_head=attn_dim_head,
            heads=config.num_attention_heads,
            attn_flash=flash_attn,
            attn_dropout=config.attention_dropout,
            ff_dropout=config.dropout,
            cross_attend=cross_attn,
            cross_attn_dim_context=config.hidden_size,
            cross_attn_num_mem_kv=config.cross_attn_num_mem_kv,
            ff_glu=config.ff_glu,
            num_mem_kv=config.num_mem_kv,
            attn_qk_norm=True,
        )

        self.to_logits = nn.Linear(config.hidden_size, config.vocab_size)
        self.pad_token_id = config.pad_token_id

        self.ctx_aggregator = None
        if self.config.with_ctx_pc:
            self.ctx_aggregator = ContextAggregator(
                ctx_dim=cond_encoder.output_dim, num_heads=8
            )
            self.ctx_aggregator.apply(self._init_weights)

        self.post_init()

    def get_input_embeddings(self):
        return self.token_embed

    def set_input_embeddings(self, value):
        self.token_embed = value

    def get_output_embeddings(self):
        return self.to_logits

    def set_output_embeddings(self, new_embeddings):
        self.to_logits = new_embeddings

    def set_decoder(self, decoder):
        raise NotImplementedError

    def get_decoder(self):
        return self.decoder

    def get_inputs_with_cond(
        self,
        input_ids=None,
        cond_pcs=None,
        cond_pcs_2d=None,
        ctx_pcs=None,
        ctx_pcs_2d=None,
        pixel_values=None,
    ):
        extra_feat_mask = None
        sampled_feats = None
        ctx_img_feats = None
        if pixel_values is not None and self.cond_encoder_img is not None:
            img_feats = self.cond_encoder_img(pixel_values=pixel_values)
            sampled_feats = F.grid_sample(
                img_feats,
                cond_pcs_2d.unsqueeze(1),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled_feats = sampled_feats.squeeze(2).permute(0, 2, 1)  # (B, N, C)
            if ctx_pcs_2d is not None:
                ctx_img_feats = F.grid_sample(
                    img_feats,
                    ctx_pcs_2d.unsqueeze(1),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                ctx_img_feats = ctx_img_feats.squeeze(2).permute(0, 2, 1)  # (B, N, C)
            if self.training:
                extra_feat_mask = (
                    torch.rand(cond_pcs.size(0)) < self.config.img_cond_drop_prob
                )
                extra_feat_mask = extra_feat_mask.to(cond_pcs.device)
        shape_embed, latents, _dec_latents = self.cond_encoder(
            cond_pcs, extra_feat=sampled_feats, extra_feat_mask=extra_feat_mask
        )
        if ctx_pcs is not None and self.ctx_aggregator is not None:
            ctx_latents = self.cond_encoder(
                ctx_pcs, extra_feat=ctx_img_feats, extra_feat_mask=extra_feat_mask
            )[1]
            latents = self.ctx_aggregator(latents, ctx_latents)
        cond_embeds = self.projector((shape_embed, latents, _dec_latents))
        return cond_embeds

    @eval_decorator
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        cond_embeds: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        temperature=1.0,
        use_cache=True,
        max_new_tokens=None,
        tqdm_position=0,
        do_sample=True,
    ):

        if exists(inputs):
            inputs = rearrange(inputs, "b ... -> b (...)")
            assert inputs.shape[-1] <= self.max_seq_len

            batch_size = inputs.shape[0]

        batch_size = default(batch_size, 1)

        codes = default(
            inputs, torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
        )

        curr_length = codes.shape[-1]

        cache = None

        if max_new_tokens is None:
            max_seq_len = self.max_seq_len
        else:
            max_seq_len = curr_length + max_new_tokens

        # predict tokens auto-regressively
        for i in tqdm(
            range(curr_length, max_seq_len),
            position=tqdm_position,
            desc=f"Process: {tqdm_position}",
            dynamic_ncols=True,
            leave=False,
        ):
            output = self.forward(
                input_ids=codes,
                labels=None,
                use_cache=use_cache,
                cond_embeds=cond_embeds,
                cache=cache,
                return_dict=True,
            )

            logits = output.logits

            if use_cache:
                cache = output.past_key_values

            # sample code from logits
            logits = logits[:, -1]
            if do_sample:
                filtered_logits = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)
            else:
                sample = logits.argmax(dim=-1, keepdim=True)
            codes, _ = pack([codes, sample], "b *")

            # check for all rows to have [eos] to terminate
            is_eos_codes = codes == self.eos_token_id

            if is_eos_codes.any(dim=-1).all():
                break

        # mask out to padding anything after the first eos
        mask = is_eos_codes.float().cumsum(dim=-1) >= 1
        codes = codes.masked_fill(mask, self.pad_token_id)

        return codes

    def forward(
        self,
        input_ids=None,
        cond_pcs=None,
        cond_pcs_2d=None,
        ctx_pcs=None,
        ctx_pcs_2d=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        pixel_values=None,
        cache=None,
        cond_embeds=None,
        **kwargs,
    ):
        if position_ids is not None or cache_position is not None:
            raise NotImplementedError

        if cond_embeds is None and self.cond_encoder is not None:
            cond_embeds = self.get_inputs_with_cond(
                input_ids=input_ids,
                cond_pcs=cond_pcs,
                cond_pcs_2d=cond_pcs_2d,
                ctx_pcs=ctx_pcs,
                ctx_pcs_2d=ctx_pcs_2d,
                pixel_values=pixel_values,
            )

        logits, (loss, loss_layout, loss_object), intermediates_with_cache = (
            self._forward_impl(
                input_ids=input_ids,
                labels=labels,
                cond_embeds=cond_embeds,
                cache=cache,
                attention_mask=attention_mask,
                **kwargs,
            )
        )

        return CustomCausalLMOutputWithTokenTypes(
            loss=loss,
            loss_layout=loss_layout,
            loss_object=loss_object,
            logits=logits,
            past_key_values=intermediates_with_cache if use_cache else None,
            hidden_states=None,
            attentions=None,
        )

    @property
    def loss_function(self):
        return causal_lm_loss_with_token_types

    def _forward_impl(
        self,
        input_ids=None,
        labels=None,
        cache=None,
        cond_embeds=None,
        attention_mask=None,
        **kwargs,
    ):
        # handle conditions
        attn_context_kwargs = dict(
            context=cond_embeds,
            context_mask=None,
        )

        # take care of codes that may be flattened
        if input_ids.ndim > 2:
            input_ids = rearrange(input_ids, "b ... -> b (...)")

        # prepare mask for position embedding of block and offset tokens
        block_mask = (0 <= input_ids) & (input_ids < self.block_size**3)
        offset_mask = (self.block_size**3 <= input_ids) & (
            input_ids < self.block_size**3 + self.offset_size**3
        )
        sp_block_mask = (self.block_size**3 + self.offset_size**3 <= input_ids) & (
            input_ids < self.block_size**3 + self.offset_size**3 + self.block_size**3
        )

        # get some variable
        batch, seq_len, device = *input_ids.shape, input_ids.device

        assert (
            seq_len <= self.max_seq_len
        ), f"received codes of length {seq_len} but needs to be less than {self.max_seq_len}"

        # token embed
        input_ids = input_ids.masked_fill(input_ids == self.pad_token_id, 0)
        input_embeds = self.token_embed(input_ids)

        # codebook embed + absolute positions
        seq_arange = torch.arange(input_embeds.shape[-2], device=device)

        # add positional embedding for block and offset token
        block_embed = repeat(self.block_embed, "1 d -> b n d", n=seq_len, b=batch)
        offset_embed = repeat(self.offset_embed, "1 d -> b n d", n=seq_len, b=batch)

        sp_block_embed = repeat(self.sp_block_embed, "1 d -> b n d", n=seq_len, b=batch)

        input_embeds = (
            input_embeds
            + self.abs_pos_emb(seq_arange)
            + block_embed * block_mask.unsqueeze(-1)
            + offset_embed * offset_mask.unsqueeze(-1)
            + sp_block_embed * sp_block_mask.unsqueeze(-1)
        )

        # auto prepend sos token
        sos = repeat(self.sos_token, "d -> b d", b=batch)
        input_embeds, _ = pack([sos, input_embeds], "b * d")

        # if attention_mask is not None:
        #     # prepend attention mask for sos token
        #     sos_mask = torch.ones(
        #         (attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype
        #     )
        #     attention_mask = torch.cat([sos_mask, attention_mask], dim=1)
        #     attention_mask = attention_mask.to(dtype=torch.bool)

        # attention
        attended, intermediates_with_cache = self.decoder(
            input_embeds,
            cache=cache,
            return_hiddens=True,
            # mask=attention_mask,
            **attn_context_kwargs,
        )

        # logits
        logits = self.to_logits(attended)

        loss = loss_layout = loss_object = None
        if labels is not None:
            extended_labels = torch.full(
                (labels.shape[0], labels.shape[1] + 1),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            # To account for the prepended SOS token
            extended_labels[:, 1:] = labels
            extended_labels = extended_labels.to(logits.device)

            if "token_type_ids" in kwargs:
                token_type_ids = kwargs["token_type_ids"]
                extended_token_type_ids = torch.full(
                    (token_type_ids.shape[0], token_type_ids.shape[1] + 1),
                    TokenType.PADDING,
                    dtype=token_type_ids.dtype,
                    device=token_type_ids.device,
                )
                extended_token_type_ids[:, 1:] = token_type_ids
                kwargs["token_type_ids"] = extended_token_type_ids.to(logits.device)

            loss, loss_layout, loss_object = self.loss_function(
                logits,
                extended_labels,
                vocab_size=self.config.vocab_size,
                loss_layout_scale=self.config.loss_layout_scale,
                **kwargs,
            )

        return logits, (loss, loss_layout, loss_object), intermediates_with_cache
