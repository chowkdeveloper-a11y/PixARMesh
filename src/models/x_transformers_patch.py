import torch
from torch import Tensor
from typing import Optional
from functools import partial
from x_transformers.x_transformers import (
    Decoder as XTransformerDecoder,
    LayerIntermediates,
    exists,
    random,
    dropout_seq,
)


class Decoder(XTransformerDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gradient_checkpointing = False

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        self_attn_kv_mask=None,
        mems=None,
        seq_start_pos: Optional[Tensor] = None,
        cache: Optional[LayerIntermediates] = None,
        cache_age=1,
        return_hiddens=False,
        rotary_pos_emb=None,
    ):
        use_gradient_checkpointing = self.gradient_checkpointing and self.training
        if use_gradient_checkpointing:
            if cache is not None:
                cache = None

        assert not (
            self.cross_attend ^ exists(context)
        ), "context must be passed in if cross_attend is set to True"

        # initialize accums

        hiddens = []
        layer_hiddens = []
        intermediates = []

        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        # handle left padded sequences

        if exists(seq_start_pos):
            seq_arange = torch.arange(x.shape[-2], device=x.device, dtype=torch.long)
            left_pad_mask = seq_arange >= seq_start_pos[..., None]

            if exists(self_attn_kv_mask):
                self_attn_kv_mask = self_attn_kv_mask & left_pad_mask
            else:
                self_attn_kv_mask = left_pad_mask

        # rotary positions

        if not exists(rotary_pos_emb) and exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(
                list(map(lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1], mems))
            )
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(
                max_rotary_emb_length
            )

        # assume cached key / values

        attn_cache = []

        if exists(cache):
            assert (
                not self.training
                and self.causal
                and not any([*map(exists, (mask, attn_mask))])
            )

            if cache_age > 0:
                x = x[:, -cache_age:]  # for spec decoding, may be greater than 1

            attn_cache = cache.attn_intermediates
            iter_attn_cache = iter(attn_cache)
        else:
            iter_attn_cache = None

        # outer residual - for resiDual paper

        outer_residual = x * self.resi_dual_scale

        # get layers to be executed

        layer_variables = (self.layer_types, self.layers, self.layer_dropouts)

        layer_variables = tuple(
            tuple(layer_variable[i] for i in self.layers_execute_order)
            for layer_variable in layer_variables
        )

        # go through the attention and feedforward layers

        for ind, (layer_type, (norm, block, residual_fn), layer_dropout) in enumerate(
            zip(*layer_variables)
        ):
            is_last = ind == (len(self.layers) - 1)

            if self.training and layer_dropout > 0.0 and random() < layer_dropout:
                continue

            if layer_type == "a":
                if return_hiddens:
                    hiddens.append(x)
                layer_mem = mems.pop(0) if mems else None

            if layer_type == "c":
                if self.training and self.cross_attn_tokens_dropout > 0.0:
                    context, context_mask = dropout_seq(
                        context, context_mask, self.cross_attn_tokens_dropout
                    )

            inner_residual = x

            if return_hiddens:
                layer_hiddens.append(x)

            pre_norm, post_branch_norm, post_main_norm = norm

            if exists(pre_norm):
                x = pre_norm(x)

            if use_gradient_checkpointing:
                block = partial(self._gradient_checkpointing_func, block.__call__)

            if layer_type == "a":
                out, inter = block(
                    x,
                    mask=mask,
                    context_mask=self_attn_kv_mask,
                    attn_mask=attn_mask,
                    rel_pos=self.rel_pos,
                    rotary_pos_emb=rotary_pos_emb,
                    prev_attn=prev_attn,
                    cache=(
                        next(iter_attn_cache, None)
                        if iter_attn_cache is not None
                        else None
                    ),
                    mem=layer_mem,
                    return_intermediates=True,
                )
            elif layer_type == "c":
                out, inter = block(
                    x,
                    context=context,
                    mask=mask,
                    context_mask=context_mask,
                    prev_attn=prev_cross_attn,
                    cache=(
                        next(iter_attn_cache, None)
                        if iter_attn_cache is not None
                        else None
                    ),
                    return_intermediates=True,
                )
            elif layer_type == "f":
                out = block(x)

            if self.resi_dual:
                outer_residual = outer_residual + out * self.resi_dual_scale

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, inner_residual)

            if layer_type in ("a", "c") and return_hiddens:
                intermediates.append(inter)

            if layer_type == "a" and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == "c" and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            layer_hiddens.append(x)

        if self.resi_dual:
            x = x + self.final_norm(outer_residual)
        else:
            x = self.final_norm(x)

        if not return_hiddens:
            return x

        intermediates = LayerIntermediates(
            hiddens=hiddens,
            last_hidden=x,
            attn_intermediates=intermediates,
            layer_hiddens=layer_hiddens,
        )

        return x, intermediates
