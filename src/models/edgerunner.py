import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import OPTForCausalLM, OPTConfig, OPTModel, PreTrainedModel
from transformers.models.opt.modeling_opt import OPTDecoder
from .cond import EdgeRunnerProjector, ContextAggregator
from .embed import CoordEmbed
from .loss import causal_lm_loss_with_token_types, CustomCausalLMOutputWithTokenTypes


class OPTLearnedPositionalEmbeddingNoOffset(nn.Embedding):
    def forward(
        self,
        attention_mask: torch.LongTensor,
        past_key_values_length: int = 0,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        if position_ids is None:
            position_ids = torch.cumsum(attention_mask, dim=1)
            position_ids = (position_ids * attention_mask - 1).long()
            position_ids = position_ids[:, past_key_values_length:]
        position_ids.clip_(0)
        return super().forward(position_ids)


class ShapeOPTConfig(OPTConfig):
    model_type = "shape-opt"

    def __init__(
        self,
        indicator_token_id=-50,
        obj_pc_token_id=-49,
        pc_token_id=-48,
        with_ctx_pc=False,
        img_cond_drop_prob=0.0,
        loss_layout_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.indicator_token_id = indicator_token_id
        self.obj_pc_token_id = obj_pc_token_id
        self.pc_token_id = pc_token_id
        self.with_ctx_pc = with_ctx_pc
        self.tie_word_embeddings = False
        self.img_cond_drop_prob = img_cond_drop_prob
        self.loss_layout_scale = loss_layout_scale


class ShapeOPTDecoder(OPTDecoder):
    config_class = ShapeOPTConfig

    def __init__(self, config: ShapeOPTConfig):
        super().__init__(config)
        self.embed_positions = OPTLearnedPositionalEmbeddingNoOffset(
            config.max_position_embeddings, config.hidden_size
        )
        self.post_init()


class ShapeOPTModel(OPTModel):
    def __init__(self, config: ShapeOPTConfig):
        super().__init__(config)
        self.decoder = ShapeOPTDecoder(config)
        self.post_init()


class ShapeOPT(OPTForCausalLM):
    _tied_weights_keys = []
    config_class = ShapeOPTConfig

    def __init__(
        self,
        config: ShapeOPTConfig,
        cond_encoder=None,
        cond_encoder_img=None,
        is_scene=False,
    ):
        super().__init__(config)
        self.model = ShapeOPTModel(config)
        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False
        )
        self.embed_num_face = nn.Embedding(10, config.word_embed_proj_dim)

        self.is_scene = is_scene
        if self.is_scene:
            self.indicator_embed = CoordEmbed(
                num_points=3,
                dim=config.word_embed_proj_dim,
                freq_embed_dim=48,
            )
        else:
            self.indicator_embed = None

        self.post_init()

        if cond_encoder is not None:
            self.projector = EdgeRunnerProjector(
                cond_encoder.output_dim,
                config.word_embed_proj_dim,
            )
            self.projector.apply(self._init_weights)
        self.cond_encoder = cond_encoder
        self.cond_encoder_img = cond_encoder_img

        self.ctx_aggregator = None
        if self.config.with_ctx_pc:
            self.ctx_aggregator = ContextAggregator(
                cond_encoder.output_dim, num_heads=8
            )
            self.ctx_aggregator.apply(self._init_weights)

    def _init_weights(self, module):
        return PreTrainedModel._init_weights(self, module)

    def get_inputs_with_cond(
        self,
        input_ids,
        cond_pcs=None,
        cond_pcs_2d=None,
        ctx_pcs=None,
        ctx_pcs_2d=None,
        cond_num_faces=None,
        obj_indices=None,
        obj_bboxes=None,
        obj_cond_pcs=None,
        pixel_values=None,
    ):
        input_ids = input_ids.clone()
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

        cond_token_mask = input_ids == self.config.pc_token_id
        indicator_mask = input_ids == self.config.indicator_token_id
        obj_cond_token_mask = input_ids == self.config.obj_pc_token_id
        input_ids[indicator_mask | obj_cond_token_mask | cond_token_mask] = (
            self.config.pad_token_id
        )
        inputs_embeds = self.model.decoder.embed_tokens(input_ids)

        if obj_indices is not None and self.is_scene:
            valid_inds = obj_indices >= 0
            if obj_bboxes is not None:
                indicator_embeds = self.indicator_embed(obj_bboxes)
                inputs_embeds.masked_scatter_(
                    indicator_mask.unsqueeze(-1), indicator_embeds[valid_inds]
                )
            if obj_cond_pcs is not None and len(obj_cond_pcs) > 0:
                obj_conds = self.cond_encoder(obj_cond_pcs)
                obj_cond_embeds = self.projector(obj_conds)
                inputs_embeds.masked_scatter_(
                    obj_cond_token_mask.unsqueeze(-1), obj_cond_embeds.flatten(0, 1)
                )

        all_cond_embeds = []
        if cond_pcs is not None:
            conds = self.cond_encoder(
                cond_pcs, extra_feat=sampled_feats, extra_feat_mask=extra_feat_mask
            )
            if ctx_pcs is not None and self.ctx_aggregator is not None:
                ctx_conds = self.cond_encoder(
                    ctx_pcs, extra_feat=ctx_img_feats, extra_feat_mask=extra_feat_mask
                )
                conds = self.ctx_aggregator(conds, ctx_conds)
            cond_embeds = self.projector(conds)
            all_cond_embeds.append(cond_embeds)
        if cond_num_faces is None:
            cond_num_faces = torch.zeros(
                (inputs_embeds.size(0), 1),
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        num_face_embeds = self.embed_num_face(cond_num_faces)
        all_cond_embeds.append(num_face_embeds)
        if len(all_cond_embeds) > 0:
            all_cond_embeds = torch.cat(all_cond_embeds, dim=1).flatten(0, 1)
            inputs_embeds.masked_scatter_(
                cond_token_mask.unsqueeze(-1), all_cond_embeds
            )
        return inputs_embeds

    @property
    def loss_function(self):
        return causal_lm_loss_with_token_types

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cond_pcs=None,
        cond_pcs_2d=None,
        ctx_pcs=None,
        ctx_pcs_2d=None,
        cond_num_faces=None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        obj_indices=None,
        obj_bboxes=None,
        obj_cond_pcs=None,
        pixel_values=None,
        **kwargs,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.cond_encoder is not None and cond_pcs is not None:
            inputs_embeds = self.get_inputs_with_cond(
                input_ids,
                cond_pcs=cond_pcs,
                cond_pcs_2d=cond_pcs_2d,
                ctx_pcs=ctx_pcs,
                ctx_pcs_2d=ctx_pcs_2d,
                cond_num_faces=cond_num_faces,
                obj_indices=obj_indices,
                obj_bboxes=obj_bboxes,
                obj_cond_pcs=obj_cond_pcs,
                pixel_values=pixel_values,
            )
            input_ids = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = loss_layout = loss_object = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss, loss_layout, loss_object = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                loss_layout_scale=self.config.loss_layout_scale,
                **kwargs,
            )

        return CustomCausalLMOutputWithTokenTypes(
            loss=loss,
            loss_layout=loss_layout,
            loss_object=loss_object,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
