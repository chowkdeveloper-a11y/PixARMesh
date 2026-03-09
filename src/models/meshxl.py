import torch
from typing import Optional
from transformers import OPTForCausalLM, OPTConfig
from transformers.models.opt.modeling_opt import CausalLMOutputWithPast
from .cond import MicheProjector


class MeshOPTConfig(OPTConfig):
    model_type = "mesh-opt"

    def __init__(
        self,
        pc_token_id=-48,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pc_token_id = pc_token_id


class MeshOPT(OPTForCausalLM):
    config_class = MeshOPTConfig

    def __init__(
        self, config: MeshOPTConfig, cond_encoder=None, is_scene=False, **kwargs
    ):
        super().__init__(config)

        self.is_scene = is_scene
        if is_scene:
            raise NotImplementedError

        if cond_encoder is not None:
            self.projector = MicheProjector(
                cond_encoder.output_dim,
                config.word_embed_proj_dim,
            )
            self.projector.apply(self._init_weights)
        self.cond_encoder = cond_encoder
        self.post_init()

    def get_inputs_with_cond(self, input_ids, cond_pcs=None):
        input_ids = input_ids.clone()
        cond_token_mask = input_ids == self.config.pc_token_id
        input_ids[cond_token_mask] = self.config.pad_token_id
        inputs_embeds = self.model.decoder.embed_tokens(input_ids)
        if cond_pcs is not None:
            conds = self.cond_encoder(cond_pcs)
            cond_embeds = self.projector(conds)
            inputs_embeds[cond_token_mask] = cond_embeds.flatten(0, 1).to(
                inputs_embeds.dtype
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cond_pcs=None,
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
        **kwargs,
    ):
        if (
            obj_indices is not None
            or obj_bboxes is not None
            or obj_cond_pcs is not None
        ):
            raise NotImplementedError

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
            inputs_embeds = self.get_inputs_with_cond(input_ids, cond_pcs)
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

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
