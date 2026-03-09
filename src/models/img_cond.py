from typing import List
import torch
import torch.nn as nn
from transformers import AutoBackbone
from transformers.models.dpt.modeling_dpt import DPTNeck, DPTConfig


class ImageConditionEncoder(nn.Module):
    def __init__(self, model_name: str, out_features: List[str] = None, **kwargs):
        super().__init__()
        self.backbone = AutoBackbone.from_pretrained(
            model_name,
            out_features=out_features,
            **kwargs,
        )
        self.out_features = out_features
        self.backbone.requires_grad_(False)

    @property
    def output_dim(self):
        return self.backbone.config.hidden_size

    @property
    def num_register_tokens(self):
        return getattr(self.backbone, "num_register_tokens", 0)

    @torch.no_grad()
    def forward(self, pixel_values):
        features = self.backbone(pixel_values=pixel_values).feature_maps
        if self.out_features is None:
            # Last layer only
            features = features[0]
        return features

    def state_dict(*args, **kwargs):
        return {}

    def load_state_dict(*args, **kwargs):
        return


class HighResImageConditionEncoder(nn.Module):
    def __init__(
        self, model_name: str, out_features: List[str], hidden_size: int = 256
    ):
        super().__init__()
        self.base_encoder = ImageConditionEncoder(
            model_name, out_features, reshape_hidden_states=False
        )
        hidden_size = self.base_encoder.output_dim
        factor = 0.5
        neck_hidden_sizes = [
            int(hidden_size / (2**i)) for i in range(len(out_features))
        ]
        reassemble_factors = [factor * (2**i) for i in range(len(out_features))]
        reassemble_factors = [int(f) if f >= 1 else f for f in reassemble_factors]
        dpt_config = DPTConfig(
            backbone_config=self.base_encoder.backbone.config,
            is_hybrid=False,
            neck_hidden_sizes=neck_hidden_sizes[::-1],
            fusion_hidden_size=hidden_size,
            reassemble_factors=reassemble_factors[::-1],
            readout_type="project",
            hidden_act="gelu",
            use_bias_in_fusion_residual=False,
            use_batch_norm_in_fusion_residual=False,
            neck_ignore_stages=[],
        )
        self.config = dpt_config
        self.neck = DPTNeck(dpt_config)
        self.num_register_tokens = self.base_encoder.num_register_tokens
        self.hidden_size = hidden_size

        self.neck.apply(self._init_weights)

    @property
    def output_dim(self):
        return self.hidden_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, pixel_values):
        _, _, height, width = pixel_values.shape
        patch_size = self.config.backbone_config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size
        features = self.base_encoder(pixel_values=pixel_values)
        if self.num_register_tokens > 0:
            out_features = []
            for f in features:
                cls_token = f[:, :1]
                patch_embed = f[:, self.num_register_tokens + 1 :]
                out_features.append(torch.cat([cls_token, patch_embed], dim=1))
        else:
            out_features = features
        features = self.neck(out_features, patch_height, patch_width)
        return features[-1]
