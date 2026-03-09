import torch
from src.utils.config import ModelConfig
from .meshxl import MeshOPT, MeshOPTConfig
from .edgerunner import ShapeOPT, ShapeOPTConfig
from .bpt import BPTModel, BPTConfig
from .pc_miche.encoder import PointCloudEncoder
from .pc_edgerunner.encoder import EdgeRunnerPointEncoder
from .cond import ConditionEncoder
from .img_cond import ImageConditionEncoder, HighResImageConditionEncoder


def get_model(
    local_model_path, model_cfg: ModelConfig, cond_encoder=None, cond_encoder_img=None
):
    extra_args = {}
    if model_cfg is not None:
        extra_args["bos_token_id"] = model_cfg.bos_token_id
        extra_args["eos_token_id"] = model_cfg.eos_token_id
        extra_args["pad_token_id"] = model_cfg.pad_token_id
        extra_args["max_position_embeddings"] = model_cfg.max_position_embeddings
        extra_args["indicator_token_id"] = model_cfg.indicator_token_id
        extra_args["obj_pc_token_id"] = model_cfg.obj_pc_token_id
        extra_args["pc_token_id"] = model_cfg.pc_token_id
        extra_args["with_ctx_pc"] = model_cfg.with_ctx_pc
        extra_args["img_cond_drop_prob"] = model_cfg.img_cond_drop_prob
        extra_args["loss_layout_scale"] = model_cfg.loss_layout_scale
        if model_cfg.sep_token_id is not None:
            extra_args["sep_token_id"] = model_cfg.sep_token_id
        model_type = model_cfg.ar_model_type
    else:
        model_type = "meshxl"

    is_scene = "-scene" in model_type

    match model_type:
        case "meshxl":
            config_class = MeshOPTConfig
            model_class = MeshOPT
        case "edgerunner" | "edgerunner-scene":
            config_class = ShapeOPTConfig
            model_class = ShapeOPT
        case "bpt":
            config_class = BPTConfig
            model_class = BPTModel
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

    config = config_class.from_pretrained(local_model_path, **extra_args)
    model = model_class.from_pretrained(
        local_model_path,
        config=config,
        torch_dtype=torch.float32,
        attn_implementation="flash_attention_2",
        cond_encoder=cond_encoder,
        cond_encoder_img=cond_encoder_img,
        is_scene=is_scene,
        ignore_mismatched_sizes=True,
    )
    if config.vocab_size != model_cfg.vocab_size:
        model.resize_token_embeddings(model_cfg.vocab_size, pad_to_multiple_of=64)
    return model


def get_condition_encoder(
    local_model_path, model_cfg: ModelConfig, cond_encoder_img=None
):
    cond_enc_type = model_cfg.cond_enc_type
    match cond_enc_type:
        case "miche":
            model_class = PointCloudEncoder
        case "edgerunner":
            model_class = EdgeRunnerPointEncoder
        case _:
            raise ValueError(f"Unknown cond enc type: {cond_enc_type}")
    extra_args = {}
    if cond_encoder_img is not None:
        extra_args["with_extra_feat"] = True
        extra_args["extra_feat_dim"] = cond_encoder_img.output_dim
    model = model_class.from_pretrained(local_model_path, **extra_args)
    return ConditionEncoder(model, freeze=model_cfg.freeze_cond_encoder)


def get_image_condition_encoder(model_cfg: ModelConfig):
    if model_cfg.high_res_image_encoder:
        out_features = model_cfg.image_encoder_layers
        model = HighResImageConditionEncoder(
            model_name=model_cfg.image_encoder,
            out_features=out_features,
            hidden_size=model_cfg.high_res_image_encoder_hidden_size,
        )
    else:
        model = ImageConditionEncoder(model_name=model_cfg.image_encoder)
    return model
