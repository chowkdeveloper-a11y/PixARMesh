from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    type: str
    path: str
    num_points: int = 4096
    norm_bound: float = 0.9995
    with_normals: bool = False
    mesh_path: str = ""
    num_pos_tokens: int = 128
    random_scale: bool = True
    random_scale_min: float = 0.75
    random_rotate_sampled_instance: bool = False
    random_rotate: bool = True
    random_rotate_min: float = 0.0
    random_rotate_max: float = 360.0
    random_jitter_point_clouds: bool = True
    random_jitter_probability: float = 0.5
    random_jitter_offset: float = 0.01
    random_jitter_depth: bool = True
    random_jitter_depth_offset: float = 0.02
    random_shift: bool = False
    random_shift_max: float = 0.2
    visualize: bool = False
    # For local condition
    local_obj_num_points: int = -1
    mask_path: str = ""
    mask_erosion_size: int = 3
    depth_path: str = ""
    use_predicted_depth: bool = False
    predicted_depth_aligned: bool = False
    local_obj_cond_drop_prob: float = 0.2
    # Object point cloud in global frame
    use_masked_obj_pc: bool = False
    # Image conditions
    load_images: bool = False
    image_preprocessor: str = "facebook/dpt-dinov2-small-nyu"
    image_size_divisor: int = 28
    # Context point clouds
    num_ctx_points: int = 0
    # Ablations
    ignore_obj_seq: bool = False
    ignore_layout_seq: bool = False


@dataclass
class ModelConfig:
    vocab_size: int
    num_pos_tokens: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    sep_token_id: Optional[int] = None
    indicator_token_id: int = -50
    obj_pc_token_id: int = -49
    pc_token_id: int = -48
    prefix_len: int = 0
    pc_latent_len: int = 0
    cond: bool = False
    obj_cond: bool = False
    local_path: str = ""
    local_cond_path: str = ""
    cond_enc_type: str = "miche"
    freeze_cond_encoder: bool = True
    ar_model_type: str = "meshxl"
    tokenization_method: str = "meshxl"
    max_seq_length: int = 8192
    max_position_embeddings: int = 8192
    pos_token_offset: int = 0
    no_layout_loss: bool = False
    layout_tokenization_method: str = "tri"
    img_cond: bool = False
    image_encoder: str = ""
    high_res_image_encoder: bool = False
    high_res_image_encoder_hidden_size: int = 256
    image_encoder_layers: List[str] = field(default_factory=lambda: [])
    with_ctx_pc: bool = False
    img_cond_drop_prob: float = 0.0
    loss_layout_scale: Optional[float] = None
