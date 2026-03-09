from functools import partial
import json
import jsonlines
import datasets
import trimesh
import torch
import numpy as np
from x_transformers.autoregressive_wrapper import top_p, top_k
from src.data import tokenize_bpt
from src.data.mesh import get_mesh_dataset, transform_3d_front
from src.models.utils import (
    get_image_condition_encoder,
    get_condition_encoder,
    get_model,
)
from src.utils.config import ModelConfig, DataConfig


def _flatten_3d_front_for_inference(examples, data_cfg, use_predicted_mask):
    examples = transform_3d_front(examples, is_train=False, data_cfg=data_cfg)
    uids = examples["uid"]
    images = examples["images"]
    depths = examples["depths"]
    all_bboxes = examples["bboxes"]
    all_obj_bounds = examples["all_obj_bounds"]
    all_obj_to_cam_transforms = examples["all_obj_to_cam_transforms"]
    all_obj_model_ids = examples["all_obj_model_ids"]
    all_obj_masks = examples["all_obj_masks"]
    all_point_clouds = examples["all_point_clouds"]
    all_point_clouds_2d = examples["all_point_clouds_2d"]

    result_uids = []
    result_images = []
    result_depths = []
    result_bboxes = []
    result_pcds = []
    result_pcds_2d = []
    result_bounds = []
    result_transforms = []
    result_model_ids = []
    result_masks = []

    if use_predicted_mask:
        placeholder_bound = np.zeros((2, 3), dtype=np.float32)
        placeholder_transform = np.eye(4, dtype=np.float32)
        model_id = ""
        for uid, image, depth, pcds, pcds_2d in zip(
            uids, images, depths, all_point_clouds, all_point_clouds_2d
        ):
            mask_info = f"datasets/grounded_sam/{uid}/mask_annotations.json"
            with open(mask_info) as fp:
                mask_data = json.load(fp)
            mask_path = f"datasets/grounded_sam/{uid}/masks.npz"
            masks = np.load(mask_path)["masks"]
            n = masks.shape[0]
            for ind, info in enumerate(mask_data):
                if info["class_name"] == "lamp":
                    continue
                result_uids.append(f"{uid}_{ind}")
                result_images.append(image)
                result_depths.append(depth)
                result_bboxes.append(np.zeros((n, 8, 3), dtype=np.float32))
                result_pcds.append(pcds)
                result_pcds_2d.append(pcds_2d)
                result_bounds.append(placeholder_bound)
                result_transforms.append(placeholder_transform)
                result_model_ids.append(model_id)
                result_masks.append(masks[ind])
    else:
        for (
            uid,
            image,
            depth,
            bboxes,
            bounds,
            transforms,
            model_ids,
            masks,
            pcds,
            pcds_2d,
        ) in zip(
            uids,
            images,
            depths,
            all_bboxes,
            all_obj_bounds,
            all_obj_to_cam_transforms,
            all_obj_model_ids,
            all_obj_masks,
            all_point_clouds,
            all_point_clouds_2d,
        ):
            for ind, (bound, transform, model_id, mask) in enumerate(
                zip(bounds, transforms, model_ids, masks)
            ):
                result_uids.append(f"{uid}_{ind}")
                result_images.append(image)
                result_depths.append(depth)
                result_bboxes.append(bboxes)
                result_pcds.append(pcds)
                result_pcds_2d.append(pcds_2d)
                result_bounds.append(bound)
                result_transforms.append(transform)
                result_model_ids.append(model_id)
                result_masks.append(mask)

    return {
        "uid": result_uids,
        "image": result_images,
        "depth": result_depths,
        "bboxes": result_bboxes,
        "point_clouds": result_pcds,
        "point_clouds_2d": result_pcds_2d,
        "bound": result_bounds,
        "transform": result_transforms,
        "model_id": result_model_ids,
        "mask": result_masks,
    }


def prepare_test_set(data_cfg, metadata, use_predicted_mask):
    test_set = get_mesh_dataset(data_cfg)[2]

    with jsonlines.open(metadata, "r") as reader:
        valid_ids = [obj["image_id"] for obj in reader]

    valid_indices = []
    all_uids = test_set["uid"]
    for i, uid in enumerate(all_uids):
        if str(uid) in valid_ids:
            valid_indices.append(i)

    subset_test_set = test_set.select(valid_indices)
    image_size = (484, 648)
    subset_test_set = subset_test_set.map(
        partial(
            _flatten_3d_front_for_inference,
            data_cfg=data_cfg,
            use_predicted_mask=use_predicted_mask,
        ),
        batched=True,
        batch_size=4,
        num_proc=4,
        remove_columns=subset_test_set.column_names,
        features=datasets.Features(
            {
                "uid": datasets.Value("string"),
                "image": datasets.Array3D(dtype="uint8", shape=(*image_size, 3)),
                "depth": datasets.Array2D(dtype="float32", shape=image_size),
                "bboxes": datasets.Sequence(
                    datasets.Array2D(dtype="float32", shape=(8, 3))
                ),
                "point_clouds": datasets.Array3D(
                    dtype="float32", shape=(*image_size, 3)
                ),
                "point_clouds_2d": datasets.Array3D(
                    dtype="float32", shape=(*image_size, 2)
                ),
                "bound": datasets.Array2D(dtype="float32", shape=(2, 3)),
                "transform": datasets.Array2D(dtype="float32", shape=(4, 4)),
                "model_id": datasets.Value("string"),
                "mask": datasets.Array2D(dtype="bool", shape=image_size),
            }
        ),
    )
    return subset_test_set


def joint_filter(logits, k=50, p=0.95):
    logits = top_k(logits, k=k)
    logits = top_p(logits, thres=p)
    return logits


def get_prefix_allowed_tokens_fn_edgerunner(model, batch_size=1):
    def prefix_allowed_tokens_fn_with_state(batch_id, input_ids, states):
        state = states[batch_id]
        idx = input_ids.shape[0]
        # print(f'=== prefix idx: {idx} ===')
        # BOS is always provided, so the first token must be BOM
        # 0=PAD, 1=BOS, 2=EOS, 3=L, 4=R, 5=BOM, 6~=coords
        if idx == 0:
            return [5]

        # update state based on the last token
        if input_ids[-1] == 5:
            state["counter"] = 9  # after BOM, there must be 9 coords tokens
        elif input_ids[-1] in [3, 4]:
            state["counter"] = 3  # after LR, there must be 3 coords tokens
        elif input_ids[-1] >= 6:
            state["counter"] -= 1  # after coords, counter -1

        # set rules for the next token
        # counter > 0 means there are still coords to be filled
        if state["counter"] > 0:
            return list(range(6, model.config.vocab_size))
        # otherwise, it could be L/R/BOM/EOS
        else:
            return [3, 4, 5, model.config.eos_token_id]

    # keep a persistent state during generation
    states = [{"counter": 0} for _ in range(batch_size)]
    prefix_allowed_tokens_fn = partial(
        prefix_allowed_tokens_fn_with_state, states=states
    )
    return prefix_allowed_tokens_fn


def decode_mesh_edgerunner(tokens, tokenizer, clean=True, verbose=False):
    tokens = tokens - 3
    vertices, faces, face_type = tokenizer.decode(tokens)

    if verbose:
        print(f"[INFO] vertices: {vertices.shape[0]}, faces: {faces.shape[0]}")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # fix flipped faces and merge close vertices
    if clean:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.fix_normals()

        if verbose:
            print(
                f"[INFO] cleaned vertices: {mesh.vertices.shape[0]}, faces: {mesh.faces.shape[0]}"
            )
    return mesh


def decode_bpt(tokens):
    vertices = tokenize_bpt.BPT_deserialize(tokens)
    num_vertices = len(vertices) // 3 * 3
    faces = np.arange(1, num_vertices + 1).reshape(-1, 3)
    mesh = tokenize_bpt.to_mesh(vertices, faces, post_process=True)
    return mesh


def recover_box_transform(P_local, Q_world):
    """
    Recover 7-DoF transform (yaw, per-axis scale, translation)
    given corresponding ordered corners.

    Args:
        P_local : (8,3) np.array — canonical box corners in object local frame
        Q_world : (8,3) np.array — observed corners in world frame (same order)

    Returns:
        yaw   : float (radians)  — rotation about Y
        scale : (3,) np.array    — [sx, sy, sz], positive
        trans : (3,) np.array    — translation vector
        T     : (4,4) np.array   — homogeneous transform (world <- local)
    """
    P = np.asarray(P_local, dtype=float)
    Q = np.asarray(Q_world, dtype=float)

    # --- remove translation ---
    muP = P.mean(0)
    muQ = Q.mean(0)
    P0, Q0 = P - muP, Q - muQ

    # --- solve in XZ plane for yaw + sx, sz ---
    P_xz, Q_xz = P0[:, [0, 2]], Q0[:, [0, 2]]
    X, *_ = np.linalg.lstsq(P_xz, Q_xz, rcond=None)
    A = X.T  # 2×2 affine part

    # Decompose A ≈ R(θ)·diag(sx, sz)
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    sx = np.hypot(a, c)
    sz = np.hypot(b, d)
    yaw1 = np.arctan2(-c, a)
    yaw2 = np.arctan2(b, d)
    # robust average of angles
    yaw = np.arctan2(np.sin(yaw1) + np.sin(yaw2), np.cos(yaw1) + np.cos(yaw2))

    # --- y-scale (since yaw leaves Y axis unchanged) ---
    Py, Qy = P0[:, 1], Q0[:, 1]
    sy = float(np.dot(Py, Qy) / np.dot(Py, Py))
    sy = abs(sy)

    scale = np.array([abs(sx), sy, abs(sz)])

    # --- translation ---
    R = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    )
    trans = muQ - (R @ (scale * muP))

    T = np.eye(4)
    T[:3, :3] = R @ np.diag(scale)
    T[:3, 3] = trans

    return yaw, scale, trans, T


def run_perspectivefields(image):
    from perspective2d import PerspectiveFields
    from perspective2d.utils.utils import general_vfov_to_focal

    img_bgr = image[..., ::-1]
    H, W, _ = img_bgr.shape
    pf_model = PerspectiveFields("Paramnet-360Cities-edina-uncentered").eval().cuda()
    pred = pf_model.inference(img_bgr=img_bgr)

    roll = np.radians(pred["pred_roll"].cpu().item())
    pitch = np.radians(pred["pred_pitch"].cpu().item())
    vfov = np.radians(pred["pred_general_vfov"].cpu().item())
    cx_rel = pred["pred_rel_cx"].cpu().item()
    cy_rel = pred["pred_rel_cy"].cpu().item()
    focal_rel = general_vfov_to_focal(cx_rel, cy_rel, 1, vfov, degree=False)
    f = focal_rel * H
    cx = (cx_rel + 0.5) * W
    cy = (cy_rel + 0.5) * H
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    return K, roll, pitch


def initialize_depth_pro(ckpt_path="checkpoint/depth_pro.pt"):
    import depth_pro

    config = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = ckpt_path
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    model.cuda()

    def get_depth_from_depth_pro(
        image: np.ndarray, f_px: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = transform(image).cuda()
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"]
        f_px = prediction["focallength_px"]
        return depth.detach().cpu(), f_px.detach().cpu()

    return get_depth_from_depth_pro


def prepare_model_for_inference(is_bpt, checkpoint):
    if is_bpt:
        cond_encoder_name = "miche-encoder-bpt"
        params = {
            "vocab_size": 5184,
            "num_pos_tokens": 128,
            "bos_token_id": -100,
            "eos_token_id": 5120,
            "pad_token_id": -1,
            "prefix_len": 0,
            "cond_enc_type": "miche",
            "ar_model_type": "bpt",
            "tokenization_method": "bpt",
            "max_seq_length": 10000,
            "max_position_embeddings": 10000,
            "pos_token_offset": 0,
            "sep_token_id": 5121,
            "indicator_token_id": 5121,
            "pc_latent_len": 0,
        }
    else:
        cond_encoder_name = "edgerunner-pc-encoder"
        params = {
            "vocab_size": 576,
            "num_pos_tokens": 512,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "prefix_len": 2049,
            "cond_enc_type": "edgerunner",
            "ar_model_type": "edgerunner",
            "tokenization_method": "edgerunner",
            "max_seq_length": 43019,
            "max_position_embeddings": 43019,
            "pos_token_offset": 6,
            "sep_token_id": 518,
            "indicator_token_id": 518,
            "pc_latent_len": 2048,
        }
    model_cfg = ModelConfig(
        cond=True,
        layout_tokenization_method="full",
        loss_layout_scale=None,
        img_cond=True,
        image_encoder="facebook/dinov2-with-registers-small",
        local_cond_path=f"zx1239856/{cond_encoder_name}",
        local_path=checkpoint,
        high_res_image_encoder=False,
        freeze_cond_encoder=False,
        with_ctx_pc=True,
        **params,
    )
    data_cfg = DataConfig(
        type="3d-front-layout",
        path="datasets/3d-front-ar-packed",
        num_pos_tokens=model_cfg.num_pos_tokens,
        num_points=4096 if is_bpt else 8192,
        norm_bound=0.95,
        mesh_path="datasets/3d-front-meshes",
        random_rotate_min=-45.0,
        random_rotate_max=45.0,
        random_shift=True,
        random_shift_max=0.2,
        visualize=False,
        mask_path="datasets/3d-front-panoptic",
        mask_erosion_size=3,
        use_masked_obj_pc=True,
        with_normals=is_bpt,
        random_jitter_point_clouds=False,
        load_images=True,
        image_preprocessor="facebook/dpt-dinov2-small-nyu",
        image_size_divisor=28,
        num_ctx_points=16384,
    )
    cond_encoder_img = get_image_condition_encoder(model_cfg)
    cond_encoder = get_condition_encoder(
        model_cfg.local_cond_path, model_cfg, cond_encoder_img=cond_encoder_img
    )
    model = get_model(
        model_cfg.local_path,
        model_cfg,
        cond_encoder=cond_encoder,
        cond_encoder_img=cond_encoder_img,
    )
    return model, model_cfg, data_cfg
