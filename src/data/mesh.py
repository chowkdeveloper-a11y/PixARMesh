import datasets
from functools import partial
from pathlib import Path
import os
import json
import trimesh
import numpy as np
import random
from transformers import AutoImageProcessor, PreTrainedTokenizerBase, ProcessorMixin
from src.utils.config import DataConfig, ModelConfig
from src.data import utils


class MeshTokenizer(PreTrainedTokenizerBase):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__(
            max_len=model_cfg.max_seq_length,
            padding_side="right",
        )
        self.is_fast = True
        self.vocab_size = model_cfg.vocab_size
        self._special_tokens_map.update(
            {
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }
        )
        self._vocab = {
            "<pad>": model_cfg.pad_token_id,
            "<bos>": model_cfg.bos_token_id,
            "<eos>": model_cfg.eos_token_id,
        }

    @property
    def added_tokens_decoder(self):
        return {}

    @property
    def added_tokens_encoder(self):
        return {}

    def save_vocabulary(self, save_directory, filename_prefix=None):
        out_path = Path(save_directory)
        if filename_prefix is None:
            filename_prefix = ""
        vocab_file = out_path / (filename_prefix + "vocab.json")
        with vocab_file.open("w") as fp:
            json.dump(self._vocab, fp, indent=2)
        return (str(vocab_file),)

    def convert_tokens_to_ids(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            tokens = [tokens]
        ids = [self._vocab.get(token, self._vocab.get("<unk>", 0)) for token in tokens]
        if len(ids) == 1:
            return ids[0]
        return ids


class MeshProcessor(ProcessorMixin):
    def __init__(self, model_cfg: ModelConfig):
        self.tokenizer = MeshTokenizer(model_cfg)

    def save_pretrained(self, *args, **kwargs):
        # No op
        pass


def subsample_point_clouds(
    valid_pts, valid_pts_2d, num_points, is_train, data_cfg: DataConfig, with_normals
):
    if len(valid_pts) == 0:
        pc_valid = False
        sampled_pc = np.zeros((num_points, 3 + (3 if with_normals else 0)))
        sampled_pc_2d = np.zeros((num_points, 2))
        sample_inds = None
    else:
        sampled_pc, sample_inds = utils.random_sample_point_clouds(
            valid_pts, num_points, return_inds=True
        )
        sampled_pc_2d = valid_pts_2d[sample_inds]
        # Augmentation: random perturbation
        if (
            is_train
            and data_cfg.random_jitter_point_clouds
            and random.random() < data_cfg.random_jitter_probability
        ):
            sampled_pc += (
                np.random.randn(*sampled_pc.shape) * data_cfg.random_jitter_offset
            )
        if with_normals:
            sampled_pc = utils.estimate_point_cloud_normals(sampled_pc)
        pc_valid = True
    return sampled_pc, sampled_pc_2d, pc_valid, sample_inds


def transform_mesh(example, is_train, data_cfg: DataConfig):
    result_vertices = []
    result_faces = []
    result_point_clouds = []
    num_points = data_cfg.num_points
    with_normals = data_cfg.with_normals
    norm_bound = data_cfg.norm_bound
    has_pc = num_points > 0
    for idx, (vertices, faces) in enumerate(zip(example["vertices"], example["faces"])):
        vertices = np.array(vertices)
        vertices = utils.normalize_vertices(vertices, bound=norm_bound)
        faces = np.array(faces)
        result_vertices.append(vertices)
        result_faces.append(faces)
        if has_pc:
            pc = utils.sample_point_cloud(vertices, faces, num_points, with_normals)
            result_point_clouds.append(pc)
    ret = {
        "vertices": result_vertices,
        "faces": result_faces,
    }
    if has_pc:
        ret["point_clouds"] = result_point_clouds
    return ret


def get_instance_mesh(vertices, faces, return_raw=False):
    if len(vertices) == 0 or len(faces) == 0:
        if return_raw:
            return None
        return None, None
    if return_raw:
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    else:
        vertices = np.array(vertices)
        faces = np.array(faces)
        return vertices, faces


def transform_3d_front(
    example,
    is_train,
    data_cfg: DataConfig,
    image_preprocessor=None,
):
    result_images = []
    result_bboxes = []
    result_T_gravity = []
    result_point_clouds = []
    result_point_clouds_2d = []
    result_ctx_point_clouds = []
    result_ctx_point_clouds_2d = []
    result_pc_valid = []
    result_obj_indices = []
    result_vertices = []
    result_faces = []
    result_obj_point_clouds = []

    result_depths = []
    result_all_point_clouds = []
    result_all_point_clouds_2d = []
    result_all_obj_bounds = []
    result_all_obj_to_cam_transform_rect = []
    result_all_obj_masks = []
    result_all_obj_model_ids = []

    result_pixel_values = []

    num_points = data_cfg.num_points
    with_normals = data_cfg.with_normals
    norm_bound = data_cfg.norm_bound
    has_pc = num_points > 0
    load_images = data_cfg.load_images and image_preprocessor is not None

    local_obj_num_points = data_cfg.local_obj_num_points
    mask_path = data_cfg.mask_path

    y_up_matrix = np.diag(np.array([-1, -1, 1, 1], dtype=np.float32))
    load_obj_mesh = data_cfg.type != "3d-front-layout"
    has_local_pc = local_obj_num_points > 0 and load_obj_mesh
    use_masked_obj_pc = data_cfg.use_masked_obj_pc and load_obj_mesh

    has_ctx_pc = has_pc and data_cfg.num_ctx_points > 0

    for idx in range(len(example["uid"])):
        scene_id = example["uid"][idx]
        image = np.array(example["image"][idx])
        objects = example["objects"][idx]
        all_model_ids = objects["model_ids"]
        bounds = np.array(objects["bounds"], dtype=np.float32)
        transforms = np.array(objects["transforms"], dtype=np.float32)
        wrd2cam_rect = np.array(example["wrd2cam_rect"][idx], dtype=np.float32)
        rect_inv = np.array(example["rect_inv"][idx], dtype=np.float32)
        K = np.array(example["K"][idx], dtype=np.float32)
        gt_depth = np.array(example["depth"][idx], dtype=np.float32)
        gt_depth = (1 - gt_depth / 255.0) * 10.0  # to meters

        if data_cfg.use_predicted_depth:
            if data_cfg.predicted_depth_aligned:
                depth = np.load(os.path.join(data_cfg.depth_path, f"{scene_id}.npy"))
            else:
                pred_depth = utils.read_depth_pro_depth(data_cfg.depth_path, scene_id)
                try:
                    depth = utils.align_depth(pred_depth, gt_depth, mask=gt_depth > 0)
                except Exception:
                    depth = gt_depth
        else:
            depth = gt_depth

        depth_trunc = 1e-6
        valid_mask = depth > depth_trunc
        if is_train and data_cfg.random_jitter_depth:
            depth_noise = (
                np.random.randn(*depth.shape) * data_cfg.random_jitter_depth_offset
            )
            depth[valid_mask] += depth_noise[valid_mask]
            depth[valid_mask].clip(min=depth_trunc)

        pad_size_left = pad_size_top = 0
        out_h, out_w = image.shape[:2]
        if load_images:
            processed = image_preprocessor(images=image, return_tensors="pt")
            pixel_values = processed["pixel_values"]
            orig_h, orig_w = image.shape[0], image.shape[1]
            out_h, out_w = pixel_values.shape[2], pixel_values.shape[3]
            pad_size_left = (out_w - orig_w) // 2
            pad_size_top = (out_h - orig_h) // 2
            result_pixel_values.append(pixel_values)

        inst_ids = objects["inst_ids"]

        t_cam = wrd2cam_rect[:3, 3]
        T_gravity = np.eye(4, dtype=np.float32)
        T_gravity[:3, :3] = rect_inv
        T_gravity[:3, 3] = t_cam - rect_inv @ t_cam
        T_gravity_inv = np.linalg.inv(T_gravity).astype(np.float32)

        K_inv = np.linalg.inv(K).astype(np.float32)
        depth_pcs, pix_pcs = utils.back_project_depth(
            depth, K_inv, return_pix_coords=True
        )
        pix_pcs = pix_pcs + np.array([pad_size_left, pad_size_top])
        # We assume align_corners=False
        pix_pcs = (pix_pcs + 0.5) / np.array([out_w, out_h]) * 2 - 1

        img_shape = depth_pcs.shape[:2]
        depth_pcs = utils.transform_3d_points(
            depth_pcs.reshape(-1, 3), T_gravity_inv @ y_up_matrix
        ).reshape(img_shape[0], img_shape[1], 3)

        depth_pcs[~valid_mask] = depth_trunc
        all_pcd = depth_pcs.reshape(-1, 3)
        valid_cols = image[valid_mask]
        valid_obj_pts = None

        obj_center = obj_scale = None

        obj_rot_mat = None
        augmented_obj_bound = None

        if mask_path:
            obj_masks = utils.get_masks_by_ids(
                example["panoptic_mask"][idx],
                inst_ids,
                erode_size=data_cfg.mask_erosion_size,
            )
            result_all_obj_masks.append(obj_masks)
        else:
            obj_masks = None

        if load_obj_mesh:
            if is_train:
                inst_idx = np.random.randint(0, len(all_model_ids))
            else:
                inst_idx = idx % len(all_model_ids)

            vertices, faces = get_instance_mesh(
                objects["vertices"][inst_idx], objects["faces"][inst_idx]
            )
            if vertices is not None and faces is not None:
                if is_train and data_cfg.random_rotate_sampled_instance:
                    obj_azimuth = np.random.choice(
                        [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
                    )
                    obj_azimuth = np.deg2rad(obj_azimuth)
                    obj_rot_mat = utils.get_rotation_y_matrix(obj_azimuth)
                    vertices = vertices @ obj_rot_mat.T
                    augmented_obj_bound = np.stack([vertices.min(0), vertices.max(0)])
                vertices, obj_center, obj_scale = utils.normalize_vertices(
                    vertices, bound=norm_bound, return_all=True
                )
                faces = np.array(faces)
            result_obj_indices.append(inst_idx)
            result_vertices.append(vertices)
            result_faces.append(faces)

            if has_local_pc:
                mask = obj_masks[inst_idx]
                valid_obj_pts = depth_pcs[mask & valid_mask]

        all_bboxes_rect = []
        all_obj_bounds = []
        all_obj_to_cam_transform_rect = []
        for obj_idx, (bound, transform) in enumerate(zip(bounds, transforms)):
            to_cam_transform_rect = wrd2cam_rect @ transform
            if augmented_obj_bound is not None and obj_idx == inst_idx:
                bound = augmented_obj_bound
                to_cam_transform_rect = to_cam_transform_rect @ utils.augment_matrix(
                    np.linalg.inv(obj_rot_mat)
                )
            points_3d = utils.make_3d_bbox(*bound)
            aug_points_3d = np.concatenate(
                [points_3d, np.ones((points_3d.shape[0], 1))], axis=1
            )
            points_3d = aug_points_3d @ to_cam_transform_rect.T
            all_bboxes_rect.append(points_3d[:, :3])
            all_obj_bounds.append(bound)
            all_obj_to_cam_transform_rect.append(to_cam_transform_rect)

        all_obj_bounds = np.array(all_obj_bounds)
        all_obj_to_cam_transform_rect = np.array(all_obj_to_cam_transform_rect)

        all_bboxes_rect = np.stack(all_bboxes_rect, axis=0)

        sampled_obj_pc = None
        if has_local_pc:
            local_pc_valid = (
                vertices is not None and faces is not None and len(valid_obj_pts) > 0
            )
            should_drop = (
                is_train and random.random() < data_cfg.local_obj_cond_drop_prob
            )
            if local_pc_valid and not should_drop:
                valid_obj_pts = utils.transform_3d_points(
                    valid_obj_pts,
                    np.linalg.inv(all_obj_to_cam_transform_rect[inst_idx]),
                )
                valid_obj_pts = (valid_obj_pts - obj_center) * obj_scale
                sampled_obj_pc = utils.random_sample_point_clouds(
                    valid_obj_pts, local_obj_num_points
                )

                if (
                    is_train
                    and data_cfg.random_jitter_point_clouds
                    and random.random() < data_cfg.random_jitter_probability
                ):
                    sampled_obj_pc += (
                        np.random.randn(*sampled_obj_pc.shape)
                        * data_cfg.random_jitter_offset
                    )
                if with_normals:
                    sampled_obj_pc = utils.estimate_point_cloud_normals(sampled_obj_pc)
            else:
                sampled_obj_pc = None

            result_obj_point_clouds.append(sampled_obj_pc)

        if has_pc:
            pc_valid = True
            # Augmentation: random rotation (along y axis) and scaling
            if is_train and data_cfg.random_scale:
                bound = np.random.uniform(data_cfg.random_scale_min, norm_bound)
            else:
                bound = norm_bound

            if is_train and data_cfg.random_rotate:
                azimuth = np.random.uniform(
                    data_cfg.random_rotate_min, data_cfg.random_rotate_max
                )
                azimuth = np.deg2rad(azimuth)
                rot_mat = utils.get_rotation_y_matrix(azimuth)
                all_bboxes_rect = all_bboxes_rect @ rot_mat.T
                all_pcd = all_pcd @ rot_mat.T
                all_obj_to_cam_transform_rect = (
                    utils.augment_matrix(rot_mat) @ all_obj_to_cam_transform_rect
                )

            all_bboxes_rect, all_pcd, normalize_matrix = (
                utils.normalize_bboxes_with_point_clouds(
                    all_bboxes_rect,
                    all_pcd,
                    bound=bound,
                    return_matrix=True,
                )
            )
            all_obj_to_cam_transform_rect = (
                normalize_matrix @ all_obj_to_cam_transform_rect
            )
            if is_train and data_cfg.random_shift:
                all_bboxes_rect, all_pcd, shift_matrix = (
                    utils.random_shift_bboxes_with_point_clouds(
                        all_bboxes_rect,
                        all_pcd,
                        max_shift=data_cfg.random_shift_max,
                        bound=bound,
                        return_matrix=True,
                    )
                )
                all_obj_to_cam_transform_rect = (
                    shift_matrix @ all_obj_to_cam_transform_rect
                )
            all_pcd = all_pcd.reshape(img_shape[0], img_shape[1], 3)
            result_all_point_clouds.append(all_pcd)
            result_all_point_clouds_2d.append(pix_pcs)
            if use_masked_obj_pc:
                pts_mask = valid_mask & obj_masks[inst_idx]
            else:
                pts_mask = valid_mask

            valid_pts = all_pcd[pts_mask]
            valid_pts_2d = pix_pcs[pts_mask]

            ctx_pts = all_pcd[valid_mask]
            ctx_pts_2d = pix_pcs[valid_mask]
            if has_ctx_pc:
                sampled_ctx_pc, sampled_ctx_pc_2d, _, _ = subsample_point_clouds(
                    ctx_pts,
                    ctx_pts_2d,
                    data_cfg.num_ctx_points,
                    is_train,
                    data_cfg,
                    with_normals,
                )
                result_ctx_point_clouds.append(sampled_ctx_pc)
                result_ctx_point_clouds_2d.append(sampled_ctx_pc_2d)

            sampled_pc, sampled_pc_2d, pc_valid, sample_inds = subsample_point_clouds(
                valid_pts, valid_pts_2d, num_points, is_train, data_cfg, with_normals
            )

            result_point_clouds.append(sampled_pc)
            result_point_clouds_2d.append(sampled_pc_2d)
            result_pc_valid.append(pc_valid)
        else:
            sampled_pc = sample_inds = None

        if data_cfg.visualize:
            from .vis import visualize_pcs_and_bboxes, visualize_obj_and_pcs
            from .utils import decode_gravity_aligned_bbox

            vis_pc = sampled_pc if sampled_pc is not None else valid_pts
            vis_col = valid_cols[sample_inds] if sampled_pc is not None else valid_cols
            vis_boxes = decode_gravity_aligned_bbox(all_bboxes_rect[:, [0, 2, 5]])
            scene = visualize_pcs_and_bboxes(vis_pc[:, :3], vis_col, vis_boxes)
            if load_obj_mesh and vertices is not None and faces is not None:
                mesh = get_instance_mesh(
                    objects["vertices"][inst_idx],
                    objects["faces"][inst_idx],
                    return_raw=True,
                )
                if obj_rot_mat is not None:
                    mesh.apply_transform(utils.augment_matrix(obj_rot_mat))
                mesh.apply_transform(all_obj_to_cam_transform_rect[inst_idx])

                if sampled_obj_pc is not None:
                    obj_scene = visualize_obj_and_pcs(
                        vertices, faces, sampled_obj_pc[:, :3]
                    )
                    obj_scene.export(f"vis_obj_{idx}.glb")

                scene.add_geometry(mesh, node_name="selected_obj")

            scene.export(f"vis_{idx}.glb")

        result_images.append(image)
        result_bboxes.append(all_bboxes_rect)
        result_depths.append(depth)
        result_T_gravity.append(T_gravity)
        result_all_obj_bounds.append(all_obj_bounds)
        result_all_obj_to_cam_transform_rect.append(all_obj_to_cam_transform_rect)
        result_all_obj_model_ids.append(all_model_ids)

    ret = {
        "uid": example["uid"],
        "images": result_images,
        "bboxes": result_bboxes,
        "depths": result_depths,
        "all_obj_bounds": result_all_obj_bounds,
        "all_obj_to_cam_transforms": result_all_obj_to_cam_transform_rect,
        "all_obj_model_ids": result_all_obj_model_ids,
    }
    if result_all_obj_masks:
        ret["all_obj_masks"] = result_all_obj_masks
    if has_pc:
        ret["all_point_clouds"] = result_all_point_clouds
        ret["all_point_clouds_2d"] = result_all_point_clouds_2d
        ret["point_clouds"] = result_point_clouds
        ret["point_clouds_2d"] = result_point_clouds_2d
        ret["point_clouds_valid"] = result_pc_valid
    if has_ctx_pc:
        ret["ctx_point_clouds"] = result_ctx_point_clouds
        ret["ctx_point_clouds_2d"] = result_ctx_point_clouds_2d
    if load_obj_mesh:
        ret["obj_indices"] = result_obj_indices
        ret["vertices"] = result_vertices
        ret["faces"] = result_faces
    if has_local_pc:
        ret["obj_point_clouds"] = result_obj_point_clouds
    if load_images:
        ret["pixel_values"] = result_pixel_values
    return ret


def get_mesh_dataset(data_cfg: DataConfig):
    local_path = Path(data_cfg.path).absolute().as_posix()
    num_proc = max(min(os.cpu_count(), 64), 16)
    data = datasets.load_dataset(local_path, num_proc=num_proc)
    data_type = data_cfg.type
    match data_type:
        case "shapenet":
            mapper = transform_mesh
        case "3d-front" | "3d-front-layout":
            mapper = transform_3d_front
        case _:
            raise ValueError(f"Unknown dataset type: {data_type}")
    kwargs = {}
    if data_cfg.load_images:
        kwargs["image_preprocessor"] = AutoImageProcessor.from_pretrained(
            data_cfg.image_preprocessor, size_divisor=data_cfg.image_size_divisor
        )
    train_data = data["train"].with_transform(
        partial(mapper, is_train=True, data_cfg=data_cfg, **kwargs)
    )
    val_key = "val" if "val" in data else "test"
    val_data = data[val_key].with_transform(
        partial(mapper, is_train=False, data_cfg=data_cfg, **kwargs)
    )
    test_data = data["test"]
    return train_data, val_data, test_data
