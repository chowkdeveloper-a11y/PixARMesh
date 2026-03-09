import numpy as np
import torch
from src.utils.config import DataConfig, ModelConfig
from src.data import tokenize_bpt
from .typing import TokenType
from .utils import quantize_mesh, quantize_gravity_aligned_bboxes, dequantize_points


def pad_tokens(all_input_ids, pad_token_id, padding_side="right", max_seq_length=None):
    max_len = max(len(input_ids) for input_ids in all_input_ids)
    padded_input_ids = np.full(
        (len(all_input_ids), max_len), pad_token_id, dtype=np.int64
    )
    attention_mask = np.zeros((len(all_input_ids), max_len), dtype=np.int64)
    for i, input_ids in enumerate(all_input_ids):
        if padding_side == "right":
            padded_input_ids[i, : len(input_ids)] = input_ids
            attention_mask[i, : len(input_ids)] = 1
        else:
            padded_input_ids[i, -len(input_ids) :] = input_ids
            attention_mask[i, -len(input_ids) :] = 1

    if max_seq_length is not None and max_len > max_seq_length:
        padded_input_ids = padded_input_ids[:, :max_seq_length]
        attention_mask = attention_mask[:, :max_seq_length]

    return padded_input_ids, attention_mask


class BaseCollator:
    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
    ):
        self.bos_token_id = model_cfg.bos_token_id
        self.eos_token_id = model_cfg.eos_token_id
        self.pad_token_id = model_cfg.pad_token_id
        self.pc_token_id = model_cfg.pc_token_id
        self.prefix_len = model_cfg.prefix_len
        self.num_pos_tokens = data_cfg.num_pos_tokens
        self.pos_token_offset = model_cfg.pos_token_offset
        self.tokenization_method = model_cfg.tokenization_method
        self.max_seq_length = model_cfg.max_seq_length

        if self.tokenization_method == "edgerunner":
            from meto import Engine

            self.tokenizer = Engine(
                discrete_bins=self.num_pos_tokens, backend="LR_ABSCO"
            )

    def _pad(self, all_input_ids, all_token_type_ids=None):
        padded_input_ids, attention_mask = pad_tokens(
            all_input_ids,
            pad_token_id=self.pad_token_id,
            padding_side="right",
            max_seq_length=self.max_seq_length,
        )
        labels = padded_input_ids.copy()
        labels[labels == self.pad_token_id] = -100
        labels[labels == self.pc_token_id] = -100
        if self.prefix_len > 0:
            labels[:, self.prefix_len] = -100  # mask out <bos> loss for prefix
        ret = {
            "input_ids": torch.as_tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.as_tensor(attention_mask, dtype=torch.long),
            "labels": torch.as_tensor(labels, dtype=torch.long),
        }
        if all_token_type_ids is not None:
            padded_token_type_ids, _ = pad_tokens(
                all_token_type_ids,
                pad_token_id=TokenType.PADDING,
                padding_side="right",
                max_seq_length=self.max_seq_length,
            )
            ret["token_type_ids"] = torch.as_tensor(
                padded_token_type_ids, dtype=torch.long
            )
        return ret

    def _tokenize_bpt(self, vertices, faces):
        vertices, faces = quantize_mesh(
            vertices, faces, num_pos_tokens=self.num_pos_tokens
        )
        vertices = dequantize_points(vertices, num_pos_tokens=self.num_pos_tokens)
        mesh = tokenize_bpt.to_mesh(vertices, faces)
        seq = tokenize_bpt.BPT_serialize(mesh)
        return seq.tolist()

    def _tokenize_meshxl(self, vertices, faces):
        vertices, faces = quantize_mesh(
            vertices, faces, num_pos_tokens=self.num_pos_tokens
        )
        soup = vertices[faces]
        seq = soup.reshape(-1) + self.pos_token_offset
        return seq.tolist()

    def _tokenize_edgerunner(self, vertices, faces):
        tokens, _, _ = self.tokenizer.encode(vertices, faces)
        seq = tokens + 3
        return seq.tolist()

    def _get_mesh_sequence(self, vertices, faces):
        match self.tokenization_method:
            case "meshxl":
                tokenize_func = self._tokenize_meshxl
            case "edgerunner":
                tokenize_func = self._tokenize_edgerunner
            case "bpt":
                tokenize_func = self._tokenize_bpt
            case _:
                raise ValueError(
                    f"Unknown tokenization method: {self.tokenization_method}"
                )
        return tokenize_func(vertices, faces)


class MeshDataCollator(BaseCollator):
    def __call__(self, examples):
        all_input_ids = []
        all_cond_pcs = []
        has_pc = "point_clouds" in examples[0]
        for example in examples:
            vertices = example["vertices"]
            faces = example["faces"]
            if has_pc:
                pc = example["point_clouds"]
                all_cond_pcs.append(pc)
            mesh_seq = self._get_mesh_sequence(vertices, faces)
            input_ids = (
                [self.pc_token_id] * self.prefix_len
                + [self.bos_token_id]
                + mesh_seq
                + [self.eos_token_id]
            )
            all_input_ids.append(input_ids)
        ret = self._pad(all_input_ids)
        if has_pc:
            all_cond_pcs = np.array(all_cond_pcs)
            ret["cond_pcs"] = torch.as_tensor(all_cond_pcs, dtype=torch.float32)
        return ret


class Front3DCollator(BaseCollator):
    def __init__(self, data_cfg, model_cfg):
        super().__init__(data_cfg, model_cfg)
        self.indicator_token_id = model_cfg.indicator_token_id
        self.obj_pc_token_id = model_cfg.obj_pc_token_id
        self.obj_cond = model_cfg.obj_cond
        self.no_layout_loss = model_cfg.no_layout_loss
        self.pc_latent_len = model_cfg.pc_latent_len
        self.layout_tokenization_method = model_cfg.layout_tokenization_method
        if self.layout_tokenization_method not in ["tri", "full"]:
            raise ValueError(
                f"Unknown layout tokenization method: {self.layout_tokenization_method}"
            )
        # Placeholder for object point cloud latent tokens
        self.obj_pc_seq = (
            [self.obj_pc_token_id] * self.pc_latent_len if self.obj_cond else []
        )
        self.pc_dim = 3 + (3 if data_cfg.with_normals else 0)
        self.obj_num_points = data_cfg.local_obj_num_points
        self.box_tri_inds = [0, 2, 5]
        # Object point cloud in global frame
        self.use_masked_obj_pc = data_cfg.use_masked_obj_pc

        self.is_bpt = self.tokenization_method == "bpt"
        self.ignore_obj_seq = data_cfg.ignore_obj_seq
        self.ignore_layout_seq = data_cfg.ignore_layout_seq

    def _tokenize_bbox(self, rect_bboxes):
        quantized_bboxes, sort_inds = quantize_gravity_aligned_bboxes(
            rect_bboxes,
            num_pos_tokens=self.num_pos_tokens,
            return_sort_inds=True,
        )
        if self.layout_tokenization_method == "full":
            tris = quantized_bboxes
        else:
            tris = quantized_bboxes[:, self.box_tri_inds]
        if self.is_bpt:
            seq = tokenize_bpt.tokenize_layout(
                tris.reshape(-1, 3),
                block_size=8,
                offset_size=16,
            )
        else:
            seq = tris.reshape(-1) + self.pos_token_offset
        return seq.tolist()

    def __call__(self, examples):
        all_input_ids = []
        all_token_type_ids = []
        all_cond_pcs = []
        all_cond_pcs_2d = []
        all_cond_pc_valid = []
        all_ctx_pcs = []
        all_ctx_pcs_2d = []
        has_pc = "point_clouds" in examples[0]
        has_obj = "obj_indices" in examples[0]
        has_obj_pc = has_obj and "obj_point_clouds" in examples[0]
        has_pixel_values = "pixel_values" in examples[0]
        has_ctx_pc = "ctx_point_clouds" in examples[0]
        all_obj_indices = []
        all_obj_bboxes = []
        all_obj_cond_pcs = []
        layout_prefix_lens = []

        all_pixel_values = []

        for example in examples:
            bboxes = example["bboxes"]

            if has_pixel_values:
                pixel_values = example["pixel_values"]
                all_pixel_values.append(pixel_values)

            if self.use_masked_obj_pc and has_obj:
                obj_index = example["obj_indices"]
                layout_seq = self._tokenize_bbox(bboxes[[obj_index]])
            else:
                layout_seq = self._tokenize_bbox(bboxes)

            if has_pc:
                pc = example["point_clouds"]
                pc_2d = example["point_clouds_2d"]
                pc_valid = example["point_clouds_valid"]
                all_cond_pcs.append(pc)
                all_cond_pcs_2d.append(pc_2d)
                all_cond_pc_valid.append(pc_valid)

            if has_ctx_pc:
                ctx_pc = example["ctx_point_clouds"]
                ctx_pc_2d = example["ctx_point_clouds_2d"]
                all_ctx_pcs.append(ctx_pc)
                all_ctx_pcs_2d.append(ctx_pc_2d)

            obj_seq = []
            obj_type_ids = []

            if self.use_masked_obj_pc:
                obj_pc = None
            else:
                obj_pc = example.get("obj_point_clouds", None)

            if has_obj:
                obj_index = example["obj_indices"]
                obj_bbox = bboxes[obj_index][self.box_tri_inds]
                vertices = example["vertices"]
                faces = example["faces"]
                if vertices is not None and faces is not None:
                    mesh_seq = self._get_mesh_sequence(vertices, faces)
                    obj_pc_seq = self.obj_pc_seq if obj_pc is not None else []
                    obj_seq = obj_pc_seq + mesh_seq
                    obj_type_ids = [TokenType.COND_PREFIX] * len(obj_pc_seq) + [
                        TokenType.OBJECT
                    ] * len(mesh_seq)
                else:
                    obj_index = -1  # no object
                    obj_bbox = np.zeros((len(self.box_tri_inds), 3), dtype=np.float32)
                all_obj_indices.append(obj_index)
                all_obj_bboxes.append(obj_bbox)

            if has_obj_pc and obj_pc is not None:
                all_obj_cond_pcs.append(obj_pc)

            bos_seq = [self.bos_token_id] if not self.is_bpt else []
            bos_len = len(bos_seq)

            if self.ignore_obj_seq:
                obj_seq = []
                obj_type_ids = []

            indicator_seq = []
            if len(obj_seq) > 0 and not self.ignore_layout_seq:
                indicator_seq = [self.indicator_token_id]

            if self.ignore_layout_seq:
                layout_seq = []

            input_ids = (
                [self.pc_token_id] * self.prefix_len
                + bos_seq
                + layout_seq
                + indicator_seq
                + obj_seq
                + [self.eos_token_id]
            )
            token_type_ids = (
                [TokenType.COND_PREFIX] * self.prefix_len
                + [TokenType.SPECIAL_TOKEN] * bos_len
                + [TokenType.LAYOUT] * len(layout_seq)
                + [TokenType.SPECIAL_TOKEN] * len(indicator_seq)
                + obj_type_ids
                + [TokenType.SPECIAL_TOKEN]
            )
            layout_prefix_lens.append(self.prefix_len + bos_len + len(layout_seq))
            all_input_ids.append(input_ids)
            all_token_type_ids.append(token_type_ids)

        ret = self._pad(all_input_ids, all_token_type_ids)
        if has_pc:
            all_cond_pcs = np.array(all_cond_pcs)
            ret["cond_pcs"] = torch.as_tensor(all_cond_pcs, dtype=torch.float32)
            all_cond_pcs_2d = np.array(all_cond_pcs_2d)
            ret["cond_pcs_2d"] = torch.as_tensor(all_cond_pcs_2d, dtype=torch.float32)
        if has_ctx_pc:
            all_ctx_pcs = np.array(all_ctx_pcs)
            ret["ctx_pcs"] = torch.as_tensor(all_ctx_pcs, dtype=torch.float32)
            all_ctx_pcs_2d = np.array(all_ctx_pcs_2d)
            ret["ctx_pcs_2d"] = torch.as_tensor(all_ctx_pcs_2d, dtype=torch.float32)
        if has_obj:
            all_obj_indices = np.array(all_obj_indices)
            ret["obj_indices"] = torch.as_tensor(all_obj_indices, dtype=torch.long)
            all_obj_bboxes = np.array(all_obj_bboxes)
            ret["obj_bboxes"] = torch.as_tensor(all_obj_bboxes, dtype=torch.float32)
        if has_obj_pc:
            if len(all_obj_cond_pcs) > 0:
                all_obj_cond_pcs = np.array(all_obj_cond_pcs)
            else:
                all_obj_cond_pcs = np.zeros(
                    (0, self.obj_num_points, self.pc_dim), dtype=np.float32
                )
            ret["obj_cond_pcs"] = torch.as_tensor(all_obj_cond_pcs, dtype=torch.float32)
        # We need to set indicator token label to -100 as it is given
        labels = ret["labels"]
        if self.indicator_token_id < 0:
            labels[labels == self.indicator_token_id] = -100
        # Set object point cloud token labels to -100
        labels[labels == self.obj_pc_token_id] = -100
        layout_prefix_lens = np.array(layout_prefix_lens)
        no_obj_batch_ids = all_obj_indices == -1
        if len(no_obj_batch_ids) > 0:
            # Mask out <eos> for samples without object mesh
            labels[no_obj_batch_ids, layout_prefix_lens[no_obj_batch_ids]] = -100
        if self.no_layout_loss:
            layout_prefix_lens = torch.as_tensor(layout_prefix_lens, dtype=torch.long)
            inds = torch.arange(labels.size(1), device=labels.device)
            mask = inds[None, :] < layout_prefix_lens[:, None]
            labels[mask] = -100
        if has_pc:
            # Drop loss for examples with invalid point clouds
            pc_valid_mask = torch.as_tensor(all_cond_pc_valid, dtype=torch.bool)
            labels[~pc_valid_mask] = -100
        ret["labels"] = labels
        if has_pixel_values:
            all_pixel_values = torch.concat(all_pixel_values, dim=0)
            ret["pixel_values"] = all_pixel_values
        return ret


def get_mesh_data_collator(data_cfg: DataConfig, model_cfg: ModelConfig):
    data_type = data_cfg.type
    match data_type:
        case "shapenet":
            collator = MeshDataCollator(data_cfg, model_cfg)
        case "3d-front" | "3d-front-layout":
            collator = Front3DCollator(data_cfg, model_cfg)
        case _:
            raise ValueError(f"Unknown dataset type: {data_type}")
    return collator
