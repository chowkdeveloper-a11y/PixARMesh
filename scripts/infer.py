import os

os.environ["OMP_NUM_THREADS"] = "1"
import warnings

warnings.filterwarnings("ignore")

import argparse
import torch
import numpy as np
from tqdm import tqdm
from accelerate import PartialState
from pathlib import Path
from transformers import set_seed, AutoImageProcessor
from src.utils.inference import (
    prepare_model_for_inference,
    get_prefix_allowed_tokens_fn_edgerunner,
    decode_mesh_edgerunner,
    decode_bpt,
    prepare_test_set,
    joint_filter,
)
from src.data.collator import get_mesh_data_collator
from src.data import utils as data_utils, tokenize_bpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (per GPU) for inference",
    )
    parser.add_argument(
        "--run-type",
        type=str,
        choices=["obj", "scene"],
        required=True,
        help="Whether to run inference on object level or scene level",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["edgerunner", "bpt"],
        required=True,
        help="Type of the model to use for inference",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to Huggingface checkpoint",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--gt-layout",
        action="store_true",
        help="Whether to use ground-truth layout for inference",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Whether to use sampling during inference",
    )
    parser.add_argument(
        "--gt-depth",
        action="store_true",
        help="Whether to use ground-truth depth maps",
    )
    parser.add_argument(
        "--gt-mask",
        action="store_true",
        help="Whether to use ground-truth masks",
    )
    args = parser.parse_args()

    is_bpt = args.model_type == "bpt"

    is_obj_level = args.run_type == "obj"
    metadata_name = "test_obj_sub_100" if is_obj_level else "test_scene"
    metadata_file = f"metadata/{metadata_name}.jsonl"

    if is_obj_level:
        use_gt_layout = True
        use_gt_mask = True
    else:
        use_gt_layout = args.gt_layout
        use_gt_mask = args.gt_mask

    use_gt_depth = args.gt_depth

    if not use_gt_mask:
        # We don't have GT layout if using predicted masks
        use_gt_layout = False

    use_gt_layout_str = "gt_layout" if use_gt_layout else "pred_layout"
    use_gt_mask_str = "gt_mask" if use_gt_mask else "pred_mask"
    use_gt_depth_str = "gt_depth" if use_gt_depth else "pred_depth"
    name = f"{use_gt_layout_str}_{use_gt_mask_str}_{use_gt_depth_str}"
    out_dir = Path(args.output_dir) / args.run_type / args.model_type / name
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    state = PartialState()

    device = state.device
    model, model_cfg, data_cfg = prepare_model_for_inference(is_bpt, args.checkpoint)
    model.to(device)
    model.compile()

    data_cfg.use_predicted_depth = not use_gt_depth
    if not use_gt_depth:
        data_cfg.depth_path = "datasets/depth_pro_aligned_npy"
        data_cfg.predicted_depth_aligned = True

    image_preprocessor = AutoImageProcessor.from_pretrained(
        data_cfg.image_preprocessor, size_divisor=data_cfg.image_size_divisor
    )

    with state.local_main_process_first():
        test_set = prepare_test_set(
            data_cfg, metadata_file, use_predicted_mask=not use_gt_mask
        )

    sharded_data = test_set.shard(state.num_processes, state.process_index)

    collator = get_mesh_data_collator(data_cfg, model_cfg)
    cond_prefix = [collator.pc_token_id] * collator.prefix_len
    bos_prefix = [collator.bos_token_id] if not is_bpt else []

    n_iters = (len(sharded_data) + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for item in tqdm(
                sharded_data.iter(batch_size=args.batch_size),
                total=n_iters,
                position=state.process_index,
                dynamic_ncols=True,
                leave=False,
            ):
                all_pcds = np.array(item["point_clouds"], dtype=np.float32)
                all_pcds_2d = np.array(item["point_clouds_2d"], dtype=np.float32)
                sampled_pcds = []
                sampled_pcds_2d = []
                sampled_ctx_pcds = []
                sampled_ctx_pcds_2d = []
                all_input_ids = []
                all_uids = []
                images = []

                gt_pose_seqs = []

                for uid, pcds, pcds_2d, bboxes, mask, transform, image in zip(
                    item["uid"],
                    all_pcds,
                    all_pcds_2d,
                    item["bboxes"],
                    item["mask"],
                    item["transform"],
                    item["image"],
                ):
                    if (out_dir / f"{uid}.ply").exists():
                        continue

                    mask = np.array(mask)

                    all_uids.append(uid)
                    obj_index = int(uid.split("_")[-1])
                    bboxes = np.array(bboxes, dtype=np.float32)
                    obj_pcd_in_global = pcds[mask]
                    obj_pcd_2d = pcds_2d[mask]
                    sampled_pcd, sample_inds = data_utils.random_sample_point_clouds(
                        obj_pcd_in_global,
                        data_cfg.num_points,
                        return_inds=True,
                    )
                    if data_cfg.with_normals:
                        sampled_pcd = data_utils.estimate_point_cloud_normals(
                            sampled_pcd
                        )
                    sampled_ctx_pcd, sampled_ctx_inds = (
                        data_utils.random_sample_point_clouds(
                            pcds.reshape(-1, 3),
                            data_cfg.num_ctx_points,
                            return_inds=True,
                        )
                    )
                    if data_cfg.with_normals:
                        sampled_ctx_pcd = data_utils.estimate_point_cloud_normals(
                            sampled_ctx_pcd
                        )
                    sampled_ctx_pcd_2d = pcds_2d.reshape(-1, 2)[sampled_ctx_inds]
                    sampled_ctx_pcds.append(sampled_ctx_pcd)
                    sampled_ctx_pcds_2d.append(sampled_ctx_pcd_2d)
                    sampled_pcds.append(sampled_pcd)
                    sampled_pcds_2d.append(obj_pcd_2d[sample_inds])
                    gt_pose_seq = collator._tokenize_bbox(bboxes[[obj_index]])
                    gt_pose_seqs.append(gt_pose_seq)
                    if use_gt_layout:
                        layout_seq = gt_pose_seq + [collator.indicator_token_id]
                    else:
                        layout_seq = []

                    prefix_seq = cond_prefix + bos_prefix + layout_seq
                    all_input_ids.append(prefix_seq)

                    images.append(np.array(image, dtype=np.uint8))

                if len(all_input_ids) == 0:
                    continue

                sampled_pcds = np.stack(sampled_pcds, axis=0)
                sampled_pcds_2d = np.stack(sampled_pcds_2d, axis=0)
                all_input_ids = np.stack(all_input_ids, axis=0)
                all_input_ids = torch.as_tensor(all_input_ids, dtype=torch.long).to(
                    device
                )
                cond_pcs = torch.as_tensor(sampled_pcds, dtype=torch.float32).to(device)
                cond_pcs_2d = torch.as_tensor(sampled_pcds_2d, dtype=torch.float32).to(
                    device
                )
                ctx_pcs = torch.as_tensor(
                    np.stack(sampled_ctx_pcds, axis=0), dtype=torch.float32
                ).to(device)
                ctx_pcs_2d = torch.as_tensor(
                    np.stack(sampled_ctx_pcds_2d, axis=0), dtype=torch.float32
                ).to(device)

                gt_pose_seqs = np.array(gt_pose_seqs)

                cond_images = image_preprocessor(images, return_tensors="pt")[
                    "pixel_values"
                ].to(device)
                bs = all_input_ids.shape[0]

                if not use_gt_layout:
                    # Decode layout first
                    inputs_embeds = model.get_inputs_with_cond(
                        all_input_ids,
                        cond_pcs=cond_pcs,
                        cond_pcs_2d=cond_pcs_2d,
                        ctx_pcs=ctx_pcs,
                        ctx_pcs_2d=ctx_pcs_2d,
                        pixel_values=cond_images,
                    )
                    if is_bpt:
                        results = model.generate(
                            cond_embeds=inputs_embeds,
                            batch_size=bs,
                            max_new_tokens=17,
                            do_sample=False,
                        )
                        results_cpu = results.cpu().numpy()
                        poses = tokenize_bpt.detokenize_layout(
                            results_cpu[:, :-1]
                        ).reshape(-1, 8, 3)
                    else:
                        results = model.generate(
                            inputs_embeds=inputs_embeds,
                            max_new_tokens=25,
                            use_cache=True,
                            do_sample=False,
                        )
                        results_cpu = results.cpu().numpy()
                        poses = (
                            results_cpu[:, :-1] - model_cfg.pos_token_offset
                        ).reshape(-1, 8, 3)
                    poses = data_utils.dequantize_points(
                        poses, model_cfg.num_pos_tokens
                    )
                    all_input_ids = torch.cat([all_input_ids, results], dim=1)
                    for uid, tokens, pose in zip(all_uids, results_cpu, poses):
                        np.savez_compressed(
                            out_dir / f"{uid}_pred_layout.npz", tokens=tokens
                        )
                        np.savez_compressed(out_dir / f"{uid}_pose.npz", pose=pose)

                inputs_embeds = model.get_inputs_with_cond(
                    all_input_ids,
                    cond_pcs=cond_pcs,
                    cond_pcs_2d=cond_pcs_2d,
                    ctx_pcs=ctx_pcs,
                    ctx_pcs_2d=ctx_pcs_2d,
                    pixel_values=cond_images,
                )

                seq_len = all_input_ids.shape[1]
                max_new_tokens = min(collator.max_seq_length - seq_len, 40960)

                if is_bpt:
                    results = model.generate(
                        inputs=all_input_ids,
                        cond_embeds=inputs_embeds,
                        max_new_tokens=max_new_tokens,
                        temperature=0.5,
                        filter_logits_fn=joint_filter,
                        filter_kwargs=dict(k=50, p=0.95),
                        do_sample=args.do_sample,
                        tqdm_position=state.process_index,
                    )
                    results = results[:, seq_len:]
                else:
                    extra_args = {"do_sample": False}
                    if args.do_sample:
                        extra_args = {
                            "do_sample": True,
                            "top_k": 10,
                        }
                    results = model.generate(
                        inputs_embeds=inputs_embeds,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        prefix_allowed_tokens_fn=get_prefix_allowed_tokens_fn_edgerunner(
                            model,
                            batch_size=bs,
                        ),
                        **extra_args,
                    )
                results = results.cpu().numpy()
                for uid, tokens in zip(all_uids, results):
                    # remove padding
                    tokens = tokens[tokens != collator.pad_token_id]
                    eos_idx = (tokens == model.config.eos_token_id).nonzero()[0]
                    if len(eos_idx) > 0:
                        tokens = tokens[: eos_idx[0]]
                    if is_bpt:
                        mesh = decode_bpt(tokens)
                    else:
                        mesh = decode_mesh_edgerunner(
                            tokens, collator.tokenizer, clean=True, verbose=False
                        )
                    mesh.export(out_dir / f"{uid}.ply")


if __name__ == "__main__":
    main()
