import argparse
import jsonlines
import datasets
import open3d as o3d
import numpy as np
import json
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from src.utils import evaluation
from src.data.utils import get_masks_by_ids


def flatten_3d_front_for_eval(examples, erode_size=0):
    result_scene_ids = []
    result_obj_ids = []
    result_model_ids = []
    result_mask_areas = []
    all_uids = examples["uid"]
    all_objects = examples["objects"]
    all_masks = examples["panoptic_mask"]
    for scene_uid, objs, pano_mask in zip(all_uids, all_objects, all_masks):
        model_ids = objs["model_ids"]
        inst_ids = objs["inst_ids"]
        n_insts = len(model_ids)
        obj_masks = get_masks_by_ids(
            pano_mask,
            inst_ids,
            erode_size=erode_size,
        )
        mask_area = obj_masks.sum(axis=(1, 2))
        result_scene_ids.extend([scene_uid] * n_insts)
        result_obj_ids.extend(list(range(n_insts)))
        result_model_ids.extend(model_ids)
        result_mask_areas.extend(mask_area.tolist())
    return {
        "uid": result_scene_ids,
        "obj_id": result_obj_ids,
        "model_id": result_model_ids,
        "mask_area": result_mask_areas,
    }


def get_mesh(mesh_path: Path):
    if mesh_path.exists():
        return o3d.io.read_triangle_mesh(str(mesh_path))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="datasets/3d-front-ar-packed")
    parser.add_argument(
        "--metadata", type=str, default="metadata/test_obj_sub_100.jsonl"
    )
    parser.add_argument("--gt-dir", type=str, default="datasets/3D-FUTURE-model-ply")
    parser.add_argument("--pred-dir", type=str, required=True)
    parser.add_argument(
        "--num-sample-points",
        type=int,
        default=10000,
        help="Number of points to sample from each mesh for evaluation",
    )
    parser.add_argument(
        "--mask-area-thresh",
        type=int,
        default=1600,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing eval results",
    )
    parser.add_argument("--save-dir", type=str, default="outputs/evaluations-obj")
    args = parser.parse_args()

    with jsonlines.open(args.metadata, "r") as reader:
        valid_uids = {line["image_id"] for line in reader}

    accelerator = Accelerator()

    with accelerator.local_main_process_first():
        dataset = datasets.load_dataset(args.dataset, split="test", num_proc=16)

    all_uids = dataset["uid"]
    valid_indices = []
    for i, uid in enumerate(all_uids):
        if str(uid) in valid_uids:
            valid_indices.append(i)

    subset = dataset.select(valid_indices)

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with accelerator.local_main_process_first():
        subset = subset.map(
            flatten_3d_front_for_eval,
            batched=True,
            batch_size=4,
            remove_columns=subset.column_names,
        )

    sharded_subset = subset.shard(
        num_shards=accelerator.num_processes,
        index=accelerator.process_index,
    )

    for item in tqdm(sharded_subset):
        uid = item["uid"]
        obj_id = item["obj_id"]
        model_id = item["model_id"]
        mask_area = item["mask_area"]
        if mask_area < args.mask_area_thresh:
            continue

        gt_mesh_path = gt_dir / f"{model_id}.ply"
        pred_mesh_path = pred_dir / f"{uid}_{obj_id}.ply"
        out_json_path = save_dir / f"{uid}_{obj_id}.json"

        if out_json_path.exists() and not args.overwrite:
            continue

        gt_mesh = get_mesh(gt_mesh_path)
        pred_mesh = get_mesh(pred_mesh_path)
        has_gt = gt_mesh is not None
        has_pred = pred_mesh is not None

        record = {
            "uid": uid,
            "obj_id": obj_id,
            "model_id": model_id,
            "has_gt": has_gt,
            "has_pred": has_pred,
            "cd": None,
            "f_score": None,
        }

        if has_gt and has_pred:
            gt_pcds = evaluation.sample_points_from_o3d_mesh(gt_mesh, 5000)
            pred_pcds = evaluation.sample_points_from_o3d_mesh(pred_mesh, 5000)
            gt_pcds = evaluation.get_normalized_pcd(gt_pcds)
            pred_pcds = evaluation.get_normalized_pcd(pred_pcds)
            transform_matrices = evaluation.get_object_transformations(
                [pred_pcds], [gt_pcds]
            )
            gt_pcds = evaluation.sample_points_from_o3d_mesh(
                gt_mesh, args.num_sample_points
            )
            pred_pcds = evaluation.sample_points_from_o3d_mesh(
                pred_mesh, args.num_sample_points
            )
            gt_pcds = evaluation.get_normalized_pcd(gt_pcds)
            pred_pcds = evaluation.get_normalized_pcd(pred_pcds)

            transformed_pred_pcds = evaluation.apply_transformation_matrix(
                pred_pcds,
                transform_matrices[0],
            )
            cd_loss = chamfer_distance(
                gt_pcds.unsqueeze(0).cuda(),
                transformed_pred_pcds.unsqueeze(0).cuda(),
            )[0].item()
            f_score = evaluation.f_score(gt_pcds.numpy(), transformed_pred_pcds.numpy())
            record["cd"] = float(cd_loss)
            record["f_score"] = float(f_score)

        with out_json_path.open("w") as f:
            json.dump(record, f)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        results = []
        all_cds = []
        all_f_scores = []
        for item in subset:
            uid = item["uid"]
            obj_id = item["obj_id"]
            out_json_path = save_dir / f"{uid}_{obj_id}.json"
            mask_area = item["mask_area"]
            if mask_area < args.mask_area_thresh:
                continue
            with out_json_path.open("r") as f:
                record = json.load(f)
            if record["cd"] is not None:
                all_cds.append(record["cd"])
                all_f_scores.append(record["f_score"])
            results.append(record)

        avg_cd = float(np.mean(all_cds))
        avg_f_scores = float(np.mean(all_f_scores))
        results.append(
            {
                "avg_cd": avg_cd,
                "avg_f_score": avg_f_scores,
                "num_evaluated": len(all_cds),
            }
        )
        results_path = save_dir / "eval_obj_results.jsonl"
        with jsonlines.open(results_path, "w") as writer:
            writer.write_all(results)
        print(
            f"""
Evaluation results saved to {results_path}.
Num valid objects: {len(all_cds)}
Average Chamfer Distance (x10^{-3}): {avg_cd * 1000:.3f}
Average F-Score (%): {avg_f_scores:.3f}
"""
        )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
