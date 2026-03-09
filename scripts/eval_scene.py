from pathlib import Path
from tqdm import tqdm
import argparse
import json
import jsonlines
import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
import open3d as o3d
from accelerate import Accelerator
from src.data.utils import make_3d_bbox
from src.utils import evaluation
from src.utils.logging import get_logger


def transform_mesh_and_get_bbox(mesh_path, transform):
    import trimesh

    mesh = trimesh.load(mesh_path)
    # trimesh.Scene
    all_bboxes = []
    for geom in mesh.geometry.values():
        geom.apply_transform(transform)
        bbox = make_3d_bbox(*geom.bounds)
        all_bboxes.append(bbox)
    return np.stack(all_bboxes, axis=0)


def get_bbox_iou(gt_bboxes, pred_bboxes):
    from pytorch3d.ops import box3d_overlap
    from scipy.optimize import linear_sum_assignment

    intersection_vol, iou_3d = box3d_overlap(gt_bboxes, pred_bboxes)
    # M gt_bboxes, N pred_bboxes
    cost_matrix = -iou_3d.numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Sometimes there are more gt boxes than pred boxes
    # We assign 0 IoU to unmatched boxes
    iou_3d = iou_3d[row_ind, col_ind]
    mean_iou = iou_3d.sum() / len(gt_bboxes)
    return mean_iou.item()


def o3d_pcd_to_tensor(
    pcd: o3d.geometry.PointCloud, num_sample_points=None
) -> torch.Tensor:
    pts = np.asarray(pcd.points)
    if num_sample_points is not None:
        random_inds = np.random.choice(pts.shape[0], num_sample_points)
        pts = pts[random_inds]
    return torch.from_numpy(pts).float()


def get_ar_mesh(pred_dir, uid):
    return pred_dir / f"{uid}.glb"


@torch.no_grad()
def evaluate_single(logger, gt_pcd_dir, pred_dir, save_dir, uid, eval_iou):
    aligned_pcd_path = save_dir / f"{uid}.npz"
    if "_" in uid:
        real_uid = uid.split("_")[0]
    else:
        real_uid = uid
    gt_pts = np.load(gt_pcd_dir / f"{real_uid}.npz")["pcds"]
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_pts)
    gt_pts = torch.from_numpy(gt_pts).float()

    if not aligned_pcd_path.exists() or eval_iou:
        pred_mesh_path = get_ar_mesh(pred_dir, uid)
        pred_mesh = o3d.io.read_triangle_mesh(pred_mesh_path)
        try:
            pred_pcd = pred_mesh.sample_points_uniformly(10000)
        except:
            logger.warning(
                f"Error sampling points from mesh {pred_mesh_path}. Skipping {uid}."
            )
            return
        scale = (
            gt_pcd.get_minimal_oriented_bounding_box().volume()
            / pred_pcd.get_minimal_oriented_bounding_box().volume()
        ) ** (1 / 3)
        pred_pcd = pred_pcd.scale(
            scale,
            center=np.array((0.0, 0.0, 0.0)),
        )
        translate = gt_pcd.get_center() - pred_pcd.get_center()
        pred_pcd = pred_pcd.translate(translate)
        pre_transform = np.diag([scale, scale, scale, 1.0])
        pre_transform[:3, 3] = translate
        pre_transform = torch.from_numpy(pre_transform).float()
        pred_pts = o3d_pcd_to_tensor(pred_pcd)
        transform = evaluation.get_object_transformations(
            [pred_pts],
            [gt_pts],
            num_steps=200,
            verbose=False,
        ).squeeze(0)
        aligned_pts = evaluation.apply_transformation_matrix(pred_pts, transform)
        final_transform = transform @ pre_transform
        np.savez(aligned_pcd_path, pcds=aligned_pts.numpy())
    else:
        aligned_pts = np.load(aligned_pcd_path)["pcds"]
        aligned_pts = torch.from_numpy(aligned_pts).float()

    if eval_iou:
        pred_bboxes = transform_mesh_and_get_bbox(
            pred_mesh_path, final_transform.numpy()
        )
        gt_bboxes = np.load(f"datasets/ar-eval-gt-undecimated/bboxes/{real_uid}.npz")[
            "bboxes"
        ]
        pred_bboxes = torch.from_numpy(pred_bboxes).float()
        gt_bboxes = torch.from_numpy(gt_bboxes).float()
        iou = get_bbox_iou(gt_bboxes, pred_bboxes)
    else:
        iou = None

    aligned_pts_cuda = aligned_pts.unsqueeze(0).cuda()
    gt_pts_cuda = gt_pts.unsqueeze(0).cuda()
    cd = chamfer_distance(aligned_pts_cuda, gt_pts_cuda)[0].item()
    cd_s = chamfer_distance(aligned_pts_cuda, gt_pts_cuda, single_directional=True)[
        0
    ].item()
    f_score = evaluation.f_score(
        aligned_pts.numpy(), gt_pts.numpy(), tau=0.002
    )  # InstPIFu
    f_score_2 = evaluation.f_score(
        aligned_pts.numpy(), gt_pts.numpy(), tau=0.1
    )  # DeepPriorAssembly
    result_dict = {
        "cd": cd,
        "cd_s": cd_s,
        "f_score": float(f_score),
        "f_score_2": float(f_score_2),
    }
    if iou is not None:
        result_dict["bbox_iou"] = float(iou)
    with open(save_dir / f"{uid}.json", "w") as f:
        json.dump(result_dict, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="metadata/test_scene.jsonl")
    parser.add_argument(
        "--gt-pcd-dir", type=str, default="datasets/ar-eval-gt-undecimated/pcds"
    )
    parser.add_argument("--pred-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument(
        "--eval-iou",
        action="store_true",
    )
    args = parser.parse_args()

    eval_iou = args.eval_iou

    with jsonlines.open(args.metadata, "r") as reader:
        all_uids = [line["image_id"] for line in reader]

    accelerator = Accelerator()

    sharded_uids = all_uids[accelerator.process_index :: accelerator.num_processes]

    pred_dir = Path(args.pred_dir)
    save_dir = args.save_dir
    if not save_dir:
        save_dir = (
            "outputs/evaluations-scene" if not eval_iou else "outputs/evaluations-iou"
        )
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(save_dir / "eval.log")

    for uid in tqdm(sharded_uids):
        evaluate_single(
            logger,
            Path(args.gt_pcd_dir),
            pred_dir,
            save_dir,
            uid,
            eval_iou,
        )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Collate results
        all_cds = []
        all_cd_s = []
        all_f_scores = []
        all_f_scores_2 = []
        all_ious = []
        for uid in all_uids:
            record_path = save_dir / f"{uid}.json"
            if not record_path.exists():
                logger.warning(f"Skipping {uid}, no record found.")
                continue
            with record_path.open() as f:
                result_dict = json.load(f)
            all_cds.append(result_dict["cd"])
            all_cd_s.append(result_dict["cd_s"])
            all_f_scores.append(result_dict["f_score"])
            all_f_scores_2.append(result_dict["f_score_2"])
            if "bbox_iou" in result_dict:
                all_ious.append(result_dict["bbox_iou"])

        logger.info("======== Summary ========")
        logger.info(f"Method: ar")
        logger.info(f"Number of samples: {len(all_cds)}")
        logger.info(f"CD (10^{-3}): {1000 * np.mean(all_cds)}")
        logger.info(f"CD-S (10^{-3}): {1000 * np.mean(all_cd_s)}")
        logger.info(f"F-Score (tau=0.002): {np.mean(all_f_scores)}")
        logger.info(f"F-Score (tau=0.1): {np.mean(all_f_scores_2)}")
        if eval_iou:
            logger.info(f"BBox IoU: {np.mean(all_ious)}")
        logger.info("==========================")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
