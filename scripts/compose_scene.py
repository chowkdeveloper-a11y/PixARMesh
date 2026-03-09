import argparse
import jsonlines
import trimesh
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.data.utils import make_3d_bbox
from src.data.vis import get_bbox_path
from src.utils.inference import recover_box_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="metadata/test_scene.jsonl")
    parser.add_argument("--pred-dir", type=str, required=True)
    parser.add_argument("--vis-bbox", action="store_true")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = pred_dir / "scenes"
    out_dir.mkdir(parents=True, exist_ok=True)

    y_up_matrix = np.diag(np.array([-1, -1, 1, 1], dtype=np.float32))

    with jsonlines.open(args.metadata, "r") as reader:
        valid_uids = {line["image_id"] for line in reader}

    valid_uids = sorted(list(valid_uids))

    red = np.array([255, 0, 0, 255], dtype=np.uint8)

    for uid in tqdm(valid_uids):
        all_objs = pred_dir.glob(f"{uid}_*.ply")
        scene = trimesh.Scene()
        for obj_path in all_objs:
            obj_idx = obj_path.stem.split("_")[-1]
            mesh = trimesh.load_mesh(obj_path)
            pose = np.load(pred_dir / f"{uid}_{obj_idx}_pose.npz")["pose"]
            transform = recover_box_transform(make_3d_bbox(*mesh.bounds), pose)[-1]
            mesh.apply_transform(y_up_matrix @ transform)

            scene.add_geometry(mesh, node_name=f"obj_{obj_idx}")

            if args.vis_bbox:
                bbox_path = get_bbox_path(pose, color=red)
                scene.add_geometry(bbox_path, node_name=f"bbox_{obj_idx}")

        scene.export(out_dir / f"{uid}.glb")


if __name__ == "__main__":
    main()
