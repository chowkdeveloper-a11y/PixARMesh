import trimesh
import numpy as np


_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),  # front face
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),  # back face
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # side edges
]


def get_bbox_path(bbox_corners: np.ndarray, color=None):
    segments = bbox_corners[_EDGES]
    path = trimesh.load_path(segments)
    if color is not None:
        path.colors = np.tile(color, (len(path.entities), 1))
    return path


def visualize_pcs_and_bboxes(
    pts: np.ndarray,
    cols: np.ndarray | None,
    bbox_corners: np.ndarray,  # (N, 8, 3)
    out_path: str | None = None,
):
    scene = trimesh.Scene()
    if cols is None:
        # Green by default
        cols = np.array([0, 255, 0, 255], dtype=np.uint8)
        cols = np.tile(cols, (len(pts), 1))
    pc = trimesh.PointCloud(pts, colors=cols)
    scene.add_geometry(pc, node_name="depth_pc")

    red = np.array([255, 0, 0, 255], dtype=np.uint8)

    for i in range(bbox_corners.shape[0]):
        path = get_bbox_path(bbox_corners[i], color=red)
        scene.add_geometry(path, node_name=f"bbox_{i}")

    if out_path is not None:
        scene.export(out_path)
    return scene


def visualize_obj_and_pcs(vertices, faces, pts):
    scene = trimesh.Scene()
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    scene.add_geometry(mesh, node_name="mesh")

    pc = trimesh.PointCloud(pts, colors=np.array([0, 255, 0, 255], dtype=np.uint8))
    scene.add_geometry(pc, node_name="pc")

    return scene
