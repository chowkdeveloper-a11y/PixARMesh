import os
import numpy as np
import trimesh
from scipy.ndimage import binary_erosion
from sklearn.linear_model import RANSACRegressor, LinearRegression
from PIL import Image

try:
    import open3d as o3d

    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False


def color_to_id(img):
    img = img.astype(np.uint32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 256 * 256 * r + 256 * g + b


def get_masks_by_ids(pan_seg_gt, inst_id_list, erode_size=0):
    pan_seg_gt = np.array(pan_seg_gt)
    pan_seg_gt = color_to_id(pan_seg_gt)
    _, inst_seg = pan_seg_gt // 1000, pan_seg_gt % 1000
    masks = inst_seg[..., None] == np.array(inst_id_list)
    masks = masks.transpose(2, 0, 1)
    if erode_size > 0:
        struct = np.ones((1, erode_size, erode_size), dtype=bool)
        masks = binary_erosion(masks, structure=struct)
    return masks


def align_depth(relative_depth, metric_depth, mask=None, min_samples=0.2):
    regressor = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=True), min_samples=min_samples
    )
    if mask is not None:
        regressor.fit(
            relative_depth[mask].reshape(-1, 1), metric_depth[mask].reshape(-1, 1)
        )
    else:
        regressor.fit(relative_depth.reshape(-1, 1), metric_depth.reshape(-1, 1))
    depth = regressor.predict(relative_depth.reshape(-1, 1)).reshape(
        relative_depth.shape
    )
    return depth


def read_depth_pro_depth(depth_path, scene_id):
    dp_depth = Image.open(os.path.join(depth_path, f"{scene_id}.png"))
    dp_depth = np.array(dp_depth) / 65535 * 20
    return dp_depth.astype(np.float32)


def augment_matrix(mat):
    aug_mat = np.eye(4, dtype=mat.dtype)
    aug_mat[:3, :3] = mat
    return aug_mat


def get_rotation_y_matrix(azimuth):
    c = np.cos(azimuth)
    s = np.sin(azimuth)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def random_sample_point_clouds(points, num_samples, return_inds=False):
    random_inds = np.random.choice(points.shape[0], num_samples)
    pts = points[random_inds]
    if return_inds:
        return pts, random_inds
    return pts


def estimate_point_cloud_normals(points, knn=30):
    if not _HAS_O3D:
        return np.concatenate([points, np.zeros_like(points)], axis=1)
    canonical_pcd = o3d.geometry.PointCloud()
    canonical_pcd.points = o3d.utility.Vector3dVector(points)
    canonical_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
    )
    return np.concatenate(
        [np.array(canonical_pcd.points), np.array(canonical_pcd.normals)], axis=1
    )


def back_project_depth(depth, K_inv, return_pix_coords=False):
    H, W = depth.shape
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    pos_pix_2d = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    ones = np.ones_like(pos_pix_2d[..., 0:1], dtype=np.float32)
    pos_pix = np.concatenate([pos_pix_2d, ones], axis=-1)
    pos_pix_flat = pos_pix.reshape(-1, 3)
    pos_cam = pos_pix_flat @ K_inv.T
    pos_cam = pos_cam.reshape(H, W, 3)
    pos_cam = pos_cam / pos_cam[..., 2:3]
    pos_cam = pos_cam * depth[..., np.newaxis]
    if return_pix_coords:
        return pos_cam, pos_pix_2d
    return pos_cam


def augment_transformation_matrix(matrix):
    size = matrix.shape[-1]
    augmented_matrix = matrix.new_zeros(*matrix.shape[:-2], size + 1, size + 1)
    augmented_matrix[..., :size, :size] = matrix
    augmented_matrix[..., -1, -1] = 1.0
    return augmented_matrix


def transform_3d_points(points_3d, transform):
    points_3d_h = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)
    transformed_points = points_3d_h @ transform.T
    return transformed_points[:, :3]


def sample_point_cloud(vertices, faces, num_points, with_normals):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points, face_idx = mesh.sample(num_points, return_index=True)
    if with_normals:
        normals = mesh.face_normals[face_idx]
        points = np.concatenate([points, normals], axis=1)
    return points


def normalize_vertices(vertices, bound=0.95, return_all=False):
    vmin, vmax = vertices.min(0), vertices.max(0)
    center = (vmin + vmax) / 2
    vertices = vertices - center
    scale = 2 * bound / (vmax - vmin).max()
    vertices = vertices * scale
    if return_all:
        return vertices, center, scale
    return vertices


def normalize_bboxes_with_point_clouds(
    bboxes, point_clouds, bound=0.95, return_matrix=False
):
    all_points = bboxes.reshape(-1, 3)
    all_points = np.concatenate([all_points, point_clouds], axis=0)
    _, center, scale = normalize_vertices(all_points, bound=bound, return_all=True)
    norm_bboxes = bboxes - center[None, None, :]
    norm_bboxes = norm_bboxes * scale
    norm_pcs = point_clouds - center[None, :]
    norm_pcs = norm_pcs * scale
    if return_matrix:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = -center
        matrix[:3, :] = matrix[:3, :] * scale
        return norm_bboxes, norm_pcs, matrix
    return norm_bboxes, norm_pcs


def random_shift_bboxes_with_point_clouds(
    bboxes, point_clouds, max_shift, bound=0.95, return_matrix=False
):
    shift = np.random.uniform(-max_shift, max_shift, size=(3,))
    all_points = bboxes.reshape(-1, 3)
    all_points = np.concatenate([all_points, point_clouds], axis=0)
    vmin, vmax = all_points.min(0), all_points.max(0)
    shift = np.clip(shift, -bound - vmin, bound - vmax)
    norm_bboxes = bboxes + shift[None, None, :]
    norm_pcs = point_clouds + shift[None, :]
    if return_matrix:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = shift
        return norm_bboxes, norm_pcs, matrix
    return norm_bboxes, norm_pcs


def quantize_points(normalized_points, num_pos_tokens):
    # [-1, 1] -> [0, 1]
    normalized_points = (normalized_points + 1) / 2
    # error bound - 1 / (2 * num_pos_tokens)
    return (
        np.floor(normalized_points * num_pos_tokens)
        .clip(0, num_pos_tokens - 1)
        .astype(np.int32)
    )


def dequantize_points(quantized_points, num_pos_tokens):
    normalized = (quantized_points + 0.5) / num_pos_tokens
    return normalized * 2 - 1


def quantize_mesh(normalized_vertices, faces, num_pos_tokens):
    quantized_vertices = quantize_points(normalized_vertices, num_pos_tokens)
    # Y-up to Z-up
    quantized_vertices = quantized_vertices[:, [2, 0, 1]]
    # ZYX sort
    sort_inds = np.lexsort(quantized_vertices.T)
    sorted_vertices = quantized_vertices[sort_inds]
    inv_inds = np.argsort(sort_inds)
    faces = inv_inds[faces]
    # sort within faces
    start_inds = faces.argmin(axis=1)  # [M]
    all_inds = start_inds[:, None] + np.arange(3)[None, :]
    all_inds = all_inds % 3
    faces = np.take_along_axis(faces, all_inds, axis=1)  # [M, 3]
    # sort among faces
    face_sort_inds = np.lexsort(faces[:, ::-1].T)
    sorted_faces = faces[face_sort_inds]
    # Z-up back to Y-up
    sorted_vertices = sorted_vertices[:, [1, 2, 0]]
    return sorted_vertices, sorted_faces


def make_3d_bbox(min_corner, max_corner):
    x_min, y_min, z_min = min_corner
    x_max, y_max, z_max = max_corner
    points_3d = np.array(
        [
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
        ]
    )
    return points_3d


def decode_gravity_aligned_bbox(encoded_bboxes):
    """
          v4_____________________v5
          /|                    /|
         / |                   / |
        /  |                  /  |
       /___|_________________/   |
    v0|    |                 |v1 |
      |    |                 |   |
      |    |                 |   |
      |    |                 |   |
      |    |_________________|___|
      |   / v7               |   /v6
      |  /                   |  /
      | /                    | /
      |/_____________________|/
      v3                     v2
    X <-----  x Z (into the screen)
    """
    # encoded_bboxes: [N, 3, 3]
    v0 = encoded_bboxes[:, 0, :]
    v2 = encoded_bboxes[:, 1, :]
    v5 = encoded_bboxes[:, 2, :]
    y_min = v2[:, 1]
    y_max = v0[:, 1]
    # v0, v5 are on XZ plane
    v1 = np.stack([v2[:, 0], y_max, v2[:, 2]], axis=-1)
    v4 = v0 + v5 - v1
    v3 = np.stack([v0[:, 0], y_min, v0[:, 2]], axis=-1)
    v6 = np.stack([v5[:, 0], y_min, v5[:, 2]], axis=-1)
    v7 = np.stack([v4[:, 0], y_min, v4[:, 2]], axis=-1)
    return np.stack([v0, v1, v2, v3, v4, v5, v6, v7], axis=1)


def quantize_gravity_aligned_bboxes(
    normalized_bboxes,
    num_pos_tokens,
    return_sort_inds=True,
):
    # all_bboxes: [N, 8, 3]
    centers = normalized_bboxes.mean(axis=1)
    quantized = quantize_points(normalized_bboxes, num_pos_tokens)
    sort_inds = np.lexsort(centers.T)
    sorted_quantized = quantized[sort_inds]
    if return_sort_inds:
        return sorted_quantized, sort_inds
    return sorted_quantized
