import torch
import torch.nn as nn
import scipy
import numpy as np
from pytorch3d.transforms import euler_angles_to_matrix
from pytorch3d.loss import chamfer_distance
from tqdm import trange
from src.data.utils import augment_transformation_matrix


class PoseEstimator(nn.Module):
    def __init__(self, num_repeats, num_instances, dof=7, enable_global_rotation=False):
        super().__init__()
        assert dof in (5, 7), "DoF must be either 5 or 7"

        if dof == 5:
            self.rotation_y = nn.Parameter(
                torch.zeros(num_repeats, num_instances), requires_grad=True
            )
        else:
            self.rotation = nn.Parameter(
                torch.zeros(num_repeats, num_instances, 3), requires_grad=True
            )
        self.translation = nn.Parameter(
            torch.zeros(num_repeats, num_instances, 3), requires_grad=True
        )
        self.scale = nn.Parameter(
            torch.ones(num_repeats, num_instances), requires_grad=True
        )
        if enable_global_rotation:
            self.global_rotation = nn.Parameter(
                torch.zeros(num_repeats, 3), requires_grad=True
            )
        else:
            self.global_rotation = None
        self.dof = dof
        self.num_repeats = num_repeats
        self.num_instances = num_instances

    @property
    def eulers(self):
        if self.dof == 5:
            angles = self.rotation_y.new_zeros(self.num_repeats, self.num_instances, 3)
            angles[..., 1] = self.rotation_y
            return angles
        else:
            return self.rotation

    @torch.no_grad()
    def get_transformation_matrix(self):
        return self._get_transformation_matrix()

    def _get_transformation_matrix(self):
        result = euler_angles_to_matrix(self.eulers, "XYZ")
        result = augment_transformation_matrix(result)
        result[..., :3, 3] = self.translation
        result[..., :3, :] *= self.scale[:, :, None, None]
        if self.global_rotation is not None:
            global_rot_matrix = euler_angles_to_matrix(self.global_rotation, "XYZ")
            global_rot_matrix = augment_transformation_matrix(global_rot_matrix)
            global_rot_matrix = global_rot_matrix[:, None, :, :]
            result = torch.matmul(global_rot_matrix, result)
        return result

    def get_cd_loss(self, transformed_points, target_points, single_directional=False):
        # We may want to use single-directional if target_points are from partial shape
        point_dim = transformed_points.shape[-1]
        if target_points.ndim == 3:
            target_points = target_points[None].repeat(self.num_repeats, 1, 1, 1)
        num_points = transformed_points.shape[-2]
        cd_loss = chamfer_distance(
            target_points.reshape(-1, num_points, point_dim),
            transformed_points.reshape(-1, num_points, point_dim),
            batch_reduction=None,
            point_reduction="mean",
            single_directional=single_directional,
        )[0]
        cd_loss = cd_loss.reshape(self.num_repeats, self.num_instances)
        return cd_loss

    def forward(self, source_points, target_points=None):
        # Num repeats: B
        # Source / Target points: [N (num_insts), K (num_points), 3]
        # [N, K, 3] x [B, N, 3, 3] -> [B, N, K, 3] + [B, N, 1, 3]
        augmented_points = torch.cat(
            [source_points, torch.ones_like(source_points[..., :1])], dim=-1
        )
        transform = self._get_transformation_matrix()
        transformed_points = torch.matmul(augmented_points, transform.transpose(-1, -2))
        transformed_points = transformed_points[..., :3]
        if target_points is None:
            return transformed_points
        return self.get_cd_loss(transformed_points, target_points)


def random_rotation_matrix_y(num_repeats, num_instances, device):
    rot_y = (
        torch.rand((num_repeats, num_instances), device=device, dtype=torch.float32)
        * 2
        * torch.pi
    )
    rot = rot_y.new_zeros(num_repeats, num_instances, 3)
    rot[..., 1] = rot_y
    return euler_angles_to_matrix(rot, "XYZ")


@torch.enable_grad()
def get_object_transformations(
    source_list,
    target_list,
    num_repeats=20,
    num_steps=50,
    lr=5e-2,
    verbose=False,
    dof=7,
):
    device = "cuda"
    source_pts = torch.stack(source_list).to(device)
    target_pts = torch.stack(target_list).to(device)
    num_instances = len(source_list)
    estimator = PoseEstimator(
        num_repeats=num_repeats,
        num_instances=num_instances,
        dof=dof,
    )
    estimator = estimator.to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)

    last_loss = None

    random_rotation = random_rotation_matrix_y(num_repeats, num_instances, device)
    transformed_source_pts = torch.matmul(source_pts, random_rotation.transpose(-1, -2))

    pbar = trange if verbose else range
    for _ in pbar(num_steps):
        optimizer.zero_grad()
        loss = estimator(transformed_source_pts, target_pts)
        last_loss = loss.detach()
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        if verbose:
            print(f"Step {_}: Best loss: {last_loss.min().item():.4f}")

    random_rotation = augment_transformation_matrix(random_rotation)
    all_transforms = estimator.get_transformation_matrix()
    all_transforms = torch.matmul(all_transforms, random_rotation)
    best_inds = torch.argmin(last_loss, dim=0)
    best_transforms = all_transforms[best_inds, torch.arange(num_instances)]
    return best_transforms.cpu()


def sample_points_from_o3d_mesh(mesh, num_points, sample_method="uniform"):
    if sample_method not in ["uniform", "poisson"]:
        raise ValueError(f"Unknown sampling method: {sample_method}")
    if sample_method == "uniform":
        points = mesh.sample_points_uniformly(num_points).points
    elif sample_method == "poisson":
        points = mesh.sample_points_poisson_disk(num_points).points
    return torch.from_numpy(np.asarray(points)).float()


def get_normalized_pcd(pcd: torch.Tensor):
    # Normalize via bounding box
    vmax = pcd.max(dim=-2, keepdim=True).values
    vmin = pcd.min(dim=-2, keepdim=True).values
    cube_size = (vmax - vmin).max(dim=-1, keepdim=True).values
    cube_center = (vmin + vmax) / 2
    pcd = (pcd - cube_center) / cube_size
    return pcd


def apply_transformation_matrix(
    points: torch.Tensor, transformation_matrix: torch.Tensor
) -> torch.Tensor:
    if points.shape[-1] == 3:
        points = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
    points = torch.matmul(points, transformation_matrix.transpose(0, 1))
    points = points[..., :3] / points[..., 3:]
    return points


def pointcloud_neighbor_distances_indices(source_points, target_points):
    target_kdtree = scipy.spatial.cKDTree(target_points)
    distances, indices = target_kdtree.query(source_points)
    return distances, indices


def percent_below(dists, thresh):
    return np.mean((dists**2 <= thresh).astype(np.float32)) * 100.0


def _f_score(a_to_b, b_to_a, thresh):
    OCCNET_FSCORE_EPS = 1e-09
    precision = percent_below(a_to_b, thresh)
    recall = percent_below(b_to_a, thresh)

    return (2 * precision * recall) / (precision + recall + OCCNET_FSCORE_EPS)


def f_score(points1, points2, tau=0.002):
    """Computes the F-Score at tau between two meshes."""
    dist12, _ = pointcloud_neighbor_distances_indices(points1, points2)
    dist21, _ = pointcloud_neighbor_distances_indices(points2, points1)
    f_score_tau = _f_score(dist12, dist21, tau)
    return f_score_tau
