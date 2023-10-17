import numpy as np
import open3d as o3d

import voxel_hash_map_pybind


def voxel_down_sample(points: np.ndarray, voxel_size: float):
    _points = voxel_hash_map_pybind._Vector3dVector(points)
    voxels, indexes = voxel_hash_map_pybind._voxel_down_sample(
        _points, voxel_size)
    return np.asarray(voxels), indexes


def compute_graph_distances(laplacian: np.ndarray):
    voxel_hash_map_pybind._compute_graph_distances(laplacian)
