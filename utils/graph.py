import numpy as np
import copy
import open3d as o3d
import torch
from voxel_hash_map.voxelize import compute_graph_distances

class Graph:
    def __init__(self, sparse_tensor, knn):
        self.knn = knn
        self.build_graph_from_point_cloud(sparse_tensor)

    def build_graph_from_point_cloud(self, pnts): #sparse_tensor):
        #pnts = np.asarray(sparse_tensor.cpu()[:, :3])
        self.num_vertices = pnts.shape[0]
        try:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pnts))
        except:
            pnts = np.asarray(pnts.cpu()[:, :3])
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pnts))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        self.laplacian_matrix = np.full(
            (self.num_vertices, self.num_vertices), np.inf)
        for i, p in enumerate(pnts):
            _, idx_neighbors, dist_neighbors = pcd_tree.search_hybrid_vector_3d(p, 0.05, 7)
            self.laplacian_matrix[i, idx_neighbors] = np.asarray(dist_neighbors)
        compute_graph_distances(self.laplacian_matrix)

