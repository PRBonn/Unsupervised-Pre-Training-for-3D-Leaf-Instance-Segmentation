#include "voxel_hash_map.h"

#include <algorithm>
#include <tuple>

namespace voxel_hash_map {

std::tuple<std::vector<Eigen::Vector3d>, std::vector<int>>
voxel_down_sample(const std::vector<Eigen::Vector3d> &frame,
                  double voxel_size) {
  VoxelHashMap grid(voxel_size);
  grid.AddPoints(frame);
  return grid.Pointcloud();
}

void VoxelHashMap::AddPoint(const Eigen::Vector3d &point, const int index) {
  auto voxel = Voxel(point, voxel_size_);
  auto search = map_.find(voxel);
  if (search != map_.end()) {
    auto &voxel_block = search->second;
    voxel_block.add_point(point);
  } else {
    map_.insert({voxel, VoxelBlock(point, 1)});
    this->indexes_.emplace_back(index);
  }
}

void VoxelHashMap::AddPoints(const std::vector<Eigen::Vector3d> &points) {
  int num_points = points.size();
  this->indexes_.reserve(num_points);
  for (int idx = 0; idx < num_points; ++idx) {
    AddPoint(points[idx], idx);
  }
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<int>>
VoxelHashMap::Pointcloud() const {
  std::vector<Eigen::Vector3d> points;
  for (const auto &[voxel, voxel_block] : map_) {
    for (int i = 0; i < voxel_block.size(); ++i) {
      points.push_back(voxel_block.points[i]);
    }
  }
  return std::make_tuple(points, this->indexes_);
}

void compute_graph_distances(Eigen::Ref<RowMatrixXd> laplacian_matrix) {
  assert(laplacian_matrix.rows() == laplacian_matrix.cols() &&
         "compute_graph_distances| mismatch rows and columns in laplacian");
  size_t num_points = laplacian_matrix.rows();
  for (size_t r = 0; r < num_points; ++r) {
    for (size_t c = r; c < num_points; ++c) {
      const Eigen::VectorXd &dists_on_graph =
          laplacian_matrix.row(r) + laplacian_matrix.row(c);
      laplacian_matrix(r, c) = dists_on_graph.minCoeff();
      laplacian_matrix(c, r) = laplacian_matrix(r, c);
    }
  }
}
} // namespace voxel_hash_map
