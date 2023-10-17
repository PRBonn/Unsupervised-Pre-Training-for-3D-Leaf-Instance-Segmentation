#pragma once

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <map>
#include <queue>
#include <tuple>
#include <vector>

namespace voxel_hash_map {
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct Voxel {
  Voxel(int x, int y, int z) : i(x), j(y), k(z) {}
  Voxel(const Eigen::Vector3d &point, double voxel_size) {
    i = static_cast<int>(point[0] / voxel_size);
    j = static_cast<int>(point[1] / voxel_size);
    k = static_cast<int>(point[2] / voxel_size);
  }
  bool operator==(const Voxel &vox) const {
    return i == vox.i && j == vox.j && k == vox.k;
  }

  int i;
  int j;
  int k;
};

struct VoxelBlock {
  explicit VoxelBlock(int num_points) : num_points_(num_points) {
    points.reserve(num_points);
  }

  explicit VoxelBlock(const Eigen::Vector3d &point, int num_points)
      : num_points_(num_points) {
    points.reserve(num_points);
    points.emplace_back(point);
  }

  void add_point(const Eigen::Vector3d &point) {
    if (IsFull())
      return;
    points.emplace_back(point);
  }

  [[nodiscard]] inline bool IsFull() const {
    return num_points_ == points.size();
  }
  [[nodiscard]] inline int size() const { return points.size(); }

  std::vector<Eigen::Vector3d> points;
  int num_points_{};
};

// Search Neighbors with VoxelHashMap lookups
using pair_distance_t = std::tuple<double, Eigen::Vector3d>;

struct Comparator {
  bool operator()(const pair_distance_t &left,
                  const pair_distance_t &right) const {
    return std::get<0>(left) < std::get<0>(right);
  }
};

using priority_queue_t =
    std::priority_queue<pair_distance_t, std::vector<pair_distance_t>,
                        Comparator>;

} // namespace voxel_hash_map
// Specialization of std::hash for our custom type Voxel
namespace std {

template <> struct hash<voxel_hash_map::Voxel> {
  size_t operator()(const voxel_hash_map::Voxel &vox) const {
    const size_t kP1 = 73856093;
    const size_t kP2 = 19349669;
    const size_t kP3 = 83492791;
    return vox.i * kP1 + vox.j * kP2 + vox.k * kP3;
  }
};
} // namespace std

namespace voxel_hash_map {

std::tuple<std::vector<Eigen::Vector3d>, std::vector<int>>
voxel_down_sample(const std::vector<Eigen::Vector3d> &frame, double voxel_size);

class VoxelHashMap {
public:
  explicit VoxelHashMap(const double voxel_size_) : voxel_size_(voxel_size_) {}

public:
  using Vector3dVector = std::vector<Eigen::Vector3d>;
  using PointCloudReturnType = std::tuple<Vector3dVector, std::vector<int>>;
  [[nodiscard]] PointCloudReturnType Pointcloud() const;
  void AddPoints(const Vector3dVector &points);

private:
  void AddPoint(const Eigen::Vector3d &point, const int index);

private:
  // mapping parameters
  double voxel_size_;
  std::vector<int> indexes_;
  // map representation
  std::unordered_map<Voxel, VoxelBlock> map_;
};
void compute_graph_distances(Eigen::Ref<RowMatrixXd> laplacian_matrix);
} // namespace voxel_hash_map
