#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <eigen3/Eigen/Core>
#include <vector>

#include "stl_vector_eigen.h"
#include "voxel_hash_map.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);

namespace voxel_hash_map {
PYBIND11_MODULE(voxel_hash_map_pybind, m) {
  auto vector3dvector = pybind_eigen_vector_of_vector<Eigen::Vector3d>(
      m, "_Vector3dVector", "std::vector<Eigen::Vector3d>",
      py::py_array_to_vectors_double<Eigen::Vector3d>);

  py::class_<VoxelHashMap> voxel_hash_map(m, "_VoxelHashMap", "Don't use this");
  voxel_hash_map
      .def(py::init([](double voxel_size) { return VoxelHashMap(voxel_size); }),
           "voxel_size"_a)
      .def("_add_points", &VoxelHashMap::AddPoints, "points"_a)
      .def("_point_cloud", &VoxelHashMap::Pointcloud);

  // Floating functions, unrelated to the mapping class
  m.def("_voxel_down_sample", &voxel_down_sample, "frame"_a, "voxel_size"_a);
  m.def("_compute_graph_distances", &compute_graph_distances,
        "laplacian_matrix"_a);
}

} // namespace voxel_hash_map
