// #include "dynotree/KDTree.h"
#include "dynoRRT/rrt.h"

#include <pybind11/functional.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

// template <typename T>
// void declare_tree(py::module &m, const std::string &name) {
//   py::class_<typename T::DistanceId>(m, (name + "dp").c_str())
//       .def(py::init<>())
//       .def_readonly("distance", &T::DistanceId::distance)
//       .def_readonly("id", &T::DistanceId::id);
//
//   py::class_<T>(m, name.c_str())
//       .def(py::init<int>())
//       .def("addPoint", &T::addPoint)
//       .def("search", &T::search)
//       .def("searchKnn", &T::searchKnn)
//       .def("getStateSpace", &T::getStateSpace);
//   // .def("interpolate", &T::interpolate);
//   //
// }
//
// template <typename T>
// void declare_treex(py::module &m, const std::string &name) {
//   // const std::string name = "TreeX";
//   py::class_<typename T::DistanceId>(m, (name + "dp").c_str())
//       .def(py::init<>())
//       .def_readonly("distance", &T::DistanceId::distance)
//       .def_readonly("id", &T::DistanceId::id);
//
//   py::class_<T>(m, name.c_str())
//       .def(py::init<int, const std::vector<std::string>>())
//       .def("addPoint", &T::addPoint)            // add point
//       .def("search", &T::search)                // search
//       .def("searchKnn", &T::searchKnn)          // search
//       .def("getStateSpace", &T::getStateSpace); //
//   //
//   //
//   //
//   // .def("interpolate", &T::interpolate);
//   //
// }
//
// template <typename T>
// void declare_state_space_x(py::module &m, const std::string &name) {
//
//   py::class_<T>(m, name.c_str())
//       .def(py::init<const std::vector<std::string>>())
//       // .def(py::init<>())
//       .def("interpolate", &T::interpolate)       // add point
//       .def("set_bounds", &T::set_bounds)         // search
//       .def("sample_uniform", &T::sample_uniform) // search
//       .def("distance", &T::distance)
//       .def("distance_to_rectangle", &T::distance_to_rectangle);
//
//   // search
//   //
// }
//
// template <typename T>
// void declare_state_space(py::module &m, const std::string &name) {
//
//   py::class_<T>(m, name.c_str())
//       // .def(py::init<int, const std::vector<std::string>>())
//       .def(py::init<>())
//       .def("interpolate", &T::interpolate)
//       .def("set_bounds", &T::set_bounds)
//       .def("sample_uniform", &T::sample_uniform)
//       .def("distance", &T::distance)
//       .def("distance_to_rectangle", &T::distance_to_rectangle);
//
//   // search
//   //
// }

PYBIND11_MODULE(pydynorrt, m) {
  m.doc() = R"pbdoc(
        pydynotree
        -----------------------

        .. currentmodule:: pydynotree

    )pbdoc";

  using RX = dynotree::Rn<double, -1>;
  using RRT_X = RRT<RX, -1>;
  // using RRT_RX = dynotree::KDTree<int, -1, 32, double, RX>;

  py::class_<RRT_options>(m, "RRT_options")
      .def(py::init<>())
      .def_readwrite("max_it", &RRT_options::max_it)
      .def_readwrite("goal_bias", &RRT_options::goal_bias)
      .def_readwrite("collision_resolution", &RRT_options::collision_resolution)
      .def_readwrite("max_step", &RRT_options::max_step)
      .def_readwrite("max_compute_time_ms", &RRT_options::max_compute_time_ms)
      .def_readwrite("goal_tolerance", &RRT_options::goal_tolerance)
      .def_readwrite("max_num_configs", &RRT_options::max_num_configs);

  py::enum_<TerminationCondition>(m, "TerminationCondition", py::arithmetic())
      .value("MAX_IT", TerminationCondition::MAX_IT)
      .value("MAX_TIME", TerminationCondition::MAX_TIME)
      .value("GOAL_REACHED", TerminationCondition::GOAL_REACHED)
      .value("MAX_NUM_CONFIGS", TerminationCondition::MAX_NUM_CONFIGS)
      .value("UNKNOWN", TerminationCondition::UNKNOWN);

  py::class_<RRT_X>(m, "RRT_X")
      .def(py::init<>())
      .def("set_start", &RRT_X::set_start) // add point
      .def("set_goal", &RRT_X::set_goal)
      .def("init_tree", &RRT_X::init_tree)
      .def("plan", &RRT_X::plan)
      .def("get_path", &RRT_X::get_path)
      .def("get_fine_path", &RRT_X::get_fine_path)
      .def("set_is_collision_free_fun", &RRT_X::set_is_collision_free_fun)
      .def("set_bounds_to_state", &RRT_X::set_bounds_to_state)
      .def("set_options", &RRT_X::set_options)
      .def("get_sample_configs", &RRT_X::get_sample_configs)
      .def("get_configs", &RRT_X::get_configs)
      .def("get_parents", &RRT_X::get_parents);

  // rrt.init_tree();
  //
  //
  //          (Eigen::Vector3d(2.0, 0.2, 0));
  //
  //
  //
  //     .def("addPoint", &T::addPoint)            // add point
  //     .def("search", &T::search)                // search
  //     .def("searchKnn", &T::searchKnn)          // search
  //     .def("getStateSpace", &T::getStateSpace); //
  //
  //
  //
  // using state_space_t = dynotree::R2SO2<double>;
  // using tree_t = dynotree::KDTree<int, 3, 32, double, state_space_t>;
  //
  // state_space_t state_space;
  // state_space.set_bounds(Eigen::Vector2d(0, 0), Eigen::Vector2d(3, 3));
  //
  // RRT<state_space_t, 3> rrt;
  //
  // std::vector<CircleObstacle> obstacles;
  //
  // obstacles.push_back(CircleObstacle{Eigen::Vector2d(1, 0.4), 0.5});
  // obstacles.push_back(CircleObstacle{Eigen::Vector2d(1, 2), 0.5});
  //
  // RRT_options options{.max_it = 1000,
  //                     .goal_bias = 0.05,
  //                     .collision_resolution = 0.01,
  //                     .max_step = 10,
  //                     .max_compute_time_ms = 1e9,
  //                     .goal_tolerance = 0.001,
  //                     .max_num_configs = 10000};
  //
  // rrt.set_options(options);
  // rrt.set_state_space(state_space);
  // rrt.set_start(Eigen::Vector3d(0.1, 0.1, M_PI / 2));
  // rrt.set_goal(Eigen::Vector3d(2.0, 0.2, 0));
  // rrt.init_tree();
  //
  // rrt.set_is_collision_free_fun(
  //     [&](const auto &x) { return !is_collision(x, obstacles, radius);
  //     });
  //
  // TerminationCondition termination_condition = rrt.plan();
  //
  // std::cout << magic_enum::enum_name(termination_condition) << std::endl;
  //
  // std::vector<Eigen::Vector3d> path, fine_path;
  // if (termination_condition == TerminationCondition::GOAL_REACHED) {
  //   rrt.get_path(path);
  //   rrt.get_fine_path(0.01, fine_path);
  // }
  //

  // using R2 = dynotree::Rn<double, 2>;
  // using R3 = dynotree::Rn<double, 3>;
  // using R4 = dynotree::Rn<double, 4>;
  // using R5 = dynotree::Rn<double, 5>;
  // using R6 = dynotree::Rn<double, 6>;
  // using R7 = dynotree::Rn<double, 7>;
  // using RX = dynotree::Rn<double, -1>;
  //
  // using SO2 = dynotree::SO2<double>;
  // using SO3 = dynotree::SO3<double>;
  //
  // using R2SO2 = dynotree::R2SO2<double>;
  // using R3SO3 = dynotree::R3SO3<double>;
  //
  // using Combined = dynotree::Combined<double>;
  //
  // declare_state_space<R2>(m, "R2");
  // declare_state_space<R3>(m, "R3");
  // declare_state_space<R4>(m, "R4");
  // declare_state_space<R5>(m, "R5");
  // declare_state_space<R6>(m, "R6");
  // declare_state_space<R7>(m, "R7");
  // declare_state_space<RX>(m, "RX");
  //
  // declare_state_space<R2SO2>(m, "R2SO2");
  // declare_state_space<R3SO3>(m, "R3SO3");
  //
  // declare_state_space<SO3>(m, "SO3");
  // declare_state_space<SO2>(m, "SO2");
  //
  // declare_state_space_x<Combined>(m, "SpaceX");
  //
  // const int bucket_size = 32;
  // using TreeRX = dynotree::KDTree<int, -1, bucket_size, double, RX>;
  // using TreeR2 = dynotree::KDTree<int, 2, bucket_size, double, R2>;
  // using TreeR4 = dynotree::KDTree<int, 4, bucket_size, double, R4>;
  // using TreeR7 = dynotree::KDTree<int, 7, bucket_size, double, R7>;
  // using TreeR2SO2 = dynotree::KDTree<int, 3, bucket_size, double, R2SO2>;
  // using TreeSO3 = dynotree::KDTree<int, 4, bucket_size, double, SO3>;
  // using TreeSO2 = dynotree::KDTree<int, 1, bucket_size, double, SO2>;
  // using TreeR3SO3 = dynotree::KDTree<int, 7, bucket_size, double, R3SO3>;
  // using TreeX = dynotree::KDTree<int, -1, bucket_size, double, Combined>;
  //
  // declare_tree<TreeRX>(m, "TreeRX");
  // declare_tree<TreeR2>(m, "TreeR2");
  // declare_tree<TreeR4>(m, "TreeR4");
  // declare_tree<TreeR7>(m, "TreeR7");
  // declare_tree<TreeR2SO2>(m, "TreeR2SO2");
  // declare_tree<TreeSO3>(m, "TreeSO3");
  // declare_tree<TreeSO2>(m, "TreeSO2");
  // declare_tree<TreeR3SO3>(m, "TreeR3SO3");
  // declare_treex<TreeX>(m, "TreeX");

  m.def("srand", [](int seed) { std::srand(seed); });
  m.def("rand", []() { return std::rand(); });
  m.def("rand01", []() { return static_cast<double>(std::rand()) / RAND_MAX; });
}
