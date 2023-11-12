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

  using namespace dynorrt;
  using RX = dynotree::Rn<double, -1>;

  using PlannerBase_RX = PlannerBase<RX, -1>;
  using RRT_X = RRT<RX, -1>;
  using BiRRT_X = BiRRT<RX, -1>;
  using RRTConnect_X = RRTConnect<RX, -1>;

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

  py::class_<PlannerBase_RX>(m, "PlannerBase_X")
      .def(py::init<>())
      .def("set_start", &PlannerBase_RX::set_start) // add point
      .def("set_goal", &PlannerBase_RX::set_goal)
      .def("init", &PlannerBase_RX::init)
      .def("plan", &PlannerBase_RX::plan)
      .def("get_path", &PlannerBase_RX::get_path)
      .def("get_fine_path", &PlannerBase_RX::get_fine_path)
      .def("set_is_collision_free_fun",
           &PlannerBase_RX::set_is_collision_free_fun)
      .def("set_bounds_to_state", &PlannerBase_RX::set_bounds_to_state)
      .def("get_sample_configs", &PlannerBase_RX::get_sample_configs)
      .def("get_configs", &PlannerBase_RX::get_configs)
      .def("get_parents", &PlannerBase_RX::get_parents)
      .def("read_cfg_file", &PlannerBase_RX::read_cfg_file)
      .def("read_cfg_string", &PlannerBase_RX::read_cfg_string);

  py::class_<RRT_X, PlannerBase_RX>(m, "RRT_X")
      .def(py::init<>())
      .def("set_options", &RRT_X::set_options);

  py::class_<BiRRT_X, PlannerBase_RX>(m, "BiRRT_X")
      .def(py::init<>())
      .def("set_options", &BiRRT_X::set_options);

  py::class_<RRTConnect_X, BiRRT_X>(m, "RRTConnect_X").def(py::init<>());

  m.def("srand", [](int seed) { std::srand(seed); });
  m.def("rand", []() { return std::rand(); });
  m.def("rand01", []() { return static_cast<double>(std::rand()) / RAND_MAX; });
}
