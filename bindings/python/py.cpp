#include <pinocchio/bindings/python/pybind11.hpp>

// This lines forces clang-format to keep the include split here
#include <pybind11/pybind11.h>

#include <boost/python.hpp>

#define SCALAR double
#define OPTIONS 0
#define JOINT_MODEL_COLLECTION ::pinocchio::JointCollectionDefaultTpl
#include <pinocchio/bindings/python/pybind11-all.hpp>

pinocchio::Model *make_model() {
  pinocchio::Model *model = new pinocchio::Model;
  std::cout << "make_model: " << reinterpret_cast<intptr_t>(model) << std::endl;
  return model;
}

pinocchio::Model &return_same_model_copy(pinocchio::Model &m) { return m; }
pinocchio::Model *return_same_model_nocopy(pinocchio::Model &m) { return &m; }

pinocchio::SE3 multiply_se3(pinocchio::SE3 const &a, pinocchio::SE3 const &b) {
  return a * b;
}

template <typename T> intptr_t get_ptr(T &m) {
  std::cout << &m << '\n' << m << std::endl;
  return reinterpret_cast<intptr_t>(&m);
}

void test1(int i) { std::cout << "no conversion: " << ' ' << i << std::endl; }
void testModel1(pinocchio::Model &model) {
  std::cout << "testModel1: " << &model << std::endl;
  model.name = "testModel1: I modified the model name";
}
intptr_t testModel2(pinocchio::Model &model, int i) {
  std::cout << "testModel2: " << &model << ' ' << i << std::endl;
  model.name = "testModel2: I modified the model name";
  return reinterpret_cast<intptr_t>(&model);
}
intptr_t testModel3(pinocchio::Model const &model, int i) {
  std::cout << "testModel3: " << &model << ' ' << i << std::endl;
  return reinterpret_cast<intptr_t>(&model);
}

void testModel_manual(pybind11::object model) {
  testModel1(pinocchio::python::from<pinocchio::Model &>(model));
}

using pinocchio::python::make_pybind11_function;

#if 0

PYBIND11_MODULE(pydynorrt, m) {
  using namespace pybind11::literals;  // For _a

  pybind11::module::import("pinocchio");
  m.def("testModel_manual", testModel_manual);

  m.def("test1", make_pybind11_function(&test1));
  m.def("make_model", make_pybind11_function(&make_model));
  // m.def("return_same_model_broken", make_pybind11_function(&return_same_model_copy),
  //       pybind11::return_value_policy::reference);
  // m.def("return_same_model", make_pybind11_function(&return_same_model_nocopy),
  //       pybind11::return_value_policy::reference);
  //
  m.def("get_ptr", make_pybind11_function(&get_ptr<pinocchio::Model>));
  // m.def("get_se3_ptr", make_pybind11_function(&get_ptr<pinocchio::SE3>));
  //
  // m.def("multiply_se3_1", make_pybind11_function(&multiply_se3), "a"_a, "b"_a);
  // m.def("multiply_se3", make_pybind11_function(&multiply_se3), "a"_a,
  //       "b"_a = pinocchio::python::default_arg(pinocchio::SE3::Identity()));
  m.def("testModel1", make_pybind11_function(&testModel1));
  // m.def("testModel2", make_pybind11_function(&testModel2));
  // m.def("testModel3", make_pybind11_function(&testModel3));
  //
  // pybind11::module no_wrapper = m.def_submodule("no_wrapper");
  // no_wrapper.def("multiply_se3", &multiply_se3, "a"_a,
  //                "b"_a
  //                // does not work = pinocchio::SE3::Identity()
  //                );
  // no_wrapper.def("testModel1", &testModel1);
  // no_wrapper.def("testModel2", &testModel2);
  // no_wrapper.def("testModel3", &testModel3);
}

#endif

#if 1

// #include "dynotree/KDTree.h"
// #include "dynoRRT/rrt_base.h"
#include "dynoRRT/birrt.h"
#include "dynoRRT/kinorrt.h"
#include "dynoRRT/lazyprm.h"
#include "dynoRRT/prm.h"
#include "dynoRRT/rrt.h"
#include "dynoRRT/rrtconnect.h"
#include "dynoRRT/rrtstar.h"

#include <pybind11/functional.h>

#include "dynoRRT/collision_manager.h"
#include "dynoRRT/pin_col_manager.h"

#include "pybind11_json.hpp"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace dynorrt;

// #include <pinocchio/bindings/python/pybind11.hpp>
// This lines forces clang-format to keep the include split here

// #include <boost/python.hpp>

// #define SCALAR double
// #define OPTIONS 0
// #define JOINT_MODEL_COLLECTION ::pinocchio::JointCollectionDefaultTpl
// #include <pinocchio/bindings/python/pybind11-all.hpp>

using pinocchio::python::make_pybind11_function;

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

#include <iostream>

void set_pin_model(dynorrt::Collision_manager_pinocchio &col_manager,
                   pinocchio::Model &t_model,
                   pinocchio::GeometryModel &t_geomodel) {
  col_manager.set_pin_model(t_model, t_geomodel);
}

void model_test(pinocchio::Model &model) {
  std::cout << "model_test" << std::endl;
  std::cout << "model.nq" << model.nq << std::endl;
  std::cout << "model.nv" << model.nv << std::endl;
}

// m.def("get_ptr", make_pybind11_function(&get_ptr<pinocchio::Model>));

template <typename StateSpace, int dim>
void add_planners_to_module(py::module &m, const std::string &name) {

  using PlannerBase_RX = PlannerBase<StateSpace, dim>;
  py::class_<PlannerBase_RX>(m, ("PlannerBase_" + name).c_str())
      .def(py::init<>())
      .def("set_collision_manager",
           &PlannerBase_RX::set_collision_manager)  // add point
      .def("set_start", &PlannerBase_RX::set_start) // add point
      .def("set_goal", &PlannerBase_RX::set_goal)
      .def("set_goal_list", &PlannerBase_RX::set_goal_list)
      .def("init", &PlannerBase_RX::init)
      .def("plan", &PlannerBase_RX::plan)
      .def("get_path", &PlannerBase_RX::get_path)
      .def("get_fine_path", &PlannerBase_RX::get_fine_path)
      .def("set_is_collision_free_fun",
           &PlannerBase_RX::set_is_collision_free_fun)
      .def("set_sample_fun", &PlannerBase_RX::set_sample_fun)
      .def("set_bounds_to_state", &PlannerBase_RX::set_bounds_to_state)
      .def("get_sample_configs", &PlannerBase_RX::get_sample_configs)
      .def("set_state_space_with_string",
           &PlannerBase_RX::set_state_space_with_string)
      .def("get_configs", &PlannerBase_RX::get_configs)
      .def("get_parents", &PlannerBase_RX::get_parents)
      .def("read_cfg_file", &PlannerBase_RX::read_cfg_file)
      .def("read_cfg_string", &PlannerBase_RX::read_cfg_string)
      .def("get_planner_data",
           [](PlannerBase_RX &planner) {
             json j;
             planner.get_planner_data(j);
             return j;
           })
      .def("set_is_collision_free_fun_from_manager",
           [](PlannerBase_RX &planner,
              Collision_manager_pinocchio &col_manager) {
             planner.set_is_collision_free_fun([&](const auto &q) {
               return col_manager.is_collision_free(q);
             });
           })
      .def("set_is_collision_free_fun_from_manager_parallel",
           [](PlannerBase_RX &planner,
              Collision_manager_pinocchio &col_manager) {
             planner.set_is_collision_free_fun_parallel([&](const auto &q) {
               return col_manager.is_collision_free_set(q, true, nullptr,
                                                        nullptr);
             });
           })
      .def("set_dev_mode_parallel", &PlannerBase_RX::set_dev_mode_parallel);

  using PlannerBase_RX = PlannerBase<StateSpace, dim>;
  using RRTStar = RRTStar<StateSpace, dim>;
  using RRT = RRT<StateSpace, dim>;
  using BiRRT = BiRRT<StateSpace, dim>;
  using RRTConnect = RRTConnect<StateSpace, dim>;
  using PRM = PRM<StateSpace, dim>;
  using LazyPRM = LazyPRM<StateSpace, dim>;

  py::class_<RRTStar, PlannerBase_RX>(m, ("PlannerRRTStar_" + name).c_str())
      .def(py::init<>())
      .def("say_hello", []() { std::cout << "hello" << std::endl; });

  py::class_<RRT, PlannerBase_RX>(m, ("PlannerRRT_" + name).c_str())
      .def(py::init<>());

  py::class_<BiRRT, PlannerBase_RX>(m, ("PlannerBiRRT_" + name).c_str())
      .def(py::init<>());

  py::class_<RRTConnect, BiRRT>(m, ("PlannerRRTConnect_" + name).c_str())
      .def(py::init<>());

  py::class_<PRM, PlannerBase_RX>(m, ("PlannerPRM_" + name).c_str())
      .def(py::init<>());
  // .def("set_options", &PRM_X::set_options)
  // .def("get_adjacency_list", &PRM_X::get_adjacency_list)
  // .def("get_check_edges_valid", &PRM_X::get_check_edges_valid)
  // .def("get_check_edges_invalid", &PRM_X::get_check_edges_invalid);

  py::class_<LazyPRM, PlannerBase_RX>(m, ("PlannerLazyPRM_" + name).c_str())
      .def(py::init<>());

  // For now, I only expose with control dim = -1.
  using KinoRRT_RX = KinoRRT<StateSpace, dim, -1>;

  py::class_<KinoRRT_RX, PlannerBase_RX>(m, ("PlannerKinoRRT_" + name).c_str())
      .def(py::init<>())
      .def("set_expand_fun", &KinoRRT_RX::set_expand_fun);

  // .def("set_options", &LazyPRM_X::set_options)
  // .def("get_adjacency_list", &LazyPRM_X::get_adjacency_list)
  // .def("get_check_edges_valid", &LazyPRM_X::get_check_edges_valid)
  // .def("get_check_edges_invalid", &LazyPRM_X::get_check_edges_invalid);

  // py::class_<RRTConnect, PlannerBase_RX>(m, ("PlannerRRTConnect_" +
  // name).c_str())
  //     .def(py::init<>());

  // py
};

// void testModel1(pinocchio::Model& model) {
//   std::cout << "testModel1: " << &model << std::endl;
//   model.name = "testModel1: I modified the model name";
// }

PYBIND11_MODULE(pydynorrt, m) {

  // pybind11::module::import("pinocchio");
  using namespace pybind11::literals; // For _a

  m.doc() = R"pbdoc(
        pydynorrt
        -----------------------

        .. currentmodule:: pydynotree

    )pbdoc";

  using namespace dynorrt;
  using RX = dynotree::Rn<double, -1>;

  using PlannerBase_RX = PlannerBase<RX, -1>;
  using RRT_X = RRT<RX, -1>;
  using BiRRT_X = BiRRT<RX, -1>;
  using RRTConnect_X = RRTConnect<RX, -1>;
  using PRM_X = PRM<RX, -1>;
  using LazyPRM_X = LazyPRM<RX, -1>;

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

  // PlannerID { RRT, BiRRT, RRTConnect, PRM, LazyPRM, RRTStar, UNKNOWN };
  py::enum_<PlannerID>(m, "PlannerID", py::arithmetic())
      .value("RRT", PlannerID::RRT)
      .value("BiRRT", PlannerID::BiRRT)
      .value("RRTConnect", PlannerID::RRTConnect)
      .value("RRTStar", PlannerID::RRTStar)
      .value("PRM", PlannerID::PRM)
      .value("LazyPRM", PlannerID::LazyPRM)
      .value("UNKNOWN", PlannerID::UNKNOWN);

  using RX = dynotree::Rn<double, -1>;
  using Combined = dynotree::Combined<double>;

  py::enum_<TerminationCondition>(m, "TerminationCondition", py::arithmetic())
      .value("MAX_IT", TerminationCondition::MAX_IT)
      .value("MAX_TIME", TerminationCondition::MAX_TIME)
      .value("GOAL_REACHED", TerminationCondition::GOAL_REACHED)
      .value("MAX_NUM_CONFIGS", TerminationCondition::MAX_NUM_CONFIGS)
      .value("RUNNING", TerminationCondition::RUNNING)
      .value("USER_DEFINED", TerminationCondition::USER_DEFINED)
      .value("MAX_IT_GOAL_REACHED", TerminationCondition::MAX_IT_GOAL_REACHED)
      .value("MAX_TIME_GOAL_REACHED",
             TerminationCondition::MAX_TIME_GOAL_REACHED)
      .value("MAX_NUM_CONFIGS_GOAL_REACHED",
             TerminationCondition::MAX_NUM_CONFIGS_GOAL_REACHED)
      .value("RUNNING_GOAL_REACHED", TerminationCondition::RUNNING_GOAL_REACHED)
      .value("USER_DEFINED_GOAL_REACHED",
             TerminationCondition::USER_DEFINED_GOAL_REACHED)
      .value("EXTERNAL_TRIGGER_GOAL_REACHED",
             TerminationCondition::EXTERNAL_TRIGGER_GOAL_REACHED)
      .value("UNKNOWN", TerminationCondition::UNKNOWN);

  add_planners_to_module<Combined, -1>(m, "Combined");
  add_planners_to_module<dynotree::Rn<double, -1>, -1>(m, "Rn");
  // add_planners_to_module<dynotree::Rn<double, 12>, 12>(m, "R12");
  // add_planner_to_module<dynotree::Rn<double, -1>, -1>(m, "Rn");
  // add_planner_to_module<dynotree::Rn<double, 2>, 2>(m, "R2");
  // add_planner_to_module<dynotree::Rn<double, 3>, 3>(m, "R3");
  // add_planner_to_module<dynotree::Rn<double, 4>, 4>(m, "R4");
  // add_planner_to_module<dynotree::Rn<double, 5>, 5>(m, "R5");
  // add_planner_to_module<dynotree::Rn<double, 6>, 6>(m, "R6");
  // add_planner_to_module<dynotree::Rn<double, 7>, 7>(m, "R7");

  // Do I have to add planner one by one?

  // py::class_<RRTStar<dynotree::Combined<double>,-1>, Planner>(m, "RRT_X")
  //     .def(py::init<>())
  //     .def("set_options", &RRT_X::set_options);

  // TODO: add until 14 DIM (e.g., two pandas). Also add SE3, SE2

  // m.def("get_planner_RX", &get_planner<Rx, -1>);
  // m.def("get_planner_R2", &get_planner<dynotree::Rn<double, 2>, 2>);
  // m.def("get_planner_R3", &get_planner<dynotree::Rn<double, 3>, 3>);
  // m.def("get_planner_R4", &get_planner<dynotree::Rn<double, 4>, 4>);
  // m.def("get_planner_R5", &get_planner<dynotree::Rn<double, 5>, 5>);
  // m.def("get_planner_R6", &get_planner<dynotree::Rn<double, 6>, 6>);
  // m.def("get_planner_R7", &get_planner<dynotree::Rn<double, 7>, 7>);
  // m.def("get_planner_X", &get_planner<Combined, -1>);

  // py::class_<RRT_X, PlannerBase_RX>(m, "RRT_X")
  //     .def(py::init<>())
  //     .def("set_options", &RRT_X::set_options);
  //
  // py::class_<BiRRT_X, PlannerBase_RX>(m, "BiRRT_X")
  //     .def(py::init<>())
  //     .def("set_options", &BiRRT_X::set_options);
  //
  // py::class_<RRTConnect_X, BiRRT_X>(m, "RRTConnect_X").def(py::init<>());

  // py::class_<PRM_X, PlannerBase_RX>(m, "PRM_X")
  //     .def(py::init<>())
  //     .def("set_options", &PRM_X::set_options)
  //     .def("get_adjacency_list", &PRM_X::get_adjacency_list)
  //     .def("get_check_edges_valid", &PRM_X::get_check_edges_valid)
  //     .def("get_check_edges_invalid", &PRM_X::get_check_edges_invalid);
  //
  // py::class_<LazyPRM_X, PlannerBase_RX>(m, "LazyPRM_X")
  //     .def(py::init<>())
  //     .def("set_options", &LazyPRM_X::set_options)
  //     .def("get_adjacency_list", &LazyPRM_X::get_adjacency_list)
  //     .def("get_check_edges_valid", &LazyPRM_X::get_check_edges_valid)
  //     .def("get_check_edges_invalid", &LazyPRM_X::get_check_edges_invalid);

  py::class_<BallObstacle<2>>(m, "BallObs2")
      .def(py::init<BallObstacle<2>::Cref, double>())
      .def_readwrite("center", &BallObstacle<2>::center)
      .def_readwrite("radius", &BallObstacle<2>::radius);

  py::class_<CollisionManagerBallWorld<2>>(m, "CM2")
      .def(py::init<>())
      .def("add_obstacle", &CollisionManagerBallWorld<2>::add_obstacle)
      .def("set_radius_robot", &CollisionManagerBallWorld<2>::set_radius_robot)
      .def("set_obstacles", &CollisionManagerBallWorld<2>::set_obstacles);

  py::class_<BallObstacle<-1>>(m, "BallObsX")
      .def(py::init<BallObstacle<-1>::Cref, double>())
      .def_readwrite("center", &BallObstacle<-1>::center)
      .def_readwrite("radius", &BallObstacle<-1>::radius);

  py::class_<CollisionManagerBallWorld<-1>>(m, "CMX")
      .def(py::init<>())
      .def("add_obstacle", &CollisionManagerBallWorld<-1>::add_obstacle)
      .def("set_radius_robot", &CollisionManagerBallWorld<-1>::set_radius_robot)
      .def("set_obstacles", &CollisionManagerBallWorld<-1>::set_obstacles);

  py::class_<FrameBounds>(m, "FrameBounds")
      .def(py::init<>())
      .def_readwrite("frame_name", &FrameBounds::frame_name)
      .def_readwrite("lower", &FrameBounds::lower)
      .def_readwrite("upper", &FrameBounds::upper);

  py::class_<Collision_manager_pinocchio>(m, "Collision_manager_pinocchio")
      .def(py::init<>())
      .def("set_urdf_filename", &Collision_manager_pinocchio::set_urdf_filename)
      .def("set_srdf_filename", &Collision_manager_pinocchio::set_srdf_filename)
      .def("set_robots_model_path",
           &Collision_manager_pinocchio::set_robots_model_path)
      .def("build", &Collision_manager_pinocchio::build)
      .def("is_collision_free", &Collision_manager_pinocchio::is_collision_free)
      .def("set_frame_bounds", &Collision_manager_pinocchio::set_frame_bounds)
      .def("set_edge_parallel", &Collision_manager_pinocchio::set_edge_parallel)
      .def("is_collision_free_set",
           [](Collision_manager_pinocchio &this_object,
              const std::vector<Eigen::VectorXd> &q_set,
              bool stop_at_first_collision) {
             int counter_infeas_out = 0;
             int counter_feas_out = 0;
             bool out = this_object.is_collision_free_set(
                 q_set, stop_at_first_collision, &counter_infeas_out,
                 &counter_feas_out);
             return std::make_tuple(out, counter_infeas_out, counter_feas_out);
           })
      .def("reset_counters", &Collision_manager_pinocchio::reset_counters)
      .def("get_num_collision_checks",
           &Collision_manager_pinocchio::get_num_collision_checks)
      .def("set_use_pool", &Collision_manager_pinocchio::set_use_pool)
      .def("get_time_ms", &Collision_manager_pinocchio::get_time_ms);

  // .def("set_pin_model",
  // make_pybind11_function(&Collision_manager_pinocchio::set_pin_model);
  //
  using PathShortCut_RX = PathShortCut<RX, -1>;

  py::class_<PathShortCut_RX>(m, "PathShortCut_RX")
      .def(py::init<>())
      .def("set_bounds_to_state", &PathShortCut_RX::set_bounds_to_state)
      .def("set_is_collision_free_fun",
           &PathShortCut_RX::set_is_collision_free_fun)
      .def("set_collision_manager", &PathShortCut_RX::set_collision_manager)
      .def("set_is_collision_free_fun_from_manager",
           [](PathShortCut_RX &planner,
              Collision_manager_pinocchio &col_manager) {
             planner.set_is_collision_free_fun([&](const auto &q) {
               return col_manager.is_collision_free(q);
             });
           })
      .def("get_path", &PathShortCut_RX::get_path)
      .def("init", &PathShortCut_RX::init)
      .def("get_fine_path", &PathShortCut_RX::get_fine_path)
      .def("get_planner_data",
           [](PathShortCut_RX &this_object) {
             json j;
             this_object.get_planner_data(j);
             return j;
           })
      .def("check_internal", &PathShortCut_RX::check_internal)
      .def("reset_internal", &PathShortCut_RX::reset_internal)
      .def("set_initial_path", &PathShortCut_RX::set_initial_path)
      .def("shortcut", &PathShortCut_RX::shortcut);

  m.def("srand", [](int seed) { std::srand(seed); });
  m.def("srandtime", []() { std::srand(time(0)); });
  m.def("rand", []() { return std::rand(); });
  m.def("rand01", []() { return static_cast<double>(std::rand()) / RAND_MAX; });

  m.def("set_pin_model", make_pybind11_function(&set_pin_model));

  // m.def("testModel1", make_pybind11_function(&testModel1));
  // m.def("model_test", make_pybind11_function(&model_test));

  // m.def( "path_shortcut_v1", &path_shortcut_v1<Eigen::VectorXd,
  // dynotree::Rn<double, -1>, std::function<bool(const Eigen::VectorXd
  // &)>);
  //
  // m.def( "path_shortcut_v1b", &path_shortcut_v1b<Eigen::VectorXd,
  // dynotree::Rn<double, -1>>);
}
#endif
