// clang-format off
#define BOOST_TEST_MODULE test_0
#define BtOST_TEST_DYN_LINK
#include <boost/test/unit_test_suite.hpp>
// clang-format on

#include "dynoRRT/dynorrt_macros.h"
#include "dynoRRT/pin_col_manager.h"
#include "dynobench/motions.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include <string>
// clang-format on

#include "dynoRRT/dynorrt_macros.h"
#include "dynotree/KDTree.h"
#include "magic_enum.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>

#include "dynoRRT/collision_manager.h"
#include "dynoRRT/eigen_conversions.hpp"
#include "dynoRRT/rrt.h"
#include <boost/test/unit_test.hpp>

#include <hpp/fcl/collision_object.h>
#include <hpp/fcl/shape/geometric_shapes.h>
#include <iostream>

#include "dynotree/KDTree.h"
#include "magic_enum.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>

#include "dynoRRT/eigen_conversions.hpp"
#include "dynoRRT/rrt.h"
#include <boost/test/unit_test.hpp>

#include <hpp/fcl/collision_object.h>
#include <hpp/fcl/shape/geometric_shapes.h>
#include <iostream>

#include "dynoRRT/dynorrt_macros.h"

#include "dynobench/planar_rotor.hpp"
#include "dynobench/unicycle1.hpp"
#include <hpp/fcl/collision_object.h>
#include <hpp/fcl/shape/geometric_shapes.h>

#include "dynoRRT/birrt.h"
#include "dynoRRT/dynorrt_macros.h"
#include "dynoRRT/kinorrt.h"
#include "dynoRRT/lazyprm.h"
#include "dynoRRT/prm.h"
#include "dynoRRT/rrt.h"
#include "dynoRRT/rrtconnect.h"
#include "dynoRRT/rrtstar.h"
#include "dynoRRT/sststar.h"

using namespace dynorrt;
using PlannerRn = PlannerBase<dynotree::Rn<double>, -1>;
using PlannerR3SO3 = PlannerBase<dynotree::R3SO3<double>, 7>;

using state_space_t = dynotree::Combined<double>;
using PlannerBase_t = PlannerBase<state_space_t, -1>;
using PlannerPtr = std::shared_ptr<PlannerBase_t>;

BOOST_AUTO_TEST_CASE(t_hello_world_dynobench) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  auto argv = boost::unit_test::framework::master_test_suite().argv;

  if (argc < 3) {
    std::cout << "Usage: ./test_dynorrt <path_to_base_dir> <path_to_dynobench>"
              << std::endl;
    BOOST_TEST(false);
    return;
  }

  // auto env = std ::string(base_path) +
  // "envs/unicycle1_v0/parallelpark_0.yaml";
  //
  // Problem problem;
  // problem.read_from_yaml(env.c_str());

  {
    auto unicycle = dynobench::Model_unicycle1();
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd u = Eigen::VectorXd::Ones(2);
    Eigen::VectorXd xnext(3);
    double dt = 0.1;
    unicycle.step(xnext, x0, u, dt);
    std::cout << "xnext: " << xnext.transpose() << std::endl;
  }

  {
    auto env = std ::string(argv[2]) + "/envs/unicycle1_v0/parallelpark_0.yaml";

    dynobench::Problem problem;
    problem.read_from_yaml(env.c_str());

    auto unicycle = dynobench::Model_unicycle1();
    dynobench::load_env(unicycle, problem);

    Eigen::Vector3d x(.7, .8, 0);

    dynobench::CollisionOut col;

    unicycle.collision_distance(x, col);

    BOOST_CHECK(std::fabs(col.distance - .25) < 1e-7);

    x = Eigen::Vector3d(1.9, .3, 0);
    unicycle.collision_distance(x, col);

    BOOST_CHECK(std::fabs(col.distance - .3) < 1e-7);

    col.write(std::cout);

    x = Eigen::Vector3d(1.5, .3, .1);
    unicycle.collision_distance(x, col);
    col.write(std::cout);

    BOOST_CHECK(std::fabs(col.distance - (-0.11123)) < 1e-5);
  }

  // load model from urdf
  std::string base_path(argv[1]);

  std::string urdf =
      base_path +
      "src/python/pydynorrt/data/models/unicycle_parallel_park.urdf";
  std::string srdf =
      base_path +
      "src/python/pydynorrt/data/models/unicycle_parallel_park.srdf";

  Collision_manager_pinocchio coll_manager;
  coll_manager.set_urdf_filename(urdf);
  coll_manager.set_srdf_filename(srdf);
  // coll_manager.set_robots_model_path("");
  coll_manager.build();

  Eigen::Vector3d x(.7, .8, 0);
  BOOST_TEST(coll_manager.is_collision_free(x));

  x = Eigen::Vector3d(1.9, .3, 0);
  BOOST_TEST(coll_manager.is_collision_free(x));

  x = Eigen::Vector3d(1.5, .3, .1);
  BOOST_TEST(!coll_manager.is_collision_free(x));
}

BOOST_AUTO_TEST_CASE(t_kinorrt) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  auto argv = boost::unit_test::framework::master_test_suite().argv;

  if (argc < 3) {
    std::cout << "Usage: ./test_dynorrt <path_to_base_dir> <path_to_dynobench>"
              << std::endl;
    BOOST_TEST(false);
    return;
  }

  std::string base_path(argv[1]);

  std::string urdf =
      base_path +
      "src/python/pydynorrt/data/models/unicycle_parallel_park.urdf";
  std::string srdf =
      base_path +
      "src/python/pydynorrt/data/models/unicycle_parallel_park.srdf";

  Collision_manager_pinocchio coll_manager;
  coll_manager.set_urdf_filename(urdf);
  coll_manager.set_srdf_filename(srdf);
  // coll_manager.set_robots_model_path("");
  coll_manager.build();

  {
    auto unicycle = dynobench::Model_unicycle1();
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd u = Eigen::VectorXd::Ones(2);
    Eigen::VectorXd xnext(3);
    double dt = 0.1;
    unicycle.step(xnext, x0, u, dt);
    std::cout << "xnext: " << xnext.transpose() << std::endl;
  }

  dynobench::Problem problem;
  auto env = std::string(argv[2]) + "/envs/unicycle1_v0/parallelpark_0.yaml";

  problem.read_from_yaml(env.c_str());

  using state_space_t = dynotree::Combined<double>;
  state_space_t state_space({"Rn:2", "SO2"});

  state_space.set_bounds(problem.p_lb, problem.p_ub);

  KinoRRT_options options;
  options.max_num_kino_steps = 10; // in each expansion,
  options.goal_tolerance = .2;
  options.collision_resolution = .1; // 10 cm

  // Work on this! we should be able to give the state space at compile time!
  // Then, i can compare timings!
  using planner_t = KinoRRT<state_space_t, -1, -1>;
  using trajectory_t = planner_t::trajectory_t;
  planner_t rrt;
  rrt.init(3);
  rrt.set_state_space(state_space);
  rrt.set_options(options);

  auto unicycle = dynobench::Model_unicycle1();
  dynobench::load_env(unicycle, problem);

  rrt.set_start(problem.start);
  rrt.set_goal(problem.goal);

  using state_t = Eigen::VectorXd;

  using is_collision_free_fun_t = std::function<bool(state_t)>;

  is_collision_free_fun_t is_collision_free_fun = [&](const state_t &x) {
    return coll_manager.is_collision_free(x);
    // return unicycle.collision_check(x);
  };

  BOOST_TEST(is_collision_free_fun(problem.start));
  BOOST_TEST(is_collision_free_fun(problem.goal));

  using expand_fun_t =
      std::function<void(state_t &, const state_t &, trajectory_t &)>;

  double dt = .1;

  expand_fun_t expand_fun = [&](const state_t &x_start, const state_t &x_goal,
                                trajectory_t &traj) {
    double best_distance_to_goal = std::numeric_limits<double>::max();
    state_t x(x_start.size());
    state_t xnext(x_start.size());
    trajectory_t traj_tmp;
    for (size_t i = 0; i < options.num_expansions; i++) {

      // sample time
      int min_steps = options.min_num_kino_steps;
      int max_steps = options.max_num_kino_steps;
      int num_steps = min_steps + rand() % (max_steps - min_steps + 1);
      x = x_start;
      traj_tmp.states.push_back(x);

      Eigen::VectorXd u_delta = (unicycle.u_ub - unicycle.u_lb);

      Eigen::VectorXd u =
          (unicycle.u_lb - u_delta) +
          (unicycle.u_ub - unicycle.u_lb + 2 * u_delta)
              .cwiseProduct(
                  (Eigen::VectorXd::Random(2) + Eigen::VectorXd::Ones(2)) / 2.);

      u = u.cwiseMax(unicycle.u_lb);
      u = u.cwiseMin(unicycle.u_ub);

      for (size_t t = 0; t < num_steps; t++) {

        unicycle.step(xnext, x, u, dt);
        traj_tmp.controls.push_back(u);
        traj_tmp.states.push_back(xnext);
        x = xnext;
      }
      double distance_to_goal = state_space.distance(xnext, x_goal);
      if (distance_to_goal < best_distance_to_goal) {
        best_distance_to_goal = distance_to_goal;
        traj = traj_tmp;
      }
    }
  };

  rrt.set_is_collision_free_fun(is_collision_free_fun);
  rrt.set_expand_fun(expand_fun);

  auto out = rrt.plan();

  std::cout << "out: " << magic_enum::enum_name(out) << std::endl;

  std::ofstream file_out("/tmp/kino.json");
  nlohmann::json j;
  rrt.get_planner_data(j);
  file_out << j;

  // how much time spent in nearest neighbour?
}

//
BOOST_AUTO_TEST_CASE(t_kinorrt2) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  auto argv = boost::unit_test::framework::master_test_suite().argv;

  if (argc < 2) {
    std::cout << "Usage: ./test_dynorrt <path_to_base_dir>" << std::endl;
    BOOST_TEST(false);
    return;
  }

  std::string base_path(argv[1]);

  std::string env_file = "benchmark/envs/pinocchio/planarquad_column.json";

  std::ifstream i(base_path + env_file);

  if (!i.is_open()) {
    std::cout << "Error opening file: " << env_file << std::endl;
    BOOST_TEST(false);
    return;
  }

  Eigen::VectorXd weights(6);
  weights << 1., 1., .5, .2, .2, .2;

  // [1., .5, .2, .2] # p , R, v , w

  json j;
  i >> j;

  // Eigen::VectorXd start(j["start"]), goal(j["goal"]), lb(j["lb"]),
  // ub(j["ub"]);

  Eigen::VectorXd start = j["start"];
  Eigen::VectorXd goal = j["goal"];
  Eigen::VectorXd lb = j["lb"];
  Eigen::VectorXd ub = j["ub"];

  std::vector<std::string> state_space_vstr = j["state_space"];

  std::string urdf(j["urdf"]), srdf(j["srdf"]);

  std::string robots_model_path = base_path + std::string(j["base_path"]);
  std::cout << "robots_model_path: " << robots_model_path << std::endl;
  // std::string robots_model_path =
  urdf = robots_model_path + urdf;
  srdf = robots_model_path + srdf;

  Collision_manager_pinocchio coll_manager;
  coll_manager.set_urdf_filename(urdf);
  coll_manager.set_srdf_filename(srdf);
  coll_manager.set_robots_model_path(robots_model_path);
  coll_manager.build();

  // check a line between start and goal
  state_space_t state_space(state_space_vstr);
  state_space.set_weights(weights);
  state_space.set_bounds(lb,
                         ub); // TODO: allow set bounds as a list of vectors!

  if (!coll_manager.is_collision_free(start.head<3>())) {
    THROW_PRETTY_DYNORRT("start is in collision");
  }

  if (!coll_manager.is_collision_free(goal.head<3>())) {
    THROW_PRETTY_DYNORRT("goal  is in collision");
  }

  KinoRRT_options options;
  options.min_num_kino_steps = 10;
  options.max_num_kino_steps = 50; // in each expansion,
  options.goal_tolerance = .5;
  options.collision_resolution = .1; // TODO: should
  // i check all points in the trajectory, or only if they are above the
  // collision resolution?

  using planner_t = KinoRRT<state_space_t, -1, -1>;
  using trajectory_t = planner_t::trajectory_t;
  planner_t rrt;
  rrt.init(state_space.get_runtime_dim());
  rrt.set_state_space(state_space);
  rrt.set_options(options);

  auto robot = dynobench::Model_quad2d();
  // dynobench::load_env(unicycle, problem);

  rrt.set_start(start);
  rrt.set_goal(goal);

  using state_t = Eigen::VectorXd;

  using is_collision_free_fun_t = std::function<bool(state_t)>;

  is_collision_free_fun_t is_collision_free_fun = [&](const state_t &x) {
    return coll_manager.is_collision_free(x.head<3>());
  };

  BOOST_TEST(is_collision_free_fun(start));
  BOOST_TEST(is_collision_free_fun(goal));

  using expand_fun_t =
      std::function<void(state_t &, const state_t &, trajectory_t &)>;

  double dt = .01;

  using V6d = Eigen::Matrix<double, 6, 1>;
  expand_fun_t expand_fun = [&](const state_t &x_start, const state_t &x_goal,
                                trajectory_t &traj) {
    // my model expects [x,y,theta,xdot,ydot,thetadot]

    double best_distance_to_goal = std::numeric_limits<double>::max();
    state_t x(x_start.size());
    state_t xnext(x_start.size());
    trajectory_t traj_tmp;
    for (size_t i = 0; i < options.num_expansions; i++) {

      // sample time
      int min_steps = options.min_num_kino_steps;
      int max_steps = options.max_num_kino_steps;
      int num_steps = min_steps + rand() % (max_steps - min_steps + 1);
      x = x_start;
      traj_tmp.states.push_back(x);
      double k_delta = 0.;

      Eigen::VectorXd u_delta = k_delta * (robot.u_ub - robot.u_lb);

      Eigen::VectorXd u =
          (robot.u_lb - u_delta) +
          (robot.u_ub - robot.u_lb + 2 * u_delta)
              .cwiseProduct(
                  (Eigen::VectorXd::Random(2) + Eigen::VectorXd::Ones(2)) / 2.);

      u = u.cwiseMax(robot.u_lb);
      u = u.cwiseMin(robot.u_ub);

      for (size_t t = 0; t < num_steps; t++) {

        robot.step(xnext, x, u, dt);
        traj_tmp.controls.push_back(u);
        traj_tmp.states.push_back(xnext);
        x = xnext;
      }
      double distance_to_goal = state_space.distance(xnext, x_goal);
      if (distance_to_goal < best_distance_to_goal) {
        best_distance_to_goal = distance_to_goal;
        traj = traj_tmp;
      }
    }
  };

  rrt.set_is_collision_free_fun(is_collision_free_fun);
  rrt.set_expand_fun(expand_fun);

  auto out = rrt.plan();

  std::cout << "planning done" << std::endl;
  std::cout << "out: " << magic_enum::enum_name(out) << std::endl;

  {
    std::ofstream file_out("/tmp/kino_planar.json");
    nlohmann::json j;
    rrt.get_planner_data(j);
    file_out << j;
  }

  // create RRT

  // TODO: SST*
}

BOOST_AUTO_TEST_CASE(t_sst) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  auto argv = boost::unit_test::framework::master_test_suite().argv;

  if (argc < 3) {
    std::cout << "Usage: ./test_dynorrt <path_to_base_dir> <path_to_dynobench>"
              << std::endl;
    BOOST_TEST(false);
    return;
  }

  std::srand(0);

  std::string base_path(argv[1]);

  // std::string urdf = base_path + "/models/unicycle_parallel_park.urdf";
  // std::string srdf = base_path + "/models/unicycle_parallel_park.srdf";

  std::string urdf =
      base_path + "src/python/pydynorrt/data/models/unicycle_bugtrap.urdf";
  std::string srdf =
      base_path + "src/python/pydynorrt/data/models/unicycle_bugtrap.srdf";

  Collision_manager_pinocchio coll_manager;
  coll_manager.set_urdf_filename(urdf);
  coll_manager.set_srdf_filename(srdf);
  // coll_manager.set_robots_model_path("");
  coll_manager.build();

  {
    auto unicycle = dynobench::Model_unicycle1();
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd u = Eigen::VectorXd::Ones(2);
    Eigen::VectorXd xnext(3);
    double dt = 0.1;
    unicycle.step(xnext, x0, u, dt);
    std::cout << "xnext: " << xnext.transpose() << std::endl;
  }

  dynobench::Problem problem;
  // auto env =
  //     std ::string("../dynobench/") +
  //     "envs/unicycle1_v0/parallelpark_0.yaml";

  auto env = std::string(argv[2]) + "envs/unicycle1_v0/bugtrap_0.yaml";

  problem.read_from_yaml(env.c_str());
  // using state_space_t = dynotree::Combined<double>;
  // state_space_t state_space({"Rn:2", "SO2"});
  using state_space_t = dynotree::R2SO2<double>;
  state_space_t state_space;

  state_space.set_bounds(problem.p_lb, problem.p_ub);

  SSTstar_options options;
  options.max_it = 1e8;
  options.max_num_configs = 1e8;
  options.max_compute_time_ms = 2e3;

  options.max_num_kino_steps = 10; // in each expansion,
  options.goal_tolerance = .2;
  options.collision_resolution = .1; // 10 cm
  options.delta_s = .2;

  // Work on this! we should be able to give the state space at compile time!
  // Then, i can compare timings!
  using planner_t = SSTstar<state_space_t, 3, 2>;
  using trajectory_t = planner_t::trajectory_t;
  using state_t = planner_t::state_t;
  using control_t = planner_t::control_t;
  planner_t rrt;
  rrt.init(3);
  rrt.set_state_space(state_space);
  rrt.set_options(options);

  auto unicycle = dynobench::Model_unicycle1();
  dynobench::load_env(unicycle, problem);

  rrt.set_start(problem.start);
  rrt.set_goal(problem.goal);

  using is_collision_free_fun_t = std::function<bool(state_t)>;

  is_collision_free_fun_t is_collision_free_fun = [&](const state_t &x) {
    return coll_manager.is_collision_free(x);
    // return unicycle.collision_check(x);
  };

  BOOST_TEST(is_collision_free_fun(problem.start));
  BOOST_TEST(is_collision_free_fun(problem.goal));

  using expand_fun_t =
      std::function<void(state_t &, const state_t &, trajectory_t &)>;

  double dt = .1;

  expand_fun_t expand_fun = [&](const state_t &x_start, const state_t &x_goal,
                                trajectory_t &traj) {
    double best_distance_to_goal = std::numeric_limits<double>::max();
    state_t x(x_start.size());
    state_t xnext(x_start.size());
    trajectory_t traj_tmp;
    for (size_t i = 0; i < options.num_expansions; i++) {

      // sample time
      int min_steps = options.min_num_kino_steps;
      int max_steps = options.max_num_kino_steps;
      int num_steps = min_steps + rand() % (max_steps - min_steps + 1);
      x = x_start;
      traj_tmp.states.push_back(x);

      Eigen::VectorXd u_delta = (unicycle.u_ub - unicycle.u_lb);

      Eigen::VectorXd u =
          (unicycle.u_lb - u_delta) +
          (unicycle.u_ub - unicycle.u_lb + 2 * u_delta)
              .cwiseProduct(
                  (Eigen::VectorXd::Random(2) + Eigen::VectorXd::Ones(2)) / 2.);

      u = u.cwiseMax(unicycle.u_lb);
      u = u.cwiseMin(unicycle.u_ub);

      for (size_t t = 0; t < num_steps; t++) {

        unicycle.step(xnext, x, u, dt);
        traj_tmp.controls.push_back(u);
        traj_tmp.states.push_back(xnext);
        x = xnext;
      }
      double distance_to_goal = state_space.distance(xnext, x_goal);
      if (distance_to_goal < best_distance_to_goal) {
        best_distance_to_goal = distance_to_goal;
        traj = traj_tmp;
      }
    }
  };

  rrt.set_is_collision_free_fun(is_collision_free_fun);
  rrt.set_expand_fun(expand_fun);

  auto out = rrt.plan();

  std::cout << "out: " << magic_enum::enum_name(out) << std::endl;

  std::ofstream file_out("/tmp/kino.json");
  nlohmann::json j;
  rrt.get_planner_data(j);
  file_out << j;

  // how much time spent in nearest neighbour?
}
