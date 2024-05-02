// clang-format off
// I need to include pinocchio before the others, otherwise it will not compile
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/parsers/srdf.hpp"
#include "dynobench/motions.hpp"
#include "dynoRRT/pin_col_manager.h"
// clang-format on
#define BOOST_TEST_DYN_LINK

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

using namespace dynorrt;
using PlannerRn = PlannerBase<dynotree::Rn<double>, -1>;
using PlannerR3SO3 = PlannerBase<dynotree::R3SO3<double>, 7>;

using state_space_t = dynotree::Combined<double>;
using PlannerBase_t = PlannerBase<state_space_t, -1>;
using PlannerPtr = std::shared_ptr<PlannerBase_t>;

PlannerPtr get_planner_from_name(const std::string &planner_name) {

  if (planner_name == "RRT") {
    return std::make_shared<RRT<state_space_t, -1>>();
  } else if (planner_name == "RRTConnect") {
    return std::make_shared<RRTConnect<state_space_t, -1>>();
  } else if (planner_name == "BiRRT") {
    return std::make_shared<BiRRT<state_space_t, -1>>();
  } else if (planner_name == "RRTStar") {
    return std::make_shared<RRTStar<state_space_t, -1>>();
  } else if (planner_name == "LazyPRM") {
    return std::make_shared<LazyPRM<state_space_t, -1>>();
  } else if (planner_name == "PRM") {
    return std::make_shared<PRM<state_space_t, -1>>();
  } else if (planner_name == "KinoRRT") {
    return std::make_shared<KinoRRT<state_space_t, -1, -1>>();
  } else if (planner_name == "SSTstar") {
    return std::make_shared<SSTstar<state_space_t, -1, -1>>();
  } else {
    THROW_PRETTY_DYNORRT("Planner not found:" + planner_name);
  }
}

// for (auto &planner_name : __planners) {
//   std::shared_ptr<PlannerBase_t> planner =
//       planner_from_name<state_space_t, 2>(planner_name);
//   planners.push_back(planner);
// }

BOOST_AUTO_TEST_CASE(t_pin_all) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  auto argv = boost::unit_test::framework::master_test_suite().argv;

  if (argc < 2) {
    std::cout << "Usage: ./test_dynorrt <path_to_base_dir>" << std::endl;
    BOOST_TEST(false);
  }
  std::string base_path(argv[1]);

  std::string options_cfg_all = base_path + "planner_config/PIN_all.toml";
  std::ifstream ifs(options_cfg_all);
  if (!ifs.good()) {
    std::stringstream ss;
    ss << "File " << options_cfg_all << " does not exist" << std::endl;
    THROW_PRETTY_DYNORRT(ss.str());
  }

  auto cfg = toml::parse(ifs);
  std::cout << cfg << std::endl;

  std::vector<std::string> envs =
      toml::find<std::vector<std::string>>(cfg, "envs");
  std::vector<std::string> planner_names =
      toml::find<std::vector<std::string>>(cfg, "planners");

  // std::vector<std::string> envs = {
  // "../../benchmark/envs/pinocchio/ur5_bin.json",
  // "../../benchmark/envs/pinocchio/se3_window.json",
  // "../../benchmark/envs/pinocchio/ur5_two_arms.json",
  // "../../benchmark/envs/pinocchio/point_mass_cables.json"};

  // std::vector<std::string> planner_names = {"RRT", "RRTConnect", "BiRRT"};

  std::vector<json> results;

  for (auto &env : envs) {
    std::ifstream i(env);
    if (!i.is_open()) {
      std::cout << "Error opening file" << std::endl;
      return;
    }

    nlohmann::json j;
    i >> j;

    Eigen::VectorXd start = j["start"];
    Eigen::VectorXd goal = j["goal"];
    Eigen::VectorXd lb = j["lb"];
    std::vector<std::string> state_space_vstr = j["state_space"];
    Eigen::VectorXd ub = j["ub"];

    std::string urdf = j["urdf"];
    std::string srdf = j["srdf"];

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

    if (!coll_manager.is_collision_free(start)) {
      THROW_PRETTY_DYNORRT("start is in collision");
    }

    if (!coll_manager.is_collision_free(goal)) {
      THROW_PRETTY_DYNORRT("goal  is in collision");
    }

    state_space_t state_space(state_space_vstr);
    // state_space_t state_space; // Use this for Rn
    state_space.set_bounds(lb, ub);

    for (auto &planner_name : planner_names) {
      srand(0);
      PlannerPtr planner = get_planner_from_name(planner_name);

      planner->init(start.size());
      planner->set_state_space(state_space);
      planner->set_start(start);
      planner->set_goal(goal);
      planner->read_cfg_file(options_cfg_all);

      planner->set_is_collision_free_fun(
          [&](const auto &q) { return coll_manager.is_collision_free(q); });

      auto termination_condition = planner->plan();

      std::cout << "termination condition is "
                << magic_enum::enum_name(termination_condition) << std::endl;

      std::cout << "is_termination_condition_solved: "
                << is_termination_condition_solved(termination_condition)
                << std::endl;

      std::cout << planner->get_name() << std::endl;
      std::cout << "planner_name " << planner_name << std::endl;
      std::cout << "problem " << env << std::endl;
      BOOST_TEST(is_termination_condition_solved(termination_condition));

      std::cout << "num_collision_checks: "
                << coll_manager.get_num_collision_checks() << std::endl;
      std::cout << "Average time per collision check [ms]: "
                << coll_manager.get_time_ms() /
                       coll_manager.get_num_collision_checks()
                << std::endl;

      auto path = planner->get_path();
      auto fine_path = planner->get_fine_path(.1);

      std::cout << "DONE" << std::endl;
      std::cout << path.size() << std::endl;

      json j;
      j["env"] = env;
      j["terminate_status"] = magic_enum::enum_name(termination_condition);
      planner->get_planner_data(j);
      results.push_back(j);
    }
  }

  namespace fs = std::filesystem;
  fs::path filePath = "/tmp/dynorrt/out.json";

  if (!fs::exists(filePath)) {
    fs::create_directories(filePath.parent_path());
    std::cout << "The directory path has been created." << std::endl;
  } else {
    std::cout << "The file already exists." << std::endl;
  }

  std::ofstream o(filePath.c_str());
  o << std::setw(2) << json(results) << std::endl;
}

BOOST_AUTO_TEST_CASE(t_col_manager) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  auto argv = boost::unit_test::framework::master_test_suite().argv;

  if (argc < 2) {
    std::cout << "Usage: ./test_dynorrt <path_to_base_dir>" << std::endl;
    BOOST_TEST(false);
    return;
  }
  std::string base_path(argv[1]);

  std::string options_cfg_all = base_path + "planner_config/PIN_all.toml";
  std::ifstream ifs(options_cfg_all);
  if (!ifs.good()) {
    std::stringstream ss;
    ss << "File " << options_cfg_all << " does not exist" << std::endl;
    THROW_PRETTY_DYNORRT(ss.str());
  }

  auto cfg = toml::parse(ifs);
  std::cout << cfg << std::endl;

  std::vector<std::string> envs = {
      "benchmark/envs/pinocchio/ur5_bin.json",
      "benchmark/envs/pinocchio/se3_window.json",
      "benchmark/envs/pinocchio/ur5_two_arms.json",
      "benchmark/envs/pinocchio/point_mass_cables.json"};

  // std::vector<std::string> planner_names = {"RRT", "RRTConnect", "BiRRT"};

  std::vector<json> results;

  for (auto &env : envs) {
    std::ifstream i(base_path + env);
    if (!i.is_open()) {
      std::cout << "Error opening file: " << env << std::endl;
      return;
    }

    std::cout << "env is " << env << std::endl;

    nlohmann::json j;
    i >> j;

    Eigen::VectorXd start = j["start"];
    Eigen::VectorXd goal = j["goal"];
    Eigen::VectorXd lb = j["lb"];
    std::vector<std::string> state_space_vstr = j["state_space"];
    Eigen::VectorXd ub = j["ub"];

    std::string urdf = j["urdf"];
    std::string srdf = j["srdf"];

    std::string robots_model_path = base_path + std::string(j["base_path"]);
    std::cout << "robots_model_path: " << robots_model_path << std::endl;
    // std::string robots_model_path =
    urdf = robots_model_path + urdf;
    srdf = robots_model_path + srdf;

    Collision_manager_pinocchio coll_manager;
    coll_manager.set_urdf_filename(urdf);
    coll_manager.set_srdf_filename(srdf);
    coll_manager.set_robots_model_path(robots_model_path);
    int edge_num_threads = 3;
    coll_manager.set_edge_parallel(edge_num_threads);
    coll_manager.build();

    // check a line between start and goal
    state_space_t state_space(state_space_vstr);

    int num_points = 1000;
    {
      if (!coll_manager.is_collision_free(start)) {
        THROW_PRETTY_DYNORRT("start is in collision");
      }

      if (!coll_manager.is_collision_free(goal)) {
        THROW_PRETTY_DYNORRT("goal  is in collision");
      }

      int free = 0;
      Eigen::VectorXd out(state_space.get_runtime_dim());

      auto tic = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < num_points; i++) {
        double alpha = (double)i / (double)(num_points - 1);
        state_space.interpolate(start, goal, alpha, out);
        auto col_free = coll_manager.is_collision_free(out);
        free += col_free;
        if (!col_free)
          break;
      }
      auto toc = std::chrono::high_resolution_clock::now();
      std::cout << "elapsed time [ms]"
                << std::chrono::duration_cast<std::chrono::microseconds>(toc -
                                                                         tic)
                           .count() /
                       1000.0
                << std::endl;
    }
    {
      std::cout << "test parallel" << std::endl;
      int num_threads = 4;

      if (!coll_manager.is_collision_free_parallel(start, num_threads)) {
        THROW_PRETTY_DYNORRT("start is in collision");
      }

      if (!coll_manager.is_collision_free_parallel(goal, num_threads)) {
        THROW_PRETTY_DYNORRT("goal  is in collision");
      }

      int free = 0;
      Eigen::VectorXd out(state_space.get_runtime_dim());

      auto tic = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < num_points; i++) {
        double alpha = (double)i / (double)(num_points - 1);
        state_space.interpolate(start, goal, alpha, out);
        auto col_free =
            coll_manager.is_collision_free_parallel(out, num_threads);
        free += col_free;
        if (!col_free)
          break;
      }

      auto toc = std::chrono::high_resolution_clock::now();
      double time_ms =
          std::chrono::duration_cast<std::chrono::microseconds>(toc - tic)
              .count() /
          1000.0;
      std::cout << "elapsed time [ms]" << time_ms << std::endl;
      std::cout << "elapesed time * num_threads [ms] " << time_ms * num_threads
                << std::endl;
      std::cout << "free/total " << free << "/" << num_points << " = "
                << double(free) / num_points << std::endl;
    }

    {

      std::vector<Eigen::VectorXd> q_set(
          num_points, Eigen::VectorXd::Zero(state_space.get_runtime_dim()));

      for (size_t i = 0; i < num_points; i++) {
        double alpha = (double)i / (double)(num_points - 1);
        state_space.interpolate(start, goal, alpha, q_set.at(i));
      }

      int counter_infeas = 0;
      int counter_feas = 0;
      auto tic = std::chrono::high_resolution_clock::now();
      bool edge_free = coll_manager.is_collision_free_set(
          q_set, true, &counter_infeas, &counter_feas);

      auto toc = std::chrono::high_resolution_clock::now();
      double time_ms =
          std::chrono::duration_cast<std::chrono::microseconds>(toc - tic)
              .count() /
          1000.0;
      // BOOST_TEST(counter_feas + counter_infeas == num_points);
      std::cout << "time [ms] " << time_ms << std::endl;
      std::cout << "time * num_threads [ms] " << time_ms * edge_num_threads
                << std::endl;
      std::cout << "infeas / total " << counter_infeas << "/" << num_points
                << " = " << double(counter_infeas) / num_points << std::endl;
    }
    // It is working better now. Lets integrate this by, and ask user to give
    // how many threads.
    // TODO: broad phase collision check! -- Tonight?

    // test how to check points in parallel in a kdtree
    //
    // Lets try a broad phase collision check!!
    // TODO:continue here

    // TODO: try to compile with pinocchio from conda-forge? Will it be faster?
  }
}

BOOST_AUTO_TEST_CASE(t_hello_world_dynobench) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  auto argv = boost::unit_test::framework::master_test_suite().argv;

  if (argc < 2) {
    std::cout << "Usage: ./test_dynorrt <path_to_base_dir>" << std::endl;
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
    auto env =
        std ::string("../dynobench/") + "envs/unicycle1_v0/parallelpark_0.yaml";

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

  std::string urdf = base_path + "/models/unicycle_parallel_park.urdf";
  std::string srdf = base_path + "/models/unicycle_parallel_park.srdf";
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

  if (argc < 2) {
    std::cout << "Usage: ./test_dynorrt <path_to_base_dir>" << std::endl;
    BOOST_TEST(false);
    return;
  }

  std::string base_path(argv[1]);

  std::string urdf = base_path + "/models/unicycle_parallel_park.urdf";
  std::string srdf = base_path + "/models/unicycle_parallel_park.srdf";
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
  auto env =
      std ::string("../dynobench/") + "envs/unicycle1_v0/parallelpark_0.yaml";

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

  if (argc < 2) {
    std::cout << "Usage: ./test_dynorrt <path_to_base_dir>" << std::endl;
    BOOST_TEST(false);
    return;
  }

  std::srand(0);

  std::string base_path(argv[1]);

  // std::string urdf = base_path + "/models/unicycle_parallel_park.urdf";
  // std::string srdf = base_path + "/models/unicycle_parallel_park.srdf";

  std::string urdf = base_path + "/models/unicycle_bugtrap.urdf";
  std::string srdf = base_path + "/models/unicycle_bugtrap.srdf";

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

  auto env = std ::string("../dynobench/") + "envs/unicycle1_v0/bugtrap_0.yaml";

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
