
#define BOOST_TEST_MODULE test_0
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test_suite.hpp>

#include "dynoRRT/dynorrt_macros.h"
#include <string>

#include "dynoRRT/pin_col_manager.h"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"
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

#include "dynoRRT/pin_ik_solver.h"

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

BOOST_AUTO_TEST_CASE(t_pin_all) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  auto argv = boost::unit_test::framework::master_test_suite().argv;

  if (argc < 2) {
    std::cout
        << "Usage: ./test_dynorrt [boost_test_options] -- <path_to_base_dir>"
        << std::endl;
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
    std::ifstream i(base_path + env);
    if (!i.is_open()) {
      THROW_PRETTY_DYNORRT("Error opening file: " + env);
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
    std::cout
        << "Usage: ./test_dynorrt [boost_test_options] -- <path_to_base_dir>"
        << std::endl;
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

    int edge_num_threads = 4;
    Collision_manager_pinocchio coll_manager;
    coll_manager.set_urdf_filename(urdf);
    coll_manager.set_srdf_filename(srdf);
    coll_manager.set_robots_model_path(robots_model_path);
    coll_manager.set_edge_parallel(edge_num_threads);
    coll_manager.set_use_pool(true);

    // NOTE: I do this so that collision checks
    // take more time, to see speedup from parallel col check
    coll_manager.set_use_aabb(false);

    coll_manager.build();

    state_space_t state_space(state_space_vstr);

    int num_points = 10000;
    if (!coll_manager.is_collision_free(start)) {
      THROW_PRETTY_DYNORRT("start is in collision");
    }

    if (!coll_manager.is_collision_free(goal)) {
      THROW_PRETTY_DYNORRT("goal  is in collision");
    }

    std::vector<Eigen::VectorXd> q_set;
    q_set.reserve(num_points);
    for (size_t i = 0; i < num_points; i++) {
      if (i % 2 == 0)
        q_set.push_back(start);
      else
        q_set.push_back(goal);
    }

    {
      auto tic = std::chrono::high_resolution_clock::now();

      bool out = true;
      for (auto &q : q_set) {
        out = out && coll_manager.is_collision_free(q);
      }
      BOOST_TEST(out);
      auto toc = std::chrono::high_resolution_clock::now();
      double dt =
          std::chrono::duration_cast<std::chrono::microseconds>(toc - tic)
              .count() /
          1000.0;

      std::cout << "SINGLE THREAD elapsed time [ms]" << dt << std::endl;
      std::cout << "time per collision check [ms] " << dt / num_points
                << std::endl;
    }
    {
      auto tic = std::chrono::high_resolution_clock::now();
      bool out = coll_manager.is_collision_free_set(q_set);
      BOOST_TEST(out);
      auto toc = std::chrono::high_resolution_clock::now();
      std::cout << "MULTI THREAD elapsed time [ms]"
                << std::chrono::duration_cast<std::chrono::microseconds>(toc -
                                                                         tic)
                           .count() /
                       1000.0
                << std::endl;
    }
  }
}

BOOST_AUTO_TEST_CASE(test_pin_ik) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  auto argv = boost::unit_test::framework::master_test_suite().argv;

  if (argc < 2) {
    std::cout << "Usage: ./test_dynorrt [] -- <path_to_base_dir>" << std::endl;
    BOOST_TEST(false);
    return;
  }

  std::string base_path(argv[1]);

  Pin_ik_solver pin_ik_solver;
  std::string env = "benchmark/envs/pinocchio/ur5_bin.json";

  std::ifstream i(base_path + env);
  if (!i.is_open()) {
    THROW_PRETTY_DYNORRT("Error opening file: " + env);
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
  urdf = robots_model_path + urdf;
  srdf = robots_model_path + srdf;
  pin_ik_solver.set_urdf_filename(urdf);
  pin_ik_solver.set_srdf_filename(srdf);
  pin_ik_solver.build();

  pin_ik_solver.set_frame_name("tool0");

  lb = start - 0.1 * Eigen::VectorXd::Ones(start.size());
  ub = start + 0.1 * Eigen::VectorXd::Ones(start.size());
  pin_ik_solver.set_bounds(lb, ub);

  auto [mptr, dptr] = pin_ik_solver.get_model_data_ptr();

  pinocchio::framesForwardKinematics(*mptr, *dptr, start);

  int frame_id = mptr->getFrameId("tool0");
  const pinocchio::SE3 iMd = dptr->oMf[frame_id];

  // pinocchio::SE3 pq_des;
  // pq_des.setIdentity();
  pin_ik_solver.set_pq_des(iMd);
  std::cout << "checking start" << std::endl;
  pin_ik_solver.get_cost(start);

  std::cout << "checking goal" << std::endl;
  pin_ik_solver.get_cost(goal);

  auto status = pin_ik_solver.solve_ik();
  BOOST_TEST(bool(status == IKStatus::SUCCESS));

  //
}
