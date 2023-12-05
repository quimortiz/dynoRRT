// clang-format off
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/parsers/srdf.hpp"
#include "dynoRRT/pin_col_manager.h"
#include <memory>

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

#include "dynoRRT/collision_manager.h"
#include "dynoRRT/eigen_conversions.hpp"
#include "dynoRRT/rrt.h"
#include <boost/test/unit_test.hpp>

#include <hpp/fcl/collision_object.h>
#include <hpp/fcl/shape/geometric_shapes.h>
#include <iostream>

#include "dynoRRT/dynorrt_macros.h"

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
  srand(0);

  std::string options_cfg_all = "../../planner_config/PIN_all.toml";
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

    std::string robots_model_path = j["base_path"];
    // std::string robots_model_path =
    urdf = robots_model_path + urdf;
    srdf = robots_model_path + srdf;

    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf, model);

    pinocchio::Data data(model);

    pinocchio::GeometryModel geom_model;
    pinocchio::urdf::buildGeom(model, urdf, pinocchio::COLLISION, geom_model,
                               robots_model_path);

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
      PlannerPtr planner = get_planner_from_name(planner_name);

      planner->init(start.size());
      planner->set_state_space(state_space);
      planner->set_start(start);
      planner->set_goal(goal);
      planner->read_cfg_file(options_cfg_all);
      // rrt_v0_PIN.toml");

      planner->set_is_collision_free_fun(
          [&](const auto &q) { return coll_manager.is_collision_free(q); });

      auto termination_condition = planner->plan();

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

#if 0
  Eigen::VectorXd w1 = Eigen::VectorXd::Zero(12);
  w1 << 1.88495559, -1.8849556, 0.62831853, 0., 0., 0., -0.9424778, -0.9424778,
      1.57079633, 0., 0., 0.;
  Eigen::VectorXd w2 = Eigen::VectorXd::Zero(12);
  w2 << 1.88495559, -1.8849556, 0.62831853, 0., 0., 0., -2.82743339, -0.9424778,
      1.57079633, 0., 0., 0.;

  using state_t = Eigen::VectorXd;
  state_t tmp;
  double resolution = .1;
  auto get_fine_path = [&](std::vector<Eigen::VectorXd> path) {

    int runtime_dim  = 12;

      tmp.resize(runtime_dim);

    std::vector<state_t> fine_path;
    if (path.size() == 0) {
      std::cout << "Warning: path.size() == 0" << std::endl;
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      return std::vector<state_t>{};
    }

    for (int i = 0; i < path.size() - 1; i++) {
      state_t _start = path[i];
      state_t _goal = path[i + 1];
      int N = int(state_space.distance(_start, _goal) / resolution) + 1;
      for (int j = 0; j < N; j++) {
        state_space.interpolate(_start, _goal, double(j) / N, tmp);
        fine_path.push_back(tmp);
      }
    }
    fine_path.push_back(path[path.size() - 1]);
    return fine_path;
  };
  auto _fine_path = get_fine_path({start,w1, w2, goal});
  // auto _fine_path = get_fine_path({start, goal});
  for (auto &s : _fine_path) {
    std::cout << s.transpose() << std::endl;
    BOOST_TEST(coll_manager.is_collision_free(s));
    BOOST_TEST(state_space.check_bounds(s));
  }

#endif
