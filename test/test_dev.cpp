#include "dynoRRT/dynorrt_macros.h"
#include <memory>
#define BOOST_TEST_DYN_LINK

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

#if 0
void solve_rrtstar(json &j) {

  using state_space_t = dynotree::Rn<double, 2>;
  using tree_t = dynotree::KDTree<int, 2, 32, double, state_space_t>;

  state_space_t state_space;
  state_space.set_bounds(lb, ub);

  RRTStar<state_space_t, 2> rrt_star;

  RRT_options options{.max_it = 5000,
                      .goal_bias = 0.05,
                      .collision_resolution = 0.01,
                      .max_step = .2,
                      .max_compute_time_ms = 1e3,
                      .goal_tolerance = 0.001,
                      .max_num_configs = 10000};

  rrt_star.set_options(options);
  rrt_star.set_state_space(state_space);
  rrt_star.set_start(start);
  rrt_star.set_goal(goal);
  rrt_star.init(-1);
  rrt_star.set_collision_manager(&col_manager);

  TerminationCondition termination_condition = rrt_star.plan();

  std::cout << magic_enum::enum_name(termination_condition) << std::endl;

  // export to json
  {
    // json j;
    j["env"] = env;
    j["terminate_status"] = magic_enum::enum_name(termination_condition);
    j["cost_to_come"] = rrt_star.get_cost_to_come();
    j["children"] = rrt_star.get_children();
    j["path"] = rrt_star.get_path();
    j["paths"] = rrt_star.get_paths();
    j["configs"] = rrt_star.get_configs();
    j["sample_configs"] = rrt_star.get_sample_configs();
    j["parents"] = rrt_star.get_parents();
    CHECK_PRETTY_DYNORRT__(ensure_childs_and_parents(rrt_star.get_children(),
                                                     rrt_star.get_parents()));

    BOOST_TEST(is_termination_condition_solved(termination_condition));
  }
}

#endif

template <typename state_space_t, int DIM>
std::shared_ptr<PlannerBase<state_space_t, DIM>>
planner_from_name(const std::string &str) {

  if (str == "RRT") {
    return std::make_shared<RRT<state_space_t, DIM>>();
  } else if (str == "BiRRT") {
    return std::make_shared<BiRRT<state_space_t, DIM>>();
  } else if (str == "RRTConnect") {
    return std::make_shared<RRTConnect<state_space_t, DIM>>();
  } else if (str == "RRTStar") {
    return std::make_shared<RRTStar<state_space_t, DIM>>();
  } else if (str == "PRM") {
    return std::make_shared<PRM<state_space_t, DIM>>();
  } else if (str == "LazyPRM") {
    return std::make_shared<LazyPRM<state_space_t, DIM>>();
  } else {
    THROW_PRETTY_DYNORRT("Unknown planner name: " + str);
  }
}

namespace fs = std::filesystem;
BOOST_AUTO_TEST_CASE(t_all_planners_circleworld) {

  std::string options_cfg_all = "../../planner_config/circleworld_2d_all.toml";
  std::ifstream ifs(options_cfg_all);
  auto cfg = toml::parse(ifs);

  std::vector<std::string> envs =
      toml::find<std::vector<std::string>>(cfg, "envs");
  std::vector<std::string> __planners =
      toml::find<std::vector<std::string>>(cfg, "planners");

  // std::vector<std::string> envs = {
  //     "../../benchmark/envs/ballworld2/one_obs.json",
  //     "../../benchmark/envs/ballworld2/empty.json",
  //     "../../benchmark/envs/ballworld2/bugtrap.json",
  //     "../../benchmark/envs/ballworld2/random1.json"
  // };

  using state_space_t = dynotree::Rn<double, 2>;
  using tree_t = dynotree::KDTree<int, 2, 32, double, state_space_t>;
  using PlannerBase_t = PlannerBase<state_space_t, 2>;

  std::vector<std::shared_ptr<PlannerBase_t>> planners;

  for (auto &planner_name : __planners) {
    std::shared_ptr<PlannerBase_t> planner =
        planner_from_name<state_space_t, 2>(planner_name);
    planners.push_back(planner);
  }

  // std::shared_ptr<PlannerBase_t> rrt =
  //     std::make_shared<RRT<state_space_t, 2>>();
  // std::shared_ptr<PlannerBase_t> birrt =
  //     std::make_shared<BiRRT<state_space_t, 2>>();
  // std::shared_ptr<PlannerBase_t> rrtconnect =
  //     std::make_shared<RRTConnect<state_space_t, 2>>();
  // std::shared_ptr<PlannerBase_t> rrtstar =
  //     std::make_shared<RRTStar<state_space_t, 2>>();

  // planners.push_back(rrtstar);
  // planners.push_back(rrt);
  // planners.push_back(birrt);
  // planners.push_back(rrtconnect);

  std::vector<json> results;

  for (auto &env : envs) {

    std::ifstream f(env);

    if (!f.good()) {
      std::stringstream ss;
      ss << "File " << env << " does not exist" << std::endl;
      // THROW_PRETTY_DYNORRT(ss.str());
      THROW_PRETTY_DYNORRT("hello");
    }
    json j;
    f >> j;

    std::cout << j << std::endl;

    Eigen::VectorXd start = j["start"];
    Eigen::VectorXd goal = j["goal"];
    Eigen::VectorXd lb = j["lb"];
    Eigen::VectorXd ub = j["ub"];

    CollisionManagerBallWorld<2> col_manager;
    col_manager.load_world(env);

    state_space_t state_space;
    state_space.set_bounds(lb, ub);

    for (auto &planner : planners) {

      planner->reset();
      planner->read_cfg_file(options_cfg_all);
      planner->set_state_space(state_space);
      planner->set_start(start);
      planner->set_goal(goal);
      planner->init(2);
      planner->set_collision_manager(&col_manager);
      TerminationCondition termination_condition = planner->plan();
      std::cout << magic_enum::enum_name(termination_condition) << std::endl;
      BOOST_TEST(is_termination_condition_solved(termination_condition));

      json j;
      j["env"] = env;
      j["terminate_status"] = magic_enum::enum_name(termination_condition);
      j["planner_name"] = planner->get_name();
      planner->get_planner_data(j);
      results.push_back(j);
    }
  }

  std::cout << "Writing all results to a file " << std::endl;
  fs::path filePath = "/tmp/dynorrt/out.json";

  if (!fs::exists(filePath)) {
    fs::create_directories(filePath.parent_path());
    std::cout << "The directory path has been created." << std::endl;
  } else {
    std::cout << "The file already exists." << std::endl;
  }

  std::ofstream o(filePath.c_str());
  json jout;
  jout["results"] = results;
  jout["envs"] = envs;
  std::vector<std::string> planners_names;
  for (auto &planner : planners) {
    planners_names.push_back(planner->get_name());
  }
  jout["planners"] = planners_names;
  o << std::setw(2) << jout << std::endl;
}
