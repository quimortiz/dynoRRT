#include "dynoRRT/dynorrt_macros.h"
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

BOOST_AUTO_TEST_CASE(t_all_planners_circleworld) {

  std::string env = "benchmark/envs/ballworld2/one_obs.json";

  std::ifstream f(env);

  if (!f.good()) {
    std::stringstream ss;
    ss << "File " << env << " does not exist" << std::endl;
    // THROW_PRETTY_DYNORRT(ss.str());
    THROW_PRETTY_DYNORRT("hello");
  }
  json j;
  f >> j;

  Eigen::VectorXd start = j["start"];
  Eigen::VectorXd goal = j["goal"];
  Eigen::VectorXd lb = j["lb"];
  Eigen::VectorXd ub = j["ub"];

  CollisionManagerBallWorld<2> col_manager;
  col_manager.load_world(env);

  using state_space_t = dynotree::Rn<double, 2>;
  using tree_t = dynotree::KDTree<int, 2, 32, double, state_space_t>;

  state_space_t state_space;
  state_space.set_bounds(Eigen::Vector2d(0, 0), Eigen::Vector2d(3, 3));

  RRTStar<state_space_t, 2> rrt_star;

  RRT_options options{.max_it = 1000,
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
    json j;
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

    namespace fs = std::filesystem;
    fs::path filePath = "/tmp/dynorrt/out.json";

    if (!fs::exists(filePath)) {
      fs::create_directories(filePath.parent_path());
      std::cout << "The directory path has been created." << std::endl;
    } else {
      std::cout << "The file already exists." << std::endl;
    }

    std::ofstream o(filePath.c_str());
    o << std::setw(2) << j << std::endl;
    BOOST_TEST(is_termination_condition_solved(termination_condition));
  }
}
