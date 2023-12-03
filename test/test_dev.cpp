#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"

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

#include <hpp/fcl/collision_object.h>
#include <hpp/fcl/shape/geometric_shapes.h>

#include "dynoRRT/pin_col_manager.h"

BOOST_AUTO_TEST_CASE(t_pin_all) {

  // std::string options_cfg_all = "../../planner_config/pinocchio_all.toml";

  std::string env = "../../benchmark/envs/pinocchio/ur5_bin.json";

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
  Eigen::VectorXd ub = j["ub"];

  std::string urdf = j["urdf"];
  std::string srdf = j["srdf"];

  std::string robots_model_path =
      "/home/quim/stg/quim-example-robot-data/example-robot-data/";
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

  using state_space_t = dynotree::Rn<double>;
  using tree_t = dynotree::KDTree<int, -1, 32, double, state_space_t>;

  state_space_t state_space;

  if (!coll_manager.is_collision_free(start)) {
    THROW_PRETTY_DYNORRT("start is in collision");
  }

  if (!coll_manager.is_collision_free(goal)) {
    THROW_PRETTY_DYNORRT("goal  is in collision");
  }

  state_space.set_bounds(lb, ub);

  RRT<state_space_t, -1> rrt;
  rrt.init(6);
  rrt.set_state_space(state_space);
  rrt.set_start(start);
  rrt.set_goal(goal);
  rrt.read_cfg_file("../../planner_config/rrt_v0_PIN.toml");

  rrt.set_is_collision_free_fun(
      [&](const auto &q) { return coll_manager.is_collision_free(q); });

  auto termination_condition = rrt.plan();

  BOOST_TEST(is_termination_condition_solved(termination_condition));

  std::cout << "num_collision_checks: "
            << coll_manager.get_num_collision_checks() << std::endl;
  std::cout << "Average time per collision check [ms]: "
            << coll_manager.get_time_ms() /
                   coll_manager.get_num_collision_checks()
            << std::endl;

  auto path = rrt.get_path();
  auto fine_path = rrt.get_fine_path(.1);

  std::cout << "DONE" << std::endl;
  std::cout << path.size() << std::endl;

  {
    json j;
    j["path"] = path;
    j["fine_path"] = fine_path;
    j["configs"] = rrt.get_configs();
    j["parents"] = rrt.get_parents();
    j["invalid_edges"] = rrt.get_invalid_edges();

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
  }
}
