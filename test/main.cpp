#include "dynoRRT/dynorrt_macros.h"

#include <string>
#define BOOST_TEST_MODULE test_0
#define BOOST_TEST_DYN_LINK

#include "dynotree/KDTree.h"
#include "magic_enum.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>

#include "dynoRRT/collision_manager.h"
#include "dynoRRT/eigen_conversions.hpp"
#include "dynoRRT/rrt.h"
#include <boost/test/unit_test.hpp>

#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio/algorithm/geometry.hpp"

#include <hpp/fcl/collision_object.h>
#include <hpp/fcl/shape/geometric_shapes.h>
#include <iostream>

#include "dynotree/KDTree.h"
#include "magic_enum.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>

#include "dynoRRT/collision_manager.h"
#include "dynoRRT/eigen_conversions.hpp"
#include "dynoRRT/rrt.h"
#include <boost/test/unit_test.hpp>

#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio/algorithm/geometry.hpp"

#include <hpp/fcl/collision_object.h>
#include <hpp/fcl/shape/geometric_shapes.h>
#include <iostream>

#include "dynoRRT/pin_col_manager.h"

using json = nlohmann::json;

using namespace dynorrt;

struct CircleObstacle {
  Eigen::Vector2d center;
  double radius;
};

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

double length = .5;
double robot_radius = 0.01;

void compute_two_points(const Eigen::Vector3d &x, Eigen::Vector2d &p1,
                        Eigen::Vector2d &p2) {
  p1 = x.head(2);
  p2 = p1 + .5 * Eigen::Vector2d(cos(x[2]), sin(x[2]));
}

double distance_point_to_segment(const Eigen::Vector2d &p1,
                                 const Eigen::Vector2d &p2,
                                 const Eigen::Vector2d &x) {
  double u = (x - p1).dot(p2 - p1) / (p2 - p1).dot(p2 - p1);
  u = std::clamp(u, 0.0, 1.0);
  return (p1 + u * (p2 - p1) - x).norm();
}

bool is_collision(const Eigen::Vector3d &x,
                  std::vector<CircleObstacle> &obstacles, double radius_) {
  Eigen::Vector2d p1, p2;
  compute_two_points(x, p1, p2);
  for (auto &obs : obstacles) {
    if (distance_point_to_segment(p1, p2, obs.center) < radius_ + obs.radius) {
      return true;
    }
  }
  return false;
}

BOOST_AUTO_TEST_CASE(test_ball_world2) {

  srand(0);
  std::cout << "hello world" << std::endl;

  using state_space_t = dynotree::Rn<double, 2>;
  using tree_t = dynotree::KDTree<int, 2, 32, double, state_space_t>;

  state_space_t state_space;
  state_space.set_bounds(Eigen::Vector2d(0, 0), Eigen::Vector2d(3, 3));

  RRT<state_space_t, 2> rrt;

  CollisionManagerBallWorld<2> collision_manager;
  BallObstacle<2> obs1, obs2;
  obs1.center = Eigen::Vector2d(1, 0.4);
  obs1.radius = 0.5;
  obs2.center = Eigen::Vector2d(1, 2);
  obs2.radius = 0.5;

  collision_manager.add_obstacle(obs1);
  collision_manager.add_obstacle(obs2);
  auto col_free = [&](const auto &x) {
    return !collision_manager.is_collision(x);
  };

  RRT_options options{.max_it = 1000,
                      .goal_bias = 0.05,
                      .collision_resolution = 0.01,
                      .max_step = 1.,
                      .max_compute_time_ms = 1e9,
                      .goal_tolerance = 0.001,
                      .max_num_configs = 10000};

  rrt.set_options(options);
  rrt.set_state_space(state_space);
  rrt.set_start(Eigen::Vector2d(0.1, 0.1));
  rrt.set_goal(Eigen::Vector2d(2.0, 0.2));
  rrt.init(-1);

  // rrt.set_is_collision_free_fun(col_free);

  rrt.set_collision_manager(&collision_manager);

  TerminationCondition termination_condition = rrt.plan();

  std::cout << magic_enum::enum_name(termination_condition) << std::endl;
  std::vector<Eigen::Vector2d> path, fine_path, shortcut_path;
  if (termination_condition == TerminationCondition::GOAL_REACHED) {
    path = rrt.get_path();
    fine_path = rrt.get_fine_path(0.1);
    shortcut_path = path_shortcut_v1(path, col_free, rrt.get_state_space(),
                                     options.collision_resolution);
  }

  // export to json
  json j;
  j["terminate_status"] = magic_enum::enum_name(termination_condition);

  std::vector<Eigen::Vector2d> configs, sample_configs;
  std::vector<int> parents;

  configs = rrt.get_configs();
  sample_configs = rrt.get_sample_configs();
  parents = rrt.get_parents();
  path = rrt.get_path();

  j["configs"] = configs;
  j["sample_configs"] = sample_configs;
  j["parents"] = parents;
  j["path"] = path;
  j["fine_path"] = fine_path;
  j["shortcut_path"] = shortcut_path;

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
  BOOST_TEST((termination_condition == TerminationCondition::GOAL_REACHED));

  // continue here!!
}

BOOST_AUTO_TEST_CASE(test_1) {

  srand(0);
  std::cout << "hello world" << std::endl;

  using state_space_t = dynotree::R2SO2<double>;
  using tree_t = dynotree::KDTree<int, 3, 32, double, state_space_t>;

  state_space_t state_space;
  state_space.set_bounds(Eigen::Vector2d(0, 0), Eigen::Vector2d(3, 3));

  RRT<state_space_t, 3> rrt;

  std::vector<CircleObstacle> obstacles;
  obstacles.push_back(CircleObstacle{Eigen::Vector2d(1, 0.4), 0.5});
  obstacles.push_back(CircleObstacle{Eigen::Vector2d(1, 2), 0.5});
  auto col_free = [&](const auto &x) {
    return !is_collision(x, obstacles, robot_radius);
  };

  CollisionManagerBallWorld<2> collision_manager;
  BallObstacle<2> obs1, obs2;
  obs1.center = Eigen::Vector2d(1, 0.4);
  obs1.radius = 0.5;
  obs2.center = Eigen::Vector2d(1, 2);
  obs2.radius = 0.5;

  collision_manager.add_obstacle(obs1);
  collision_manager.add_obstacle(obs2);
  // auto col_free = [&](const auto &x) {
  //   return !collision_manager.is_collision(x);
  // };

  RRT_options options{.max_it = 1000,
                      .goal_bias = 0.05,
                      .collision_resolution = 0.01,
                      .max_step = 1.,
                      .max_compute_time_ms = 1e9,
                      .goal_tolerance = 0.001,
                      .max_num_configs = 10000};

  rrt.set_options(options);
  rrt.set_state_space(state_space);
  rrt.set_start(Eigen::Vector3d(0.1, 0.1, M_PI / 2));
  rrt.set_goal(Eigen::Vector3d(2.0, 0.2, 0));
  rrt.init(-1);

  rrt.set_is_collision_free_fun(col_free);

  TerminationCondition termination_condition = rrt.plan();

  std::cout << magic_enum::enum_name(termination_condition) << std::endl;
  std::vector<Eigen::Vector3d> path, fine_path, shortcut_path;
  if (termination_condition == TerminationCondition::GOAL_REACHED) {
    path = rrt.get_path();
    fine_path = rrt.get_fine_path(0.1);
    shortcut_path = path_shortcut_v1(path, col_free, rrt.get_state_space(),
                                     options.collision_resolution);
  }

  // export to json
  json j;
  j["terminate_status"] = magic_enum::enum_name(termination_condition);

  std::vector<Eigen::Vector3d> configs, sample_configs;
  std::vector<int> parents;

  configs = rrt.get_configs();
  sample_configs = rrt.get_sample_configs();
  parents = rrt.get_parents();
  path = rrt.get_path();

  j["configs"] = configs;
  j["sample_configs"] = sample_configs;
  j["parents"] = parents;
  j["path"] = path;
  j["fine_path"] = fine_path;
  j["shortcut_path"] = shortcut_path;

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
  BOOST_TEST((termination_condition == TerminationCondition::GOAL_REACHED));

  // continue here!!
}

BOOST_AUTO_TEST_CASE(test_birrt) {

  using state_space_t = dynotree::R2SO2<double>;
  using tree_t = dynotree::KDTree<int, 3, 32, double, state_space_t>;

  state_space_t state_space;
  state_space.set_bounds(Eigen::Vector2d(0, 0), Eigen::Vector2d(3, 3));

  BiRRT<state_space_t, 3> birrt;

  std::vector<CircleObstacle> obstacles;

  obstacles.push_back(CircleObstacle{Eigen::Vector2d(1, 0.4), 0.5});
  obstacles.push_back(CircleObstacle{Eigen::Vector2d(1, 2), 0.5});
  obstacles.push_back(CircleObstacle{Eigen::Vector2d(2.2, .9), 0.5});

  std::srand(time(NULL));
  BiRRT_options options{.max_it = 10000,
                        .goal_bias = .8,
                        .collision_resolution = 0.01,
                        .backward_probability = 0.5,
                        .max_step = .5,
                        .max_compute_time_ms = 1e9,
                        .goal_tolerance = 0.001,
                        .max_num_configs = 10000,
                        .max_num_trials_col_free = 1000};

  birrt.set_options(options);
  birrt.set_state_space(state_space);
  birrt.set_start(Eigen::Vector3d(0.1, 0.1, M_PI / 2));
  birrt.set_goal(Eigen::Vector3d(2.0, 0.2, 0));
  birrt.init(-1);

  auto col_free = [&](const auto &x) {
    return !is_collision(x, obstacles, robot_radius);
  };
  birrt.set_is_collision_free_fun(col_free);

  TerminationCondition termination_condition = birrt.plan();

  std::cout << magic_enum::enum_name(termination_condition) << std::endl;
  std::vector<Eigen::Vector3d> path, fine_path, shortcut_path;
  if (termination_condition == TerminationCondition::GOAL_REACHED) {
    path = birrt.get_path();
    fine_path = birrt.get_fine_path(0.1);
    shortcut_path = path_shortcut_v1(path, col_free, birrt.get_state_space(),
                                     options.collision_resolution);
  }

  // export to json
  json j;
  j["terminate_status"] = magic_enum::enum_name(termination_condition);

  std::vector<Eigen::Vector3d> configs, sample_configs;
  std::vector<int> parents;

  configs = birrt.get_configs();
  sample_configs = birrt.get_sample_configs();
  parents = birrt.get_parents();
  path = birrt.get_path();

  j["configs"] = configs;
  j["sample_configs"] = sample_configs;
  j["parents"] = parents;

  j["configs_backward"] = birrt.get_configs_backward();
  j["parents_backward"] = birrt.get_parents_backward();

  j["path"] = path;
  j["fine_path"] = fine_path;
  j["shortcut_path"] = shortcut_path;

  namespace fs = std::filesystem;
  fs::path filePath = "/tmp/dynorrt/test_birrt.json";

  if (!fs::exists(filePath)) {
    fs::create_directories(filePath.parent_path());
    std::cout << "The directory path has been created." << std::endl;
  } else {
    std::cout << "The file already exists." << std::endl;
  }

  std::ofstream o(filePath.c_str());
  o << std::setw(2) << j << std::endl;
  BOOST_TEST((termination_condition == TerminationCondition::GOAL_REACHED));
}

BOOST_AUTO_TEST_CASE(test_rrt_connect) {

  using state_space_t = dynotree::R2SO2<double>;
  using tree_t = dynotree::KDTree<int, 3, 32, double, state_space_t>;

  state_space_t state_space;
  state_space.set_bounds(Eigen::Vector2d(0, 0), Eigen::Vector2d(3, 3));

  RRTConnect<state_space_t, 3> birrt;

  std::vector<CircleObstacle> obstacles;

  obstacles.push_back(CircleObstacle{Eigen::Vector2d(1, 0.4), 0.5});
  obstacles.push_back(CircleObstacle{Eigen::Vector2d(1, 2), 0.5});
  obstacles.push_back(CircleObstacle{Eigen::Vector2d(2.2, .9), 0.5});

  std::srand(time(NULL));
  BiRRT_options options{.max_it = 10000,
                        .goal_bias = .8,
                        .collision_resolution = 0.01,
                        .backward_probability = 0.5,
                        .max_step = 1.,
                        .max_compute_time_ms = 1e9,
                        .goal_tolerance = 0.001,
                        .max_num_configs = 10000,
                        .max_num_trials_col_free = 1000};

  birrt.set_options(options);
  birrt.set_state_space(state_space);
  birrt.set_start(Eigen::Vector3d(0.1, 0.1, M_PI / 2));
  birrt.set_goal(Eigen::Vector3d(2.0, 0.2, 0));
  birrt.init(-1);

  auto col_free = [&](const auto &x) {
    return !is_collision(x, obstacles, robot_radius);
  };
  birrt.set_is_collision_free_fun(col_free);

  TerminationCondition termination_condition = birrt.plan();

  std::cout << magic_enum::enum_name(termination_condition) << std::endl;
  std::vector<Eigen::Vector3d> path, fine_path, shortcut_path;
  if (termination_condition == TerminationCondition::GOAL_REACHED) {
    path = birrt.get_path();
    fine_path = birrt.get_fine_path(0.1);
    shortcut_path = path_shortcut_v1(path, col_free, birrt.get_state_space(),
                                     options.collision_resolution);
  }

  // export to json
  json j;
  j["terminate_status"] = magic_enum::enum_name(termination_condition);

  std::vector<Eigen::Vector3d> configs, sample_configs;
  std::vector<int> parents;

  configs = birrt.get_configs();
  sample_configs = birrt.get_sample_configs();
  parents = birrt.get_parents();
  path = birrt.get_path();

  j["configs"] = configs;
  j["sample_configs"] = sample_configs;
  j["parents"] = parents;

  j["configs_backward"] = birrt.get_configs_backward();
  j["parents_backward"] = birrt.get_parents_backward();

  j["path"] = path;
  j["fine_path"] = fine_path;
  j["shortcut_path"] = shortcut_path;

  namespace fs = std::filesystem;
  fs::path filePath = "/tmp/dynorrt/test_rrt_connect.json";

  if (!fs::exists(filePath)) {
    fs::create_directories(filePath.parent_path());
    std::cout << "The directory path has been created." << std::endl;
  } else {
    std::cout << "The file already exists." << std::endl;
  }

  std::ofstream o(filePath.c_str());
  o << std::setw(2) << j << std::endl;
  BOOST_TEST((termination_condition == TerminationCondition::GOAL_REACHED));
}

#ifndef PINOCCHIO_MODEL_DIR
#define PINOCCHIO_MODEL_DIR

#endif

#if 0
BOOST_AUTO_TEST_CASE(test_PIN_ur5) {

  using namespace pinocchio;


  Eigen::VectorXd q_i(6), q_g(6);

  q_i << 0, -1.5, 2.1, -0.5, -0.5, 0;
  q_g << 3.1, -1.0, 1, -0.5, -0.5, 0;
  std::srand(1);

  std::string robots_model_path =
      "/home/quim/croco/lib/python3.8/site-packages/cmeel.prefix/share/";
  bool use_reduced_model = true;
  if (use_reduced_model) {
    robots_model_path = "/home/quim/stg/quim-example-robot-data";
  }

  std::string urdf_filename =
      robots_model_path +
      std::string(
          "/example-robot-data/robots/ur_description/urdf/ur5_robot.urdf");

  std::string srdf_filename =
      robots_model_path +
      std::string("/example-robot-data/robots/ur_description/srdf/ur5.srdf");

  if (use_reduced_model) {

    urdf_filename =
        robots_model_path +
        std::string(
            "/example-robot-data/robots/ur_description/urdf/ur5_robot.urdf");

    srdf_filename =
        robots_model_path +
        std::string("/example-robot-data/robots/ur_description/srdf/ur5.srdf");
  }

  Model model;
  pinocchio::urdf::buildModel(urdf_filename, model);

  Data data(model);

  GeometryModel geom_model;
  pinocchio::urdf::buildGeom(model, urdf_filename, pinocchio::COLLISION,
                             geom_model, robots_model_path);

  double radius = 0.1;
  double length = 1.0;

  PinExternalObstacle obs1{.shape = "cylinder",
                           .name = "cylinder",
                           .data = Eigen::Vector2d(radius, length),
                           .translation = Eigen::Vector3d(-0.5, 0.4, 0.5),
                           .rotation_angle_axis =
                               Eigen::Vector4d(M_PI / 2, 0, 0, 1)};

  PinExternalObstacle obs2{.shape = "cylinder",
                           .name = "cylinder",
                           .data = Eigen::Vector2d(radius, length),
                           .translation = Eigen::Vector3d(-0.5, -0.4, 0.5),
                           .rotation_angle_axis =
                               Eigen::Vector4d(M_PI / 2, 0, 0, 1)};

  PinExternalObstacle obs3{.shape = "cylinder",
                           .name = "cylinder",
                           .data = Eigen::Vector2d(radius, length),
                           .translation = Eigen::Vector3d(-0.5, 0.7, 0.5),
                           .rotation_angle_axis =
                               Eigen::Vector4d(M_PI / 2, 0, 0, 1)};

  Collision_manager_pinocchio coll_manager;
  coll_manager.add_external_obstacle(obs1);
  coll_manager.add_external_obstacle(obs2);
  coll_manager.add_external_obstacle(obs3);

  coll_manager.set_urdf_filename(urdf_filename);
  coll_manager.set_srdf_filename(srdf_filename);
  coll_manager.set_robots_model_path(robots_model_path);
  coll_manager.build();

  // SE3 placement3 = SE3::Identity();
  // placement3.rotation() = Eigen::AngleAxisd(M_PI / 2,
  // Eigen::Vector3d::UnitZ()); placement3.translation() = Eigen::Vector3d(-0.5,
  // 0.7, 0.5);
  //
  // Model::JointIndex idx_geom1 =
  //     geom_model.addGeometryObject(GeometryObject("cyl1", 0, cyl1,
  //     placement1));
  // Model::JointIndex idx_geom2 =
  //     geom_model.addGeometryObject(GeometryObject("cyl2", 0, cyl2,
  //     placement2));
  // Model::JointIndex idx_geom3 =
  //     geom_model.addGeometryObject(GeometryObject("cyl3", 0, cyl3,
  //     placement3));
  //
  // geom_model.geometryObjects[idx_geom1].parentJoint = 0;
  // geom_model.geometryObjects[idx_geom2].parentJoint = 0;
  // geom_model.geometryObjects[idx_geom3].parentJoint = 0;

  // geom_model.addAllCollisionPairs();
  // pinocchio::srdf::removeCollisionPairs(model, geom_model, srdf_filename);
  // GeometryData geom_data(geom_model);

  using state_space_t = dynotree::Rn<double>;
  using tree_t = dynotree::KDTree<int, -1, 32, double, state_space_t>;

  state_space_t state_space;

  Eigen::VectorXd lb = Eigen::VectorXd::Constant(6, -3.2);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(6, 3.2);
  ub(1) = 0.0;

  if (!coll_manager.is_collision_free(q_i)) {
    THROW_PRETTY_DYNORRT("start is in collision");
  }

  if (!coll_manager.is_collision_free(q_g)) {
    THROW_PRETTY_DYNORRT("goal  is in collision");
  }

  state_space.set_bounds(lb, ub);

  RRT<state_space_t, -1> rrt;
  rrt.init(6);
  rrt.set_state_space(state_space);
  rrt.set_start(q_i);
  rrt.set_goal(q_g);
  rrt.read_cfg_file("../planner_config/rrt_v0_PIN.toml");

  int num_collision_checks = 0;

  double col_time_ms = 0;

  rrt.set_is_collision_free_fun(
      [&](const auto &q) { return coll_manager.is_collision_free(q); });

  rrt.plan();

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

  json j;
  j["path"] = path;
  j["fine_path"] = fine_path;

  namespace fs = std::filesystem;
  fs::path filePath = "/tmp/dynorrt/rrt_path.json";

  if (!fs::exists(filePath)) {
    fs::create_directories(filePath.parent_path());
    std::cout << "The directory path has been created." << std::endl;
  } else {
    std::cout << "The file already exists." << std::endl;
  }

  std::ofstream o(filePath.c_str());
  o << std::setw(2) << j << std::endl;
}
#endif

BOOST_AUTO_TEST_CASE(t_rrtstar) {

  srand(0);

  using state_space_t = dynotree::Rn<double, 2>;
  using tree_t = dynotree::KDTree<int, 2, 32, double, state_space_t>;

  state_space_t state_space;
  state_space.set_bounds(Eigen::Vector2d(0, 0), Eigen::Vector2d(3, 3));

  RRTStar<state_space_t, 2> rrt_star;

  CollisionManagerBallWorld<2> collision_manager;

  RRT_options options{.max_it = 1000,
                      .goal_bias = 0.05,
                      .collision_resolution = 0.01,
                      .max_step = .2,
                      .max_compute_time_ms = 1e9,
                      .goal_tolerance = 0.001,
                      .max_num_configs = 10000};

  rrt_star.set_options(options);
  rrt_star.set_state_space(state_space);
  rrt_star.set_start(Eigen::Vector2d(0.5, 1.5));
  rrt_star.set_goal(Eigen::Vector2d(2.0, 1.5));
  rrt_star.init(-1);

  // rrt.set_is_collision_free_fun(col_free);

  rrt_star.set_collision_manager(&collision_manager);

  TerminationCondition termination_condition = rrt_star.plan();

  std::cout << magic_enum::enum_name(termination_condition) << std::endl;

  // export to json
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

namespace fs = std::filesystem;
BOOST_AUTO_TEST_CASE(t_all_planners_circleworld) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  auto argv = boost::unit_test::framework::master_test_suite().argv;

  if (argc < 2) {
    std::cout << "Usage: ./test_dynorrt <path_to_base_dir>" << std::endl;
    BOOST_TEST(false);
  }
  std::string base_path(argv[1]);

  std::string options_cfg_all =
      base_path + "planner_config/circleworld_2d_all.toml";
  std::ifstream ifs(options_cfg_all);
  if (!ifs.good()) {
    std::stringstream ss;
    ss << "File " << options_cfg_all << " does not exist" << std::endl;
    THROW_PRETTY_DYNORRT(ss.str());
  }
  auto cfg = toml::parse(ifs);

  std::vector<std::string> envs =
      toml::find<std::vector<std::string>>(cfg, "envs");
  std::vector<std::string> __planners =
      toml::find<std::vector<std::string>>(cfg, "planners");

  using state_space_t = dynotree::Rn<double, 2>;
  using tree_t = dynotree::KDTree<int, 2, 32, double, state_space_t>;
  using PlannerBase_t = PlannerBase<state_space_t, 2>;

  std::vector<std::shared_ptr<PlannerBase_t>> planners;

  for (auto &planner_name : __planners) {
    std::shared_ptr<PlannerBase_t> planner =
        planner_from_name<state_space_t, 2>(planner_name);
    planners.push_back(planner);
  }

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
