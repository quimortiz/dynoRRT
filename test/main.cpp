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

using json = nlohmann::json;

using namespace dynorrt;

struct CircleObstacle {
  Eigen::Vector2d center;
  double radius;
};

// def compute_two_points(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
//     """
//     x: 3D vector (x, y, theta)
//
//     """
//     p1 = x[0:2]
//     p2 = p1 + length * np.array([np.cos(x[2]), np.sin(x[2])])
//     return p1, p2

double length = .5;
double robot_radius = 0.01;

void compute_two_points(const Eigen::Vector3d &x, Eigen::Vector2d &p1,
                        Eigen::Vector2d &p2) {
  p1 = x.head(2);
  p2 = p1 + .5 * Eigen::Vector2d(cos(x[2]), sin(x[2]));
}

// def distance_point_to_segment(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray)
// -> float:
//     """
//     p1, p2: two points defining the segment
//     p3: the point
//     """
//     u = np.dot(p3 - p1, p2 - p1) / np.dot(p2 - p1, p2 - p1)
//     u = np.clip(u, 0, 1)
//     return np.linalg.norm(p1 + u * (p2 - p1) - p3)

double distance_point_to_segment(const Eigen::Vector2d &p1,
                                 const Eigen::Vector2d &p2,
                                 const Eigen::Vector2d &x) {
  double u = (x - p1).dot(p2 - p1) / (p2 - p1).dot(p2 - p1);
  u = std::clamp(u, 0.0, 1.0);
  return (p1 + u * (p2 - p1) - x).norm();
}

// def is_collision(x: np.ndarray) -> bool:
//     """
//     x: 3D vector (x, y, theta)
//
//     """
//     p1, p2 = compute_two_points(x)
//     for obs in obstacles:
//         if distance_point_to_segment(p1, p2, obs[0]) < radius + obs[1]:
//             return True
//     return False

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
                        .xrand_collision_free = true,
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
                        .xrand_collision_free = true,
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

struct PinExternalObstacle {

  std::string shape;
  std::string name;
  Eigen::VectorXd data;
  Eigen::VectorXd translation;
  Eigen::VectorXd rotation_angle_axis;
};

class Collision_manager_pinocchio {

public:
  void set_urdf_filename(const std::string &t_urdf_filename) {
    urdf_filename = t_urdf_filename;
  }

  void set_srdf_filename(const std::string &t_srdf_filename) {
    srdf_filename = t_srdf_filename;
  }
  void set_robots_model_path(const std::string &t_robot_model_path) {
    robots_model_path = t_robot_model_path;
  }

  void add_external_obstacle(const PinExternalObstacle &obs) {
    obstacles.push_back(obs);
  }

  void build() {

    using namespace pinocchio;
    pinocchio::urdf::buildModel(urdf_filename, model);

    data = pinocchio::Data(model);

    pinocchio::urdf::buildGeom(model, urdf_filename, pinocchio::COLLISION,
                               geom_model, robots_model_path);

    double radius = 0.1;
    double length = 1.0;

    std::string obstacle_base_name = "external_obs_";
    int i = 0;
    for (auto &obs : obstacles) {

      boost::shared_ptr<fcl::CollisionGeometry> geometry;

      if (obs.name == "cylinder") {
        CHECK_PRETTY_DYNORRT(obs.data.size() == 2,
                             "cylinder needs 2 parameters");
        double radius = obs.data[0];
        double length = obs.data[1];
        geometry = boost::make_shared<fcl::Cylinder>(radius, length);
      } else {
        THROW_PRETTY_DYNORRT("only cylinder are supported");
      }

      SE3 placement = SE3::Identity();
      CHECK_PRETTY_DYNORRT(obs.translation.size() == 3,
                           "translation needs 3 parameters");
      CHECK_PRETTY_DYNORRT(obs.rotation_angle_axis.size() == 4,
                           "rotation_angle_axis needs 4 parameters");
      placement.rotation() = Eigen::AngleAxisd(
          obs.rotation_angle_axis[0], obs.rotation_angle_axis.tail<3>());
      placement.translation() = obs.translation;

      Model::JointIndex idx_geom = geom_model.addGeometryObject(GeometryObject(
          obstacle_base_name + std::to_string(i) + "_" + obs.name, 0, geometry,
          placement));

      geom_model.geometryObjects[idx_geom].parentJoint = 0;
      i++;
    }

    geom_model.addAllCollisionPairs();
    pinocchio::srdf::removeCollisionPairs(model, geom_model, srdf_filename);
    geom_data = GeometryData(geom_model);
    build_done = true;
  }

  bool is_collision_free(const Eigen::VectorXd &q) {

    num_collision_checks++;
    auto tic = std::chrono::high_resolution_clock::now();
    if (!build_done) {
      THROW_PRETTY_DYNORRT("build not done");
    }
    bool out = !computeCollisions(model, data, geom_model, geom_data, q, true);
    time_ms += std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now() - tic)
                   .count() /
               1000.0;
    return out;
  };

  void reset_counters() {
    time_ms = 0;
    num_collision_checks = 0;
  }
  int get_num_collision_checks() { return num_collision_checks; }
  double get_time_ms() { return time_ms; }

private:
  std::string urdf_filename;
  std::string srdf_filename;
  std::string env_urdf;
  std::string robots_model_path;

  std::vector<PinExternalObstacle> obstacles;
  pinocchio::Model model;
  pinocchio::Data data;
  pinocchio::GeometryData geom_data;
  bool build_done = false;
  pinocchio::GeometryModel geom_model;
  double time_ms = 0;
  int num_collision_checks = 0;
};

BOOST_AUTO_TEST_CASE(test_PIN_ur5) {

  using namespace pinocchio;

  // Eige ::VectorXd q_i(6), q_g(6);
  // q_i << 1.0, -1.5, 2.1, -0.5, -0.5, 0;
  // q_g << 3.0, -1.0, 1, -0.5, -0.5, 0;

  // q_i = np.array([0, -1.5, 2.1, -0.5, -0.5, 0])
  // q_g = np.array([3.1, -1.0, 1, -0.5, -0.5, 0])

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

  // rrt.set_is_collision_free_fun([&](const auto &q) {
  //   num_collision_checks++;
  //
  //   auto tic = std::chrono::high_resolution_clock::now();
  //   bool out = !computeCollisions(model, data, geom_model, geom_data, q,
  //   true); auto toc = std::chrono::high_resolution_clock::now(); auto
  //   duration =
  //       std::chrono::duration_cast<std::chrono::microseconds>(toc - tic);
  //   col_time_ms += duration.count() / 1000.0;
  //   return out;
  // });

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
  // BOOST_TEST((termination_condition ==
  // TerminationCondition::GOAL_REACHED));
}

BOOST_AUTO_TEST_CASE(t_rrtstar) {

  srand(0);

  using state_space_t = dynotree::Rn<double, 2>;
  using tree_t = dynotree::KDTree<int, 2, 32, double, state_space_t>;

  state_space_t state_space;
  state_space.set_bounds(Eigen::Vector2d(0, 0), Eigen::Vector2d(3, 3));

  RRTStar<state_space_t, 2> rrt_star;

  CollisionManagerBallWorld<2> collision_manager;
  // BallObstacle<2> obs1, obs2;
  // obs1.center = Eigen::Vector2d(1, 0.4);
  // obs1.radius = 0.5;
  // obs2.center = Eigen::Vector2d(1, 2);
  // obs2.radius = 0.5;
  //
  // collision_manager.add_obstacle(obs1);
  // collision_manager.add_obstacle(obs2);

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
