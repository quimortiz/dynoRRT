#define BOOST_TEST_MODULE test_0
#define BOOST_TEST_DYN_LINK

#include "dynotree/KDTree.h"
#include "eigen_conversions.hpp"
#include "magic_enum.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>

#include "dynoRRT/rrt.h"
#include <boost/test/unit_test.hpp>

using json = nlohmann::json;

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
double radius = 0.01;

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
  // obstacles.push_back(CircleObstacle{Eigen::Vector2d(2.2, .9), 0.5});

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
  rrt.init_tree();

  auto col_free = [&](const auto &x) {
    return !is_collision(x, obstacles, radius);
  };
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
  birrt.init_tree();
  birrt.init_backward_tree();

  auto col_free = [&](const auto &x) {
    return !is_collision(x, obstacles, radius);
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
  birrt.init_tree();
  birrt.init_backward_tree();

  auto col_free = [&](const auto &x) {
    return !is_collision(x, obstacles, radius);
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
