#include <Eigen/Dense>
#include <dynoRRT/collision_manager.h>
#include <dynoRRT/rrt.h>
#include <iostream>

using namespace dynorrt;

int main() {

  std::cout << "hello world" << std::endl;
  srand(0);
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

  rrt.set_is_collision_free_fun(
      [&](const auto &x) { return !collision_manager.is_collision(x); });

  TerminationCondition termination_condition = rrt.plan();

  std::cout << magic_enum::enum_name(termination_condition) << std::endl;
  std::vector<Eigen::Vector2d> path, fine_path;
  // shortcut_path;
  if (termination_condition == TerminationCondition::GOAL_REACHED) {
    path = rrt.get_path();
    fine_path = rrt.get_fine_path(0.1);
    // shortcut_path = path_shortcut_v1(path, col_free, rrt.get_state_space(),
    //                                  options.collision_resolution);
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
  // j["shortcut_path"] = shortcut_path;

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
  if (termination_condition == TerminationCondition::GOAL_REACHED) {
    std::cout << "Path found" << std::endl;
  } else {
    std::cout << "Path not found" << std::endl;
    throw std::runtime_error("An error has occured! -- Path not found");
  }
}
