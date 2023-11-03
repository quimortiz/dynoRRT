#include "dynotree/KDTree.h"
#include "magic_enum.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

template <typename T, typename StateSpace>
bool is_edge_collision_free(T x_start, T x_end,
                            std::function<bool(T)> is_collision_free_fun,
                            StateSpace state_space, double resolution) {

  T tmp = x_start;
  if (is_collision_free_fun(x_end)) {
    int N = int(state_space.distance(x_start, x_end) / resolution) + 1;
    for (int j = 0; j < N; j++) {
      state_space.interpolate(x_start, x_end, double(j) / N, tmp);
      if (!is_collision_free_fun(tmp)) {
        return false;
      }
    }
  } else {
    return false;
  }
  return true;
}

struct RRT_options {
  int max_it = 10000;
  double goal_bias = 0.05;
  double collision_resolution = 0.01;
  double max_step = 0.1;
  double max_compute_time_ms = 1e9;
  double goal_tolerance = 0.001;
  int max_num_configs = 10000;
};

enum class TerminationCondition {
  MAX_IT,
  MAX_TIME,
  GOAL_REACHED,
  MAX_NUM_CONFIGS,
  UNKNOWN
};

template <typename StateSpace, int DIM> class RRT {

public:
  using state_t = Eigen::Matrix<double, DIM, 1>;
  using tree_t = dynotree::KDTree<int, DIM, 32, double, StateSpace>;
  using is_collision_free_fun_t = std::function<bool(state_t)>;

  void set_options(RRT_options t_options) { options = t_options; }

  void set_state_space(StateSpace t_state_space) {
    state_space = t_state_space;
  }

  void
  set_is_collision_free_fun(is_collision_free_fun_t t_is_collision_free_fun) {
    is_collision_free_fun = t_is_collision_free_fun;
  }

  void set_start(state_t t_start) { start = t_start; }
  void set_goal(state_t t_goal) { goal = t_goal; }
  void get_path(std::vector<state_t> &t_path) { t_path = path; }

  void init_tree(int runtime_dim = -1) {
    tree = tree_t(runtime_dim, state_space);
  }

  void get_valid_configs(std::vector<state_t> &t_valid_configs) {
    t_valid_configs = valid_configs;
  }

  void get_sample_configs(std::vector<state_t> &t_sample_configs) {
    t_sample_configs = sample_configs;
  }

  void get_parents(std::vector<int> &t_parents) { t_parents = parents; }

  void get_fine_path(double resolution, std::vector<state_t> &fine_path) {

    // in Python
    // for i in range(len(path) - 1):
    //     _start = path[i]
    //     _goal = path[i + 1]
    //     for i in range(N):
    //         out = np.zeros(3)
    //         interpolate_fun(_start, _goal, i / N, out)
    //         plot_robot(ax, out, color="gray", alpha=0.5)
    // in c++

    for (int i = 0; i < path.size() - 1; i++) {
      state_t _start = path[i];
      state_t _goal = path[i + 1];
      int N = int(state_space.distance(_start, _goal) / resolution) + 1;
      for (int j = 0; j < N; j++) {
        state_t out;
        state_space.interpolate(_start, _goal, double(j) / N, out);
        fine_path.push_back(out);
      }
    }
    // add last
    fine_path.push_back(path[path.size() - 1]);
  }

  TerminationCondition plan() {

    bool finished = false;
    state_t x_rand, x_new, x_near;

    parents.push_back(-1);
    valid_configs.push_back(start);
    tree.addPoint(start, 0);

    int num_it = 0;

    auto tic = std::chrono::steady_clock::now();
    TerminationCondition termination_condition = TerminationCondition::UNKNOWN;
    while (!finished) {
      num_it++;

      if (double(rand()) / RAND_MAX < options.goal_bias) {
        x_rand = goal;
      } else {
        state_space.sample_uniform(x_rand);
      }

      sample_configs.push_back(x_rand);

      auto nn = tree.search(x_rand);
      x_near = valid_configs[nn.id];

      if (nn.distance < options.max_step) {
        x_new = x_rand;
      } else {
        state_space.interpolate(x_near, x_rand, options.max_step / nn.distance,
                                x_new);
      }
      bool is_collision_free =
          is_edge_collision_free(x_near, x_new, is_collision_free_fun,
                                 state_space, options.collision_resolution);

      if (is_collision_free) {
        tree.addPoint(x_new, valid_configs.size());
        valid_configs.push_back(x_new);
        parents.push_back(nn.id);

        if (state_space.distance(x_new, goal) < options.goal_tolerance) {
          finished = true;
          termination_condition = TerminationCondition::GOAL_REACHED;
        }
      }

      // update finish condition
      if (valid_configs.size() > options.max_num_configs) {
        finished = true;
        termination_condition = TerminationCondition::MAX_NUM_CONFIGS;
      }
      if (num_it > options.max_it) {
        finished = true;
        termination_condition = TerminationCondition::MAX_IT;
      }
      double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - tic)
                              .count();
      if (elapsed_ms > options.max_compute_time_ms) {
        finished = true;
        termination_condition = TerminationCondition::MAX_TIME;
      }
    }

    if (termination_condition == TerminationCondition::GOAL_REACHED) {

      int i = valid_configs.size() - 1;
      path.push_back(valid_configs[i]);
      while (i != -1) {
        path.push_back(valid_configs[i]);
        i = parents[i];
      }
    }

    std::cout << valid_configs.size() << std::endl;

    return termination_condition;
  };

private:
  StateSpace state_space;
  state_t start;
  state_t goal;
  tree_t tree;
  is_collision_free_fun_t is_collision_free_fun;
  RRT_options options;
  std::vector<state_t> path;
  std::vector<state_t> valid_configs, sample_configs;
  std::vector<int> parents;
};
