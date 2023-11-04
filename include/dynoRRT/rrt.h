#include "dynotree/KDTree.h"
#include "magic_enum.hpp"

#include <Eigen/Core>
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

  RRT() = default;
  // RRT(int runtime_dim) {
  //   state_space = t_state_space;
  //   // init_tree(runtime_dim);
  // }

  void set_options(RRT_options t_options) { options = t_options; }

  void set_state_space(StateSpace t_state_space) {
    state_space = t_state_space;
  }

  void set_bounds_to_state(Eigen::VectorXd min, Eigen::VectorXd max) {
    // TODO: remove from here?
    state_space.set_bounds(min, max);
  }

  void
  set_is_collision_free_fun(is_collision_free_fun_t t_is_collision_free_fun) {
    is_collision_free_fun = t_is_collision_free_fun;
  }

  void set_start(state_t t_start) { start = t_start; }
  void set_goal(state_t t_goal) { goal = t_goal; }

  std::vector<state_t> get_path() {

    if (path.size() == 0) {
      std::cout << "Warning: path.size() == 0" << std::endl;
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      return {};
    }
    return path;
  }

  void init_tree(int t_runtime_dim = -1) {
    runtime_dim = t_runtime_dim;
    std::cout << "init tree" << std::endl;
    std::cout << "DIM: " << DIM << std::endl;
    std::cout << "runtime_dim: " << runtime_dim << std::endl;

    if constexpr (DIM == -1) {
      if (runtime_dim == -1) {
        throw std::runtime_error("DIM == -1 and runtime_dim == -1");
      }
    }
    tree = tree_t();
    tree.init_tree(runtime_dim, state_space);
  }
  std::vector<state_t> get_valid_configs() { return valid_configs; }

  std::vector<state_t> get_sample_configs() { return sample_configs; }

  void get_parents(std::vector<int> &t_parents) { t_parents = parents; }

  std::vector<state_t> get_fine_path(double resolution) {

    // in Python
    // for i in range(len(path) - 1):
    //     _start = path[i]
    //     _goal = path[i + 1]
    //     for i in range(N):
    //         out = np.zeros(3)
    //         interpolate_fun(_start, _goal, i / N, out)
    //         plot_robot(ax, out, color="gray", alpha=0.5)
    // in c++

    state_t tmp;

    if constexpr (DIM == -1) {
      if (runtime_dim == -1) {
        throw std::runtime_error("DIM == -1 and runtime_dim == -1");
      }
      tmp.resize(runtime_dim);
    }

    std::vector<state_t> fine_path;
    if (path.size() == 0) {
      std::cout << "Warning: path.size() == 0" << std::endl;
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      return {};
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
    // add last
    fine_path.push_back(path[path.size() - 1]);
    return fine_path;
  }

  TerminationCondition plan() {

    if (path.size() != 0) {
      std::cout << "Warning: path.size() != 0" << std::endl;
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      return TerminationCondition::UNKNOWN;
    }

    bool finished = false;
    state_t x_rand, x_new, x_near;
    if constexpr (DIM == -1) {
      if (runtime_dim == -1) {
        throw std::runtime_error("DIM == -1 and runtime_dim == -1");
      }
      x_rand.resize(runtime_dim);
      x_new.resize(runtime_dim);
      x_near.resize(runtime_dim);
    }

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

      total_distance = 0;
      for (size_t i = 0; i < path.size() - 1; i++) {
        total_distance += state_space.distance(path[i], path[i + 1]);
      }
    }

    // std::cout << valid_configs.size() << std::endl;

    std::cout << "RRT PLAN" << std::endl;
    std::cout << "Terminate status: "
              << magic_enum::enum_name(termination_condition) << std::endl;
    std::cout << "num_it: " << num_it << std::endl;
    std::cout << "valid_configs.size(): " << valid_configs.size() << std::endl;
    std::cout << "compute time (ms): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - tic)
                     .count()
              << std::endl;
    std::cout << "path.size(): " << path.size() << std::endl;
    std::cout << "total_distance: " << total_distance << std::endl;

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
  int runtime_dim = DIM;
  double total_distance = -1;
};
