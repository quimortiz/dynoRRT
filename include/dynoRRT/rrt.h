#include "dynotree/KDTree.h"
#include "magic_enum.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

class pretty_runtime_exception : public std::runtime_error {
  // adapted from
  // https://stackoverflow.com/questions/348833/how-to-know-the-exact-line-of-code-where-an-exception-has-been-caused
  std::string msg;

public:
  pretty_runtime_exception(const std::string &arg, const char *file, int line,
                           const char *function)
      : std::runtime_error(arg) {
    std::ostringstream o;
    o << "Error in " << function << " (" << file << ":" << line << "): " << arg
      << std::endl;
    msg = o.str();
  }
  ~pretty_runtime_exception() throw() {}
  const char *what() const throw() { return msg.c_str(); }
};

#define THROW_PRETTY_DYNORRT(arg)                                              \
  throw pretty_runtime_exception(arg, __FILE__, __LINE__, __FUNCTION__);

template <typename T, typename StateSpace, typename Fun>
bool is_edge_collision_free(T x_start, T x_end, Fun &is_collision_free_fun,
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
  bool xrand_collision_free = true;
  int max_num_trias_xrand_col_free = 1000;
};

enum class TerminationCondition {
  MAX_IT,
  MAX_TIME,
  GOAL_REACHED,
  MAX_NUM_CONFIGS,
  UNKNOWN
};

template <typename StateSpace, int DIM> class PlannerBase {

public:
  using state_t = Eigen::Matrix<double, DIM, 1>;
  using state_cref_t = const Eigen::Ref<const state_t> &;
  using tree_t = dynotree::KDTree<int, DIM, 32, double, StateSpace>;
  using is_collision_free_fun_t = std::function<bool(state_t)>;

  void set_state_space(StateSpace t_state_space) {
    state_space = t_state_space;
  }

  void set_bounds_to_state(Eigen::VectorXd min, Eigen::VectorXd max) {
    // TODO: remove from here?

    if (min.size() != max.size()) {
      throw std::runtime_error("min.size() != max.size()");
    }
    if constexpr (DIM == -1) {
      if (runtime_dim == -1) {
        throw std::runtime_error("DIM == -1 and runtime_dim == -1");
      }
      if (min.size() != runtime_dim) {
        throw std::runtime_error("min.size() != runtime_dim");
      }
    } else {
      if (min.size() != DIM) {
        throw std::runtime_error("min.size() != DIM");
      }
    }

    state_space.set_bounds(min, max);
  }

  void
  set_is_collision_free_fun(is_collision_free_fun_t t_is_collision_free_fun) {
    is_collision_free_fun = t_is_collision_free_fun;
  }

  bool is_collision_free_fun_timed(state_cref_t x) {
    auto tic = std::chrono::steady_clock::now();
    bool is_collision_free = is_collision_free_fun(x);
    double elapsed_mcs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - tic)
                             .count();
    collisions_time_ms += elapsed_mcs / 1000.;
    return is_collision_free;
  }

  void set_start(state_cref_t t_start) { start = t_start; }
  void set_goal(state_cref_t t_goal) { goal = t_goal; }

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

  std::vector<int> get_parents() { return parents; }

  std::vector<state_t> get_fine_path(double resolution) {

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
    fine_path.push_back(path[path.size() - 1]);
    return fine_path;
  }

  virtual TerminationCondition plan() = 0;

protected:
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
  double collisions_time_ms = 0.;
};

template <typename StateSpace, int DIM>
class RRT : public PlannerBase<StateSpace, DIM> {

  using Base = PlannerBase<StateSpace, DIM>;

public:
  RRT() = default;

  void set_options(RRT_options t_options) { options = t_options; }

  virtual TerminationCondition plan() override {

    if (Base::path.size() != 0) {
      std::cout << "Warning: path.size() != 0" << std::endl;
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      return TerminationCondition::UNKNOWN;
    }

    typename Base::state_t x_rand, x_new, x_near;

    if constexpr (DIM == -1) {
      if (Base::runtime_dim == -1) {
        THROW_PRETTY_DYNORRT("DIM == -1 and runtime_dim == -1");
      }
      x_rand.resize(Base::runtime_dim);
      x_new.resize(Base::runtime_dim);
      x_near.resize(Base::runtime_dim);
    }

    if (Base::parents.size()) {
      THROW_PRETTY_DYNORRT("parents.size() != 0");
    }
    if (Base::valid_configs.size()) {
      THROW_PRETTY_DYNORRT("valid_configs.size() != 0");
    }
    if (Base::sample_configs.size()) {
      THROW_PRETTY_DYNORRT("sample_configs.size() != 0");
    }
    if (Base::path.size()) {
      THROW_PRETTY_DYNORRT("path.size() != 0");
    }

    Base::parents.push_back(-1);
    Base::valid_configs.push_back(Base::start);
    Base::tree.addPoint(Base::start, 0);

    int num_it = 0;

    auto tic = std::chrono::steady_clock::now();
    TerminationCondition termination_condition = TerminationCondition::UNKNOWN;
    bool finished = false;

    auto col = [&](const auto &x) {
      return Base::is_collision_free_fun_timed(x);
    };

    while (!finished) {
      num_it++;

      if (double(rand()) / RAND_MAX < options.goal_bias) {
        x_rand = Base::goal;
      } else {
        if (options.xrand_collision_free) {
          bool is_collision_free = false;
          int num_tries = 0;
          while (!is_collision_free &&
                 num_tries < options.max_num_trias_xrand_col_free) {
            Base::state_space.sample_uniform(x_rand);
            is_collision_free = Base::is_collision_free_fun_timed(x_rand);
            num_tries++;
          }
        } else {
          Base::state_space.sample_uniform(x_rand);
        }
      }

      Base::sample_configs.push_back(x_rand);

      auto nn = Base::tree.search(x_rand);
      x_near = Base::valid_configs[nn.id];

      if (nn.distance < options.max_step) {
        x_new = x_rand;
      } else {
        Base::state_space.interpolate(x_near, x_rand,
                                      options.max_step / nn.distance, x_new);
      }
      bool is_collision_free = is_edge_collision_free(
          x_near, x_new, col,
          // Base::is_collision_free_fun_timed
          Base::state_space, options.collision_resolution);

      if (is_collision_free) {
        Base::tree.addPoint(x_new, Base::valid_configs.size());
        Base::valid_configs.push_back(x_new);
        Base::parents.push_back(nn.id);

        if (Base::state_space.distance(x_new, Base::goal) <
            options.goal_tolerance) {
          finished = true;
          termination_condition = TerminationCondition::GOAL_REACHED;
        }
      }

      // update finish condition
      if (Base::valid_configs.size() > options.max_num_configs) {
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

      int i = Base::valid_configs.size() - 1;
      Base::path.push_back(Base::valid_configs[i]);
      if (Base::state_space.distance(Base::valid_configs[i], Base::goal) >
          options.goal_tolerance) {
        throw std::runtime_error("state_space.distance(valid_configs[i], goal) "
                                 "< options.goal_tolerance");
      }
      i = Base::parents[i];
      while (i != -1) {
        Base::path.push_back(Base::valid_configs[i]);
        i = Base::parents[i];
      }

      std::reverse(Base::path.begin(), Base::path.end());

      Base::total_distance = 0;
      for (size_t i = 0; i < Base::path.size() - 1; i++) {
        Base::total_distance +=
            Base::state_space.distance(Base::path[i], Base::path[i + 1]);
      }
    }

    // std::cout << valid_configs.size() << std::endl;

    std::cout << "RRT PLAN" << std::endl;
    std::cout << "Terminate status: "
              << magic_enum::enum_name(termination_condition) << std::endl;
    std::cout << "num_it: " << num_it << std::endl;
    std::cout << "valid_configs.size(): " << Base::valid_configs.size()
              << std::endl;
    std::cout << "compute time (ms): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - tic)
                     .count()
              << std::endl;
    std::cout << "collisions time (ms): " << Base::collisions_time_ms
              << std::endl;
    std::cout << "path.size(): " << Base::path.size() << std::endl;
    std::cout << "total_distance: " << Base::total_distance << std::endl;

    return termination_condition;
  };

protected:
  //   StateSpace state_space;
  //   state_t start;
  //   state_t goal;
  //   tree_t tree;
  //   is_collision_free_fun_t is_collision_free_fun;
  RRT_options options;
  //   std::vector<state_t> path;
  //   std::vector<state_t> valid_configs, sample_configs;
  //   std::vector<int> parents;
  //   int runtime_dim = DIM;
  //   double total_distance = -1;
  //   double collisions_time_ms = 0;
};
