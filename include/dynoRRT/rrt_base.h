#pragma once

#include "dynotree/KDTree.h"

#include "collision_manager.h"
#include "dynorrt_macros.h"
#include "options.h"
#include <chrono>

namespace dynorrt {
template <typename StateSpace, int DIM> class PlannerBase {

public:
  using state_t = Eigen::Matrix<double, DIM, 1>;
  using state_ref_t = Eigen::Ref<state_t>;
  using state_cref_t = const Eigen::Ref<const state_t> &;
  using tree_t = dynotree::KDTree<int, DIM, 32, double, StateSpace>;
  using is_collision_free_fun_t = std::function<bool(state_t)>;
  using is_collision_free_fun_parallel_t = std::function<bool(const std::vector<state_t> & )>;
  using sample_fun_t = std::function<void(state_ref_t)>;
  using edge_t = std::pair<state_t, state_t>;

  virtual ~PlannerBase() = default;

  virtual void reset() {
    THROW_PRETTY_DYNORRT("Not implemented in base class!");
  }

  virtual std::string get_name() {
    THROW_PRETTY_DYNORRT("Not implemented in base class!");
  }

  void set_state_space(StateSpace t_state_space) {
    state_space = t_state_space;
    tree = tree_t();
    tree.init_tree(runtime_dim, state_space);
  }

  void set_state_space_with_string(
      const std::vector<std::string> &state_space_vstring) {
    if constexpr (std::is_same<StateSpace, dynotree::Combined<double>>::value) {
      state_space = StateSpace(state_space_vstring);
      tree = tree_t();
      tree.init_tree(runtime_dim, state_space);
    } else {
      THROW_PRETTY_DYNORRT("use set_state_string only with dynotree::Combined");
    }
  }

  virtual void print_options(std::ostream &out = std::cout) {
    THROW_PRETTY_DYNORRT("Not implemented in base class!");
  }

  void virtual set_options_from_toml(toml::value &cfg) {
    THROW_PRETTY_DYNORRT("Not implemented in base class!");
  }

  virtual void read_cfg_file(const std::string &cfg_file) {

    std::ifstream ifs(cfg_file);
    if (!ifs) {
      THROW_PRETTY_DYNORRT("Cannot open cfg_file: " + cfg_file);
    }
    auto cfg = toml::parse(ifs);
    set_options_from_toml(cfg);
  }

  void virtual read_cfg_string(const std::string &cfg_string) {
    std::stringstream ss(cfg_string);
    auto cfg = toml::parse(ss);
    set_options_from_toml(cfg);
  }

  std::vector<edge_t> get_valid_edges() { return valid_edges; }

  std::vector<edge_t> get_invalid_edges() { return invalid_edges; }

  void set_bounds_to_state(const Eigen::VectorXd &lb,
                           const Eigen::VectorXd &ub) {
    CHECK_PRETTY_DYNORRT__(lb.size() == ub.size());

    if constexpr (DIM == -1) {
      CHECK_PRETTY_DYNORRT__(runtime_dim != -1);
      CHECK_PRETTY_DYNORRT__(lb.size() == runtime_dim);
      CHECK_PRETTY_DYNORRT__(ub.size() == runtime_dim);
    }

    state_space.set_bounds(lb, ub);
  }

  void
  set_is_collision_free_fun(is_collision_free_fun_t t_is_collision_free_fun) {
    is_collision_free_fun = t_is_collision_free_fun;
  }

  void
  set_is_collision_free_fun_parallel(is_collision_free_fun_parallel_t t_is_collision_free_fun_parallel) {
    is_collision_free_fun_parallel = t_is_collision_free_fun_parallel;
  }



  void set_sample_fun(sample_fun_t t_sample_fun) {
    sample_fun = t_sample_fun;
    custom_sample_fun = true;
  }

  void
  set_collision_manager(CollisionManagerBallWorld<DIM> *collision_manager) {
    is_collision_free_fun = [collision_manager](state_t x) {
      return !collision_manager->is_collision(x);
    };
  }

  // TODO: timing collisions take a lot of overhead, specially for
  // very simple envs where collisions are very fast.
  bool is_collision_free_fun_timed(state_cref_t x) {
    auto tic = std::chrono::high_resolution_clock::now();
    bool is_collision_free = is_collision_free_fun(x);
    double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now() - tic)
                            .count();
    collisions_time_ms += elapsed_ns / double(1e6);
    number_collision_checks++;
    return is_collision_free;
  }

  StateSpace &get_state_space() { return state_space; }

  void set_start(state_cref_t t_start) { start = t_start; }
  void set_goal(state_cref_t t_goal) { goal = t_goal; }

  // NOTE: Goal list has priority over goal
  void set_goal_list(std::vector<state_t> t_goal_list) {
    goal_list = t_goal_list;
  }

  std::vector<state_t> get_path() {

    if (path.size() == 0) {
      std::cout << "Warning: path.size() == 0" << std::endl;
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      return {};
    }
    return path;
  }

  virtual void init(int t_runtime_dim = -1) {

    runtime_dim = t_runtime_dim;

    if constexpr (DIM == -1) {
      x_rand.resize(this->runtime_dim);
      x_new.resize(this->runtime_dim);
      x_near.resize(this->runtime_dim);
      if (runtime_dim == -1) {
        throw std::runtime_error("DIM == -1 and runtime_dim == -1");
      }
    }
    tree = tree_t();
    tree.init_tree(runtime_dim, state_space);
  }
  std::vector<state_t> get_configs() { return configs; }

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

  virtual void get_planner_data(json &j) {
    j["planner_name"] = this->get_name();
    j["path"] = path;
    j["fine_path"] = get_fine_path(0.01);
    j["configs"] = configs;
    j["sample_configs"] = sample_configs;
    j["parents"] = parents;
    j["evaluated_edges"] = evaluated_edges;
    j["infeasible_edges"] = infeasible_edges;
    j["total_distance"] = total_distance;
    j["collisions_time_ms"] = collisions_time_ms;
    j["number_collision_checks"] = number_collision_checks;
    j["valid_edges"] = valid_edges;
    j["invalid_edges"] = invalid_edges;

    // THROW_PRETTY_DYNORRT("Not implemented in base class!");
  }

  virtual void check_internal() const {

    CHECK_PRETTY_DYNORRT__(tree.size() == 0);
    CHECK_PRETTY_DYNORRT__(parents.size() == 0);
    CHECK_PRETTY_DYNORRT__(configs.size() == 0);
    CHECK_PRETTY_DYNORRT__(sample_configs.size() == 0);
    CHECK_PRETTY_DYNORRT__(path.size() == 0);
    CHECK_PRETTY_DYNORRT__(evaluated_edges == 0);
    CHECK_PRETTY_DYNORRT__(infeasible_edges == 0);
    CHECK_PRETTY_DYNORRT__(total_distance == -1);
    CHECK_PRETTY_DYNORRT__(collisions_time_ms == 0);

    if (goal_list.size() > 0) {
      for (const auto &g : goal_list) {
        CHECK_PRETTY_DYNORRT__(is_collision_free_fun(g));
      }
    } else {
      CHECK_PRETTY_DYNORRT__(is_collision_free_fun(goal));
    }

    CHECK_PRETTY_DYNORRT__(is_collision_free_fun(start));
  }

  virtual void reset_internal() {
    parents.clear();
    configs.clear();
    sample_configs.clear();
    path.clear();
    evaluated_edges = 0;
    infeasible_edges = 0;
    total_distance = -1;
    collisions_time_ms = 0;
    init();
  }

  virtual TerminationCondition plan() {
    THROW_PRETTY_DYNORRT("Not implemented in base class!");
  }

  void set_dev_mode_parallel(bool t_dev_mode_parallel) {
    dev_mode_parallel = t_dev_mode_parallel;
  }


protected:
  StateSpace state_space;
  state_t start;
  // User can define a goal or goal_list.
  // NOTE: Goal list has priority over goal
  state_t goal;
  std::vector<state_t> goal_list;
  tree_t tree;
  is_collision_free_fun_t is_collision_free_fun = [](const auto &) {
    THROW_PRETTY_DYNORRT("You have to define a collision free fun!");
    return false;
  };

  is_collision_free_fun_parallel_t is_collision_free_fun_parallel = [](const auto &) {
    THROW_PRETTY_DYNORRT("You have to define a collision free fun parallel!");
    return false;
  };


  sample_fun_t sample_fun;
  bool custom_sample_fun = false;
  std::vector<state_t> path;
  std::vector<state_t> configs;
  std::vector<state_t> sample_configs; // TODO: only with  flag
  std::vector<int> parents;
  int runtime_dim = DIM;
  double total_distance = -1;
  double collisions_time_ms = 0.;
  int number_collision_checks = 0;
  int evaluated_edges = 0;
  int infeasible_edges = 0;
  bool dev_mode_parallel = false;
  std::vector<std::pair<state_t, state_t>>
      valid_edges; // TODO: only with a flag
  std::vector<std::pair<state_t, state_t>>
      invalid_edges; // TODO: only rrth a flag

  state_t x_rand, x_new, x_near;
};

} // namespace dynorrt
