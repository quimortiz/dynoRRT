#include "dynotree/KDTree.h"
#include "magic_enum.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

#include <climits>
#include <cstdlib>
#include <ctime>
#include <iostream>

class pretty_runtime_exception : public std::runtime_error {
  // Adapted from:
  // https://stackoverflow.com/questions/348833/how-to-know-the-exact-line-of-code-where-an-exception-has-been-caused

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

private:
  std::string msg;
};

#define THROW_PRETTY_DYNORRT(arg)                                              \
  throw pretty_runtime_exception(arg, __FILE__, __LINE__, __FUNCTION__);

#define CHECK_PRETTY_DYNORRT(condition, arg)                                   \
  if (!(condition)) {                                                          \
    throw pretty_runtime_exception(arg, __FILE__, __LINE__, __FUNCTION__);     \
  }

#define CHECK_PRETTY_DYNORRT__(condition)                                      \
  if (!(condition)) {                                                          \
    throw pretty_runtime_exception(#condition, __FILE__, __LINE__,             \
                                   __FUNCTION__);                              \
  }

#define MESSAGE_PRETTY_DYNORRT(arg)                                            \
  std::cout << "Message in " << __FUNCTION__ << " (" << __FILE__ << ":"        \
            << __LINE__ << "): " << arg << std::endl;

template <typename T> void print_path(const std::vector<T> &path) {

  std::cout << "PATH" << std::endl;
  std::cout << "path.size(): " << path.size() << std::endl;
  for (size_t i = 0; i < path.size(); i++) {
    std::cout << path[i].transpose() << std::endl;
  }
}

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

template <typename T, typename StateSpace, typename Fun>
std::vector<T> path_shortcut_v1(const std::vector<T> path,
                                Fun &is_collision_free_fun,
                                StateSpace state_space, double resolution) {

  CHECK_PRETTY_DYNORRT__(path.size() >= 2);

  if (path.size() == 2) {
    // [ start, goal ]
    return path;
  }

  std::vector<T> path_out;
  int start_index = 0;
  path_out.push_back(path[start_index]);
  while (true) {
    int target_index = start_index + 2;
    // We know that +1 is always feasible!
    while (target_index < path.size() &&
           is_edge_collision_free(path[start_index], path[target_index],
                                  is_collision_free_fun, state_space,
                                  resolution)) {
      target_index++;
    }
    target_index--; // Reduce one, to get the last collision free edge.

    CHECK_PRETTY_DYNORRT__(target_index >= start_index + 1);
    CHECK_PRETTY_DYNORRT__(target_index < path.size());

    path_out.push_back(path[target_index]);

    if (target_index == path.size() - 1) {
      break;
    }

    start_index = target_index;
  }

  CHECK_PRETTY_DYNORRT__(path_out.size() >= 2);
  CHECK_PRETTY_DYNORRT__(path_out.size() <= path.size());

  MESSAGE_PRETTY_DYNORRT("\nPath_shortcut_v1: From "
                         << path.size() << " to " << path_out.size() << "\n");

  return path_out;
}

template <typename T>
auto trace_back_solution(int id, std::vector<T> configs,
                         const std::vector<int> &parents) {
  std::vector<T> path;
  path.push_back(configs[id]);
  while (parents[id] != -1) {
    id = parents[id];
    path.push_back(configs[id]);
  }
  std::reverse(path.begin(), path.end());
  return path;
};

struct RRT_options {
  int max_it = 10000;
  double goal_bias = 0.05;
  double collision_resolution = 0.01;
  double max_step = 1.;
  double max_compute_time_ms = 1e9;
  double goal_tolerance = 0.001;
  int max_num_configs = 10000;
  bool xrand_collision_free = true;
  int max_num_trials_col_free = 1000;
};

enum class TerminationCondition {
  MAX_IT,
  MAX_TIME,
  GOAL_REACHED,
  MAX_NUM_CONFIGS,
  RUNNING,
  UNKNOWN
};

template <typename StateSpace, int DIM> class PlannerBase {

public:
  using state_t = Eigen::Matrix<double, DIM, 1>;
  using state_cref_t = const Eigen::Ref<const state_t> &;
  using tree_t = dynotree::KDTree<int, DIM, 32, double, StateSpace>;
  using is_collision_free_fun_t = std::function<bool(state_t)>;
  using edge_t = std::pair<state_t, state_t>;

  void set_state_space(StateSpace t_state_space) {
    state_space = t_state_space;
  }

  std::vector<edge_t> get_valid_edges() { return valid_edges; }

  std::vector<edge_t> get_invalid_edges() { return invalid_edges; }

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

  StateSpace &get_state_space() { return state_space; }

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
    CHECK_PRETTY_DYNORRT__(is_collision_free_fun(start));
    CHECK_PRETTY_DYNORRT__(is_collision_free_fun(goal));
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
    init_tree();
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
  std::vector<state_t> configs;
  std::vector<state_t> sample_configs;
  std::vector<int> parents;
  int runtime_dim = DIM;
  double total_distance = -1;
  double collisions_time_ms = 0.;
  int evaluated_edges = 0;
  int infeasible_edges = 0;
  std::vector<std::pair<state_t, state_t>> valid_edges;
  std::vector<std::pair<state_t, state_t>> invalid_edges;

  state_t x_rand, x_new, x_near;
};

struct BiRRT_options {
  int max_it = 10000;
  double goal_bias = 0.05;
  double collision_resolution = 0.01;
  double backward_probability = 0.5;
  double max_step = 0.1;
  double max_compute_time_ms = 1e9;
  double goal_tolerance = 0.001;
  int max_num_configs = 10000;
  bool xrand_collision_free = true;
  int max_num_trials_col_free = 1000;
};

// Continue here!
// template <typename StateSpace, int DIM>

template <typename StateSpace, int DIM>
class BiRRT : public PlannerBase<StateSpace, DIM> {
public:
  using Base = PlannerBase<StateSpace, DIM>;
  using tree_t = typename Base::tree_t;
  using Configs = std::vector<typename Base::state_t>;

  void set_options(BiRRT_options t_options) { options = t_options; }
  BiRRT_options get_options() { return options; }

  Configs &get_configs_backward() { return configs_backward; }
  std::vector<int> &get_parents_backward() { return parents_backward; }

  void init_backward_tree(int t_runtime_dim = -1) {
    this->runtime_dim = t_runtime_dim;
    std::cout << "init tree" << std::endl;
    std::cout << "DIM: " << DIM << std::endl;
    std::cout << "runtime_dim: " << this->runtime_dim << std::endl;

    if constexpr (DIM == -1) {
      if (this->runtime_dim == -1) {
        throw std::runtime_error("DIM == -1 and runtime_dim == -1");
      }
    }
    tree_backward = tree_t();
    tree_backward.init_tree(this->runtime_dim, this->state_space);
  }

  virtual void reset_internal() override {
    this->Base::reset_internal();
    init_backward_tree();
    parents_backward.clear();
    configs_backward.clear();
  }

  virtual void check_internal() const override {

    CHECK_PRETTY_DYNORRT(tree_backward.size() == 0,
                         "tree_backward.size() != 0");
    CHECK_PRETTY_DYNORRT(parents_backward.size() == 0,
                         "parents_backward.size() != 0");

    CHECK_PRETTY_DYNORRT(configs_backward.size() == 0,
                         "configs_backward.size() != 0");

    this->Base::check_internal();
  }

  virtual TerminationCondition plan() override {

    check_internal();

    // forward tree
    this->parents.push_back(-1);
    this->configs.push_back(this->start);
    this->tree.addPoint(this->start, 0);

    // backward tree
    parents_backward.push_back(-1);
    configs_backward.push_back(this->goal);
    tree_backward.addPoint(this->goal, 0);

    int num_it = 0;
    auto tic = std::chrono::steady_clock::now();
    bool path_found = false;

    auto col = [this](const auto &x) {
      return this->Base::is_collision_free_fun_timed(x);
    };

    auto get_elapsed_ms = [&] {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - tic)
          .count();
    };

    auto should_terminate = [&] {
      if (this->configs.size() + configs_backward.size() >
          options.max_num_configs) {
        return TerminationCondition::MAX_NUM_CONFIGS;
      } else if (num_it > options.max_it) {
        return TerminationCondition::MAX_IT;
      } else if (get_elapsed_ms() > options.max_compute_time_ms) {
        return TerminationCondition::MAX_TIME;
      } else if (path_found) {
        return TerminationCondition::GOAL_REACHED;
      } else {
        return TerminationCondition::RUNNING;
      }
    };

    TerminationCondition termination_condition = should_terminate();
    bool expand_forward = true;

    int connect_id_forward = -1;
    int connect_id_backward = -1;

    while (termination_condition == TerminationCondition::RUNNING) {

      expand_forward = static_cast<double>(std::rand()) / RAND_MAX >
                       options.backward_probability;
      auto tgt_configs_ptr =
          expand_forward ? &configs_backward : &this->configs;

      auto src_tree_ptr = expand_forward ? &this->tree : &tree_backward;
      auto src_configs_ptr =
          expand_forward ? &this->configs : &configs_backward;
      auto src_parents_ptr =
          expand_forward ? &this->parents : &parents_backward;

      auto connect_id_src_ptr =
          expand_forward ? &connect_id_forward : &connect_id_backward;
      auto connect_id_tgt_ptr =
          expand_forward ? &connect_id_backward : &connect_id_forward;

      int goal_id = -1;
      bool goal_connection_attempt =
          static_cast<double>(std::rand()) / RAND_MAX < options.goal_bias;
      if (goal_connection_attempt) {
        goal_id = std::rand() % tgt_configs_ptr->size();
        this->x_rand = tgt_configs_ptr->at(goal_id);
      } else {
        if (options.xrand_collision_free) {
          bool is_collision_free = false;
          int num_tries = 0;
          while (!is_collision_free &&
                 num_tries < options.max_num_trials_col_free) {
            this->state_space.sample_uniform(this->x_rand);
            is_collision_free = col(this->x_rand);
            num_tries++;
          }
          CHECK_PRETTY_DYNORRT(is_collision_free,
                               "cannot generate a valid xrand");
        } else {
          this->state_space.sample_uniform(this->x_rand);
        }
      }
      this->sample_configs.push_back(this->x_rand);
      auto nn = src_tree_ptr->search(this->x_rand);
      std::cout << "nn.id: " << nn.id << std::endl;
      this->x_near = src_configs_ptr->at(nn.id);

      bool full_step_attempt = nn.distance < options.max_step;
      if (full_step_attempt) {
        this->x_new = this->x_rand;
      } else {
        this->state_space.interpolate(this->x_near, this->x_rand,
                                      options.max_step / nn.distance,
                                      this->x_new);
      }

      std::cout << "goal goal_connection_attempt" << goal_connection_attempt
                << std::endl;
      std::cout << "expand_forward" << expand_forward << std::endl;
      std::cout << "x_rand: " << this->x_rand.transpose() << std::endl;
      std::cout << "x_new: " << this->x_new.transpose() << std::endl;
      std::cout << "x_near: " << this->x_near.transpose() << std::endl;

      this->evaluated_edges += 1;
      bool is_collision_free = is_edge_collision_free(
          this->x_near, this->x_new, col, this->state_space,
          options.collision_resolution);

      if (is_collision_free) {
        this->valid_edges.push_back({this->x_near, this->x_new});
      } else if (!is_collision_free) {
        this->invalid_edges.push_back({this->x_near, this->x_new});
      }

      this->infeasible_edges += !is_collision_free;

      if (is_collision_free) {
        src_tree_ptr->addPoint(this->x_new, src_tree_ptr->size());
        src_configs_ptr->push_back(this->x_new);
        src_parents_ptr->push_back(nn.id);
        CHECK_PRETTY_DYNORRT__(src_tree_ptr->size() == src_configs_ptr->size());
        CHECK_PRETTY_DYNORRT__(src_tree_ptr->size() == src_parents_ptr->size());

        // TODO: decide if I Should do this or do a second KD tree?
        if (full_step_attempt && goal_connection_attempt) {
          path_found = true;
          MESSAGE_PRETTY_DYNORRT("path found");
          CHECK_PRETTY_DYNORRT__(
              this->state_space.distance(this->x_new,
                                         tgt_configs_ptr->at(goal_id)) <
              options.goal_tolerance);
          *connect_id_src_ptr = src_tree_ptr->size() - 1;
          *connect_id_tgt_ptr = goal_id;
        }
        // Alternative: Second KD tree
      }

      num_it++;
      termination_condition = should_terminate();

    } // RRT CONNECT terminated

    if (termination_condition == TerminationCondition::GOAL_REACHED) {

      // forward tree
      //
      //

      CHECK_PRETTY_DYNORRT__(this->configs.size() >= 1);
      CHECK_PRETTY_DYNORRT__(configs_backward.size() >= 1);

      auto fwd_path =
          trace_back_solution(connect_id_forward, this->configs, this->parents);

      auto bwd_path = trace_back_solution(connect_id_backward, configs_backward,
                                          parents_backward);

      CHECK_PRETTY_DYNORRT__(
          this->state_space.distance(fwd_path.back(), bwd_path.back()) <
          options.goal_tolerance);

      std::reverse(bwd_path.begin(), bwd_path.end());

      print_path(fwd_path);
      print_path(bwd_path);

      this->path.insert(this->path.end(), fwd_path.begin(), fwd_path.end());
      this->path.insert(this->path.end(), bwd_path.begin() + 1, bwd_path.end());
    }
    // else

    return termination_condition;
    //
  }

protected:
  BiRRT_options options;
  typename Base::tree_t tree_backward;
  std::vector<int> parents_backward;
  std::vector<typename Base::state_t> configs_backward;
};

template <typename StateSpace, int DIM>
class RRTConnect : public BiRRT<StateSpace, DIM> {

  using Base = BiRRT<StateSpace, DIM>;
  using state_t = typename Base::Base::state_t;
  using tree_t = typename Base::Base::tree_t;

public:
  virtual TerminationCondition plan() override {

    auto &options = this->options;

    this->check_internal();

    // forward tree
    this->parents.push_back(-1);
    this->configs.push_back(this->start);
    this->tree.addPoint(this->start, 0);

    // backward tree
    this->parents_backward.push_back(-1);
    this->configs_backward.push_back(this->goal);
    this->tree_backward.addPoint(this->goal, 0);

    int num_it = 0;
    auto tic = std::chrono::steady_clock::now();
    bool path_found = false;

    auto col = [this](const auto &x) {
      return this->Base::is_collision_free_fun_timed(x);
    };

    auto get_elapsed_ms = [&] {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - tic)
          .count();
    };

    auto should_terminate = [&] {
      if (this->configs.size() + this->configs_backward.size() >
          options.max_num_configs) {
        return TerminationCondition::MAX_NUM_CONFIGS;
      } else if (num_it > options.max_it) {
        return TerminationCondition::MAX_IT;
      } else if (get_elapsed_ms() > options.max_compute_time_ms) {
        return TerminationCondition::MAX_TIME;
      } else if (path_found) {
        return TerminationCondition::GOAL_REACHED;
      } else {
        return TerminationCondition::RUNNING;
      }
    };

    TerminationCondition termination_condition = should_terminate();
    bool expand_forward = true;

    int connect_id_forward = -1;
    int connect_id_backward = -1;

    struct T {
      std::vector<int> *parents;
      std::vector<state_t> *configs;
      tree_t *tree;
      bool is_forward;
    };

    T Ta{.parents = &this->parents,
         .configs = &this->configs,
         .tree = &this->tree,
         .is_forward = true

    };

    T Tb{.parents = &this->parents_backward,
         .configs = &this->configs_backward,
         .tree = &this->tree_backward,
         .is_forward = false};

    while (termination_condition == TerminationCondition::RUNNING) {

      if (options.xrand_collision_free) {
        bool is_collision_free = false;
        int num_tries = 0;
        while (!is_collision_free &&
               num_tries < options.max_num_trials_col_free) {
          this->state_space.sample_uniform(this->x_rand);
          is_collision_free = col(this->x_rand);
          num_tries++;
        }
        CHECK_PRETTY_DYNORRT(is_collision_free,
                             "cannot generate a valid xrand");
      } else {
        this->state_space.sample_uniform(this->x_rand);
      }

      this->sample_configs.push_back(this->x_rand); // store for debugging
      //
      //

      std::cout << "x_rand: " << this->x_rand.transpose() << std::endl;
      // Expand Ta toward x_rand
      auto nn_a = Ta.tree->search(this->x_rand);
      this->x_near = Ta.configs->at(nn_a.id);

      bool full_step_attempt = nn_a.distance < options.max_step;
      if (full_step_attempt) {
        this->x_new = this->x_rand;
      } else {
        this->state_space.interpolate(this->x_near, this->x_rand,
                                      options.max_step / nn_a.distance,
                                      this->x_new);
      }

      this->evaluated_edges += 1;
      bool is_collision_free = is_edge_collision_free(
          this->x_near, this->x_new, col, this->state_space,
          options.collision_resolution);

      if (is_collision_free) {
        this->valid_edges.push_back({this->x_near, this->x_new});
      } else if (!is_collision_free) {
        this->invalid_edges.push_back({this->x_near, this->x_new});
      }

      if (is_collision_free) {
        Ta.tree->addPoint(this->x_new, Ta.tree->size());
        Ta.configs->push_back(this->x_new);
        Ta.parents->push_back(nn_a.id);

        // RRT Connect Strategy
        this->x_rand = this->x_new;
        auto nn_b = Tb.tree->search(this->x_rand);
        this->x_near = Tb.configs->at(nn_b.id);

        bool full_step_attempt = nn_b.distance < options.max_step;
        if (full_step_attempt) {
          this->x_new = this->x_rand;
        } else {
          this->state_space.interpolate(this->x_near, this->x_rand,
                                        options.max_step / nn_b.distance,
                                        this->x_new);
        }
        this->evaluated_edges += 1;
        bool is_collision_free = is_edge_collision_free(
            this->x_near, this->x_new, col, this->state_space,
            options.collision_resolution);
        if (is_collision_free) {
          Tb.tree->addPoint(this->x_new, Tb.tree->size());
          Tb.configs->push_back(this->x_new);
          Tb.parents->push_back(nn_b.id);

          if (full_step_attempt) {
            path_found = true;

            if (Ta.is_forward) {
              connect_id_forward = Ta.tree->size() - 1;
              connect_id_backward = Tb.tree->size() - 1;
            } else {
              connect_id_forward = Tb.tree->size() - 1;
              connect_id_backward = Ta.tree->size() - 1;
            }
          }
        }
      }

      num_it++;
      termination_condition = should_terminate();
      std::swap(Ta, Tb);

    } // RRT CONNECT terminated

    if (termination_condition == TerminationCondition::GOAL_REACHED) {

      // forward tree
      //
      //

      CHECK_PRETTY_DYNORRT__(this->configs.size() >= 1);
      CHECK_PRETTY_DYNORRT__(this->configs_backward.size() >= 1);

      auto fwd_path =
          trace_back_solution(connect_id_forward, this->configs, this->parents);

      auto bwd_path = trace_back_solution(
          connect_id_backward, this->configs_backward, this->parents_backward);

      CHECK_PRETTY_DYNORRT__(
          this->state_space.distance(fwd_path.back(), bwd_path.back()) <
          options.goal_tolerance);

      std::reverse(bwd_path.begin(), bwd_path.end());

      print_path(fwd_path);
      print_path(bwd_path);

      this->path.insert(this->path.end(), fwd_path.begin(), fwd_path.end());
      this->path.insert(this->path.end(), bwd_path.begin() + 1, bwd_path.end());
    }
    // else

    return termination_condition;
    //
  }
};

template <typename StateSpace, int DIM>
class RRT : public PlannerBase<StateSpace, DIM> {

  using Base = PlannerBase<StateSpace, DIM>;

public:
  RRT() = default;

  void set_options(RRT_options t_options) { options = t_options; }

  virtual TerminationCondition plan() override {

    Base::check_internal();

    Base::parents.push_back(-1);
    Base::configs.push_back(Base::start);
    Base::tree.addPoint(Base::start, 0);

    int num_it = 0;

    auto tic = std::chrono::steady_clock::now();
    bool path_found = false;

    auto col = [this](const auto &x) {
      return this->Base::is_collision_free_fun_timed(x);
    };

    auto get_elapsed_ms = [&] {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - tic)
          .count();
    };

    auto should_terminate = [&] {
      if (Base::configs.size() > options.max_num_configs) {
        return TerminationCondition::MAX_NUM_CONFIGS;
      } else if (num_it > options.max_it) {
        return TerminationCondition::MAX_IT;
      } else if (get_elapsed_ms() > options.max_compute_time_ms) {
        return TerminationCondition::MAX_TIME;
      } else if (path_found) {
        return TerminationCondition::GOAL_REACHED;
      } else {
        return TerminationCondition::RUNNING;
      }
    };

    TerminationCondition termination_condition = should_terminate();

    while (termination_condition == TerminationCondition::RUNNING) {

      if (static_cast<double>(std::rand()) / RAND_MAX < options.goal_bias) {
        this->x_rand = Base::goal;
      } else {
        if (options.xrand_collision_free) {
          bool is_collision_free = false;
          int num_tries = 0;
          while (!is_collision_free &&
                 num_tries < options.max_num_trials_col_free) {
            Base::state_space.sample_uniform(this->x_rand);
            is_collision_free = col(this->x_rand);
            num_tries++;
          }
          CHECK_PRETTY_DYNORRT(is_collision_free,
                               "cannot generate a valid xrand");
        } else {
          Base::state_space.sample_uniform(this->x_rand);
        }
      }

      Base::sample_configs.push_back(this->x_rand);

      auto nn = Base::tree.search(this->x_rand);
      this->x_near = Base::configs[nn.id];

      if (nn.distance < options.max_step) {
        this->x_new = this->x_rand;
      } else {
        Base::state_space.interpolate(this->x_near, this->x_rand,
                                      options.max_step / nn.distance,
                                      this->x_new);
      }

      Base::evaluated_edges += 1;
      bool is_collision_free = is_edge_collision_free(
          this->x_near, this->x_new, col, Base::state_space,
          options.collision_resolution);
      Base::infeasible_edges += !is_collision_free;

      if (is_collision_free) {
        this->valid_edges.push_back({this->x_near, this->x_new});
      } else if (!is_collision_free) {
        this->invalid_edges.push_back({this->x_near, this->x_new});
      }

      if (is_collision_free) {
        Base::tree.addPoint(this->x_new, Base::configs.size());
        Base::configs.push_back(this->x_new);
        Base::parents.push_back(nn.id);

        if (Base::state_space.distance(this->x_new, Base::goal) <
            options.goal_tolerance) {
          path_found = true;
          MESSAGE_PRETTY_DYNORRT("path found");
        }
      }

      num_it++;
      termination_condition = should_terminate();

    } // RRT terminated

    if (termination_condition == TerminationCondition::GOAL_REACHED) {

      int i = Base::configs.size() - 1;

      CHECK_PRETTY_DYNORRT__(
          Base::state_space.distance(Base::configs[i], Base::goal) <
          options.goal_tolerance);

      Base::path = trace_back_solution(i, Base::configs, Base::parents);

      CHECK_PRETTY_DYNORRT__(
          Base::state_space.distance(Base::path[0], Base::start) < 1e-6);
      CHECK_PRETTY_DYNORRT__(
          Base::state_space.distance(Base::path.back(), Base::goal) < 1e-6);

      Base::total_distance = 0;
      for (size_t i = 0; i < Base::path.size() - 1; i++) {

        double distance =
            Base::state_space.distance(Base::path[i], Base::path[i + 1]);
        MESSAGE_PRETTY_DYNORRT(distance);
        CHECK_PRETTY_DYNORRT__(distance <= options.max_step + 1e-6);

        Base::total_distance +=
            Base::state_space.distance(Base::path[i], Base::path[i + 1]);
      }
    }

    MESSAGE_PRETTY_DYNORRT("Output from RRT PLANNER");
    std::cout << "Terminate status: "
              << magic_enum::enum_name(termination_condition) << std::endl;
    std::cout << "num_it: " << num_it << std::endl;
    std::cout << "configs.size(): " << Base::configs.size() << std::endl;
    std::cout << "compute time (ms): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - tic)
                     .count()
              << std::endl;
    std::cout << "collisions time (ms): " << Base::collisions_time_ms
              << std::endl;
    std::cout << "evaluated_edges: " << Base::evaluated_edges << std::endl;
    std::cout << "infeasible_edges: " << Base::infeasible_edges << std::endl;
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
  //   std::vector<state_t> configs, sample_configs;
  //   std::vector<int> parents;
  //   int runtime_dim = DIM;
  //   double total_distance = -1;
  //   double collisions_time_ms = 0;
};
