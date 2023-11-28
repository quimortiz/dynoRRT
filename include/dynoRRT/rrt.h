#pragma once

#include "collision_manager.h"
#include "dynoRRT/dynorrt_macros.h"
#include "dynoRRT/toml_extra_macros.h"

#include "dynotree/KDTree.h"
#include "magic_enum.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/fusion/functional/invocation/invoke.hpp>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <ctime>
#include <iostream>

#include <algorithm>
#include <cstdlib>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::json;

using namespace dynorrt;

inline auto index_2d_to_1d_symmetric(int i, int j, int n) {
  if (i > j) {
    std::swap(i, j);
  }
  return i * n + j;
};
inline auto index_1d_to_2d(int index, int n) {
  int i = index / n;
  int j = index % n;
  return std::make_pair(i, j);
};

struct pair_hash {
  template <class T, class U>
  std::size_t operator()(std::pair<T, U> const &pair) const {
    return std::hash<T>()(pair.first) ^ std::hash<U>()(pair.second);
  }
};

template <typename T, typename priority_t> struct PriorityQueue {
  typedef std::pair<priority_t, T> PQElement;
  std::priority_queue<PQElement, std::vector<PQElement>,
                      std::greater<PQElement>>
      elements;

  inline bool empty() const { return elements.empty(); }

  inline void put(T item, priority_t priority) {
    elements.emplace(priority, item);
  }

  T get() {
    T best_item = elements.top().second;
    elements.pop();
    return best_item;
  }
};

template <typename Location, typename Graph>
void dijkstra_search(Graph graph, Location start, Location goal,
                     std::unordered_map<Location, Location> &came_from,
                     std::unordered_map<Location, double> &cost_so_far) {
  PriorityQueue<Location, double> frontier;
  frontier.put(start, 0);

  came_from[start] = start;
  cost_so_far[start] = 0;

  while (!frontier.empty()) {
    Location current = frontier.get();

    if (current == goal) {
      break;
    }

    for (Location next : graph.neighbors(current)) {
      double new_cost = cost_so_far[current] + graph.cost(current, next);
      if (cost_so_far.find(next) == cost_so_far.end() ||
          new_cost < cost_so_far[next]) {
        cost_so_far[next] = new_cost;
        came_from[next] = current;
        frontier.put(next, new_cost);
      }
    }
  }
}

template <typename Location>
std::vector<Location>
reconstruct_path(Location start, Location goal,
                 std::unordered_map<Location, Location> came_from) {
  std::vector<Location> path;
  Location current = goal;
  if (came_from.find(goal) == came_from.end()) {
    return path; // no path can be found
  }
  while (current != start) {
    path.push_back(current);
    current = came_from[current];
  }
  path.push_back(start); // optional
  std::reverse(path.begin(), path.end());
  return path;
}

template <typename Location, typename Graph>
void a_star_search(Graph graph, Location start, Location goal,
                   std::unordered_map<Location, Location> &came_from,
                   std::unordered_map<Location, double> &cost_so_far,
                   std::function<double(Location, Location)> cost,
                   std::function<double(Location, Location)> heuristic) {
  PriorityQueue<Location, double> frontier;
  frontier.put(start, 0);

  came_from[start] = start;
  cost_so_far[start] = 0;

  while (!frontier.empty()) {
    Location current = frontier.get();

    if (current == goal) {
      break;
    }

    for (Location next : graph[current]) {
      double edge_cost = cost(current, next);
      if (edge_cost == std::numeric_limits<double>::infinity()) {
        continue;
      }
      double new_cost = cost_so_far[current] + edge_cost;
      if (cost_so_far.find(next) == cost_so_far.end() ||
          new_cost < cost_so_far[next]) {
        cost_so_far[next] = new_cost;
        double priority = new_cost + heuristic(next, goal);
        frontier.put(next, priority);
        came_from[next] = current;
      }
    }
  }
}

namespace dynorrt {

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

  void print(std::ostream & = std::cout);
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

  void print(std::ostream & = std::cout);
};

struct PRM_options {
  int num_vertices_0 = 200;
  double increase_vertices_rate = 2.;
  double collision_resolution = 0.01;
  int max_it = 10;
  double connection_radius = 1.;
  double max_compute_time_ms = 1e9;
  bool xrand_collision_free = true;
  int max_num_trials_col_free = 1000;
  bool incremental_collision_check = false;
  void print(std::ostream & = std::cout);
};

struct PRMlazy_options {
  int num_vertices_0 = 200;
  double increase_vertices_rate = 2.;
  double collision_resolution = 0.01;
  int max_lazy_iterations = 1000;
  double connection_radius = .5;
  double max_compute_time_ms = 1e9;
  bool xrand_collision_free = true;
  int max_num_trials_col_free = 1000;
  void print(std::ostream & = std::cout);
};

} // namespace dynorrt

TOML11_DEFINE_CONVERSION_NON_INTRUSIVE_OR(dynorrt::RRT_options, max_it,
                                          goal_bias, collision_resolution,
                                          max_step, max_compute_time_ms,
                                          goal_tolerance, max_num_configs,
                                          xrand_collision_free,
                                          max_num_trials_col_free);

TOML11_DEFINE_CONVERSION_NON_INTRUSIVE_OR(dynorrt::BiRRT_options, max_it,
                                          goal_bias, collision_resolution,
                                          backward_probability, max_step,
                                          max_compute_time_ms, goal_tolerance,
                                          max_num_configs, xrand_collision_free,
                                          max_num_trials_col_free);

TOML11_DEFINE_CONVERSION_NON_INTRUSIVE_OR(
    dynorrt::PRM_options, num_vertices_0, increase_vertices_rate,
    collision_resolution, max_it, connection_radius, max_compute_time_ms,
    xrand_collision_free, max_num_trials_col_free, incremental_collision_check);

TOML11_DEFINE_CONVERSION_NON_INTRUSIVE_OR(
    dynorrt::PRMlazy_options, num_vertices_0, increase_vertices_rate,
    collision_resolution, max_lazy_iterations, connection_radius,
    max_compute_time_ms, xrand_collision_free, max_num_trials_col_free);

inline void dynorrt::RRT_options::print(std::ostream &out) {
  toml::value v = *this;
  out << v << std::endl;
}

inline void dynorrt::BiRRT_options::print(std::ostream &out) {
  toml::value v = *this;
  out << v << std::endl;
}

inline void dynorrt::PRM_options::print(std::ostream &out) {
  toml::value v = *this;
  out << v << std::endl;
}

inline void dynorrt::PRMlazy_options::print(std::ostream &out) {
  toml::value v = *this;
  out << v << std::endl;
}

namespace dynorrt {

template <typename T> void print_path(const std::vector<T> &path) {

  std::cout << "PATH" << std::endl;
  std::cout << "path.size(): " << path.size() << std::endl;
  for (size_t i = 0; i < path.size(); i++) {
    std::cout << path[i].transpose() << std::endl;
  }
}

/**
 * @brief      Check if a path is collision free.
 * @tparam     T     The type of the state
 * @tparam     StateSpace  The type of the state space
 * @tparam     Fun   The type of the is collision free function
 * @param[in]  path  The path we want to shortcut
 * @param[in]  is_collision_free_fun  The is collision free function
 * @param[in]  state_space  The state
 * @param[in]  resolution  The resolution
 * @return     True if collision free, False otherwise.
 */
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

enum class TerminationCondition {
  MAX_IT,
  MAX_TIME,
  GOAL_REACHED,
  MAX_NUM_CONFIGS,
  RUNNING,
  // The following are for Anytime Assymp optimal planners
  MAX_IT_GOAL_REACHED,
  MAX_TIME_GOAL_REACHED,
  MAX_NUM_CONFIGS_GOAL_REACHED,
  RUNNING_GOAL_REACHED,
  EXTERNAL_TRIGGER_GOAL_REACHED,
  UNKNOWN
};

inline void ensure_connected_tree_with_no_cycles(
    const std::vector<std::set<int>> &childrens) {

  int num_configs = childrens.size();
  std::vector<bool> visited(num_configs, false);

  int start = 0;

  std::queue<int> q;
  q.push(start);

  // Check that I visit all nodes once
  while (!q.empty()) {
    int current = q.front();
    q.pop();
    CHECK_PRETTY_DYNORRT__(visited[current] == false);
    visited[current] = true;
    for (auto child : childrens[current]) {
      q.push(child);
    }
  }

  // check that all nodes are visited
  for (int i = 0; i < num_configs; i++) {
    if (!visited[i]) {
      std::cout << "i: " << i << std::endl;
      std::cout << "num_configs: " << num_configs << std::endl;
      std::cout << "visited[i]: " << visited[i] << std::endl;
      CHECK_PRETTY_DYNORRT__(visited[i]);
    }
  }
}

inline bool is_termination_condition_solved(
    const TerminationCondition &termination_condition) {
  return termination_condition == TerminationCondition::GOAL_REACHED ||
         termination_condition == TerminationCondition::MAX_IT_GOAL_REACHED ||
         termination_condition == TerminationCondition::MAX_TIME_GOAL_REACHED ||
         termination_condition ==
             TerminationCondition::MAX_NUM_CONFIGS_GOAL_REACHED ||
         termination_condition ==
             TerminationCondition::EXTERNAL_TRIGGER_GOAL_REACHED;
}

template <typename StateSpace, int DIM> class PlannerBase {

public:
  using state_t = Eigen::Matrix<double, DIM, 1>;
  using state_cref_t = const Eigen::Ref<const state_t> &;
  using tree_t = dynotree::KDTree<int, DIM, 32, double, StateSpace>;
  using is_collision_free_fun_t = std::function<bool(state_t)>;
  using edge_t = std::pair<state_t, state_t>;

  virtual ~PlannerBase() = default;

  void set_state_space(StateSpace t_state_space) {
    state_space = t_state_space;
    tree = tree_t();
    tree.init_tree(runtime_dim, state_space);
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

  void
  set_collision_manager(CollisionManagerBallWorld<DIM> *collision_manager) {
    is_collision_free_fun = [collision_manager](state_t x) {
      return !collision_manager->is_collision(x);
    };
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

  virtual void init(int t_runtime_dim = -1) {

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
    init();
  }

  virtual TerminationCondition plan() {
    THROW_PRETTY_DYNORRT("Not implemented in base class!");
  }

protected:
  StateSpace state_space;
  state_t start;
  state_t goal;
  tree_t tree;
  is_collision_free_fun_t is_collision_free_fun = [](const auto &) {
    THROW_PRETTY_DYNORRT("define a collision free fun!");
    return false;
  };

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

// Continue here!
// template <typename StateSpace, int DIM>

template <typename StateSpace, int DIM>
class BiRRT : public PlannerBase<StateSpace, DIM> {
public:
  using Base = PlannerBase<StateSpace, DIM>;
  using tree_t = typename Base::tree_t;
  using state_t = typename Base::state_t;
  using Configs = std::vector<typename Base::state_t>;

  struct T_helper {
    std::vector<int> *parents;
    std::vector<state_t> *configs;
    tree_t *tree;
    bool is_forward;
  };

  virtual ~BiRRT() = default;

  void set_options(BiRRT_options t_options) { options = t_options; }
  BiRRT_options get_options() { return options; }

  virtual void print_options(std::ostream &out = std::cout) override {
    options.print(out);
  }

  virtual void set_options_from_toml(toml::value &cfg) override {
    options = toml::find<BiRRT_options>(cfg, "RRT_options");
  }

  Configs &get_configs_backward() { return configs_backward; }
  std::vector<int> &get_parents_backward() { return parents_backward; }

  virtual void init(int t_runtime_dim = -1) override {
    Base::init(t_runtime_dim);
    init_backward_tree(t_runtime_dim);
  }

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

    MESSAGE_PRETTY_DYNORRT("Options");
    this->print_options();

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

    T_helper Ta{.parents = &this->parents,
                .configs = &this->configs,
                .tree = &this->tree,
                .is_forward = true

    };

    T_helper Tb{.parents = &parents_backward,
                .configs = &configs_backward,
                .tree = &tree_backward,
                .is_forward = false};

    T_helper Tsrc, Ttgt;

    while (termination_condition == TerminationCondition::RUNNING) {

      expand_forward = static_cast<double>(std::rand()) / RAND_MAX >
                       options.backward_probability;

      if (expand_forward) {
        Tsrc = Ta;
        Ttgt = Tb;
      } else {
        Tsrc = Tb;
        Ttgt = Ta;
      }

      int goal_id = -1;
      bool goal_connection_attempt =
          static_cast<double>(std::rand()) / RAND_MAX < options.goal_bias;
      if (goal_connection_attempt) {
        goal_id = std::rand() % Ttgt.configs->size();
        this->x_rand = Ttgt.configs->at(goal_id);
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
      auto nn = Tsrc.tree->search(this->x_rand);
      std::cout << "nn.id: " << nn.id << std::endl;
      this->x_near = Tsrc.configs->at(nn.id);

      bool full_step_attempt = nn.distance < options.max_step;
      if (full_step_attempt) {
        this->x_new = this->x_rand;
      } else {
        this->state_space.interpolate(this->x_near, this->x_rand,
                                      options.max_step / nn.distance,
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

      this->infeasible_edges += !is_collision_free;

      if (is_collision_free) {
        Tsrc.tree->addPoint(this->x_new, Tsrc.tree->size());
        Tsrc.configs->push_back(this->x_new);
        Tsrc.parents->push_back(nn.id);
        CHECK_PRETTY_DYNORRT__(Tsrc.tree->size() == Tsrc.configs->size());
        CHECK_PRETTY_DYNORRT__(Tsrc.tree->size() == Tsrc.parents->size());

        if (full_step_attempt && goal_connection_attempt) {
          path_found = true;
          MESSAGE_PRETTY_DYNORRT("Path found!");
          CHECK_PRETTY_DYNORRT__(this->state_space.distance(
                                     this->x_new, Ttgt.configs->at(goal_id)) <
                                 options.goal_tolerance);

          if (expand_forward) {
            connect_id_forward = Tsrc.tree->size() - 1;
            connect_id_backward = goal_id;
          } else {
            connect_id_forward = goal_id;
            connect_id_backward = Tsrc.tree->size() - 1;
          }
        }
      }

      num_it++;
      termination_condition = should_terminate();

    } // RRT CONNECT terminated

    if (termination_condition == TerminationCondition::GOAL_REACHED) {

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

      this->path.insert(this->path.end(), fwd_path.begin(), fwd_path.end());
      this->path.insert(this->path.end(), bwd_path.begin() + 1, bwd_path.end());
    }

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
  virtual ~RRTConnect() = default;

  virtual TerminationCondition plan() override {
    this->check_internal();

    MESSAGE_PRETTY_DYNORRT("Options");
    this->print_options();

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
          this->options.max_num_configs) {
        return TerminationCondition::MAX_NUM_CONFIGS;
      } else if (num_it > this->options.max_it) {
        return TerminationCondition::MAX_IT;
      } else if (get_elapsed_ms() > this->options.max_compute_time_ms) {
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

    typename Base::T_helper Ta{.parents = &this->parents,
                               .configs = &this->configs,
                               .tree = &this->tree,
                               .is_forward = true

    };

    typename Base::T_helper Tb{.parents = &this->parents_backward,
                               .configs = &this->configs_backward,
                               .tree = &this->tree_backward,
                               .is_forward = false};

    while (termination_condition == TerminationCondition::RUNNING) {

      if (this->options.xrand_collision_free) {
        bool is_collision_free = false;
        int num_tries = 0;
        while (!is_collision_free &&
               num_tries < this->options.max_num_trials_col_free) {
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

      bool full_step_attempt = nn_a.distance < this->options.max_step;
      if (full_step_attempt) {
        this->x_new = this->x_rand;
      } else {
        this->state_space.interpolate(this->x_near, this->x_rand,
                                      this->options.max_step / nn_a.distance,
                                      this->x_new);
      }

      this->evaluated_edges += 1;
      bool is_collision_free = is_edge_collision_free(
          this->x_near, this->x_new, col, this->state_space,
          this->options.collision_resolution);

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

        bool full_step_attempt = nn_b.distance < this->options.max_step;
        if (full_step_attempt) {
          this->x_new = this->x_rand;
        } else {
          this->state_space.interpolate(this->x_near, this->x_rand,
                                        this->options.max_step / nn_b.distance,
                                        this->x_new);
        }
        this->evaluated_edges += 1;
        bool is_collision_free = is_edge_collision_free(
            this->x_near, this->x_new, col, this->state_space,
            this->options.collision_resolution);
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
          this->options.goal_tolerance);

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
class LazyPRM : public PlannerBase<StateSpace, DIM> {
  // TODO: continue here!!

public:
  virtual ~LazyPRM() = default;

  virtual void print_options(std::ostream &out = std::cout) override {
    options.print(out);
  }

  virtual void set_options_from_toml(toml::value &cfg) override {
    options = toml::find<PRMlazy_options>(cfg, "PRMlazy_options");
  }

  std::vector<std::vector<int>> &get_adjacency_list() { return adjacency_list; }

  void set_options(PRMlazy_options t_options) { options = t_options; }

  virtual TerminationCondition plan() override {

    TerminationCondition termination_condition = TerminationCondition::UNKNOWN;

    CHECK_PRETTY_DYNORRT__(adjacency_list.size() == 0);

    this->check_internal();

    MESSAGE_PRETTY_DYNORRT("Options");
    this->print_options();

    auto col = [this](const auto &x) {
      return this->is_collision_free_fun_timed(x);
    };

    CHECK_PRETTY_DYNORRT__(options.num_vertices_0 >= 2);

    this->configs.push_back(this->start);
    this->configs.push_back(this->goal);

    // Generate N random collision free configs

    // REFACTOR THIS CODE!!! only on one place
    auto tic = std::chrono::steady_clock::now();
    for (size_t i = 0; i < options.num_vertices_0 - 2; i++) {
      bool is_collision_free = false;
      int num_tries = 0;
      if (options.xrand_collision_free) {
        while (!is_collision_free &&
               num_tries < options.max_num_trials_col_free) {
          this->state_space.sample_uniform(this->x_rand);
          is_collision_free = col(this->x_rand);
          num_tries++;
        }
        CHECK_PRETTY_DYNORRT(is_collision_free,
                             "cannot generate a valid xrand");
        this->configs.push_back(this->x_rand);
      } else {
        this->state_space.sample_uniform(this->x_rand);
      }
    }
    double time_sample_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - tic)
            .count();

    // Vertices build.

    // Now lets get the connections
    adjacency_list.resize(this->configs.size());

    for (size_t i = 0; i < this->configs.size(); i++) {
      this->tree.addPoint(this->configs[i], i, false);
    }
    this->tree.splitOutstanding();

    auto tic2 = std::chrono::steady_clock::now();
    // NOTE: using a K-d Tree helps only if there are a lot of points!
    for (int i = 0; i < this->configs.size(); i++) {

      auto nn =
          this->tree.searchBall(this->configs[i], options.connection_radius);
      for (int j = 0; j < nn.size(); j++) {
        auto &src = this->configs[i];
        auto &tgt = this->configs[nn[j].id];
        if (i >= nn[j].id) {
          continue;
        }
        adjacency_list[i].push_back(nn[j].id);
        adjacency_list[nn[j].id].push_back(i);
      }
    }

    // for (int j = i + 1; j < this->configs.size(); j++) {
    //   auto &src = this->configs[i];
    //   auto &tgt = this->configs[j];
    //   if (this->state_space.distance(src, tgt) < options.connection_radius)
    //   {
    //
    // if ( options.incremental_collision_check ||
    //     is_edge_collision_free(src, tgt, col, this->state_space,
    //                            this->options.collision_resolution)) {
    //
    //
    //       adjacency_list[i].push_back(j);
    //       adjacency_list[j].push_back(i);
    //     }
    //   }
    // }
    double time_build_graph_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - tic2)
            .count();

    MESSAGE_PRETTY_DYNORRT("graph built!");

    // Search a Path between start and goal
    //

    // NOTE: I can just compute the collisions for edges lazily when required!
    // (if no graph is available)
    using Location = int;
    int start_id = 0;
    int goal_id = 1;
    std::unordered_map<Location, Location> came_from;
    std::unordered_map<Location, double> cost_so_far;

    // Use this to avoid recomputing collisions --> save such that (i,j) with i
    // < j
    std::unordered_map<int, int> edges_map; // -1, 0 , 1
    // { -1: not checked, 0: collision, 1: no collision}

    for (size_t i = 0; i < this->configs.size(); i++) {
      for (size_t j = i + 1; j < this->configs.size(); j++) {
        edges_map[index_2d_to_1d_symmetric(i, j, this->configs.size())] = -1;
      }
    }

    // TODO: get evaluated edges afterwards!
    auto twod_index_to_one_d_index = [](int i, int j, int n) {
      if (i > j) {
        std::swap(i, j);
      }
      return i * n + j;
    };

    auto index_1d_to_2d = [](int index, int n) {
      int i = index / n;
      int j = index % n;
      return std::make_pair(i, j);
    };

    std::function<double(Location, Location)> cost = [&](Location a,
                                                         Location b) {
      //
      int index = index_2d_to_1d_symmetric(a, b, this->configs.size());
      int status = edges_map[index];

      if (status == 0) {
        return std::numeric_limits<double>::infinity();
      } else {
        return this->state_space.distance(this->configs[a], this->configs[b]);
      }
    };

    std::function<double(Location, Location)> heuristic = [this](Location a,
                                                                 Location b) {
      return this->state_space.distance(this->configs[a], this->configs[b]);
    };

    for (size_t it = 0; it < options.max_lazy_iterations; it++) {

      came_from.clear();
      cost_so_far.clear();
      a_star_search(adjacency_list, start_id, goal_id, came_from, cost_so_far,
                    cost, heuristic);

      if (cost_so_far.find(goal_id) == cost_so_far.end()) {
        MESSAGE_PRETTY_DYNORRT("failed to find a solution!");
        return TerminationCondition::UNKNOWN;

      } else {

        std::vector<Location> path_id;
        path_id = reconstruct_path(start_id, goal_id, came_from);
        bool path_valid = true;

        std::cout << "path id is " << std::endl;
        for (auto &x : path_id) {
          std::cout << x << " ";
        }

        for (size_t i = 0; i < path_id.size() - 1; i++) {

          int index = index_2d_to_1d_symmetric(path_id.at(i), path_id.at(i + 1),
                                               this->configs.size());

          int status = edges_map[index];
          if (status == 0) {
            THROW_PRETTY_DYNORRT("why?");
          } else if (status == 1) {

          } else {
            auto &src = this->configs.at(path_id.at(i));
            auto &tgt = this->configs.at(path_id.at(i + 1));
            bool free =
                is_edge_collision_free(src, tgt, col, this->state_space,
                                       this->options.collision_resolution);
            edges_map[index] = static_cast<int>(free);
            if (!free) {
              path_valid = false;
              break;
            }
          }
        }
        if (path_valid) {
          MESSAGE_PRETTY_DYNORRT("solved");

          this->path.clear();
          for (size_t i = 0; i < path_id.size(); i++) {
            this->path.push_back(this->configs[path_id[i]]);
          }

          termination_condition = TerminationCondition::GOAL_REACHED;
          break;
        } else {

          MESSAGE_PRETTY_DYNORRT("collision in PATH -- Running again");
        }
      }
    }

    auto tic3 = std::chrono::steady_clock::now();
    double time_search_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - tic3)
            .count();

    std::cout << "time_sample_ms: " << time_sample_ms << std::endl;
    std::cout << "time_build_graph_ms: " << time_build_graph_ms << std::endl;
    std::cout << "time build - time col:"
              << time_build_graph_ms - this->collisions_time_ms << std::endl;
    std::cout << "time_search_ms: " << time_search_ms << std::endl;
    std::cout << "time_collisions_ms: " << this->collisions_time_ms
              << std::endl;

    // REFACTOR!!
    for (auto &t : edges_map) {
      std::pair<int, int> pair = index_1d_to_2d(t.first, this->configs.size());
      if (pair.first > pair.second) {
        std::swap(pair.first, pair.second);
      }
      if (pair.first == pair.second) {
        THROW_PRETTY_DYNORRT("pair.first == pair.second");
      }
      if (t.second == 1) {
        check_edges_valid.push_back(pair);
      } else if (t.second == 0) {
        check_edges_invalid.push_back(pair);
      }
    }

    return termination_condition;

    // if (cost_so_far.find(goal_id) == cost_so_far.end()) {
    //   MESSAGE_PRETTY_DYNORRT("failed to find a solution!");
    //   return TerminationCondition::UNKNOWN;
    //
    // } else {
    //
    //   std::vector<Location> path_id;
    //   path_id = reconstruct_path(start_id, goal_id, came_from);
    //   this->path.clear();
    //
    //   for (size_t i = 0; i < path_id.size(); i++) {
    //     this->path.push_back(this->configs[path_id[i]]);
    //   }
    //   return TerminationCondition::GOAL_REACHED;
    // }
  }

  std::vector<std::pair<int, int>> get_check_edges_valid() {
    return check_edges_valid;
  }

  std::vector<std::pair<int, int>> get_check_edges_invalid() {
    return check_edges_invalid;
  }

private:
  std::vector<std::vector<int>> adjacency_list;
  std::vector<std::pair<int, int>> check_edges_valid;
  std::vector<std::pair<int, int>> check_edges_invalid;
  PRMlazy_options options;
};

template <typename StateSpace, int DIM>
class PRM : public PlannerBase<StateSpace, DIM> {

  using AdjacencyList = std::vector<std::vector<int>>;
  using Base = PlannerBase<StateSpace, DIM>;

public:
  PRM() = default;
  virtual ~PRM() = default;

  virtual void print_options(std::ostream &out = std::cout) override {
    options.print(out);
  }

  virtual void set_options_from_toml(toml::value &cfg) override {
    options = toml::find<PRM_options>(cfg, "PRM_options");
  }

  AdjacencyList &get_adjacency_list() { return adjacency_list; }

  void set_options(PRM_options t_options) { options = t_options; }

  virtual TerminationCondition plan() override {
    TerminationCondition termination_condition = TerminationCondition::UNKNOWN;

    CHECK_PRETTY_DYNORRT__(adjacency_list.size() == 0);

    Base::check_internal();

    MESSAGE_PRETTY_DYNORRT("Options");
    this->print_options();

    auto col = [this](const auto &x) {
      return this->Base::is_collision_free_fun_timed(x);
    };

    CHECK_PRETTY_DYNORRT__(options.num_vertices_0 >= 2);

    this->configs.push_back(Base::start);
    this->configs.push_back(Base::goal);

    // Generate N random collision free configs

    auto tic = std::chrono::steady_clock::now();
    for (size_t i = 0; i < options.num_vertices_0 - 2; i++) {
      bool is_collision_free = false;
      int num_tries = 0;
      if (options.xrand_collision_free) {
        while (!is_collision_free &&
               num_tries < options.max_num_trials_col_free) {
          Base::state_space.sample_uniform(this->x_rand);
          is_collision_free = col(this->x_rand);
          num_tries++;
        }
        CHECK_PRETTY_DYNORRT(is_collision_free,
                             "cannot generate a valid xrand");
        this->configs.push_back(this->x_rand);
      } else {
        Base::state_space.sample_uniform(this->x_rand);
      }
    }
    double time_sample_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - tic)
            .count();

    // Vertices build.

    // Now lets get the connections
    adjacency_list.resize(this->configs.size());

    for (size_t i = 0; i < this->configs.size(); i++) {
      this->tree.addPoint(this->configs[i], i, false);
    }
    this->tree.splitOutstanding();

    auto tic2 = std::chrono::steady_clock::now();
    // NOTE: using a K-d Tree helps only if there are a lot of points!
    for (int i = 0; i < this->configs.size(); i++) {

      auto nn =
          this->tree.searchBall(this->configs[i], options.connection_radius);
      for (int j = 0; j < nn.size(); j++) {
        auto &src = this->configs[i];
        auto &tgt = this->configs[nn[j].id];
        if (i >= nn[j].id) {
          continue;
        }
        if (options.incremental_collision_check ||
            is_edge_collision_free(src, tgt, col, this->state_space,
                                   this->options.collision_resolution)

        ) {
          adjacency_list[i].push_back(nn[j].id);
          adjacency_list[nn[j].id].push_back(i);
        }
      }

      // for (int j = i + 1; j < this->configs.size(); j++) {
      //   auto &src = this->configs[i];
      //   auto &tgt = this->configs[j];
      //   if (this->state_space.distance(src, tgt) <
      //   options.connection_radius)
      //   {
      //
      // if ( options.incremental_collision_check ||
      //     is_edge_collision_free(src, tgt, col, this->state_space,
      //                            this->options.collision_resolution)) {
      //
      //
      //       adjacency_list[i].push_back(j);
      //       adjacency_list[j].push_back(i);
      //     }
      //   }
      // }
    }
    double time_build_graph_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - tic2)
            .count();

    MESSAGE_PRETTY_DYNORRT("graph built!");

    // Search a Path between start and goal
    //

    // NOTE: I can just compute the collisions for edges lazily when
    // required! (if no graph is available)
    using Location = int;
    int start_id = 0;
    int goal_id = 1;
    std::unordered_map<Location, Location> came_from;
    std::unordered_map<Location, double> cost_so_far;

    // Use this to avoid recomputing collisions --> save such that (i,j)
    // with i < j
    std::unordered_map<int, bool> incremental_checked_edges;

    // TODO: get evaluated edges afterwards!

    std::function<double(Location, Location)> cost = [&](Location a,
                                                         Location b) {
      if (this->options.incremental_collision_check) {

        int index = index_2d_to_1d_symmetric(a, b, this->configs.size());

        // check if the edge has been checked before
        if (incremental_checked_edges.find(index) !=
            incremental_checked_edges.end()) {
          if (incremental_checked_edges[index]) {
            return this->state_space.distance(this->configs[a],
                                              this->configs[b]);
          } else {
            return std::numeric_limits<double>::infinity();
          }
        }

        bool cfree = is_edge_collision_free(this->configs[a], this->configs[b],
                                            col, this->state_space,
                                            this->options.collision_resolution);
        incremental_checked_edges[index] = cfree;

        if (cfree) {
          return this->state_space.distance(this->configs[a], this->configs[b]);

        } else {
          return std::numeric_limits<double>::infinity();
        }
      } else {
        return this->state_space.distance(this->configs[a], this->configs[b]);
      }
    };

    std::function<double(Location, Location)> heuristic = [this](Location a,
                                                                 Location b) {
      return this->state_space.distance(this->configs[a], this->configs[b]);
    };

    auto tic3 = std::chrono::steady_clock::now();
    a_star_search(adjacency_list, start_id, goal_id, came_from, cost_so_far,
                  cost, heuristic);
    double time_search_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - tic3)
            .count();

    std::cout << "time_sample_ms: " << time_sample_ms << std::endl;
    std::cout << "time_build_graph_ms: " << time_build_graph_ms << std::endl;
    std::cout << "time build - time col:"
              << time_build_graph_ms - this->collisions_time_ms << std::endl;
    std::cout << "time_search_ms: " << time_search_ms << std::endl;
    std::cout << "time_collisions_ms: " << this->collisions_time_ms
              << std::endl;

    for (auto &t : incremental_checked_edges) {
      std::pair<int, int> pair = index_1d_to_2d(t.first, this->configs.size());
      if (pair.first > pair.second) {
        std::swap(pair.first, pair.second);
      }
      if (pair.first == pair.second) {
        THROW_PRETTY_DYNORRT("pair.first == pair.second");
      }
      if (t.second) {
        check_edges_valid.push_back(pair);
      } else {
        check_edges_invalid.push_back(pair);
      }
    }

    if (cost_so_far.find(goal_id) == cost_so_far.end()) {
      MESSAGE_PRETTY_DYNORRT("failed to find a solution!");
      return TerminationCondition::UNKNOWN;

    } else {

      std::vector<Location> path_id;
      path_id = reconstruct_path(start_id, goal_id, came_from);
      this->path.clear();

      for (size_t i = 0; i < path_id.size(); i++) {
        this->path.push_back(this->configs[path_id[i]]);
      }
      return TerminationCondition::GOAL_REACHED;
    }
  }

  std::vector<std::pair<int, int>> &get_check_edges_valid() {
    return check_edges_valid;
  }

  std::vector<std::pair<int, int>> &get_check_edges_invalid() {
    return check_edges_invalid;
  }

private:
  PRM_options options;
  AdjacencyList adjacency_list;
  std::vector<std::pair<int, int>> check_edges_valid;
  std::vector<std::pair<int, int>> check_edges_invalid;
};

bool inline ensure_childs_and_parents(
    const std::vector<std::set<int>> &children,
    const std::vector<int> &parents) {

  if (parents.size() != children.size()) {
    MESSAGE_PRETTY_DYNORRT("parents.size() != children.size()");
    return false;
  }

  for (size_t i = 0; i < parents.size(); i++) {
    if (parents[i] == -1) {
      continue;
    }
    if (children[parents[i]].find(i) == children[parents[i]].end()) {
      MESSAGE_PRETTY_DYNORRT("i " + std::to_string(i));
      std::cout << "parents[i] " << parents[i] << std::endl;
      std::cout << "children[parents[i]] " << std::endl;
      for (auto &x : children[parents[i]]) {
        std::cout << x << " ";
      }
      return false;
    }
  }

  for (size_t i = 0; i < children.size(); i++) {
    // MESSAGE_PRETTY_DYNORRT("i " + std::to_string(i));
    for (auto &child : children[i]) {
      std::cout << "child " << child << std::endl;
      std::cout << "parents[child] " << parents[child] << std::endl;
      if (parents[child] != i) {
        return false;
      }
    }
  }

  return true;
}

// TODO
// AO-RRT: TODO
// RRT-Kinodynamic: Reuse RRT?
// SST:

// Reference:
// Sampling-based Algorithms for Optimal Motion Planning
// Sertac Karaman Emilio Frazzoli
//
// Algorithm 6
// https://arxiv.org/pdf/1105.1186.pdf
template <typename StateSpace, int DIM>
class RRTStar : public PlannerBase<StateSpace, DIM> {

  using Base = PlannerBase<StateSpace, DIM>;
  using state_t = typename Base::state_t;

public:
  RRTStar() = default;

  virtual ~RRTStar() = default;

  virtual void print_options(std::ostream &out = std::cout) override {
    options.print(out);
  }

  virtual void set_options_from_toml(toml::value &cfg) override {
    options = toml::find<RRT_options>(cfg, "RRT_options");
  }

  // Lets do recursive version first
  void update_children(int start, double difference,
                       const std::vector<int> &parents,
                       const std::vector<std::set<int>> &children,
                       std::vector<double> &cost_to_come, int &counter) {
    counter++;
    CHECK_PRETTY_DYNORRT__(counter < parents.size());
    std::cout << "update children on " << start << std::endl;
    CHECK_PRETTY_DYNORRT__(difference < 0);
    CHECK_PRETTY_DYNORRT__(start > 0);
    CHECK_PRETTY_DYNORRT__(start < parents.size());
    CHECK_PRETTY_DYNORRT__(start < children.size());
    CHECK_PRETTY_DYNORRT__(start < cost_to_come.size());
    CHECK_PRETTY_DYNORRT__(parents.size() == children.size());
    CHECK_PRETTY_DYNORRT__(parents.size() == cost_to_come.size());
    for (auto &child : children.at(start)) {
      cost_to_come.at(child) += difference;
      update_children(child, difference, parents, children, cost_to_come,
                      counter);
    }
  }

  void set_options(RRT_options t_options) { options = t_options; }

  std::vector<double> get_cost_to_come() { return cost_to_come; }
  std::vector<std::set<int>> get_children() { return children; }
  std::vector<std::vector<state_t>> get_paths() { return paths; }

  void ensure_datastructures() {

    int number_configs = this->configs.size();

    CHECK_PRETTY_DYNORRT__(this->parents.size() == number_configs);
    CHECK_PRETTY_DYNORRT__(this->children.size() == number_configs);
    CHECK_PRETTY_DYNORRT__(this->cost_to_come.size() == number_configs);
    CHECK_PRETTY_DYNORRT__(this->tree.size() == number_configs);
  }

  virtual TerminationCondition plan() override {
    Base::check_internal();

    MESSAGE_PRETTY_DYNORRT("Options");
    this->print_options();

    this->parents.push_back(-1);
    this->children.push_back(std::set<int>());
    this->configs.push_back(Base::start);
    this->cost_to_come.push_back(0.);
    this->tree.addPoint(Base::start, 0);

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
        if (path_found) {
          return TerminationCondition::MAX_NUM_CONFIGS_GOAL_REACHED;
        } else {
          return TerminationCondition::MAX_NUM_CONFIGS;
        }
      } else if (num_it > options.max_it) {
        if (path_found) {
          return TerminationCondition::MAX_IT_GOAL_REACHED;
        } else {
          return TerminationCondition::MAX_IT;
        }
      } else if (get_elapsed_ms() > options.max_compute_time_ms) {
        if (path_found) {
          return TerminationCondition::MAX_TIME_GOAL_REACHED;
        } else {
          return TerminationCondition::MAX_TIME;
        }
      } else {
        if (path_found) {
          return TerminationCondition::RUNNING_GOAL_REACHED;
        } else {
          return TerminationCondition::RUNNING;
        }
      }
    };

    TerminationCondition termination_condition = should_terminate();
    int goal_id = -1;
    double best_cost = std::numeric_limits<double>::infinity();

    while (termination_condition == TerminationCondition::RUNNING ||
           termination_condition ==
               TerminationCondition::RUNNING_GOAL_REACHED) {

      // check that the goal_id is never a parent of a node
      if (goal_id != -1) {
        for (auto &p : this->parents) {
          if (p == goal_id) {
            THROW_PRETTY_DYNORRT("goal_id is a parent of a node");
          }
        }
      }

      ensure_connected_tree_with_no_cycles(this->children);
      ensure_datastructures();

      if (!ensure_childs_and_parents(children, this->parents)) {
        THROW_PRETTY_DYNORRT("parents and children are not consistent it " +
                             std::to_string(num_it));
      }

      bool informed_rrt_star = false;
      int max_attempts_informed = 1000;
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
            if (informed_rrt_star) {
              THROW_PRETTY_DYNORRT("not implemented");
            }
            num_tries++;
          }
          CHECK_PRETTY_DYNORRT(is_collision_free,
                               "cannot generate a valid xrand");
        } else {
          Base::state_space.sample_uniform(this->x_rand);
        }
      }

      Base::sample_configs.push_back(this->x_rand);

      auto nn1 = Base::tree.search(this->x_rand);

      if (nn1.id == goal_id) {
        // I dont want to put nodes as children of the goal
        nn1 = Base::tree.searchKnn(this->x_rand, 2).at(1);
      }

      this->x_near = Base::configs.at(nn1.id);

      if (nn1.distance < options.max_step) {
        this->x_new = this->x_rand;
      } else {
        Base::state_space.interpolate(this->x_near, this->x_rand,
                                      options.max_step / nn1.distance,
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

        // Be Careful not to add the goal twice!
        double distance_to_goal =
            this->get_state_space().distance(this->x_new, this->goal);

        // Compute all the nodes inside a ball
        double radius_search = options.max_step;
        auto _nns = this->tree.searchBall(this->x_new, radius_search);

        bool is_goal = distance_to_goal < options.goal_tolerance;
        int id_new;
        if (path_found && is_goal) {
          id_new = goal_id;
        } else {
          id_new = Base::configs.size();
        }

        std::unordered_map<int, int> edges_map; // -1, 0 , 1
        // { -1: not checked, 0: collision, 1: no collision}

        // NOTE: at most, there will be configs.size() + 1 points
        int max_configs_index = Base::configs.size() + 1;
        for (auto &_nn : _nns) {
          edges_map[index_2d_to_1d_symmetric(_nn.id, id_new,
                                             max_configs_index)] = -1;
        }

        int id_min = nn1.id;
        std::cout << "nn1.id: " << nn1.id << std::endl;
        double cost_min = cost_to_come.at(id_min) +
                          this->state_space.distance(this->x_near, this->x_new);

        for (auto &_nn : _nns) {

          if (_nn.id == goal_id) {
            // do not add a node as child of the goal
            continue;
          }

          double tentative_g =
              cost_to_come.at(_nn.id) +
              this->state_space.distance(this->configs.at(_nn.id), this->x_new);
          if (tentative_g < cost_min) {
            if (edges_map[index_2d_to_1d_symmetric(_nn.id, id_new,
                                                   max_configs_index)] == -1) {
              bool is_collision_free = is_edge_collision_free(
                  this->configs.at(_nn.id), this->x_new, col, Base::state_space,
                  options.collision_resolution);
              edges_map[index_2d_to_1d_symmetric(_nn.id, id_new,
                                                 max_configs_index)] =
                  static_cast<int>(is_collision_free);
            }
            if (edges_map[index_2d_to_1d_symmetric(_nn.id, id_new,
                                                   max_configs_index)] == 1) {
              id_min = _nn.id;
              cost_min = tentative_g;
            }
          }
        }

        if (path_found && is_goal) {
          if (id_min != this->parents.at(id_new)) {
            // Rewire the goal
            this->children.at(this->parents.at(id_new)).erase(id_new);
            this->children.at(id_min).insert(id_new);
            this->parents.at(id_new) = id_min;
            this->cost_to_come.at(id_new) = cost_min;
          }
        } else {
          Base::tree.addPoint(this->x_new, id_new);
          Base::configs.push_back(this->x_new);
          this->parents.push_back(id_min);
          this->cost_to_come.push_back(cost_min);
          this->children.push_back(std::set<int>());
          this->children.at(id_min).insert(id_new);
        }

        if (!path_found && Base::state_space.distance(this->x_new, Base::goal) <
                               options.goal_tolerance) {
          path_found = true;
          goal_id = id_new;
          MESSAGE_PRETTY_DYNORRT("First Path Found");
          MESSAGE_PRETTY_DYNORRT("Goal id: " << goal_id);
        }

        // std::cout << "CHILDREN" << std::endl;
        //
        // int counter = 0;
        // for (auto &x : this->children) {
        //   std::cout << counter++ << ": ";
        //   for (auto &y : x) {
        //     std::cout << y << " ";
        //   }
        //   std::cout << std::endl;
        // }
        //
        // std::cout << "PARENTS" << std::endl;
        // counter = 0;
        // for (auto &x : this->parents) {
        //   std::cout << counter++ << ": " << x << std::endl;
        // }

        if (id_new != goal_id) {
          // Rewire the tree
          for (auto &_nn : _nns) {
            double tentative_g =
                cost_to_come.at(id_new) +
                this->state_space.distance(this->configs.at(id_new),
                                           this->configs.at(_nn.id));
            double current_g = cost_to_come.at(_nn.id);
            if (tentative_g < current_g) {
              if (edges_map[index_2d_to_1d_symmetric(
                      _nn.id, id_new, Base::configs.size())] == -1) {
                bool is_collision_free = is_edge_collision_free(
                    this->configs.at(_nn.id), this->x_new, col,
                    Base::state_space, options.collision_resolution);
                edges_map[index_2d_to_1d_symmetric(_nn.id, id_new,
                                                   Base::configs.size())] =
                    static_cast<int>(is_collision_free);
              }
              if (edges_map[index_2d_to_1d_symmetric(
                      _nn.id, id_new, Base::configs.size())] == 1) {
                // Improve! I should rewire the tree
                this->children.at(this->parents.at(_nn.id)).erase(_nn.id);
                this->parents.at(_nn.id) = id_new;
                this->children.at(id_new).insert(_nn.id);
                this->cost_to_come.at(_nn.id) = tentative_g;
                MESSAGE_PRETTY_DYNORRT("rewiring");
                std::cout << "_nn.id: " << _nn.id << std::endl;
                std::cout << "id new: " << id_new << std::endl;
                std::cout << "tentative_g: " << tentative_g << std::endl;
                std::cout << "current_g: " << current_g << std::endl;
                double difference = tentative_g - current_g;
                CHECK_PRETTY_DYNORRT__(difference < 0);
                int counter = 0;

                ensure_childs_and_parents(children, this->parents);

                std::cout << "CHILDREN" << std::endl;

                int _counter = 0;
                for (auto &x : this->children) {
                  std::cout << _counter++ << ": ";
                  for (auto &y : x) {
                    std::cout << y << " ";
                  }
                  std::cout << std::endl;
                }

                std::cout << "PARENTS" << std::endl;
                counter = 0;
                for (auto &x : this->parents) {
                  std::cout << counter++ << ": " << x << std::endl;
                }

                counter = 0;
                update_children(_nn.id, difference, this->parents,
                                this->children, this->cost_to_come, counter);
              }
            }
          }
        }
      }
      if (path_found) {
        double goal_cost_tentative = cost_to_come[goal_id];
        // most of the times, the cost is the same!
        if (goal_cost_tentative < best_cost) {
          best_cost = goal_cost_tentative;
          paths.push_back(
              trace_back_solution(goal_id, Base::configs, Base::parents));
          MESSAGE_PRETTY_DYNORRT("New Path Found!"
                                 << " Number paths" << paths.size());
        }
      }
      num_it++;
      termination_condition = should_terminate();
    } // RRT terminated

    if (is_termination_condition_solved(termination_condition)) {

      int i = goal_id;
      MESSAGE_PRETTY_DYNORRT("goal id is" << i);

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
  std::vector<double> cost_to_come;
  std::vector<std::set<int>> children;
  std::vector<std::vector<state_t>> paths;
  RRT_options options;
};

template <typename StateSpace, int DIM>
class RRT : public PlannerBase<StateSpace, DIM> {

  using Base = PlannerBase<StateSpace, DIM>;

public:
  RRT() = default;

  virtual ~RRT() = default;

  virtual void print_options(std::ostream &out = std::cout) override {
    options.print(out);
  }

  virtual void set_options_from_toml(toml::value &cfg) override {
    options = toml::find<RRT_options>(cfg, "RRT_options");
  }

  void set_options(RRT_options t_options) { options = t_options; }

  virtual TerminationCondition plan() override {
    Base::check_internal();

    MESSAGE_PRETTY_DYNORRT("Options");
    this->print_options();

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
  RRT_options options;
};

} // namespace dynorrt
//
//
//

//
