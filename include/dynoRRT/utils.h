#pragma once

#include "dynorrt_macros.h"
#include <algorithm>
#include <queue>

namespace dynorrt {

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

template <typename T, typename StateSpace>
double get_path_length(const std::vector<T> &path, StateSpace &state_space) {
  double total_distance = 0;
  for (size_t i = 0; i < path.size() - 1; i++) {
    total_distance += state_space.distance(path[i], path[i + 1]);
  }
  return total_distance;
}

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
                            StateSpace state_space, double resolution,
                            bool check_end_points = true) {

  if (check_end_points &&
      (!is_collision_free_fun(x_end) || !is_collision_free_fun(x_start))) {
    return false;
  }

  T tmp;
  tmp.resize(x_start.size());

  double d = state_space.distance(x_start, x_end);
  if (d < resolution) {
    return true;
  }
  int N = int(d / resolution) + 1;
  // for (int j = 1; j < N; j++) {
  //   state_space.interpolate(x_start, x_end, double(j) / N, tmp);
  //   if (!is_collision_free_fun(tmp)) {
  //     return false;
  //   }
  //
  // }

  std::vector<T> path;
  path.reserve(N);

  for (int j = 1; j < N; j++) {
    state_space.interpolate(x_start, x_end, double(j) / N, tmp);
    path.push_back(tmp);
  }

  std::vector<T> path_order;
  path_order.reserve(N);

  using Segment = std::pair<size_t, size_t>;
  std::queue<Segment> queue;

  int index_start = 0;
  int index_last = path.size() - 1;
  queue.push(Segment{index_start, index_last});

  while (!queue.empty()) {
    auto [si, gi] = queue.front();
    queue.pop();
    size_t ii = int((si + gi) / 2);
    if (ii == si || ii == gi) {
      continue;
    } else {
      path_order.push_back(path.at(ii));
      if (gi - si < 2) {
        continue;
      } else {
        queue.push(Segment{ii, gi});
        queue.push(Segment{si, ii});
      }
    }
  }

  path_order.push_back(path.at(index_start));
  path_order.push_back(path.at(index_last));

  // std::cout << "adding index " << index_start << std::endl;
  // std::cout << "adding index " << index_last << std::endl;

  // std::cout << "path.size(): " << path.size() << std::endl;
  // std::cout << "path_order.size(): " << path_order.size() << std::endl;

  // for (auto &x : path_order) {
  //   if (!is_collision_free_fun(x)) {
  //     return false;
  //   }
  // }

  for (auto &x : path_order) {
    if (!is_collision_free_fun(x)) {
      return false;
    }
  }

  // bool stop_at_first_collision = true;
  // return is_collision_free_fun(path);

  return true;
}

template <typename T, typename StateSpace, typename Fun>
bool is_edge_collision_free_parallel(T x_start, T x_end,
                                     Fun &is_collision_free_fun,
                                     StateSpace state_space, double resolution,
                                     bool check_end_points = true) {

  // TODO: can I avoid memory allocation?
  std::vector<T> path;
  if (check_end_points &&
      !is_collision_free_fun(std::vector<T>{x_end, x_start})) {
    // (!is_collision_free_fun(std::vector<T>{x_end}) ||
    // !is_collision_free_fun(std::vector<T>{x_start}))) {
    return false;
  }

  T tmp;
  tmp.resize(x_start.size());

  double d = state_space.distance(x_start, x_end);
  if (d < resolution) {
    return true;
  }
  int N = int(d / resolution) + 1;
  path.reserve(N);
  for (int j = 1; j < N; j++) {
    state_space.interpolate(x_start, x_end, double(j) / N, tmp);
    path.push_back(tmp);
  }

  std::vector<T> path_order;
  path_order.reserve(N);

  using Segment = std::pair<size_t, size_t>;
  std::queue<Segment> queue;

  int index_start = 0;
  int index_last = path.size() - 1;
  queue.push(Segment{index_start, index_last});

  while (!queue.empty()) {
    auto [si, gi] = queue.front();
    queue.pop();
    size_t ii = int((si + gi) / 2);
    if (ii == si || ii == gi) {
      continue;
    } else {
      path_order.push_back(path.at(ii));
      if (gi - si < 2) {
        continue;
      } else {
        queue.push(Segment{ii, gi});
        queue.push(Segment{si, ii});
      }
    }
  }

  path_order.push_back(path.at(index_start));
  path_order.push_back(path.at(index_last));

  return is_collision_free_fun(path_order);
}

// NOTE: it is recommended to call this function with a "fine path",
// TODO: test this function!!
template <typename T, typename StateSpace, typename Fun>
std::vector<T> path_shortcut_v2(const std::vector<T> path,
                                Fun &is_collision_free_fun,
                                StateSpace state_space, double resolution,
                                double path_resolution) {

  CHECK_PRETTY_DYNORRT__(path.size() >= 2);

  std::vector<T> path_local = path;
  if (path.size() == 2) {
    // [ start, goal ]
    return path;
  }
  int max_shortcut_attempts = 100;
  int shortcut_it = 0;

  bool finished = false;
  int max_index_diff_rate = .5;

  while (!finished) {
    // choose two indexes at random
    int start_i = rand() % path_local.size();
    int start_f = rand() % path_local.size();

    if (start_i > start_f) {
      std::swap(start_i, start_f);
    } else if (start_i == start_f) {
      continue;
    }

    start_f =
        std::min(start_f, start_i + max_index_diff_rate * path_local.size());

    if (is_edge_collision_free(path_local[start_i], path_local[start_f],
                               is_collision_free_fun, state_space,
                               resolution)) {
      // change the path

      std::vector<T> path_local_new;
      path_local_new.insert(path_local_new.end(), path_local.begin(),
                            path_local.begin() + start_i);
      // additionally, i could sample at a given resolution

      std::vector<T> fine_path{};
      int N =
          int(state_space.distance(path_local[start_i], path_local[start_f]) /
              path_resolution) +
          1;
      T tmp;
      for (int j = 0; j < N; j++) {
        state_space.interpolate(path_local[start_i], path_local[start_f],
                                double(j) / N, tmp);
        fine_path.push_back(tmp);
      }
      // fine_path does not contain the start_f

      // we add start_f here
      path_local_new.insert(path_local_new.end(), path_local.begin() + start_f,
                            path_local.end());
    }

    shortcut_it++;
  }
}

template <typename T, typename StateSpace, typename Fun>
std::vector<T> path_shortcut_v1(const std::vector<T> path,
                                Fun is_collision_free_fun,
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

  MESSAGE_PRETTY_DYNORRT("\nPath_shortcut_v1: Path length Reduced From "
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

template <typename T> void print_path(const std::vector<T> &path) {

  std::cout << "PATH" << std::endl;
  std::cout << "path.size(): " << path.size() << std::endl;
  for (size_t i = 0; i < path.size(); i++) {
    std::cout << path[i].transpose() << std::endl;
  }
}

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
      // MESSAGE_PRETTY_DYNORRT("i " + std::to_string(i));
      // std::cout << "parents[i] " << parents[i] << std::endl;
      // std::cout << "children[parents[i]] " << std::endl;
      // for (auto &x : children[parents[i]]) {
      //   std::cout << x << " ";
      // }
      return false;
    }
  }

  for (size_t i = 0; i < children.size(); i++) {
    // MESSAGE_PRETTY_DYNORRT("i " + std::to_string(i));
    for (auto &child : children[i]) {
      if (parents[child] != i) {
        return false;
      }
    }
  }

  return true;
}

template <typename StateSpace, int DIM> class PathShortCut {

public:
  using state_t = Eigen::Matrix<double, DIM, 1>;
  using state_ref_t = Eigen::Ref<state_t>;
  using state_cref_t = const Eigen::Ref<const state_t> &;
  using edge_t = std::pair<state_t, state_t>;
  using is_collision_free_fun_t = std::function<bool(state_t)>;

  virtual ~PathShortCut() = default;

  void set_state_space(StateSpace t_state_space) {
    state_space = t_state_space;
  }

  void set_state_space_with_string(
      const std::vector<std::string> &state_space_vstring) {
    if constexpr (std::is_same<StateSpace, dynotree::Combined<double>>::value) {
      state_space = StateSpace(state_space_vstring);
    } else {
      THROW_PRETTY_DYNORRT("use set_state_string only with dynotree::Combined");
    }
  }

  virtual void print_options(std::ostream &out = std::cout) {
    THROW_PRETTY_DYNORRT("Not implemented Yet");
  }

  void virtual set_options_from_toml(toml::value &cfg) {
    THROW_PRETTY_DYNORRT("Not implemented");
  }

  virtual void read_cfg_file(const std::string &cfg_file) {
    THROW_PRETTY_DYNORRT("Not implemented");
  }

  void virtual read_cfg_string(const std::string &cfg_string) {
    THROW_PRETTY_DYNORRT("Not implemented");
  }

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
  set_collision_manager(CollisionManagerBallWorld<DIM> *collision_manager) {
    is_collision_free_fun = [collision_manager](state_t x) {
      return !collision_manager->is_collision(x);
    };
  }

  // TODO: timing collisions take a lot of overhead, specially for
  // very simple envs where collisions are very fast.
  bool is_collision_free_fun_timed(state_cref_t x) {
    auto tic = std::chrono::steady_clock::now();
    bool is_collision_free = is_collision_free_fun(x);
    double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - tic)
                            .count();
    collisions_time_ms += elapsed_ns / 1e6;
    number_collision_checks++;
    return is_collision_free;
  }

  StateSpace &get_state_space() { return state_space; }

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
      if (runtime_dim == -1) {
        throw std::runtime_error("DIM == -1 and runtime_dim == -1");
      }
    }
  }

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

  virtual std::string get_name() { return "PathShortCut"; }

  virtual void get_planner_data(json &j) {
    j["planner_name"] = this->get_name();
    j["path"] = path;
    j["fine_path"] = get_fine_path(0.01);
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

    CHECK_PRETTY_DYNORRT__(path.size() == 0);
    CHECK_PRETTY_DYNORRT__(evaluated_edges == 0);
    CHECK_PRETTY_DYNORRT__(infeasible_edges == 0);
    CHECK_PRETTY_DYNORRT__(total_distance == -1);
    CHECK_PRETTY_DYNORRT__(collisions_time_ms == 0);
  }

  virtual void reset_internal() {
    path.clear();
    evaluated_edges = 0;
    infeasible_edges = 0;
    total_distance = -1;
    collisions_time_ms = 0;
    init();
  }

  virtual void set_initial_path(const std::vector<state_t> &t_path) {
    initial_path = t_path;
  }

  virtual void shortcut() {
    CHECK_PRETTY_DYNORRT__(initial_path.size() >= 2);

    if (initial_path.size() == 2) {
      // [ start, goal ]
      path = initial_path;
    }

    path.clear();
    int start_index = 0;
    path.push_back(initial_path.at(start_index));
    while (true) {
      int target_index = start_index + 2;
      // We know that +1 is always feasible!

      auto is_edge_collision_free_ = [&] {
        bool out = is_edge_collision_free(
            initial_path[start_index], initial_path[target_index],
            is_collision_free_fun, state_space, resolution);
        evaluated_edges++;
        if (!out) {
          infeasible_edges++;
        }
        return out;
      };

      while (target_index < initial_path.size() && is_edge_collision_free_()) {
        target_index++;
      }
      target_index--; // Reduce one, to get the last collision free edge.

      CHECK_PRETTY_DYNORRT__(target_index >= start_index + 1);
      CHECK_PRETTY_DYNORRT__(target_index < initial_path.size());

      path.push_back(initial_path[target_index]);

      if (target_index == initial_path.size() - 1) {
        break;
      }

      start_index = target_index;
    }

    CHECK_PRETTY_DYNORRT__(path.size() >= 2);
    CHECK_PRETTY_DYNORRT__(path.size() <= initial_path.size());

    MESSAGE_PRETTY_DYNORRT("\nPath_shortcut: Num of waypoints Reduced From "
                           << initial_path.size() << " to " << path.size()
                           << "\n");

    double total_distance_before = get_path_length(initial_path, state_space);
    total_distance = get_path_length(path, state_space);

    MESSAGE_PRETTY_DYNORRT("\nPath_shortcut: Distance reduced from "
                           << total_distance_before << " to " << total_distance
                           << "\n");
  }

protected:
  StateSpace state_space;
  // User can define a goal or goal_list.
  // NOTE: Goal list has priority over goal
  is_collision_free_fun_t is_collision_free_fun = [](const auto &) {
    THROW_PRETTY_DYNORRT("You have to define a collision free fun!");
    return false;
  };
  std::vector<state_t> initial_path;
  std::vector<state_t> path;
  int runtime_dim = DIM;
  double total_distance = -1;
  double collisions_time_ms = 0.;
  int number_collision_checks = 0;
  int evaluated_edges = 0;
  int infeasible_edges = 0;
  double resolution = .05;
  std::vector<std::pair<state_t, state_t>>
      valid_edges; // TODO: only with a flag
  std::vector<std::pair<state_t, state_t>>
      invalid_edges; // TODO: only rrth a flag
};

template <int DIM_state = -1, int DIM_control = -1> struct Trajectory {
  using state_t = Eigen::Matrix<double, DIM_state, 1>;
  using control_t = Eigen::Matrix<double, DIM_control, 1>;
  std::vector<state_t> states;
  std::vector<control_t> controls;
};

template <int DIM_state = -1, int DIM_control = -1>
void to_json(json &j, const Trajectory<DIM_state, DIM_control> &p) {
  j = json{{"states", p.states}, {"controls", p.controls}};
}

template <int DIM_state = -1, int DIM_control = -1>
void from_json(const json &j, Trajectory<DIM_state, DIM_control> &p) {
  j.at("states").get_to(p.states);
  j.at("controls").get_to(p.controls);
}

// NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Trajectory<-1,-1>, states, controls);

// json auto serialization with macro

template <int DIM_state = -1, int DIM_control = -1>
inline Trajectory<DIM_state, DIM_control> trace_back_full_traj(
    std::vector<int> parents, int i,
    std::vector<Trajectory<DIM_state, DIM_control>> small_trajectories) {

  Trajectory<DIM_state, DIM_control> full_trajectory;
  int id = i;
  std::vector<int> path_id;
  path_id.push_back(id);
  while (parents[id] != -1) {
    id = parents[id];
    path_id.push_back(id);
  }

  std::reverse(path_id.begin(), path_id.end());

  for (size_t j = 1; j < path_id.size(); j++) {
    int id = path_id.at(j);
    full_trajectory.controls.insert(full_trajectory.controls.end(),
                                    small_trajectories[id].controls.begin(),
                                    small_trajectories[id].controls.end());
    if (j == path_id.size() - 1) {
      full_trajectory.states.insert(full_trajectory.states.end(),
                                    small_trajectories[id].states.begin(),
                                    small_trajectories[id].states.end());
    } else {
      full_trajectory.states.insert(full_trajectory.states.end(),
                                    small_trajectories[id].states.begin(),
                                    small_trajectories[id].states.end() - 1);
    }
  }
  return full_trajectory;
};

} // namespace dynorrt
