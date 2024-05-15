#pragma once

#include "dynorrt_macros.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <queue>
#include <toml.hpp>
using json = nlohmann::json;
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

template <typename T>
void create_binary_order(const std::vector<T> &path_in,
                         std::vector<T> &path_out) {

  const int N = path_in.size();
  path_out.reserve(N);

  using Segment = std::pair<size_t, size_t>;
  std::queue<Segment> queue;

  int index_start = 0;
  int index_last = path_in.size() - 1;
  queue.push(Segment{index_start, index_last});

  while (!queue.empty()) {
    auto [si, gi] = queue.front();
    queue.pop();
    size_t ii = int((si + gi) / 2);
    if (ii == si || ii == gi) {
      continue;
    } else {
      path_out.push_back(path_in.at(ii));
      if (gi - si < 2) {
        continue;
      } else {
        queue.push(Segment{ii, gi});
        queue.push(Segment{si, ii});
      }
    }
  }

  path_out.push_back(path_in.at(index_start));
  path_out.push_back(path_in.at(index_last));
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
                            bool check_end_points = true,
                            bool binary_order = false) {

  if (check_end_points &&
      (!is_collision_free_fun(x_end) || !is_collision_free_fun(x_start))) {
    return false;
  }

  double d = state_space.distance(x_start, x_end);
  if (d < resolution) {
    return true;
  }

  int N = int(d / resolution) + 1;

  if (!binary_order) {

    T tmp;
    tmp.resize(x_start.size());

    for (int j = 1; j < N; j++) {
      state_space.interpolate(x_start, x_end, double(j) / N, tmp);
      if (!is_collision_free_fun(tmp)) {
        return false;
      }
    }
  } else {

    // TODO::  A better implementation can avoid (or reduce) memory allocation
    // of vectors!
    T tmp;
    tmp.resize(x_start.size());

    std::vector<T> path;
    path.reserve(N);

    for (int j = 1; j < N; j++) {
      state_space.interpolate(x_start, x_end, double(j) / N, tmp);
      path.push_back(tmp);
    }

    std::vector<T> path_order;

    create_binary_order(path, path_order);

    for (auto &x : path_order) {
      if (!is_collision_free_fun(x)) {
        return false;
      }
    }
  }

  return true;
}

template <typename T, typename StateSpace, typename Fun>
bool is_edge_collision_free_set(T x_start, T x_end, Fun &is_collision_free_fun,
                                StateSpace state_space, double resolution,
                                bool check_end_points = true) {

  std::vector<T> path;
  if (check_end_points &&
      !is_collision_free_fun(std::vector<T>{x_end, x_start})) {
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

  create_binary_order(path, path_order);
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

void finite_diff_grad(const Eigen::VectorXd &q,
                      std::function<double(const Eigen::VectorXd &)> f,
                      Eigen::VectorXd &grad_out, double eps = 1e-5) {

  Eigen::VectorXd q_plus = q;
  Eigen::VectorXd q_minus = q;

  for (size_t i = 0; i < q.size(); i++) {
    q_plus = q;
    q_minus = q;
    q_plus(i) += eps;
    q_minus(i) -= eps;

    double f_plus = f(q_plus);
    double f_minus = f(q_minus);

    double grad = (f_plus - f_minus) / (2 * eps);
    grad_out(i) = grad;
  }
}

void finite_diff_hess(const Eigen::VectorXd &q,
                      std::function<double(const Eigen::VectorXd &)> f,
                      Eigen::MatrixXd &H_out, double eps = 1e-5) {

  Eigen::VectorXd q_plus = q;
  Eigen::VectorXd q_minus = q;
  for (size_t i = 0; i < q.size(); i++) {
    q_plus = q;
    q_minus = q;
    q_plus(i) += eps;
    q_minus(i) -= eps;

    Eigen::VectorXd grad_plus(q.size());
    Eigen::VectorXd grad_minus(q.size());
    finite_diff_grad(q_plus, f, grad_plus);
    finite_diff_grad(q_minus, f, grad_minus);

    Eigen::VectorXd grad_diff = grad_plus - grad_minus;
    H_out.col(i) = grad_diff / (2 * eps);
  }
}

} // namespace dynorrt
