#pragma once

#include <algorithm>

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
      (!is_collision_free_fun(x_end) || !is_collision_free_fun(x_start)))
    return false;

  T tmp;
  tmp.resize(x_start.size());

  double d = state_space.distance(x_start, x_end);
  if (d < resolution) {
    return true;
  }
  int N = int(d / resolution) + 1;
  for (int j = 1; j < N; j++) {
    state_space.interpolate(x_start, x_end, double(j) / N, tmp);
    if (!is_collision_free_fun(tmp)) {
      return false;
    }
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

} // namespace dynorrt
