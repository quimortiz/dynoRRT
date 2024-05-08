#pragma once
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "magic_enum.hpp"
#include "nlohmann/json_fwd.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/fusion/functional/invocation/invoke.hpp>
#include <boost/test/tools/detail/fwd.hpp>
#include <nlohmann/json.hpp>

#include "dynorrt_macros.h"
#include "options.h"
#include "rrt_base.h"
#include "utils.h"

#include "prm.h"

// NOTE: possible bug in TOML? connection_radius = 3 is not parsed correctly as
// a doube?
namespace dynorrt {
using json = nlohmann::json;

// Continue here!
// template <typename StateSpace, int DIM>

template <typename StateSpace, int DIM>
class LazyPRM : public PRM<StateSpace, DIM> {
  using Base = PlannerBase<StateSpace, DIM>;

public:
  LazyPRM() = default;

  virtual ~LazyPRM() = default;

  virtual void reset() override { *this = LazyPRM(); }

  virtual void print_options(std::ostream &out = std::cout) override {
    options.print(out);
  }

  virtual std::string get_name() override { return "LazyPRM"; }

  virtual void set_options_from_toml(toml::value &cfg) override {
    options = toml::find<LazyPRM_options>(cfg, "LazyPRM_options");
  }

  void set_options(LazyPRM_options t_options) { options = t_options; }

  virtual TerminationCondition plan() override {

    TerminationCondition termination_condition = TerminationCondition::UNKNOWN;

    CHECK_PRETTY_DYNORRT__(this->adjacency_list.size() == 0);

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
      while (!is_collision_free &&
             num_tries < options.max_num_trials_col_free) {
        this->state_space.sample_uniform(this->x_rand);
        is_collision_free = col(this->x_rand);
        num_tries++;
      }
      CHECK_PRETTY_DYNORRT(is_collision_free, "cannot generate a valid xrand");
      this->configs.push_back(this->x_rand);
    }
    double time_sample_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - tic)
            .count();

    // Vertices build.

    auto tic2 = std::chrono::steady_clock::now();

    // Now lets get the connections
    this->adjacency_list.resize(this->configs.size());

    auto __tic = std::chrono::steady_clock::now();
    for (size_t i = 0; i < this->configs.size(); i++) {
      this->tree.addPoint(this->configs[i], i, false);
    }
    this->tree.splitOutstanding();

    std::cout << "time build tree "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - __tic)
                     .count()
              << std::endl;
    // using linear = dynotree::LinearKNN<int, -1, double, StateSpace>;
    //
    // auto __l = linear(this->runtime_dim, this->state_space);
    //
    // for (size_t i = 0; i < this->configs.size(); i++) {
    //   __l.addPoint(this->configs[i], i, true);
    // }
    //
    double time_nn = 0;
    using DistanceId = typename Base::tree_t::DistanceId;
    for (int i = 0; i < this->configs.size(); i++) {

      std::vector<typename Base::tree_t::DistanceId> nn;
      // std::vector<typename linear::DistanceId> nn;

      auto __tic = std::chrono::steady_clock::now();
      if (options.k_near > 0) {
        // nn = __l.searchKnn(this->configs[i], options.k_near);
        nn = this->tree.searchKnn(this->configs[i], options.k_near);
      } else {
        // nn = __l.searchBall(this->configs[i], options.connection_radius);
        nn = this->tree.searchBall(this->configs[i], options.connection_radius);
      }
      time_nn += std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - __tic)
                     .count();

      for (int j = 0; j < nn.size(); j++) {
        if (i >= nn[j].id) {
          continue;
        }
        // note: this assumes that the distance is symmetric!
        this->adjacency_list[i].push_back(nn[j].id);
        this->adjacency_list[nn[j].id].push_back(i);
      }
    }
    std::cout << "time_nn: " << time_nn / 1000. << std::endl;

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

    std::unordered_map<int, bool> incremental_checked_edges;
    // { -1: not checked, 0: collision, 1: no collision}

    auto tic_star_search = std::chrono::steady_clock::now();

    // Use this to avoid recomputing collisions --> save such that (i,j) with
    // i < j
    // std::unordered_map<int, int> edges_map; // -1, 0 , 1

    // for (size_t i = 0; i < this->configs.size(); i++) {
    //   for (size_t j = i + 1; j < this->configs.size(); j++) {
    //     edges_map[index_2d_to_1d_symmetric(i, j, this->configs.size())] = -1;
    //   }
    // }

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

      if (auto it = incremental_checked_edges.find(index);
          it != incremental_checked_edges.end() && !it->second)
        return std::numeric_limits<double>::infinity();
      else
        return this->state_space.distance(this->configs[a], this->configs[b]);
    };

    std::function<double(Location, Location)> heuristic = [this](Location a,
                                                                 Location b) {
      return this->state_space.distance(this->configs[a], this->configs[b]);
    };

    for (size_t it = 0; it < options.max_lazy_iterations; it++) {

      came_from.clear();
      cost_so_far.clear();
      a_star_search(this->adjacency_list, start_id, goal_id, came_from,
                    cost_so_far, cost, heuristic);

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
          auto it = incremental_checked_edges.find(index);

          int status = -1;
          if (it != incremental_checked_edges.end()) {
            status = it->second;
          }
          if (status == 0) {
            THROW_PRETTY_DYNORRT("why?");
          } else if (status == 1) {

          } else {
            auto &src = this->configs.at(path_id.at(i));
            auto &tgt = this->configs.at(path_id.at(i + 1));
            bool free =
                is_edge_collision_free(src, tgt, col, this->state_space,
                                       this->options.collision_resolution);
            if (free) {
              this->check_edges_valid.push_back(
                  {path_id.at(i), path_id.at(i + 1)});
            } else {
              this->check_edges_invalid.push_back(
                  {path_id.at(i), path_id.at(i + 1)});
            }

            incremental_checked_edges[index] = free;
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
          std::cout << "Path not valid -- Running again" << std::endl;
        }
      }
    }

    // auto tic3 = std::chrono::steady_clock::now();
    double time_search_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - tic_star_search)
            .count();

    double time_total = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - tic)
                            .count();

    std::cout << "time_total " << time_total << std::endl;
    std::cout << "time_sample_ms: " << time_sample_ms << std::endl;
    std::cout << "time_build_graph_ms: " << time_build_graph_ms << std::endl;
    std::cout << "time build - time col:"
              << time_build_graph_ms - this->collisions_time_ms << std::endl;
    std::cout << "time_search_ms: " << time_search_ms << std::endl;
    std::cout << "time_collisions_ms: " << this->collisions_time_ms
              << std::endl;

    // REFACTOR!!
    // for (auto &t : edges_map) {
    //   std::pair<int, int> pair = index_1d_to_2d(t.first,
    //   this->configs.size()); if (pair.first > pair.second) {
    //     std::swap(pair.first, pair.second);
    //   }
    //   if (pair.first == pair.second) {
    //     THROW_PRETTY_DYNORRT("pair.first == pair.second");
    //   }
    //   if (t.second == 1) {
    //     this->check_edges_valid.push_back(pair);
    //   } else if (t.second == 0) {
    //     this->check_edges_invalid.push_back(pair);
    //   }
    // }

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

  // std::vector<std::pair<int, int>> get_check_edges_invalid() {
  //   return check_edges_invalid;
  // }

protected:
  // std::vector<std::vector<int>> adjacency_list;
  // std::vector<std::pair<int, int>> check_edges_valid;
  // std::vector<std::pair<int, int>> check_edges_invalid;
  LazyPRM_options options;
};

} // namespace dynorrt
