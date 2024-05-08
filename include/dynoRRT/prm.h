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

// NOTE: possible bug in TOML? connection_radius = 3 is not parsed correctly as
// a doube?
namespace dynorrt {
using json = nlohmann::json;

// Continue here!
// template <typename StateSpace, int DIM>

template <typename StateSpace, int DIM>
class PRM : public PlannerBase<StateSpace, DIM> {

  using AdjacencyList = std::vector<std::vector<int>>;
  using Base = PlannerBase<StateSpace, DIM>;
  using tree_t = typename Base::tree_t;

public:
  PRM() = default;
  virtual ~PRM() = default;

  virtual std::string get_name() override { return "PRM"; }

  virtual void reset() override { *this = PRM(); }

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

    std::cout << "Options:" << std::endl;
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
      while (!is_collision_free &&
             num_tries < options.max_num_trials_col_free) {
        Base::state_space.sample_uniform(this->x_rand);
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

    // Now lets get the connections
    adjacency_list.resize(this->configs.size());

    for (size_t i = 0; i < this->configs.size(); i++) {
      this->tree.addPoint(this->configs[i], i, true);
    }
    // this->tree.splitOutstanding();

    auto tic2 = std::chrono::steady_clock::now();
    // NOTE: using a K-d Tree helps only if there are a lot of points!
    for (int i = 0; i < this->configs.size(); i++) {

      std::vector<typename tree_t::DistanceId> nn;

      if (options.k_near > 0) {
        nn = this->tree.searchKnn(this->configs[i], options.k_near);
      } else {
        nn = this->tree.searchBall(this->configs[i], options.connection_radius);
      }

      for (int j = 0; j < nn.size(); j++) {
        if (i >= nn[j].id) {
          continue;
        }
        if (options.incremental_collision_check) {
          adjacency_list[i].push_back(nn[j].id);
          adjacency_list[nn[j].id].push_back(i);
        } else {
          auto &src = this->configs[i];
          auto &tgt = this->configs[nn[j].id];
          bool col_free =
              is_edge_collision_free(src, tgt, col, this->state_space,
                                     this->options.collision_resolution);
          if (col_free) {
            adjacency_list[i].push_back(nn[j].id);
            adjacency_list[nn[j].id].push_back(i);
            check_edges_valid.push_back({i, nn[j].id});
          } else {
            check_edges_invalid.push_back({i, nn[j].id});
          }
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

    // MESSAGE_PRETTY_DYNORRT("graph built!");

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

        if (cfree) {
          check_edges_valid.push_back({a, b});
        } else {
          check_edges_invalid.push_back({a, b});
        }

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

    double time_total = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - tic)
                            .count();

    std::cout << "total time (ms): " << time_total << std::endl;
    std::cout << "time_sample_ms: " << time_sample_ms << std::endl;
    std::cout << "time_build_graph_ms: " << time_build_graph_ms << std::endl;
    std::cout << "time build - time col:"
              << time_build_graph_ms - this->collisions_time_ms << std::endl;
    std::cout << "time_search_ms: " << time_search_ms << std::endl;
    std::cout << "time_collisions_ms: " << this->collisions_time_ms
              << std::endl;

    // if (options.incremental_collision_check) {
    //
    //   for (auto &t : incremental_checked_edges) {
    //     std::pair<int, int> pair =
    //         index_1d_to_2d(t.first, this->configs.size());
    //     if (pair.first > pair.second) {
    //       std::swap(pair.first, pair.second);
    //     }
    //     if (pair.first == pair.second) {
    //       THROW_PRETTY_DYNORRT("pair.first == pair.second");
    //     }
    //     if (t.second) {
    //       check_edges_valid.push_back(pair);
    //     } else {
    //       check_edges_invalid.push_back(pair);
    //     }
    //   }
    // } else {
    //   // then all edges have been checked!
    // }

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

  virtual void get_planner_data(nlohmann::json &j) override {
    Base::get_planner_data(j);
    j["adjacency_list"] = adjacency_list;
    j["check_edges_valid"] = check_edges_valid;
    j["check_edges_invalid"] = check_edges_invalid;
  }

protected:
  PRM_options options;
  AdjacencyList adjacency_list;
  std::vector<std::pair<int, int>> check_edges_valid;
  std::vector<std::pair<int, int>> check_edges_invalid;
};



} // namespace dynorrt
