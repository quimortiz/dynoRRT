#pragma once
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <utility>

#include "magic_enum.hpp"
#include "nlohmann/json_fwd.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/fusion/functional/invocation/invoke.hpp>
#include <boost/test/tools/detail/fwd.hpp>
#include <nlohmann/json.hpp>

#include "birrt.h"
#include "dynorrt_macros.h"
#include "options.h"
#include "utils.h"

// NOTE: possible bug in TOML? connection_radius = 3 is not parsed correctly as
// a doube?
namespace dynorrt {
using json = nlohmann::json;

// Continue here!
// template <typename StateSpace, int DIM>

template <typename StateSpace, int DIM>
class RRTConnect : public BiRRT<StateSpace, DIM> {

  using Base = BiRRT<StateSpace, DIM>;
  using state_t = typename Base::Base::state_t;
  using tree_t = typename Base::Base::tree_t;

public:
  virtual ~RRTConnect() = default;

  virtual std::string get_name() override { return "RRTConnect"; }

  virtual void set_options_from_toml(toml::value &cfg) override {
    this->options = toml::find<BiRRT_options>(cfg, "RRTConnect_options");
  }

  virtual void reset() override { *this = RRTConnect(); }

  virtual TerminationCondition plan() override {
    this->check_internal();
    CHECK_PRETTY_DYNORRT__(!this->goal_list.size());

    std::cout << "Options" << std::endl;
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

      bool xrand_collision_free = false;
      int num_tries = 0;
      while (!xrand_collision_free &&
             num_tries < this->options.max_num_trials_col_free) {
        this->state_space.sample_uniform(this->x_rand);
        xrand_collision_free = col(this->x_rand);
        num_tries++;
      }
      CHECK_PRETTY_DYNORRT(xrand_collision_free,
                           "cannot generate a valid xrand");

      this->sample_configs.push_back(this->x_rand); // store for debugging
      //
      //

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
    //
    //
    //
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - tic)
                          .count();

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

      this->path.insert(this->path.end(), fwd_path.begin(), fwd_path.end());
      this->path.insert(this->path.end(), bwd_path.begin() + 1, bwd_path.end());
      this->total_distance = get_path_length(this->path, this->state_space);
    }
    // else

    // PRINT SOMETHING TO SCREEN

    std::cout << "Terminate status: "
              << magic_enum::enum_name(termination_condition) << std::endl;
    std::cout << "num_it: " << num_it << std::endl;
    std::cout << "configs.size(): " << this->configs.size() << std::endl;
    std::cout << "configs_backwared.size(): " << this->configs_backward.size()
              << std::endl;
    std::cout << "compute time (ms): " << elapsed_ms << std::endl;
    std::cout << "collisions time (ms): " << this->collisions_time_ms
              << std::endl;
    std::cout << "evaluated_edges: " << this->evaluated_edges << std::endl;
    std::cout << "infeasible_edges: " << this->infeasible_edges << std::endl;
    std::cout << "path.size(): " << this->path.size() << std::endl;
    std::cout << "total_distance: " << this->total_distance << std::endl;

    return termination_condition;
    //
  }
};

} // namespace dynorrt
