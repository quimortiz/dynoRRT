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

template <typename StateSpace, int DIM, int DIM_control>
class KinoRRT : public PlannerBase<StateSpace, DIM> {

public:
  using Base = PlannerBase<StateSpace, DIM>;
  using state_t = typename Base::state_t;
  using control_t = Eigen::Matrix<double, DIM_control, 1>;
  using trajectory_t = Trajectory<DIM, DIM_control>;
  KinoRRT() = default;

  virtual ~KinoRRT() = default;

  virtual void print_options(std::ostream &out = std::cout) override {
    std::cout << "Options in " << this->get_name() << std::endl;
    options.print(out);
  }

  virtual void reset() override { *this = KinoRRT(); }

  virtual std::string get_name() override { return "kinoRRT"; }

  virtual void set_options_from_toml(toml::value &cfg) override {
    options = toml::find<KinoRRT_options>(cfg, "KinoRRT_options");
  }

  void set_options(KinoRRT_options t_options) { options = t_options; }

  KinoRRT_options get_options() const { return options; }

  virtual TerminationCondition plan() override {
    Base::check_internal();
    this->print_options();

    this->parents.push_back(-1);
    this->configs.push_back(this->start);
    this->tree.addPoint(this->start, 0);
    this->small_trajectories.push_back(
        trajectory_t()); // empty samll trajectory to reach
    // the start

    int num_it = 0;
    auto tic = std::chrono::steady_clock::now();
    bool path_found = false;
    int num_collisions = 0;

    auto col = [&, this](const auto &x) {
      // return this->is_collision_free_fun_timed(x);
      num_collisions++;
      return this->is_collision_free_fun(x);
    };

    this->state_space.print(std::cout);

    CHECK_PRETTY_DYNORRT__(col(this->start));
    CHECK_PRETTY_DYNORRT__(col(this->goal));
    CHECK_PRETTY_DYNORRT__(this->state_space.check_bounds(this->start));
    CHECK_PRETTY_DYNORRT__(this->state_space.check_bounds(this->goal));

    auto get_elapsed_ms = [&] {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - tic)
          .count();
    };

    auto should_terminate = [&] {
      if (this->configs.size() > options.max_num_configs) {
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
    bool is_goal = false;
    double nn_search_time = 0;
    double time_expand_fun = 0;
    double time_store_info = 0;

    auto timed_tree_search = [&](const state_t &x) {
      auto tic = std::chrono::steady_clock::now();
      auto nn = this->tree.search(x);
      nn_search_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - tic)
                            .count() /
                        double(10e6);

      return nn;
    };

    while (termination_condition == TerminationCondition::RUNNING) {

      if (static_cast<double>(std::rand()) / RAND_MAX < options.goal_bias) {
        this->x_rand = this->goal;
        is_goal = true;
      } else {
        is_goal = false;
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
      }
      if (options.debug) {
        this->sample_configs.push_back(this->x_rand);
      }

      auto nn = timed_tree_search(this->x_rand);

      this->x_near = this->configs.at(nn.id);

      if (is_goal) {
        // If the goal is sampled, then I will try to connect
        // a random state with half probability, not only the closest one!
        // This is to avoid getting stuck in a local minima
        if (static_cast<double>(std::rand()) / RAND_MAX < 0.5) {
          int rand_id = std::rand() % this->configs.size();
          this->x_near = this->configs.at(rand_id);
          nn.id = rand_id;
          nn.distance = this->state_space.distance(this->x_rand, this->x_near);
        }
      }

      if (nn.distance < options.max_step) {
        this->x_new = this->x_rand;
      } else {
        this->state_space.interpolate(this->x_near, this->x_rand,
                                      options.max_step / nn.distance,
                                      this->x_new);
      }

      trajectory_t small_trajectory;

      auto _tic = std::chrono::steady_clock::now();
      this->expand_fun(this->x_near, this->x_new, small_trajectory);
      time_expand_fun += std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - _tic)
                             .count() /
                         double(10e6);

      bool is_collision_free = true;
      // TODO: check this at spatial resolution, instead
      // of temporal resolution!
      for (size_t i = 0; i < small_trajectory.states.size() - 1; i++) {

        if (!this->state_space.check_bounds(small_trajectory.states[i])) {
          is_collision_free = false;
          break;
        }

        if (!col(small_trajectory.states[i])) {
          is_collision_free = false;
          break;
        }
        // check the edge at a resolution
        if (!is_edge_collision_free(
                small_trajectory.states.at(i),
                small_trajectory.states.at(i + 1), col, this->state_space,
                this->options.collision_resolution, false)) {
          is_collision_free = false;
          break;
        }
      }

      if (!col(small_trajectory.states.back())) {
        is_collision_free = false;
      }
      this->evaluated_edges += 1;
      this->infeasible_edges += !is_collision_free;

      this->x_new = small_trajectory.states.back();

      if (options.store_all) {
        if (is_collision_free) {
          this->valid_edges.push_back({this->x_near, this->x_new});
        } else if (!is_collision_free) {
          this->invalid_edges.push_back({this->x_near, this->x_new});
        }
      }

      if (is_collision_free) {

        auto _tic = std::chrono::steady_clock::now();

        this->tree.addPoint(this->x_new, this->configs.size());
        this->configs.push_back(this->x_new);
        this->parents.push_back(nn.id);
        this->small_trajectories.push_back(small_trajectory);

        if (this->state_space.distance(this->x_new, Base::goal) <
            options.goal_tolerance) {
          path_found = true;
          MESSAGE_PRETTY_DYNORRT("path found");
        }

        time_store_info += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - _tic)
                               .count() /
                           double(10e6);
      }

      num_it++;
      termination_condition = should_terminate();

    } // RRT terminated

    double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - tic)
                      .count();

    if (termination_condition == TerminationCondition::GOAL_REACHED) {

      int i = this->configs.size() - 1;

      CHECK_PRETTY_DYNORRT__(
          this->state_space.distance(this->configs[i], this->goal) <
          options.goal_tolerance);

      this->path = trace_back_solution(i, this->configs, this->parents);

      full_trajectory =
          trace_back_full_traj(this->parents, i, this->small_trajectories);

      std::cout << this->configs.back().transpose() << std::endl;
      std::cout << this->goal.transpose() << std::endl;
      std::cout << this->path.back().transpose() << std::endl;

      CHECK_PRETTY_DYNORRT__(
          this->state_space.distance(this->full_trajectory.states.front(),
                                     this->start) < 1e-6);

      CHECK_PRETTY_DYNORRT__(full_trajectory.states.size() ==
                             full_trajectory.controls.size() + 1);

      CHECK_PRETTY_DYNORRT__(
          this->state_space.distance(this->full_trajectory.states.back(),
                                     this->goal) < options.goal_tolerance)

      CHECK_PRETTY_DYNORRT__(
          this->state_space.distance(this->path[0], this->start) < 1e-6);

      std::cout << this->state_space.distance(this->path.back(), this->goal)
                << std::endl;

      CHECK_PRETTY_DYNORRT__(
          this->state_space.distance(this->path.back(), this->goal) <
          options.goal_tolerance)

      this->total_distance = 0;
      for (size_t i = 0; i < this->path.size() - 1; i++) {
        this->total_distance +=
            this->state_space.distance(this->path[i], this->path[i + 1]);
      }
    } else {
      MESSAGE_PRETTY_DYNORRT("failed to find a solution!");
      double min_distance = std::numeric_limits<double>::infinity();
      int min_id = -1;
      for (size_t i = 0; i < this->configs.size(); i++) {
        double distance =
            this->state_space.distance(this->configs[i], this->goal);
        if (distance < min_distance) {
          min_distance = distance;
          min_id = i;
        }
      }
      MESSAGE_PRETTY_DYNORRT("min_distance: " << min_distance);
      MESSAGE_PRETTY_DYNORRT("min_id: " << min_id);
    }

    MESSAGE_PRETTY_DYNORRT("Output from RRT PLANNER");
    // TODO: factor this this code to write reports
    std::cout << "Terminate status: "
              << magic_enum::enum_name(termination_condition) << std::endl;
    std::cout << "num_it: " << num_it << std::endl;
    std::cout << "compute time (ms): " << time << std::endl;
    std::cout << "configs.size(): " << this->configs.size() << std::endl;
    std::cout << "collisions time (ms): " << this->collisions_time_ms
              << std::endl;
    std::cout << "evaluated_edges: " << this->evaluated_edges << std::endl;
    std::cout << "infeasible_edges: " << this->infeasible_edges << std::endl;
    std::cout << "path.size(): " << this->path.size() << std::endl;
    std::cout << "full_trajectory.states.size(): "
              << full_trajectory.states.size() << std::endl;
    std::cout << "total_distance: " << this->total_distance << std::endl;
    std::cout << "number collision checks when timed: "
              << this->number_collision_checks << std::endl;
    std::cout << "number collision checks always: " << num_collisions
              << std::endl;
    std::cout << "nn search [ms] " << nn_search_time << std::endl;
    std::cout << "time expand [ms] " << time_expand_fun << std::endl;
    std::cout << "time store info [ms] " << time_store_info << std::endl;

    return termination_condition;
  };

  virtual void get_planner_data(nlohmann::json &j) override {
    Base::get_planner_data(j);
    j["small_trajectories"] = small_trajectories;
    j["full_trajectory"] = full_trajectory;
  }

  void
  set_expand_fun(std::function<void(state_t &, const state_t &, trajectory_t &)>
                     t_expand_fun) {
    expand_fun = t_expand_fun;
  }

protected:
  KinoRRT_options options;
  std::vector<trajectory_t> small_trajectories;
  trajectory_t full_trajectory;
  std::function<void(state_t &, const state_t &, trajectory_t &)> expand_fun;
};



} // namespace dynorrt
