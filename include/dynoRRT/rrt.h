#pragma once
#include <cstdlib>
#include <ctime>
#include <iostream>

#include "magic_enum.hpp"
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
class RRT : public PlannerBase<StateSpace, DIM> {

  using Base = PlannerBase<StateSpace, DIM>;
  using state_t = typename Base::state_t;

public:
  RRT() = default;

  virtual ~RRT() = default;

  virtual void print_options(std::ostream &out = std::cout) override {
    options.print(out);
  }

  virtual void reset() override { *this = RRT(); }

  virtual std::string get_name() override { return "RRT"; }

  virtual void set_options_from_toml(toml::value &cfg) override {
    options = toml::find<RRT_options>(cfg, "RRT_options");
  }

  void set_options(RRT_options t_options) { options = t_options; }

  virtual TerminationCondition plan() override {
    Base::check_internal();

    std::cout << "Options " << std::endl;
    this->print_options();

    this->parents.push_back(-1);
    this->configs.push_back(this->start);
    this->tree.addPoint(this->start, 0);

    int num_it = 0;
    auto tic = std::chrono::steady_clock::now();
    bool path_found = false;

    auto col = [this](const auto &x) {
      return this->is_collision_free_fun_timed(x);
    };

    // auto col_parallel = [this](const auto &x) {
    //   return this->is_collision_free_fun_parallel(x);
    // };

    std::cout << "state_space" << std::endl;
    this->state_space.print(std::cout);

    CHECK_PRETTY_DYNORRT__(col(this->start));
    CHECK_PRETTY_DYNORRT__(this->state_space.check_bounds(this->start));

    if (this->goal_list.size()) {
      for (auto &goal : this->goal_list) {
        CHECK_PRETTY_DYNORRT__(col(goal));
        CHECK_PRETTY_DYNORRT__(this->state_space.check_bounds(goal));
      }
    } else {
      CHECK_PRETTY_DYNORRT__(col(this->goal));
      CHECK_PRETTY_DYNORRT__(this->state_space.check_bounds(this->goal));
    }

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

    int goal_id = -1; // only used when user defines a list of goals

    while (termination_condition == TerminationCondition::RUNNING) {

      if (static_cast<double>(std::rand()) / RAND_MAX < options.goal_bias) {
        if (this->goal_list.size()) {
          int rand_goal_id = std::rand() % this->goal_list.size();
          this->x_rand = this->goal_list[rand_goal_id];
          is_goal = true;
        } else {
          this->x_rand = this->goal;
          is_goal = true;
        }

      } else {
        is_goal = false;
        bool is_collision_free = false;
        int num_tries = 0;
        while (!is_collision_free &&
               num_tries < options.max_num_trials_col_free) {

          if (this->custom_sample_fun) {
            this->sample_fun(this->x_rand);
          } else {
            this->state_space.sample_uniform(this->x_rand);
          }
          is_collision_free = col(this->x_rand);
          num_tries++;
        }
        CHECK_PRETTY_DYNORRT(is_collision_free,
                             "cannot generate a valid xrand");
      }
      if (options.debug) {
        this->sample_configs.push_back(this->x_rand);
      }

      auto nn = this->tree.search(this->x_rand);
      this->x_near = this->configs.at(nn.id);

      if (is_goal) {
        // NOTE: if the goal is sampled, then I will try to connect
        // a random state, not only the closest one!
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

      this->evaluated_edges += 1;

      bool is_collision_free = true;

      // TODO: check firss the end point!

      if (this->custom_is_set_collision_free_fun) {
        is_collision_free = is_edge_collision_free_set(
            this->x_near, this->x_new, this->is_set_collision_free_fun,
            this->state_space, options.collision_resolution);

      } else {
        is_collision_free = is_edge_collision_free(
            this->x_near, this->x_new, col, this->state_space,
            options.collision_resolution);
      }

      this->infeasible_edges += !is_collision_free;

      if (options.store_all) {
        if (is_collision_free) {
          this->valid_edges.push_back({this->x_near, this->x_new});
        } else if (!is_collision_free) {
          this->invalid_edges.push_back({this->x_near, this->x_new});
        }
      }

      if (is_collision_free) {
        this->tree.addPoint(this->x_new, this->configs.size());
        this->configs.push_back(this->x_new);
        this->parents.push_back(nn.id);

        if (this->goal_list.size()) {
          for (size_t i = 0; i < this->goal_list.size(); i++) {
            if (this->state_space.distance(this->x_new, this->goal_list[i]) <
                options.goal_tolerance) {
              path_found = true;
              goal_id = i;
              MESSAGE_PRETTY_DYNORRT("path found -- goal " + std::to_string(i));
              break;
            }
          }

        }

        else {
          if (this->state_space.distance(this->x_new, Base::goal) <
              options.goal_tolerance) {
            path_found = true;
            MESSAGE_PRETTY_DYNORRT("path found");
          }
        }
      }

      num_it++;
      termination_condition = should_terminate();

    } // RRT terminated

    if (termination_condition == TerminationCondition::GOAL_REACHED) {

      int i = this->configs.size() - 1;

      state_t goal;

      if (this->goal_list.size()) {
        goal = this->goal_list[goal_id];
      } else {
        goal = this->goal;
      }

      CHECK_PRETTY_DYNORRT__(
          this->state_space.distance(this->configs[i], goal) <
          options.goal_tolerance);

      this->path = trace_back_solution(i, this->configs, this->parents);

      CHECK_PRETTY_DYNORRT__(
          this->state_space.distance(this->path[0], this->start) < 1e-6);

      std::cout << this->state_space.distance(this->path.back(), goal)
                << std::endl;

      CHECK_PRETTY_DYNORRT__(
          this->state_space.distance(this->path.back(), goal) <
          options.goal_tolerance);

      this->total_distance = 0;
      for (size_t i = 0; i < this->path.size() - 1; i++) {

        double distance =
            this->state_space.distance(this->path[i], this->path[i + 1]);
        // The distance between two points can be slightly bigger than the
        // max_step for non euclidean spaces
        CHECK_PRETTY_DYNORRT__(distance <= 2 * options.max_step);

        this->total_distance +=
            this->state_space.distance(this->path[i], this->path[i + 1]);
      }
    } else {
      MESSAGE_PRETTY_DYNORRT("failed to find a solution!");
      double min_distance = std::numeric_limits<double>::infinity();
      int min_id = -1;
      for (size_t i = 0; i < this->configs.size(); i++) {
        if (this->goal_list.size()) {
          for (size_t j = 0; j < this->goal_list.size(); j++) {
            double distance = this->state_space.distance(this->configs[i],
                                                         this->goal_list[j]);
            if (distance < min_distance) {
              min_distance = distance;
              min_id = i;
            }
          }
        } else {
          double distance =
              this->state_space.distance(this->configs[i], this->goal);
          if (distance < min_distance) {
            min_distance = distance;
            min_id = i;
          }
        }
      }
      MESSAGE_PRETTY_DYNORRT("min_distance: " << min_distance);
      MESSAGE_PRETTY_DYNORRT("min_id: " << min_id);
    }

    MESSAGE_PRETTY_DYNORRT("Output from RRT PLANNER");
    std::cout << "Terminate status: "
              << magic_enum::enum_name(termination_condition) << std::endl;
    std::cout << "num_it: " << num_it << std::endl;
    std::cout << "configs.size(): " << this->configs.size() << std::endl;
    std::cout << "compute time (ms): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - tic)
                     .count()
              << std::endl;
    std::cout << "collisions time (ms): " << this->collisions_time_ms
              << std::endl;
    std::cout << "evaluated_edges: " << this->evaluated_edges << std::endl;
    std::cout << "infeasible_edges: " << this->infeasible_edges << std::endl;
    std::cout << "path.size(): " << this->path.size() << std::endl;
    std::cout << "total_distance: " << this->total_distance << std::endl;

    return termination_condition;
  };

  virtual void get_planner_data(nlohmann::json &j) override {
    Base::get_planner_data(j);
  }

protected:
  RRT_options options;
};

// template <typename StateSpace, int DIM>
// std::shared_ptr<PlannerBase<StateSpace, DIM>>
// get_planner(PlannerID planner_id) {
//   std::cout << "planner_id: " << magic_enum::enum_name(planner_id) <<
//   std::endl; switch (planner_id) { case PlannerID::RRT:
//     return std::make_shared<RRT<StateSpace, DIM>>();
//   case PlannerID::BiRRT:
//     return std::make_shared<BiRRT<StateSpace, DIM>>();
//   case PlannerID::RRTConnect:
//     return std::make_shared<RRTConnect<StateSpace, DIM>>();
//   case PlannerID::RRTStar:
//     return std::make_shared<RRTStar<StateSpace, DIM>>();
//   case PlannerID::PRM:
//     return std::make_shared<PRM<StateSpace, DIM>>();
//   case PlannerID::LazyPRM:
//     return std::make_shared<LazyPRM<StateSpace, DIM>>();
//   default:
//     THROW_PRETTY_DYNORRT("Planner not implemented");
//   }
// }

} // namespace dynorrt
