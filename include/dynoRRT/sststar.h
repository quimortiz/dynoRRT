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

// Reference:
// Asymptotically Optimal Sampling-based Kinodynamic Planning
// Yanbo Li, Zakary Littlefield, Kostas E. Bekris
// https://arxiv.org/pdf/1407.2896.pdf
template <typename StateSpace, int DIM, int DIM_CONTROL>
class SSTstar : public PlannerBase<StateSpace, DIM> {

public:
  using Base = PlannerBase<StateSpace, DIM>;
  using state_t = typename Base::state_t;
  using tree_t = typename Base::tree_t;
  using control_t = Eigen::Matrix<double, DIM_CONTROL, 1>;
  using trajectory_t = Trajectory<DIM, DIM_CONTROL>;
  using DistanceId_t = typename Base::tree_t::DistanceId;

  SSTstar() = default;

  virtual ~SSTstar() = default;

  virtual void print_options(std::ostream &out = std::cout) override {
    std::cout << "Options in " << this->get_name() << std::endl;
    options.print(out);
  }

  virtual void reset() override { *this = SSTstar(); }

  virtual std::string get_name() override { return "SSTstar"; }

  virtual void set_options_from_toml(toml::value &cfg) override {
    options = toml::find<SSTstar_options>(cfg, "SSTstar_options");
  }

  void set_options(SSTstar_options t_options) { options = t_options; }

  SSTstar_options get_options() const { return options; }

  virtual TerminationCondition plan() override {
    Base::check_internal();
    int num_inactive_states_in_tree = 0;
    double max_rate_inactive_in_tree = 0.6;

    int num_collisions = 0;
    auto col = [&, this](const auto &x) {
      num_collisions++;
      // return this->is_collision_free_fun_timed(x);
      return this->is_collision_free_fun(x);
    };

    CHECK_PRETTY_DYNORRT__(col(this->start));
    CHECK_PRETTY_DYNORRT__(col(this->goal));
    CHECK_PRETTY_DYNORRT__(this->state_space.check_bounds(this->start));
    CHECK_PRETTY_DYNORRT__(this->state_space.check_bounds(this->goal));

    std::cout << "State Space: " << std::endl;
    this->state_space.print(std::cout);
    this->print_options();

    this->parents.push_back(-1);
    this->childrens.push_back({});
    this->configs.push_back(this->start);
    this->costs.push_back(0);
    this->tree.addPoint(this->start, 0);
    this->small_trajectories.push_back(trajectory_t());
    v_active.insert(0);
    // witnesses.push_back(0);

    double nn_search_time = 0;
    // TODO: change by a kd-tree!!

    auto nearest_s = [&](const auto &x) {
      auto tic = std::chrono::high_resolution_clock::now();
      // double distance = std::numeric_limits<double>::infinity();
      // int best_id = -1;
      // for (size_t i = 0; i < witnesses.size(); i++) {
      //   double d = this->state_space.distance(x,
      //   this->configs[witnesses[i]]); if (d < distance) {
      //     distance = d;
      //     best_id = i;
      //   }
      // }

      auto nn = this->tree.search(x);
      double elapsed_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::high_resolution_clock::now() - tic)
                              .count() /
                          double(10e6);
      CHECK_PRETTY_DYNORRT__(elapsed_ms > 0.);
      nn_search_time += elapsed_ms;
      return nn;
      // return DistanceId_t{nn_distance, nn_id};
    };

    auto timed_tree_search = [&](const state_t &x) {
      auto tic = std::chrono::high_resolution_clock::now();
      // int nn_id = -1;
      // double nn_distance = std::numeric_limits<double>::infinity();
      // for (auto &id : this->v_active) {
      //   double distance = this->state_space.distance(x, this->configs[id]);
      //   if (distance < nn_distance) {
      //     nn_distance = distance;
      //     nn_id = id;
      //   }
      // }
      auto nn = this->tree.search(x);
      double elapsed_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::high_resolution_clock::now() - tic)
                              .count() /
                          double(10e6);
      CHECK_PRETTY_DYNORRT__(elapsed_ms > 0.);
      nn_search_time += elapsed_ms;

      return nn;
      // return DistanceId_t{nn_distance, nn_id};
    };

    int num_it = 0;
    auto tic = std::chrono::steady_clock::now();
    bool path_found = false;
    bool user_defined_terminate = false;

    auto get_elapsed_ms = [&] {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - tic)
          .count();
    };

    auto should_terminate = [&] {
      if (user_defined_terminate) {
        if (path_found) {
          return TerminationCondition::USER_DEFINED_GOAL_REACHED;
        } else {
          return TerminationCondition::USER_DEFINED;
        }
      }
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
    bool is_goal = false;
    double time_expand_fun = 0;
    double time_store_info = 0;

    std::set<int> goal_ids;
    bool debug = false;
    double best_goal_cost = std::numeric_limits<double>::infinity();

    while (termination_condition == TerminationCondition::RUNNING ||
           termination_condition ==
               TerminationCondition::RUNNING_GOAL_REACHED) {

      // check consistency

      if (num_it % 1000 == 0)
        std::cout << "it " << num_it << std::endl;
      if (debug) {
        CHECK_PRETTY_DYNORRT__(v_active.size() + v_inactive.size() +
                                   v_dead.size() ==
                               this->configs.size());

        CHECK_PRETTY_DYNORRT__(costs.size() == this->configs.size());
        CHECK_PRETTY_DYNORRT__(this->parents.size() == this->configs.size());
        CHECK_PRETTY_DYNORRT__(this->parents.size() == this->childrens.size());
        // std::cout << "witnesses.size(): " << witnesses.size() << std::endl;
        // std::cout << "v_active.size(): " << v_active.size() << std::endl;
        // CHECK_PRETTY_DYNORRT__(witnesses.size() == v_active.size());

        // check consistency between parents and childrens
        for (size_t i = 0; i < this->parents.size(); i++) {
          if (this->parents[i] != -1) {
            auto it = std::find(this->childrens[this->parents[i]].begin(),
                                this->childrens[this->parents[i]].end(), i);
            CHECK_PRETTY_DYNORRT__(it !=
                                   this->childrens[this->parents[i]].end());
          }
        }
        // the other way around

        for (size_t i = 0; i < this->childrens.size(); i++) {
          for (auto &child : this->childrens[i]) {
            CHECK_PRETTY_DYNORRT__(this->parents[child] == i);
          }
        }

        {

          std::vector<int> common_data;
          set_intersection(v_active.begin(), v_active.end(), v_inactive.begin(),
                           v_inactive.end(), std::back_inserter(common_data));
          CHECK_PRETTY_DYNORRT__(common_data.size() == 0);
        }
        {

          std::vector<int> common_data;
          set_intersection(v_active.begin(), v_active.end(), v_dead.begin(),
                           v_dead.end(), std::back_inserter(common_data));
          CHECK_PRETTY_DYNORRT__(common_data.size() == 0);
        }
        {
          std::vector<int> common_data;
          set_intersection(v_inactive.begin(), v_inactive.end(), v_dead.begin(),
                           v_dead.end(), std::back_inserter(common_data));
          CHECK_PRETTY_DYNORRT__(common_data.size() == 0);
        }
      }

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

      auto motion_cost = [&](const auto &trajectory) {
        return trajectory.states.size() - 1;
      };

      auto __tic = std::chrono::steady_clock::now();
      if (is_collision_free) {

        // compute cost

        double cost = costs.at(nn.id) + motion_cost(small_trajectory);

        // Check if node is locally the best

        // if is locally best

        // maybe add here something about Goal Reach!?  -- then I always add it
        // as if it was a novel node.

        enum class STATUS {
          NOVEL,
          BEST,
          DOMINATED,
          UNKNOWN,
          GOAL_BEST,
          GOAL_WORSE
        };
        STATUS status = STATUS::UNKNOWN;

        // If the node is the goal (up to goal tolerance, we take the node)
        DistanceId_t nn_s{std::numeric_limits<double>::infinity(), -1};

        if (this->state_space.distance(this->x_new, this->goal) <
            options.goal_tolerance) {
          if (cost < best_goal_cost) {
            status = STATUS::GOAL_BEST;
          } else {
            status = STATUS::GOAL_WORSE;
          }

        } else {
          nn_s = nearest_s(this->x_new);
          if (nn_s.distance > options.delta_s) {
            status = STATUS::NOVEL;
          } else if (cost < costs.at(nn_s.id)) {
            status = STATUS::BEST;
          } else {
            status = STATUS::DOMINATED;
          }
        }

        if (status == STATUS::BEST || status == STATUS::NOVEL ||
            status == STATUS::GOAL_BEST) {

          // I create a new node
          int next_id = this->configs.size();
          this->configs.push_back(this->x_new);
          this->parents.push_back(nn.id);

          childrens.push_back(std::set<int>());
          childrens.at(nn.id).insert(next_id);

          this->small_trajectories.push_back(small_trajectory);
          v_active.insert(next_id);
          this->tree.addPoint(this->x_new, next_id);
          costs.push_back(cost);

          if (status == STATUS::GOAL_BEST) {
            if (cost < best_goal_cost) {
              MESSAGE_PRETTY_DYNORRT("new best goal cost: " << cost);
              best_goal_cost = cost;
              goal_ids.insert(next_id);

              goal_paths.push_back(
                  trace_back_solution(next_id, Base::configs, Base::parents));

              goal_trajectories.push_back(trace_back_full_traj(
                  this->parents, next_id, this->small_trajectories));
              path_found = true;
              solution_callback(goal_trajectories.back());
              time_stamp_ms.push_back(get_elapsed_ms());
            }
          } else if (status == STATUS::NOVEL) {
            // witnesses.push_back(next_id);
          }

          else if (status == STATUS::BEST) {
            // I set the config pointed by the previous wintess to inactive
            int previous_representative = nn_s.id;
            // witnesses.at(nn_s.id) = next_id;
            v_active.erase(previous_representative);

            // set the node of the tree to inactive

            v_inactive.insert(previous_representative);
            this->tree.set_inactive(this->configs.at(previous_representative));
            num_inactive_states_in_tree++;

            int node = previous_representative;
            while (node != -1 && !childrens.at(node).size() &&
                   v_inactive.find(node) != v_inactive.end()) {
              // I should not kill nodes that are goals!
              if (goal_ids.find(node) != goal_ids.end()) {
                break;
              }

              v_inactive.erase(node);
              v_dead.insert(node);
              int parent = this->parents.at(node);
              // this deletes the edge
              childrens.at(parent).erase(node);
              this->parents.at(node) = -1;
              node = parent;
            }

            // NOTE: I skip the part of erasing nodes from the
            // tree.
          }
        }
      }

      time_store_info += std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - __tic)
                             .count() /
                         double(10e6);

      num_it++;
      termination_condition = should_terminate();

      if (num_inactive_states_in_tree >
          max_rate_inactive_in_tree * this->tree.size()) {
        std::cout << num_inactive_states_in_tree << " " << this->tree.size()
                  << std::endl;
        // lets rebuild the tree.
        this->tree = tree_t();
        this->tree.init_tree(this->runtime_dim, this->state_space);
        std::cout << "Too many inactive states in the tree, rebuilding it"
                  << std::endl;
        for (auto &id : v_active) {
          this->tree.addPoint(this->configs[id], id);
        }
        num_inactive_states_in_tree = 0;
      }

    } // RRT terminated

    double time = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - tic)
                      .count();

    if (is_termination_condition_solved(termination_condition)) {
      this->path = goal_paths.back();
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

    MESSAGE_PRETTY_DYNORRT("Output from SST star PLANNER");
    std::cout << "Terminate status: "
              << magic_enum::enum_name(termination_condition) << std::endl;
    std::cout << "time store info [ms] " << time_store_info << std::endl;
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

    std::cout << "v_active.size(): " << v_active.size() << std::endl;
    std::cout << "v_inactive.size(): " << v_inactive.size() << std::endl;
    std::cout << "v_dead.size(): " << v_dead.size() << std::endl;
    // std::cout << "witnesses.size(): " << witnesses.size() << std::endl;
    std::cout << "paths.size(): " << goal_paths.size() << std::endl;
    std::cout << "goal_ids.size(): " << goal_ids.size() << std::endl;
    std::cout << "goal_trajectory.size(): " << goal_trajectories.size()
              << std::endl;
    std::cout << "states per goal trajectory: " << std::endl;
    for (auto &p : goal_trajectories) {
      std::cout << p.states.size() << std::endl;
    }

    return termination_condition;
  };

  virtual void get_planner_data(nlohmann::json &j) override {
    Base::get_planner_data(j);
    j["small_trajectories"] = small_trajectories;
    j["full_trajectory"] = full_trajectory;

    j["childrens"] = childrens;
    j["costs"] = costs;
    j["v_active"] = v_active;
    j["v_inactive"] = v_inactive;
    j["v_dead"] = v_dead;
    // j["witnesses"] = witnesses;
    j["goal_paths"] = goal_paths;
    j["goal_trajectories"] = goal_trajectories;
    j["time_stamp_ms"] = time_stamp_ms;
  }

  void
  set_expand_fun(std::function<void(state_t &, const state_t &, trajectory_t &)>
                     t_expand_fun) {
    expand_fun = t_expand_fun;
  }

  void set_solution_callback(
      std::function<void(const trajectory_t &traj)> t_solution_callback) {
    solution_callback = t_solution_callback;
  }

protected:
  SSTstar_options options;

  std::function<void(const trajectory_t &traj)> solution_callback =
      [](const trajectory_t &traj) {};

  std::vector<trajectory_t> small_trajectories;
  std::vector<std::vector<state_t>> goal_paths;
  std::vector<trajectory_t> goal_trajectories;
  trajectory_t full_trajectory;
  std::function<void(state_t &, const state_t &, trajectory_t &)> expand_fun;

  std::vector<std::set<int>> childrens;
  std::vector<double> costs;
  std::set<int> v_active;   // index in this->configs
  std::set<int> v_inactive; // index in this->configs
  std::set<int> v_dead;
  std::vector<double> time_stamp_ms;
};

} // namespace dynorrt
