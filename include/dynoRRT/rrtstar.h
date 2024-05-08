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



// Reference:
// Sampling-based Algorithms for Optimal Motion Planning
// Sertac Karaman Emilio Frazzoli
// Algorithm 6
// https://arxiv.org/pdf/1105.1186.pdf
template <typename StateSpace, int DIM>
class RRTStar : public PlannerBase<StateSpace, DIM> {

  // TODO: add flag to be a pure rrt at the beginning, and only transition to
  // rewiring once we have a solution!

  using Base = PlannerBase<StateSpace, DIM>;
  using state_t = typename Base::state_t;

public:
  RRTStar() = default;

  virtual ~RRTStar() = default;

  virtual void print_options(std::ostream &out = std::cout) override {
    options.print(out);
  }

  virtual void reset() override { *this = RRTStar(); }

  virtual std::string get_name() override { return "RRTStar"; }

  virtual void set_options_from_toml(toml::value &cfg) override {
    options = toml::find<RRT_options>(cfg, "RRTStar_options");
  }

  // Lets do recursive version first
  void update_children(int start, double difference,
                       const std::vector<int> &parents,
                       const std::vector<std::set<int>> &children,
                       std::vector<double> &cost_to_come, int &counter) {
    counter++;
    CHECK_PRETTY_DYNORRT__(counter < parents.size());
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

    std::cout << "Options" << std::endl;
    this->print_options();

    this->parents.push_back(-1);
    this->children.push_back(std::set<int>());
    this->configs.push_back(Base::start);
    this->cost_to_come.push_back(0.);
    this->tree.addPoint(Base::start, 0);

    int num_it = 0;
    int goal_id = -1;
    double best_cost = std::numeric_limits<double>::infinity();
    bool path_found = false;
    auto tic = std::chrono::steady_clock::now();

    auto col = [this](const auto &x) {
      return this->Base::is_collision_free_fun_timed(x);
    };

    auto get_elapsed_ms = [&] {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - tic)
          .count();
    };

    bool user_defined_terminate = false;

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

    bool informed_rrt_star = true;
    std::cout << "informed_rrt_star: " << informed_rrt_star << std::endl;
    int max_attempts_informed = 1000;
    bool is_goal = false;
    while (termination_condition == TerminationCondition::RUNNING ||
           termination_condition ==
               TerminationCondition::RUNNING_GOAL_REACHED) {

      // check that the goal_id is never a parent of a node

      if (this->options.debug) {

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
      }

      if (static_cast<double>(std::rand()) / RAND_MAX < options.goal_bias) {
        this->x_rand = Base::goal;
        is_goal = true;
      } else {
        bool is_collision_free = false;
        int num_tries = 0;
        while (!is_collision_free &&
               num_tries < options.max_num_trials_col_free) {

          if (informed_rrt_star) {
            int max_attempts_informed = 1000;
            int num_attempts = 0;
            bool found = false;
            while (num_attempts < max_attempts_informed) {
              num_attempts++;

              // TODO: change the bounds that you sample on!!!
              Base::state_space.sample_uniform(this->x_rand);
              double distance_to_goal =
                  Base::state_space.distance(this->x_rand, this->goal);
              double distance_to_goal_start =
                  Base::state_space.distance(this->x_rand, this->start);
              if (distance_to_goal + distance_to_goal_start < best_cost) {
                found = true;
                break;
              }
            }
            if (!found) {
              user_defined_terminate = true;
              MESSAGE_PRETTY_DYNORRT("using informed RRT star -- suitable "
                                     "sample not found in " +
                                     std::to_string(max_attempts_informed) +
                                     " attempt\n  "
                                     "set user_defined_terminate to true");
            }
          } else {
            Base::state_space.sample_uniform(this->x_rand);
          }
          is_collision_free = col(this->x_rand);
          num_tries++;
        }
        CHECK_PRETTY_DYNORRT(is_collision_free,
                             "cannot generate a valid xrand");
      }

      Base::sample_configs.push_back(this->x_rand);

      auto nn1 = Base::tree.search(this->x_rand);

      if (nn1.id == goal_id) {
        // I dont want to put nodes as children of the goal
        nn1 = Base::tree.searchKnn(this->x_rand, 2).at(1);
      }

      this->x_near = Base::configs.at(nn1.id);

      if (is_goal) {
        // Experimental: if the goal is sampled, then I will try to connect
        // a random state with 50%, not only the closest one!
        if (static_cast<double>(std::rand()) / RAND_MAX < 0.5) {
          int rand_id = -1;
          while (rand_id == -1 || rand_id == goal_id) {
            rand_id = std::rand() % this->configs.size();
          }
          this->x_near = this->configs.at(rand_id);
          nn1.id = rand_id;
          nn1.distance = this->state_space.distance(this->x_rand, this->x_near);
        }
      }

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

        std::vector<typename Base::tree_t::DistanceId> _nns;
        if (options.k_near > 0) {
          _nns = this->tree.searchKnn(this->x_new, options.k_near);
        } else {
          _nns = this->tree.searchBall(this->x_new, radius_search);
        }
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
        // std::cout << "nn1.id: " << nn1.id << std::endl;
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
                // MESSAGE_PRETTY_DYNORRT("rewiring");
                // std::cout << "_nn.id: " << _nn.id << std::endl;
                // std::cout << "id new: " << id_new << std::endl;
                // std::cout << "tentative_g: " << tentative_g << std::endl;
                // std::cout << "current_g: " << current_g << std::endl;
                double difference = tentative_g - current_g;
                CHECK_PRETTY_DYNORRT__(difference < 0);
                int counter = 0;

                if (this->options.debug)
                  ensure_childs_and_parents(children, this->parents);

                // std::cout << "CHILDREN" << std::endl;
                // int _counter = 0;
                // for (auto &x : this->children) {
                //   std::cout << _counter++ << ": ";
                //   for (auto &y : x) {
                //     std::cout << y << " ";
                //   }
                //   std::cout << std::endl;
                // }

                // std::cout << "PARENTS" << std::endl;
                // counter = 0;
                // for (auto &x : this->parents) {
                //   std::cout << counter++ << ": " << x << std::endl;
                // }

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
          MESSAGE_PRETTY_DYNORRT("New Path Found!" << " Number paths "
                                                   << paths.size());
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
          Base::state_space.distance(Base::path.back(), Base::goal) <
          options.goal_tolerance);

      Base::total_distance = 0;
      for (size_t i = 0; i < Base::path.size() - 1; i++) {

        double distance =
            Base::state_space.distance(Base::path[i], Base::path[i + 1]);
        CHECK_PRETTY_DYNORRT__(distance <=
                               2 * options.max_step); // note, distance can be
                                                      // slightly bigger in non
        // euclidean spaces

        Base::total_distance +=
            Base::state_space.distance(Base::path[i], Base::path[i + 1]);
      }
    }

    MESSAGE_PRETTY_DYNORRT("Output from RRT Star PLANNER");
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
    std::cout << "paths.size(): " << this->paths.size() << std::endl;
    std::cout << "total_distance: " << Base::total_distance << std::endl;
    std::cout << "Straight-line distance start-goal: "
              << this->state_space.distance(this->start, this->goal)
              << std::endl;

    return termination_condition;
  };

  virtual void get_planner_data(nlohmann::json &j) override {
    Base::get_planner_data(j);
    j["cost_to_come"] = cost_to_come;
    j["children"] = children;
    j["paths"] = paths;
  }

protected:
  std::vector<double> cost_to_come;
  std::vector<std::set<int>> children;
  std::vector<std::vector<state_t>> paths;
  RRT_options options;
};


} // namespace dynorrt
