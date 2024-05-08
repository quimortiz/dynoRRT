#pragma once
#include <algorithm>
#include <iostream>
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


template <typename StateSpace, int DIM>
class BiRRT : public PlannerBase<StateSpace, DIM> {
public:
  using Base = PlannerBase<StateSpace, DIM>;
  using tree_t = typename Base::tree_t;
  using state_t = typename Base::state_t;
  using Configs = std::vector<typename Base::state_t>;

  struct T_helper {
    std::vector<int> *parents;
    std::vector<state_t> *configs;
    tree_t *tree;
    bool is_forward;
  };

  virtual ~BiRRT() = default;

  virtual std::string get_name() override { return "BiRRT"; }

  virtual void reset() override { *this = BiRRT(); }

  void set_options(BiRRT_options t_options) { options = t_options; }
  BiRRT_options get_options() { return options; }

  virtual void print_options(std::ostream &out = std::cout) override {
    options.print(out);
  }

  virtual void set_options_from_toml(toml::value &cfg) override {
    options = toml::find<BiRRT_options>(cfg, "BiRRT_options");
  }

  Configs &get_configs_backward() { return configs_backward; }
  std::vector<int> &get_parents_backward() { return parents_backward; }

  virtual void init(int t_runtime_dim = -1) override {
    Base::init(t_runtime_dim);
    init_backward_tree(t_runtime_dim);
  }

  void init_backward_tree(int t_runtime_dim = -1) {
    this->runtime_dim = t_runtime_dim;
    std::cout << "init tree" << std::endl;
    std::cout << "DIM: " << DIM << std::endl;
    std::cout << "runtime_dim: " << this->runtime_dim << std::endl;

    if constexpr (DIM == -1) {
      if (this->runtime_dim == -1) {
        throw std::runtime_error("DIM == -1 and runtime_dim == -1");
      }
    }
    tree_backward = tree_t();
    tree_backward.init_tree(this->runtime_dim, this->state_space);
  }

  virtual void reset_internal() override {
    Base::reset_internal();
    init_backward_tree();
    parents_backward.clear();
    configs_backward.clear();
  }

  virtual void check_internal() const override {

    CHECK_PRETTY_DYNORRT(tree_backward.size() == 0,
                         "tree_backward.size() != 0");
    CHECK_PRETTY_DYNORRT(parents_backward.size() == 0,
                         "parents_backward.size() != 0");

    CHECK_PRETTY_DYNORRT(configs_backward.size() == 0,
                         "configs_backward.size() != 0");

    this->Base::check_internal();
  }

  virtual TerminationCondition plan() override {

    check_internal();

    std::cout << "Options" << std::endl;
    this->print_options();

    // forward tree
    this->parents.push_back(-1);
    this->configs.push_back(this->start);
    this->tree.addPoint(this->start, 0);

    // backward tree
    parents_backward.push_back(-1);
    configs_backward.push_back(this->goal);
    tree_backward.addPoint(this->goal, 0);

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
      if (this->configs.size() + configs_backward.size() >
          options.max_num_configs) {
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
    bool expand_forward = true;

    int connect_id_forward = -1;
    int connect_id_backward = -1;

    T_helper Ta{.parents = &this->parents,
                .configs = &this->configs,
                .tree = &this->tree,
                .is_forward = true

    };

    T_helper Tb{.parents = &parents_backward,
                .configs = &configs_backward,
                .tree = &tree_backward,
                .is_forward = false};

    T_helper Tsrc, Ttgt;

    while (termination_condition == TerminationCondition::RUNNING) {

      expand_forward = static_cast<double>(std::rand()) / RAND_MAX >
                       options.backward_probability;

      if (expand_forward) {
        Tsrc = Ta;
        Ttgt = Tb;
      } else {
        Tsrc = Tb;
        Ttgt = Ta;
      }

      int goal_id = -1;
      bool goal_connection_attempt =
          static_cast<double>(std::rand()) / RAND_MAX < options.goal_bias;
      if (goal_connection_attempt) {
        goal_id = std::rand() % Ttgt.configs->size();
        this->x_rand = Ttgt.configs->at(goal_id);
      } else {
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
      this->sample_configs.push_back(this->x_rand);
      auto nn = Tsrc.tree->search(this->x_rand);
      // std::cout << "nn.id: " << nn.id << std::endl;
      this->x_near = Tsrc.configs->at(nn.id);

      bool full_step_attempt = nn.distance < options.max_step;
      if (full_step_attempt) {
        this->x_new = this->x_rand;
      } else {
        this->state_space.interpolate(this->x_near, this->x_rand,
                                      options.max_step / nn.distance,
                                      this->x_new);
      }

      this->evaluated_edges += 1;
      bool is_collision_free = is_edge_collision_free(
          this->x_near, this->x_new, col, this->state_space,
          options.collision_resolution);

      if (is_collision_free) {
        this->valid_edges.push_back({this->x_near, this->x_new});
      } else if (!is_collision_free) {
        this->invalid_edges.push_back({this->x_near, this->x_new});
      }

      this->infeasible_edges += !is_collision_free;

      if (is_collision_free) {
        Tsrc.tree->addPoint(this->x_new, Tsrc.tree->size());
        Tsrc.configs->push_back(this->x_new);
        Tsrc.parents->push_back(nn.id);
        CHECK_PRETTY_DYNORRT__(Tsrc.tree->size() == Tsrc.configs->size());
        CHECK_PRETTY_DYNORRT__(Tsrc.tree->size() == Tsrc.parents->size());

        if (full_step_attempt && goal_connection_attempt) {
          path_found = true;
          MESSAGE_PRETTY_DYNORRT("Path found!");
          CHECK_PRETTY_DYNORRT__(this->state_space.distance(
                                     this->x_new, Ttgt.configs->at(goal_id)) <
                                 options.goal_tolerance);

          if (expand_forward) {
            connect_id_forward = Tsrc.tree->size() - 1;
            connect_id_backward = goal_id;
          } else {
            connect_id_forward = goal_id;
            connect_id_backward = Tsrc.tree->size() - 1;
          }
        }
      }

      num_it++;
      termination_condition = should_terminate();

    } // RRT CONNECT terminated

    if (termination_condition == TerminationCondition::GOAL_REACHED) {

      CHECK_PRETTY_DYNORRT__(this->configs.size() >= 1);
      CHECK_PRETTY_DYNORRT__(configs_backward.size() >= 1);

      auto fwd_path =
          trace_back_solution(connect_id_forward, this->configs, this->parents);

      auto bwd_path = trace_back_solution(connect_id_backward, configs_backward,
                                          parents_backward);

      CHECK_PRETTY_DYNORRT__(
          this->state_space.distance(fwd_path.back(), bwd_path.back()) <
          options.goal_tolerance);

      std::reverse(bwd_path.begin(), bwd_path.end());

      this->path.insert(this->path.end(), fwd_path.begin(), fwd_path.end());
      this->path.insert(this->path.end(), bwd_path.begin() + 1, bwd_path.end());
      this->total_distance = get_path_length(this->path, this->state_space);
    }

    return termination_condition;
    //
  }

  void get_planner_data(json &j) override {
    Base::get_planner_data(j);
    j["parents_backward"] = parents_backward;
    j["configs_backward"] = configs_backward;
  }

protected:
  BiRRT_options options;
  typename Base::tree_t tree_backward;
  std::vector<int> parents_backward;
  std::vector<typename Base::state_t> configs_backward;
};


} // namespace dynorrt
