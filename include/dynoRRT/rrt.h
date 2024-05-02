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

    this->state_space.print(std::cout);

    CHECK_PRETTY_DYNORRT__(col(this->start));

    if (this->goal_list.size()) {
      for (auto &goal : this->goal_list) {
        CHECK_PRETTY_DYNORRT__(col(goal));
        CHECK_PRETTY_DYNORRT__(this->state_space.check_bounds(goal));
      }
    } else {
      CHECK_PRETTY_DYNORRT__(col(this->goal));
      CHECK_PRETTY_DYNORRT__(this->state_space.check_bounds(this->goal));
    }
    CHECK_PRETTY_DYNORRT__(this->state_space.check_bounds(this->start));

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
          // get a goal at random
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
        CHECK_PRETTY_DYNORRT(is_collision_free,
                             "cannot generate a valid xrand");
      }
      if (options.debug) {
        this->sample_configs.push_back(this->x_rand);
      }

      auto nn = this->tree.search(this->x_rand);
      this->x_near = this->configs.at(nn.id);

      if (is_goal) {
        // Experimental: if the goal is sampled, then I will try to connect
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
      bool is_collision_free = is_edge_collision_free(
          this->x_near, this->x_new, col, this->state_space,
          options.collision_resolution);
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
          // check againts all the goal

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

template <typename StateSpace, int DIM>
std::shared_ptr<PlannerBase<StateSpace, DIM>>
get_planner(PlannerID planner_id) {
  std::cout << "planner_id: " << magic_enum::enum_name(planner_id) << std::endl;
  switch (planner_id) {
  case PlannerID::RRT:
    return std::make_shared<RRT<StateSpace, DIM>>();
  case PlannerID::BiRRT:
    return std::make_shared<BiRRT<StateSpace, DIM>>();
  case PlannerID::RRTConnect:
    return std::make_shared<RRTConnect<StateSpace, DIM>>();
  case PlannerID::RRTStar:
    return std::make_shared<RRTStar<StateSpace, DIM>>();
  case PlannerID::PRM:
    return std::make_shared<PRM<StateSpace, DIM>>();
  case PlannerID::LazyPRM:
    return std::make_shared<LazyPRM<StateSpace, DIM>>();
  default:
    THROW_PRETTY_DYNORRT("Planner not implemented");
  }
}

// TODO: template on the DIM of state and control.

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
