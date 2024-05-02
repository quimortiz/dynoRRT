#pragma once

#include "toml_extra_macros.h"
#include <iostream>
#include <ostream>

namespace dynorrt {

// Note: PRM* and LazyPRM* are implemented by
// setting the suitable options in PRM and LazyPRM respectively.
enum class PlannerID { RRT, BiRRT, RRTConnect, RRTStar, PRM, LazyPRM, UNKNOWN };

enum class TerminationCondition {
  MAX_IT,
  MAX_TIME,
  GOAL_REACHED,
  MAX_NUM_CONFIGS,
  RUNNING,
  USER_DEFINED,
  // The following are for Anytime Assymp optimal planners
  MAX_IT_GOAL_REACHED,
  MAX_TIME_GOAL_REACHED,
  MAX_NUM_CONFIGS_GOAL_REACHED,
  RUNNING_GOAL_REACHED,
  USER_DEFINED_GOAL_REACHED,
  EXTERNAL_TRIGGER_GOAL_REACHED,
  UNKNOWN
};

inline bool is_termination_condition_solved(
    const TerminationCondition &termination_condition) {
  return termination_condition == TerminationCondition::GOAL_REACHED ||
         termination_condition == TerminationCondition::MAX_IT_GOAL_REACHED ||
         termination_condition == TerminationCondition::MAX_TIME_GOAL_REACHED ||
         termination_condition ==
             TerminationCondition::MAX_NUM_CONFIGS_GOAL_REACHED ||
         termination_condition ==
             TerminationCondition::EXTERNAL_TRIGGER_GOAL_REACHED ||
         termination_condition ==
             TerminationCondition::USER_DEFINED_GOAL_REACHED;
}

struct RRT_options {
  int max_it = 10000;
  double goal_bias = 0.05;
  double collision_resolution = 0.01;
  double max_step = 1.;
  double max_compute_time_ms = 1e9;
  double goal_tolerance = 0.001;
  int max_num_configs = 10000;
  int max_num_trials_col_free = 1000;
  bool debug = false;
  bool store_all = false;
  int k_near = -1; // if different than -1, it is used in RRT star -- TODO: RRT
                   // star options!

  void print(std::ostream & = std::cout);
};

struct KinoRRT_options {
  int max_it = 10000;
  double goal_bias = 0.05;
  double collision_resolution = 0.01;
  double max_step = 1.;
  double max_compute_time_ms = 1e9;
  double goal_tolerance = 0.1;
  int max_num_configs = 10000;
  int max_num_trials_col_free = 1000;
  bool debug = false;
  int max_num_kino_steps = 10; // in each expansion,
  int min_num_kino_steps = 5;
  int num_expansions = 1; // if more than one, we keep the one that gets closer
                          // to the random target
  bool store_all = false; // TODO: debug and store all should be only one.

  void print(std::ostream & = std::cout);
};

struct SSTstar_options {
  int max_it = 10000;
  double goal_bias = 0.05;
  double collision_resolution = 0.01;
  double max_step = 1.;
  double max_compute_time_ms = 1e9;
  double goal_tolerance = 0.1;
  int max_num_configs = 10000;
  int max_num_trials_col_free = 1000;
  bool debug = false;
  int max_num_kino_steps = 10; // in each expansion,
  int min_num_kino_steps = 5;
  int num_expansions = 1; // if more than one, we keep the one that gets closer
                          // to the random target
  bool store_all = false;
  double delta_s = .5;

  void print(std::ostream & = std::cout);
};

struct BiRRT_options {
  int max_it = 10000;
  double goal_bias = 0.05;
  double collision_resolution = 0.01;
  double backward_probability = 0.5;
  double max_step = 0.1;
  double max_compute_time_ms = 1e9;
  double goal_tolerance = 0.001;
  int max_num_configs = 10000;
  int max_num_trials_col_free = 1000;

  void print(std::ostream & = std::cout);
};

struct PRM_options {
  // TODO: incrementally + option for PRM star;
  int num_vertices_0 = 200;
  double increase_vertices_rate = 2.;
  double collision_resolution = 0.01;
  int max_it = 10;
  double connection_radius = 1.;
  double max_compute_time_ms = 1e9;
  int max_num_trials_col_free = 1000;
  bool incremental_collision_check = false;
  int k_near = -1;
  void print(std::ostream & = std::cout);
};

struct LazyPRM_options {
  // TODO: incrementally + option for PRM star;
  int num_vertices_0 = 200;
  double increase_vertices_rate = 2.;
  double collision_resolution = 0.01;
  int max_lazy_iterations = 1000;
  double connection_radius = .5;
  double max_compute_time_ms = 1e9;
  int max_num_trials_col_free = 1000;
  int k_near = -1;
  void print(std::ostream & = std::cout);
};

} // namespace dynorrt

TOML11_DEFINE_CONVERSION_NON_INTRUSIVE_OR(dynorrt::RRT_options, max_it,
                                          goal_bias, collision_resolution,
                                          max_step, max_compute_time_ms,
                                          goal_tolerance, max_num_configs,
                                          max_num_trials_col_free, debug,
                                          store_all, k_near);

TOML11_DEFINE_CONVERSION_NON_INTRUSIVE_OR(
    dynorrt::KinoRRT_options, max_it, goal_bias, collision_resolution, max_step,
    max_compute_time_ms, goal_tolerance, max_num_configs,
    max_num_trials_col_free, debug, max_num_kino_steps, min_num_kino_steps,
    num_expansions, store_all);

TOML11_DEFINE_CONVERSION_NON_INTRUSIVE_OR(
    dynorrt::SSTstar_options, max_it, goal_bias, collision_resolution, max_step,
    max_compute_time_ms, goal_tolerance, max_num_configs,
    max_num_trials_col_free, debug, max_num_kino_steps, min_num_kino_steps,
    num_expansions, store_all);

TOML11_DEFINE_CONVERSION_NON_INTRUSIVE_OR(dynorrt::BiRRT_options, max_it,
                                          goal_bias, collision_resolution,
                                          backward_probability, max_step,
                                          max_compute_time_ms, goal_tolerance,
                                          max_num_configs,
                                          max_num_trials_col_free);

TOML11_DEFINE_CONVERSION_NON_INTRUSIVE_OR(
    dynorrt::PRM_options, num_vertices_0, increase_vertices_rate,
    collision_resolution, max_it, connection_radius, max_compute_time_ms,
    max_num_trials_col_free, incremental_collision_check, k_near);

TOML11_DEFINE_CONVERSION_NON_INTRUSIVE_OR(
    dynorrt::LazyPRM_options, num_vertices_0, increase_vertices_rate,
    collision_resolution, max_lazy_iterations, connection_radius,
    max_compute_time_ms, max_num_trials_col_free, k_near);

inline void dynorrt::RRT_options::print(std::ostream &out) {
  toml::value v = *this;
  out << v << std::endl;
}

inline void dynorrt::BiRRT_options::print(std::ostream &out) {
  toml::value v = *this;
  out << v << std::endl;
}

inline void dynorrt::PRM_options::print(std::ostream &out) {
  toml::value v = *this;
  out << v << std::endl;
}

inline void dynorrt::LazyPRM_options::print(std::ostream &out) {
  toml::value v = *this;
  out << v << std::endl;
}

inline void dynorrt::KinoRRT_options::print(std::ostream &out) {
  toml::value v = *this;
  out << v << std::endl;
}

inline void dynorrt::SSTstar_options::print(std::ostream &out) {
  toml::value v = *this;
  out << v << std::endl;
}
