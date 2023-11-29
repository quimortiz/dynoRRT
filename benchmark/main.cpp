#include "dynoRRT/dynorrt_macros.h"
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>

#include <string>
#define BOOST_TEST_MODULE benchmark
#define BOOST_TEST_DYN_LINK

#include "dynotree/KDTree.h"
#include "magic_enum.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>

#include "dynoRRT/collision_manager.h"
#include "dynoRRT/eigen_conversions.hpp"
#include "dynoRRT/rrt.h"
#include <boost/test/unit_test.hpp>
#include <ompl/geometric/planners/rrt/RRT.h>

// #include "pinocchio/parsers/srdf.hpp"
// #include "pinocchio/parsers/urdf.hpp"
// #include "pinocchio/algorithm/geometry.hpp"

// #include <hpp/fcl/collision_object.h>
// #include <hpp/fcl/shape/geometric_shapes.h>
#include <iostream>

#include "dynotree/KDTree.h"
#include "magic_enum.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>

#include "dynoRRT/collision_manager.h"
#include "dynoRRT/eigen_conversions.hpp"
#include "dynoRRT/rrt.h"
#include <boost/test/unit_test.hpp>

// #include "pinocchio/parsers/srdf.hpp"
// #include "pinocchio/parsers/urdf.hpp"

// #include "pinocchio/algorithm/geometry.hpp"

// #include <hpp/fcl/collision_object.h>
// #include <hpp/fcl/shape/geometric_shapes.h>
#include <iostream>

// benchmark against OMPL
//
//
//

#include <iostream>

#include "ompl/base/SpaceInformation.h"
#include "ompl/base/spaces/RealVectorStateSpace.h"
#include "ompl/geometric/PathGeometric.h"

#include "ompl/base/ScopedState.h"

namespace ob = ompl::base;
namespace og = ompl::geometric;

class RRTstar_with_NodeCount : public ompl::geometric::RRTstar {
public:
  using ompl::geometric::RRTstar::RRTstar;

  // Function to return the number of nodes
  std::size_t getNodeCount() const { return nn_->size(); }
};

struct Statistic {
  double median;
  double first_quartile;
  double third_quartile;

  friend std::ostream &operator<<(std::ostream &os, const Statistic &stat) {
    os << "median: " << stat.median
       << " first_quartile: " << stat.first_quartile
       << " third_quartile: " << stat.third_quartile;
    return os;
  }
};

Statistic get_statistic(const std::vector<double> &v) {
  std::vector<double> v_copy = v;
  std::sort(v_copy.begin(), v_copy.end());
  return Statistic{.median = v_copy[v_copy.size() / 2],
                   .first_quartile = v_copy[v_copy.size() / 4],
                   .third_quartile = v_copy[3 * v_copy.size() / 4]};
}

// template <typename T> double get_median(const std::vector<T> &v) {
//   std::vector<T> v_copy = v;
//   std::sort(v_copy.begin(), v_copy.end());
//   return v_copy[v_copy.size() / 2];
// }
//
// double get_first_quartile(const std::vector<double> &v) {
//   std::vector<double> v_copy = v;
//   std::sort(v_copy.begin(), v_copy.end());
//   return v_copy[v_copy.size() / 4];
// }
//
// double get_third_quartile(const std::vector<double> &v) {
//   std::vector<double> v_copy = v;
//   std::sort(v_copy.begin(), v_copy.end());
//   return v_copy[3 * v_copy.size() / 4];
// }

BOOST_AUTO_TEST_CASE(t_bench_rrt) {
  std::cout << "bench_rrt" << std::endl;

  double goal_bias = 0.05;
  double collision_resolution = 0.01;
  double max_step = .005;
  int num_runs = 100;

  std::vector<double> run_ms(num_runs);
  std::vector<double> run_cost(num_runs);
  std::vector<double> run_length(num_runs);

  for (int i = 0; i < num_runs; i++) {
    using state_space_t = dynotree::Rn<double, 3>;
    using tree_t = dynotree::KDTree<int, -1, 32, double, state_space_t>;

    state_space_t state_space;
    state_space.set_bounds(Eigen::Vector3d(-2, -2, -2),
                           Eigen::Vector3d(2, 2, 2));
    // tree_t tree(state_space);

    Eigen::Vector3d q_i(-1., -1., -1.);
    Eigen::Vector3d q_g(1, 1, 1);
    RRT<state_space_t, -1> rrt;
    rrt.init(3);
    rrt.set_state_space(state_space);
    rrt.set_start(q_i);
    rrt.set_goal(q_g);

    RRT_options options{.max_it = 100000,
                        .goal_bias = goal_bias,
                        .collision_resolution = collision_resolution,
                        .max_step = max_step,
                        .max_compute_time_ms = 1e9,
                        .goal_tolerance = 1e-8,
                        .max_num_configs = 10000};

    rrt.set_is_collision_free_fun([](const auto &s) { return true; });
    rrt.set_options(options);

    auto tic = std::chrono::steady_clock::now();
    rrt.plan();
    auto toc = std::chrono::steady_clock::now();
    double elapsemd_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(toc - tic)
            .count() /
        1000.;
    run_ms[i] = elapsemd_time_ms;
    std::cout << "done " << std::endl;
    std::cout << "elapsemd_time_ms " << elapsemd_time_ms << std::endl;

    run_length[i] = rrt.get_path().size();
    double cost = 0;

    for (size_t i = 0; i < rrt.get_path().size() - 1; i++) {
      cost += (rrt.get_path()[i + 1] - rrt.get_path()[i]).norm();
    }

    run_cost[i] = cost;
  }

  // run OMPL RRT
  //

  std::vector<double> run_ms_ompl(num_runs);
  std::vector<double> runs_cost_ompl(num_runs);
  std::vector<double> run_length_ompl(num_runs);
  for (size_t i = 0; i < num_runs; i++) {
    auto space(std::make_shared<ob::RealVectorStateSpace>());
    space->addDimension(-2., 2.);
    space->addDimension(-2., 2.);
    space->addDimension(-2., 2.);
    auto si(std::make_shared<ob::SpaceInformation>(space));
    si->setStateValidityChecker([](const ob::State *state) { return true; });
    si->setStateValidityCheckingResolution(collision_resolution);
    si->setup();

    ob::ScopedState<> start(space);
    start[0] = -1.;
    start[1] = -1.;
    start[2] = -1.;
    ob::ScopedState<> goal(space);
    goal[0] = 1.;
    goal[1] = 1.;
    goal[2] = 1.;

    auto planner = std::make_shared<og::RRT>(si);

    // create a problem instance
    auto pdef(std::make_shared<ob::ProblemDefinition>(si));

    // set the start and goal states
    pdef->setStartAndGoalStates(start, goal);

    // create a planner for the defined space

    // set the problem we are trying to solve for the planner
    planner->setProblemDefinition(pdef);

    // perform setup steps for the planner
    planner->setup();

    // print the settings for this space
    si->printSettings(std::cout);

    // print the problem settings
    pdef->print(std::cout);

    planner->setGoalBias(goal_bias);
    planner->setRange(max_step);

    // attempt to solve the problem within one second of planning time

    auto tic = std::chrono::steady_clock::now();
    ob::PlannerStatus solved = planner->ob::Planner::solve(1.0);
    auto toc = std::chrono::steady_clock::now();

    double elapsed_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(toc - tic)
            .count() /
        1000.;

    run_ms_ompl[i] = elapsed_time_ms;

    std::cout << "elapsed ms " << elapsed_time_ms << std::endl;

    if (solved) {
      // get the goal representation from the problem definition (not the same
      // as the goal state) and inquire about the found path
      ob::PathPtr path = pdef->getSolutionPath();
      std::cout << "Found solution:" << std::endl;
      // runs_cost_ompl[i] = path->cost();
      run_length_ompl[i] = path->as<og::PathGeometric>()->getStateCount();

      std::vector<Eigen::Vector3d> path_points;

      for (size_t i = 0; i < path->as<og::PathGeometric>()->getStateCount();
           i++) {
        auto state = path->as<og::PathGeometric>()->getState(i);
        Eigen::Vector3d p;
        p[0] = state->as<ob::RealVectorStateSpace::StateType>()->values[0];
        p[1] = state->as<ob::RealVectorStateSpace::StateType>()->values[1];
        p[2] = state->as<ob::RealVectorStateSpace::StateType>()->values[2];
        std::cout << p.transpose() << std::endl;
        path_points.push_back(p);
      }

      double cost = 0;
      for (size_t i = 0; i < path_points.size() - 1; i++) {
        cost += (path_points[i + 1] - path_points[i]).norm();
      }
      runs_cost_ompl[i] = cost;

      // print the path to screen
    } else
      std::cout << "No solution found" << std::endl;
  }

  auto stat_ms = get_statistic(run_ms);
  auto stat_cost = get_statistic(run_cost);
  auto stat_length = get_statistic(run_length);

  std::cout << "DynoRRT" << std::endl;
  std::cout << "stat_ms: " << stat_ms << std::endl;
  std::cout << "stat_cost: " << stat_cost << std::endl;
  std::cout << "stat_length: " << stat_length << std::endl;

  auto stat_ms_ompl = get_statistic(run_ms_ompl);
  auto stat_cost_ompl = get_statistic(runs_cost_ompl);
  auto stat_length_ompl = get_statistic(run_length_ompl);

  std::cout << "OMPL" << std::endl;
  std::cout << "stat_ms: " << stat_ms_ompl << std::endl;
  std::cout << "stat_cost: " << stat_cost_ompl << std::endl;
  std::cout << "stat_length: " << stat_length_ompl << std::endl;
}

BOOST_AUTO_TEST_CASE(t_bench_rrtstar) {

  double goal_bias = 0.05;
  double collision_resolution = 0.01;
  double max_step = .1;
  double max_compute_time_ms = 100;
  int num_runs = 100;

  std::vector<double> run_ms(num_runs);
  std::vector<double> run_cost(num_runs);
  std::vector<double> run_length(num_runs);
  std::vector<double> solved(num_runs);
  std::vector<double> num_configs(num_runs);
  for (int i = 0; i < num_runs; i++) {
    using state_space_t = dynotree::Rn<double, 3>;
    using tree_t = dynotree::KDTree<int, -1, 32, double, state_space_t>;

    state_space_t state_space;
    state_space.set_bounds(Eigen::Vector3d(-2, -2, -2),
                           Eigen::Vector3d(2, 2, 2));

    Eigen::Vector3d q_i(-1., -1., -1.);
    Eigen::Vector3d q_g(1, 1, 1);
    RRTStar<state_space_t, -1> rrt;

    rrt.init(3);
    rrt.set_state_space(state_space);
    rrt.set_start(q_i);
    rrt.set_goal(q_g);

    RRT_options options{.max_it = 1000000,
                        .goal_bias = goal_bias,
                        .collision_resolution = collision_resolution,
                        .max_step = max_step,
                        .max_compute_time_ms = max_compute_time_ms,
                        .goal_tolerance = 1e-8,
                        .max_num_configs = 1000000};

    rrt.set_is_collision_free_fun([](const auto &s) { return true; });
    rrt.set_options(options);

    auto tic = std::chrono::steady_clock::now();
    auto status = rrt.plan();
    auto toc = std::chrono::steady_clock::now();
    double elapsemd_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(toc - tic)
            .count() /
        1000.;
    std::cout << "elapsemd_time_ms " << elapsemd_time_ms << std::endl;
    run_ms[i] = elapsemd_time_ms;
    solved[i] = is_termination_condition_solved(status);
    num_configs[i] = rrt.get_configs().size();

    std::cout << "done " << std::endl;
    std::cout << "elapsemd_time_ms " << elapsemd_time_ms << std::endl;

    run_length[i] = rrt.get_path().size();
    double cost = 0;

    for (size_t i = 0; i < rrt.get_path().size() - 1; i++) {
      cost += (rrt.get_path()[i + 1] - rrt.get_path()[i]).norm();
    }

    run_cost[i] = cost;
  }

  // run OMPL RRT Star

  // continue here! ...
  std::vector<double> run_ms_ompl(num_runs);
  std::vector<double> runs_cost_ompl(num_runs);
  std::vector<double> run_length_ompl(num_runs);
  std::vector<double> solved_ompl(num_runs);
  std::vector<double> num_configs_ompl(num_runs);
  for (size_t i = 0; i < num_runs; i++) {
    auto space(std::make_shared<ob::RealVectorStateSpace>());
    space->addDimension(-2., 2.);
    space->addDimension(-2., 2.);
    space->addDimension(-2., 2.);
    auto si(std::make_shared<ob::SpaceInformation>(space));
    si->setStateValidityChecker([](const ob::State *state) { return true; });
    si->setStateValidityCheckingResolution(collision_resolution);
    si->setup();

    ob::ScopedState<> start(space);
    start[0] = -1.;
    start[1] = -1.;
    start[2] = -1.;
    ob::ScopedState<> goal(space);
    goal[0] = 1.;
    goal[1] = 1.;
    goal[2] = 1.;

    // create a problem instance
    auto pdef(std::make_shared<ob::ProblemDefinition>(si));

    // set the start and goal states
    pdef->setStartAndGoalStates(start, goal);

    // create a planner for the defined space

    // set the problem we are trying to solve for the planner

    pdef->setOptimizationObjective(
        std::make_shared<ob::PathLengthOptimizationObjective>(si));

    // print the settings for this space
    si->printSettings(std::cout);

    // print the problem settings
    pdef->print(std::cout);

    ob::PlannerPtr planner = std::make_shared<RRTstar_with_NodeCount>(si);
    planner->setProblemDefinition(pdef);
    // perform setup steps for the planner
    planner->setup();

    planner->as<og::RRTstar>()->setGoalBias(goal_bias);
    planner->as<og::RRTstar>()->setRange(max_step);

    // attempt to solve the problem within one second of planning time
    auto tic = std::chrono::steady_clock::now();
    ob::PlannerStatus solved =
        planner->ob::Planner::solve(max_compute_time_ms / 1000.);
    auto toc = std::chrono::steady_clock::now();

    double elapsed_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(toc - tic)
            .count() /
        1000.;

    run_ms_ompl[i] = elapsed_time_ms;

    std::cout << "elapsed ms " << elapsed_time_ms << std::endl;

    if (solved) {
      solved_ompl[i] = 1;
      // get the goal representation from the problem definition (not the same
      // as the goal state) and inquire about the found path

      // get the number of vertices in the tree in the RRT
      num_configs_ompl[i] =
          planner->as<RRTstar_with_NodeCount>()->getNodeCount();

      ob::PathPtr path = pdef->getSolutionPath();
      std::cout << "Found solution:" << std::endl;
      // runs_cost_ompl[i] = path->cost();
      run_length_ompl[i] = path->as<og::PathGeometric>()->getStateCount();

      std::vector<Eigen::Vector3d> path_points;

      for (size_t i = 0; i < path->as<og::PathGeometric>()->getStateCount();
           i++) {
        auto state = path->as<og::PathGeometric>()->getState(i);
        Eigen::Vector3d p;
        p[0] = state->as<ob::RealVectorStateSpace::StateType>()->values[0];
        p[1] = state->as<ob::RealVectorStateSpace::StateType>()->values[1];
        p[2] = state->as<ob::RealVectorStateSpace::StateType>()->values[2];
        std::cout << p.transpose() << std::endl;
        path_points.push_back(p);
      }

      double cost = 0;
      for (size_t i = 0; i < path_points.size() - 1; i++) {
        cost += (path_points[i + 1] - path_points[i]).norm();
      }
      runs_cost_ompl[i] = cost;

      // print the path to screen
    } else
      solved_ompl[i] = 0.;
  }

  std::cout << "DynoRRT" << std::endl;
  auto stat_ms = get_statistic(run_ms);
  auto stat_cost = get_statistic(run_cost);
  auto stat_length = get_statistic(run_length);

  std::cout << "stat_ms: " << stat_ms << std::endl;
  std::cout << "stat_cost: " << stat_cost << std::endl;
  std::cout << "stat_length: " << stat_length << std::endl;
  std::cout << "num configs: " << get_statistic(num_configs) << std::endl;
  std::cout << "solved: "
            << std::accumulate(solved.begin(), solved.end(), 0.) /
                   double(solved.size())
            << std::endl;

  std::cout << "OMPL" << std::endl;
  auto stat_ms_ompl = get_statistic(run_ms_ompl);
  auto stat_cost_ompl = get_statistic(runs_cost_ompl);
  auto stat_length_ompl = get_statistic(run_length_ompl);
  std::cout << "stat_ms: " << stat_ms_ompl << std::endl;
  std::cout << "stat_cost: " << stat_cost_ompl << std::endl;
  std::cout << "stat_length: " << stat_length_ompl << std::endl;
  std::cout << "num configs: " << get_statistic(num_configs_ompl) << std::endl;
  std::cout << "solved: "
            << std::accumulate(solved_ompl.begin(), solved_ompl.end(), 0.) /
                   double(solved_ompl.size())
            << std::endl;
}

// return si;

// int main(){
//   bench_rrt();
//
//
//
//
//
// }
