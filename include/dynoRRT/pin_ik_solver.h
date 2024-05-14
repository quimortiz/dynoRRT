#pragma once
//
#include "dynoRRT/dynorrt_macros.h"
#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"
//

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include <Eigen/Dense>

#include "dynoRRT/utils.h"
#include <hpp/fcl/collision_object.h>

//
//
// #include "BS_thread_pool.hpp"

#include "thread-pool/BS_thread_pool.hpp"

#include "pinocchio/multibody/fwd.hpp"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>

#include <hpp/fcl/collision_object.h>
#include <string>
#include <vector>

enum class IKStatus { RUNNING, SUCCESS, UNKNOWN, MAX_ATTEMPTS, MAX_TIME };

namespace dynorrt {

class Pin_ik_solver {

public:
  void set_urdf_filename(const std::string &t_urdf_filename) {
    urdf_filename = t_urdf_filename;
  }

  void set_srdf_filename(const std::string &t_srdf_filename) {
    srdf_filename = t_srdf_filename;
  }
  void set_robots_model_path(const std::string &t_robot_model_path) {
    robots_model_path = t_robot_model_path;
  }

  void set_num_threads(int num_threads) { num_threads_edges = num_threads; }

  void set_pin_model(pinocchio::Model &t_model,
                     pinocchio::GeometryModel &t_geomodel);

  void build();

  void set_use_aabb(bool use) {
    THROW_PRETTY_DYNORRT("not implemented");
    // use_aabb = use;
  }

  void set_bounds(const Eigen::VectorXd &t_x_lb,
                  const Eigen::VectorXd &t_x_ub) {
    x_lb = t_x_lb;
    x_ub = t_x_ub;
  }

  void set_p_des(const Eigen::VectorXd &t_p_des) { p_des = t_p_des; }

  // void set_q_des(const Eigen::VectorXd &t_q_des) { q_des = t_q_des; }

  void set_pq_des(const pinocchio::SE3 &t_pq_des) {
    pq_des = t_pq_des;
    flag_desired_pq = true;
  }

  void set_max_num_attempts(int num) { max_num_attempts = num; }

  void set_max_solutions(int num) { max_solutions = num; }

  void set_max_time_ms(int num) { max_time_ms = num; }

  auto get_model_data_ptr() { return std::make_tuple(&model, &data); }

  double get_distance_cost(const Eigen::VectorXd &q) {

    double cost_dist = 0;
    // Collisions
    size_t min_index = computeDistances(model, data, geom_model, geom_data, q);

    double min_dist =
        geom_data.distanceResults[min_index].min_distance - col_margin;

    // std::cout << "min_dist: " << min_dist << std::endl;

    // lets assume this is a signed distance function!!
    if (min_dist < 0) {
      cost_dist = min_dist * min_dist;
    } else {
      cost_dist = 0;
    }
    return .5 * cost_dist;
  }

  double get_bounds_cost(const Eigen::VectorXd &q) {

    double cost_bounds = 0;

    for (size_t i = 0; i < q.size(); i++) {
      if (q[i] < x_lb[i]) {
        cost_bounds += (x_lb[i] - q[i]) * (x_lb[i] - q[i]);
      } else if (q[i] > x_ub[i]) {
        cost_bounds += (q[i] - x_ub[i]) * (q[i] - x_ub[i]);
      }
    }
    return .5 * cost_bounds;
  }

  double get_frame_cost(const Eigen::VectorXd &q) {

    pinocchio::framesForwardKinematics(model, data, q);

    double cost_frame = 0;
    if (flag_desired_pq) {
      const pinocchio::SE3 iMd = data.oMf[frame_id].actInv(pq_des);
      Eigen::VectorXd err = pinocchio::log6(iMd).toVector(); // in joint frame
      cost_frame = .5 * err.squaredNorm();
    } else {
      // std::cout << "trans:" << data.oMf[frame_id].translation() << std::endl;
      // std::cout << "p_des:" << p_des << std::endl;
      Eigen::VectorXd tool_nu = data.oMf[frame_id].translation() - p_des;
      cost_frame = .5 * tool_nu.squaredNorm();
    }

    return cost_frame;
  }

  double get_cost(const Eigen::VectorXd &q) {

    double cost_dist = get_distance_cost(q);
    double cost_bounds = get_bounds_cost(q);
    double cost_frame = get_frame_cost(q);

    // std::cout << "evaluating at q: " << q.transpose() << " cost: "
    //           << col_penalty * cost_dist + bound_penalty * cost_bounds +
    //                  frame_penalty * cost_frame
    //           << std::endl;
    // std::cout << "cost_dist: " << cost_dist << " cost_bounds: " << cost_bounds
    //           << " cost_frame: " << cost_frame << std::endl;
    return col_penalty * cost_dist + bound_penalty * cost_bounds +
           frame_penalty * cost_frame;
  }

  void get_cost_derivative(const Eigen::VectorXd &q, Eigen::VectorXd &dq) {
    double eps = 1e-5;
    double fq = get_cost(q);
    Eigen::VectorXd qplus = q;

    for (size_t i = 0; i < q.size(); i++) {
      qplus = q;
      qplus[i] += eps;
      dq[i] = (get_cost(qplus) - fq) / eps;
    }
  }

  IKStatus solve_ik();

  std::vector<Eigen::VectorXd> get_ik_solutions() { return ik_solutions; }

  void set_tolerances(double t_bound_tol, double t_col_tol,
                      double t_frame_tol) {
    bound_tol = t_bound_tol;
    col_tol = t_col_tol;
    frame_tol = t_frame_tol;
  }

  void set_frame_id(int id) { frame_name = id; }

  void set_frame_name(const std::string &t_name) {
    frame_name = t_name;
    frame_id = model.getFrameId(frame_name);
  }

  void set_col_margin(double t_col_margin) { col_margin = t_col_margin; }

private:
  int frame_id = -1;
  std::string frame_name;

  double col_penalty = 1e4;
  double bound_penalty = 1e4;
  double frame_penalty = 1e2;
  bool ineq_squared_penalties = true; // or LOG

  double bound_tol = 1e-4;
  double col_tol = 1e-4;
  double frame_tol = 1e-4;
  double col_margin = .001;

  int max_time_ms = 1000;
  int max_num_attempts = 1;
  int max_solutions = 1;

  int num_threads_edges = -1;
  std::string urdf_filename;
  std::string srdf_filename;
  std::string env_urdf;
  std::string robots_model_path;
  std::vector<Eigen::VectorXd> ik_solutions;

  // std::vector<FrameBounds> frame_bounds;
  Eigen::VectorXd x_lb;
  Eigen::VectorXd x_ub;
  Eigen::VectorXd p_des; // or maybe use pinocchio::SE?
  // Eigen::VectorXd q_des;
  pinocchio::SE3 pq_des;
  bool flag_desired_pq = false;

  pinocchio::Model model;
  pinocchio::Data data;
  pinocchio::GeometryModel geom_model;
  pinocchio::GeometryData geom_data;

  std::vector<pinocchio::Data> data_parallel;
  std::vector<pinocchio::GeometryData> geom_data_parallel;

  bool build_done = false;
  std::vector<hpp::fcl::CollisionObject> collision_objects;
  std::vector<std::vector<hpp::fcl::CollisionObject>>
      collision_objects_parallel;

  std::unique_ptr<BS::thread_pool> pool;

  bool use_pool = false;
  bool use_aabb = false; // what to do here?
};
} // namespace dynorrt
