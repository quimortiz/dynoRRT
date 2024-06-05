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

enum class OPTStatus {
  RUNNING,
  SUCCESS,
  UNKNOWN,
  MAX_IT,
  STEP_TOL,
  STEP_TOL_MAXREG,
  GRAD_TOL
};

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

  void set_frame_positions(const std::vector<Eigen::VectorXd> &t_p_des) {
    frame_positions = t_p_des;
  }

  void set_frame_poses(const std::vector<pinocchio::SE3> &t_pq_des) {
    frame_poses = t_pq_des;
    flag_desired_pq = true;
  }

  void set_max_num_attempts(int num) { max_num_attempts = num; }

  void set_max_solutions(int num) { max_solutions = num; }

  void set_max_time_ms(int num) { max_time_ms = num; }

  auto get_model_data_ptr() { return std::make_tuple(&model, &data); }

  double get_bounds_cost(const Eigen::VectorXd &q,
                         Eigen::VectorXd *grad = nullptr,
                         Eigen::MatrixXd *H = nullptr);

  double get_distance_cost(const Eigen::VectorXd &q,
                           Eigen::VectorXd *grad = nullptr,
                           Eigen::MatrixXd *H = nullptr);

  double get_frame_cost(const Eigen::VectorXd &q,
                        Eigen::VectorXd *grad = nullptr,
                        Eigen::MatrixXd *H = nullptr);

  double get_cost(const Eigen::VectorXd &q, Eigen::VectorXd *grad = nullptr,
                  Eigen::MatrixXd *H = nullptr);

  void get_cost_derivative(const Eigen::VectorXd &q, Eigen::VectorXd &dq);

  IKStatus solve_ik();

  std::vector<Eigen::VectorXd> get_ik_solutions() { return ik_solutions; }

  void set_tolerances(double t_bound_tol, double t_col_tol,
                      double t_frame_tol) {
    bound_tol = t_bound_tol;
    col_tol = t_col_tol;
    frame_tol = t_frame_tol;
  }

  void set_frame_ids(const std::vector<int> &id) { frame_ids = id; }

  void set_frame_names(const std::vector<std::string> &t_name) {
    frame_ids.clear();

    for (auto &t_name : t_name) {
      if (!model.existFrame(t_name)) {
        THROW_PRETTY_DYNORRT("frame not found " + t_name);
      }
      frame_ids.push_back(model.getFrameId(t_name));
    }
  }

  void set_col_margin(double t_col_margin) { col_margin = t_col_margin; }
  void set_max_it(int t_max_it) { max_it = t_max_it; }
  void set_use_gradient_descent(bool use) { use_gradient_descent = use; }
  void set_use_finite_diff(bool fd) { use_finite_diff = fd; }

  double get_joint_reg_cost(const Eigen::VectorXd &q,
                            Eigen::VectorXd *grad = nullptr,
                            Eigen::MatrixXd *H = nullptr);

  void set_joint_reg_penalty(double t_joint_reg_penalty) {
    joint_reg_penalty = t_joint_reg_penalty;
  }
  void set_joint_reg(const Eigen::VectorXd &t_joint_reg) {
    joint_reg = t_joint_reg;
  }

private:
  std::vector<int> frame_ids;
  std::vector<std::string> frame_names;
  std::vector<pinocchio::SE3> frame_poses;
  std::vector<Eigen::VectorXd> frame_positions;
  bool flag_desired_pq = false;

  // int frame_id = -1;

  double joint_reg_penalty = 0.;
  double col_penalty = 1e4;
  double bound_penalty = 1e4;
  double frame_penalty = 1e2;
  bool ineq_squared_penalties = true; // or LOG

  double bound_tol = 1e-4;
  double col_tol = 1e-4;
  double frame_tol = 1e-4;
  double col_margin = .001;
  double max_it = 100; // max iterations for the solver

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
  Eigen::VectorXd joint_reg;

  std::unique_ptr<BS::thread_pool> pool;

  bool use_finite_diff = true;
  bool use_gradient_descent = false;
  bool use_pool = false;
  bool use_aabb = false; // what to do here?
};
} // namespace dynorrt
