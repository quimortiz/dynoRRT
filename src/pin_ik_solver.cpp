

#include "dynoRRT/pin_ik_solver.h"
#include "magic_enum.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/collision/collision.hpp"
#include "pinocchio/multibody/geometry.hpp"
#include "pinocchio/multibody/sample-models.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/utils/timer.hpp"

#include <iostream>

namespace dynorrt {

void Pin_ik_solver::build() {

  pinocchio::urdf::buildModel(urdf_filename, model);

  data = pinocchio::Data(model);

  pinocchio::urdf::buildGeom(model, urdf_filename, pinocchio::COLLISION,
                             geom_model, robots_model_path);

  geom_model.addAllCollisionPairs();
  if (srdf_filename != "") {
    pinocchio::srdf::removeCollisionPairs(model, geom_model, srdf_filename);
  }

  for (auto &go : geom_model.geometryObjects) {
    go.geometry->computeLocalAABB();
  }

  collision_objects.reserve(geom_model.geometryObjects.size());
  for (size_t i = 0; i < geom_model.geometryObjects.size(); i++) {
    collision_objects.emplace_back(
        hpp::fcl::CollisionObject(geom_model.geometryObjects[i].geometry));
  }

  geom_data = pinocchio::GeometryData(geom_model);
  build_done = true;

  if (num_threads_edges > 0) {
    THROW_PRETTY_DYNORRT("IK parallel -- not implemented");

    for (size_t i = 0; i < num_threads_edges; i++) {
      data_parallel.push_back(pinocchio::Data(model));
      geom_data_parallel.push_back(pinocchio::GeometryData(geom_model));

      std::vector<hpp::fcl::CollisionObject> _collision_objects;
      _collision_objects.reserve(geom_model.geometryObjects.size());
      for (size_t j = 0; j < geom_model.geometryObjects.size(); j++) {
        _collision_objects.emplace_back(
            hpp::fcl::CollisionObject(geom_model.geometryObjects[j].geometry));
      }
      collision_objects_parallel.push_back(_collision_objects);
    }

    pool = std::make_unique<BS::thread_pool>(num_threads_edges);
  }
}

IKStatus Pin_ik_solver::solve_ik() {

  ik_solutions.clear();

  if (!build_done) {
    THROW_PRETTY_DYNORRT("build not done");
  }

  if (!x_lb.size() || !x_ub.size()) {
    THROW_PRETTY_DYNORRT("x_lb and x_ub not set");
  }

  if (x_lb.size() != x_ub.size()) {
    THROW_PRETTY_DYNORRT("x_lb and x_ub must have the same size");
  }
  if (!frame_ids.size()) {
    THROW_PRETTY_DYNORRT("frame_ids not set");
  }

  if (flag_desired_pq) {
    if (frame_poses.size() != frame_ids.size()) {
      THROW_PRETTY_DYNORRT("frame_poses and frame_ids must have the same size");
    }
  } else {
    if (frame_positions.size() != frame_ids.size()) {
      THROW_PRETTY_DYNORRT(
          "frame_positions and frame_ids must have the same size");
    }
  }

  IKStatus status = IKStatus::RUNNING;

  int attempts = 0;
  auto tic = std::chrono::high_resolution_clock::now();
  const int gd_it = 500;
  Eigen::VectorXd one_v = Eigen::VectorXd::Ones(x_lb.size());
  Eigen::VectorXd r = Eigen::VectorXd::Random(x_lb.size());

  Eigen::VectorXd grad = Eigen::VectorXd::Zero(x_lb.size());
  Eigen::VectorXd dq = Eigen::VectorXd::Zero(x_lb.size());

  Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(x_lb.size(), x_lb.size());
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(x_lb.size(), x_lb.size());

  const double max_alpha_gd = .01;
  const int num_ls_it = 20;

  double reg_factor = 1e-4;

  const double min_reg_factor = 1e-6;
  const double max_ref_factor = 1e2;
  const double grad_tol = 1e-6;
  const double step_tol = 1e-6;
  const double alpha_too_small = .1;
  const double alpha_success = .8;

  double fq;

  bool verbose = false;

  while (status == IKStatus::RUNNING) {

    r = Eigen::VectorXd::Random(x_lb.size()); // [-1, 1]
    Eigen::VectorXd q = x_lb + (x_ub - x_lb).cwiseProduct(one_v + r) / 2.;

    double sc = get_cost(q);

    if (verbose)
      std::cout << "it: -1 " << " cost: " << sc << " q:" << q.transpose()
                << std::endl;

    bool converged = false;
    int it = 0;
    int counter_print = 1;
    OPTStatus opt_status = OPTStatus::RUNNING;
    while (opt_status == OPTStatus::RUNNING) {
      if (use_gradient_descent) {
        grad.setZero();

        if (use_finite_diff) {
          fq = get_cost(q);
          finite_diff_grad(
              q, [&](const auto &__q) { return get_cost(__q); }, grad);
        } else {
          fq = get_cost(q, &grad);
        }

        dq = -max_alpha_gd * grad;
      } else {
        grad.setZero();
        H.setZero();

        if (use_finite_diff) {
          fq = get_cost(q);
          finite_diff_grad(
              q, [&](const auto &__q) { return get_cost(__q); }, grad);
          finite_diff_hess(
              q, [&](const auto &__q) { return get_cost(__q); }, H);
        } else {
          fq = get_cost(q, &grad, &H);
        }

        H += reg_factor * Id;
        dq = H.ldlt().solve(-grad);
        if (dq.dot(grad) > 0) {
          if (verbose)
            std::cout << "Not a descent direction -- using gradient instead"
                      << std::endl;
          reg_factor *= 10; // I increase the regularization for next iteration
          dq = -max_alpha_gd * grad;
        }
      }

      // line search
      int num_ls_it_counter = 0;
      double alpha = 1.;
      while (true) {

        if (get_cost(q + alpha * dq) <
            fq - 0.001 * alpha * dq.norm() * dq.norm()) {
          break;
        } else {
          alpha *= 0.5;
        }
        num_ls_it_counter++;
        if (num_ls_it_counter > num_ls_it) {
          break;
        }
      }

      q += alpha * dq;

      // TODO: Increase the regularization if the step is too small and I am
      // using second order.

      if (it % counter_print == 0) {
        double sc = get_cost(q);
        if (verbose)
          std::cout << "it: " << it << " cost: " << sc << " alpha: " << alpha
                    << "||a*dq||" << alpha * dq.norm() << " q:" << q.transpose()
                    << " dq:" << dq.transpose() << std::endl;
      }

      it++;

      if (it >= max_it) {
        opt_status = OPTStatus::MAX_IT;
      }

      if (grad.norm() < grad_tol) {
        opt_status = OPTStatus::GRAD_TOL;
      }

      if (use_gradient_descent) {
        if (alpha * dq.norm() < step_tol) {
          opt_status = OPTStatus::STEP_TOL;
        }
      } else {

        if (reg_factor > max_ref_factor - 1e-6 &&
            alpha * dq.norm() < step_tol) {
          opt_status = OPTStatus::STEP_TOL_MAXREG;
        }

        if (alpha > alpha_success) {
          reg_factor /= 10.;
          reg_factor = std::max(reg_factor, min_reg_factor);
        } else if (alpha < alpha_too_small) {
          reg_factor *= 10.;
          reg_factor = std::min(reg_factor, max_ref_factor);
        }
      }
    }
    if (verbose)
      std::cout << "Terminate Status: " << magic_enum::enum_name(opt_status)
                << std::endl;

    double ds = get_distance_cost(q);
    double bs = get_bounds_cost(q);
    double fs = get_frame_cost(q);

    if (verbose) {
      std::cout << "At convergence: " << std::endl;
      std::cout << "cost_dist: " << ds << " cost_bounds: " << bs
                << " cost_frame: " << fs << std::endl;
    }
    if (ds < col_tol && bs < bound_tol && fs < frame_tol) {
      if (verbose)
        std::cout << "Solution found!" << std::endl;
      ik_solutions.push_back(q);
    } else {
      if (verbose)
        std::cout << "Solution not found!" << std::endl;
    }

    attempts++;

    auto toc = std::chrono::high_resolution_clock::now();

    double elapsed_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic)
            .count();

    if (ik_solutions.size() >= max_solutions) {
      status = IKStatus::SUCCESS;
    } else if (attempts >= max_num_attempts) {
      status = IKStatus::MAX_ATTEMPTS;
    } else if (elapsed_time_ms > max_time_ms) {
      status = IKStatus::MAX_TIME;
    }
  }
  return status;
}

void Pin_ik_solver::set_pin_model(pinocchio::Model &t_model,
                                  pinocchio::GeometryModel &t_geomodel) {

  model = t_model;
  geom_model = t_geomodel;
  data = pinocchio::Data(model);
  // geom_data = pinocchio::GeometryData(geom_model);

  // geom_model.addAllCollisionPairs();
  // if (srdf_filename != "") {
  //   pinocchio::srdf::removeCollisionPairs(model, geom_model,
  //   srdf_filename);
  // }

  for (auto &go : geom_model.geometryObjects) {
    go.geometry->computeLocalAABB();
  }

  collision_objects.reserve(geom_model.geometryObjects.size());
  for (size_t i = 0; i < geom_model.geometryObjects.size(); i++) {
    collision_objects.emplace_back(
        hpp::fcl::CollisionObject(geom_model.geometryObjects[i].geometry));
  }

  geom_data = pinocchio::GeometryData(geom_model);
  build_done = true;

  if (num_threads_edges > 0) {

    for (size_t i = 0; i < num_threads_edges; i++) {
      data_parallel.push_back(pinocchio::Data(model));
      geom_data_parallel.push_back(pinocchio::GeometryData(geom_model));

      std::vector<hpp::fcl::CollisionObject> _collision_objects;
      _collision_objects.reserve(geom_model.geometryObjects.size());
      for (size_t j = 0; j < geom_model.geometryObjects.size(); j++) {
        _collision_objects.emplace_back(
            hpp::fcl::CollisionObject(geom_model.geometryObjects[j].geometry));
      }
      collision_objects_parallel.push_back(_collision_objects);
    }
    // collision_objects_parallel.resize(num_threads_edges);
  }

  if (num_threads_edges > 0) {
    THROW_PRETTY_DYNORRT("IK parallel -- not implemented");
  }
}

double Pin_ik_solver::get_distance_cost(const Eigen::VectorXd &q,
                                        Eigen::VectorXd *grad,
                                        Eigen::MatrixXd *H) {

  double cost_dist = 0;
  // Collisions
  size_t min_index = computeDistances(model, data, geom_model, geom_data, q);

  double min_dist =
      geom_data.distanceResults[min_index].min_distance - col_margin;

  // std::cout << "min_dist: " << min_dist << std::endl;

  // lets assume this is a signed distance function!!
  if (min_dist < 0) {
    cost_dist = .5 * min_dist * min_dist;

    if (grad) {

      // we compute the gradient using finite differences

      Eigen::VectorXd qplus = q;
      double eps = 1e-4;
      Eigen::VectorXd grad_distance = Eigen::VectorXd::Zero(q.size());
      for (size_t j = 0; j < q.size(); j++) {
        qplus = q;
        qplus[j] += eps;

        int _min_index =
            computeDistances(model, data, geom_model, geom_data, qplus);

        double _min_dist =
            geom_data.distanceResults[_min_index].min_distance - col_margin;

        grad_distance[j] = (_min_dist - min_dist) / eps; // Missing a -1 here?
      }

      (*grad) += min_dist * grad_distance;

      if (H) {
        *H += grad_distance * grad_distance.transpose();
      }
    }

  } else {
    cost_dist = 0;
  }
  return cost_dist;
}

double Pin_ik_solver::get_joint_reg_cost(const Eigen::VectorXd &q,
                                         Eigen::VectorXd *grad,
                                         Eigen::MatrixXd *H) {

  if (!joint_reg.size()) {
    THROW_PRETTY_DYNORRT("joint_reg not set");
  }

  double cost_joints = 0;
  cost_joints = .5 * (q - joint_reg).squaredNorm();
  if (grad) {
    (*grad) += (q - joint_reg);
    if (H) {
      *H += Eigen::MatrixXd::Identity(q.size(), q.size());
    }
  }
  return cost_joints;
}

double Pin_ik_solver::get_bounds_cost(const Eigen::VectorXd &q,
                                      Eigen::VectorXd *grad,
                                      Eigen::MatrixXd *H) {

  double cost_bounds = 0;

  for (size_t i = 0; i < q.size(); i++) {
    if (q[i] < x_lb[i]) {
      cost_bounds += .5 * (x_lb[i] - q[i]) * (x_lb[i] - q[i]);

      if (grad) {
        grad->operator()(i) += (x_lb[i] - q[i]) * -1;

        if (H) {
          H->operator()(i, i) += 1;
        }
      }

    } else if (q[i] > x_ub[i]) {
      cost_bounds += .5 * (q[i] - x_ub[i]) * (q[i] - x_ub[i]);

      if (grad) {
        grad->operator()(i) += (q[i] - x_ub[i]) * 1;

        if (H) {
          H->operator()(i, i) += 1;
        }
      }
    }
  }
  return cost_bounds;
}

double Pin_ik_solver::get_frame_cost(const Eigen::VectorXd &q,
                                     Eigen::VectorXd *grad,
                                     Eigen::MatrixXd *H) {
  // REFERENCE:
  // https://scaron.info/robotics/jacobian-of-a-kinematic-task-and-derivatives-on-manifolds.html

  pinocchio::framesForwardKinematics(model, data, q);

  double cost_frame = 0;

  for (size_t frame_j = 0; frame_j < frame_ids.size(); frame_j++) {

    int frame_id = frame_ids[frame_j];

    if (flag_desired_pq) {
      auto &pq_des = frame_poses[frame_j];
      const pinocchio::SE3 iMd = data.oMf[frame_id].actInv(pq_des);
      Eigen::VectorXd err = pinocchio::log6(iMd).toVector(); // in joint frame
      cost_frame += .5 * err.squaredNorm();

      if (grad) {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, model.nv);
        pinocchio::computeFrameJacobian(model, data, q, frame_id,
                                        pinocchio::LOCAL, J);

        pinocchio::Data::Matrix6 Jlog;
        pinocchio::Jlog6(iMd.inverse(), Jlog);

        J = -Jlog * J;
        *grad += J.transpose() * err;

        if (H) {
          *H += J.transpose() * J;
        }
      }

    } else {
      auto &p_des = frame_positions[frame_j];
      Eigen::VectorXd tool_nu = data.oMf[frame_id].translation() - p_des;
      cost_frame += .5 * tool_nu.squaredNorm();

      if (grad) {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, model.nv);
        Eigen::MatrixXd J3 = Eigen::MatrixXd::Zero(3, model.nv);
        pinocchio::computeFrameJacobian(model, data, q, frame_id,
                                        pinocchio::LOCAL_WORLD_ALIGNED, J);
        J3 = J.topRows(3);
        *grad += J3.transpose() * tool_nu;

        if (H) {
          *H += J3.transpose() * J3;
        }
      }
    }
  }

  return cost_frame;
}

double Pin_ik_solver::get_cost(const Eigen::VectorXd &q, Eigen::VectorXd *grad,
                               Eigen::MatrixXd *H) {

  double cost = 0;
  double eps = 1e-6;

  Eigen::VectorXd *grad_local = nullptr;
  Eigen::MatrixXd *H_local = nullptr;

  if (grad) {
    grad_local = new Eigen::VectorXd;
    grad_local->resize(q.size());
    grad_local->setZero();

    if (H) {
      H_local = new Eigen::MatrixXd;
      H_local->resize(q.size(), q.size());
      H_local->setZero();
    }
  }

  if (bound_penalty > eps) {

    if (grad_local)
      grad_local->setZero();
    if (H_local)
      H_local->setZero();

    cost += bound_penalty * get_bounds_cost(q, grad_local, H_local);

    if (grad) {
      grad->operator+=(bound_penalty * (*grad_local));
      if (H) {
        H->operator+=(bound_penalty * (*H_local));
      }
    }
  }

  if (col_penalty > eps) {

    if (grad_local)
      grad_local->setZero();
    if (H_local)
      H_local->setZero();

    cost += col_penalty * get_distance_cost(q, grad_local, H_local);

    if (grad) {
      grad->operator+=(col_penalty * (*grad_local));
      if (H) {
        H->operator+=(col_penalty * (*H_local));
      }
    }
  }

  if (frame_penalty > eps) {

    if (grad_local)
      grad_local->setZero();
    if (H_local)
      H_local->setZero();

    cost += frame_penalty * get_frame_cost(q, grad_local, H_local);
    if (grad) {
      grad->operator+=(frame_penalty * (*grad_local));
      if (H) {
        H->operator+=(frame_penalty * (*H_local));
      }
    }
  }

  if (joint_reg_penalty > eps) {

    if (grad_local)
      grad_local->setZero();
    if (H_local)
      H_local->setZero();

    cost += joint_reg_penalty * get_joint_reg_cost(q, grad_local, H_local);
    if (grad) {
      grad->operator+=(joint_reg_penalty * (*grad_local));
      if (H) {
        H->operator+=(joint_reg_penalty * (*H_local));
      }
    }
  }

  if (grad) {
    delete grad_local;
    if (H) {
      delete H_local;
    }
  }

  return cost;
}

void Pin_ik_solver::get_cost_derivative(const Eigen::VectorXd &q,
                                        Eigen::VectorXd &dq) {
  double eps = 1e-5;
  double fq = get_cost(q);
  Eigen::VectorXd qplus = q;

  for (size_t i = 0; i < q.size(); i++) {
    qplus = q;
    qplus[i] += eps;
    dq[i] = (get_cost(qplus) - fq) / eps;
  }
}

} // namespace dynorrt
