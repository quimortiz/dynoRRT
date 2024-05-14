

#include "dynoRRT/pin_ik_solver.h"

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

  if (!build_done) {
    THROW_PRETTY_DYNORRT("build not done");
  }

  if (!x_lb.size() || !x_ub.size()) {
    THROW_PRETTY_DYNORRT("x_lb and x_ub not set");
  }

  if (x_lb.size() != x_ub.size()) {
    THROW_PRETTY_DYNORRT("x_lb and x_ub must have the same size");
  }
  if (frame_id == -1) {
    THROW_PRETTY_DYNORRT("frame_id not set");
  }

  IKStatus status = IKStatus::RUNNING;

  int attempts = 0;
  auto tic = std::chrono::high_resolution_clock::now();
  const int gd_it = 500;
  while (status == IKStatus::RUNNING) {

    // sample a random configuration
    Eigen::VectorXd q =
        x_lb + (x_ub - x_lb)
                       .cwiseProduct(Eigen::VectorXd::Random(x_lb.size()) +
                                     Eigen::VectorXd::Ones(x_lb.size())) /
                   2.0;

    // evaluate the cost
    double fq = get_cost(q);
    int counter_print = 1;

    double sc = get_cost(q);
    std::cout << "it: -1 " << " cost: " << sc << " q:" << q.transpose()
              << std::endl;

    for (size_t it = 0; it < gd_it; it++) {
      Eigen::VectorXd dq = Eigen::VectorXd::Zero(q.size());
      double fq = get_cost(q);
      get_cost_derivative(q, dq);
      // TODO: change this to work in SE3!!
      //
      //
      //
      // lets do backtraing line search

      const int num_ls_it = 20;
      int num_ls_it_counter = 0;

      double alpha = 0.1;
      while (true) {

        if (get_cost(q - alpha * dq) <
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

      q -= alpha * dq;

      if (it % counter_print == 0) {
        double sc = get_cost(q);
        std::cout << "it: " << it << " cost: " << sc << " q:" << q.transpose()
                  << " dq:" << dq.transpose() << std::endl;
      }
    }

    double ds = get_distance_cost(q);
    double bs = get_bounds_cost(q);
    double fs = get_frame_cost(q);

    std::cout << "At convergence: " << std::endl;
    std::cout << "cost_dist: " << ds << " cost_bounds: " << bs
              << " cost_frame: " << fs << std::endl;
    if (ds < col_tol && bs < bound_tol && fs < frame_tol) {
      std::cout << "Solution found!" << std::endl;
      ik_solutions.push_back(q);
    } else {
      std::cout << "Solution not found!" << std::endl;
    }

    attempts++;

    if (attempts >= max_num_attempts) {
      status = IKStatus::MAX_ATTEMPTS;
    }

    if (ik_solutions.size() >= max_solutions) {
      status = IKStatus::SUCCESS;
    }

    auto toc = std::chrono::high_resolution_clock::now();

    double elapsed_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic)
            .count();

    if (elapsed_time_ms > max_time_ms) {
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

  pool = std::make_unique<BS::thread_pool>(num_threads_edges);
  // 12);
}

} // namespace dynorrt
