
#include "dynoRRT/pin_col_manager.h"

#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "dynoRRT/dynorrt_macros.h"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/algorithm/parallel/geometry.hpp"
#include "pinocchio/multibody/fcl.hpp"

// #include <hpp/fcl/broadphase/broadphase.h>
#include <thread>

namespace dynorrt {

template <typename T>
// using shared_ptr = boost::shared_ptr<T>;
using shared_ptr = std::shared_ptr<T>;

template <typename T, typename... Args> auto make_shared_ptr(Args &&...args) {
  // return boost::make_shared<T>(std::forward<Args>(args)...);
  return std::make_shared<T>(std::forward<Args>(args)...);
}

void Collision_manager_pinocchio::build() {

  using namespace pinocchio;
  pinocchio::urdf::buildModel(urdf_filename, model);

  data = pinocchio::Data(model);

  pinocchio::urdf::buildGeom(model, urdf_filename, pinocchio::COLLISION,
                             geom_model, robots_model_path);

  double radius = 0.1;
  double length = 1.0;

  std::string obstacle_base_name = "external_obs_";
  int i = 0;

  for (auto &obs : obstacles) {

    shared_ptr<fcl::CollisionGeometry> geometry;

    if (obs.name == "cylinder") {
      CHECK_PRETTY_DYNORRT(obs.data.size() == 2, "cylinder needs 2 parameters");
      double radius = obs.data[0];
      double length = obs.data[1];
      geometry = make_shared_ptr<fcl::Cylinder>(radius, length);
    } else {
      THROW_PRETTY_DYNORRT("only cylinder are supported");
    }

    SE3 placement = SE3::Identity();
    CHECK_PRETTY_DYNORRT(obs.translation.size() == 3,
                         "translation needs 3 parameters");
    CHECK_PRETTY_DYNORRT(obs.rotation_angle_axis.size() == 4,
                         "rotation_angle_axis needs 4 parameters");
    placement.rotation() = Eigen::AngleAxisd(obs.rotation_angle_axis[0],
                                             obs.rotation_angle_axis.tail<3>());
    placement.translation() = obs.translation;

    Model::JointIndex idx_geom = geom_model.addGeometryObject(
        GeometryObject(obstacle_base_name + std::to_string(i) + "_" + obs.name,
                       0, geometry, placement));

    geom_model.geometryObjects[idx_geom].parentJoint = 0;
    i++;
  }

  geom_model.addAllCollisionPairs();
  pinocchio::srdf::removeCollisionPairs(model, geom_model, srdf_filename);
  for (auto &go : geom_model.geometryObjects) {
    go.geometry->computeLocalAABB();
  }

  collision_objects.reserve(geom_model.geometryObjects.size());
  for (size_t i = 0; i < geom_model.geometryObjects.size(); i++) {
    collision_objects.emplace_back(
        hpp::fcl::CollisionObject(geom_model.geometryObjects[i].geometry));
  }

  geom_data = GeometryData(geom_model);
  build_done = true;

  if (num_threads_edges > 0) {

    for (size_t i = 0; i < num_threads_edges; i++) {
      data_parallel.push_back(pinocchio::Data(model));
      geom_data_parallel.push_back(pinocchio::GeometryData(geom_model));
    }
  }
}

bool Collision_manager_pinocchio::is_collision_free(const Eigen::VectorXd &q) {

  const bool use_aabb = true;
  if (use_aabb) {
    // continue here!!

    updateGeometryPlacements(model, data, geom_model, geom_data, q);
    bool isColliding = false;
    bool stopAtFirstCollision = true;

    for (size_t i = 0; i < geom_model.geometryObjects.size(); i++) {
      if (!geom_model.geometryObjects[i].disableCollision) {
        collision_objects.at(i).setTransform(
            toFclTransform3f(geom_data.oMg[i]));
        collision_objects.at(i).computeAABB();
      }
    }

    for (std::size_t cp_index = 0; cp_index < geom_model.collisionPairs.size();
         ++cp_index) {
      const pinocchio::CollisionPair &cp = geom_model.collisionPairs[cp_index];

      if (geom_data.activeCollisionPairs[cp_index] &&
          !(geom_model.geometryObjects[cp.first].disableCollision ||
            geom_model.geometryObjects[cp.second].disableCollision)) {

        if (collision_objects[cp.first].getAABB().overlap(
                collision_objects[cp.second].getAABB())) {

          bool res = computeCollision(geom_model, geom_data, cp_index);
          if (!isColliding && res) {
            isColliding = true;
            geom_data.collisionPairIndex = cp_index; // first pair to be in
                                                     // collision
            if (stopAtFirstCollision)
              return false;
          }
        }
      }
    }
    return true;

  } else {
    num_collision_checks++;
    // auto tic = std::chrono::high_resolution_clock::now();
    if (!build_done) {
      THROW_PRETTY_DYNORRT("build not done");
    }
    bool out = !computeCollisions(model, data, geom_model, geom_data, q, true);

    // time_ms += std::chrono::duration_cast<std::chrono::microseconds>(
    //                std::chrono::high_resolution_clock::now() - tic)
    //                .count() /
    //            1000.0;

    return out;
  }
};

bool Collision_manager_pinocchio::is_collision_free_set(
    const std::vector<Eigen::VectorXd> &q_set, bool stop_at_first_collision,
    int *counter_infeas_out, int *counter_feas_out) {

  num_collision_checks++;
  if (!build_done) {
    THROW_PRETTY_DYNORRT("build not done");
  }

  int checks_per_thread = int(q_set.size() / num_threads_edges) + 1;

  if (num_threads_edges > q_set.size()) {
    checks_per_thread = 1;
  }

  std::vector<std::thread> threads;
  std::atomic_bool infeasible_found = false;
  std::atomic_int counter_infeas = 0;
  std::atomic_int counter_feas = 0;

  for (size_t j = 0; j < num_threads_edges; j++) {

    auto fun = [&](int thread_id) {
      for (int i = thread_id; i < q_set.size(); i += num_threads_edges) {

        if (stop_at_first_collision && infeasible_found) {
          return;
        } else {
          if (computeCollisions(model, data_parallel[thread_id], geom_model,
                                geom_data_parallel[thread_id], q_set[i],
                                true)) {
            counter_infeas++;
            infeasible_found = true;
          } else {
            counter_feas++;
          }
        }
      }
    };
    threads.push_back(std::thread(fun, j));
  };

  for (auto &t : threads) {
    t.join();
  }
  *counter_infeas_out = counter_infeas.load();
  *counter_feas_out = counter_feas.load();

  return !infeasible_found;
}

bool Collision_manager_pinocchio::is_collision_free_parallel(
    const Eigen::VectorXd &q, int num_threads) {

  num_collision_checks++;
  auto tic = std::chrono::high_resolution_clock::now();
  if (!build_done) {
    THROW_PRETTY_DYNORRT("build not done");
  }
  bool out = !computeCollisions(num_threads, model, data, geom_model, geom_data,
                                q, true);

  time_ms += std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::high_resolution_clock::now() - tic)
                 .count() /
             1000.0;
  return out;
};

} // namespace dynorrt
