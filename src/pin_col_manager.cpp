#include "dynoRRT/pin_col_manager.h"

#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio/algorithm/geometry.hpp"

#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "dynoRRT/dynorrt_macros.h"
#include "hpp/fcl/shape/geometric_shapes.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/multibody/fcl.hpp"
#include <atomic>

// #include <hpp/fcl/broadphase/broadphase.h>
#include <future>
#include <thread>

#include "dynoRRT/pin_col_manager.h"

#include "dynoRRT/dynorrt_macros.h"
#include "pinocchio/multibody/fcl.hpp"
#include <atomic>

#include <memory>
#include <thread>

namespace dynorrt {

// using namespace hpp;

template <typename T>
// using shared_ptr = boost::shared_ptr<T>;
using shared_ptr = std::shared_ptr<T>;

template <typename T, typename... Args> auto make_shared_ptr(Args &&...args) {
  // return boost::make_shared<T>(std::forward<Args>(args)...);
  return std::make_shared<T>(std::forward<Args>(args)...);
}

void Collision_manager_pinocchio::build() {

  // using namespace pinocchio;
  pinocchio::urdf::buildModel(urdf_filename, model);

  data = pinocchio::Data(model);

  pinocchio::urdf::buildGeom(model, urdf_filename, pinocchio::COLLISION,
                             geom_model, robots_model_path);

  double radius = 0.1;
  double length = 1.0;

  std::string obstacle_base_name = "external_obs_";
  int i = 0;

  for (auto &obs : obstacles) {

    shared_ptr<hpp::fcl::CollisionGeometry> geometry;

    if (obs.name == "cylinder") {
      CHECK_PRETTY_DYNORRT(obs.data.size() == 2, "cylinder needs 2 parameters");
      double radius = obs.data[0];
      double length = obs.data[1];
      geometry = make_shared_ptr<hpp::fcl::Cylinder>(radius, length);
    } else {
      THROW_PRETTY_DYNORRT("only cylinder are supported");
    }

    pinocchio::SE3 placement = pinocchio::SE3::Identity();
    CHECK_PRETTY_DYNORRT(obs.translation.size() == 3,
                         "translation needs 3 parameters");
    CHECK_PRETTY_DYNORRT(obs.rotation_angle_axis.size() == 4,
                         "rotation_angle_axis needs 4 parameters");
    placement.rotation() = Eigen::AngleAxisd(obs.rotation_angle_axis[0],
                                             obs.rotation_angle_axis.tail<3>());
    placement.translation() = obs.translation;

    pinocchio::Model::JointIndex idx_geom =
        geom_model.addGeometryObject(pinocchio::GeometryObject(
            obstacle_base_name + std::to_string(i) + "_" + obs.name, 0,
            geometry, placement));

    geom_model.geometryObjects[idx_geom].parentJoint = 0;
    i++;
  }

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

bool Collision_manager_pinocchio::is_inside_frame_bounds(
    const Eigen::VectorXd &q) {

  // TODO: Check if this is the most efficient way to do this
  pinocchio::forwardKinematics(model, data, q, Eigen::VectorXd::Zero(model.nv));
  pinocchio::updateFramePlacements(model, data);
  Eigen::Vector3d frame_pos;
  for (const auto &frame_bound : frame_bounds) {
    frame_pos = data.oMf[frame_bound.frame_id].translation();
    for (int i = 0; i < 3; i++) {
      if (frame_pos(i) < frame_bound.lower(i) ||
          frame_pos(i) > frame_bound.upper(i)) {
        return false;
      }
    }
  }
  return true;
}

void Collision_manager_pinocchio::set_pin_model(
    pinocchio::Model &t_model, pinocchio::GeometryModel &t_geomodel) {
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

    pool = std::make_unique<BS::thread_pool>(num_threads_edges);
  }

  // 12);
}

bool Collision_manager_pinocchio::is_collision_free_v2(const Eigen::VectorXd &q,
                                                       int thread_id) {

  // TODO: is inside frame bounds does not work in the parallel setting
  if (!is_inside_frame_bounds(q)) {
    return false;
  }

  // num_collision_checks++;

  if (use_aabb) {

    updateGeometryPlacements(model, data_parallel.at(thread_id), geom_model,
                             geom_data_parallel.at(thread_id), q);
    bool isColliding = false;
    bool stopAtFirstCollision = true;

    for (size_t i = 0; i < geom_model.geometryObjects.size(); i++) {
      if (!geom_model.geometryObjects[i].disableCollision) {
        collision_objects_parallel.at(thread_id).at(i).setTransform(
            toFclTransform3f(geom_data_parallel.at(thread_id).oMg[i]));
        collision_objects_parallel.at(thread_id).at(i).computeAABB();
      }
    }

    for (std::size_t cp_index = 0; cp_index < geom_model.collisionPairs.size();
         ++cp_index) {
      const pinocchio::CollisionPair &cp = geom_model.collisionPairs[cp_index];

      if (geom_data_parallel.at(thread_id).activeCollisionPairs[cp_index] &&
          !(geom_model.geometryObjects[cp.first].disableCollision ||
            geom_model.geometryObjects[cp.second].disableCollision)) {

        if (collision_objects_parallel.at(thread_id)[cp.first]
                .getAABB()
                .overlap(collision_objects_parallel.at(thread_id)[cp.second]
                             .getAABB())) {

          bool res = pinocchio::computeCollision(
              geom_model, geom_data_parallel.at(thread_id), cp_index);
          if (!isColliding && res) {
            isColliding = true;
            geom_data_parallel.at(thread_id).collisionPairIndex =
                cp_index; // first pair to be in
                          // collision
            if (stopAtFirstCollision)
              return false;
          }
        }
      }
    }
    return true;

  } else {
    // auto tic = std::chrono::high_resolution_clock::now();
    if (!build_done) {
      THROW_PRETTY_DYNORRT("build not done");
    }
    bool out = !pinocchio::computeCollisions(
        model, data_parallel.at(thread_id), geom_model,
        geom_data_parallel.at(thread_id), q, true);

    // time_ms += std::chrono::duration_cast<std::chrono::microseconds>(
    //                std::chrono::high_resolution_clock::now() - tic)
    //                .count() /
    //            1000.0;

    return out;
  }
};

bool Collision_manager_pinocchio::is_collision_free(const Eigen::VectorXd &q) {

  if (!build_done) {
    THROW_PRETTY_DYNORRT("build not done");
  }

  if (!is_inside_frame_bounds(q)) {
    return false;
  }

  num_collision_checks++;

  if (use_aabb) {

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

          bool res =
              pinocchio::computeCollision(geom_model, geom_data, cp_index);
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
    // auto tic = std::chrono::high_resolution_clock::now();
    if (!build_done) {
      THROW_PRETTY_DYNORRT("build not done");
    }
    bool out = !pinocchio::computeCollisions(model, data, geom_model, geom_data,
                                             q, true);

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

  if (!build_done) {
    THROW_PRETTY_DYNORRT("build not done");
  }

  std::atomic_bool infeasible_found = false;
  if (this->num_threads_edges > 1) {
    // std::cout << "num_threads_edges: " << num_threads_edges << std::endl;
    // std::cout << "use pool" << use_pool << std::endl;

    int checks_per_thread = int(q_set.size() / num_threads_edges) + 1;

    if (num_threads_edges > q_set.size()) {
      checks_per_thread = 1;
    }

    std::atomic_int counter_infeas = 0;
    std::atomic_int counter_feas = 0;

    if (!use_pool) {
      std::vector<std::thread> threads(num_threads_edges);

      for (size_t j = 0; j < num_threads_edges; j++) {
        auto fun = [&](int thread_id) {
          for (int i = thread_id; i < q_set.size(); i += num_threads_edges) {

            if (stop_at_first_collision && infeasible_found) {
              return;
            } else {
              num_collision_checks++;
              if (!is_collision_free_v2(q_set[i], thread_id)) {
                counter_infeas++;
                infeasible_found = true;
                if (stop_at_first_collision)
                  break;
              } else {
                counter_feas++;
              }
            }
          }
        };
        threads[j] = std::thread(fun, j);
      };

      for (auto &t : threads) {
        t.join();
      }
    }

    else {
      for (size_t i = 0; i < q_set.size(); i++) {
        pool->detach_task([&, i] {
          if (infeasible_found && stop_at_first_collision) {
          } else {
            auto _thread_id = BS::this_thread::get_index();
            int thread_id = _thread_id.value();
            if (!is_collision_free_v2(q_set[i], thread_id)) {
              counter_infeas++;
              infeasible_found = true;
            } else {
              counter_feas++;
            }
          }
        });
      };

      pool->wait();
    }

    if (counter_infeas_out)
      *counter_infeas_out = counter_infeas.load();
    if (counter_feas_out)
      *counter_feas_out = counter_feas.load();

    // return true;
    return !infeasible_found;
  }

  else {
    bool edge_col_free = true;
    for (auto &q : q_set) {

      bool col_free = is_collision_free(q);
      if (col_free) {
        if (counter_feas_out)
          (*counter_feas_out)++;

      } else {
        edge_col_free = false;
        if (counter_infeas_out)
          (*counter_infeas_out)++;
      }

      if (stop_at_first_collision && !col_free) {
        break;
      }
    }
    return edge_col_free;
  }
}

// bool Collision_manager_pinocchio::is_collision_free_parallel(
//     const Eigen::VectorXd &q, int num_threads) {
//
//   num_collision_checks++;
//   auto tic = std::chrono::high_resolution_clock::now();
//   if (!build_done) {
//     THROW_PRETTY_DYNORRT("build not done");
//   }
//   bool out = !pinocchio::computeCollisions(num_threads, model, data,
//   geom_model,
//                                            geom_data, q, true);
//
//   time_ms += std::chrono::duration_cast<std::chrono::microseconds>(
//                  std::chrono::high_resolution_clock::now() - tic)
//                  .count() /
//              1000.0;
//   return out;
// };

void Collision_manager_pinocchio::set_pin_model0(pinocchio::Model &t_model) {
  this->model = t_model;
}

void Collision_manager_pinocchio::set_pin_geomodel0(
    pinocchio::GeometryModel &t_geomodel) {
  this->geom_model = t_geomodel;
}

} // namespace dynorrt
