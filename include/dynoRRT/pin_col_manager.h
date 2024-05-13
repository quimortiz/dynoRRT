#pragma once
//
#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"
//

#include "pinocchio/algorithm/geometry.hpp"
#include <Eigen/Dense>

#include <hpp/fcl/collision_object.h>

//
//
// #include "BS_thread_pool.hpp"

#include "BS_thread_pool.hpp"

#include "pinocchio/multibody/fwd.hpp"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>

#include <hpp/fcl/collision_object.h>
#include <string>
#include <vector>

namespace dynorrt {

struct PinExternalObstacle {

  std::string shape;
  std::string name;
  Eigen::VectorXd data;
  Eigen::VectorXd translation;
  Eigen::VectorXd rotation_angle_axis;
};

struct FrameBounds {
  std::string frame_name;
  Eigen::Vector3d lower;
  Eigen::Vector3d upper;
  int frame_id = -1;
};

class Collision_manager_pinocchio {

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

  void add_external_obstacle(const PinExternalObstacle &obs) {
    obstacles.push_back(obs);
  }
  void set_edge_parallel(int num_threads) { num_threads_edges = num_threads; }

  void build();

  bool is_collision_free(const Eigen::VectorXd &q);

  bool is_collision_free_set(const std::vector<Eigen::VectorXd> &q_set,
                             bool stop_at_first_collision = true,
                             int *counter_infeas_out = nullptr,
                             int *counter_feas_out = nullptr);

  void reset_counters() {
    time_ms = 0;
    num_collision_checks = 0;
  }
  int get_num_collision_checks() { return num_collision_checks; }
  double get_time_ms() { return time_ms; }

  void set_frame_bounds(const std::vector<FrameBounds> &t_frame_bounds) {
    frame_bounds = t_frame_bounds;

    for (auto &fb : frame_bounds) {
      fb.frame_id = model.getFrameId(fb.frame_name);
    }
  }

  bool is_inside_frame_bounds(const Eigen::VectorXd &q);

  bool is_collision_free_v2(const Eigen::VectorXd &q, int thread_id);

  void set_pin_model(pinocchio::Model &t_model,
                     pinocchio::GeometryModel &t_geomodel);

  void set_pin_model0(pinocchio::Model &t_model);
  void set_pin_geomodel0(pinocchio::GeometryModel &t_geomodel);
  void set_use_pool(bool use) { use_pool = use; }


  void set_use_aabb(bool use) { use_aabb = use; }

private:
  int num_threads_edges = -1;
  std::string urdf_filename;
  std::string srdf_filename;
  std::string env_urdf;
  std::string robots_model_path;

  std::vector<FrameBounds> frame_bounds;

  std::vector<PinExternalObstacle> obstacles;
  pinocchio::Model model;
  pinocchio::Data data;
  pinocchio::GeometryData geom_data;

  std::vector<pinocchio::Data> data_parallel;
  std::vector<pinocchio::GeometryData> geom_data_parallel;

  bool build_done = false;
  pinocchio::GeometryModel geom_model;
  double time_ms = 0;
  int num_collision_checks = 0;
  bool aabb_compute_done = false;
  std::vector<hpp::fcl::CollisionObject> collision_objects;
  std::vector<std::vector<hpp::fcl::CollisionObject>>
      collision_objects_parallel;

  std::unique_ptr<BS::thread_pool> pool;

  bool use_pool = false;
  bool use_aabb = true;
};
} // namespace dynorrt
