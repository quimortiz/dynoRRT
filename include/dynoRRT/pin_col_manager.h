#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio/algorithm/geometry.hpp"

#include <Eigen/Dense>
#include <boost/make_shared.hpp>

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

  // one thread, check a list of configurations
  bool is_collision_free_list(const std::vector<Eigen::VectorXd> qs);

  bool is_collision_free_parallel(const Eigen::VectorXd &q, int num_threads);

  void reset_counters() {
    time_ms = 0;
    num_collision_checks = 0;
  }
  int get_num_collision_checks() { return num_collision_checks; }
  double get_time_ms() { return time_ms; }

  bool is_collision_free_set(const std::vector<Eigen::VectorXd> &q_set,
                             bool stop_at_first_collision,
                             int *counter_infeas_out, int *counter_feas_out);

  void set_frame_bounds(const std::vector<FrameBounds> &t_frame_bounds) {
    frame_bounds = t_frame_bounds;

    for (auto &fb : frame_bounds) {
      fb.frame_id = model.getFrameId(fb.frame_name);
    }
  }

  bool is_inside_frame_bounds(const Eigen::VectorXd &q);

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
};
} // namespace dynorrt
