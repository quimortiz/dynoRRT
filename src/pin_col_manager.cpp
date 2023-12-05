
#include "dynoRRT/pin_col_manager.h"

#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "dynoRRT/dynorrt_macros.h"
#include "pinocchio/algorithm/geometry.hpp"

namespace dynorrt {

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

    boost::shared_ptr<fcl::CollisionGeometry> geometry;

    if (obs.name == "cylinder") {
      CHECK_PRETTY_DYNORRT(obs.data.size() == 2, "cylinder needs 2 parameters");
      double radius = obs.data[0];
      double length = obs.data[1];
      geometry = boost::make_shared<fcl::Cylinder>(radius, length);
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
  geom_data = GeometryData(geom_model);
  build_done = true;
}

bool Collision_manager_pinocchio::is_collision_free(const Eigen::VectorXd &q) {

  num_collision_checks++;
  auto tic = std::chrono::high_resolution_clock::now();
  if (!build_done) {
    THROW_PRETTY_DYNORRT("build not done");
  }
  bool out = !computeCollisions(model, data, geom_model, geom_data, q, true);
  time_ms += std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::high_resolution_clock::now() - tic)
                 .count() /
             1000.0;
  return out;
};

} // namespace dynorrt
