#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include <hpp/fcl/collision_object.h>
#include <hpp/fcl/shape/geometric_shapes.h>
#include <iostream>

// PINOCCHIO_MODEL_DIR is defined by the CMake but you can define your own
// modeldirectory here.
#ifndef PINOCCHIO_MODEL_DIR
#define PINOCCHIO_MODEL_DIR                                                    \
  "/home/quim/croco/lib/python3.8/site-packages/cmeel.prefix/share/"
#endif

int main(int /*argc*/, char ** /*argv*/) {
  using namespace pinocchio;
  const std::string robots_model_path = PINOCCHIO_MODEL_DIR;

  // You should change here to set up your own URDF file
  const std::string urdf_filename =
      robots_model_path +
      std::string(
          "/example-robot-data/robots/ur_description/urdf/ur5_robot.urdf");

  // You should change here to set up your own SRDF file
  const std::string srdf_filename =
      robots_model_path +
      std::string("/example-robot-data/robots/ur_description/srdf/ur5.srdf");

  // Load the URDF model contained in urdf_filename
  Model model;
  pinocchio::urdf::buildModel(urdf_filename, model);

  // Build the data associated to the model
  Data data(model);

  // Load the geometries associated to model which are contained in the URDF
  // file
  GeometryModel geom_model;
  pinocchio::urdf::buildGeom(model, urdf_filename, pinocchio::COLLISION,
                             geom_model, robots_model_path);

  // geom = pin.GeometryObject(name, 0, hppfcl.Cylinder(radius, length),
  // placement) new_id = collision_model.addGeometryObject(geom) geom.meshColor
  // = np.array(color visual_model.addGeometryObject(geom)
  //
  // for link_id in range(robot.model.nq):
  //     collision_model.addCollisionPair(pin.CollisionPair(link_id, new_id))
  // return geom

  // model.addBodyFrame("universe_body", 0, SE3::Identity());
  // Model::FrameIndex body_id_3 = model.getBodyId("universe_body");
  // Model::JointIndex joint_parent_3 = model.frames[body_id_3].parent;
  // SE3 universe_body_placement = SE3::Random();

  double radius = 0.1;
  double length = 0.5;

  boost::shared_ptr<fcl::CollisionGeometry> cyl1(
      new fcl::Cylinder(radius, length));
  boost::shared_ptr<fcl::CollisionGeometry> cyl2(
      new fcl::Cylinder(radius, length));
  boost::shared_ptr<fcl::CollisionGeometry> cyl3(
      new fcl::Cylinder(radius, length));

  SE3 placement1 = SE3::Identity();
  placement1.rotation() = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ());
  placement1.translation() = Eigen::Vector3d(-0.5, 0.4, 0.5);

  SE3 placement2 = SE3::Identity();
  placement2.rotation() = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ());
  placement2.translation() = Eigen::Vector3d(-0.5, -0.4, 0.5);

  SE3 placement3 = SE3::Identity();
  placement3.rotation() = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ());
  placement3.translation() = Eigen::Vector3d(-0.5, 0.7, 0.5);

  Model::JointIndex idx_geom1 =
      geom_model.addGeometryObject(GeometryObject("cyl1", 0, cyl1, placement1));
  Model::JointIndex idx_geom2 =
      geom_model.addGeometryObject(GeometryObject("cyl2", 0, cyl2, placement2));
  Model::JointIndex idx_geom3 =
      geom_model.addGeometryObject(GeometryObject("cyl3", 0, cyl3, placement3));

  geom_model.geometryObjects[idx_geom1].parentJoint = 0;
  geom_model.geometryObjects[idx_geom2].parentJoint = 0;
  geom_model.geometryObjects[idx_geom3].parentJoint = 0;

  // cylID = "world/cyl1"
  // placement = pin.SE3(pin.SE3(rotate("z", np.pi / 2), np.array([-0.5, 0.4,
  // 0.5]))) addCylinderToUniverse(cylID, radius, length, placement, color=[0.7,
  // 0.7, 0.98, 1])
  //
  //
  // cylID = "world/cyl2"
  // placement = pin.SE3(pin.SE3(rotate("z", np.pi / 2), np.array([-0.5, -0.4,
  // 0.5]))) addCylinderToUniverse(cylID, radius, length, placement, color=[0.7,
  // 0.7, 0.98, 1])
  //
  // cylID = "world/cyl3"
  // placement = pin.SE3(pin.SE3(rotate("z", np.pi / 2), np.array([-0.5, 0.7,
  // 0.5]))) addCylinderToUniverse(cylID, radius, length, placement, color=[0.7,
  // 0.7, 0.98, 1])
  //
  //
  // cylID = "world/cyl4"
  // placement = pin.SE3(pin.SE3(rotate("z", np.pi / 2), np.array([-0.5, -0.7,
  // 0.5]))) addCylinderToUniverse(cylID, radius, length, placement, color=[0.7,
  // 0.7, 0.98, 1])

  // from
  // Model::JointIndex idx;
  // idx =
  // model.addJoint(model.getJointId("universe"),JointModelPlanar(),SE3::Identity(),"planar1_joint");
  // model.addJointFrame(idx);
  // model.appendBodyToJoint(idx,Inertia::Random(),SE3::Identity());
  // model.addBodyFrame("planar1_body", idx, SE3::Identity());
  //
  // idx =
  // model.addJoint(model.getJointId("universe"),JointModelPlanar(),SE3::Identity(),"planar2_joint");
  // model.addJointFrame(idx);
  // model.appendBodyToJoint(idx,Inertia::Random(),SE3::Identity());
  // model.addBodyFrame("planar2_body", idx, SE3::Identity());
  //

  // boost::shared_ptr
  //   <fcl::Box> sample(new fcl::Box(1, 1, 1));
  // Model::FrameIndex body_id_1 = model.getBodyId("planar1_body");
  // Model::JointIndex joint_parent_1 = model.frames[body_id_1].parent;
  // Model::JointIndex idx_geom1 =
  // geomModel.addGeometryObject(GeometryObject("ff1_collision_object",
  //                                                                          model.getBodyId("planar1_body"),joint_parent_1,
  //                                                                          sample,SE3::Identity(),
  //                                                                          "",
  //                                                                          Eigen::Vector3d::Ones())
  //                                                           );
  // geomModel.geometryObjects[idx_geom1].parentJoint =
  // model.frames[body_id_1].parent;
  //
  //
  // shared_ptr<fcl::Box> sample2(new fcl::Box(1, 1, 1));
  // Model::FrameIndex body_id_2 = model.getBodyId("planar2_body");
  // Model::JointIndex joint_parent_2 = model.frames[body_id_2].parent;
  // Model::JointIndex idx_geom2 =
  // geomModel.addGeometryObject(GeometryObject("ff2_collision_object",
  //                                                                          model.getBodyId("planar2_body"),joint_parent_2,
  //                                                                          sample2,SE3::Identity(),
  //                                                                          "",
  //                                                                          Eigen::Vector3d::Ones()),
  //                                                           model);
  // BOOST_CHECK(geomModel.geometryObjects[idx_geom2].parentJoint ==
  // model.frames[body_id_2].parent);

  //                                                              ,
  //                                                              "universe_collision_object",
  //                                                                            model.getBodyId("universe_body"),SE3::Identity(),
  //                                                                            cyl,SE3::Identity(), "", Eigen::Vector3d::Ones()),
  //                                                             model);
  //
  // universe_collision_object",
  //                                                                            model.getBodyId("universe_body"),joint_parent_3,
  //                                                                            universe_body_geometry,universe_body_placement, "", Eigen::Vector3d::Ones()),
  //                                                             model);

  // Add all possible collision pairs and remove the ones collected in the SRDF
  // file
  geom_model.addAllCollisionPairs();
  pinocchio::srdf::removeCollisionPairs(model, geom_model, srdf_filename);

  // Build the data associated to the geom_model
  GeometryData geom_data(
      geom_model); // contained the intermediate computations, like the
                   // placement of all the geometries with respect to the world
                   // frame

  // Load the reference configuration of the robots (this one should be
  // collision free)
  // pinocchio::srdf::loadReferenceConfigurations(model,srdf_filename); // the
  // reference configuratio stored in the SRDF file is called half_sitting

  // const Model::ConfigVectorType & q =
  // model.referenceConfigurations["half_sitting"];

  std::cout << "model: " << std::endl;
  std::cout << model;
  std::cout << "geom_model: " << std::endl;
  std::cout << geom_model;
  std::cout << "geom_data: " << std::endl;
  std::cout << geom_data;

  std::vector<Eigen::VectorXd> q_vecs;

  Eigen::VectorXd q1 = Eigen::VectorXd::Zero(model.nq);
  q1 << 1.0, -1.5, 2.1, -0.5, -0.5, 0;
  q_vecs.push_back(q1);

  Eigen::VectorXd q2 = Eigen::VectorXd::Zero(model.nq);
  q2 << 1.0, -1.5, 2.1, -0.5, -0.5, 0;
  q_vecs.push_back(q2);

  // [ 2.66364844 -1.16572247  1.39070035 -2.29374365  0.68460081 -3.09567634]
  // [ 0.15543802 -1.62053444  3.02576015 -1.32789258  1.73668927  0.17116787]
  // [-0.96708521 -2.99465177 -3.07185249 -0.27070888 -2.79618663 -1.67500829]

  Eigen::VectorXd q3 = Eigen::VectorXd::Zero(model.nq);
  q3 << 2.66364844, -1.16572247, 1.39070035, -2.29374365, 0.68460081,
      -3.09567634;
  q_vecs.push_back(q3);
  Eigen::VectorXd q4 = Eigen::VectorXd::Zero(model.nq);
  q4 << 0.15543802, -1.62053444, 3.02576015, -1.32789258, 1.73668927,
      0.17116787;
  q_vecs.push_back(q4);
  Eigen::VectorXd q5 = Eigen::VectorXd::Zero(model.nq);
  q5 << -0.96708521, -2.99465177, -3.07185249, -0.27070888, -2.79618663,
      -1.67500829;
  q_vecs.push_back(q5);

  // q_i = np.array([1.0, -1.5, 2.1, -0.5, -0.5, 0])
  // q_g = np.array([3.0, -1.0, 1, -0.5, -0.5, 0])

  // Eigen::VectorXd q = pinocchio::neutral(model);

  // And test all the collision pairs

  for (auto &q : q_vecs) {
    std::cout << "q: " << q.transpose() << std::endl;

    computeCollisions(model, data, geom_model, geom_data, q);

    // Print the status of all the collision pairs
    for (size_t k = 0; k < geom_model.collisionPairs.size(); ++k) {
      const CollisionPair &cp = geom_model.collisionPairs[k];
      const hpp::fcl::CollisionResult &cr = geom_data.collisionResults[k];

      std::cout << "collision pair: " << cp.first << " , " << cp.second
                << " - collision: ";
      std::cout << (cr.isCollision() ? "yes" : "no") << std::endl;
    }

    // If you want to stop as soon as a collision is encounter, just add false
    // for the final default argument stopAtFirstCollision

    std::cout << "summary "
              << computeCollisions(model, data, geom_model, geom_data, q, true)
              << std::endl;

    // And if you to check only one collision pair, e.g. the third one, at the
    // neutral element of the Configuration Space of the robot
    const PairIndex pair_id = 2;
    const Model::ConfigVectorType q_neutral = neutral(model);
    updateGeometryPlacements(
        model, data, geom_model, geom_data,
        q_neutral); // performs a forward kinematics over the whole kinematics
                    // model + update the placement of all the geometries
                    // contained inside geom_model
    computeCollision(geom_model, geom_data, pair_id);
  }

  return 0;
}
