from __future__ import print_function
import pinocchio as pin, hppfcl
import time

import os
from os.path import dirname, join, abspath
from pinocchio.visualize import MeshcatVisualizer
import sys


pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")

pinocchio_model_dir = "/home/quim/croco/lib/python3.8/site-packages/cmeel.prefix/share/"


model_path = join(pinocchio_model_dir, "example-robot-data/robots")
mesh_dir = pinocchio_model_dir
urdf_filename = "romeo_small.urdf"
urdf_model_path = join(join(model_path, "romeo_description/urdf"), urdf_filename)

# Load model
# model = pin.buildModelFromUrdf(urdf_model_path,pin.JointModelFreeFlyer())
#
# # Load collision geometries
# geom_model = pin.buildGeomFromUrdf(model,urdf_model_path,mesh_dir,pin.GeometryType.COLLISION)


model, geom_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)


viz = MeshcatVisualizer(model, geom_model, visual_model)


try:
    viz.initViewer(open=True)
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)

# Load the robot in the viewer.
viz.loadViewerModel()

# while True:
#     time.sleep(.001)

#
#
# # Add collisition pairs
# geom_model.addAllCollisionPairs()
# print("num collision pairs - initial:",len(geom_model.collisionPairs))
#
# # Remove collision pairs listed in the SRDF file
# srdf_filename = "romeo.srdf"
# srdf_model_path = model_path + "/romeo_description/srdf/" + srdf_filename
#
# pin.removeCollisionPairs(model,geom_model,srdf_model_path)
# print("num collision pairs - after removing useless collision pairs:",len(geom_model.collisionPairs))
#
# # Load reference configuration
# pin.loadReferenceConfigurations(model,srdf_model_path)
#
# # Retrieve the half sitting position from the SRDF file
# q = model.referenceConfigurations["half_sitting"]
#
# # Create data structures
# data = model.createData()
# geom_data = pin.GeometryData(geom_model)
#
# # Compute all the collisions
# pin.computeCollisions(model,data,geom_model,geom_data,q,False)
#
# # Print the status of collision for all collision pairs
# for k in range(len(geom_model.collisionPairs)):
#   cr = geom_data.collisionResults[k]
#   cp = geom_model.collisionPairs[k]
#   print("collision pair:",cp.first,",",cp.second,"- collision:","Yes" if cr.isCollision() else "No")
#
# # Compute for a single pair of collision
# pin.updateGeometryPlacements(model,data,geom_model,geom_data,q)
# pin.computeCollision(geom_model,geom_data,0)
