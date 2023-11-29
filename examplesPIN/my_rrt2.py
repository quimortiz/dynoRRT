from utils.meshcat_viewer_wrapper import colors
import sys  # noqa

sys.path.append(".")  # noqa

import pydynorrt
from pinocchio.utils import rotate
import meshcat
from tp4.collision_wrapper import CollisionWrapper
import matplotlib.pylab as plt
import example_robot_data as robex
import hppfcl
import math
import numpy as np
import pinocchio as pin
import time
from pinocchio.robot_wrapper import buildModelsFromUrdf
from pinocchio.visualize import MeshcatVisualizer
from tqdm import tqdm
import os
import sys
from pinocchio.visualize.meshcat_visualizer import *
import warnings


from pinocchio.shortcuts import createDatas

try:
    import hppfcl

    WITH_HPP_FCL_BINDINGS = True
except ImportError:
    WITH_HPP_FCL_BINDINGS = False


def _applyConfiguration(viz, name, placement):
    if isinstance(placement, list) or isinstance(placement, tuple):
        placement = np.array(placement)
    if isinstance(placement, pin.SE3):
        R, p = placement.rotation, placement.translation
        T = np.r_[np.c_[R, p], [[0, 0, 0, 1]]]
    elif isinstance(placement, np.ndarray):
        if placement.shape == (7,):  # XYZ-quat
            R = pin.Quaternion(np.reshape(placement[3:], [4, 1])).matrix()
            p = placement[:3]
            T = np.r_[np.c_[R, p], [[0, 0, 0, 1]]]
        else:
            print("Error, np.shape of placement is not accepted")
            return False
    else:
        print("Error format of placement is not accepted")
        return False
    viz.viewer[name].set_transform(T)


def _materialFromColor(color):
    if isinstance(color, meshcat.geometry.MeshPhongMaterial):
        return color
    elif isinstance(color, str):
        material = colors.colormap[color]
    elif isinstance(color, list):
        material = meshcat.geometry.MeshPhongMaterial()
        material.color = colors.rgb2int(*[int(c * 255) for c in color[:3]])
        if len(color) == 3:
            material.transparent = False
        else:
            material.transparent = color[3] < 1
            material.opacity = float(color[3])
    elif color is None:
        material = random.sample(list(colors.colormap), 1)[0]
    else:
        material = colors.black
    return material


def isMesh(geometry_object):
    """Check whether the geometry object contains a Mesh supported by MeshCat"""
    if geometry_object.meshPath == "":
        return False

    _, file_extension = os.path.splitext(geometry_object.meshPath)
    if file_extension.lower() in [".dae", ".obj", ".stl"]:
        return True

    return False


def _addSphere(meshcat_vis, name, radius, color):
    material = _materialFromColor(color)
    meshcat_vis.viewer[name].set_object(meshcat.geometry.Sphere(radius), material)


def _addCylinder(meshcat_vis, name, length, radius, color):
    material = _materialFromColor(color)
    meshcat_vis.viewer[name].set_object(
        meshcat.geometry.Cylinder(length, radius), material
    )


# In[ ]:


plt.ion()


# In[ ]:


# from utils.datastructures.storage import Storage
# from utils.datastructures.pathtree import PathTree
# from utils.datastructures.mtree import MTree


def _materialFromColor(color):
    if isinstance(color, meshcat.geometry.MeshPhongMaterial):
        return color
    elif isinstance(color, str):
        material = colors.colormap[color]
    elif isinstance(color, list):
        material = meshcat.geometry.MeshPhongMaterial()
        material.color = colors.rgb2int(*[int(c * 255) for c in color[:3]])
        if len(color) == 3:
            material.transparent = False
        else:
            material.transparent = color[3] < 1
            material.opacity = float(color[3])
    elif color is None:
        material = random.sample(list(colors.colormap), 1)[0]
    else:
        material = colors.black
    return material


def _loadViewerGeometryObject(viz, geometry_object, geometry_type, color=None):
    """Load a single geometry object"""
    import meshcat.geometry

    viewer_name = viz.getViewerNodeName(geometry_object, geometry_type)

    is_mesh = False
    try:
        if WITH_HPP_FCL_BINDINGS and isinstance(
            geometry_object.geometry, hppfcl.ShapeBase
        ):
            obj = viz.loadPrimitive(geometry_object)
        elif isMesh(geometry_object):
            obj = viz.loadMesh(geometry_object)
            is_mesh = True
        elif WITH_HPP_FCL_BINDINGS and isinstance(
            geometry_object.geometry, (hppfcl.BVHModelBase, hppfcl.HeightFieldOBBRSS)
        ):
            obj = loadMesh(geometry_object.geometry)
        else:
            msg = (
                "The geometry object named "
                + geometry_object.name
                + " is not supported by Pinocchio/MeshCat for vizualization."
            )
            warnings.warn(msg, category=UserWarning, stacklevel=2)
            return
        if obj is None:
            return
    except Exception as e:
        msg = "Error while loading geometry object: %s\nError message:\n%s" % (
            geometry_object.name,
            e,
        )
        warnings.warn(msg, category=UserWarning, stacklevel=2)
        return

    if isinstance(obj, meshcat.geometry.Object):
        viz.viewer[viewer_name].set_object(obj)
    elif isinstance(obj, meshcat.geometry.Geometry):
        material = meshcat.geometry.MeshPhongMaterial()
        # Set material color from URDF, converting for triplet of doubles to a single int.
        if color is None:
            meshColor = geometry_object.meshColor
        else:
            meshColor = color
        material.color = (
            int(meshColor[0] * 255) * 256**2
            + int(meshColor[1] * 255) * 256
            + int(meshColor[2] * 255)
        )
        # Add transparency, if needed.
        if float(meshColor[3]) != 1.0:
            material.transparent = True
            material.opacity = float(meshColor[3])
        viz.viewer[viewer_name].set_object(obj, material)

    if is_mesh:  # Apply the scaling
        scale = list(np.asarray(geometry_object.meshScale).flatten())
        viz.viewer[viewer_name].set_property("scale", scale)


def _loadViewerModel(viz, rootNodeName="pinocchio", color=None):
    """Load the robot in a MeshCat viewer.
    Parameters:
        rootNodeName: name to give to the robot in the viewer
        color: optional, color to give to the robot. This overwrites the color present in the urdf.
               Format is a list of four RGBA floats (between 0 and 1)
    """

    # Set viewer to use to gepetto-gui.
    viz.viewerRootNodeName = rootNodeName

    # Collisions
    viz.viewerCollisionGroupName = viz.viewerRootNodeName + "/" + "collisions"

    for collision in viz.collision_model.geometryObjects:
        _loadViewerGeometryObject(
            viz, collision, pin.GeometryType.COLLISION, np.array([0.7, 0.7, 0.98, 0.5])
        )
    viz.displayCollisions(False)

    # Visuals
    viz.viewerVisualGroupName = viz.viewerRootNodeName + "/" + "visuals"

    for visual in viz.visual_model.geometryObjects:
        _loadViewerGeometryObject(viz, visual, pin.GeometryType.VISUAL, color)
    viz.displayVisuals(True)


# ## Load UR5
# In[ ]:
base_path = "/home/quim/stg/quim-example-robot-data/example-robot-data/"
# model_path = join(pinocchio_model_dir, "example-robot-data/robots")
# mesh_dir = pinocchio_model_dir
# urdf_filename = "talos_reduced.urdf"
# urdf_model_path = join(join(model_path,"talos_data/robots"),urdf_filename)
# urdf_filename = "solo.urdf"
# urdf_model_path = join(join(model_path, "solo_description/robots"), urdf_filename)
# model, collision_model, visual_model = pin.buildModelsFromUrdf(
#     base_path + "robots/ur_description/urdf/ur5_robot.urdf",
#     base_path + "robots/ur_description/meshes/"
# )


robot = pin.RobotWrapper.BuildFromURDF(
    base_path + "robots/ur_description/urdf/ur5_robot.urdf",
    base_path + "robots/ur_description/meshes/",
)

collision_model = robot.collision_model
visual_model = robot.visual_model
model = robot.model

# robot = robex.load("ur5")
# collision_model = robot.collision_model
# visual_model = robot.visual_model


# Recall some placement for the UR5

# In[ ]:

# a = model.placement(model.q0, 6)  # Placement of the end effector joint.
# b = model.framePlacement(model.q0, 22)  # Placement of the end effector tip.

# tool_axis = b.rotation[:, 2]  # Axis of the tool
# tool_position = b.translation


# In[ ]:


# viz = MeshcatVisualizer(None, model, collision_model, visual_model)
# viz = MeshcatVisualizer(robot)

# viz.display(q0)


# for a in robot.collision_model.geometryObjects:
# robot.collision_model[a.name].
# color = np.array([0.7, 0.7, 0.98, .5])
# a.material = _materialFromColor(color)

# print(a.name)
# collison
# def materialFromColor(color):

# change the material of the collision objects


# In[ ]:


# viz.viewer.jupyter_cell()


# Set a start and a goal configuration

# In[ ]:

viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer()
_loadViewerModel(viz)

viz_start = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz_start.initViewer(viz.viewer)
_loadViewerModel(viz_start, "start", color=[0.0, 1.0, 0.0, 0.5])

viz_goal = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz_goal.initViewer(viz.viewer)
_loadViewerModel(viz_goal, "goal", color=[1.0, 0.0, 0.0, 0.5])


# q_i = np.array([1.0, -1.5, 2.1, -0.5, -0.5, 0])
# q_g = np.array([3.0, -1.0, 1, -0.5, -0.5, 0])

q_i = np.array([0, -1.5, 2.1, -0.5, -0.5, 0])
q_g = np.array([3.1, -1.0, 1, -0.5, -0.5, 0])


radius = 0.05

# print(viz.viewerCollisionGroupName)
viz.display(pin.neutral(robot.model))
viz_start.display(q_i)
viz_goal.display(q_g)

# viz.displayCollisions(True)
viz.viewer[viz.viewerCollisionGroupName].set_property("visible", True)
viz.updatePlacements(pin.GeometryType.COLLISION)
# viz.viewer[viz.viewerCollisionGroupName].set_property("opacity", .5)
# viz.viewer[viz.viewerCollisionGroupName + "/base_link_0"].set_property("visible", False)
# set_property("visible", False)
# viz.viewer[viz.viewerCollisionGroupName + "/base_link_0"].material = _materialFromColor([1, 0, 0, 1])


# set_property("opacity", .5)

# if visibility:
#     self.updatePlacements(pin.GeometryType.COLLISION)


# pin.forwardKinematics(model, data, q_i)
# update

# M =

# M = robot.framePlacement(q_i, 22)
# name = "world/sph_initial"
# viz.addSphere(name, radius, [0.0, 1.0, 0.0, 0.5])
# viz.applyConfiguration(name, M)


if os.environ.get("INTERACTIVE") is not None:
    input("Press Enter to continue...")


# viz.display(q_g)
# M = robot.framePlacement(q_g, 22)
# name = "world/sph_goal"
# # viz.addSphere(name, radius, [0.0, 0.0, 1.0, 0.5])
# # viz.applyConfiguration(name, M)
#
# q = np.copy(q_g)


print(robot.__dict__)
robot.gmodel = robot.collision_model

colwrap = CollisionWrapper(robot)  # For collision checking

counter_collision = 0


def addCylinderToUniverse(name, radius, length, placement, color=colors.red):
    geom = pin.GeometryObject(name, 0, hppfcl.Cylinder(radius, length), placement)
    new_id = collision_model.addGeometryObject(geom)
    geom.meshColor = np.array(color)
    visual_model.addGeometryObject(geom)

    for link_id in range(robot.model.nq):
        collision_model.addCollisionPair(pin.CollisionPair(link_id, new_id))
    return geom


# def addBoxToUniverse(name, radius, length, placement, color=colors.red):
#     geom = pin.GeometryObject(name, 0, hppfcl.Cylinder(radius, length), placement)
#     new_id = collision_model.addGeometryObject(geom)
#     geom.meshColor = np.array(color)
#     visual_model.addGeometryObject(geom)
#
#     for link_id in range(robot.model.nq):
#         collision_model.addCollisionPair(pin.CollisionPair(link_id, new_id))
#     return geom


[
    collision_model.removeGeometryObject(e.name)
    for e in collision_model.geometryObjects
    if e.name.startswith("world/")
]

# Add a red box in the viewer
radius = 0.1
length = 1.0

cylID = "world/cyl1"
placement = pin.SE3(pin.SE3(rotate("z", np.pi / 2), np.array([-0.5, 0.4, 0.5])))
addCylinderToUniverse(cylID, radius, length, placement, color=[0.7, 0.7, 0.98, 1])


cylID = "world/cyl2"
placement = pin.SE3(pin.SE3(rotate("z", np.pi / 2), np.array([-0.5, -0.4, 0.5])))
addCylinderToUniverse(cylID, radius, length, placement, color=[0.7, 0.7, 0.98, 1])

cylID = "world/cyl3"
placement = pin.SE3(pin.SE3(rotate("z", np.pi / 2), np.array([-0.5, 0.7, 0.5])))
addCylinderToUniverse(cylID, radius, length, placement, color=[0.7, 0.7, 0.98, 1])


cylID = "world/cyl4"
placement = pin.SE3(pin.SE3(rotate("z", np.pi / 2), np.array([-0.5, -0.7, 0.5])))
addCylinderToUniverse(cylID, radius, length, placement, color=[0.7, 0.7, 0.98, 1])


# update the data
# robot.classicalAcceleration

# robot.data, robot.collision_data, robot.visual_data = createDatas(robot.model,robot.collision_model,robot.visual_model)


viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer()
_loadViewerModel(viz)

viz_start = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz_start.initViewer(viz.viewer)
_loadViewerModel(viz_start, "start", color=[0.0, 1.0, 0.0, 0.5])

viz_goal = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz_goal.initViewer(viz.viewer)
_loadViewerModel(viz_goal, "goal", color=[1.0, 0.0, 0.0, 0.5])


# In[ ]:


# tmp = np.copy(q_i)
# q_i = np.copy(q_g)
# q_g = np.copy(tmp)


# q_g = np.array([2.5, -1.0, 1, -0.5, -0.5, 0])
radius = 0.05


# We need to reload the viewer

# In[ ]:


# viz = MeshcatVisualizer(robot)


# In[ ]:


viz.display(pin.neutral(model))
viz_start.display(q_i)
viz_goal.display(q_g)
M = robot.framePlacement(q_i, 22)
name = "world/sph_initial"
_addSphere(viz, name, radius, [0.0, 1.0, 0.0, 0.5])
_applyConfiguration(viz, name, M)


# In[ ]:


viz.display(q_g)
M = robot.framePlacement(q_g, 22)
name = "world/sph_goal"
_addSphere(viz, name, radius, [0.1, 0.0, 0.0, 0.5])
_applyConfiguration(viz, name, M)

# viz.addSphere(name, radius, [0.0, 0.0, 1.0, 0.5])
# viz.applyConfiguration(name, M)

colwrap = CollisionWrapper(robot)  # For collision checking
colwrap.computeCollisions(q_g)

if os.environ.get("INTERACTIVE") is not None:
    input("Press Enter to continue...")


# pydynorrt.srand(0)
pydynorrt.srand(1)


options = "./planner_config/rrt_v0_PIN.toml"

print("hello")
rrt = pydynorrt.RRT_X()
# rrt.init_tree(6)
rrt.init(6)
rrt.set_start(q_i)
rrt.set_goal(q_g)

print("done")


if options is not None:
    if options.endswith(".toml"):
        rrt.read_cfg_file(options)
    else:
        rrt.read_cfg_string(options)

robot.collision_data = pin.GeometryData(robot.collision_model)
robot.visual_data = pin.GeometryData(robot.visual_model)


def is_colliding(q):
    """
    Use CollisionWrapper to decide if a configuration is in collision
    """
    global counter_collision
    counter_collision += 1
    colwrap.computeCollisions(q)
    collisions = colwrap.getCollisionList()
    return len(collisions) > 0


invalid_configurations = []


def coll(q):
    """Return True if in collision, false otherwise."""
    global counter_collision
    counter_collision += 1
    pin.updateGeometryPlacements(
        robot.model, robot.data, robot.collision_model, robot.collision_data, q
    )
    out = pin.computeCollisions(robot.collision_model, robot.collision_data, False)
    # print(f"evaluating collision, q ={q} out={out} ")
    if out:
        invalid_configurations.append(q)
    return out


# TODO: add option to a custom function to check the distance. Exploration in tree seems so much better!

# rrt.set_is_collision_free_fun(lambda x: not is_colliding(x))
rrt.set_is_collision_free_fun(lambda x: not coll(x))
lb = np.array([-3.2, -3.2, -3.2, -3.2, -3.2, -3.2])
ub = np.array([3.2, 0, 3.2, 3.2, 3.2, 3.2])
rrt.set_bounds_to_state(lb, ub)
# rrt.set_options(rrt_options)


tic = time.time()
out = rrt.plan()
elasped_time_ms = (time.time() - tic) * 1000
print("Compute time (ms) : ", elasped_time_ms)
print("Number of collision checks: ", counter_collision)
print("Collision checks per second: ", counter_collision / elasped_time_ms * 1000)


print("configurations in collision")
for q in invalid_configurations:
    print(q)


path = rrt.get_path()
fine_path = rrt.get_fine_path(0.1)
valid = rrt.get_configs()
sample = rrt.get_sample_configs()
parents = rrt.get_parents()

if os.environ.get("INTERACTIVE") is not None:
    input("Press Enter to continue...")


display_count = 0


def display_edge(q1, q2, radius=0.01, color=[1.0, 0.0, 0.0, 1]):
    global display_count
    M1 = robot.framePlacement(q1, 22)  # Placement of the end effector tip.
    M2 = robot.framePlacement(q2, 22)  # Placement of the end effector tip.
    middle = 0.5 * (M1.translation + M2.translation)
    direction = M2.translation - M1.translation
    length = np.linalg.norm(direction)
    dire = direction / length
    orth = np.cross(dire, np.array([0.0, 0.0, 1.0]))
    orth = orth / np.linalg.norm(orth)
    orth2 = np.cross(dire, orth)
    orth2 = orth2 / np.linalg.norm(orth2)
    assert np.abs(np.linalg.norm(dire) - 1) < 1e-8
    assert np.abs(np.linalg.norm(orth) - 1) < 1e-8
    assert np.abs(np.linalg.norm(orth2) - 1) < 1e-8

    Mcyl = pin.SE3(np.stack([orth2, dire, orth], axis=1), middle)
    name = f"world/sph2_{display_count}"
    _addSphere(viz, name, radius, color=[1.0, 0.0, 0.0, 1])
    _applyConfiguration(viz, name, M2)

    name = f"world/sph1_{display_count}"
    _addSphere(viz, name, radius, color=[1.0, 0.0, 0.0, 1])
    _applyConfiguration(viz, name, M1)

    # addCylinderToUniverse(f"world/cyl_vis_{display_count}", radius, length, Mcyl, color=color)

    name = f"world/cil{display_count}"
    # print(f"adding cylinder {name}")
    # print(f"radius={radius} length={length}")
    # _radius = radius / length
    _radius = radius
    # print(f"_radius={_radius}")
    _addCylinder(viz, name, length, _radius, color)
    _applyConfiguration(viz, name, Mcyl)
    display_count += 1


# def addCylinderToUniverse(name, radius, length, placement, color=colors.red):
#     geom = pin.GeometryObject(name, 0, hppfcl.Cylinder(radius, length), placement)
#     new_id = collision_model.addGeometryObject(geom)
#     geom.meshColor = np.array(color)
#     visual_model.addGeometryObject(geom)


parents = rrt.get_parents()
assert len(parents) == len(valid)
print("path is ")

for i, p in enumerate(path):
    print(f"{i}: {p}")


if os.environ.get("INTERACTIVE") is not None:
    input("Press Enter to continue...")

for i, p in enumerate(parents):
    assert p < i
    if p != -1:
        print(f"{i} -> {p}")
        q1 = valid[i]
        q2 = valid[p]
        display_edge(q1, q2, radius=0.005, color=[0.5, 0.5, 0.5, 1])
        time.sleep(0.05)

for i in range(len(path) - 1):
    print(f"{i} -> {i+1}")
    q1 = path[i]
    q2 = path[i + 1]
    print(f"q1={q1} q2={q2}")
    display_edge(q1, q2, radius=0.02, color=[1.0, 0.0, 0.0, 0.5])

for i in range(len(fine_path) - 1):
    print(f"{i} -> {i+1}")
    q1 = fine_path[i]
    q2 = fine_path[i + 1]
    print(f"q1={q1} q2={q2}")
    display_edge(q1, q2, radius=0.01, color=[0.0, 1.0, 0.0, 1])


# for i in range(len(qs) - 1):
step = 0.1
for p in fine_path:
    viz.display(p)
    time.sleep(step)
# viz.display(qs[-1])
