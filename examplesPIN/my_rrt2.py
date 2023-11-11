import example_robot_data as robex
import hppfcl
import math
import numpy as np
import pinocchio as pin
import time
from tqdm import tqdm
import os
import sys


sys.path.append(".")


# In[ ]:


import matplotlib.pylab as plt

plt.ion()


# In[ ]:


from utils.meshcat_viewer_wrapper import MeshcatVisualizer, colors
from utils.datastructures.storage import Storage
from utils.datastructures.pathtree import PathTree
from utils.datastructures.mtree import MTree
from tp4.collision_wrapper import CollisionWrapper


# ## Load UR5

# In[ ]:


robot = robex.load("ur5")
collision_model = robot.collision_model
visual_model = robot.visual_model


# Recall some placement for the UR5

# In[ ]:


a = robot.placement(robot.q0, 6)  # Placement of the end effector joint.
b = robot.framePlacement(robot.q0, 22)  # Placement of the end effector tip.

tool_axis = b.rotation[:, 2]  # Axis of the tool
tool_position = b.translation


# In[ ]:


viz = MeshcatVisualizer(robot)


# In[ ]:


# viz.viewer.jupyter_cell()


# Set a start and a goal configuration

# In[ ]:


q_i = np.array([1.0, -1.5, 2.1, -0.5, -0.5, 0])
q_g = np.array([3.0, -1.0, 1, -0.5, -0.5, 0])
radius = 0.05


# In[ ]:


viz.display(q_i)
M = robot.framePlacement(q_i, 22)
name = "world/sph_initial"
viz.addSphere(name, radius, [0.0, 1.0, 0.0, 0.5])
viz.applyConfiguration(name, M)


if os.environ.get("INTERACTIVE") is not None:
    input("Press Enter to continue...")


viz.display(q_g)
M = robot.framePlacement(q_g, 22)
name = "world/sph_goal"
viz.addSphere(name, radius, [0.0, 0.0, 1.0, 0.5])
viz.applyConfiguration(name, M)

q = np.copy(q_g)


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


from pinocchio.utils import rotate

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


# In[ ]:


q_i = np.array([0, -1.5, 2.1, -0.5, -0.5, 0])
q_g = np.array([3.1, -1.0, 1, -0.5, -0.5, 0])

# tmp = np.copy(q_i)
# q_i = np.copy(q_g)
# q_g = np.copy(tmp)


# q_g = np.array([2.5, -1.0, 1, -0.5, -0.5, 0])
radius = 0.05


# We need to reload the viewer

# In[ ]:


viz = MeshcatVisualizer(robot)


# In[ ]:


viz.display(q_i)
M = robot.framePlacement(q_i, 22)
name = "world/sph_initial"
viz.addSphere(name, radius, [0.0, 1.0, 0.0, 0.5])
viz.applyConfiguration(name, M)


# In[ ]:


viz.display(q_g)
M = robot.framePlacement(q_g, 22)
name = "world/sph_goal"
viz.addSphere(name, radius, [0.0, 0.0, 1.0, 0.5])
viz.applyConfiguration(name, M)


if os.environ.get("INTERACTIVE") is not None:
    input("Press Enter to continue...")

colwrap = CollisionWrapper(robot)  # For collision checking

import pydynorrt

# pydynorrt.srand(0)
pydynorrt.srand(1)

rrt_options = pydynorrt.RRT_options()
rrt_options.max_it = 2000
# rrt_options.max_step =2.0
rrt_options.max_step = 1.0
rrt_options.collision_resolution = 0.2

rrt = pydynorrt.RRT_X()
rrt.init_tree(6)
rrt.set_start(q_i)
rrt.set_goal(q_g)


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


def coll(q):
    """Return True if in collision, false otherwise."""
    global counter_collision
    counter_collision += 1
    pin.updateGeometryPlacements(
        robot.model, robot.data, robot.collision_model, robot.collision_data, q
    )
    out = pin.computeCollisions(robot.collision_model, robot.collision_data, False)
    # print(f"evaluating collision, q ={q} out={out} ")
    return out


# TODO: add option to a custom function to check the distance. Exploration in tree seems so much better!

# rrt.set_is_collision_free_fun(lambda x: not is_colliding(x))
rrt.set_is_collision_free_fun(lambda x: not coll(x))
lb = np.array([-3.2, -3.2, -3.2, -3.2, -3.2, -3.2])
ub = np.array([3.2, 0, 3.2, 3.2, 3.2, 3.2])
rrt.set_bounds_to_state(lb, ub)
rrt.set_options(rrt_options)


out = rrt.plan()
path = rrt.get_path()
fine_path = rrt.get_fine_path(0.1)
valid = rrt.get_configs()
sample = rrt.get_sample_configs()
parents = rrt.get_parents()
print("Number of collision checks: ", counter_collision)

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
    viz.addSphere(name, radius, color=[1.0, 0.0, 0.0, 1])
    viz.applyConfiguration(name, M2)

    name = f"world/sph1_{display_count}"
    viz.addSphere(name, radius, color=[1.0, 0.0, 0.0, 1])
    viz.applyConfiguration(name, M1)

    # addCylinderToUniverse(f"world/cyl_vis_{display_count}", radius, length, Mcyl, color=color)

    name = f"world/cil{display_count}"
    # print(f"adding cylinder {name}")
    # print(f"radius={radius} length={length}")
    # _radius = radius / length
    _radius = radius
    # print(f"_radius={_radius}")
    viz.addCylinder(name, length, _radius, color)
    viz.applyConfiguration(name, Mcyl)
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
