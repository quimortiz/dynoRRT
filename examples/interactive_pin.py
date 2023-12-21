import tkinter
import sys  # noqa

sys.path.append(".")  # noqa
sys.path.append("utils/python")  # noqa

sys.path.append("src/python/pydynorrt")  # noqa
import pin_more


import ballworld_2d
import json
import subprocess
import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np
import sys
from pathlib import Path

# from utils.meshcat_viewer_wrapper import colors
import sys  # noqa

sys.path.append(".")  # noqa

# import pydynorrt
from pinocchio.utils import rotate
import meshcat

# from tp4.collision_wrapper import CollisionWrapper
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

# base_path = "/home/quim/stg/quim-example-robot-data/example-robot-data/"
# robot = "robots/ur_description/urdf/ur5_two_robots.urdf"


robots = "/home/quim/stg/dynoRRT/src/python/pydynorrt/data/models/unicycle_parallel_park.urdf"


# robot = "/home/quim/stg/dynoRRT/benchmark/models/se3_window.urdf"

robot = pin.RobotWrapper.BuildFromURDF(
    robots
    # "benchmark/models/point_payload_two_robots.urdf"
    # base_path + robot,
    # robot
    # base_path + "/
    #     "robots/ur_description/urdf/ur5_robot_with_box.urdf",
    # base_path + "robots/ur_description/meshes/",
    # base_path + "robots/ur_description/meshes/",
)

collision_model = robot.collision_model
visual_model = robot.visual_model
model = robot.model

viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer()
pin_more._loadViewerModel(viz)

start = np.array([0.7, 0.8, 0])  # x,y,theta
goal = np.array([1.9, 0.3, 0])  # x,y,theta

q_i = start
# np.zeros(robot.nq)

q_g = goal

# q_i = np.array([-0.62831853, 0.0, 0.0, 0.0, 0.9424778, 0.0, -0.9424778])
#
#
# q_g = np.array([0.62831853, 0.2, 0.3, 0.0, 0.9424778, 0.0, -0.9424778])


# q_g = np.zeros(robot.nq)

qs = pin.neutral(robot.model)

# q_i = np.array([1.88495559, -0.9424778,   1.88495559,  0.,          0.,          0.,
#                -0.9424778,  -0.9424778,   1.57079633,  0.,          0.,          0.])
#
# qs = np.copy(q_i)
#
# q_g = np.array([0.62831853, -1.25663707,  1.88495559,  0.,          0.,          0.,
#                -2.82743339, -0.9424778,   1.57079633,  0.,          0.,          0.,])
#
# w1 = np.array([1.88495559, -1.8849556,   0.62831853,  0.,          0.,   0.,
#                -0.9424778,  -0.9424778,   1.57079633,  0.,          0.,          0.])
#
# w2 = np.array([1.88495559, -1.8849556,   0.62831853,  0.,          0.,   0.,
#                -2.82743339, -0.9424778,   1.57079633,  0.,          0.,          0.,])


viz.display(q_i)

# q_g = np.array([3.1, -1.0, 1, -0.5, -0.5, 0])
# q_g = np.array([3.1, -1.0, 1, -0.5, -0.5, 0])

show_start_and_goal = True
if show_start_and_goal:
    viz_start = MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )
    viz_start.initViewer(viz.viewer)
    pin_more._loadViewerModel(viz_start, "start", color=[0.0, 1.0, 0.0, 0.5])

    viz_goal = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz_goal.initViewer(viz.viewer)
    pin_more._loadViewerModel(viz_goal, "goal", color=[1.0, 0.0, 0.0, 0.5])

    viz_start.display(q_i)
    viz_goal.display(q_g)

# viz_goal.display(q_g)

# radius = 0.05
#
# IDX_TOOL = robot.model.getFrameId("tool0")
# print("IDX_TOOL", IDX_TOOL)
#
#
# M = robot.framePlacement(q_i, IDX_TOOL)
# name = "world/sph_initial"
# pin_more._addSphere(viz, name, radius, [0.0, 1.0, 0.0, 0.5])
# pin_more._applyConfiguration(viz, name, M)
#
#
# viz.display(q_g)
# M = robot.framePlacement(q_g, IDX_TOOL)
# name = "world/sph_goal"
# pin_more._addSphere(viz, name, radius, [0.1, 0.0, 0.0, 0.5])
# pin_more._applyConfiguration(viz, name, M)
# pin_more._applyConfiguration(viz, name, M)


root = tkinter.Tk()
a = tkinter.Label(root, text="Hello World")


def buu(event):
    print(scale.get())


n = len(pin.neutral(robot.model))

delta = np.zeros(model.nq)


class Modifier:
    def __init__(self, i, delta):
        print("creating modifier", i)
        self.i = i
        self.delta = delta

    def update(self, val):
        delta[self.i] = float(val) / 10 * math.pi
        print(np.array(self.delta) + x)
        viz.display(np.array(self.delta) + x)


# x = np.copy(pin.neutral(robot.model))
x = np.copy(qs)
# x = [0 for i in range(n)]
# modifiers = []

for i in range(n):
    # modifiers.append(Modifier(i, x))
    scale = tkinter.Scale(
        orient="horizontal", from_=-10, to=10, command=Modifier(i, delta).update
    )
    scale.pack()

root.mainloop()


# interplote

path = [np.copy(q_i), np.copy(w1), np.copy(w2), np.copy(q_g)]
N = 10
path_fine = []

for i in range(len(path) - 1):
    for j in range(N):
        path_fine.append(path[i] + (path[i + 1] - path[i]) * j / N)


while True:
    for p in path_fine:
        viz.display(np.array(p))
        time.sleep(0.1)
