import tkinter
import sys  # noqa

sys.path.append(".")  # noqa
sys.path.append("utils/python")  # noqa

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


base_path = "/home/quim/stg/quim-example-robot-data/example-robot-data/"
robot = pin.RobotWrapper.BuildFromURDF(
    base_path + "robots/ur_description/urdf/ur5_robot_with_box.urdf",
    base_path + "robots/ur_description/meshes/",
)

collision_model = robot.collision_model
visual_model = robot.visual_model
model = robot.model

viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer()
pin_more._loadViewerModel(viz)

viz_start = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz_start.initViewer(viz.viewer)
pin_more._loadViewerModel(viz_start, "start", color=[0.0, 1.0, 0.0, 0.5])

viz_goal = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz_goal.initViewer(viz.viewer)
pin_more._loadViewerModel(viz_goal, "goal", color=[1.0, 0.0, 0.0, 0.5])


q_i = np.array([0, -1.5, 2.1, -0.5, -0.5, 0])
q_g = np.array([3.1, -1.0, 1, -0.5, -0.5, 0])


viz.display(pin.neutral(robot.model))
viz_start.display(q_i)
viz_goal.display(q_g)


radius = 0.05

IDX_TOOL = robot.model.getFrameId("tool0")
print("IDX_TOOL", IDX_TOOL)


M = robot.framePlacement(q_i, IDX_TOOL)
name = "world/sph_initial"
pin_more._addSphere(viz, name, radius, [0.0, 1.0, 0.0, 0.5])
pin_more._applyConfiguration(viz, name, M)


viz.display(q_g)
M = robot.framePlacement(q_g, IDX_TOOL)
name = "world/sph_goal"
pin_more._addSphere(viz, name, radius, [0.1, 0.0, 0.0, 0.5])
pin_more._applyConfiguration(viz, name, M)


root = tkinter.Tk()
a = tkinter.Label(root, text="Hello World")


def buu(event):
    print(scale.get())


n = len(pin.neutral(robot.model))


class Modifier:
    def __init__(self, i, x):
        print("creating modifier", i)
        self.i = i
        self.x = x

    def update(self, val):
        self.x[self.i] = float(val) / 10 * math.pi
        print(self.x)
        viz.display(np.array(self.x))


x = np.copy(pin.neutral(robot.model))
# x = [0 for i in range(n)]
# modifiers = []

for i in range(n):
    # modifiers.append(Modifier(i, x))
    scale = tkinter.Scale(
        orient="horizontal", from_=-10, to=10, command=Modifier(i, x).update
    )
    scale.pack()

root.mainloop()
