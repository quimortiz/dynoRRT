from meshcat.animation import Animation
import numpy as np
import pydynorrt as pyrrt
from pydynorrt import pin_more as pyrrt_vis
import pinocchio as pin
import meshcat
import time
import matplotlib.pyplot as plt
import math
import sys
import meshcat
import scipy.optimize as opt
import scipy
import pickle
from pinocchio.visualize import MeshcatVisualizer


import os

interactive = False
if os.environ.get("DYNORRT_I") == "1":
    interactive = True


base_path = pyrrt.DATADIR
urdf = base_path + "models/iiwa.urdf"
srdf = base_path + "models/kuka.srdf"


robot = pin.RobotWrapper.BuildFromURDF(urdf, base_path + "models")
viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)


try:
    viz.initViewer(open=True)
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)

viz.loadViewerModel()


TARGET = np.array([0.4, 0.0, 0.1])
oMgoal = pin.SE3(np.eye(3), TARGET)
IDX_VIS = robot.model.getFrameId("contact")
id_A5 = robot.model.getFrameId("A5")
id_A6 = robot.model.getFrameId("A6")
id_A7 = robot.model.getFrameId("A7")

viz.displayCollisions(True)

rad = 1.0
q_lim = rad * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-5])
q0 = np.array([0.0, 1.0, 0.0, -1.4, -0.7, 0.0, 0.0])
lb = q0 - q_lim
ub = q0 + q_lim
p_lb = np.array([0.0, -0.2, 0])
p_ub = np.array([1.0, 0.2, 0.9])


generate_valid_goals = True
num_goals = 20


use_ik_solver = True

if generate_valid_goals:

    valid_goals = []
    if use_ik_solver:
        solver = pyrrt.Pin_ik_solver()
        solver.set_urdf_filename(urdf)
        solver.set_srdf_filename(srdf)
        solver.build()
        solver.set_frame_positions([oMgoal.translation])
        solver.set_bounds(lb, ub)
        solver.set_max_num_attempts(1000)
        solver.set_frame_names(["contact"])
        solver.set_max_time_ms(3000)
        solver.set_max_solutions(20)
        solver.set_max_it(1000)
        solver.set_use_gradient_descent(False)
        solver.set_use_finite_diff(False)

        solver.set_joint_reg_penalty(1e-3)
        solver.set_joint_reg(np.ones(7))

        out = solver.solve_ik()
        ik_solutions = solver.get_ik_solutions()
        print("number of solutions", len(ik_solutions))
        print("out", out)

        # solver.get_ik_solutions(ik_solutions)

        for q in ik_solutions:
            viz.display(q)
            if interactive:
                input("press enter")
