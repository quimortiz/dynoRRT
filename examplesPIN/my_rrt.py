import time
import pinocchio as pin
from utils.meshcat_viewer_wrapper import MeshcatVisualizer
import time
import numpy as np
from numpy.linalg import inv, norm, pinv, svd, eig
from scipy.optimize import fmin_bfgs, fmin_slsqp
from utils.load_ur5_with_obstacles import load_ur5_with_obstacles, Target
import matplotlib.pylab as plt
import os

import sys

sys.path.append(".")

# NOTE: this ur5 robot is using meshes for collision checking.
# If we change the meshes for primitive shapes, we
# will get much faster collision checking
robot = load_ur5_with_obstacles(reduced=True)


# robot.collision_model.addAllCollisionPairs()
import pydynorrt

cm = pydynorrt.Collision_manager_pinocchio()
cm.set_edge_parallel(1)
cm.set_use_pool(True)
pydynorrt.set_pin_model(cm, robot.model, robot.collision_model)


viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
# q_start = np.array([1.5, -1])
# q_goal = np.array([1.5,  1])


q_start = np.array([-2.5, -1])
q_goal = np.array([-2, 3])


robot_start = load_ur5_with_obstacles(reduced=True)
robot_goal = load_ur5_with_obstacles(reduced=True)


# Display another robot.
viz_start = MeshcatVisualizer(robot_start)
viz_start.initViewer(viz.viewer)
viz_start.loadViewerModel(rootNodeName="start")
viz_start.display(q_start)

viz_goal = MeshcatVisualizer(robot_goal)
viz_goal.initViewer(viz.viewer)
viz_goal.loadViewerModel(rootNodeName="goal")
viz_goal.display(q_goal)


assert cm.is_collision_free(q_start)
assert cm.is_collision_free(q_goal)
# cm.is_collision_free(np.zeros(2))
# cm.is_collision_free(np.zeros(2))
# cm = pydynorrt.Collision_manager_pinocchio()
# pydynorrt.set_pin_model(cm,robot.model, robot.collision_model)

target_pos = np.array([0.5, 0.5])
print("the q_goal is touching the target pos")
target = Target(viz, position=target_pos)


def coll(q):
    """Return True if in collision, false otherwise."""
    pin.updateGeometryPlacements(
        robot.model, robot.data, robot.collision_model, robot.collision_data, q
    )
    out = pin.computeCollisions(robot.collision_model, robot.collision_data, False)
    # print(f"evaluating collision, q ={q} out={out} ")

    # assert cm.is_collision_free(q) != out

    return out


import sys

sys.path.append("../buildRelease/bindings/python")
import pydynorrt

config_string = """
[RRT_options]
max_it = 1000
max_step = 1.0
collision_resolution = 0.01
"""

# rrt_options = pydynorrt.RRT_options()
pydynorrt.srand(0)
rrt = pydynorrt.PlannerRRT_Rn()
rrt.set_start(q_start)
rrt.set_goal(q_goal)
rrt.init(2)
# rrt.set_is_collision_free_fun(lambda x: not coll(x))
# rrt.
rrt.set_is_set_collision_free_fun_from_manager_parallel(cm)
rrt.set_is_collision_free_fun_from_manager(cm)
lb = np.array([-3.2, -3.2])
ub = np.array([3.2, 3.2])
rrt.set_bounds_to_state(lb, ub)
rrt.read_cfg_string(config_string)

tic = time.time()
out = rrt.plan()
toc = time.time()
print("Planning time", toc - tic)
path = rrt.get_path()
fine_path = rrt.get_fine_path(0.1)
valid = rrt.get_configs()
sample = rrt.get_sample_configs()


print("planning done")

# def plot_robot(ax, x, color="black", alpha=1.0):
#     ax.plot([x[0]], [x[1]], marker="o", color=color, alpha=alpha)


# fig, ax = plt.subplots()
#
# ax.set_xlim(lb[0], ub[0])
# ax.set_ylim(lb[1], ub[1])


# for v in sample:
#     plot_robot(ax, v, color="blue")
#
# for v in valid:
#     plot_robot(ax, v, color="gray")
#
#
# for i in range(len(path)):
#     plot_robot(ax, path[i], color="black")
#
# for i in range(len(fine_path)):
#     plot_robot(ax, fine_path[i], color="yellow")

# parents = rrt.get_parents()

#
# for i, p in enumerate(parents):
#     if p != -1:
#         print(f"{i} -> {p}")
#         ax.plot(
#             [valid[i][0], valid[p][0]],
#             [valid[i][1], valid[p][1]],
#             color="black",
#             alpha=0.5,
#         )
#
#
# plot_robot(ax, q_start, "green")
# plot_robot(ax, q_goal, "red")
#
# plt.show()
#
# index = 0
# print(path)
# if os.environ.get("INTERACTIVE") is not None:
#     while True:
#         if index == len(path):
#             index = 0
#         if os.environ.get("INTERACTIVE") is not None:
#             input("press enter to continue")
#         q = path[index]
#         print(f"i={index}/{len(path)} q={q}")
#         viz.display(q)
#         time.sleep(0.01)
#         index += 1
#         if index == len(path):
#             break
#
#     index = 0
#
#     while True:
#         if index == len(fine_path):
#             index = 0
#         if os.environ.get("INTERACTIVE") is not None:
#             input("press enter to continue")
#         q = fine_path[index]
#         print(f"i={index}/{len(fine_path)} q={q}")
#         viz.display(q)
#         time.sleep(0.01)
#         index += 1
#
# print("planning done")
