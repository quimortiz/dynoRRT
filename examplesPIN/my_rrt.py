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

robot = load_ur5_with_obstacles(reduced=True)


# The next few lines initialize a 3D viewer.

# In[4]:


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


target_pos = np.array([0.5, 0.5])
print("the q_goal is touching the target pos")
target = Target(viz, position=target_pos)


def coll(q):
    """Return True if in collision, false otherwise."""
    pin.updateGeometryPlacements(
        robot.model, robot.data, robot.collision_model, robot.collision_data, q
    )
    out = pin.computeCollisions(robot.collision_model, robot.collision_data, False)
    print(f"evaluating collision, q ={q} out={out} ")
    return out


import sys

sys.path.append("../buildRelease/bindings/python")
import pydynorrt

rrt_options = pydynorrt.RRT_options()
rrt_options.max_it = 1000
rrt_options.max_step = 1.0
rrt_options.collision_resolution = 0.1

rrt = pydynorrt.RRT_X()
rrt.set_start(q_start)
rrt.set_goal(q_goal)
rrt.init(2)
rrt.set_is_collision_free_fun(lambda x: not coll(x))
lb = np.array([-3.2, -3.2])
ub = np.array([3.2, 3.2])
rrt.set_bounds_to_state(lb, ub)
rrt.set_options(rrt_options)

out = rrt.plan()
path = rrt.get_path()
fine_path = rrt.get_fine_path(0.1)
valid = rrt.get_configs()
sample = rrt.get_sample_configs()


def plot_robot(ax, x, color="black", alpha=1.0):
    ax.plot([x[0]], [x[1]], marker="o", color=color, alpha=alpha)


fig, ax = plt.subplots()

ax.set_xlim(lb[0], ub[0])
ax.set_ylim(lb[1], ub[1])


for v in sample:
    plot_robot(ax, v, color="blue")

for v in valid:
    plot_robot(ax, v, color="gray")


for i in range(len(path)):
    plot_robot(ax, path[i], color="black")

for i in range(len(fine_path)):
    plot_robot(ax, fine_path[i], color="yellow")

parents = rrt.get_parents()


for i, p in enumerate(parents):
    if p != -1:
        print(f"{i} -> {p}")
        ax.plot(
            [valid[i][0], valid[p][0]],
            [valid[i][1], valid[p][1]],
            color="black",
            alpha=0.5,
        )


plot_robot(ax, q_start, "green")
plot_robot(ax, q_goal, "red")

plt.show()

index = 0
print(path)
if os.environ.get("INTERACTIVE") is not None:
    while True:
        if index == len(path):
            index = 0
        if os.environ.get("INTERACTIVE") is not None:
            input("press enter to continue")
        q = path[index]
        print(f"i={index}/{len(path)} q={q}")
        viz.display(q)
        time.sleep(0.01)
        index += 1
        if index == len(path):
            break

    index = 0

    while True:
        if index == len(fine_path):
            index = 0
        if os.environ.get("INTERACTIVE") is not None:
            input("press enter to continue")
        q = fine_path[index]
        print(f"i={index}/{len(fine_path)} q={q}")
        viz.display(q)
        time.sleep(0.01)
        index += 1
