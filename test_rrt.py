import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from dataclasses import dataclass
import time

# build_cmd = ["make"]
# cwd = "buildRelease"
# subprocess.run(build_cmd, cwd=cwd)
#
#
# sys.path.append(f"./{cwd}/bindings/python")
sys.path.append("bindings/python")
import pydynorrt

pydynorrt.srand(2)

xlim = [0, 3]
ylim = [0, 3]


@dataclass
class Obstacle:
    center: np.ndarray
    radius: float


obstacles = [Obstacle(np.array([1, 0.4]), 0.5), Obstacle(np.array([1, 2]), 0.5)]


def is_collision_free(x: np.ndarray) -> bool:
    """
    x: 3D vector (x, y, theta)

    """
    for obs in obstacles:
        if np.linalg.norm(x - obs.center) < obs.radius:
            return False
    return True


def plot_env(ax, env):
    for obs in obstacles:
        ax.add_patch(plt.Circle((obs.center), obs.radius, color="blue", alpha=0.5))


def plot_robot(ax, x, color="black", alpha=1.0):
    ax.plot([x[0]], [x[1]], marker="o", color=color, alpha=alpha)


fig, ax = plt.subplots()
start = np.array([0.1, 0.1])
goal = np.array([2.0, 0.2])


plot_env(ax, obstacles)


ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect("equal")
ax.set_title("env, start and goal configurations")


rrt_options = pydynorrt.RRT_options()
rrt_options.max_it = 80
rrt_options.max_step = 1.0
rrt_options.collision_resolution = 0.1
rrt_options.goal_bias = 0.1

rrt = pydynorrt.RRT_X()
rrt.set_start(start)
rrt.set_goal(goal)
rrt.init_tree(2)
rrt.set_is_collision_free_fun(is_collision_free)
rrt.set_bounds_to_state([xlim[0], ylim[0]], [xlim[1], ylim[1]])
rrt.set_options(rrt_options)


out = rrt.plan()
path = rrt.get_path()
fine_path = rrt.get_fine_path(0.1)
valid = rrt.get_configs()
sample = rrt.get_sample_configs()

for v in sample:
    plot_robot(ax, v, color="blue", alpha=0.5)

for v in valid:
    plot_robot(ax, v, color="gray", alpha=0.5)


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

plot_robot(ax, start, "green")
plot_robot(ax, goal, "red")

plt.show()


# TODO: print the tree using the parent pointers!
