import sys

sys.path.append(".")

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

import subprocess


import json


build_cmd = ["make"]
run_cmd = ["./main"]

subprocess.run(build_cmd, cwd="build")
subprocess.run(run_cmd, cwd="build")


with open("out.json", "r") as f:
    d = json.load(f)

sample_configs = d["sample_configs"]
valid_configs = d["valid_configs"]
parents = d["parents"]

import sys

sys.path.append(".")

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import pydynotree as dynotree

# np.random.seed(0)
dynotree.srand(1)


# ref: https://paulbourke.net/geometry/pointlineplane/
def distance_point_to_segment(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    p1, p2: two points defining the segment
    p3: the point
    """
    u = np.dot(p3 - p1, p2 - p1) / np.dot(p2 - p1, p2 - p1)
    u = np.clip(u, 0, 1)
    return np.linalg.norm(p1 + u * (p2 - p1) - p3)


# R2xSO2
# a robot is a segment.


length = 0.5
radius = 0.01


def compute_two_points(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    x: 3D vector (x, y, theta)

    """
    p1 = x[0:2]
    p2 = p1 + length * np.array([np.cos(x[2]), np.sin(x[2])])
    return p1, p2


# env : list( Tuple(np.array, float ) = []

xlim = [0, 3]
ylim = [0, 3]

obstacles = [(np.array([1, 0.4]), 0.5), (np.array([1, 2]), 0.5)]


def plot_env(ax, env):
    for obs in obstacles:
        ax.add_patch(plt.Circle((obs[0]), obs[1], color="blue", alpha=0.5))


def plot_robot(ax, x, color="black", alpha=1.0):
    p1, p2 = compute_two_points(x)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=alpha)
    ax.plot([p1[0]], [p1[1]], marker="o", color=color, alpha=alpha)


def is_collision(x: np.ndarray) -> bool:
    """
    x: 3D vector (x, y, theta)

    """
    p1, p2 = compute_two_points(x)
    for obs in obstacles:
        if distance_point_to_segment(p1, p2, obs[0]) < radius + obs[1]:
            return True
    return False


fig, ax = plt.subplots()
start = np.array([0.1, 0.1, np.pi / 2])
goal = np.array([2.0, 0.2, 0])


plot_env(ax, obstacles)


plot_robot(ax, start, "green")
plot_robot(ax, goal, "red")


ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect("equal")
ax.set_title("env, start and goal configurations")


for i in range(len(sample_configs)):
    plot_robot(ax, sample_configs[i], color="blue")


for i in range(len(valid_configs)):
    print(valid_configs[i])
    plot_robot(ax, valid_configs[i], color="gray")


plot_robot(ax, start, "green")
plot_robot(ax, goal, "red")


for p, i in enumerate(parents):
    if i != -1:
        # plot_robot(ax, valid_configs[i], color="gray")
        # plot_robot(ax, valid_configs[p], color="gray")
        ax.plot(
            [valid_configs[i][0], valid_configs[p][0]],
            [valid_configs[i][1], valid_configs[p][1]],
            color="yellow",
            linestyle="dashed",
        )

#
# # trace back the solution
#
# i = len(valid_configs) - 1
# path = []
# path.append(np.copy(valid_configs[i]))
# while i != -1:
#     path.append(np.copy(valid_configs[i]))
#     i = parents[i]
#
# # interpolate the path
#
# # reverse
# path.reverse()
#
# for i in range(len(path) - 1):
#     _start = path[i]
#     _goal = path[i + 1]
#     for i in range(N):
#         out = np.zeros(3)
#         interpolate_fun(_start, _goal, i / N, out)
#         plot_robot(ax, out, color="gray", alpha=0.5)
# # add the last configuration
# plot_robot(ax, path[-1], color="gray", alpha=0.5)
#
#
# for p in path:
#     plot_robot(ax, p, color="blue", alpha=1)
#
#
# print("path", path)
#
# ax.set_title("rrt solution")
plt.show()
#
#
# # just build rrt
