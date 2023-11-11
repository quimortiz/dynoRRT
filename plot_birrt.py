import sys

sys.path.append(".")

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

import subprocess
import json


build_cmd = ["make"]
run_cmd = ["./main", "--run_test=test_birrt"]

out = subprocess.run(build_cmd, cwd="build")

assert out.returncode == 0

out = subprocess.run(run_cmd, cwd="build")
assert out.returncode == 0


with open("/tmp/dynorrt/test_birrt.json", "r") as f:
    d = json.load(f)

sample_configs = d["sample_configs"]
configs = d["configs"]
parents = d["parents"]
path = d["path"]
fine_path = d["fine_path"]
shortcut_path = d["shortcut_path"]

configs_backward = d["configs_backward"]
parents_backward = d["parents_backward"]

import sys

sys.path.append(".")

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

# np.random.seed(0)


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
    plot_robot(ax, sample_configs[i], color="blue", alpha=0.1)
    plt.plot(
        [sample_configs[i][0]],
        [sample_configs[i][1]],
        marker="o",
        markersize=12,
        color="blue",
        alpha=0.1,
    )


for i in range(len(configs)):
    plot_robot(ax, configs[i], color="green", alpha=0.6)


for i in range(len(configs_backward)):
    plot_robot(ax, configs_backward[i], color="red", alpha=0.6)


plot_robot(ax, start, "green")
plot_robot(ax, goal, "red")


print(len(parents))

for p, i in enumerate(parents):
    if i != -1:
        # plot_robot(ax, configs[i], color="gray")
        # plot_robot(ax, configs[p], color="gray")
        ax.plot(
            [configs[i][0], configs[p][0]],
            [configs[i][1], configs[p][1]],
            color="green",
            alpha=0.2,
            # linestyle="dashed",
        )

for p, i in enumerate(parents_backward):
    if i != -1:
        # plot_robot(ax, configs[i], color="gray")
        # plot_robot(ax, configs[p], color="gray")
        ax.plot(
            [configs_backward[i][0], configs_backward[p][0]],
            [configs_backward[i][1], configs_backward[p][1]],
            color="red",
            alpha=0.2,
            # linestyle="dashed",
        )


for i in range(1, len(path) - 1):
    plot_robot(ax, path[i], color="black", alpha=0.5)
