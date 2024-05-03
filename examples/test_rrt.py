import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from dataclasses import dataclass
import time
import os

# build_cmd = ["make"]
# cwd = "buildRelease"
# subprocess.run(build_cmd, cwd=cwd)
#
#
# sys.path.append(f"./{cwd}/bindings/python")
# sys.path.append("bindings/python")

# cwd = os.environ.get("CWD", "build")
# # cwd = "buildRelease"
# build_cmd = ["make", "-j4"]
# out = subprocess.run(build_cmd, cwd=cwd)
# assert out.returncode == 0

import pydynorrt as pyrrt
import pydynorrt

pydynorrt.srand(2)

xlim = [0, 3]
ylim = [0, 3]


@dataclass
class Obstacle:
    center: np.ndarray
    radius: float


obstacles = [
    Obstacle(np.array([1, 0.4]), 0.5),
    Obstacle(np.array([1, 2]), 0.5),
    Obstacle(np.array([2.5, 2]), 0.5),
]

obs1 = pydynorrt.BallObsX([1, 0.4], 0.5)
obs2 = pydynorrt.BallObsX([1, 2], 0.5)

cm2 = pydynorrt.CMX()
cm2.add_obstacle(obs1)
cm2.add_obstacle(obs2)
cm2.set_radius_robot(0)

counter = 0


def is_collision_free(x: np.ndarray) -> bool:
    """
    x: 3D vector (x, y, theta)

    """
    global counter
    counter += 1
    for obs in obstacles:
        if np.linalg.norm(x - obs.center) < obs.radius:
            return False
    return True


def plot_env(ax, env):
    for obs in env:
        ax.add_patch(plt.Circle((obs.center), obs.radius, color="blue", alpha=0.5))


def plot_robot(ax, x, color="black", alpha=1.0):
    ax.plot([x[0]], [x[1]], marker="o", color=color, alpha=alpha)


# rrt_options = pydynorrt.RRT_options()
# rrt_options.max_it = 80
# rrt_options.max_step = 1.0
# rrt_options.collision_resolution = 0.1
# rrt_options.goal_bias = 0.1


# options_rrt_str = r"""
# [RRT_options]
# max_it = 100
# max_step = 0.1
# goal_bias = 0.5
# """

options_rrt_str = "planner_config/rrt_v0.toml"
options_prm_str = "planner_config/prm_v0.toml"
options_lazyprm_str = "planner_config/lazyprm_v0.toml"

planners = [
    pyrrt.PlannerRRT_Rn
    # pydynorrt.RRT_X,
    # pydynorrt.BiRRT_X,
    # pydynorrt.RRTConnect_X,
    # pydynorrt.PRM_X,
    # pydynorrt.LazyPRM_X
]
options = [
    options_rrt_str,
    None,
    None,
    # options_prm_str
    # options_lazyprm_str
]

names = [
    "RRT",
    # "BiRRT", "RRT_Connect",
    # "PRM",
    # "LazyPRM",
]

# planners = [
#     pydynorrt.RRT_X,
#     # pydynorrt.BiRRT_X,
#     # pydynorrt.RRTConnect_X,
# ]
# options = [
#     options_rrt_str,
#     # None, None
# ]
# names = [
#     "RRT",
#     # "BiRRT", "RRT_Connect"
# ]


for name, planner, options in zip(names, planners, options):
    fig, ax = plt.subplots()
    start = np.array([0.1, 0.1])
    goal = np.array([2.0, 0.2])
    plot_env(ax, obstacles)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title("env, start and goal configurations")

    rrt = planner()
    rrt.set_start(start)

    goal2 = np.array([3.0, 2.8])
    # goal3 = np.array([2.0, 0.2])

    goal_list = [goal, goal2]

    # rrt.set_goal(goal)
    rrt.set_goal_list(goal_list)
    rrt.init(2)
    rrt.set_is_collision_free_fun(is_collision_free)
    # rrt.set_collision_manager(cm2)
    rrt.set_bounds_to_state([xlim[0], ylim[0]], [xlim[1], ylim[1]])

    if options is not None:
        if options.endswith(".toml"):
            rrt.read_cfg_file(options)
        else:
            rrt.read_cfg_string(options)
        # rrt.set_options(rrt_options)

    out = rrt.plan()
    print("counter", counter)
    path = rrt.get_path()
    fine_path = rrt.get_fine_path(0.1)
    valid = rrt.get_configs()
    sample = rrt.get_sample_configs()

    if name == "PRM" or name == "LazyPRM":
        # get adjacency matrix
        adjacency = rrt.get_adjacency_list()
        # print(adjacency)

        for i in range(len(adjacency)):
            for j in adjacency[i]:
                ax.plot(
                    [valid[i][0], valid[j][0]],
                    [valid[i][1], valid[j][1]],
                    color="black",
                    alpha=0.2,
                )

        edges_valid = rrt.get_check_edges_valid()
        edges_invalid = rrt.get_check_edges_invalid()
        print("edges_valid", edges_valid)
        print("edges_invalid", edges_invalid)

        for i in range(len(edges_valid)):
            ax.plot(
                [valid[edges_valid[i][0]][0], valid[edges_valid[i][1]][0]],
                [valid[edges_valid[i][0]][1], valid[edges_valid[i][1]][1]],
                color="green",
                alpha=0.5,
            )
        for i in range(len(edges_invalid)):
            ax.plot(
                [valid[edges_invalid[i][0]][0], valid[edges_invalid[i][1]][0]],
                [valid[edges_invalid[i][0]][1], valid[edges_invalid[i][1]][1]],
                color="red",
                alpha=0.5,
            )

    # .def("get_check_edges_valid", &LazyPRM_X::get_check_edges_valid)
    # .def("get_check_edges_invalid", &LazyPRM_X::get_check_edges_invalid);

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
    # plot_robot(ax, goal, "red")

    for goal in goal_list:
        plot_robot(ax, goal, "red")

    plt.title(name)

    plt.show()


# TODO: print the tree using the parent pointers!
