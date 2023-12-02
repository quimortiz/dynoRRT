import sys  # noqa

sys.path.append(".")  # noqa
sys.path.append("utils/python")  # noqa


import ballworld_2d
import json
import subprocess
import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np
import sys
from pathlib import Path


build_cmd = ["make", "-j"]
run_cmd = ["./main", "--run_test=t_all_planners_circleworld"]

cwd = "buildAll/buildDebug/"

print("Running: ", " ".join(run_cmd))
print("CWD: ", cwd)
out = subprocess.run(build_cmd, cwd=cwd)

assert out.returncode == 0


file_stdout = "/tmp/dynorrt/stdout.txt"
Path(file_stdout).parent.mkdir(parents=True, exist_ok=True)

with open(file_stdout, "w") as f:
    print("Running: ", " ".join(run_cmd))
    print("CWD: ", cwd)
    out = subprocess.run(run_cmd, cwd=cwd, stdout=f)
    assert out.returncode == 0

# out = subprocess.run(run_cmd, cwd=cwd)
# assert out.returncode == 0


with open("/tmp/dynorrt/out.json", "r") as f:
    D = json.load(f)

envs = D["envs"]
planners = D["planners"]

print("envs: ", envs)
print("planners: ", planners)

for d in D["results"]:

    env = ballworld_2d.BallWorldEnv()
    env.load_env(cwd + d["env"])
    sample_configs = d["sample_configs"]
    configs = d["configs"]
    parents = d["parents"]
    path = d["path"]
    paths = d.get("paths", [])

    parents_backward = d.get("parents_backward", [])
    configs_backward = d.get("configs_backward", [])
    adjacency_list = d.get("adjacency_list", [])

    check_edges_valid = d.get("check_edges_valid", [])
    check_edges_invalid = d.get("check_edges_invalid", [])

    # plot he world

    # print the solution path

    # plt.figure(figsize=(10, 10))
    plt.figure(1)
    plt.axis("equal")

    for path in paths:
        X_path = [X[0] for X in path]
        Y_path = [X[1] for X in path]
        plt.plot(X_path, Y_path, "o-", alpha=0.2, color="blue")

    # plot the graph

    if len(adjacency_list) > 0 and len(configs) > 0:
        for i in range(len(adjacency_list)):
            for j in adjacency_list[i]:
                X = [configs[i][0], configs[j][0]]
                Y = [configs[i][1], configs[j][1]]
                if i < j:
                    plt.plot(X, Y, "-", alpha=0.2, color="black")

    if len(check_edges_valid) > 0 and len(configs) > 0:
        for i in range(len(check_edges_valid)):
            X = [
                configs[check_edges_valid[i][0]][0],
                configs[check_edges_valid[i][1]][0],
            ]
            Y = [
                configs[check_edges_valid[i][0]][1],
                configs[check_edges_valid[i][1]][1],
            ]
            plt.plot(X, Y, "-", alpha=0.4, color="green")

    if len(check_edges_invalid) > 0 and len(configs) > 0:
        for i in range(len(check_edges_invalid)):
            X = [
                configs[check_edges_invalid[i][0]][0],
                configs[check_edges_invalid[i][1]][0],
            ]
            Y = [
                configs[check_edges_invalid[i][0]][1],
                configs[check_edges_invalid[i][1]][1],
            ]
            plt.plot(X, Y, "-", alpha=0.4, color="red")

    if len(parents) > 0 and len(configs) > 0:
        for i in range(len(configs)):
            if parents[i] == -1:
                continue

            X = [configs[i][0], configs[parents[i]][0]]
            Y = [configs[i][1], configs[parents[i]][1]]
            plt.plot(X, Y, "-", alpha=0.2, color="red")

    if len(parents_backward) > 0 and len(configs_backward) > 0:
        for i in range(len(configs_backward)):
            if parents_backward[i] == -1:
                continue

            X = [configs_backward[i][0], configs_backward[parents_backward[i]][0]]
            Y = [configs_backward[i][1], configs_backward[parents_backward[i]][1]]
            plt.plot(X, Y, "-", alpha=0.2, color="blue")

    X_path = [X[0] for X in path]
    Y_path = [X[1] for X in path]
    plt.plot(X_path, Y_path, "o-", alpha=1.0, color="black")

    # env.plot_obstacles(plt.gca(), color="black", alpha=0.5)
    # env.
    env.plot_problem(plt.gca())

    plt.title(d["planner_name"] + " " + d["env"])
    plt.show()

# plt.show()
#
#
# # just build rrt
