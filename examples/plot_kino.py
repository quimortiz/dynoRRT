import json
import subprocess
import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np
import sys
import pathlib

sys.path.append(".")


build_cmd = ["make", "-j"]
run_cmd = ["./test_dynorrt", "--run_test=t_kinorrt2", "--", "../"]
cwd = "buildCondaDebug"

out = subprocess.run(build_cmd, cwd=cwd)
assert out.returncode == 0
stdout_file = "/tmp/dynorrt/stdout.txt"
pathlib.Path(stdout_file).parent.mkdir(parents=True, exist_ok=True)
with open(stdout_file, "w") as f:
    out = subprocess.run(run_cmd, cwd=cwd, stdout=f)
    assert out.returncode == 0


with open("/tmp/kino_planar.json", "r") as f:
    d = json.load(f)

sample_configs = d["sample_configs"]
configs = d["configs"]
parents = d["parents"]
path = d["path"]
small_trajectories = d["small_trajectories"]
full_trajectory = d["full_trajectory"]
# print("small trajectories")
# print(small_trajectories)

# for i,traj in enumerate(small_trajectories):
#     print("traj", i)
#     print(len(traj["states"]))
# print(traj["states"])
# print(traj["controls"])


sys.path.append(".")

add_90 = True


def draw_tri(ax, X, add_90=False, fill=None, color="k", l=0.05, alpha=1.0):
    x = X[0]
    y = X[1]
    t = X[2]
    pi2 = 3.1416 / 2
    ratio = 4
    if add_90:
        t += pi2

    vertices = np.array(
        [
            [x + l / ratio * np.cos(t + pi2), y + l / ratio * np.sin(t + pi2)],
            [x + l * np.cos(t), y + l * np.sin(t)],
            [x + l / ratio * np.cos(t - pi2), y + l / ratio * np.sin(t - pi2)],
        ]
    )
    t1 = plt.Polygon(vertices, fill=fill, color=color, alpha=alpha)
    ax.add_patch(t1)
    return t1


def plot_traj(ax, Xs):
    XX = [Xs[nx * i] for i in range(N)]
    YY = [Xs[nx * i + 1] for i in range(N)]
    p = ax.plot(XX, YY)
    color = p[0].get_color()
    for i in range(N):
        draw_tri(
            ax,
            Xs[nx * i : nx * i + 3],
            add_90=add_90,
            fill=False,
            l=0.05,
            alpha=1.0,
            color=color,
        )


# np.random.seed(0)


#
def plot_env(ax, env):
    pass
    # for obs in obstacles:
    #     ax.add_patch(plt.Circle((obs[0]), obs[1], color="blue", alpha=0.5))


def plot_robot(ax, x, color="black", alpha=1.0):
    # p1, p2 = compute_two_points(x)

    draw_tri(ax, x, add_90=add_90, fill=False, l=0.05, alpha=1.0, color=color)


fig, ax = plt.subplots()
# start = np.array( [0.7, 0.8, 0]) # x,y,theta
# goal = np.array( [1.9, 0.3, 0] ) # x,y,theta

start = np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
goal = np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0])


start = np.array([2.0, 1.0, 0.0])
goal = np.array([2.0, 2.0, 0.0])


env = None
plot_env(ax, env)


plot_robot(ax, start, "green")
plot_robot(ax, goal, "red")


min = [0.0, 0.0]
max = [3.0, 1.2]

# ax.set_xlim(min[0], max[0])
# ax.set_ylim(min[1], max[1])
ax.set_aspect("equal")
ax.set_title("env, start and goal configurations")


for i in range(len(sample_configs)):
    plot_robot(ax, sample_configs[i], color="blue")


for i in range(len(configs)):
    # print(configs[i])
    plot_robot(ax, configs[i], color="gray")


# print(len(parents))

for p, i in enumerate(parents):
    if i != -1:
        # plot_robot(ax, configs[i], color="gray")
        # plot_robot(ax, configs[p], color="gray")
        ax.plot(
            [configs[i][0], configs[p][0]],
            [configs[i][1], configs[p][1]],
            color="yellow",
            alpha=0.5,
            # linestyle="dashed",
        )


for i in range(1, len(path) - 1):
    plot_robot(ax, path[i], color="black", alpha=0.5)

for x in full_trajectory["states"]:
    plot_robot(ax, x, color="blue", alpha=0.5)


# for traj in small_trajectories = d["small_trajectories"]

for i in range(len(small_trajectories)):
    traj = small_trajectories[i]
    X = [x[0] for x in traj["states"]]
    Y = [x[1] for x in traj["states"]]
    plt.plot(X, Y, color="gray", alpha=0.5)


plot_robot(ax, start, "green")
plot_robot(ax, goal, "red")


print(len(full_trajectory["states"]))


# plot all small trajectories

# plot_traj(ax, traj)
# total_path += traj


# for p, i in enumerate(parents):
#     if i != -1:
#         # plot_robot(ax, configs[i], color="gray")
#         # plot_robot(ax, configs[p], color="gray")
#         ax.plot(
#             [configs[i][0], configs[p][0]],
#             [configs[i][1], configs[p][1]],
#             color="yellow",
#             alpha=0.5,
#             # linestyle="dashed",
#         )


# for i in range(1, len(fine_path) - 1):
#     plot_robot(ax, fine_path[i], color="black", alpha=0.2)
#
# # for i in range(1,len(shortcut_path) - 1):
# for i in range(len(shortcut_path)):
#     plot_robot(ax, shortcut_path[i], color="orange", alpha=1)


#
# # trace back the solution
#
# i = len(configs) - 1
# path = []
# path.append(np.copy(configs[i]))
# while i != -1:
#     path.append(np.copy(configs[i]))
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

# print states and controls, in two plots, sharing the same x axis
fig, ax = plt.subplots(2, 1, sharex=True)
fig.suptitle("states and controls")
ax[0].set_title("states")
ax[1].set_title("controls")

states = full_trajectory["states"]
controls = full_trajectory["controls"]
X = [x[0] for x in states]
Y = [x[1] for x in states]
TH = [x[2] for x in states]
V = [x[0] for x in controls]
W = [x[1] for x in controls]


ax[0].plot(X, label="x")
ax[0].plot(Y, label="y")
ax[0].plot(TH, label="theta")

ax[1].plot(V, label="v")
ax[1].plot(W, label="w")


ax[0].legend()
ax[1].legend()
plt.show()
