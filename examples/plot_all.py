import sys
from pathlib import Path

sys.path.append(".")

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

import subprocess
import json


build_cmd = ["make", "-j"]
run_cmd = ["./main", "--run_test=t_all_planners_circleworld"]

cwd = "buildAll/buildDebug/"
out = subprocess.run(build_cmd, cwd=cwd)

assert out.returncode == 0


file_stdout = "/tmp/dynorrt/stdout.txt"
Path(file_stdout).parent.mkdir(parents=True, exist_ok=True)

with open(file_stdout, "w") as f:
    out = subprocess.run(run_cmd, cwd=cwd, stdout=f)
    assert out.returncode == 0

# out = subprocess.run(run_cmd, cwd=cwd)
# assert out.returncode == 0


with open("/tmp/dynorrt/out.json", "r") as f:
    d = json.load(f)

sample_configs = d["sample_configs"]
configs = d["configs"]
parents = d["parents"]
path = d["path"]
paths = d["paths"]


# plot he world


import sys

sys.path.append(".")

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


# print the solution path

plt.figure(figsize=(10, 10))
plt.axis("equal")

for path in paths:
    X_path = [X[0] for X in path]
    Y_path = [X[1] for X in path]
    plt.plot(X_path, Y_path, "o-", alpha=0.2, color="blue")

X_path = [X[0] for X in path]
Y_path = [X[1] for X in path]
plt.plot(X_path, Y_path, "o-", alpha=1.0, color="black")


# plot the tree

for i in range(len(configs)):
    if parents[i] == -1:
        continue

    X = [configs[i][0], configs[parents[i]][0]]
    Y = [configs[i][1], configs[parents[i]][1]]
    plt.plot(X, Y, "-", alpha=0.2, color="red")

plt.show()

# plt.show()
#
#
# # just build rrt
