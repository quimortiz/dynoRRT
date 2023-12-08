import matplotlib.pyplot as plt
import json
from dataclasses import dataclass
import numpy as np
from typing import Tuple, List


@dataclass
class Obstacle:
    center: np.ndarray
    radius: float


# @dataclass
class BallWorldEnv:
    def __init__(self) -> None:
        self.ub = np.array([1, 1])
        self.lb = np.array([-1, -1])
        self.obstacles: List[Obstacle] = []
        self.start = np.array([0, 0])
        self.goal = np.array([0, 0])
        self.radius_robot = 0.0

    def load_env(self, json_file: str):
        with open(json_file, "r") as f:
            d = json.load(f)
        self.obstacles = [
            Obstacle(np.array(obs["center"]), obs["radius"]) for obs in d["obstacles"]
        ]
        self.ub = d["ub"]
        self.lb = d["lb"]
        self.start = d["start"]
        self.goal = d["goal"]

    def save_env(self, json_file: str):
        d = {}
        d["obstacles"] = [
            {"center": obs.center.tolist(), "radius": obs.radius}
            for obs in self.obstacles
        ]
        d["ub"] = self.ub.tolist()
        d["lb"] = self.lb.tolist()
        d["start"] = self.start.tolist()
        d["goal"] = self.goal.tolist()
        d["radius_robot"] = self.radius_robot
        with open(json_file, "w") as f:
            json.dump(d, f)

    def plot_obstacles(self, ax, color="black", alpha=0.5):
        for obs in self.obstacles:
            ax.add_patch(plt.Circle((obs.center), obs.radius, color=color, alpha=alpha))

    def plot_robot(self, ax, x, color="black", alpha=1.0):
        ax.plot([x[0]], [x[1]], marker="o", color=color, alpha=alpha)

    def plot_problem(self, ax):
        self.plot_obstacles(ax)
        self.plot_robot(ax, self.start, color="green")
        self.plot_robot(ax, self.goal, color="red")
        ax.set_xlim(self.lb[0], self.ub[0])
        ax.set_ylim(self.lb[1], self.ub[1])
        ax.set_aspect("equal")


def create_random_world():
    env = BallWorldEnv()
    num_obstacles = 20

    max_radius = 0.4
    min_radius = 0.1

    env.start = np.array([-1.0, -1.0])
    env.goal = np.array([1, 1])
    env.ub = np.array([2, 2])
    env.lb = np.array([-2, -2])
    margin = 0.1

    for i in range(num_obstacles):
        radius = np.random.uniform(min_radius, max_radius)
        center = np.random.uniform(env.lb, env.ub)

        # check if the obstacle is too close to the start or goal
        if np.linalg.norm(center - env.start) < radius + margin:
            continue
        if np.linalg.norm(center - env.goal) < radius + margin:
            continue

        env.obstacles.append(Obstacle(center, radius))

    plt.figure(figsize=(10, 10))
    env.plot_problem(plt.gca())
    plt.show()

    fileout = "/tmp/my_world.json"
    print("Saving to ", fileout)

    env.save_env(fileout)


# create a bugtrap
def create_bugtrap():
    angles = np.linspace(0, 2 * np.pi, 40)

    env = ballworld_2d.BallWorldEnv()
    env.ub = np.array([2, 2])
    env.lb = np.array([-1, -1])
    env.start = np.array([0, 0])
    env.goal = np.array([1, -0.5])
    env.radius_robot = 0.0

    # leave open at PI

    radius_trap = 0.7
    radius_obstacle = 0.1
    for a in angles:
        if np.abs(a - np.pi) < 0.1:
            continue
        else:
            env.obstacles.append(
                ballworld_2d.Obstacle(
                    np.array([0.1 + radius_trap * np.cos(a), radius_trap * np.sin(a)]),
                    radius_obstacle,
                )
            )

    plt.figure(figsize=(10, 10))
    env.plot_problem(plt.gca())
    env.save_env("/tmp/dynorrt/env.json")
    plt.show()
