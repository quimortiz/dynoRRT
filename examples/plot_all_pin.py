import sys  # noqa

sys.path.append(".")  # noqa
sys.path.append("utils/python")  # noqa

import pin_more


import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# from utils.meshcat_viewer_wrapper import colors
import sys  # noqa

sys.path.append(".")  # noqa

# import pydynorrt
import meshcat

# from tp4.collision_wrapper import CollisionWrapper
import example_robot_data as robex
import pinocchio as pin
import time
from pinocchio.robot_wrapper import buildModelsFromUrdf
from pinocchio.visualize import MeshcatVisualizer
import os
import sys
from pinocchio.visualize.meshcat_visualizer import *


build_cmd = ["make", "-j"]
run_cmd = ["./main", "--run_test=t_pin_all"]

cwd = "buildAll/buildRelease/"

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
    # assert out.returncode == 0

# out = subprocess.run(run_cmd, cwd=cwd)
# assert out.returncode == 0


viewer = meshcat.Visualizer()
with open("/tmp/dynorrt/out.json", "r") as f:
    DD = json.load(f)

for D in DD:
    # ../../benchmark/envs/pinocchio/point_mass_cables.json

    # remove ../../
    env = D["env"]
    env = env[6:]

    # base_path = "/home/quim/stg/quim-example-robot-data/example-robot-data/"

    # env = "./benchmark/envs/pinocchio/se3_window.json"
    # env = "./benchmark/envs/pinocchio/ur5_two_arms.json"
    # env = "./benchmark/envs/pinocchio/point_mass_cables.json"
    # envs/pinocchio/ur5_two_arms.json"

    print("loading env", env)
    with open(env) as f:
        ENV = json.load(f)

    start = np.array(ENV["start"])
    goal = np.array(ENV["goal"])

    idx_vis_name = ENV.get("idx_vis_name", "")
    idx_vis_name2 = ENV.get("idx_vis_name2", "")

    print("distance start to goal")
    print(np.linalg.norm(start - goal))

    # frame_reduced = "tool0"
    # frame_reduced2 = "tool0ROBOT2"

    # q_i = np.array([0, -1.5, 2.1, -0.5, -0.5, 0])
    # q_g = np.array([3.1, -1.0, 1, -0.5, -0.5, 0])

    radius = 0.05

    # IDX_VIS = robot.model.getFrameId(frame_reduced)
    # IDX_VIS2 = robot.model.getFrameId(frame_reduced2)

    # robot = pin.RobotWrapper.BuildFromURDF(
    #     base_path + "robots/ur_description/urdf/ur5_robot_with_box.urdf",
    #     base_path + "robots/ur_description/meshes/",
    # )

    # robot = pin.RobotWrapper.BuildFromURDF(
    #     "/home/quim/stg/dynoRRT/benchmark/models/se3_window.urdf",
    #     "meshes")

    # robot = pin.RobotWrapper.BuildFromURDF(
    #     base_path + "robots/ur_description/urdf/ur5_two_robots.urdf",
    #     base_path + "robots/ur_description/meshes/",
    # )

    robot = pin.RobotWrapper.BuildFromURDF(
        ENV["base_path"] + ENV["urdf"], ENV["meshes"]
    )

    collision_model = robot.collision_model
    visual_model = robot.visual_model
    model = robot.model

    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(viewer)
    pin_more._loadViewerModel(viz)

    viz_start = MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )
    viz_start.initViewer(viz.viewer)
    pin_more._loadViewerModel(viz_start, "start", color=[0.0, 1.0, 0.0, 0.5])

    viz_goal = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz_goal.initViewer(viz.viewer)
    pin_more._loadViewerModel(viz_goal, "goal", color=[1.0, 0.0, 0.0, 0.5])

    viz.display(pin.neutral(robot.model))
    viz_start.display(start)
    viz_goal.display(goal)

    IDX_VIS = -1
    IDX_VIS2 = -1

    if idx_vis_name != "":
        IDX_VIS = robot.model.getFrameId(idx_vis_name)

    if idx_vis_name2 != "":
        IDX_VIS2 = robot.model.getFrameId(idx_vis_name2)

    # sys.exit()

    if IDX_VIS != -1:
        M = robot.framePlacement(start, IDX_VIS)
        name = "world/sph_initial"
        pin_more._addSphere(viz, name, radius, [0.0, 1.0, 0.0, 0.5])
        pin_more._applyConfiguration(viz, name, M)

    if IDX_VIS2 != -1:
        M = robot.framePlacement(goal, IDX_VIS)
        name = "world/sph_goal"
        pin_more._addSphere(viz, name, radius, [0.1, 0.0, 0.0, 0.5])
        pin_more._applyConfiguration(viz, name, M)

    viz.display(goal)

    # root = tkinter.Tk()
    # a = tkinter.Label(root, text="Hello World")

    # sys.exit()

    if os.environ.get("INTERACTIVE") is not None:

        input("Press Enter to continue...")

    path = [np.array(x) for x in D["path"]]
    fine_path = [np.array(x) for x in D["fine_path"]]
    print(len(fine_path))
    print(len(path))

    parents = D["parents"]
    configs = [np.array(x) for x in D["configs"]]

    parents_backward = D.get("parents_backward", [])
    configs_backward = [np.array(x) for x in D.get("configs_backward", [])]

    paths = [[np.array(x) for x in path] for path in D.get("paths", [])]

    max_edges = 0

    adjacency_list = D.get("adjacency_list", [])
    check_edges_valid = D.get("check_edges_valid", [])
    check_edges_invalid = D.get("check_edges_invalid", [])

    #
    #
    #
    #

    display_count = 0
    #

    if IDX_VIS != -1:
        for i, p in enumerate(parents):
            if i > max_edges:
                continue
            if p != -1:
                print(f"{i} -> {p}")
                q1 = configs[i]
                q2 = configs[p]
                print(f"q1={q1} q2={q2}")
                pin_more.display_edge(
                    robot,
                    q1,
                    q2,
                    IDX_VIS,
                    display_count,
                    viz,
                    radius=0.005,
                    color=[0.2, 0.8, 0.2, 0.9],
                )
                display_count += 1

    if IDX_VIS2 != -1:
        for i, p in enumerate(parents):
            if i > max_edges:
                continue
            assert p < i
            if p != -1:
                print(f"{i} -> {p}")
                q1 = configs[i]
                q2 = configs[p]
                pin_more.display_edge(
                    robot,
                    q1,
                    q2,
                    IDX_VIS2,
                    display_count,
                    viz,
                    radius=0.005,
                    color=[0.2, 0.6, 0.2, 0.9],
                )
                display_count += 1

    if IDX_VIS != -1:
        if len(parents_backward) > 0 and len(configs_backward) > 0:
            for i, p in enumerate(parents_backward):
                if i > max_edges:
                    continue
                assert p < i
                if p != -1:
                    print(f"{i} -> {p}")
                    q1 = configs_backward[i]
                    q2 = configs_backward[p]
                    pin_more.display_edge(
                        robot,
                        q1,
                        q2,
                        IDX_VIS,
                        display_count,
                        viz,
                        radius=0.005,
                        color=[0.8, 0.2, 0.2, 0.9],
                    )
                    display_count += 1

    if IDX_VIS2 != -1:
        if len(parents_backward) > 0 and len(configs_backward) > 0:
            for i, p in enumerate(parents_backward):
                if i > max_edges:
                    continue
                assert p < i
                if p != -1:
                    print(f"{i} -> {p}")
                    q1 = configs_backward[i]
                    q2 = configs_backward[p]
                    pin_more.display_edge(
                        robot,
                        q1,
                        q2,
                        IDX_VIS2,
                        display_count,
                        viz,
                        radius=0.005,
                        color=[0.6, 0.2, 0.2, 0.9],
                    )
                    display_count += 1

    # if len(adjacency_list) > 0:
    #     if IDX_VIS != 1:
    #         for i, neighbours in enumerate(adjacency_list):
    #             q1 = configs[i]
    #             for e in neighbours:
    #                 if e < i:
    #                     continue
    #                 else:
    #                     q1 = configs[i]
    #                     q2 = configs[e]
    #                     pin_more.display_edge(robot, q1, q2, IDX_VIS, display_count, viz, radius=0.005,
    #                                           color=[0.6, 0.2, 0.2, .9])
    #                     display_count += 1
    #         while True:
    #             time.sleep(.01)

    if len(adjacency_list) and len(check_edges_invalid):
        for e in check_edges_invalid:
            q1 = configs[e[0]]
            q2 = configs[e[1]]
            pin_more.display_edge(
                robot,
                q1,
                q2,
                IDX_VIS,
                display_count,
                viz,
                radius=0.005,
                color=[0.6, 0.2, 0.2, 0.9],
            )
            display_count += 1

        for e in check_edges_valid:
            q1 = configs[e[0]]
            q2 = configs[e[1]]
            pin_more.display_edge(
                robot,
                q1,
                q2,
                IDX_VIS,
                display_count,
                viz,
                radius=0.005,
                color=[0.2, 0.6, 0.2, 0.9],
            )
            display_count += 1

    # input("Press Enter to continue...")
    #
    # for i, p in enumerate(configs):
    #     print(f"config {i}")
    #     viz.display(np.array(p))
    #     input("Press Enter to continue...")

    # for e in D["invalid_edges"]:
    #     print(f"invalid q1={e} q2={e}")
    #     pin_more.display_edge(robot, np.array(e[0]), np.array(e[1]), IDX_VIS,  display_count, viz,
    #                           radius=0.01, color=[1.0, 0.0, 0.0, 1])
    #     display_count += 1

    if len(path):
        if IDX_VIS != -1:
            for i in range(len(path) - 1):
                print(f"{i} -> {i+1}")
                q1 = path[i]
                q2 = path[i + 1]
                print(f"q1={q1} q2={q2}")
                pin_more.display_edge(
                    robot,
                    q1,
                    q2,
                    IDX_VIS,
                    display_count,
                    viz,
                    radius=0.02,
                    color=[0.0, 0.0, 1.0, 0.5],
                )
                display_count += 1

        if IDX_VIS2 != -1:
            for i in range(len(path) - 1):
                print(f"{i} -> {i+1}")
                q1 = path[i]
                q2 = path[i + 1]
                print(f"q1={q1} q2={q2}")
                pin_more.display_edge(
                    robot,
                    q1,
                    q2,
                    IDX_VIS2,
                    display_count,
                    viz,
                    radius=0.02,
                    color=[0.0, 0.0, 1.0, 0.5],
                )
                display_count += 1
    if len(paths):
        for _path in paths:
            if IDX_VIS != -1:
                for i in range(len(path) - 1):
                    print(f"{i} -> {i+1}")
                    q1 = _path[i]
                    q2 = _path[i + 1]
                    print(f"q1={q1} q2={q2}")
                    pin_more.display_edge(
                        robot,
                        q1,
                        q2,
                        IDX_VIS,
                        display_count,
                        viz,
                        radius=0.02,
                        color=[0.1, 0.1, 0.1, 0.5],
                    )
                    display_count += 1

            if IDX_VIS2 != -1:
                for i in range(len(path) - 1):
                    print(f"{i} -> {i+1}")
                    q1 = _path[i]
                    q2 = _path[i + 1]
                    print(f"q1={q1} q2={q2}")
                    pin_more.display_edge(
                        robot,
                        q1,
                        q2,
                        IDX_VIS2,
                        display_count,
                        viz,
                        radius=0.02,
                        color=[0.1, 0.1, 0.1, 0.5],
                    )
                    display_count += 1

    # if len(fine_path):
    #     for i in range(len(fine_path) - 1):
    #         print(f"{i} -> {i+1}")
    #         q1 = fine_path[i]
    #         q2 = fine_path[i + 1]
    #         print(f"q1={q1} q2={q2}")
    #         pin_more.display_edge(robot, q1, q2, IDX_VIS, display_count, viz, radius=0.01,
    #                               color=[0.0, 1.0, 0.0, 1])
    #         display_count += 1

    while True:
        for p in fine_path:
            viz.display(np.array(p))
            time.sleep(0.01)
        break

    viz.clean()

    # envs = D["envs"]
    # planners = D["planners"]
    #
    # print("envs: ", envs)
    # print("planners: ", planners)

    # for d in D["results"]:
    #
    #     env = ballworld_2d.BallWorldEnv()
    #     env.load_env(cwd + d["env"])
    #     sample_configs = d["sample_configs"]
    #     configs = d["configs"]
    #     parents = d["parents"]
    #     path = d["path"]
    #     paths = d.get("paths", [])
    #
    #     parents_backward = d.get("parents_backward", [])
    #     configs_backward = d.get("configs_backward", [])
    #     adjacency_list = d.get("adjacency_list", [])
    #
    #     check_edges_valid = d.get("check_edges_valid", [])
    #     check_edges_invalid = d.get("check_edges_invalid", [])
    #
    #     # plot he world
    #
    #     # print the solution path
    #
    #     # plt.figure(figsize=(10, 10))
    #     plt.figure(1)
    #     plt.axis("equal")
    #
    #     for path in paths:
    #         X_path = [X[0] for X in path]
    #         Y_path = [X[1] for X in path]
    #         plt.plot(X_path, Y_path, "o-", alpha=0.2, color="blue")
    #
    #     # plot the graph
    #
    #     if len(adjacency_list) > 0 and len(configs) > 0:
    #         for i in range(len(adjacency_list)):
    #             for j in adjacency_list[i]:
    #                 X = [configs[i][0], configs[j][0]]
    #                 Y = [configs[i][1], configs[j][1]]
    #                 if i < j:
    #                     plt.plot(X, Y, "-", alpha=0.2, color="black")
    #
    #     if len(check_edges_valid) > 0 and len(configs) > 0:
    #         for i in range(len(check_edges_valid)):
    #             X = [
    #                 configs[check_edges_valid[i][0]][0],
    #                 configs[check_edges_valid[i][1]][0],
    #             ]
    #             Y = [
    #                 configs[check_edges_valid[i][0]][1],
    #                 configs[check_edges_valid[i][1]][1],
    #             ]
    #             plt.plot(X, Y, "-", alpha=0.4, color="green")
    #
    #     if len(check_edges_invalid) > 0 and len(configs) > 0:
    #         for i in range(len(check_edges_invalid)):
    #             X = [
    #                 configs[check_edges_invalid[i][0]][0],
    #                 configs[check_edges_invalid[i][1]][0],
    #             ]
    #             Y = [
    #                 configs[check_edges_invalid[i][0]][1],
    #                 configs[check_edges_invalid[i][1]][1],
    #             ]
    #             plt.plot(X, Y, "-", alpha=0.4, color="red")
    #
    #     if len(parents) > 0 and len(configs) > 0:
    #         for i in range(len(configs)):
    #             if parents[i] == -1:
    #                 continue
    #
    #             X = [configs[i][0], configs[parents[i]][0]]
    #             Y = [configs[i][1], configs[parents[i]][1]]
    #             plt.plot(X, Y, "-", alpha=0.2, color="red")
    #
    #     if len(parents_backward) > 0 and len(configs_backward) > 0:
    #         for i in range(len(configs_backward)):
    #             if parents_backward[i] == -1:
    #                 continue
    #
    #             X = [configs_backward[i][0], configs_backward[parents_backward[i]][0]]
    #             Y = [configs_backward[i][1], configs_backward[parents_backward[i]][1]]
    #             plt.plot(X, Y, "-", alpha=0.2, color="blue")
    #
    #     X_path = [X[0] for X in path]
    #     Y_path = [X[1] for X in path]
    #     plt.plot(X_path, Y_path, "o-", alpha=1.0, color="black")
    #
    #     # env.plot_obstacles(plt.gca(), color="black", alpha=0.5)
    #     # env.
    #     env.plot_problem(plt.gca())
    #
    #     plt.title(d["planner_name"] + " " + d["env"])
    #     plt.show()

    # plt.show()
    #
    #
    # # just build rrt
