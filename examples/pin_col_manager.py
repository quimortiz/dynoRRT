# import sys  # noqa
# sys.path.append(".")  # noqa


import pin_more
import numpy as np
import time

from pinocchio.visualize.meshcat_visualizer import *
import sys
from pinocchio.visualize import MeshcatVisualizer
from pinocchio.robot_wrapper import buildModelsFromUrdf
import pinocchio as pin
import example_robot_data as robex
import meshcat


import pydynorrt  # noqa


import os


sys.path.append("utils/python")


# import time
# import numpy as np
#


cm = pydynorrt.Collision_manager_pinocchio()
#
#
base_path = os.environ["DYNORRT_PATH"]
#
urdf = base_path + "benchmark/models/point_payload_two_robots.urdf"
srdf = base_path + "benchmark/models/point_payload_two_robots.srdf"
#
# # urdf = "/io/benchmark/models/point_payload_two_robots.urdf"
# # srdf = "/io/benchmark/models/point_payload_two_robots.srdf"
#
# # "/io/benchmark/models/point_payload_two_robots.urdf"
#
cm.set_urdf_filename(urdf)
cm.set_srdf_filename(srdf)
cm.build()


start = np.array([-0.62831853, 0.0, 0.0, 0.0, 0.9424778, 0.0, -0.9424778])
goal = np.array([0.62831853, 0.2, 0.3, 0.0, 0.9424778, 0.0, -0.9424778])

tic = time.time()
N = 1000
for i in range(N):
    o = cm.is_collision_free(start)
toc = time.time()
elapsed = toc - tic
print("Time of 1 Collision in ms: (including python overhead)", 1000.0 / N * elapsed)
print("Time Purely on C++", cm.get_time_ms() / N)
cm.reset_counters()

print("second time")
tic = time.time()
N = 1000
for i in range(N):
    o = cm.is_collision_free(goal)
toc = time.time()
elapsed = toc - tic
print("Time of 1 Collision in ms: (including python overhead)", 1000.0 / N * elapsed)
print("Time Purely on C++", cm.get_time_ms() / N)


print("middle point")
tic = time.time()
N = 1000
mid = (goal + start) / 2.0
for i in range(N):
    o = cm.is_collision_free(mid)
toc = time.time()
elapsed = toc - tic
print("Time of 1 Collision in ms: (including python overhead)", 1000.0 / N * elapsed)
print("Time Purely on C++", cm.get_time_ms() / N)


#
assert cm.is_collision_free(goal)
assert not cm.is_collision_free((goal + start) / 2.0)
cm.reset_counters()
options_cfg_all = base_path + "planner_config/PIN_all.toml"


viewer = meshcat.Visualizer()
viewer_helper = ViewerHelperRRT(viewer, urdf, srdf, start, goal)


input("Press Enter to continue...")

idx_vis_name = "point_mass"
IDX_VIS = -1
# IDX_VIS2 = -1
#
if idx_vis_name != "":
    IDX_VIS = robot.model.getFrameId(idx_vis_name)
#
#     #
#     # ENV["urdf"],
#     # ENV["meshes"])
#
#
# # if idx_vis_name2 != "":
# #     IDX_VIS2 = robot.model.getFrameId(idx_vis_name2)
#
#
# # now lets do planning!
rrt = pydynorrt.RRT_X()

rrt.set_start(start)
rrt.set_goal(goal)
rrt.init(7)
rrt.set_is_collision_free_fun_from_manager(cm)

lb = [-1, -1, -1, -1.5708, -1.5708, -1.5708, -1.5708]
ub = [1, 1, 1, 1.5708, 1.5708, 1.5708, 1.5708]

rrt.set_bounds_to_state(lb, ub)


rrt.read_cfg_file(options_cfg_all)

out = rrt.plan()
parents = rrt.get_parents()
configs = rrt.get_configs()
path = rrt.get_path()
fine_path = rrt.get_fine_path(0.01)

input("Planning DONE! --yes, it was fast. Enter to continue...")

#
max_edges = 1000
display_count = 0


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


for p in fine_path:
    viz.display(np.array(p))
    time.sleep(0.01)


# Visualize the path
