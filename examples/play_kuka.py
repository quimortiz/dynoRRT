from meshcat.animation import Animation
import numpy as np
import pydynorrt as pyrrt
from pydynorrt import pin_more as pyrrt_vis
import pinocchio as pin
import meshcat
import time
import matplotlib.pyplot as plt
import math
import sys
import meshcat
import scipy.optimize as opt
import scipy
import pickle
from pinocchio.visualize import MeshcatVisualizer


import os

interactive = False
if os.environ.get("DYNORRT_I") == "1":
    interactive = True


def solve_ik_with_scipy(x0):
    """ """

    def cost(q):
        weight_bounds = 10
        weight_collisions = 10
        c = 0

        pin.framesForwardKinematics(robot.model, robot.data, q)
        tool_nu = robot.data.oMf[IDX_VIS].translation - oMgoal.translation
        c += 0.5 * tool_nu @ tool_nu

        x = q

        delta = 1e-3
        for i in range(len(q)):
            if x[i] > ub[i] - delta:
                c += weight_bounds**2 * 0.5 * (x[i] - ub[i] + delta) ** 2
            if x[i] < lb[i] + delta:
                c += weight_bounds**2 * 0.5 * (x[i] - lb[i] - delta) ** 2

        contact_frames = [IDX_VIS, id_A5, id_A6, id_A7]

        obs_poses = [
            np.array([0.50, 0.18, 0.32]),
            np.array([0.50, 0.00, 0.32]),
            np.array([0.50, -0.18, 0.32]),
        ]

        radius = 0.15 + 0.02

        for id_frame in contact_frames:
            for obs in obs_poses:
                distance = np.linalg.norm(robot.data.oMf[id_frame].translation - obs)
                if distance < radius:
                    c += weight_collisions**2 * 0.5 * (distance - radius) ** 2

        return c

    res = scipy.optimize.minimize(cost, x0, method="trust-constr")
    return res


base_path = pyrrt.DATADIR
urdf = base_path + "models/iiwa.urdf"
srdf = base_path + "models/kuka.srdf"


robot = pin.RobotWrapper.BuildFromURDF(urdf, base_path + "models")
viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)


try:
    viz.initViewer(open=True)
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)

viz.loadViewerModel()


TARGET = np.array([0.4, 0.0, 0.1])
oMgoal = pin.SE3(np.eye(3), TARGET)
IDX_VIS = robot.model.getFrameId("contact")
id_A5 = robot.model.getFrameId("A5")
id_A6 = robot.model.getFrameId("A6")
id_A7 = robot.model.getFrameId("A7")

viz.displayCollisions(True)

rad = 1.0
q_lim = rad * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-5])
q0 = np.array([0.0, 1.0, 0.0, -1.4, -0.7, 0.0, 0.0])
lb = q0 - q_lim
ub = q0 + q_lim
p_lb = np.array([0.0, -0.2, 0])
p_ub = np.array([1.0, 0.2, 0.9])

cm = pyrrt.Collision_manager_pinocchio()
cm.set_urdf_filename(urdf)
cm.set_srdf_filename(srdf)
cm.build()
cm.reset_counters()

genertate_valid_goals = True
num_goals = 20


if genertate_valid_goals:

    valid_goals = []
    for i in range(1000):
        if i % 100 == 0:
            print("attempt", i)
        x0 = np.random.uniform(lb, ub)
        # q, f = solve_ik(robot, oMgoal, IDX_VIS, x0)
        res = solve_ik_with_scipy(x0)
        q = res.x
        # res = solve_ik_with_scipy(x0)
        # q = res.x
        # check that I am inisie the bounds

        eps_bound = 1e-3

        pin.framesForwardKinematics(robot.model, robot.data, q)
        tool_nu = robot.data.oMf[IDX_VIS].translation - oMgoal.translation
        if np.linalg.norm(tool_nu) < 1e-2:
            if np.all(q > lb - eps_bound) and np.all(q < ub + eps_bound):

                if cm.is_collision_free(q):
                    valid_goals.append(q)
                    # viz.display(q)
                    # input("press enter")
                    if len(valid_goals) == num_goals:
                        print("We have enough goal configurations")
                        break
                else:
                    print("collision")
            else:
                print("not in bounds")
                print(q)
                print(lb)
                print(ub)
        else:
            print("not close to goal")

        with open("valid_goals.pkl", "wb") as f:
            pickle.dump(valid_goals, f)
else:
    with open("valid_goals.pkl", "rb") as f:
        valid_goals = pickle.load(f)


# how many iterations
num_starts = 20


display = False
valid_trajs = []
for i in range(num_starts):

    use_rrt_connect = False
    if use_rrt_connect:
        rrt = pyrrt.PlannerRRTConnect_Rn()
        config_str = """
        [RRTConnect_options]
        max_it = 100000
        collision_resolution = 0.05
        max_step = 1.0
        max_num_configs = 100000
        """

    else:
        rrt = pyrrt.PlannerRRT_Rn()
        config_str = """
        [RRT_options]
        max_it = 20000
        max_num_configs = 20000
        max_step = 1.0
        goal_tolerance = 0.001
        collision_resolution = 0.05
        goal_bias = 0.1
        store_all = false
        """

    # Define sampling fun in python
    def sample_fun(x):
        max_it = 1000
        it = 0
        while it < max_it:
            _x = np.random.uniform(lb, ub)
            pin.framesForwardKinematics(robot.model, robot.data, _x)
            p = robot.data.oMf[IDX_VIS].translation
            if np.all(p >= p_lb) and np.all(p <= p_ub):
                x[:] = _x
                return
            it += 1
        else:
            raise ValueError("Could not find a valid sample")

    def is_valid_fun(x):
        if not cm.is_collision_free(x):
            return False
        pin.framesForwardKinematics(robot.model, robot.data, x)
        p = robot.data.oMf[IDX_VIS].translation

        if not (np.all(p >= p_lb) and np.all(p <= p_ub)):
            return False
        return True

    start = None
    valid_start = False
    while not valid_start:
        s = np.random.uniform(lb, ub)
        if is_valid_fun(s):
            valid_start = True
            start = s

    rrt.set_start(start)
    rrt.set_goal_list(valid_goals)
    rrt.init(7)
    rrt.set_bounds_to_state(lb, ub)

    # rrt.set_sample_fun(sample_fun)
    # rrt.set_is_collision_free_fun(is_valid_fun)

    frame_bound = pyrrt.FrameBounds()
    frame_bound.frame_name = "contact"
    frame_bound.lower = p_lb
    frame_bound.upper = p_ub
    cm.set_frame_bounds([frame_bound])

    rrt.set_is_collision_free_fun_from_manager(cm)

    rrt.read_cfg_string(config_str)

    # Let's plan -- it is fast.

    tic = time.time()
    out = rrt.plan()
    toc = time.time()
    print("Planning Time [s]:", toc - tic)

    assert out == pyrrt.TerminationCondition.GOAL_REACHED

    parents = rrt.get_parents()
    configs = rrt.get_configs()
    path = rrt.get_path()
    fine_path = rrt.get_fine_path(0.1)

    # lets try to shortcut the path
    resolution = 0.05
    path_shortcut = pyrrt.PathShortCut_RX()
    path_shortcut.init(7)
    path_shortcut.set_bounds_to_state(lb, ub)
    cm.reset_counters()
    path_shortcut.set_is_collision_free_fun_from_manager(cm)
    path_shortcut.set_initial_path(fine_path)
    tic = time.time()
    path_shortcut.shortcut()
    toc = time.time()
    print("Shortcutting Time [s]:", toc - tic)
    print("number of collision checks", cm.get_num_collision_checks())
    shortcut_data = path_shortcut.get_planner_data()
    print("evaluated edges", shortcut_data["evaluated_edges"])
    print("infeasible edges", shortcut_data["infeasible_edges"])
    new_path = path_shortcut.get_path()
    new_path_fine = path_shortcut.get_fine_path(0.1)

    valid_trajs.append(new_path_fine)

    shortcut_data = path_shortcut.get_planner_data()

    # NOTE: we display only one of the possible goals

    #     viewer, urdf, base_path + "models", start, goal
    # )

    if display:

        viewer = meshcat.Visualizer()
        goal = valid_goals[0]
        viewer_helper = pyrrt_vis.ViewerHelperRRT(
            viewer, urdf, package_dirs=base_path + "models", start=start, goal=goal
        )

        robot = viewer_helper.robot
        viz = viewer_helper.viz
        idx_vis_name = "contact"
        IDX_VIS = robot.model.getFrameId(idx_vis_name)
        display_count = 0  # Just to enumerate the number
        for i, p in enumerate(parents):
            if p != -1:
                q1 = configs[i]
                q2 = configs[p]

                pin.framesForwardKinematics(robot.model, robot.data, q1)
                p = robot.data.oMf[IDX_VIS].translation
                print("p", p)

                pyrrt_vis.display_edge(
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

        for i in range(len(path) - 1):
            q1 = path[i]
            q2 = path[i + 1]
            pyrrt_vis.display_edge(
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

        for i in range(len(new_path_fine) - 1):
            q1 = new_path_fine[i]
            q2 = new_path_fine[i + 1]
            pyrrt_vis.display_edge(
                robot,
                q1,
                q2,
                IDX_VIS,
                display_count,
                viz,
                radius=0.02,
                color=[0.0, 1.0, 1.0, 0.5],
            )
            display_count += 1

        anim = Animation()
        __v = viewer_helper.viz.viewer
        for i in range(len(fine_path)):
            with anim.at_frame(viewer, i) as frame:
                viewer_helper.viz.viewer = frame
                viz.display(fine_path[i])

        viewer.set_animation(anim)
        viewer_helper.viz.viewer = __v

        html = viewer.static_html()

# lets see the trajectories!!
for traj in valid_trajs:
    if interactive:
        input("press enter")
        for state in traj:
            viz.display(state)
            time.sleep(0.05)
    else:
        pass

        # viz.display(q)
        # input("press enter")


# save in a file
# with open("kuka.html", "w") as f:
#     f.write(html)


if interactive:
    input("Press Enter to finish")
