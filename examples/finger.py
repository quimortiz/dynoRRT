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
import hppfcl as fcl


import os


def custom_configuration_vector(model: pin.Model, **kwargs) -> np.ndarray:
    """Generate a configuration vector where named joints have specific values.

    Args:
        robot: Robot model.
        kwargs: Custom values for joint coordinates.

    Returns:
        Configuration vector where named joints have the values specified in
        keyword arguments, and other joints have their neutral value.
    """
    q = pin.neutral(model)
    for joint_name, joint_value in kwargs.items():
        joint_id = model.getJointId(joint_name)
        print("joint_id", joint_id)
        print(model.joints.tolist())
        print(len(model.joints))
        joint = model.joints[joint_id]
        print("joint", joint)
        print("joint idx_q", joint.idx_q)
        print("joint value", joint_value)
        if type(joint_value) == np.ndarray:
            # assert len(joint_value) == joint.nq
            for i in range(joint.nq):
                q[joint.idx_q + i] = joint_value[i]
        else:
            q[joint.idx_q] = joint_value
    return q


def CreateCube(name, color=[1.0, 0, 0.0, 1.0], fidNames=[]):
    ## Cube model
    parent_id = 0
    mass = 0.5
    cube_length = 0.05

    rmodel = pin.Model()
    rmodel.name = name
    gmodel = pin.GeometryModel()

    ## Joints
    joint_name = name + "_floating_joint"
    joint_placement = pin.SE3.Identity()
    base_id = rmodel.addJoint(
        parent_id, pin.JointModelFreeFlyer(), joint_placement, joint_name
    )
    rmodel.addJointFrame(base_id, -1)

    cube_inertia = pin.Inertia.FromBox(mass, cube_length, cube_length, cube_length)
    cube_placement = pin.SE3.Identity()
    rmodel.appendBodyToJoint(base_id, cube_inertia, cube_placement)

    geom_name = name
    shape = fcl.Box(cube_length, cube_length, cube_length)
    shape_placement = cube_placement.copy()

    geom_obj = pin.GeometryObject(geom_name, base_id, shape, shape_placement)
    geom_obj.meshColor = np.array(color)
    gmodel.addGeometryObject(geom_obj)

    delta = 0.02
    # Contact Frames
    rot = pin.utils.rpyToMatrix(0, 0, 0)
    cnt_placement = pin.SE3(rot, np.array([cube_length / 2.0 + delta, 0.00, 0.0]))
    cnt_frame = pin.Frame(name + "_cnt1", 1, 2, cnt_placement, pin.OP_FRAME)
    fidNames.append(name + "_cnt1")
    rmodel.addFrame(cnt_frame)

    rot = pin.utils.rpyToMatrix(0, 0, np.pi)
    cnt_placement = pin.SE3(rot, np.array([-cube_length / 2.0 - delta, 0.00, 0.0]))
    cnt_frame = pin.Frame(name + "_cnt2", 1, 2, cnt_placement, pin.OP_FRAME)
    fidNames.append(name + "_cnt2")
    rmodel.addFrame(cnt_frame)

    rot = pin.utils.rpyToMatrix(0, 0, np.pi / 2.0)
    cnt_placement = pin.SE3(rot, np.array([0, cube_length / 2.0 + delta, 0.0]))
    cnt_frame = pin.Frame(name + "_cnt3", 1, 2, cnt_placement, pin.OP_FRAME)
    fidNames.append(name + "_cnt3")
    rmodel.addFrame(cnt_frame)
    # fids.append(rmodel.addFrame(cnt_frame))

    rot = pin.utils.rpyToMatrix(0, 0, -np.pi / 2.0)
    cnt_placement = pin.SE3(rot, np.array([0, -cube_length / 2.0 - delta, 0.0]))
    cnt_frame = pin.Frame(name + "_cnt4", 1, 2, cnt_placement, pin.OP_FRAME)
    rmodel.addFrame(cnt_frame)
    fidNames.append(name + "_cnt4")
    # fids.append(rmodel.addFrame(cnt_frame))

    return rmodel, gmodel, fidNames


base_path = pyrrt.DATADIR
urdf = base_path + "/nyu_fingers/nyu_finger_double_w_collision.urdf"
srdf = base_path + "/nyu_fingers/nyu_finger_double_w_collision.srdf"
mesh_path = base_path + "/nyu_fingers"
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf, mesh_path)
viz = MeshcatVisualizer(model, collision_model, visual_model)
collision_model.addAllCollisionPairs()
print("num collision pairs - initial:", len(collision_model.collisionPairs))

geom_model = collision_model
data = model.createData()
geom_data = pin.GeometryData(geom_model)

q = pin.neutral(model)

pin.computeCollisions(model, data, geom_model, geom_data, q, False)

for k in range(len(geom_model.collisionPairs)):
    cr = geom_data.collisionResults[k]
    cp = geom_model.collisionPairs[k]
    print(
        "collision pair:",
        cp.first,
        ",",
        cp.second,
        "- collision:",
        "Yes" if cr.isCollision() else "No",
    )


# lets add two cubes
rmodel1, gmodel1, fidNames = CreateCube("RedCube", [1, 0, 0, 1])
rmodel2, gmodel2, fidNames = CreateCube("BlueCube", [0, 0, 1, 1], fidNames)

rmodel, gmodel = pin.appendModel(
    rmodel2,
    rmodel1,
    gmodel2,
    gmodel1,
    0,
    pin.SE3.Identity(),
)

model_all, geom_model_all = pin.appendModel(
    model,
    rmodel,
    geom_model,
    gmodel,
    0,
    pin.SE3.Identity(),
)


_, visual_model_all = pin.appendModel(
    model,
    rmodel,
    visual_model,
    gmodel,
    0,
    pin.SE3.Identity(),
)


full_robot = pin.RobotWrapper(
    model_all, collision_model=geom_model_all, visual_model=visual_model_all
)
print("full robot")
q = custom_configuration_vector(
    model_all,
    RedCube_floating_joint=np.array([0.03, 0.1, 0, 0, 0, 0, 1.0]),
    BlueCube_floating_joint=np.array([-0.03, -0.1, 0, 0, 0, 0, 1.0]),
)


full_robot_red = full_robot.buildReducedRobot(
    ["RedCube_floating_joint", "BlueCube_floating_joint"], reference_configuration=q
)


import meshcat.geometry as mg


viz = MeshcatVisualizer(
    full_robot_red.model, full_robot_red.collision_model, full_robot_red.visual_model
)


try:
    viz.initViewer(open=True)
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)

viz.loadViewerModel()
viz.displayCollisions(True)
viz.displayFrames(True)


input("press")

full_robot_red.collision_model.addAllCollisionPairs()
print(
    "num collision pairs - initial:", len(full_robot_red.collision_model.collisionPairs)
)
pin.removeCollisionPairs(full_robot_red.model, full_robot_red.collision_model, srdf)
print(
    "num collision pairs - after removing useless collision pairs:",
    len(full_robot_red.collision_model.collisionPairs),
)

for k in range(len(full_robot_red.collision_model.collisionPairs)):
    cr = full_robot_red.collision_data.collisionResults[k]
    cp = full_robot_red.collision_model.collisionPairs[k]
    print(full_robot_red.collision_model.geometryObjects[cp.first].name)
    print(full_robot_red.collision_model.geometryObjects[cp.second].name)
    print(
        "collision pair:",
        cp.first,
        ",",
        cp.second,
        "- collision:",
        "Yes" if cr.isCollision() else "No",
    )


cm = pyrrt.Collision_manager_pinocchio()
cm.set_pin_model(full_robot_red.model, full_robot_red.collision_model)
q_start = pin.neutral(full_robot_red.model)
assert cm.is_collision_free(q_start)

lb = np.array(
    [-1.57079632679, -1.57079632679, -3.14159, -1.57079632679, -1.57079632679, -3.14159]
)
ub = np.array(
    [1.57079632679, 1.57079632679, 3.14159, 1.57079632679, 1.57079632679, 3.14159]
)


print("the position of all the frames in the robot")

q = pin.neutral(full_robot_red.model)
full_robot_red.framesForwardKinematics(q)
pin.updateFramePlacements(full_robot_red.model, full_robot_red.data)

for frame in full_robot_red.model.frames:
    name = frame.name
    id = full_robot_red.model.getFrameId(name)
    print("frame name", name)
    print("frame id", id)
    print("frame placement", full_robot_red.data.oMf[id])

print("different value of q")
q = np.random.uniform(lb, ub)
# q = pin.neutral(full_robot_red.model)
full_robot_red.framesForwardKinematics(q)
pin.updateFramePlacements(full_robot_red.model, full_robot_red.data)

for frame in full_robot_red.model.frames:
    name = frame.name
    id = full_robot_red.model.getFrameId(name)
    print("frame name", name)
    print("frame id", id)
    print("frame placement", full_robot_red.data.oMf[id])


frames = ["RedCube_cnt1", "RedCube_cnt2", "BlueCube_cnt2", "BlueCube_cnt1"]
placements = [
    full_robot_red.data.oMf[full_robot_red.model.getFrameId(name)] for name in frames
]

solver = pyrrt.Pin_ik_solver()
pyrrt.set_pin_model_ik(solver, full_robot_red.model, full_robot_red.collision_model)
delta = 1e-4
solver.set_bounds(lb - delta * np.ones(6), ub - delta * np.ones(6))
solver.set_max_time_ms(3000)
solver.set_max_solutions(5)
solver.set_max_it(1000)
solver.set_joint_reg_penalty(0.001)
q = np.zeros(6)
q[1] = math.pi / 3.0
q[2] = -math.pi / 2.0
q[4] = math.pi / 3.0
q[5] = -math.pi / 2.0
solver.set_joint_reg(q)
solver.set_col_margin(0.01)
solver.set_use_gradient_descent(False)
solver.set_use_finite_diff(False)
solver.set_max_num_attempts(20)


finger_tips = ["finger0_tip_link", "finger1_tip_link"]


solve_ik = False

# lets solve IK
if solve_ik:
    for placement in placements:
        solver.set_frame_positions([placement.translation])
        for finger in finger_tips:

            solver.set_frame_names([finger])
            out = solver.solve_ik()
            tic = time.time()
            ik_solutions = solver.get_ik_solutions()
            toc = time.time()
            print(f"trying to reach {placement.translation}")
            print("number of ik solutions")
            assert len(ik_solutions)
            print(len(ik_solutions))
            for ik_solution in ik_solutions:
                print(ik_solution)
                viz.display(ik_solution)
                input("press")

start = np.copy(q)
# lets do some motion planning
for placement in placements:
    solver.set_frame_positions([placement.translation])
    for finger in finger_tips:
        solver.set_frame_names([finger])
        out = solver.solve_ik()
        tic = time.time()
        ik_solutions = solver.get_ik_solutions()
        assert len(ik_solutions)

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

        toc = time.time()
        print(f"trying to reach {placement.translation}")
        print("number of ik solutions")

        print("valide ik solutions after filtering: ", len(ik_solutions))
        assert cm.is_collision_free(start)
        print("start", start)
        rrt.set_start(start)
        rrt.set_goal_list(ik_solutions)
        rrt.init(6)
        rrt.set_bounds_to_state(lb, ub)
        rrt.set_is_collision_free_fun_from_manager(cm)

        rrt.read_cfg_string(config_str)

        tic = time.time()
        out = rrt.plan()
        toc = time.time()
        print("Planning Time [s]:", toc - tic)

        parents = rrt.get_parents()

        configs = rrt.get_configs()

        robot = full_robot_red
        path = rrt.get_path()
        fine_path = rrt.get_fine_path(0.1)
        # idx_vis_name = finger

        display_count = 0

        # lets try to shortcut the path
        resolution = 0.05
        path_shortcut = pyrrt.PathShortCut_RX()
        path_shortcut.init(6)
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

        for q in new_path_fine:
            viz.display(q)
            time.sleep(0.1)


# now lets do grasping of the cubes


frames = ["RedCube_cnt1", "RedCube_cnt2", "BlueCube_cnt2", "BlueCube_cnt1"]
placements = [
    full_robot_red.data.oMf[full_robot_red.model.getFrameId(name)] for name in frames
]


# pin.rpyToMatrix(0,0,math.pi/4)


# lets rotate the cubes
q = custom_configuration_vector(
    model_all,
    RedCube_floating_joint=np.array([0.03, 0.1, 0, 0, 0, 0, 1.0]),
    BlueCube_floating_joint=np.array([-0.03, -0.1, 0, 0, 0, 0, 1.0]),
)


finger_and_frames = [
    [
        {
            "finger": "finger1_tip_link",
            "frame": "RedCube_cnt1",
        },
        {
            "finger": "finger0_tip_link",
            "frame": "RedCube_cnt2",
        },
    ],
    [
        {
            "finger": "finger1_tip_link",
            "frame": "RedCube_cnt3",
        },
        {
            "finger": "finger0_tip_link",
            "frame": "RedCube_cnt4",
        },
    ],
    [
        {
            "finger": "finger1_tip_link",
            "frame": "RedCube_cnt4",
        },
        {
            "finger": "finger0_tip_link",
            "frame": "RedCube_cnt3",
        },
    ],
    [
        {
            "finger": "finger1_tip_link",
            "frame": "BlueCube_cnt1",
        },
        {
            "finger": "finger0_tip_link",
            "frame": "BlueCube_cnt2",
        },
    ],
    [
        {
            "finger": "finger1_tip_link",
            "frame": "BlueCube_cnt3",
        },
        {
            "finger": "finger0_tip_link",
            "frame": "BlueCube_cnt4",
        },
    ],
    [
        {
            "finger": "finger1_tip_link",
            "frame": "BlueCube_cnt3",
        },
        {
            "finger": "finger0_tip_link",
            "frame": "BlueCube_cnt4",
        },
    ],
]

solver.set_max_solutions(1)
for ffs in finger_and_frames:

    finger_names = []
    frame_positions = []

    for ff in ffs:
        finger_names.append(ff["finger"])
        frame_positions.append(
            full_robot_red.data.oMf[
                full_robot_red.model.getFrameId(ff["frame"])
            ].translation
        )

    print("finger names are")
    print(finger_names)

    print("finger positions are")
    print(frame_positions)

    solver.set_frame_positions(frame_positions)
    solver.set_frame_names(finger_names)
    out = solver.solve_ik()
    tic = time.time()
    ik_solutions = solver.get_ik_solutions()

    if len(ik_solutions) == 0:
        print("no ik solutions")
        continue

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

    assert cm.is_collision_free(start)
    print("start", start)
    rrt.set_start(start)
    rrt.set_goal_list(ik_solutions)
    rrt.init(6)
    rrt.set_bounds_to_state(lb, ub)
    rrt.set_is_collision_free_fun_from_manager(cm)

    rrt.read_cfg_string(config_str)

    tic = time.time()
    out = rrt.plan()
    toc = time.time()
    print("Planning Time [s]:", toc - tic)

    parents = rrt.get_parents()
    configs = rrt.get_configs()

    robot = full_robot_red
    path = rrt.get_path()
    fine_path = rrt.get_fine_path(0.1)
    # idx_vis_name = finger

    display_count = 0

    # lets try to shortcut the path
    resolution = 0.05
    path_shortcut = pyrrt.PathShortCut_RX()
    path_shortcut.init(6)
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

    for q in new_path_fine:
        viz.display(q)
        time.sleep(0.1)
