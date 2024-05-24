import unittest
import pydynorrt as pyrrt
import numpy as np
import time
import example_robot_data as robex
import pinocchio as pin
import itertools


def XYZRPYtoSE3(xyzrpy):
    rotate = pin.utils.rotate
    R = rotate("x", xyzrpy[3]) @ rotate("y", xyzrpy[4]) @ rotate("z", xyzrpy[5])
    p = np.array(xyzrpy[:3])
    return pin.SE3(R, p)


def load_ur5_with_obstacles(robotname="ur5", reduced=False):
    ### Robot
    # Load the robot
    robot = robex.load(robotname)

    ### If reduced, then only keep should-tilt and elbow joint, hence creating a simple R2 robot.
    if reduced:
        unlocks = [1, 2]
        robot.model, [
            robot.visual_model,
            robot.collision_model,
        ] = pin.buildReducedModel(
            robot.model,
            [robot.visual_model, robot.collision_model],
            [i + 1 for i in range(robot.nq) if i not in unlocks],
            robot.q0,
        )
        robot.data = robot.model.createData()
        robot.collision_data = robot.collision_model.createData()
        robot.visual_data = robot.visual_model.createData()
        robot.q0 = robot.q0[unlocks].copy()

    ### Obstacle map
    # Capsule obstacles will be placed at these XYZ-RPY parameters
    oMobs = [
        [0.40, 0.0, 0.30, np.pi / 2, 0, 0],
        [-0.08, -0.0, 0.69, np.pi / 2, 0, 0],
        [0.23, -0.0, 0.04, np.pi / 2, 0, 0],
        [-0.32, 0.0, -0.08, np.pi / 2, 0, 0],
    ]

    # Load visual objects and add them in collision/visual models
    color = [1.0, 0.2, 0.2, 1.0]  # color of the capsules
    rad, length = 0.1, 0.4  # radius and length of capsules
    for i, xyzrpy in enumerate(oMobs):
        obs = pin.GeometryObject.CreateCapsule(rad, length)  # Pinocchio obstacle object
        obs.meshColor = np.array(
            [1.0, 0.2, 0.2, 1.0]
        )  # Don't forget me, otherwise I am transparent ...
        obs.name = "obs%d" % i  # Set object name
        obs.parentJoint = 0  # Set object parent = 0 = universe
        obs.placement = XYZRPYtoSE3(xyzrpy)  # Set object placement wrt parent
        robot.collision_model.addGeometryObject(obs)  # Add object to collision model
        robot.visual_model.addGeometryObject(obs)  # Add object to visual model

    ### Collision pairs
    nobs = len(oMobs)
    nbodies = robot.collision_model.ngeoms - nobs
    robotBodies = range(nbodies)
    envBodies = range(nbodies, nbodies + nobs)
    robot.collision_model.removeAllCollisionPairs()
    for a, b in itertools.product(robotBodies, envBodies):
        robot.collision_model.addCollisionPair(pin.CollisionPair(a, b))

    ### Geom data
    # Collision/visual models have been modified => re-generate corresponding data.
    robot.collision_data = pin.GeometryData(robot.collision_model)
    robot.visual_data = pin.GeometryData(robot.visual_model)

    return robot


class TestCols(unittest.TestCase):
    def test_1(self):
        """ """

        urdf = pyrrt.DATADIR + "models/ur5_two_robots.urdf"
        srdf = pyrrt.DATADIR + "models/ur5_two_robots.srdf"

        start = np.array(
            [
                1.88495559,
                -0.9424778,
                1.88495559,
                0.0,
                0.0,
                0.0,
                -0.9424778,
                -0.9424778,
                1.57079633,
                0.0,
                0.0,
                0.0,
            ]
        )

        goal = np.array(
            [
                0.62831853,
                -1.25663707,
                1.88495559,
                0.0,
                0.0,
                0.0,
                -2.82743339,
                -0.9424778,
                1.57079633,
                0.0,
                0.0,
                0.0,
            ]
        )

        cm = pyrrt.Collision_manager_pinocchio()
        cm.set_urdf_filename(urdf)
        cm.set_srdf_filename(srdf)
        cm.build()
        self.assertTrue(cm.is_collision_free(start))
        self.assertTrue(cm.is_collision_free(goal))

        N = 100
        col_free = []
        for i in range(N):
            q = start + (goal - start) * i / N
            col_free.append(cm.is_collision_free(q))

        self.assertTrue(np.sum(np.array(col_free) == 0) > 0)
        self.assertTrue(np.sum(np.array(col_free) == 1) > 0)

        confs = []
        for i in range(N):
            q = start + (goal - start) * i / N
            confs.append(q)

        col_free_all, _, _ = cm.is_collision_free_set(confs, True)
        self.assertFalse(col_free_all)

        col_free_sg, _, _ = cm.is_collision_free_set([start, goal], True)
        self.assertTrue(col_free_sg)

    def test_parallel(self):
        """ """

        urdf = pyrrt.DATADIR + "models/ur5_two_robots.urdf"
        srdf = pyrrt.DATADIR + "models/ur5_two_robots.srdf"

        start = np.array(
            [
                1.88495559,
                -0.9424778,
                1.88495559,
                0.0,
                0.0,
                0.0,
                -0.9424778,
                -0.9424778,
                1.57079633,
                0.0,
                0.0,
                0.0,
            ]
        )

        goal = np.array(
            [
                0.62831853,
                -1.25663707,
                1.88495559,
                0.0,
                0.0,
                0.0,
                -2.82743339,
                -0.9424778,
                1.57079633,
                0.0,
                0.0,
                0.0,
            ]
        )

        options = [
            {"num_cores": 1, "pool": False},
            {"num_cores": 4, "pool": False},
            {"num_cores": 4, "pool": True},
        ]

        count_feass = []
        count_infeass = []
        for option in options:

            print("option")
            print(option)
            cm = pyrrt.Collision_manager_pinocchio()
            cm.set_urdf_filename(urdf)
            cm.set_srdf_filename(srdf)
            cm.set_edge_parallel(option["num_cores"])
            cm.build()
            cm.set_use_pool(option["pool"])

            N = 10000
            confs = []
            for i in range(N):
                q = start + (goal - start) * i / N
                confs.append(q)

            tic = time.time()
            col_free_all, count_infeas, count_feas = cm.is_collision_free_set(
                confs, False
            )

            count_feass.append(count_feas)
            count_infeass.append(count_infeas)

            toc = time.time()
            print(count_infeas, count_feas)
            print(f"elapsed [s] {toc - tic}")

            self.assertFalse(col_free_all)

        for e in count_feass:
            self.assertEqual(e, count_feass[0])
        for i in count_infeass:
            self.assertEqual(i, count_infeass[0])

    def test_pin_model(self):
        """ """
        robot = load_ur5_with_obstacles(robotname="ur5", reduced=True)
        cm = pyrrt.Collision_manager_pinocchio()
        pyrrt.set_pin_model(cm, robot.model, robot.collision_model)
        q_start = np.array([-2.5, -1])
        q_goal = np.array([-2, 3])

        self.assertTrue(cm.is_collision_free(q_start))
        self.assertTrue(cm.is_collision_free(q_goal))
        N = 100
        confs = [q_start + (q_goal - q_start) * i / N for i in range(N)]
        col_free_all, _, _ = cm.is_collision_free_set(confs, False)
        self.assertFalse(col_free_all)


if __name__ == "__main__":
    unittest.main()
