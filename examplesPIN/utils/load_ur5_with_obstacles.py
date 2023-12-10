"""
Load a UR5 robot model, display it in the viewer.  Also create an obstacle
field made of several capsules, display them in the viewer and create the
collision detection to handle it.
"""

import pinocchio as pin
import example_robot_data as robex
import numpy as np
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


class Target:
    """
    Simple class target that stores and display the position of a target.
    """

    def __init__(
        self, viz=None, color=[0.0, 1.0, 0.2, 1.0], radius=0.05, position=None
    ):
        self.position = position if position is not None else np.array([0.0, 0.0])
        self.initVisual(viz, color, radius)
        self.display()

    def initVisual(self, viz, color, radius):
        self.viz = viz
        if viz is None:
            return
        self.name = "world/pinocchio/target"

        if isinstance(viz, pin.visualize.MeshcatVisualizer):
            import meshcat

            obj = meshcat.geometry.Sphere(radius)
            material = meshcat.geometry.MeshPhongMaterial()
            material.color = (
                int(color[0] * 255) * 256**2
                + int(color[1] * 255) * 256
                + int(color[2] * 255)
            )
            if float(color[3]) != 1.0:
                material.transparent = True
                material.opacity = float(color[3])
            self.viz.viewer[self.name].set_object(obj, material)

        elif isinstance(viz, pin.visualize.GepettoVisualizer):
            self.viz.viewer.gui.addCapsule(self.name, radius, 0.0, color)

    def display(self):
        if self.viz is None or self.position is None:
            return

        if isinstance(self.viz, pin.visualize.MeshcatVisualizer):
            T = np.eye(4)
            T[[0, 2], 3] = self.position
            self.viz.viewer[self.name].set_transform(T)
        elif isinstance(self.viz, pin.visualize.GepettoVisualizer):
            self.viz.viewer.gui.applyConfiguration(
                self.name, [self.position[0], 0, self.position[1], 1.0, 0.0, 0.0, 0.0]
            )
            self.viz.viewer.gui.refresh()
