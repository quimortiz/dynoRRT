import numpy as np
import pinocchio as pin
from example_robot_data import load


def load_ur5_parallel():
    """
    Create a robot composed of 4 UR5

    >>> ur5 = load('ur5')
    >>> ur5.nq
    6
    >>> len(ur5.visual_model.geometryObjects)
    7
    >>> robot = load_ur5_parallel()
    >>> robot.nq
    24
    >>> len(robot.visual_model.geometryObjects)
    28
    """
    robot = load("ur5")
    nbRobots = 4

    models = [robot.model.copy() for _ in range(nbRobots)]
    vmodels = [robot.visual_model.copy() for _ in range(nbRobots)]

    # Build the kinematic model by assembling 4 UR5
    fullmodel = pin.Model()

    for irobot, model in enumerate(models):
        # Change frame names
        for i, f in enumerate(model.frames):
            f.name = "%s_#%d" % (f.name, irobot)
        # Change joint names
        for i, n in enumerate(model.names):
            model.names[i] = "%s_#%d" % (n, irobot)

        # Choose the placement of the new arm to be added
        Mt = pin.SE3(
            np.eye(3), np.array([0.3, 0, 0.0])
        )  # First robot is simply translated
        basePlacement = (
            pin.SE3(pin.utils.rotate("z", np.pi * irobot / 2), np.zeros(3)) * Mt
        )

        # Append the kinematic model
        fullmodel = pin.appendModel(fullmodel, model, 0, basePlacement)

    # Build the geometry model
    fullvmodel = pin.GeometryModel()

    for irobot, (model, vmodel) in enumerate(zip(models, vmodels)):
        # Change geometry names
        for i, g in enumerate(vmodel.geometryObjects):
            # Change the name to avoid conflict
            g.name = "%s_#%d" % (g.name, irobot)

            # Refere to new parent names in the full kinematic tree
            g.parentFrame = fullmodel.getFrameId(model.frames[g.parentFrame].name)
            g.parentJoint = fullmodel.getJointId(model.names[g.parentJoint])

            # Append the geometry model
            fullvmodel.addGeometryObject(g)
            # print('add %s on frame %d "%s"' % (g.name, g.parentFrame, fullmodel.frames[g.parentFrame].name))

    fullrobot = pin.RobotWrapper(fullmodel, fullvmodel, fullvmodel)
    # fullrobot.q0 = np.array([-0.375, -1.2  ,  1.71 , -0.51 , -0.375,  0.   ]*4)
    fullrobot.q0 = np.array(
        [np.pi / 4, -np.pi / 4, -np.pi / 2, np.pi / 4, np.pi / 2, 0] * nbRobots
    )

    return fullrobot


if __name__ == "__main__":
    from utils.meshcat_viewer_wrapper import MeshcatVisualizer

    robot = load_ur5_parallel()
    viz = MeshcatVisualizer(robot, url="classical")
    viz.display(robot.q0)
