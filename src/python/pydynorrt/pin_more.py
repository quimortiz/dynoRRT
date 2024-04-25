import pinocchio as pin
import numpy as np
import warnings
import meshcat
import os

from pinocchio.visualize.meshcat_visualizer import *

try:
    from pinocchio.visualize.meshcat_visualizer import loadPrimitive
except ImportError:
    pass

import sys
from pinocchio.visualize import MeshcatVisualizer
from pinocchio.robot_wrapper import buildModelsFromUrdf
import pinocchio as pin
import meshcat


try:
    import hppfcl

    WITH_HPP_FCL_BINDINGS = True
except ImportError:
    WITH_HPP_FCL_BINDINGS = False


def rgb2int(r, g, b):
    """
    Convert 3 integers (chars) 0 <= r, g, b < 256 into one single integer = 256**2*r+256*g+b, as expected by Meshcat.

    >>> rgb2int(0, 0, 0)
    0
    >>> rgb2int(0, 0, 255)
    255
    >>> rgb2int(0, 255, 0) == 0x00FF00
    True
    >>> rgb2int(255, 0, 0) == 0xFF0000
    True
    >>> rgb2int(255, 255, 255) == 0xFFFFFF
    True
    """
    return int((r << 16) + (g << 8) + b)


def _materialFromColor(color):
    if isinstance(color, meshcat.geometry.MeshPhongMaterial):
        return color
    elif isinstance(color, str):
        material = colors.colormap[color]
    elif isinstance(color, list):
        material = meshcat.geometry.MeshPhongMaterial()
        material.color = rgb2int(*[int(c * 255) for c in color[:3]])
        if len(color) == 3:
            material.transparent = False
        else:
            material.transparent = color[3] < 1
            material.opacity = float(color[3])
    elif color is None:
        material = random.sample(list(colors.colormap), 1)[0]
    else:
        material = colors.black
    return material


def _addSphere(meshcat_vis, name, radius, color):
    material = _materialFromColor(color)
    meshcat_vis.viewer[name].set_object(meshcat.geometry.Sphere(radius), material)


def _addCylinder(meshcat_vis, name, length, radius, color):
    material = _materialFromColor(color)
    meshcat_vis.viewer[name].set_object(
        meshcat.geometry.Cylinder(length, radius), material
    )


def isMesh(geometry_object):
    """Check whether the geometry object contains a Mesh supported by MeshCat"""
    if geometry_object.meshPath == "":
        return False

    _, file_extension = os.path.splitext(geometry_object.meshPath)
    if file_extension.lower() in [".dae", ".obj", ".stl"]:
        return True

    return False


def _loadViewerGeometryObject(viz, geometry_object, geometry_type, color=None):
    """Load a single geometry object"""
    import meshcat.geometry

    viewer_name = viz.getViewerNodeName(geometry_object, geometry_type)

    is_mesh = False
    try:
        if WITH_HPP_FCL_BINDINGS and isinstance(
            geometry_object.geometry, hppfcl.ShapeBase
        ):
            # Hack to work with different version of Pinocchio
            try:
                obj = viz.loadPrimitive(geometry_object)
            except:
                obj = loadPrimitive(geometry_object)
        elif isMesh(geometry_object):
            obj = viz.loadMesh(geometry_object)
            is_mesh = True
        elif WITH_HPP_FCL_BINDINGS and isinstance(
            geometry_object.geometry, (hppfcl.BVHModelBase, hppfcl.HeightFieldOBBRSS)
        ):
            obj = loadMesh(geometry_object.geometry)
        else:
            msg = (
                "The geometry object named "
                + geometry_object.name
                + " is not supported by Pinocchio/MeshCat for vizualization."
            )
            warnings.warn(msg, category=UserWarning, stacklevel=2)
            return
        if obj is None:
            return
    except Exception as e:
        msg = "Error while loading geometry object: %s\nError message:\n%s" % (
            geometry_object.name,
            e,
        )
        warnings.warn(msg, category=UserWarning, stacklevel=2)
        return

    if isinstance(obj, meshcat.geometry.Object):
        viz.viewer[viewer_name].set_object(obj)
    elif isinstance(obj, meshcat.geometry.Geometry):
        material = meshcat.geometry.MeshPhongMaterial()
        # Set material color from URDF, converting for triplet of doubles to a single int.
        if color is None:
            meshColor = geometry_object.meshColor
        else:
            meshColor = color
        material.color = (
            int(meshColor[0] * 255) * 256**2
            + int(meshColor[1] * 255) * 256
            + int(meshColor[2] * 255)
        )
        # Add transparency, if needed.
        if float(meshColor[3]) != 1.0:
            material.transparent = True
            material.opacity = float(meshColor[3])
        viz.viewer[viewer_name].set_object(obj, material)

    if is_mesh:  # Apply the scaling
        scale = list(np.asarray(geometry_object.meshScale).flatten())
        viz.viewer[viewer_name].set_property("scale", scale)


def _loadViewerModel(viz, rootNodeName="pinocchio", color=None):
    """Load the robot in a MeshCat viewer.
    Parameters:
        rootNodeName: name to give to the robot in the viewer
        color: optional, color to give to the robot. This overwrites the color present in the urdf.
               Format is a list of four RGBA floats (between 0 and 1)
    """

    # Set viewer to use to gepetto-gui.
    viz.viewerRootNodeName = rootNodeName

    # Collisions
    viz.viewerCollisionGroupName = viz.viewerRootNodeName + "/" + "collisions"

    for collision in viz.collision_model.geometryObjects:
        _loadViewerGeometryObject(
            viz, collision, pin.GeometryType.COLLISION, np.array([0.7, 0.7, 0.98, 0.5])
        )
    viz.displayCollisions(True)

    # Visuals
    viz.viewerVisualGroupName = viz.viewerRootNodeName + "/" + "visuals"

    for visual in viz.visual_model.geometryObjects:
        _loadViewerGeometryObject(viz, visual, pin.GeometryType.VISUAL, color)
    viz.displayVisuals(True)


def _applyConfiguration(viz, name, placement):
    if isinstance(placement, list) or isinstance(placement, tuple):
        placement = np.array(placement)
    if isinstance(placement, pin.SE3):
        R, p = placement.rotation, placement.translation
        T = np.r_[np.c_[R, p], [[0, 0, 0, 1]]]
    elif isinstance(placement, np.ndarray):
        if placement.shape == (7,):  # XYZ-quat
            R = pin.Quaternion(np.reshape(placement[3:], [4, 1])).matrix()
            p = placement[:3]
            T = np.r_[np.c_[R, p], [[0, 0, 0, 1]]]
        else:
            print("Error, np.shape of placement is not accepted")
            return False
    else:
        print("Error format of placement is not accepted")
        return False
    viz.viewer[name].set_transform(T)


def display_edge(
    robot, q1, q2, IDX_TOOL, display_count, viz, radius=0.01, color=[1.0, 0.0, 0.0, 1]
):
    # Placement of the end effector tip.
    M1 = robot.framePlacement(q1, IDX_TOOL)
    # Placement of the end effector tip.
    M2 = robot.framePlacement(q2, IDX_TOOL)

    # print(M1)
    # print(M2)

    middle = 0.5 * (M1.translation + M2.translation)
    direction = M2.translation - M1.translation
    length = np.linalg.norm(direction)
    dire = direction / (length + 1e-6)
    dire /= np.linalg.norm(dire)
    if (
        np.linalg.norm(dire - np.array([0.0, 0.0, 1.0])) < 1e-6
        or np.linalg.norm(dire - np.array([0.0, 0.0, -1.0])) < 1e-6
    ):
        print("adding a little bit of noise")
        dire += 0.001 * np.random.rand(3)
        dire /= np.linalg.norm(dire)

    orth = np.cross(dire, np.array([0.0, 0.0, 1.0]))
    orth = orth / np.linalg.norm(orth)
    orth2 = np.cross(dire, orth)
    orth2 = orth2 / np.linalg.norm(orth2)

    # print(f"orth={orth}")
    # print(f"orth2={orth2}")
    # print(f"dire={dire}")
    assert np.abs(np.linalg.norm(dire) - 1) < 1e-8
    assert np.abs(np.linalg.norm(orth) - 1) < 1e-8
    assert np.abs(np.linalg.norm(orth2) - 1) < 1e-8

    Mcyl = pin.SE3(np.stack([orth2, dire, orth], axis=1), middle)
    name = f"world/sph2_{display_count}"
    _addSphere(viz, name, radius, color=[1.0, 0.0, 0.0, 1])
    _applyConfiguration(viz, name, M2)

    name = f"world/sph1_{display_count}"
    _addSphere(viz, name, radius, color=[1.0, 0.0, 0.0, 1])
    _applyConfiguration(viz, name, M1)

    # addCylinderToUniverse(f"world/cyl_vis_{display_count}", radius, length, Mcyl, color=color)

    name = f"world/cil{display_count}"
    # print(f"adding cylinder {name}")
    # print(f"radius={radius} length={length}")
    # _radius = radius / length
    _radius = radius
    # print(f"_radius={_radius}")
    # _addCylinder(viz, name, length, _radius, color)
    # _applyConfiguration(viz, name, Mcyl)
    # display_count += 1
    _addCylinder(viz, name, length, _radius, color=color)
    _applyConfiguration(viz, name, Mcyl)
    display_count += 1


class ViewerHelperRRT:
    def __init__(self, viewer, urdf, package_dirs, start, goal):
        self.viewer = viewer
        # TODO:
        # This version already works in the notebook
        # with 22.
        # self.robot = pin.RobotWrapper.BuildFromURDF(
        #     urdf,
        #     srdf)
        print(package_dirs)
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf, package_dirs=package_dirs)

        self.viz = MeshcatVisualizer(
            self.robot.model, self.robot.collision_model, self.robot.visual_model
        )
        self.viz.initViewer(self.viewer)
        _loadViewerModel(self.viz)

        self.viz_start = MeshcatVisualizer(
            self.robot.model, self.robot.collision_model, self.robot.visual_model
        )
        self.viz_start.initViewer(self.viz.viewer)
        _loadViewerModel(self.viz_start, "start", color=[0.0, 1.0, 0.0, 0.5])

        self.viz_goal = MeshcatVisualizer(
            self.robot.model, self.robot.collision_model, self.robot.visual_model
        )
        self.viz_goal.initViewer(self.viz.viewer)
        _loadViewerModel(self.viz_goal, "goal", color=[1.0, 0.0, 0.0, 0.5])

        # self.viz.display_frames = False
        self.viz_start.display_frames = False
        # self.viz_goal.display_frames = False

        self.viz.display(np.copy(start))
        self.viz_start.display(start)
        self.viz_goal.display(goal)
