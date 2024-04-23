# DynoRRT

Try it out online! [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quimortiz/dynorrt/main?labpath=notebooks%2Ftutorial0.ipynb)

<a target="_blank" href="https://colab.research.google.com/github/quimortiz/dynoRRT/blob/main/notebooks/tutorial0.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


DynoRRT is a C++/Python library for sampling-based motion planning, such as Rapidly Exploring Random Trees (RRT) and Probabilistic Roadmaps (PRM).

It delivers state-of-the-art performance with an easy-to-use and install Python interface. The installation is straightforward: no need for ROS, system-wide packages, MOVEIT, or OMPL—just a simple `pip install`. Plus, it's significantly faster than OMPL.

With DynoRRT, you can define and solve a motion planning problem within 60 seconds and solve it in milliseconds. Planning problems can be described using URDF Files. We rely on Pinocchio and HPP-FCL for collision detection and forward kinematics calculations. These libraries are statically linked, so you don't need them at runtime—or it's fine if you have another version of these libraries.

The Python package is created using pybind11, and the API is very similar to the C++ interface. Additionally, the Python package provides a couple of utilities to visualize the problems using Pinocchio and Meshcat, but you're free to use any viewer you prefer.

The library is currently in its alpha state. We are aiming for a public release of version 0.1 in January. The C++ packaging is still under development. Feel free to open a GitHub issue or pull request! Special help is needed for Mac support.

```bash
pip3 install pydynorrt
```

## Tutorial

You can try it online! Run the `tutorial0` notebook in Binder.

## Run tests in C++

```bash
cd build
./test_dynorrt --log_sink=z.log  -- ../ _deps/dynobench-src/
```

# Planners

**Geometric Planners:**
- RRT
- RRT Connect
- Bidirectional RRT
- PRM (Optionally, check edges using Lazy Astar, similar to BIT*)
- LAZY PRM
- RRT\*
- (Coming soon) LazyPRM *, PRM*

**Kinodynamic Planners:**
- (Coming soon) Kinodynamic RRT
- (Coming soon) SST*
- AO RRT

Looking for more planners? Check out our latest work in kinodynamic motion planning using Search And Trajectory Optimization. Our state-of-the-art motion planners for optimal kinodynamic motion planning are available [here](https://github.com/quimortiz/dynoplan).

# Collision and Robots

- Using Python bindings: You can define the collision function in PYTHON
- Robotics: Examples using Python Bindings of Pinocchio for collision checking
- In C++: Flexibility to implement whatever you want.
- In Python: Similar flexibility, but it may be slower.
- C++ Pinocchio Interface: Define your problem using a URDF FILE.
- C++ Collision for Point robot in 2D and 3D and spheres/boxes (maybe using hpp-fcl?, or just coding)

# How To Use the Code

- Python Package
- (Coming soon) C++: Easy integration into CMAKE build system. As submodule, Fetch Content...
- (Coming soon) Conda Package

# Test and CI

We will have comprehensive tests and CI setup, including code coverage analysis.

# Performance Benchmark

How fast is DynoRRT?

Benchmarking motion planning is challenging. In some algorithms, most of the time is spent in collision checking. Pinocchio and Hpp-fcl offer state-of-the-art performance for collision checking.

Let's evaluate only the motion planning algorithm:

**RRT:**
- Compared to OMPL, RRT is 3 times faster in a 3D environment without collisions using the same values of goal bias and max step size. (i.e., all compute time is spent only on nearest neighbor computations, interpolation, and memory allocation) - see test `t_bench_rrt` in `benchmark/main`.

**RRT star:**
- After 100 ms of compute time, we get a tree with 15x more samples and better cost.

**PRM star**
- TODO


# Build C++ Library and Python Package

Coming soon

# Roadmap


Release 0.1

- [ ] built graph in multicore? (parallel knn searches!)
- [ ] Pinocchio check edge in parallel
- [ ] Get the tests working again
- [ ] Implement good CI in Github
- [ ] Add the missing planner PRM\*, LazyPRM\*.
- [ ] Implement SST star and Kyno RRT
- [ ] Create a Conda package
- [ ] Add a video
- [ ] Evaluate in MotionBenchMaker
- [ ] Offer problem as two files (robot vs. environment)
- [ ] Integrate in Dynoplan
