# dynoRRT

A simple and performant motion planning library for C++ and Python.
Install in seconds, solve in milliseconds!

<!-- ROS free, OMPL free, MOVE-it free. -->

Pre alpha state!

The code that uses Pinocchio is taken from:
https://github.com/ymontmarin/_tps_robotique


How fast is DynoRRT?
Benchmarking Motion planning is hard. In some algorithms, most of the time is spent in collision checking. Pinocchio and Hpp-FCL offer state of the art performance for collision checking.


Let's evaluate only the motion planning algorithm:


RRT
Compare to ompl, RRT is 3 times faster in a 3D environment without collisions
using same values of goal bias and max step size.
(i.e., all compute time is spent only on neaerest neighbour computations, interpolation and memory allocation) -- see test t_bench_rrt in benchmark/main.

RRT star
After 100 ms of compute time we get a tree with 15x more samples and better cost.

PRM star


# TODO's




# Roadmap to Release 0.1

- [x] Pinocchio Collision from URDF of environment
- [ ] Pin col checkig in python binding
- [ ] Example 1: UR5 Column
- [ ] Example 2: Panda? Box
- [ ] Example 3: Two Ur5
- [ ] Example 4: Flying Ball
- [ ] Script to run all planner on all problems
- [ ] Pinocchio collision binary instead of distance. Faster?
- [ ] PRM* as optimal planner?
- [ ] SST star and Kyno RRT
- [ ] AO RRT?
- [ ] Options to use K-nearest neigh in planners graph
- [ ] Pip Package
- [ ] Conda package
- [ ] Tutorial with Jupyter lab, tested in CI.
- [ ] Video

# Planners

* RRT
* RRT Connect
* Bidirectional RRT
* PRM
* Incremental PRM (similar BIT\*)
* LAZY PRM
* RRT\*
* COMING SOON: PRM\*

# Collision and Robots

* Using Python bindings: You can define the collision function in PYTHON
* Robotics: Examples using Python Bindings of Pinocchio for collision checking
* In C++: Flexibility to implement whatever you want.
* In Python: Same, but it will be slower.
* COMING SOON: C++ Pinocchio Interface
* COMING SOON: C++ Collision for Point robot in 2D and 3D and spheres/boxes (maybe using hpp-fcl?, or just coding )


# How To use the code

* COMING SOON: Python Package
* COMING SOON: C++: Easy integration in CMAKE build system. As submodule, Fetch Content...
* COMING SOON: Conda Pacakge

# Benchmark

* COMING SOON:

# Test and CI

* We will have good tests and CI
