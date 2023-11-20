# dynoRRT

A simple and performant motion planning library for C++ and Python.
Install in seconds, solve in milliseconds!

Demo: Pip install and Solve Motion Planning with state-of-the-art performance (2 minutes real time screencast)

Pre alpha state!

The code that uses Pinocchio is taken from:
https://github.com/ymontmarin/_tps_robotique and
https://github.com/Gepetto/supaero2022


# Planners

* RRT
* RRT Connect
* Bidirectional RRT
* PRM (Optionally, Check edges using Lazy Astar, Similar to BIT*)
* LAZY PRM
* COMING SOON: RRT\*
* COMING SOON: AO-RRT
* COMING SOON: PRM\*
* COMING SOON: Sparse?

Kinodynamic Planners (Think if can just reuse code from above?)
* Kinodynamic RRT
* SST*
* AO-Kinodynamic RRT

More Planners?
Check our latest work in kinodynamic motion planning using Search And Trajectory Optimization.
Our recent state-of-the art motion planners optimal kinodynamic motion planning are in,
https://github.com/quimortiz/dynoplan


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
* Code Coverage

# Roadmap
