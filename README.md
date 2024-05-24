# DynoRRT

**Try it out online!**

<a target="_blank" href="https://colab.research.google.com/github/quimortiz/dynoRRT/blob/main/notebooks/tutorial0.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

DynoRRT is a C++/Python library for sampling-based motion planning using algorithms such as Rapidly Exploring Random Trees (RRT) and Probabilistic Roadmaps (PRM).

It delivers state-of-the-art performance with an easy-to-use and install Python interface. The installation is straightforward—no need for ROS, system-wide packages, MoveIt, or OMPL—just a simple `pip install` or `conda install`. It's significantly faster than OMPL and includes out-of-the-box integration with collision checking for robotics.

With DynoRRT, you can define and solve a motion planning problem within 60 seconds and solve it in milliseconds. Planning problems can be described using URDF files, through a collision function in Python, or a Pinocchio Model. Internally, we rely on Pinocchio and HPP-FCL for collision detection and forward kinematics calculations. In the pip package, these libraries are statically linked, so you don't need them at runtime—or it's fine if you have another version of these libraries. If you want to pass a Pinocchio model to the planner using the Python interface, you have to use the conda package or compile from source and install locally.

The Python package is created using pybind11, and the API is very similar to the C++ interface. Additionally, the Python package provides a couple of utilities to visualize the problems using Pinocchio and Meshcat, but you're free to use any viewer you prefer.

The library is currently in its alpha state. We are aiming for a public release in June. Feel free to open a GitHub issue or pull request!

```bash
conda install quimortiz::dynorrt -c conda-forge
```

or

```bash
pip3 install pydynorrt
```

## Tutorial

You can try it online! Run the `tutorial0` notebook in Google Colab (links at the top of the README).

## Examples and Continuous Integration

Check the GitHub CI [cpp-py](.github/workflows/cpp-py.yml) [conda](.github/workflows/conda.yml) to see how to compile the code and how to run tests and examples.

## Development

You can build the Python package using `setup.py` or using `cmake`.

With Cmake

```bash
mkdir build
cd build
cmake -DBUILD_EXAMPLES=ON -DBUILD_DYNOBENCH=ON -DBUILD_PYRRT=ON -DBUILD_BENCHMARK=OFF -DCMAKE_PREFIX_PATH="/opt/openrobots/"   -DBUILD_TESTS_RRT=1 ..
make
make install
```


With `setup.py` (adjust accordingly):

```bash
CMAKE_PREFIX_PATH=/opt/openrobots/ python3 -m build
cd dist
pip install pydynorrt-0.0.9-cp310-cp310-linux_x86_64.whl --force-reinstall
```

After changing C++, we can build Pybindings and copy by hand:

```bash
cp bindings/python/pydynorrt.cpython-310-x86_64-linux-gnu.so /home/quim/envs/mim/lib/python3.10/site-packages/pydynorrt
```

## Creating the conda package

The conda package is created in the separate repository [dynorrt-conda](https://github.com/quimortiz/dynorrt-conda).
[dynorrt-conda](https://github.com/quimortiz/dynorrt-conda) contains the recipe files to create the package from a chosen commit of this repository.

## Creating the pip Package

From the root of the repository:

```bash
docker run -it -v $(pwd):/io quay.io/pypa/manylinux2014_x86_64
```

Inside Docker:

```bash
bash /io/install_all_docker.sh
```

Then, I can create a custom build directory and:

```bash
bash ../build_cmd.sh
```

Or directly use the setup.py of Python to build all the wheels:

Run inside Docker, in the /io folder:

```bash
bash build-wheels.sh
```

## Planners

**Geometric Planners:**
- RRT
- RRT Connect
- Bidirectional RRT
- PRM
- BIT*
- LAZY PRM
- RRT*

**Kinodynamic Planners:**
- Kinodynamic RRT
- SST*

Looking for more planners? Check out our latest work in kinodynamic motion planning using Search And Trajectory Optimization. Our state-of-the-art motion planners for optimal kinodynamic motion planning are available [here](https://github.com/quimortiz/dynoplan).

## Collision and Robots

- Using Python bindings: You can define the collision function in Python.
- Robotics: Examples using Python Bindings of Pinocchio for collision checking.
- In C++: Flexibility to implement whatever you want.
- In Python: Similar flexibility, but it may be slower.
- C++ Pinocchio Interface: Define your problem using a URDF file.
- C++ Collision for Point robot in 2D and 3D and spheres/boxes (maybe using HPP-FCL, or just coding).

## How

 To Use the Code

- Python Package
- (Coming soon) C++: Easy integration into CMake build system. As submodule, Fetch Content...
- (Coming soon) Conda Package

## Test and CI

We will have comprehensive tests and CI setup, including code coverage analysis.

## Performance Benchmark

How fast is DynoRRT?

TODO

## Roadmap

Release 0.1

- [ ] Build graphs with multicore?
- [ ] Add the missing planners PRM*, LazyPRM*.
- [ ] Add a video.
- [ ] Evaluate in MotionBenchMaker.
- [ ] Offer problem as two files (robot vs. environment).
- [ ] Integrate in Dynoplan.
