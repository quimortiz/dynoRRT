# yum install -y eigen3-devel
yum install -y wget
yum install -y tar

cd /
mkdir deps
cd /deps
wget -O boost_1_85_0.tar.gz https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz
tar -xvf boost_1_85_0.tar.gz
cd boost_1_85_0
./bootstrap.sh
./b2 install -j8

cd /deps
git clone https://gitlab.com/libeigen/eigen
cd eigen
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cmake --install build

cd /deps
git clone https://github.com/leethomason/tinyxml2
cd tinyxml2
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-fPIC"
cmake --build build -j8
cmake --install build

cd /deps
git clone https://github.com/ros/console_bridge
cd console_bridge
rm -rf build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cmake --install build

cd /deps
git clone https://github.com/ros/urdfdom_headers
cd urdfdom_headers
rm -rf build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cmake --install build

cd /deps
git clone https://github.com/ros/urdfdom
cd urdfdom/
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cmake --install build

cd /deps
git clone https://github.com/OctoMap/octomap
cd octomap/
rm -rf build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cmake --install build

cd /deps
git clone https://github.com/assimp/assimp
cd assimp
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DASSIMP_BUILD_TESTS=0
cmake --build build -j8
cmake --install build

cd /deps
git clone https://github.com/humanoid-path-planner/hpp-fcl
cd hpp-fcl
rm -rf build
cmake -B build -S . -DBUILD_PYTHON_INTERFACE=0 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=0
cmake --build build -j8
cmake --install build

cd /deps
git clone https://github.com/stack-of-tasks/pinocchio
cd pinocchio
rm -rf build
cmake -B build -S .  -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_HPP_FCL_SUPPORT=1 -DBUILD_TESTING=0 -DBUILD_PYTHON_INTERFACE=0 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cmake --install build
