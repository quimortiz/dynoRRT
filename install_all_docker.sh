# cd boost/boost_1_71_0 && cd ../..
# ls


yum install -y wget
cd /etc/yum.repos.d/
wget https://download.opensuse.org/repositories/shells:fish:release:3/CentOS_7/shells:fish:release:3.repo
yum install -y fish

yum install -y yum-utils
yum-config-manager --add-repo=https://copr.fedorainfracloud.org/coprs/carlwgeorge/ripgrep/repo/epel-7/carlwgeorge-ripgrep-epel-7.repo
yum install -y ripgrep

curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim.appimage
chmod u+x nvim.appimage
./nvim.appimage --appimage-extract
./squashfs-root/AppRun --version
# Optional: exposing nvim globally.
mv squashfs-root /
ln -s /squashfs-root/AppRun /usr/bin/nvim

git clone https://github.com/nvim-lua/kickstart.nvim.git "${XDG_CONFIG_HOME:-$HOME/.config}"/nvim
# nvim --headless "+Lazy! sync" +qa




echo $PWD
yum install -y eigen3-devel
cd /io/dev_docker
cd boost/boost_1_71_0 && ./bootstrap.sh && ./b2 install
cd ../..
echo $PWD
cd tinyxml/build && make -j install && cd ../..
cd console_bridge/build && make -j  install && cd ../..
cd urdfdom_headers/build && make -j  install && cd ../..
cd urdfdom/build && make -j  install && cd ../..
cd assimp/build && make -j install && cd ../..
cd octomap/build && make -j install && cd ../..
cd hpp-fcl/build && make -j install && cd ../..
cd pinocchio/build && make -j install && cd ../..


# cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYRRT=1 -DBoost_USE_STATIC_LIBS=1 -DPYTHON_EXECUTABLE=/opt/python/cp38-cp38/bin/python  -DBoost_CHRONO_LIBRARY_RELEASE=/usr/local/lib/libboost_chrono.a ..
