find_package(Eigen3 3.3 REQUIRED)

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11/
  GIT_TAG v2.11.1)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(pydynorrt py.cpp)
target_link_libraries(
  pydynorrt
  PRIVATE 
  Boost::python
# /opt/openrobots/lib/python3.10/site-packages/pinocchio/pinocchio_pywrap.cpython-310-x86_64-linux-gnu.so
  dynorrt::dynorrt dynotree::dynotree Eigen3::Eigen toml11::toml11
          pinocchio::pinocchio dynorrt::pin_col_manager)

set_target_properties(
  pydynorrt
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED YES
             CXX_EXTENSIONS NO
             POSITION_INDEPENDENT_CODE ON)
