# TODO

find_package(Eigen3 3.3 REQUIRED)

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11/
  GIT_TAG v2.11.1)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(pydynorrt py.cpp)
target_link_libraries(pydynorrt PRIVATE dynorrt::dynorrt dynotree::dynotree
                                        Eigen3::Eigen)
set_property(TARGET pydynorrt PROPERTY POSITION_INDEPENDENT_CODE ON)
