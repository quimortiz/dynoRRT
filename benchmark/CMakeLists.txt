include(FetchContent)

# FetchContent_Declare( ompl GIT_REPOSITORY https://github.com/ompl/ompl GIT_TAG
# 1.6.0 )
#
# FetchContent_MakeAvailable(ompl)
find_package(ompl REQUIRED)

message(STATUS "OMPL_FOUND: ${OMPL_FOUND}")
message(STATUS "OMPL_INCLUDE_DIRS: ${OMPL_INCLUDE_DIRS}")
message(STATUS "OMPL_LIBRARIES: ${OMPL_LIBRARIES}")
message(STATUS "OMPL_VERSION: ${OMPL_VERSION}")

add_executable(benchmark main.cpp)
target_include_directories(benchmark PRIVATE ${OMPL_INCLUDE_DIRS})
target_link_libraries(
  benchmark PRIVATE dynotree::dynotree Eigen3::Eigen dynorrt::dynorrt
                    toml11::toml11 Boost::unit_test_framework ${OMPL_LIBRARIES})
