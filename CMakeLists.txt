# I have followed the tutorial from here:
# https://pabloariasal.github.io/2018/02/19/its-time-to-do-cmake-right/

cmake_minimum_required(VERSION 3.5)
project(
  dynoRRT
  VERSION 0.0.1
  LANGUAGES CXX)


include(FetchContent)



FetchContent_Declare(magic_enum GIT_REPOSITORY https://github.com/Neargye/magic_enum)
FetchContent_MakeAvailable(magic_enum)

# FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

find_package(Boost REQUIRED COMPONENTS unit_test_framework system filesystem
                                       serialization)


# # find_package(pinocchio REQUIRED)
# find_package(Boost REQUIRED COMPONENTS program_options serialization
#                                        stacktrace_basic)
# find_package(fcl REQUIRED)
# find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(pybind11 REQUIRED)

# add_subdirectory(deps/pybind11)
# add_subdirectory(deps/json)

add_subdirectory(dynotree)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

add_executable(main ./src/main.cpp)

target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_test(NAME test COMMAND main)



target_link_libraries(
  main 
  PRIVATE dynotree::dynotree
  Eigen3::Eigen
  magic_enum::magic_enum
  nlohmann_json::nlohmann_json
  Boost::unit_test_framework
  )
#   dynobench
#   PUBLIC fcl yaml-cpp Boost::program_options Boost::serialization
#          Boost::stacktrace_basic ${CMAKE_DL_LIBS}
#   PUBLIC nlohmann_json::nlohmann_json)
#
# target_link_libraries(
#   check_trajectory
#   PUBLIC dynobench
#   PRIVATE fcl yaml-cpp Boost::program_options Boost::serialization)
#
# target_link_libraries(
#   check_trajectory_multirobot
#   PUBLIC dynobench
#   PRIVATE fcl yaml-cpp Boost::program_options Boost::serialization)
#
# target_link_libraries(robot_python PUBLIC Eigen3::Eigen dynobench fcl)
#
# # Installation instructions
#
# #
# # include(GNUInstallDirs) set(INSTALL_CONFIGDIR
# # ${CMAKE_INSTALL_LIBDIR}/cmake/dynobench)
# #
# # install( TARGETS dynobench EXPORT dynobench-targets LIBRARY DESTINATION
# # ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
# #
# # install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# #
# # # Export the targets to a script install( EXPORT dynobench-targets FILE
# # dynobenchTargets.cmake NAMESPACE dynobench:: DESTINATION ${INSTALL_CONFIGDIR})
# #
# # # Create a ConfigVersion.cmake file include(CMakePackageConfigHelpers)
# # write_basic_package_version_file(
# # ${CMAKE_CURRENT_BINARY_DIR}/dynobenchConfigVersion.cmake VERSION
# # ${PROJECT_VERSION} COMPATIBILITY AnyNewerVersion)
# #
# # configure_package_config_file(
# # ${CMAKE_CURRENT_LIST_DIR}/cmake/dynobenchConfig.cmake.in
# # ${CMAKE_CURRENT_BINARY_DIR}/dynobenchConfig.cmake INSTALL_DESTINATION
# # ${INSTALL_CONFIGDIR})
# #
# # # Install the config, configversion and custom find modules install(FILES
# # ${CMAKE_CURRENT_BINARY_DIR}/dynobenchConfig.cmake
# # ${CMAKE_CURRENT_BINARY_DIR}/dynobenchConfigVersion.cmake DESTINATION
# # ${INSTALL_CONFIGDIR})
# #
# # #
# # ##############################################################################
# # # Exporting from the build tree #
# # configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/FindRapidJSON.cmake #
# # ${CMAKE_CURRENT_BINARY_DIR}/FindRapidJSON.cmake COPYONLY)
# #
# # export( EXPORT dynobench-targets FILE
# # ${CMAKE_CURRENT_BINARY_DIR}/dynobenchTargets.cmake NAMESPACE dynobench::)
# #
# # # Register package in the User Package Registry export(PACKAGE dynobench)
#
# # Add test
#
# if(NOT DEFINED DISABLE_DYNOBENCH_TESTS)
#   enable_testing()
#   add_subdirectory(test)
# endif()
