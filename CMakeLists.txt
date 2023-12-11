# I have followed the tutorial from here:
# https://pabloariasal.github.io/2018/02/19/its-time-to-do-cmake-right/

cmake_minimum_required(VERSION 3.5)
project(
  dynoRRT
  VERSION 0.0.1
  LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(magic_enum
                     GIT_REPOSITORY https://github.com/Neargye/magic_enum)
FetchContent_MakeAvailable(magic_enum)

# TODO: I don't like this here! but toml requires the standard to be set
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

FetchContent_Declare(toml11 GIT_REPOSITORY https://github.com/ToruNiina/toml11)

FetchContent_MakeAvailable(toml11)

# FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

# NOTE: still linking error when I try to compile in the conda env from
# pinocchio.
find_package(
  Boost REQUIRED COMPONENTS unit_test_framework system filesystem serialization
                            # PATHS /usr/lib/x86_64-linux-gnu/
)

find_package(Eigen3 REQUIRED)

find_package(pinocchio REQUIRED)

# Use this to avoid conflict with conda eigen find_package(Eigen3 3.3.7 REQUIRED
# EXACT)

option(BUILD_EXAMPLES OFF)
option(BUILD_PYRRT OFF)
option(BUILD_BENCHMARK OFF)
option(USE_ZLIBSTATIC OFF)
message(STATUS "BUILD_PYRRT ${BUILD_PYRRT}")
message(STATUS "BUILD_BENCHMARK ${BUILD_BENCHMARK}")
message(STATUS "BUILD_EXAMPLES ${BUILD_EXAMPLES}")
message(STATUS "USE_ZLIBSTATIC ${USE_ZLIBSTATIC}")

include(GNUInstallDirs)
add_library(dynorrt INTERFACE)
add_library(dynorrt::dynorrt ALIAS dynorrt)

add_library(pin_col_manager src/pin_col_manager.cpp)
add_library(dynorrt::pin_col_manager ALIAS pin_col_manager)

target_include_directories(
  dynorrt INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(dynorrt INTERFACE Eigen3::Eigen magic_enum::magic_enum
                                        nlohmann_json::nlohmann_json)

target_compile_features(dynorrt INTERFACE cxx_std_17)

add_subdirectory(dynotree)

# set(CMAKE_CXX_STANDARD 17) set(CMAKE_CXX_STANDARD_REQUIRED On)

target_link_libraries(pin_col_manager dynorrt::dynorrt pin_with_zlib)

set_target_properties(pin_col_manager PROPERTIES POSITION_INDEPENDENT_CODE ON)

# TODO: add external flag!!

# set(LIB_MYLIB /usr/local/lib64/libzlibstatic.a)
if(USE_ZLIBSTATIC)
  set(LIB_MYLIB /usr/local/lib64/libzlibstatic.a)
else()
  set(LIB_MYLIB)
endif()

add_library(pin_with_zlib INTERFACE)
target_link_libraries(pin_with_zlib INTERFACE pinocchio::pinocchio ${LIB_MYLIB})

# add_library(assimp_with_zlib INTERFACE) target_link_libraries(assimp_with_zlib
# INTERFACE /usr/local/lib64/libassimp.a ${LIB_MYLIB})

if(BUILD_PYRRT)
  add_subdirectory(bindings/python)
endif()

if(BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

if(BUILD_EXAMPLES)
  add_executable(main_test test/main.cpp test/test_dev.cpp)

  add_executable(pin_col test/pin_col.cpp)
  add_executable(pin_ur5_example test/pin_ur5_example.cpp)
  add_executable(pin_ur5_with_obs test/pin_ur5_with_obs.cpp)
  add_executable(play_toml test/play_toml.cpp)

  target_link_libraries(pin_col pin_with_zlib)
  target_link_libraries(pin_ur5_example pin_with_zlib)
  target_link_libraries(pin_ur5_with_obs pin_with_zlib)

  target_link_libraries(
    play_toml
    PRIVATE dynorrt::dynorrt
            dynotree::dynotree
            Eigen3::Eigen
            magic_enum::magic_enum
            nlohmann_json::nlohmann_json
            Boost::unit_test_framework
            toml11::toml11)

  target_link_libraries(
    main_test
    PUBLIC Eigen3::Eigen
    PRIVATE dynorrt::dynorrt
            dynotree::dynotree
            Eigen3::Eigen
            magic_enum::magic_enum
            nlohmann_json::nlohmann_json
            Boost::unit_test_framework
            toml11::toml11
            pinocchio::pinocchio
            dynorrt::pin_col_manager)

  set_target_properties(
    main
    PROPERTIES CXX_STANDARD 17
               CXX_STANDARD_REQUIRED YES
               CXX_EXTENSIONS NO)

  add_test(NAME test COMMAND main)
endif()

# dynobench PUBLIC fcl yaml-cpp Boost::program_options Boost::serialization
# Boost::stacktrace_basic ${CMAKE_DL_LIBS} PUBLIC nlohmann_json::nlohmann_json)
#
# target_link_libraries( check_trajectory PUBLIC dynobench PRIVATE fcl yaml-cpp
# Boost::program_options Boost::serialization)
#
# target_link_libraries( check_trajectory_multirobot PUBLIC dynobench PRIVATE
# fcl yaml-cpp Boost::program_options Boost::serialization)
#
# target_link_libraries(robot_python PUBLIC Eigen3::Eigen dynobench fcl)
#
# # Installation instructions
#
# # # include(GNUInstallDirs) set(INSTALL_CONFIGDIR #
# ${CMAKE_INSTALL_LIBDIR}/cmake/dynobench) # # install( TARGETS dynobench EXPORT
# dynobench-targets LIBRARY DESTINATION # ${CMAKE_INSTALL_LIBDIR} ARCHIVE
# DESTINATION ${CMAKE_INSTALL_LIBDIR}) # # install(DIRECTORY include/
# DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}) # # # Export the targets to a script
# install( EXPORT dynobench-targets FILE # dynobenchTargets.cmake NAMESPACE
# dynobench:: DESTINATION ${INSTALL_CONFIGDIR}) # # # Create a
# ConfigVersion.cmake file include(CMakePackageConfigHelpers) #
# write_basic_package_version_file( #
# ${CMAKE_CURRENT_BINARY_DIR}/dynobenchConfigVersion.cmake VERSION #
# ${PROJECT_VERSION} COMPATIBILITY AnyNewerVersion) # #
# configure_package_config_file( #
# ${CMAKE_CURRENT_LIST_DIR}/cmake/dynobenchConfig.cmake.in #
# ${CMAKE_CURRENT_BINARY_DIR}/dynobenchConfig.cmake INSTALL_DESTINATION #
# ${INSTALL_CONFIGDIR}) # # # Install the config, configversion and custom find
# modules install(FILES # ${CMAKE_CURRENT_BINARY_DIR}/dynobenchConfig.cmake #
# ${CMAKE_CURRENT_BINARY_DIR}/dynobenchConfigVersion.cmake DESTINATION #
# ${INSTALL_CONFIGDIR}) # # # #
# ##############################################################################
# # # Exporting from the build tree # #
# configure_file(${CMAKE_CURRENT_LIST_DIR}/cmake/FindRapidJSON.cmake # #
# ${CMAKE_CURRENT_BINARY_DIR}/FindRapidJSON.cmake COPYONLY) # # export( EXPORT
# dynobench-targets FILE # ${CMAKE_CURRENT_BINARY_DIR}/dynobenchTargets.cmake
# NAMESPACE dynobench::) # # # Register package in the User Package Registry
# export(PACKAGE dynobench)
#
# # Add test
#
# if(NOT DEFINED DISABLE_DYNOBENCH_TESTS) enable_testing()
# add_subdirectory(test) endif()
