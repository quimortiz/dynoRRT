# I have followed the tutorial from here:
# https://pabloariasal.github.io/2018/02/19/its-time-to-do-cmake-right/

cmake_minimum_required(VERSION 3.5)
project(
  dynorrt
  VERSION 0.0.12
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

# NOTE: still linking error when I try to compile in the conda env from
# pinocchio.
find_package(Boost REQUIRED COMPONENTS unit_test_framework system filesystem
                                       serialization python)
# PATHS /usr/lib/x86_64-linux-gnu/
# )

find_package(Eigen3 REQUIRED)

find_package(pinocchio REQUIRED)
find_package(OpenMP REQUIRED) # WHY??
# find_package(octomap REQUIRED)
include(FindThreads)

option(BUILD_DYNOBENCH "" OFF)
message(STATUS "BUILD_DYNOBENCH ${BUILD_DYNOBENCH}")

if(BUILD_DYNOBENCH)
  FetchContent_Declare(
    dynobench
    GIT_REPOSITORY https://github.com/quimortiz/dynobench
    GIT_TAG 70f52b03199cdcae44e4bd49863ec57533138d7e)
  FetchContent_MakeAvailable(dynobench)
endif()

# if (BUILD_DYNOBENCH) add_subdirectory(dynobench) endif()

if(NOT TARGET nlohmann_json)
  FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
  FetchContent_MakeAvailable(json)
endif()

# TODO: WHY??
if(OPENMP_FOUND) #
  # If found, add OpenMP flags to the compiler
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message(FATAL_ERROR "OpenMP not found")
endif()

# Use this to avoid conflict with conda eigen find_package(Eigen3 3.3.7 REQUIRED
# EXACT)

option(BUILD_TESTS_RRT "" OFF)
option(BUILD_PYRRT "" OFF)

option(PIN_PYTHON_OBJECT "" OFF)



option(BUILD_BENCHMARK "" OFF)
# option(USE_ZLIBSTATIC "" OFF)
message(STATUS "BUILD_PYRRT ${BUILD_PYRRT}")
message(STATUS "BUILD_BENCHMARK ${BUILD_BENCHMARK}")
message(STATUS "BUILD_TESTS_RRT ${BUILD_TESTS_RRT}")
message(STATUS "PIN_PYTHON_OBJECT ${PIN_PYTHON_OBJECT}")
# message(STATUS "USE_ZLIBSTATIC ${USE_ZLIBSTATIC}")

include(GNUInstallDirs)
add_library(dynorrt INTERFACE)
add_library(dynorrt::dynorrt ALIAS dynorrt)

add_library(pin_col_manager SHARED src/pin_col_manager.cpp src/pin_ik_solver.cpp)
add_library(dynorrt::pin_col_manager ALIAS pin_col_manager)

target_include_directories(
  dynorrt INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(dynorrt INTERFACE Eigen3::Eigen magic_enum::magic_enum
                                        nlohmann_json::nlohmann_json)

target_compile_features(dynorrt INTERFACE cxx_std_17)

add_subdirectory(dynotree)

# set(CMAKE_CXX_STANDARD 17) set(CMAKE_CXX_STANDARD_REQUIRED On)

target_link_libraries(pin_col_manager dynorrt::dynorrt pinocchio::pinocchio toml11::toml11 OpenMP::OpenMP_CXX)

target_include_directories(pin_col_manager PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/thread-pool/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)


set_target_properties(pin_col_manager PROPERTIES POSITION_INDEPENDENT_CODE ON)

# TODO: add external flag!!

# set(LIB_MYLIB /usr/local/lib64/libzlibstatic.a)
# if(USE_ZLIBSTATIC)
#   set(LIB_MYLIB /usr/local/lib64/libzlibstatic.a)
# else()
#   set(LIB_MYLIB)
# endif()

# add_library(pin_with_zlib INTERFACE)
# target_link_libraries(pin_with_zlib INTERFACE pinocchio::pinocchio ${LIB_MYLIB})

# add_library(assimp_with_zlib INTERFACE) target_link_libraries(assimp_with_zlib
# INTERFACE /usr/local/lib64/libassimp.a ${LIB_MYLIB})

if(BUILD_PYRRT)
  add_subdirectory(bindings/python)
endif()

if(BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

if(BUILD_TESTS_RRT)

  # add_subdirectory(dynobench)

  add_executable(test_main test/test_main.cpp) 
  add_executable(test_pin test/test_pin.cpp)

  if (BUILD_DYNOBENCH)
  add_executable(test_dyno test/test_dyno.cpp)
  endif()

  # add_executable(play_toml test/play_toml.cpp)


  # add_executable(pin_col test/pin_col.cpp) add_executable(pin_ur5_example
  # test/pin_ur5_example.cpp) add_executable(pin_ur5_with_obs
  # test/pin_ur5_with_obs.cpp)


  # target_link_libraries(pin_col pin_with_zlib)
  # target_link_libraries(pin_ur5_example pin_with_zlib)
  # target_link_libraries(pin_ur5_with_obs pin_with_zlib)

  # target_link_libraries(
  #   play_toml
  #   PRIVATE dynorrt::dynorrt
  #           dynotree::dynotree
  #           Eigen3::Eigen
  #           magic_enum::magic_enum
  #           nlohmann_json::nlohmann_json
  #           Boost::unit_test_framework
  #           toml11::toml11)

  target_link_libraries(
    test_main
    PUBLIC Eigen3::Eigen
    PRIVATE dynorrt::dynorrt
            dynotree::dynotree
            Eigen3::Eigen
            magic_enum::magic_enum
            nlohmann_json::nlohmann_json
            Boost::unit_test_framework
            toml11::toml11
            pinocchio::pinocchio
            dynorrt::pin_col_manager
            # octomap::octomap
  )


  target_link_libraries(
    test_pin
    PUBLIC Eigen3::Eigen
    PRIVATE dynorrt::dynorrt
            dynotree::dynotree
            Eigen3::Eigen
            magic_enum::magic_enum
            nlohmann_json::nlohmann_json
            Boost::unit_test_framework
            toml11::toml11
            pinocchio::pinocchio
            dynorrt::pin_col_manager
            # octomap::octomap
  )




  if (BUILD_DYNOBENCH)

  target_link_libraries(
      test_dyno
    PUBLIC Eigen3::Eigen
    PRIVATE dynorrt::dynorrt
            dynotree::dynotree
            Eigen3::Eigen
            magic_enum::magic_enum
            nlohmann_json::nlohmann_json
            Boost::unit_test_framework
            toml11::toml11
            pinocchio::pinocchio
            dynorrt::pin_col_manager
            dynobench::dynobench
            # octomap::octomap
  )

  endif()




  # set_target_properties(
  #   test_dynorrt
  #   PROPERTIES CXX_STANDARD 17
  #              CXX_STANDARD_REQUIRED YES
  #              CXX_EXTENSIONS NO)

  # add_test(NAME test COMMAND test_dynorrt)
endif()

# install

install(TARGETS nlohmann_json
    EXPORT dynorrtTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS magic_enum
    EXPORT dynorrtTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})



install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export dynorrt targets for easy import in other projects
install(TARGETS dynorrt
        EXPORT dynorrtTargets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Installation and export for pin_col_manager compiled library
install(TARGETS pin_col_manager
        EXPORT dynorrtTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Install the exports as a CMake file
install(EXPORT dynorrtTargets
        FILE dynorrtTargets.cmake
        NAMESPACE dynorrt::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/dynorrt)

# Optionally create and install a config file for the project
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "dynorrtConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/dynorrtConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/dynorrtConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/dynorrt
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/dynorrtConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/dynorrtConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/dynorrt
)

if(BUILD_PYRRT)
# Add the install command for Python files


  find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Installing Python modules to: ${PYTHON_SITE_PACKAGES}")

# Other project configurations, add_subdirectory, etc.



# I can hardcode the python path
# set(PYTHON_SITE_PACKAGES "lib/python3.12/site-packages")

# find_package(Python3 REQUIRED COMPONENTS Interpreter)
# execute_process(
#     COMMAND "${Python3_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
#     OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )

message(STATUS "Python site-packages directory: ${PYTHON_SITE_PACKAGES}")

# Other configuration and build setup...

# install(DIRECTORY ${CMAKE_BINARY_DIR}/src/python/
#         DESTINATION ${PYTHON_SITE_PACKAGES}
#         FILES_MATCHING PATTERN "*.py"
#         PATTERN "*.so")

# Example install command using the determined site-packages directory
install(DIRECTORY ${CMAKE_BINARY_DIR}/bindings/python/
        DESTINATION ${PYTHON_SITE_PACKAGES}/pydynorrt
        FILES_MATCHING PATTERN "*.so")


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/python/pydynorrt/
        DESTINATION ${PYTHON_SITE_PACKAGES}/pydynorrt
        # FILES_MATCHING PATTERN "*.py" # NOTE: I also want to install all the data files
        )

# Install shared libraries (.so files) associated with the Python bindings
# install(DIRECTORY ${CMAKE_BINARY_DIR}/bindings/python/
#         DESTINATION ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_PACKAGES}
#         FILES_MATCHING PATTERN "*.so")
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
