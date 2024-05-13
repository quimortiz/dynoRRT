cmake_minimum_required(VERSION 3.26)
include(FetchContent)


project(hello_dynorrt)
find_package(dynorrt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(OpenMP REQUIRED)

FetchContent_Declare(magic_enum
                     GIT_REPOSITORY https://github.com/Neargye/magic_enum)
FetchContent_MakeAvailable(magic_enum)

if(NOT TARGET nlohmann_json)
  FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
  FetchContent_MakeAvailable(json)
endif()



add_executable(main main.cc)
add_executable(main_pin main_pin.cc)


target_link_libraries(main dynorrt::dynorrt magic_enum::magic_enum Eigen3::Eigen nlohmann_json::nlohmann_json)


target_link_libraries( main_pin pinocchio::pinocchio dynorrt::dynorrt dynorrt::pin_col_manager magic_enum::magic_enum Eigen3::Eigen nlohmann_json::nlohmann_json)





