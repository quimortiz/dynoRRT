cmake_minimum_required(VERSION 3.26)
include(FetchContent)


project(hello_dynorrt)
find_package(dynorrt REQUIRED)
find_package(Eigen3 REQUIRED)

FetchContent_Declare(magic_enum
                     GIT_REPOSITORY https://github.com/Neargye/magic_enum)
FetchContent_MakeAvailable(magic_enum)

if(NOT TARGET nlohmann_json)
  FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
  FetchContent_MakeAvailable(json)
endif()



add_executable(hello_dynorrt main.cc)
target_link_libraries(hello_dynorrt dynorrt::dynorrt magic_enum::magic_enum Eigen3::Eigen nlohmann_json::nlohmann_json)





