cmake_minimum_required(VERSION 3.22)
project(halcyon_core)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external/pybind11)
include_directories(external/)


pybind11_add_module(halcyon_core core/halcyon_core.cpp)
target_link_libraries(halcyon_core PRIVATE pybind11::module)
