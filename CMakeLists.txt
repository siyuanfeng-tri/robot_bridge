cmake_minimum_required(VERSION 3.5.1)
project(robot_bridge)

find_package(drake CONFIG REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/include)

add_definitions(-std=c++1y -g -Wall -O2 -fPIC)

add_library(robot_bridge
  src/capsule.cc
  src/jacobian_ik.cc
  src/robot_bridge_common.cc
  src/motion_primitive.cc
  src/iiwa_controller.cc
  src/robot_bridge.cc)
target_link_libraries(robot_bridge drake::drake libgurobi70.so)
