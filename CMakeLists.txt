cmake_minimum_required(VERSION 3.5.1)
project(robot_bridge)

find_package(drake CONFIG REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/include)
link_directories(
  $ENV{GUROBI_PATH}/lib)

add_library(robot_bridge SHARED
  src/capsule.cc
  src/jacobian_ik.cc
  src/robot_bridge_common.cc
  src/motion_primitive.cc
  src/iiwa_controller.cc
  src/robot_bridge.cc)
target_link_libraries(robot_bridge drake::drake libgurobi75.so)

install(TARGETS robot_bridge
 DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(DIRECTORY include/robot_bridge
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
