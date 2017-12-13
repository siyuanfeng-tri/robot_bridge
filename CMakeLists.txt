cmake_minimum_required(VERSION 3.5.1)
project(robot_bridge)

find_package(drake CONFIG REQUIRED)

# to preserve path for the exectuable after install
#set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

include_directories(
  ${PROJECT_SOURCE_DIR}/include)
#link_directories(
#  $ENV{GUROBI_PATH}/lib)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_library(robot_bridge SHARED
  src/capsule.cc
  src/jacobian_ik.cc
  src/robot_bridge_common.cc
  src/motion_primitive.cc
  src/iiwa_controller.cc
  src/robot_bridge.cc)
#target_link_libraries(robot_bridge drake::drake gurobi75)
target_link_libraries(robot_bridge drake::drake gurobi75)

add_executable(test_robot_bridge src/test_robot_bridge.cc)
target_link_libraries(test_robot_bridge robot_bridge)

install(TARGETS robot_bridge
 DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

#install(TARGETS test_robot_bridge
# DESTINATION ${CMAKE_INSTALL_PREFIX})

install(DIRECTORY include/robot_bridge
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
