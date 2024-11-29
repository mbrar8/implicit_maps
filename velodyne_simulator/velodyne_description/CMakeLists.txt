cmake_minimum_required(VERSION 3.27.2)
project(velodyne_description)

set(CMAKE_WARN_DEVELOPER OFF)

find_package(ament_cmake REQUIRED)
find_package(gazebo REQUIRED)
find_package(gazebo_ros_pkgs REQUIRED)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

ament_export_dependencies(gazebo_ros_pkgs)

ament_package()

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY meshes/
  DESTINATION share/${PROJECT_NAME}/meshes
)

install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

install(DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

install(DIRECTORY world/
  DESTINATION share/${PROJECT_NAME}/world
)

install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)
