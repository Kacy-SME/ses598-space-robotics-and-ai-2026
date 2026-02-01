cmake_minimum_required(VERSION 3.8)
project(first_order_boustrophedon_navigator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ControllerMetrics.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Install Python script directly
install(PROGRAMS
  first_order_boustrophedon_navigator/boustrophedon_controller.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME boustrophedon_controller
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
