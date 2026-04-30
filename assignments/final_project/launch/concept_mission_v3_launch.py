#!/usr/bin/env python3
"""
concept_mission_launch.py
=========================
Launches the two-phase concept terrain mission for SES 598 final project.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('final_project')
    model_path = os.path.join(pkg_share, 'models')

    # Gazebo model paths
    for var in ('GZ_SIM_MODEL_PATH', 'GZ_SIM_RESOURCE_PATH'):
        existing = os.environ.get(var, '')
        paths = [model_path]
        if existing:
            paths.append(existing)
        os.environ[var] = os.pathsep.join(paths)

    # CRITICAL: Exactly 6 space-separated values (x y z roll pitch yaw)
    os.environ['PX4_GZ_MODEL_POSE'] = '0.0 0.0 0.5 0.0 0.0 0.0'

    # PX4 SITL
    px4_sitl = ExecuteProcess(
        cmd=['make', 'px4_sitl', 'gz_x500_depth_mono'],
        cwd=os.path.expanduser('~/PX4-Autopilot'),
        output='screen'
    )

    # Spawn Mars terrain mesh
    spawn_terrain = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-file', os.path.join(model_path, 'terrain', 'model.sdf'),
            '-name', 'mars_terrain',
            '-x', '0', '-y', '0', '-z', '-1',
        ],
        output='screen'
    )

    # ROS-Gazebo bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='bridge',
        parameters=[{'use_sim_time': True}],
        arguments=[
            '/mono_camera@sensor_msgs/msg/Image@gz.msgs.Image',
            '/mono_camera@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/depth_camera@sensor_msgs/msg/Image@gz.msgs.Image',
            '/depth_camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked',
            '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
            '/model/x500_depth_mono_0/odometry_with_covariance'
            '@nav_msgs/msg/Odometry@gz.msgs.OdometryWithCovariance',
        ],
        remappings=[
            ('/mono_camera', '/drone/front_rgb'),
            ('/mono_camera', '/drone/front_rgb/camera_info'),
            ('/depth_camera', '/drone/front_depth'),
            ('/depth_camera/points', '/drone/front_depth/points'),
            ('/model/x500_depth_mono_0/odometry_with_covariance', '/fmu/out/vehicle_odometry'),
        ],
        output='screen'
    )

    # Concept terrain mission node
    concept_mission = Node(
        package='final_project',
        executable='concept_terrain_mission_v3',
        name='concept_terrain_mission',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )
    viz_node = Node(
        package='final_project',
        executable='mission_viz',
        name='mission_viz',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='True'),
        px4_sitl,
        TimerAction(period=2.0, actions=[spawn_terrain]),
        TimerAction(period=3.0, actions=[bridge]),
        TimerAction(period=5.0, actions=[concept_mission]),
        TimerAction(period=4.0, actions=[viz_node]),
    ])
