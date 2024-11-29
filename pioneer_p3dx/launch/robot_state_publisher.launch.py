#!/usr/bin/env python3

import os

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # PioneerP3DX URDF
    # pioneer_urdf = os.path.join(
    #     get_package_share_directory('pioneer_p3dx'),
    #     'urdf', 'td_robot.urdf')
    
    turtlebot_urdf = os.path.join(
        get_package_share_directory('turtlebot3_description'),
        'urdf', 'turtlebot3_waffle.urdf')
    
    # with open(pioneer_urdf, 'r') as infp:
    #     pioneer_robot_desc = infp.read()

    with open(turtlebot_urdf, 'r') as infp:
        turtlebot_robot_desc = infp.read()

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation (Gazebo) clock if true'),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='pioneer_state_publisher',
            output='screen',
            # parameters=[{'use_sim_time': use_sim_time, 'robot_description': pioneer_robot_desc,}],
            # arguments=[pioneer_urdf]
            parameters=[{'use_sim_time': use_sim_time, 'robot_description': turtlebot_robot_desc,}],
            arguments=[turtlebot_urdf]
        ),
    ])