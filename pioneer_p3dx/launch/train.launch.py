#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction

def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    launch_file_dir = os.path.join(get_package_share_directory('pioneer_p3dx'), 'launch')
    world = os.path.join(get_package_share_directory('pioneer_p3dx'), 'worlds', 'env.world')
    rviz_file = os.path.join(get_package_share_directory('pioneer_p3dx'), 'rviz', 'agent.rviz')


    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='Set to "false" to run headless.'),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world, 'use_sim_time': use_sim_time}.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')]),
            condition=IfCondition(LaunchConfiguration('gui'))
        ),

        Node(
            package='pioneer_p3dx',
            executable='train.py',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_file_dir, '/robot_state_publisher.launch.py']),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', rviz_file],
                    parameters=[{'use_sim_time': use_sim_time}],
                    output='screen'
                )
            ]
        ),
    ])