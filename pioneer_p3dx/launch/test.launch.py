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
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file_name = 'env.world'
    world = os.path.join(get_package_share_directory('dream'), 'worlds', world_file_name)
    launch_file_dir = os.path.join(get_package_share_directory('dream'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    rviz_file = os.path.join(get_package_share_directory('dream'), 'launch', 'agent.rviz')
    gui_arg = DeclareLaunchArgument('gui', default_value='true', description='Set to "false" to run headless.')

    return LaunchDescription([
        gui_arg,

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world}.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')),
            condition=IfCondition(LaunchConfiguration('gui'))
        ),

        Node(package='dream',
             executable='test.py',
             output='screen'
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_file_dir, '/robot_state_publisher.launch.py']),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        # Node(
        #     # namespace='robot_3',
        #     package='slam_toolbox',
        #     executable='async_slam_toolbox_node',
        #     name='slam_toolbox',
        #     output='screen',
        #     parameters=[
        #         get_package_share_directory("slam_toolbox") + '/config/mapper_params_online_async.yaml',
        #         {'use_sim_time': use_sim_time}
        #     ],
        # ),

        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', rviz_file],
                    output='screen'
                )
            ]
        ),
    ])
