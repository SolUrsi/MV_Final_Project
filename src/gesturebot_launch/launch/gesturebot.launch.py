import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    start_rviz = LaunchConfiguration('start_rviz')
    start_gesture_detector = LaunchConfiguration('start_gesture_detector')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_rviz',
            default_value='false',
            description='Start rviz alongside gazebo'
        ),
        DeclareLaunchArgument(
            'start_gesture_detector',
            default_value='true',
            description='Start the gesture detector node'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('turtlebot3_manipulation_moveit_config'),
                    'launch',
                    'move_group.launch.py'
                )
            ),
            launch_arguments={'use_sim': use_sim_time}.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('turtlebot3_manipulation_moveit_config'),
                    'launch',
                    'moveit_rviz.launch.py'
                )
            ),
            condition=IfCondition(start_rviz)
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('turtlebot3_manipulation_moveit_config'),
                    'launch',
                    'servo.launch.py'
                )
            ),
            launch_arguments={'use_sim': use_sim_time}.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('turtlebot3_manipulation_gazebo'),
                    'launch',
                    'gazebo.launch.py'
                )
            ),
            launch_arguments={'start_rviz': 'False'}.items(),
            condition=IfCondition(use_sim_time)
        ),
        Node(
            package='gesturebot_gestures',
            executable='gesture_detector',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
            condition=IfCondition(start_gesture_detector)
        ),
        Node(
            package='gesturebot_manipulation',
            executable='manipulation_handler',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        )
    ])
