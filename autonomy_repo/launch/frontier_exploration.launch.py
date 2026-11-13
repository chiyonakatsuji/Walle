#!/usr/bin/env python3
"""
Launch file for autonomous frontier exploration.
Does NOT launch the simulator or the hardware bringup. Start those separately.

VERIFY: Compare the RViz / relay / state_publisher / navigator entries below
against your HW2 navigator.launch.py and copy any differences from yours.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

PACKAGE = "autonomy_repo"   # CHANGE if your package has a different name


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),

        # ---------------- RViz (requirement 4: see the map build in real time) ----------------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare("asl_tb3_sim"), "launch", "rviz.launch.py"])
            ),
            launch_arguments={
                "config": PathJoinSubstitution([FindPackageShare(PACKAGE), "rviz", "default.rviz"]),
                "use_sim_time": use_sim_time,
            }.items(),
        ),

        # ---------------- Helpers from asl_tb3_lib (same as HW2) ----------------
        Node(
            executable="rviz_goal_relay.py",
            package="asl_tb3_lib",
            parameters=[{"output_channel": "/cmd_nav"}],
        ),
        Node(
            executable="state_publisher.py",
            package="asl_tb3_lib",
        ),

        # ---------------- Your navigator from HW2 ----------------
        Node(
            executable="navigator.py",
            package=PACKAGE,
            parameters=[{"use_sim_time": use_sim_time}],
        ),

        # ---------------- Frontier explorer (delayed so map + navigator are up) ----------------
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    executable="frontier_explorer.py",
                    package=PACKAGE,
                    output="screen",
                    parameters=[{
                        "use_sim_time": use_sim_time,
                        "window_size": 13,
                    }],
                ),
            ],
        ),
    ])
