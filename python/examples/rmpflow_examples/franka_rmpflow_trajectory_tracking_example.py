#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This example demonstrates how to generate and track a motion plan for the Franka Emika Panda arm.

The motion plan is generated using Jt-RRT and the trajectory is tracked with an RMPflow policy.
"""

# Standard Library
import os
import sys
from time import sleep

# Third Party
import numpy as np

# cuMotion
import cumotion
try:
    from cumotion_vis.visualizer import FrankaVisualization, RenderableType, Visualizer

    ENABLE_VIS = True
except ImportError:
    print("Visualizer not installed. Disabling visualization.")
    ENABLE_VIS = False

# Set cuMotion root directory
CUMOTION_ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def cumotion_print_status(success):
    """Print the final status of the example."""
    if success:
        print("CUMOTION EXAMPLE COMPLETED SUCCESSFULLY")
    else:
        print("CUMOTION EXAMPLE FAILED")


if __name__ == '__main__':
    # Parse --verbose flag from sys.argv. If not provided, default to False.
    verbose = False
    for arg in sys.argv[1:]:
        if arg.startswith('--verbose'):
            if '=' in arg:
                value = arg.split('=', 1)[1].strip().lower()
                verbose = value in ('1', 'true', 'yes', 'y', 'on')
            else:
                verbose = True
    np.set_printoptions(linewidth=200)

    # ==============================================================================================
    # Load all data to represent robot (Franka) and configure motion generators
    # (motion planner and RMPflow).

    # Set directory for RMPflow configuration, motion planner configuration and robot description
    # YAML files.
    config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')

    # Set absolute path to planning configuration for Franka.
    planning_config_path = os.path.join(config_path, "franka_planner_config.yaml")

    # Set absolute path to RMPflow configuration for Franka.
    rmpflow_config_path = os.path.join(
        config_path, 'franka_rmpflow_config_without_point_cloud.yaml')

    # Set absolute path to the XRDF for Franka.
    xrdf_path = os.path.join(config_path, 'franka.xrdf')

    # Set absolute path to URDF file for Franka.
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'franka.urdf')

    # Load robot description.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

    # ==============================================================================================
    # Create world with obstacles. Views into this world will be used by both the motion planner
    # (for global planning) and RMPflow (for trajectory following).

    # Create world.
    world = cumotion.create_world()

    # Add box obstacles in front of robot arm.
    # Each pair represents the center position and side lengths of a box.
    position_length_pairs = [
        [np.array([0.55, 0.0, 0.6]), np.array([0.4, 1.5, 0.02])],
        [np.array([0.35, -0.5, 0.4]), np.array([0.02, 0.5, 0.4])],
        [np.array([0.35, 0.5, 0.4]), np.array([0.02, 0.5, 0.4])]]

    for pair in position_length_pairs:
        box_obstacle_pose = cumotion.Pose3.from_translation(pair[0])
        box = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
        box.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, pair[1])
        world.add_obstacle(box, box_obstacle_pose)

    # ==============================================================================================
    # Set initial configuration (in configuration space) and target (in task space) for task.

    # Set initial configuration.
    q0 = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.5, 0.0])

    # Choose frame to represent end effector. Any link name from the URDF loaded above can be used
    # here (though the position target defined below is chosen to be reachable by a frame with
    # origin near the gripper of the robot).
    tool_frame_name = 'right_gripper'

    # Set task space target for end effector.
    translation_target = np.array([0.67, 0.0, 0.10])
    print("Translation target: ", translation_target)

    # ==============================================================================================
    # Create motion planner for Franka.

    # Create a world view to be used by the motion planner.
    planner_world_view = world.add_world_view()

    # Create motion planner.
    config = cumotion.create_motion_planner_config_from_file(planning_config_path,
                                                             robot_description,
                                                             tool_frame_name,
                                                             world.add_world_view())
    planner = cumotion.create_motion_planner(config)

    # ==============================================================================================
    # Create RMPflow for Franka.

    # Create a world view to be used by RMPflow. This world can be synchronized independently from
    # the view used by the motion planner, but both will look to the original `world` for ground
    # truth.
    rmpflow_world_view = world.add_world_view()

    # Create RMPflow configuration.
    rmpflow_config = cumotion.create_rmpflow_config_from_file(rmpflow_config_path,
                                                              robot_description,
                                                              rmpflow_world_view)

    # Create RMPflow policy.
    rmpflow = cumotion.create_rmpflow(rmpflow_config)

    # Add a task-space target frame and set a position-only target.
    rmpflow.add_target_frame(tool_frame_name)
    rmpflow.set_position_target(tool_frame_name, translation_target)

    # ==============================================================================================
    # Create visualization with Franka arm and obstacles from world.

    if ENABLE_VIS:
        # Initialize visualization
        visualizer = Visualizer()

        # Add robot arm visualization to scene
        mesh_folder = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'meshes')
        franka_visualization = FrankaVisualization(robot_description, mesh_folder, visualizer, q0)

        # Add visualization marker for end effector target position to scene.
        target_config = {
            'position': translation_target,
            'radius': 0.05,
            'color': [1.0, 0.5, 0.0]  # orange
        }
        target_handle = 'target'
        visualizer.add(RenderableType.MARKER, target_handle, target_config)

        # Add obstacles to visualization.
        for index, pair in enumerate(position_length_pairs):
            box_config = {
                'position': pair[0],
                'side_lengths': pair[1],
                'color': [0.0, 0.0, 1.0]  # blue
            }
            visualizer.add(RenderableType.BOX, 'box_' + str(index), box_config)

    # ==============================================================================================
    # Demonstrate using only RMPflow with end effector position target (with no global planning).
    # This is expected to fail, with the arm getting stuck on top of the table with nothing guiding
    # it to back up and enter the table from below.

    # Set initial state and acceleration for robot.
    cspace_position = q0
    cspace_velocity = np.zeros(7)
    cspace_accel = np.zeros(7)

    dt = 0.005  # seconds
    total_time = 2.0  # seconds
    time = 0.0  # seconds

    while time < total_time:
        # Evaluate acceleration from c-space state.
        rmpflow.eval_accel(cspace_position, cspace_velocity, cspace_accel)

        # Update position and velocity with Euler integration.
        cspace_position += dt * cspace_velocity
        cspace_velocity += dt * cspace_accel

        # Update Franka in the visualization window.
        if ENABLE_VIS:
            franka_visualization.set_joint_positions(cspace_position)
            visualizer.update()

            # Pause to approximate real-time visualization.
            sleep(dt)

        if verbose:
            print("c-space position: ", cspace_position)

        time += dt

    print("Final c-space position: ", cspace_position)

    # ==============================================================================================
    # Demonstrate using only the motion planner with end effector position target
    # (using Jacobian-transpose RRT). This is expected to succeed but result in a somewhat jagged,
    # unnatural path.

    # The `generate_interpolated_path` option is used to create a path with (approximately)
    # equidistant nodes, with c-space length between nodes equal to the step size set in the motion
    # planner configuration.
    generate_interpolated_path = True
    planning_results = planner.plan_to_translation_target(q0,
                                                          translation_target,
                                                          generate_interpolated_path)

    # Visualize path to task space target (if found).
    if ENABLE_VIS and planning_results.path_found:
        for q in planning_results.interpolated_path:
            # Update Franka in the visualization window.
            franka_visualization.set_joint_positions(q)

            visualizer.update()

            # The generated path is purely a sequence of feasible positions, so the sleep time
            # chosen here is arbitrary.
            sleep(0.1)

        # Add a bit of extra time at the target for visualization purposes.
        extra_frames = 15
        for frame_index in range(extra_frames):
            sleep(0.1)
            visualizer.update()

    # ==============================================================================================
    # Finally, we demonstrate how the path from the motion planner can be used as waypoints for
    # RMPflow to generate a smooth path that reaches the target.

    # Reset initial state and acceleration for robot.
    cspace_position = q0
    cspace_velocity = np.zeros(7)
    cspace_accel = np.zeros(7)

    # We will use `cumotion.Kinematics` to compute forward kinematics from c-space to the
    # end effector frame.
    kinematics = robot_description.kinematics()

    if planning_results.path_found:
        # For this tracking, we are using a sparse set of waypoints from the non-interpolated path
        # from the motion planner.
        for q_target in planning_results.path:
            # For each c-space waypoint, we will convert to an end effector target.
            # In this case we are discarding the orientation and only tracking the position from the
            # motion planned path.
            target_pose = kinematics.pose(q_target, tool_frame_name)
            rmpflow.set_position_target(tool_frame_name, target_pose.translation)

            # RMPflow will direct Franka towards the position of `target_pose` until the end
            # effector is within a specified radius. In this case, the radius is hand-selected to
            # work for this particular example.
            termination_radius = 0.15

            # Set the current end effector pose for testing against `target_pose`. This value will
            # be updated for each RMPflow evaluation.
            x_pose = kinematics.pose(cspace_position, tool_frame_name)

            while np.linalg.norm(x_pose.translation - target_pose.translation) > termination_radius:
                # Evaluate acceleration from c-space state.
                rmpflow.eval_accel(cspace_position, cspace_velocity, cspace_accel)

                # Update position and velocity with Euler integration.
                cspace_position += dt * cspace_velocity
                cspace_velocity += dt * cspace_accel

                # Update end effector pose.
                x_pose = kinematics.pose(cspace_position, tool_frame_name)

                # Update Franka in the visualization window.
                if ENABLE_VIS:
                    franka_visualization.set_joint_positions(cspace_position)
                    visualizer.update()

                    # Pause to approximate real-time visualization.
                    sleep(dt)

                if verbose:
                    print("x_pose: ", x_pose.translation)

                time += dt

        print("Final tool position: ", x_pose.translation)

        # Add some time at the end for RMPflow to converge to final target.
        extra_time = 4.0  # seconds
        time = 0.0
        while time < extra_time:
            # Evaluate acceleration from c-space state.
            rmpflow.eval_accel(cspace_position, cspace_velocity, cspace_accel)

            # Update position and velocity with Euler integration.
            cspace_position += dt * cspace_velocity
            cspace_velocity += dt * cspace_accel

            # Update Franka in the visualization window.
            if ENABLE_VIS:
                franka_visualization.set_joint_positions(cspace_position)
                visualizer.update()

                # Pause to approximate real-time visualization.
                sleep(dt)

            if verbose:
                print("c-space position: ", cspace_position)

            time += dt

        print("Final c-space position: ", cspace_position)
        x_pose = kinematics.pose(cspace_position, tool_frame_name)
        print("Final tool position (after 4 sec): ", x_pose.translation)
        error = np.linalg.norm(x_pose.translation - translation_target)
        print("Final error: ", error)
        success = error < termination_radius
        cumotion_print_status(success)
