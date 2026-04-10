#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES.
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

"""This example demonstrates how to generate and execute an RMPflow policy for Franka."""

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

    # Set directory for RMPflow configuration and robot description YAML files.
    config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')

    # Set absolute path to RMPflow configuration for Franka.
    rmpflow_config_path = os.path.join(
        config_path, 'franka_rmpflow_config_without_point_cloud.yaml')

    # Set absolute path to the XRDF for Franka.
    xrdf_path = os.path.join(config_path, 'franka.xrdf')

    # Set absolute path to URDF file for Franka.
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'franka.urdf')

    # Set end effector frame for Franka.
    end_effector_frame_name = 'right_gripper'

    # Load robot description.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

    # Print some basic information about the robot to console.
    print('Number of c-space coordinates: {}'.format(robot_description.num_cspace_coords()))
    print('C-space names: ')
    for i in range(0, robot_description.num_cspace_coords()):
        print('  [{}] {}'.format(i, robot_description.cspace_coord_name(i)))

    # Create world for obstacles.
    world = cumotion.create_world()
    world_view = world.add_world_view()

    # Create RMPflow configuration.
    rmpflow_config = cumotion.create_rmpflow_config_from_file(rmpflow_config_path,
                                                              robot_description,
                                                              world_view)

    # Create RMPflow policy.
    rmpflow = cumotion.create_rmpflow(rmpflow_config)

    # Add spherical obstacle in front of robot arm.
    sphere_obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere_obstacle_radius = 0.05
    sphere_obstacle.set_attribute(cumotion.Obstacle.Attribute.RADIUS, sphere_obstacle_radius)
    sphere_obstacle_pose = cumotion.Pose3.from_translation(np.array([0.3, 0.0, 0.6]))
    world.add_obstacle(sphere_obstacle, sphere_obstacle_pose)

    # Add box obstacle in front of robot arm.
    box_obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
    box_obstacle_length = 0.1
    box_obstacle.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS,
                               box_obstacle_length * np.ones(3))
    box_obstacle_pose = cumotion.Pose3.from_translation(np.array([0.4, 0.0, 0.2]))
    world.add_obstacle(box_obstacle, box_obstacle_pose)

    # Update RMPflow world view after adding obstacles.
    world_view.update()

    # Set end effector position target.
    rmpflow.add_target_frame(end_effector_frame_name)
    end_effector_position_target = np.array([0.8, 0.0, 0.35])
    rmpflow.set_position_target(end_effector_frame_name, end_effector_position_target)

    # Set initial state and acceleration for robot.
    cspace_position = np.zeros(7)
    cspace_velocity = np.zeros(7)
    cspace_accel = np.zeros(7)

    # Set timing for policy execution.
    total_time = 60.0  # seconds
    timestep = 0.01  # seconds
    time = 0.0  # seconds

    if ENABLE_VIS:
        # Initialize visualization
        visualizer = Visualizer()

        # Add robot arm visualization to scene
        mesh_folder = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'meshes')
        franka_visualization = FrankaVisualization(
            robot_description, mesh_folder, visualizer, cspace_position)

        # Add visualization marker for end effector target position to scene.
        # Note: The visualization handle is saved so that the position can be updated later.
        target_config = {
            'position': end_effector_position_target,
            'radius': 0.05,
            'color': [1.0, 0.5, 0.0]  # orange
        }
        target_handle = 'target'
        visualizer.add(RenderableType.MARKER,
                       target_handle,
                       target_config)

        # Add visualization marker for obstacle sphere to scene.
        sphere_config = {
            'position': sphere_obstacle_pose.translation,
            'radius': sphere_obstacle_radius,
            'color': [0.0, 0.0, 1.0]  # blue
        }
        visualizer.add(RenderableType.MARKER,
                       'sphere',
                       sphere_config)

        # Add visualization box for obstacle box to scene.
        box_config = {
            'position': box_obstacle_pose.translation,
            'side_lengths': np.array(
                [box_obstacle_length, box_obstacle_length, box_obstacle_length]),
            'color': [0.0, 0.0, 1.0]  # blue
        }
        visualizer.add(RenderableType.BOX,
                       'box',
                       box_config)

    # Execute RMPflow policy for each timestep, set state with Euler integration and update
    # visualization.
    while time < total_time:
        # Update target position.
        end_effector_position_target[1] = 0.5 * np.sin(0.5 * time)
        rmpflow.set_position_target(end_effector_frame_name, end_effector_position_target)

        # Evaluate acceleration from c-space state.
        rmpflow.eval_accel(cspace_position, cspace_velocity, cspace_accel)

        # Update world view to reflect any changes to obstacles. For cases where obstacles have NOT
        # changed, this call is very low overhead.
        world_view.update()

        # Update position and velocity with Euler integration.
        cspace_position += timestep * cspace_velocity
        cspace_velocity += timestep * cspace_accel

        # If the visualization window has not been manually closed, update visualization and add
        # delay to approximate real-time visualization.
        if ENABLE_VIS and visualizer.is_active():
            visualizer.set_position(target_handle, end_effector_position_target)
            franka_visualization.set_joint_positions(cspace_position)
            visualizer.update()
            sleep(timestep)

        if verbose:
            print("c-space position: ", cspace_position)

        time += timestep

    np.set_printoptions(linewidth=200)
    expected_final_cspace_position = np.array(
        [-0.45364672, 1.21257119, -0.34068849, -0.60333264, -0.35049334, 2.9215992, 0.7503792])
    print("Final c-space position: ", cspace_position)
    success = np.allclose(cspace_position, expected_final_cspace_position, atol=1e-6)
    cumotion_print_status(success)
