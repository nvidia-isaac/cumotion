#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Python script demonstrating collision-free inverse kinematics (IK) with cuMotion.

This example demonstrates how to generate IK solutions for the Franka Panda robot with obstacles
using the cuMotion library.
"""

# Standard Library
import os

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


def visualize_franka_ik_solutions(robot_description,
                                  cspace_positions,
                                  target_pose,
                                  cuboid_side_lengths=None,
                                  cuboid_pose=None):
    """Launch a visualization window to display IK solutions for Franka.

    The visualization includes a set of coordinate axes for `target_pose` and an independent Franka
    visualization for each c-space position in `cspace_positions`.

    An optional cuboid obstacle may be specified by setting `cuboid_side_lengths` and
    `cuboid_pose`, where the pose is relative to the base of the robot.

    The visualization window remains open until manually closed.
    """
    if not ENABLE_VIS:
        return

    # Initialize visualization.
    visualizer = Visualizer()

    # Add Franka visualizations.
    mesh_folder = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'meshes')
    for index, cspace_position in enumerate(cspace_positions):
        # NOTE: A unique name is required for each robot arm.
        name = "franka_" + str(index)
        FrankaVisualization(robot_description, mesh_folder, visualizer, cspace_position, name)

    # Add coordinate axes for the target pose.
    target_config = {
        'size': 0.2,
    }
    visualizer.add(RenderableType.COORDINATE_FRAME, "target", target_config)
    visualizer.set_pose("target", target_pose)

    # Optionally add cuboid visualization.
    if cuboid_side_lengths is not None:
        assert cuboid_pose is not None
        cuboid_config = {
            'side_lengths': cuboid_side_lengths,
            'color': [0.0, 0.0, 1.0],
            'position': [0.0, 0.0, 0.0]
        }
        visualizer.add(RenderableType.BOX, "cuboid", cuboid_config)
        visualizer.set_pose("cuboid", cuboid_pose)

    visualizer.fit_camera_to_scene()

    # Update visualization until manually closed.
    while visualizer.is_active():
        visualizer.update()

    # Close visualization.
    visualizer.close()


def main():
    """Run the collision-free IK example."""
    # Set absolute path to URDF file for Franka.
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'franka.urdf')

    # Set absolute path to the XRDF for Franka.
    #
    # XRDF extends the URDF with additional information such as semantic labeling of configuration
    # space, acceleration limits, jerk limits, and collision spheres. For additional details,
    # see: https://nvidia-isaac-ros.github.io/concepts/manipulation/xrdf.html
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'franka.xrdf')

    # Load robot description..
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

    # Set end effector frame for Franka.
    end_effector_frame_name = "right_gripper"

    # The `create_world()` function creates an empty world that will be populated with obstacles.
    # A `World` represents a collection of obstacles that the robot must avoid.
    world = cumotion.create_world()

    # Calling `add_world_view()` creates a view into the world that can be used for collision checks
    # and distance evaluations. Each world view maintains a static snapshot of the world until it
    # is updated (via a call to `world_view.update()`).
    world_view = world.add_world_view()

    # Create configuration for collision-free inverse kinematics (IK).
    config = cumotion.create_default_collision_free_ik_solver_config(robot_description,
                                                                     end_effector_frame_name,
                                                                     world_view)

    # Increase the number of seeds so that statistics help ensure the check that there are more
    # solutions in an empty world than one with an obstacle.
    config.set_param("num_seeds", 36)

    # Create a collision-free IK solver.
    ik_solver = cumotion.create_collision_free_ik_solver(config)

    # Define a target pose for the end effector.
    target_translation = np.array([0.5, 0.2, 0.6])
    target_orientation = cumotion.Rotation3.from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.5 * np.pi)
    target_pose = cumotion.Pose3(target_orientation, target_translation)

    # Use the target pose to created task-space constraints for the IK solver.
    translation_constraint = (
        cumotion.CollisionFreeIkSolver.TranslationConstraint.target(target_translation))
    orientation_constraint = (
        cumotion.CollisionFreeIkSolver.OrientationConstraint.target(target_orientation))
    task_space_target = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(translation_constraint,
                                                                       orientation_constraint)

    # Solve IK, expecting multiple distinct c-space solutions.
    results = ik_solver.solve(task_space_target)
    success = results.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    print("Collision-free IK with no obstacles:")
    print("  - Found {} distinct IK solutions.".format(len(results.cspace_positions())))
    print("  - Static visualization will remain open until manually closed.")

    # Visualize c-space solutions with no cuboid obstacle added.
    visualize_franka_ik_solutions(robot_description, results.cspace_positions(), target_pose)

    # Add a cuboid obstacle to the `world`.
    cuboid_side_lengths = np.array([0.5, 0.35, 0.8])
    cuboid_position = np.array([0.4, -0.2, 0.4])
    cuboid = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
    cuboid.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, cuboid_side_lengths)
    world.add_obstacle(cuboid, cumotion.Pose3.from_translation(cuboid_position))

    # Updating `world_view` updates the snapshot of the `world` used by `ik_solver`. With this
    # update, `ik_solver` will actively try to avoid the newly added cuboid obstacle.
    world_view.update()

    # Solve IK, expecting multiple distinct c-space solutions.
    results_with_cuboid = ik_solver.solve(task_space_target)
    if results_with_cuboid.status() != cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS:
        success = False
    print("Collision-free IK with cuboid obstacle:")
    print("  - Found {} distinct IK solutions.".format(len(results_with_cuboid.cspace_positions())))
    print("  - Static visualization will remain open until manually closed.")

    # Visualize c-space solutions with cuboid obstacle added.
    visualize_franka_ik_solutions(robot_description,
                                  results_with_cuboid.cspace_positions(),
                                  target_pose,
                                  cuboid_side_lengths,
                                  cumotion.Pose3.from_translation(cuboid_position))

    if len(results_with_cuboid.cspace_positions()) >= len(results.cspace_positions()):
        success = False
    cumotion_print_status(success)


if __name__ == "__main__":
    main()
