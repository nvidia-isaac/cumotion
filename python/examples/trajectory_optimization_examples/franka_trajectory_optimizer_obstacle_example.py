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

"""Python script demonstrating collision-free trajectory optimization with cuMotion.

This example demonstrates how to generate a trajectory for the Franka Panda robot with obstacles
using the cuMotion library. This builds upon the basic trajectory example by adding collision
avoidance capabilities through geometric obstacle representation.
"""

# Standard Library
import os
import time

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


def main():
    """Run the trajectory optimization example with obstacle avoidance.

    This function demonstrates a complete workflow for collision-free trajectory optimization:
    1. Load robot description and kinematics.
    2. Create world environment with obstacles.
    3. Configure trajectory optimizer with collision checking.
    4. Plan collision-free trajectory around obstacles.
    5. Visualize the resulting motion with obstacles.
    """
    # ==============================================================================================
    # Load Robot Description
    # ==============================================================================================

    np.set_printoptions(linewidth=np.inf)
    print("Loading Franka robot description")

    # The cuMotion library uses Extended Robot Description Format (XRDF) and Universal Robot
    # Description Format (URDF) to describe robots. URDF defines the robot's kinematics, and
    # XRDF extends it with additional information such as semantic labeling of configuration
    # space, acceleration limits, jerk limits, and collision spheres. For additional details,
    # see: https://nvidia-isaac-ros.github.io/concepts/manipulation/xrdf.html
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'franka.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'franka.urdf')

    # The `load_robot_from_file()` function loads the robot description containing the semantic and
    # kinematic information from the XRDF and URDF files.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

    # The `kinematics()` function returns the robot kinematics object, which provides functions to
    # compute the robot's forward kinematics, Jacobians, frame transformations, and configuration
    # space limits.
    kinematics = robot_description.kinematics()

    # ==============================================================================================
    # Create cuMotion World with Obstacles
    # ==============================================================================================

    print("Creating world with obstacles")

    # The `create_world()` function creates an empty world that will be populated with obstacles.
    # A World represents a collection of obstacles that the robot must avoid during motion
    # planning. This provides the environment context for collision-free trajectory optimization.
    world = cumotion.create_world()

    # Create a cuboid obstacle.
    obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)

    # Set the side lengths for the cuboid obstacle.
    cuboid_side_lengths = np.array([0.2, 0.2, 0.15])
    obstacle.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, cuboid_side_lengths)

    # Define the pose of the cuboid obstacle. The pose is specified in the world coordinate frame
    # (i.e., the base frame of the robot).
    cuboid_position = np.array([0.3, 0.15, 0.3])
    cuboid_orientation = cumotion.Rotation3.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 3)
    cuboid_pose = cumotion.Pose3(cuboid_orientation, cuboid_position)

    # Add the configured obstacle to the world at the specified pose.
    world.add_obstacle(obstacle, cuboid_pose)
    print(f"Added cuboid obstacle at {cuboid_position} with size {cuboid_side_lengths}")

    # Calling `add_world_view()` creates a view into the world that can be used for collision checks
    # and distance evaluations. Each world view maintains a static snapshot of the world until it
    # is updated (via a call to `world_view.update()`). Upon creation, the `world_view` contains
    # the latest snapshot of `world` with all presently enabled obstacles.
    world_view = world.add_world_view()

    # ==============================================================================================
    # Create Trajectory Optimizer
    # ==============================================================================================

    print("Creating trajectory optimizer")

    # Use `right_gripper` frame as the tool frame for Franka. This specifies which robot frame
    # will be controlled and constrained during trajectory optimization. The tool frame is
    # typically the end effector or gripper that needs to reach target positions.
    tool_frame_name = "right_gripper"

    # Create a default trajectory optimizer config for this example. This function combines
    # default optimization parameters with the robot description, tool frame, and world view
    # to create a complete configuration for trajectory optimization. The default parameters
    # include settings for collision checking, optimization convergence, and trajectory resolution.
    # The world_view enables collision-aware planning.
    trajectory_optimizer_config = cumotion.create_default_trajectory_optimizer_config(
        robot_description,
        tool_frame_name,
        world_view
    )

    # Create the trajectory optimizer using the configuration. This instantiates the numerical
    # optimization engine that will generate collision-free trajectories using a combination
    # of particle-based optimization (PBO) and L-BFGS methods.
    trajectory_optimizer = cumotion.create_trajectory_optimizer(trajectory_optimizer_config)

    # ==============================================================================================
    # Define initial c-space position
    # ==============================================================================================

    # Use the robot's default configuration as the initial c-space (i.e., "configuration space")
    # position. This retrieves the default c-space position defined in the robot
    # description, which specifies the joint angles for all actively controlled joints.
    q_initial = robot_description.default_cspace_configuration()

    # ==============================================================================================
    # Create task-space constraints
    # ==============================================================================================

    # Set the target position in world coordinates (i.e., the base frame of the robot). This
    # defines the desired position for the origin of the tool frame. The coordinates are in meters.
    target_position = np.array([0.3, 0.1, 0.1])

    # Create a translation constraint that requires the tool frame origin to reach the specified
    # target position at the end of the trajectory. The constraint is active at termination but not
    # along the path.
    translation_constraint = cumotion.TrajectoryOptimizer.TranslationConstraint.target(
        target_position
    )

    # Create an orientation constraint that does not restrict the orientation of the tool frame
    # along the path or at termination.
    orientation_constraint = cumotion.TrajectoryOptimizer.OrientationConstraint.none()

    # Create a task-space target combining translation and orientation constraints.
    task_space_target = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_constraint,
        orientation_constraint
    )

    # ==============================================================================================
    # Plan Trajectory
    # ==============================================================================================

    print("Planning trajectory with obstacle avoidance")

    # Plan a collision-free trajectory from the initial c-space position to the task-space target.
    # This is implemented with GPU-accelerated numerical optimization algorithms.
    results = trajectory_optimizer.plan_to_task_space_target(q_initial, task_space_target)

    # ==============================================================================================
    # Check Results and Visualize
    # ==============================================================================================

    # Check if trajectory optimization was successful. The status indicates whether a
    # valid collision-free trajectory was found that satisfies all constraints and limits
    # while avoiding obstacles.
    success = False
    if results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS:
        # Extract the optimized trajectory. This is a time-parameterized path through c-space.
        trajectory = results.trajectory()

        # Get the time domain of the trajectory. The domain specifies the valid time
        # range [lower, upper] over which the trajectory is defined and can be evaluated.
        # The lower bound of the domain will always be set to zero, such that the upper
        # bound of the domain represents the time span of the trajectory.
        domain = trajectory.domain()

        print("Trajectory generation successful")
        print("Trajectory properties:")
        print(f"Trajectory duration: {domain.span():.2f} seconds")

        # Evaluate trajectory at start and end times to show c-space positions. The `eval()`
        # function computes the c-space position at any time within the domain.
        print(f"Trajectory start position: {trajectory.eval(domain.lower)}")
        print(f"Trajectory end position: {trajectory.eval(domain.upper)}")

        # Check final position error by computing forward kinematics. This verifies that
        # the final c-space position actually places the tool frame at the target.
        # while avoiding obstacles.
        q_final = trajectory.eval(domain.upper)
        final_position = kinematics.position(q_final, tool_frame_name)
        position_error = np.linalg.norm(final_position - target_position)
        print(f"Final task-space position error: {position_error * 1000:.2f} mm")
        if position_error < 1e-3:
            success = True

        cumotion_print_status(success)

        # ==========================================================================================
        # Visualize the Trajectory with Obstacles
        # ==========================================================================================

        if not ENABLE_VIS:
            return

        print("Starting Open3D visualization")

        # Create Open3D-based visualizer for rendering robot motion and obstacles.
        visualizer = Visualizer()

        # Create Franka visualization with mesh rendering.
        mesh_folder = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'meshes')
        franka_vis = FrankaVisualization(
            robot_description,
            mesh_folder,
            visualizer,
            q_initial
        )

        # Add obstacle visualization to show the geometry that the robot must avoid.
        obstacle_config = {
            'position': [0.0, 0.0, 0.0],
            'side_lengths': cuboid_side_lengths,
            'color': [0.8, 0.2, 0.2]
        }
        visualizer.add(RenderableType.BOX, "obstacle", obstacle_config)

        visualizer.set_rotation("obstacle", cuboid_orientation)

        visualizer.set_position("obstacle", cuboid_position)

        # Add target marker to visualize the goal position. Since only a position target is
        # specified, a sphere is used to visualize the target (as orientation is not constrained).
        target_config = {
            'position': target_position,
            'radius': 0.03,
            'color': [1.0, 0.5, 0.0]  # Orange color.
        }
        visualizer.add(RenderableType.MARKER, "target", target_config)

        # Add coordinate frame at origin for reference.
        frame_config = {
            'size': 0.2,
            'position': [0.0, 0.0, 0.0]
        }
        visualizer.add(RenderableType.COORDINATE_FRAME, "origin_frame", frame_config)

        # Animate the final trajectory at 20 FPS. This creates a real-time playback of
        # the optimized trajectory, showing how the robot moves from start to target.
        print("Animating trajectory with obstacle avoidance, close window to exit")
        dt = 0.05  # 50ms time steps for 20 FPS.

        # Main animation loop that continues until user closes the visualization window.
        while visualizer.is_active():
            # Iterate through trajectory time domain with specified time step.
            for t in np.arange(domain.lower, domain.upper + dt, dt):
                if not visualizer.is_active():
                    break

                # Evaluate trajectory at current time to get joint positions.
                q_t = trajectory.eval(min(t, domain.upper))

                # Update robot visualization with current joint positions.
                franka_vis.set_joint_positions(q_t)

                # Update the visualization.
                visualizer.update()

                # Sleep for the specified time step.
                time.sleep(dt)

            time.sleep(1.0)

        visualizer.close()

    else:
        # Report trajectory optimization failure. This could occur if no collision-free
        # path exists given the obstacle configuration and constraints.
        status = results.status()
        print(f"Failed to find trajectory. Status: {status}")


if __name__ == "__main__":
    main()
