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

"""Python script demonstrating collision-free trajectory optimization with rotated SDF obstacles.

This example demonstrates how to create a SDF (Signed Distance Field) cuboid obstacle
and use it for trajectory optimization with the Franka Panda robot. SDFs provide a more
flexible and continuous representation of obstacle geometry compared to primitive shapes.
"""

# Standard Library
import os
import time

# Third Party
import numpy as np

# cuMotion
import cumotion
try:
    from cumotion_vis.visualizer import SdfVisualizer, FrankaVisualization, RenderableType, \
        Visualizer

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


def create_sdf_obstacle_from_rotated_cuboid(
        cuboid_center, cuboid_size, rotation,
        num_voxels_x, num_voxels_y, num_voxels_z, voxel_size,
        host_precision, device_precision):
    """Create an SDF obstacle from a rotated cuboid.

    This function creates a temporary world with a primitive cuboid obstacle and generates
    an SDF grid from it. The SDF grid is then used to create an SDF obstacle that can be
    added to the main world.
    """
    # Create a temporary world to house the primitive cuboid obstacle for SDF generation.
    # This temporary world isolates the SDF computation from the main trajectory optimization world.
    temp_world = cumotion.create_world()

    # Create a primitive CUBOID obstacle using cuMotion's built-in geometry types.
    # Primitive obstacles provide exact analytical distance computation for SDF generation.
    cuboid_obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
    cuboid_obstacle.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, cuboid_size)

    # Combine rotation and translation into a single pose transformation for the cuboid.
    # The pose defines both the spatial position and orientation of the obstacle in the world.
    cuboid_pose = cumotion.Pose3(rotation, cuboid_center)

    # Add the configured cuboid to the temporary world at the specified pose.
    # This makes the obstacle available for distance queries through the world's collision system.
    temp_world.add_obstacle(cuboid_obstacle, cuboid_pose)

    # Create a world view from the temporary world to enable distance queries.
    world_view = temp_world.add_world_view()

    # Create a world inspector from the world view.
    world_inspector = cumotion.create_world_inspector(world_view)

    # Update the world view via its handle, which also updates `world_inspector`.
    world_view.update()

    # Allocate memory for the SDF grid using double precision for maximum accuracy.
    # The SDF will store signed distance values: negative inside obstacles, positive outside.
    sdf_data = np.empty((num_voxels_x, num_voxels_y, num_voxels_z), dtype=np.float64)

    # Iterate through every voxel in the 3D grid to compute signed distance values.
    # Each voxel stores the minimum distance to any obstacle in the temporary world.
    for i in range(num_voxels_x):
        for j in range(num_voxels_y):
            for k in range(num_voxels_z):
                # Convert voxel indices to world coordinates by scaling and offsetting.
                # The +0.5 offset centers the query point within each voxel cell.
                voxel_position = np.array([i, j, k]) * voxel_size + voxel_size * 0.5

                # Query the minimum distance from this point to any enabled obstacle.
                min_distance = world_inspector.min_distance(voxel_position)
                sdf_data[i, j, k] = min_distance

    # Create a new SDF obstacle to hold the computed distance field data.
    # SDF obstacles provide continuous distance gradients for smooth trajectory optimization.
    sdf_obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.SDF)

    # Configure the SDF grid structure with dimensions, resolution, and precision settings.
    # This defines the 3D voxel grid that will store the computed signed distance values.
    sdf_obstacle.set_attribute(cumotion.Obstacle.Attribute.GRID,
                               cumotion.Obstacle.Grid(num_voxels_x, num_voxels_y, num_voxels_z,
                                                      voxel_size, host_precision,
                                                      device_precision))

    return sdf_obstacle, sdf_data


def main():
    """Run the trajectory optimization example with rotated SDF obstacle avoidance.

    This function demonstrates a complete workflow for collision-free trajectory optimization
    using SDF obstacles:
    1. Load robot description and kinematics.
    2. Create world environment with SDF cuboid obstacle.
    3. Configure trajectory optimizer with collision checking.
    4. Plan collision-free trajectory around rotated SDF obstacle.
    5. Visualize the planned trajectory with SDF visualization.
    """
    # ==============================================================================================
    # Load Robot Description
    # ==============================================================================================

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
    # Create cuMotion World with Rotated SDF Obstacles
    # ==============================================================================================

    print("Creating world with rotated SDF cuboid obstacle")

    # The `create_world()` function creates an empty world that will be populated with SDF
    # obstacles. A World represents a collection of obstacles that the robot must avoid during
    # motion planning. This provides the environment context for collision-free trajectory
    # optimization.
    world = cumotion.create_world()

    # SDF grid parameters define the resolution and precision of the signed distance field.
    # Higher resolution provides more accurate collision detection but increases memory usage.
    num_voxels_x = 40
    num_voxels_y = 40
    num_voxels_z = 40
    voxel_size = 0.02  # 2 cm voxels provide good balance of accuracy and efficiency
    host_precision = cumotion.Obstacle.GridPrecision.FLOAT
    device_precision = cumotion.Obstacle.GridPrecision.FLOAT

    # Define cuboid properties within the SDF grid coordinate system. The generated SDF obstacle
    # will be the SDF that is generated based on this cuboid obstacle.
    cuboid_center_in_grid = np.array([0.4, 0.4, 0.3])
    cuboid_size = np.array([0.2, 0.2, 0.15])

    # Define rotation for the cuboid obstacle.
    rotation_angle = np.pi / 3  # 60 degree rotation
    rotation_axis = np.array([0.0, 0.0, 1.0])  # Rotate around Z-axis
    cuboid_rotation = cumotion.Rotation3.from_axis_angle(rotation_axis, rotation_angle)

    # Create SDF obstacle and populate the cuMotion world with it.
    # This creates a temporary world with a rotated cuboid, generates SDF data from it,
    # and returns a configured SDF obstacle that can be added to the main world. In real-world
    # applications, the SDF obstacle could be generated from a 3D reconstruction library such
    # as nvblox.
    sdf_obstacle, sdf_data = create_sdf_obstacle_from_rotated_cuboid(
        cuboid_center_in_grid, cuboid_size, cuboid_rotation,
        num_voxels_x, num_voxels_y, num_voxels_z, voxel_size,
        host_precision, device_precision
    )

    # Add SDF obstacle to world and get handle for further operations.
    # The handle allows us to update the SDF data and pose after creation.
    sdf_handle = world.add_obstacle(sdf_obstacle)

    # Set the SDF grid values from the computed distance field. This transfers the
    # SDF data from host memory to the GPU for efficient collision queries.
    world.set_sdf_grid_values_from_host(sdf_handle, sdf_data)

    # Position the SDF obstacle in the world coordinate system. This transforms the
    # SDF grid coordinate system to place the obstacle at the desired world location.
    world_obstacle_position = np.array([0.3, 0.15, 0.3])
    sdf_world_pose = cumotion.Pose3.from_translation(
        world_obstacle_position - cuboid_center_in_grid
    )
    world.set_pose(sdf_handle, sdf_world_pose)

    # Calling `add_world_view()` creates a view into the world that can be used for collision checks
    # and distance evaluations. Each world view maintains a static snapshot of the world until it
    # is updated (via a call to `world_view.update()`). Upon creation, the `world_view` contains
    # the latest snapshot of `world` with all presently enabled obstacles.
    world_view = world.add_world_view()

    # ==============================================================================================
    # Create Trajectory Optimizer
    # ==============================================================================================

    np.set_printoptions(linewidth=np.inf)
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

    print("Planning trajectory with SDF obstacle avoidance")

    # Plan a collision-free trajectory from the initial c-space position to the task-space target.
    # This is implemented with GPU-accelerated numerical optimization algorithms.
    results = trajectory_optimizer.plan_to_task_space_target(q_initial, task_space_target)

    # ==============================================================================================
    # Check Results and Visualize
    # ==============================================================================================

    # Check if trajectory optimization was successful. The status indicates whether a
    # valid collision-free trajectory was found that satisfies all constraints and limits
    # while avoiding the SDF obstacle.
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
        # the final c-space position actually places the tool frame at the target
        # while maintaining collision-free operation with the SDF obstacle.
        q_final = trajectory.eval(domain.upper)
        final_position = kinematics.position(q_final, tool_frame_name)
        position_error = np.linalg.norm(final_position - target_position)
        print(f"Final task-space position error: {position_error * 1000:.2f} mm")

        if position_error < 1e-3:
            success = True

        cumotion_print_status(success)

        # ==========================================================================================
        # Create Visualization
        # ==========================================================================================

        if not ENABLE_VIS:
            return

        print("Creating visualization")

        # Create Open3D-based visualizer for rendering robot motion and SDF obstacles. This provides
        # a 3D visualization environment for displaying the robot, SDF obstacles, and trajectory,
        # allowing users to see how the robot navigates around complex obstacle geometries.
        visualizer = Visualizer()

        # Create Franka robot visualization with mesh rendering. This loads the robot's 3D mesh
        # geometry and creates renderable objects for each link, enabling realistic
        # visualization of the robot's appearance and motion.
        mesh_folder = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'meshes')
        franka_vis = FrankaVisualization(
            robot_description,
            mesh_folder,
            visualizer,
            q_initial
        )

        # Add target marker to visualize the goal position. This creates a visual marker
        # at the target location so users can see where the robot is trying to reach
        # and understand the spatial relationship between start, target, and SDF obstacle.
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

        # Create SDF visualizer to generate wireframe voxel representation of the obstacle.
        # This provides a visual representation of the signed distance field, showing
        # the continuous obstacle representation used for collision detection.
        obstacle_configs = [
            {
                'type': 'cube',
                'position': world_obstacle_position,
                'size': max(cuboid_size)  # Use maximum dimension for visualization bounds
            }
        ]

        # Initialize SDF visualizer with world view and obstacle configuration.
        # This creates a specialized visualizer for rendering signed distance fields
        # as wireframe voxel grids that show obstacle boundaries.
        sdf_visualizer = SdfVisualizer(
            world_view=world_view,
            obstacle_configs=obstacle_configs,
            bounds=(-1.0, 1.0),  # SDF value range for visualization
            resolution=voxel_size  # Match the SDF grid resolution
        )

        # Generate SDF wireframe visualization by querying the signed distance field.
        # This creates a point cloud and wireframe representation showing obstacle surfaces.
        sdf_visualizer.add_to_visualizer(visualizer)

        # Animate the robot following the trajectory in the same window. This creates
        # a real-time playback of the optimized collision-free trajectory, showing
        # how the robot moves from start to target while avoiding the SDF obstacle.
        dt = 0.05  # 50ms time steps for 20 FPS

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
        # path exists given the SDF obstacle configuration and constraints, or if the
        # SDF representation contains invalid values.
        status = results.status()
        print(f"Failed to find trajectory. Status: {status}")


if __name__ == "__main__":
    main()
