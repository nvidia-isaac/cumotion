#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Demonstrate how to build a robot with the public `RobotDescriptionBuilder` API."""

# Third Party
import numpy as np

# cuMotion
import cumotion

POSITION_TOLERANCE = 1e-9
JOINT_LIMITS = cumotion.Kinematics.Limits(-np.pi, np.pi)
JOINT_AXIS = np.array([0.0, 0.0, 1.0])


def cumotion_print_status(success):
    """Print the final status of the example."""
    if success:
        print("CUMOTION EXAMPLE COMPLETED SUCCESSFULLY")
    else:
        print("CUMOTION EXAMPLE FAILED")


def has_expected_cspace_coords(robot, expected_cspace_coord_names):
    """Return whether `robot` has the expected c-space coordinate names in the same order."""
    if robot.num_cspace_coords() != len(expected_cspace_coord_names):
        return False
    return all(robot.cspace_coord_name(index) == name
               for index, name in enumerate(expected_cspace_coord_names))


def main():
    """Build and inspect a three-link planar arm."""
    # Build a three-link planar arm. Each revolute joint rotates around the z-axis, and each
    # unit-length link extends along its local x-axis.
    builder = cumotion.create_robot_description_builder("base")
    if builder is None:
        cumotion_print_status(False)
        return

    link_offset = cumotion.Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
    velocity_limit = 2.0
    acceleration_limit = 4.0
    jerk_limit = 8.0

    # Procedural construction is expected to succeed as all frames, joints, and c-space coordinates
    # are added.
    #
    # The generally recommended flow of:
    #   1. adding all frames,
    #   2. adding all joints, and
    #   3. adding all c-space coordinates
    # is used.
    #
    # This ordering is not required, but each component must be added prior to being referenced.
    # For example, adding "joint_1" requires that the frames "base" and "link_1" have been added.
    # Following the frames -> joint -> c-space ordering is a simple way to ensure these dependencies
    # are met.
    #
    # A c-space coordinate is intentionally omitted for "joint_3" at this time. This is to first
    # show the expected behavior of joints not controlled by a c-space coordinate.
    if not (
        builder.add_frame("link_1")
        and builder.add_frame("link_2")
        and builder.add_frame("link_3")
        and builder.add_frame("end_effector", is_tool_frame=True)
        and builder.add_revolute_joint(
            "joint_1",
            "base",
            "link_1",
            cumotion.Pose3.identity(),
            JOINT_AXIS,
            JOINT_LIMITS,
            velocity_limit)
        and builder.add_revolute_joint(
            "joint_2",
            "link_1",
            "link_2",
            link_offset,
            JOINT_AXIS,
            JOINT_LIMITS,
            velocity_limit)
        and builder.add_revolute_joint(
            "joint_3",
            "link_2",
            "link_3",
            link_offset,
            JOINT_AXIS,
            JOINT_LIMITS,
            velocity_limit)
        and builder.add_fixed_joint(
            "end_effector_joint", "link_3", "end_effector", link_offset)
        and builder.add_cspace_coord(
            "joint_1", ["joint_1"], acceleration_limit, jerk_limit)
        and builder.add_cspace_coord(
            "joint_2", ["joint_2"], acceleration_limit, jerk_limit)
    ):
        cumotion_print_status(False)
        return

    # First build the arm with only two c-space coordinates. `joint_3` has no c-space control, so
    # the resulting robot holds it at zero.
    #
    # Note: the default joint position for an uncontrolled joint will be "zero clipped to the
    # position limits"; in this case no clipping is needed. If a different default is preferred
    # *and* it is desired that the joint not be controlled by standard c-space, then an auxiliary
    # c-space coordinate can be used (via `add_aux_cspace_coord()`).
    two_coord_robot = builder.build()
    expected_two_coord_names = ["joint_1", "joint_2"]
    if (
        two_coord_robot is None
        or not has_expected_cspace_coords(two_coord_robot, expected_two_coord_names)
        or two_coord_robot.tool_frame_names() != ["end_effector"]
    ):
        cumotion_print_status(False)
        return

    # Extract a kinematic structure for the `two_coord_robot`, expecting the number of c-space
    # coordinates to be `2`.
    two_coord_kinematics = two_coord_robot.kinematics()
    if two_coord_kinematics is None or two_coord_kinematics.num_cspace_coords() != 2:
        cumotion_print_status(False)
        return
    two_coord_position = two_coord_kinematics.position(
        0.5 * np.pi * np.ones(2), "end_effector")
    two_coord_success = np.allclose(
        two_coord_position, np.array([-2.0, 1.0, 0.0]), rtol=0.0, atol=POSITION_TOLERANCE)
    print("Two-coordinate end-effector position (`joint_3` at zero):", two_coord_position)

    # Add control of `joint_3` and build an independent three-coordinate robot description.
    if not builder.add_cspace_coord(
        "joint_3", ["joint_3"], acceleration_limit, jerk_limit
    ):
        cumotion_print_status(False)
        return
    three_coord_robot = builder.build()
    expected_three_coord_names = ["joint_1", "joint_2", "joint_3"]
    if (
        three_coord_robot is None
        or not has_expected_cspace_coords(three_coord_robot, expected_three_coord_names)
        or two_coord_robot.num_cspace_coords() != 2
    ):
        cumotion_print_status(False)
        return

    # Extract a kinematic structure for the `three_coord_robot`, expecting the number of c-space
    # coordinates to be `3`.
    three_coord_kinematics = three_coord_robot.kinematics()
    if three_coord_kinematics is None or three_coord_kinematics.num_cspace_coords() != 3:
        cumotion_print_status(False)
        return
    three_coord_position = three_coord_kinematics.position(
        0.5 * np.pi * np.ones(3), "end_effector")
    three_coord_success = np.allclose(
        three_coord_position, np.array([-1.0, 0.0, 0.0]), rtol=0.0, atol=POSITION_TOLERANCE)
    print("Three-coordinate end-effector position (`joint_3` at pi / 2):", three_coord_position)

    # Attach a sphere to the frame "link_3". The sphere center is expressed in the local coordinates
    # of that frame. Populating both optional buffer distances enables the sphere for world- and
    # self-collision.
    collision_sphere = cumotion.RobotDescriptionBuilder.CollisionSphere(
        center=np.array([0.5, 0.0, 0.0]),
        radius=0.1,
        world_collision_buffer_distance=0.0,
        self_collision_buffer_distance=0.0)
    if not builder.add_collision_spheres("link_3", [collision_sphere]):
        cumotion_print_status(False)
        return

    # Build a new robot containing the collision geometry. Previously built robot descriptions
    # remain unchanged.
    collision_robot = builder.build()
    if collision_robot is None:
        cumotion_print_status(False)
        return
    collision_inspector = cumotion.create_robot_world_inspector(collision_robot, None)
    if collision_inspector is None:
        cumotion_print_status(False)
        return
    collision_sphere_success = (
        collision_inspector.num_world_collision_spheres() == 1
        and collision_inspector.num_self_collision_spheres() == 1
    )
    print(
        "Collision geometry:",
        collision_inspector.num_world_collision_spheres(),
        "world-collision sphere and",
        collision_inspector.num_self_collision_spheres(),
        "self-collision sphere")

    success = two_coord_success and three_coord_success and collision_sphere_success
    cumotion_print_status(success)


if __name__ == "__main__":
    main()
