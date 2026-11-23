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

"""Unit tests for robot_description_builder_python.h.

These tests mirror the C++ tests in `robot_description_builder_test.cpp`. Those tests inspect the
built robot through `RobotDescriptionInternal` and `KinematicStructure`, neither of which is
exposed to Python, so the following checks are instead expressed through the public Python API:
  - Self-collision masks are checked by the frame pairs reported in self-collision by
    `RobotWorldInspector.frames_in_self_collision()`, rather than by the contents of
    `RobotDescriptionInternal::selfCollisionMask()`. The order in which masked frames are recorded
    is therefore not observable from Python.
  - Collision spheres are counted by `RobotWorldInspector.num_world_collision_spheres()` and
    `RobotWorldInspector.num_self_collision_spheres()`, and attributed to frames by
    `RobotWorldInspector.world_collision_sphere_frame_name()` and
    `RobotWorldInspector.self_collision_sphere_frame_name()`, rather than by
    `RobotDescriptionInternal::numCollisionSpheres()`, `::collisionFrameNames()`, and
    `::collisionSpheres()`.

The following C++ checks have no public Python equivalent and so are not mirrored here:
  - The joint names of the built robot (`KinematicStructure::jointNames()`), so the discarding of
    a joint that drives a frame disconnected from the root frame is observed only indirectly,
    through `Kinematics.frame_names()` and `RobotDescription.num_cspace_coords()`.
  - The count and names of the auxiliary c-space coordinates of the built robot
    (`KinematicStructure::numAuxCSpaceCoords()` and `::auxCSpaceCoordName()`). Only the effect of
    an auxiliary c-space coordinate on the "standard" c-space is observable, via
    `RobotDescription.num_cspace_coords()`.
  - The content of log output, which cannot be captured from Python, so a warning or a nonfatal
    error is never asserted. Each use of `warnings_disabled` or `errors_disabled` below marks a
    call that is expected to log, and keeps the expected output from cluttering the test run.
"""

# Standard Library
from collections import Counter

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion

# Local Folder
from ._test_helper import errors_disabled, warnings_disabled

# Axis about which every joint in these tests rotates or translates.
JOINT_AXIS = np.array([1.0, 0.0, 0.0])


def _create_builder_with_frames(frame_names):
    """Create a builder with root frame "root" and add each frame in `frame_names` to it."""
    builder = cumotion.create_robot_description_builder('root')
    assert builder is not None
    for frame_name in frame_names:
        assert builder.add_frame(frame_name)
    return builder


def _add_limited_revolute_joint(builder, name, parent_frame_name, child_frame_name):
    """Add a revolute joint named `name` to `builder`, suitable for c-space control.

    The joint connects `parent_frame_name` to `child_frame_name` with an identity fixed
    transform, and rotates about `JOINT_AXIS`. It is given the position and velocity limits
    required of a joint controlled by a "standard" c-space coordinate.
    """
    return builder.add_revolute_joint(name, parent_frame_name, child_frame_name,
                                      cumotion.Pose3.identity(), JOINT_AXIS,
                                      cumotion.Kinematics.Limits(-1.0, 1.0), 1.0)


def _dual_collision_sphere(radius):
    """Return a collision sphere of `radius` enabled for both world- and self-collision."""
    return cumotion.RobotDescriptionBuilder.CollisionSphere(np.zeros(3), radius, 0.0, 0.0)


def _colliding_frame_pairs(builder, cspace_position):
    """Return the set of frame pairs in self-collision in the robot specified by `builder`.

    The robot built from `builder` is evaluated at `cspace_position`, whose length must equal the
    number of c-space coordinates of that robot. Each pair is returned as a `frozenset` of the two
    frame names, since the order in which a pair is reported is arbitrary.
    """
    robot = builder.build()
    assert len(cspace_position) == robot.num_cspace_coords()
    inspector = cumotion.create_robot_world_inspector(robot, None)
    return {frozenset(pair) for pair in inspector.frames_in_self_collision(cspace_position)}


def _collision_sphere_counts(builder):
    """Return the world- and self-collision sphere counts of the robot specified by `builder`."""
    return _inspector_collision_sphere_counts(
        cumotion.create_robot_world_inspector(builder.build(), None))


def _inspector_collision_sphere_counts(inspector):
    """Return the world- and self-collision sphere counts of the robot held by `inspector`."""
    return inspector.num_world_collision_spheres(), inspector.num_self_collision_spheres()


def _collision_spheres_per_frame(inspector):
    """Return the world- and self-collision sphere counts per frame of `inspector`.

    Each count is returned as a `Counter` keyed on frame name, standing in for the per-frame
    sphere counts that the C++ tests read from `RobotDescriptionInternal::collisionSpheres()`.
    """
    world_counts = Counter(inspector.world_collision_sphere_frame_name(index)
                           for index in range(inspector.num_world_collision_spheres()))
    self_counts = Counter(inspector.self_collision_sphere_frame_name(index)
                          for index in range(inspector.num_self_collision_spheres()))
    return world_counts, self_counts


def _check_joint_input_validation(builder, add_joint_fn, add_other_joint_fn):
    """Check the input validation shared by `add_prismatic_joint()` and `add_revolute_joint()`.

    The `builder` must have frames "frame_a" through "frame_d" added to the root frame "root".
    The `add_joint_fn` is the bound method under test, while `add_other_joint_fn` is the bound
    method of the other joint type, used to check that joint names are unique across all joint
    types.
    """
    identity = cumotion.Pose3.identity()

    with errors_disabled:
        # An empty joint name is invalid.
        assert not add_joint_fn('', 'root', 'frame_a', identity, JOINT_AXIS, None, None)

        # A nonexistent (or empty) parent or child frame name is invalid.
        assert not add_joint_fn('joint', 'not_a_root', 'frame_a', identity, JOINT_AXIS, None, None)
        assert not add_joint_fn('joint', '', 'frame_a', identity, JOINT_AXIS, None, None)
        assert not add_joint_fn('joint', 'root', 'not_a_frame', identity, JOINT_AXIS, None, None)
        assert not add_joint_fn('joint', 'root', '', identity, JOINT_AXIS, None, None)

        # The child frame cannot be the root frame.
        assert not add_joint_fn('joint', 'frame_a', 'root', identity, JOINT_AXIS, None, None)

        # The same frame cannot be used for both the parent and the child frame.
        assert not add_joint_fn('joint', 'frame_a', 'frame_a', identity, JOINT_AXIS, None, None)

    # Expect that `builder` remains in a valid state and joints can be added.
    assert add_joint_fn('joint', 'root', 'frame_a', identity, JOINT_AXIS, None, None)

    with errors_disabled:
        # Expect that the name "joint" cannot be reused for a new joint.
        assert not add_joint_fn('joint', 'frame_a', 'frame_b', identity, JOINT_AXIS, None, None)

    # Add a joint of the other type with a unique name.
    assert add_other_joint_fn('other_joint', 'frame_a', 'frame_b', identity, JOINT_AXIS, None, None)

    with errors_disabled:
        # Expect that the name "other_joint" cannot be reused, even though the joint under test is
        # of a different joint type.
        assert not add_joint_fn('other_joint', 'frame_b', 'frame_c', identity, JOINT_AXIS, None,
                                None)

    # Expect that a joint with a unique name can be added.
    assert add_joint_fn('unique_joint', 'frame_b', 'frame_c', identity, JOINT_AXIS, None, None)

    with errors_disabled:
        # Expect that a cyclic dependency cannot be created, because the child frame cannot have
        # been previously added as the child frame of another joint.
        assert not add_joint_fn('cyclic_joint', 'frame_c', 'frame_a', identity, JOINT_AXIS, None,
                                None)

        # The `joint_axis` must not be zero or nearly zero.
        assert not add_joint_fn('zero_axis_joint', 'frame_c', 'frame_d', identity, np.zeros(3),
                                None, None)
        assert not add_joint_fn('near_zero_axis_joint', 'frame_c', 'frame_d', identity,
                                1e-9 * JOINT_AXIS, None, None)

        # The `position_limits` (if provided) must represent a valid, non-zero range.
        assert not add_joint_fn('zero_range_joint', 'frame_c', 'frame_d', identity, JOINT_AXIS,
                                cumotion.Kinematics.Limits(1.0, 1.0), None)
        assert not add_joint_fn('negative_range_joint', 'frame_c', 'frame_d', identity,
                                JOINT_AXIS, cumotion.Kinematics.Limits(1.0, -1.0), None)

        # The `velocity_limit` (if provided) must be positive.
        assert not add_joint_fn('zero_velocity_limit_joint', 'frame_c', 'frame_d', identity,
                                JOINT_AXIS, None, 0.0)
        assert not add_joint_fn('negative_velocity_limit_joint', 'frame_c', 'frame_d', identity,
                                JOINT_AXIS, None, -1.0)

        # The `joint_map` (if provided) may not have a `cspace_multiplier` that is zero or near
        # zero.
        assert not add_joint_fn('zero_multiplier_joint', 'frame_c', 'frame_d', identity, JOINT_AXIS,
                                None, None, cumotion.RobotDescriptionBuilder.JointMap(0.0, 0.0))
        assert not add_joint_fn('near_zero_multiplier_joint', 'frame_c', 'frame_d', identity,
                                JOINT_AXIS, None, None,
                                cumotion.RobotDescriptionBuilder.JointMap(1e-9, 0.0))

    # Expect that `builder` can create a `RobotDescription`, discarding "frame_d" (with a warning)
    # because no joint drives it.
    with warnings_disabled:
        robot = builder.build()
    assert robot is not None


def test_create_robot_description_builder():
    """Test valid and invalid calls to `create_robot_description_builder()`."""
    # Expect that a builder is successfully created with a valid root frame name.
    valid_builder = cumotion.create_robot_description_builder('valid_root_name')
    assert valid_builder is not None
    assert 'cumotion.RobotDescriptionBuilder' == repr(valid_builder)

    # Expect failure to create a builder with an invalid (empty) root frame name.
    with errors_disabled:
        assert cumotion.create_robot_description_builder('') is None


def test_add_frame():
    """Test valid and invalid calls to `RobotDescriptionBuilder.add_frame()`."""
    builder = cumotion.create_robot_description_builder('root')
    assert builder is not None

    with errors_disabled:
        # Expect failure to add a frame repeating the root name (with or without annotation as a
        # tool frame).
        assert not builder.add_frame('root', True)
        assert not builder.add_frame('root', False)

        # An empty frame name is invalid (with or without annotation as a tool frame).
        assert not builder.add_frame('', True)
        assert not builder.add_frame('', False)

    # Expect that `builder` remains in a valid state and frames can be added (with or without
    # annotation as a tool frame).
    assert builder.add_frame('this_is_a_tool_frame', True)
    assert builder.add_frame('this_is_not_a_tool_frame', False)
    assert builder.add_frame('this_is_also_a_tool_frame', is_tool_frame=True)

    # Expect that `is_tool_frame` defaults to a frame that is not a tool frame.
    assert builder.add_frame('this_is_not_a_tool_frame_either')

    with errors_disabled:
        # Expect a repeated frame name to be invalid (with or without annotation as a tool frame).
        assert not builder.add_frame('this_is_a_tool_frame', True)
        assert not builder.add_frame('this_is_not_a_tool_frame', False)

        # Expect that `builder` can create a `RobotDescription`. Every added frame is discarded
        # because no joint connects it to the root frame, and each discarded tool frame is
        # reported as a nonfatal error.
        robot = builder.build()

    assert robot is not None
    assert 0 == len(robot.tool_frame_names())


def test_tool_frames():
    """Test that tool frames are reported in the order in which they were added."""
    builder = cumotion.create_robot_description_builder('root')
    assert builder is not None

    # Add four frames to the robot description. This kinematic structure will connect as:
    #   "root" -> "frame_a" -> "frame_b" -> "frame_c" -> "frame_d".
    # The frames, however, are added in a different order, demonstrating that addition order in
    # the builder dictates the resulting order for `RobotDescription.tool_frame_names()`.
    assert builder.add_frame('frame_d', True)
    assert builder.add_frame('frame_c', False)  # Not a tool frame.
    assert builder.add_frame('frame_a', True)
    assert builder.add_frame('frame_b', True)

    # Adding joints is required for the frames to not be discarded when calling `build()`.
    identity = cumotion.Pose3.identity()
    assert builder.add_fixed_joint('joint0', 'root', 'frame_a', identity)
    assert builder.add_fixed_joint('joint1', 'frame_a', 'frame_b', identity)
    assert builder.add_fixed_joint('joint2', 'frame_b', 'frame_c', identity)
    assert builder.add_fixed_joint('joint3', 'frame_c', 'frame_d', identity)

    robot = builder.build()
    assert robot is not None

    # Expect tool frames to be reported in order of addition to `builder`.
    assert ['frame_d', 'frame_a', 'frame_b'] == robot.tool_frame_names()


def test_add_fixed_joint():
    """Test valid and invalid calls to `RobotDescriptionBuilder.add_fixed_joint()`."""
    builder = _create_builder_with_frames(['frame_a', 'frame_b', 'frame_c'])
    identity = cumotion.Pose3.identity()

    with errors_disabled:
        # An empty joint name is invalid.
        assert not builder.add_fixed_joint('', 'root', 'frame_a', identity)

        # A nonexistent (or empty) parent or child frame name is invalid.
        assert not builder.add_fixed_joint('joint', 'not_a_root', 'frame_a', identity)
        assert not builder.add_fixed_joint('joint', '', 'frame_a', identity)
        assert not builder.add_fixed_joint('joint', 'root', 'not_a_frame', identity)
        assert not builder.add_fixed_joint('joint', 'root', '', identity)

        # The child frame cannot be the root frame.
        assert not builder.add_fixed_joint('joint', 'frame_a', 'root', identity)

        # The same frame cannot be used for both the parent and the child frame.
        assert not builder.add_fixed_joint('joint', 'frame_a', 'frame_a', identity)

    # Expect that `builder` remains in a valid state and joints can be added.
    assert builder.add_fixed_joint('joint', 'root', 'frame_a', identity)

    with errors_disabled:
        # Expect that the name "joint" cannot be reused for a new joint.
        assert not builder.add_fixed_joint('joint', 'frame_a', 'frame_b', identity)

    # Add a non-fixed joint with a unique name.
    assert builder.add_prismatic_joint('other_joint', 'frame_a', 'frame_b', identity, JOINT_AXIS,
                                       None, None)

    with errors_disabled:
        # Expect that the name "other_joint" cannot be reused for a fixed joint, even though it is
        # of a different joint type.
        assert not builder.add_fixed_joint('other_joint', 'frame_b', 'frame_c', identity)

    # Expect that a fixed joint with a unique name can be added.
    assert builder.add_fixed_joint('unique_joint', 'frame_b', 'frame_c', identity)

    with errors_disabled:
        # Expect that a cyclic dependency cannot be created, because the child frame cannot have
        # been previously added as the child frame of another joint.
        assert not builder.add_fixed_joint('cyclic_joint', 'frame_c', 'frame_a', identity)

    # Expect that `builder` can create a `RobotDescription`.
    robot = builder.build()
    assert robot is not None


def test_add_prismatic_joint():
    """Test valid and invalid calls to `RobotDescriptionBuilder.add_prismatic_joint()`."""
    builder = _create_builder_with_frames(['frame_a', 'frame_b', 'frame_c', 'frame_d'])
    _check_joint_input_validation(builder, builder.add_prismatic_joint,
                                  builder.add_revolute_joint)


def test_add_revolute_joint():
    """Test valid and invalid calls to `RobotDescriptionBuilder.add_revolute_joint()`."""
    builder = _create_builder_with_frames(['frame_a', 'frame_b', 'frame_c', 'frame_d'])
    _check_joint_input_validation(builder, builder.add_revolute_joint,
                                  builder.add_prismatic_joint)


def test_add_cspace_coord():
    """Test valid and invalid calls to `RobotDescriptionBuilder.add_cspace_coord()`."""
    # Create a basic robot with three revolute joints, where all joints have position and velocity
    # limits (making them suitable to be controlled by c-space coordinates).
    builder = _create_builder_with_frames(['frame_a', 'frame_b', 'frame_c'])
    assert _add_limited_revolute_joint(builder, 'joint_0', 'root', 'frame_a')
    assert _add_limited_revolute_joint(builder, 'joint_1', 'frame_a', 'frame_b')
    assert _add_limited_revolute_joint(builder, 'joint_2', 'frame_b', 'frame_c')

    with errors_disabled:
        # An empty c-space coord name is invalid.
        assert not builder.add_cspace_coord('', ['joint_0', 'joint_1'], 1.0, 1.0)

    # A c-space coordinate *can* have the same name as a joint.
    # NOTE: This convention is used when loading from URDF/XRDF.
    assert builder.add_cspace_coord('joint_0', ['joint_0', 'joint_1'], 1.0, 1.0)

    with errors_disabled:
        # A c-space coordinate can *NOT* be the same as another c-space coordinate.
        assert not builder.add_cspace_coord('joint_0', ['joint_2'], 1.0, 1.0)

        # A c-space coordinate can *NOT* control a joint that hasn't been added.
        assert not builder.add_cspace_coord('coord', ['joint_2', 'joint_3'], 1.0, 1.0)

        # A c-space coordinate can *NOT* control a joint that is already controlled by another
        # c-space coordinate.
        assert not builder.add_cspace_coord('coord', ['joint_2', 'joint_0'], 1.0, 1.0)

        # A c-space coordinate can *NOT* control the same joint more than once.
        assert not builder.add_cspace_coord('coord', ['joint_2', 'joint_2'], 1.0, 1.0)

        # A c-space coordinate must control at least one joint.
        assert not builder.add_cspace_coord('coord', [], 1.0, 1.0)

        # A c-space coordinate must have a positive acceleration limit.
        assert not builder.add_cspace_coord('coord', ['joint_2'], 0.0, 1.0)
        assert not builder.add_cspace_coord('coord', ['joint_2'], -1.0, 1.0)

        # A c-space coordinate must have a positive jerk limit.
        assert not builder.add_cspace_coord('coord', ['joint_2'], 1.0, 0.0)
        assert not builder.add_cspace_coord('coord', ['joint_2'], 1.0, -1.0)

        # A c-space coordinate must have a valid default position (if provided).
        assert not builder.add_cspace_coord('coord', ['joint_2'], 1.0, 1.0, 2.0)

    # Expect that `builder` is unmodified by failures and a valid c-space coordinate can still be
    # added.
    assert builder.add_cspace_coord('coord', ['joint_2'], 1.0, 1.0, default_position=None)

    # Expect that `builder` can create a `RobotDescription` reporting both c-space coordinates in
    # the order in which they were added.
    robot = builder.build()
    assert robot is not None
    assert 2 == robot.num_cspace_coords()
    assert 'joint_0' == robot.cspace_coord_name(0)
    assert 'coord' == robot.cspace_coord_name(1)


def test_add_cspace_coord_pos_vel_limits():
    """Test `RobotDescriptionBuilder.add_cspace_coord()` with invalid position or velocity limits.

    This test is isolated from `test_add_cspace_coord` to reduce the complexity of the builder
    used in each test.
    """
    # Create a basic robot with four revolute joints carrying different limits.
    builder = _create_builder_with_frames(['frame_a', 'frame_b', 'frame_c', 'frame_d'])
    identity = cumotion.Pose3.identity()
    assert builder.add_revolute_joint('joint_with_pos_limits', 'root', 'frame_a', identity,
                                      JOINT_AXIS, cumotion.Kinematics.Limits(-1.0, 1.0), None)
    assert builder.add_revolute_joint('joint_with_vel_limit', 'frame_a', 'frame_b', identity,
                                      JOINT_AXIS, None, 1.0)
    assert builder.add_revolute_joint('joint_with_no_limits', 'frame_b', 'frame_c', identity,
                                      JOINT_AXIS, None, None)
    assert builder.add_revolute_joint('joint_with_nonoverlapping_pos_limits', 'frame_c', 'frame_d',
                                      identity, JOINT_AXIS,
                                      cumotion.Kinematics.Limits(2.0, 3.0), None)

    with errors_disabled:
        # Expect that a group of joints with no position limits cannot be controlled by a c-space
        # coord.
        assert not builder.add_cspace_coord('coord',
                                            ['joint_with_vel_limit', 'joint_with_no_limits'],
                                            1.0, 1.0)

        # Expect that a group of joints with no velocity limits cannot be controlled by a c-space
        # coord.
        assert not builder.add_cspace_coord('coord',
                                            ['joint_with_pos_limits', 'joint_with_no_limits'],
                                            1.0, 1.0)

        # Expect that a group of joints with non-overlapping position limits cannot be controlled
        # by a c-space coord.
        assert not builder.add_cspace_coord('coord',
                                            ['joint_with_pos_limits',
                                             'joint_with_nonoverlapping_pos_limits',
                                             'joint_with_vel_limit'],
                                            1.0, 1.0)

    # Expect that a c-space coord *CAN* be added to a group of joints with a single
    # position-limited joint and a single velocity-limited joint.
    assert builder.add_cspace_coord('coord',
                                    ['joint_with_pos_limits',
                                     'joint_with_vel_limit',
                                     'joint_with_no_limits'],
                                    1.0, 1.0)

    # Expect that `builder` can create a `RobotDescription`.
    robot = builder.build()
    assert robot is not None


def test_add_cspace_coord_existing_aux():
    """Test `RobotDescriptionBuilder.add_cspace_coord()` with joints under auxiliary c-space.

    This test is isolated from `test_add_cspace_coord` to reduce the complexity of the builder
    used in each test.
    """
    # Create a basic robot with two revolute joints.
    builder = _create_builder_with_frames(['frame_a', 'frame_b'])
    assert _add_limited_revolute_joint(builder, 'joint_0', 'root', 'frame_a')
    assert _add_limited_revolute_joint(builder, 'joint_1', 'frame_a', 'frame_b')

    # Add an auxiliary c-space coordinate controlling the first joint.
    assert builder.add_aux_cspace_coord('aux', ['joint_0'])

    with errors_disabled:
        # Expect that an (otherwise valid) c-space coordinate cannot reuse the name of an existing
        # auxiliary c-space coordinate.
        assert not builder.add_cspace_coord('aux', ['joint_1'], 1.0, 1.0)

        # Expect that a c-space coordinate cannot control a joint already controlled by an
        # auxiliary c-space coordinate.
        assert not builder.add_cspace_coord('coord', ['joint_0'], 1.0, 1.0)

    # Expect that `builder` remains valid and a valid c-space coordinate can be added.
    assert builder.add_cspace_coord('coord', ['joint_1'], 1.0, 1.0)

    # Expect that `builder` can create a `RobotDescription` in which the auxiliary c-space
    # coordinate is excluded from the c-space used for motion generation.
    robot = builder.build()
    assert robot is not None
    assert 1 == robot.num_cspace_coords()
    assert 'coord' == robot.cspace_coord_name(0)


def test_add_cspace_coord_fixed_joint():
    """Test `RobotDescriptionBuilder.add_cspace_coord()` with fixed joints.

    This test is isolated from `test_add_cspace_coord` to reduce the complexity of the builder
    used in each test.
    """
    # Create a basic robot with one fixed and one revolute joint.
    builder = _create_builder_with_frames(['frame_a', 'frame_b'])
    assert builder.add_fixed_joint('fixed_joint', 'root', 'frame_a', cumotion.Pose3.identity())
    assert _add_limited_revolute_joint(builder, 'revolute_joint', 'frame_a', 'frame_b')

    with errors_disabled:
        # Expect that a c-space coordinate cannot control a fixed joint.
        assert not builder.add_cspace_coord('coord', ['fixed_joint', 'revolute_joint'], 1.0, 1.0)

    # Expect that `builder` remains valid and a valid c-space coordinate can be added.
    assert builder.add_cspace_coord('coord', ['revolute_joint'], 1.0, 1.0)

    # Expect that `builder` can create a `RobotDescription`.
    robot = builder.build()
    assert robot is not None


def test_add_aux_cspace_coord():
    """Test valid and invalid calls to `RobotDescriptionBuilder.add_aux_cspace_coord()`."""
    # Create a basic robot with three revolute joints, where all joints have position and velocity
    # limits (making them suitable to be controlled by c-space coordinates).
    builder = _create_builder_with_frames(['frame_a', 'frame_b', 'frame_c'])
    assert _add_limited_revolute_joint(builder, 'joint_0', 'root', 'frame_a')
    assert _add_limited_revolute_joint(builder, 'joint_1', 'frame_a', 'frame_b')
    assert _add_limited_revolute_joint(builder, 'joint_2', 'frame_b', 'frame_c')

    with errors_disabled:
        # An empty auxiliary c-space coord name is invalid.
        assert not builder.add_aux_cspace_coord('', ['joint_0', 'joint_1'])

    # An auxiliary c-space coordinate *can* have the same name as a joint.
    assert builder.add_aux_cspace_coord('joint_0', ['joint_0', 'joint_1'])

    with errors_disabled:
        # An auxiliary c-space coordinate can *NOT* be the same as another c-space coordinate.
        assert not builder.add_aux_cspace_coord('joint_0', ['joint_2'])

        # An auxiliary c-space coordinate can *NOT* control a joint that hasn't been added.
        assert not builder.add_aux_cspace_coord('coord', ['joint_2', 'joint_3'])

        # An auxiliary c-space coordinate can *NOT* control a joint that is already controlled by
        # another c-space coordinate.
        assert not builder.add_aux_cspace_coord('coord', ['joint_2', 'joint_0'])

        # An auxiliary c-space coordinate can *NOT* control the same joint more than once.
        assert not builder.add_aux_cspace_coord('coord', ['joint_2', 'joint_2'])

        # An auxiliary c-space coordinate must control at least one joint.
        assert not builder.add_aux_cspace_coord('coord', [])

        # An auxiliary c-space coordinate must have a valid default position (if provided).
        assert not builder.add_aux_cspace_coord('coord', ['joint_2'], 2.0)

    # Expect that `builder` is unmodified by failures and a valid auxiliary c-space coordinate can
    # still be added.
    assert builder.add_aux_cspace_coord('coord', ['joint_2'], default_position=None)

    # Expect that `builder` can create a `RobotDescription` in which every joint is controlled by
    # an auxiliary c-space coordinate, and so excluded from the c-space used for motion
    # generation.
    robot = builder.build()
    assert robot is not None
    assert 0 == robot.num_cspace_coords()


def test_add_aux_cspace_coord_pos_vel_limits():
    """Test `RobotDescriptionBuilder.add_aux_cspace_coord()` with invalid position limits.

    This test is isolated from `test_add_aux_cspace_coord` to reduce the complexity of the builder
    used in each test.
    """
    # Create a basic robot with four revolute joints carrying different limits.
    builder = _create_builder_with_frames(['frame_a', 'frame_b', 'frame_c', 'frame_d'])
    identity = cumotion.Pose3.identity()
    assert builder.add_revolute_joint('joint_with_pos_limits', 'root', 'frame_a', identity,
                                      JOINT_AXIS, cumotion.Kinematics.Limits(-1.0, 1.0), None)
    assert builder.add_revolute_joint('joint_with_vel_limit', 'frame_a', 'frame_b', identity,
                                      JOINT_AXIS, None, 1.0)
    assert builder.add_revolute_joint('joint_with_no_limits', 'frame_b', 'frame_c', identity,
                                      JOINT_AXIS, None, None)
    assert builder.add_revolute_joint('joint_with_nonoverlapping_pos_limits', 'frame_c', 'frame_d',
                                      identity, JOINT_AXIS,
                                      cumotion.Kinematics.Limits(2.0, 3.0), None)

    with errors_disabled:
        # Expect that a group of joints with non-overlapping position limits cannot be controlled
        # by an auxiliary c-space coord.
        assert not builder.add_aux_cspace_coord('coord_0',
                                                ['joint_with_pos_limits',
                                                 'joint_with_nonoverlapping_pos_limits',
                                                 'joint_with_vel_limit'])

    # Expect that a group of joints with no position limits and no velocity limits *CAN* be
    # controlled by an auxiliary c-space coord.
    # NOTE: This is not true for "standard" c-space coordinates.
    assert builder.add_aux_cspace_coord('coord_0', ['joint_with_no_limits'])

    # Expect that an auxiliary c-space coord *CAN* be added to a group of joints with a single
    # position-limited joint and a single velocity-limited joint.
    assert builder.add_aux_cspace_coord('coord_1',
                                        ['joint_with_pos_limits', 'joint_with_vel_limit'])

    # Expect that `builder` can create a `RobotDescription`.
    robot = builder.build()
    assert robot is not None


def test_add_aux_cspace_coord_existing_cspace():
    """Test `RobotDescriptionBuilder.add_aux_cspace_coord()` with joints controlled by c-space.

    This test is isolated from `test_add_aux_cspace_coord` to reduce the complexity of the builder
    used in each test.
    """
    # Create a basic robot with two revolute joints.
    builder = _create_builder_with_frames(['frame_a', 'frame_b'])
    assert _add_limited_revolute_joint(builder, 'joint_0', 'root', 'frame_a')
    assert _add_limited_revolute_joint(builder, 'joint_1', 'frame_a', 'frame_b')

    # Add a c-space coordinate controlling the first joint.
    assert builder.add_cspace_coord('aux', ['joint_0'], 1.0, 1.0)

    with errors_disabled:
        # Expect that an (otherwise valid) auxiliary c-space coordinate cannot reuse the name of an
        # existing c-space coordinate.
        assert not builder.add_aux_cspace_coord('aux', ['joint_1'])

        # Expect that an auxiliary c-space coordinate cannot control a joint already controlled by
        # a "standard" c-space coordinate.
        assert not builder.add_aux_cspace_coord('coord', ['joint_0'])

    # Expect that `builder` remains valid and a valid auxiliary c-space coordinate can be added.
    assert builder.add_aux_cspace_coord('coord', ['joint_1'])

    # Expect that `builder` can create a `RobotDescription`.
    robot = builder.build()
    assert robot is not None
    assert 1 == robot.num_cspace_coords()


def test_add_aux_cspace_coord_fixed_joint():
    """Test `RobotDescriptionBuilder.add_aux_cspace_coord()` with fixed joints.

    This test is isolated from `test_add_aux_cspace_coord` to reduce the complexity of the builder
    used in each test.
    """
    # Create a basic robot with one fixed and one revolute joint.
    builder = _create_builder_with_frames(['frame_a', 'frame_b'])
    assert builder.add_fixed_joint('fixed_joint', 'root', 'frame_a', cumotion.Pose3.identity())
    assert _add_limited_revolute_joint(builder, 'revolute_joint', 'frame_a', 'frame_b')

    with errors_disabled:
        # Expect that an auxiliary c-space coordinate cannot control a fixed joint.
        assert not builder.add_aux_cspace_coord('coord', ['fixed_joint', 'revolute_joint'])

    # Expect that `builder` remains valid and a valid auxiliary c-space coordinate can be added.
    assert builder.add_aux_cspace_coord('coord', ['revolute_joint'])

    # Expect that `builder` can create a `RobotDescription`.
    robot = builder.build()
    assert robot is not None


def test_add_collision_spheres():
    """Test valid and invalid calls to `RobotDescriptionBuilder.add_collision_spheres()`."""
    # Create representative collision spheres for testing.
    zero = np.zeros(3)
    sphere_type = cumotion.RobotDescriptionBuilder.CollisionSphere
    world_only_sphere = sphere_type(zero, 1.0, 0.5, None)
    self_only_sphere = sphere_type(zero, 1.0, None, 0.2)
    dual_sphere = sphere_type(zero, 1.0, 0.1, 0.2)
    inactive_sphere = sphere_type(zero, 1.0, None, None)
    negative_radius_sphere = sphere_type(zero, -0.5, 0.0, 0.0)
    negative_effective_radius_sphere = sphere_type(zero, 1.0, -2.0, 0.0)
    positive_effective_radius_sphere = sphere_type(zero, -1.0, 2.0, None)

    # Create a single-joint robot to which collision spheres can be attached.
    builder = _create_builder_with_frames(['frame_a'])
    assert _add_limited_revolute_joint(builder, 'joint_0', 'root', 'frame_a')
    assert builder.add_cspace_coord('coord', ['joint_0'], 1.0, 1.0)

    # NOTE: `RobotDescriptionInternal::numCollisionSpheres()`, used by the C++ tests to count the
    #       spheres attached to the built robot, has no Python equivalent. The world- and
    #       self-collision sphere counts reported by `RobotWorldInspector` are used instead, so a
    #       sphere enabled for both kinds of collision is counted once in each.
    with errors_disabled:
        # Expect that spheres cannot be added to a nonexistent frame.
        assert not builder.add_collision_spheres('not_a_frame', [world_only_sphere,
                                                                 self_only_sphere,
                                                                 dual_sphere])
        assert (0, 0) == _collision_sphere_counts(builder)

        # Expect that spheres with a non-positive `radius` are invalid, and that no sphere from
        # the call is added.
        assert not builder.add_collision_spheres('root', [world_only_sphere,
                                                          self_only_sphere,
                                                          negative_radius_sphere,
                                                          dual_sphere])
        assert (0, 0) == _collision_sphere_counts(builder)

        # Expect that inactive spheres (i.e., neither world- nor self-collision) are invalid, and
        # that no sphere from the call is added.
        assert not builder.add_collision_spheres('root', [world_only_sphere,
                                                          self_only_sphere,
                                                          inactive_sphere,
                                                          dual_sphere])
        assert (0, 0) == _collision_sphere_counts(builder)

        # Expect that spheres with a negative effective radius (i.e., `radius` plus
        # `world_collision_buffer_distance` or `self_collision_buffer_distance`) are invalid, and
        # that no sphere from the call is added.
        assert not builder.add_collision_spheres('root', [world_only_sphere,
                                                          self_only_sphere,
                                                          negative_effective_radius_sphere,
                                                          dual_sphere])
        assert (0, 0) == _collision_sphere_counts(builder)

        # Expect that spheres with a negative `radius` are invalid, even with a positive effective
        # radius, and that no sphere from the call is added.
        assert not builder.add_collision_spheres('root', [world_only_sphere,
                                                          self_only_sphere,
                                                          positive_effective_radius_sphere,
                                                          dual_sphere])
        assert (0, 0) == _collision_sphere_counts(builder)

    # Expect that an empty set of collision spheres is a *valid* no-op.
    assert builder.add_collision_spheres('root', [])
    assert (0, 0) == _collision_sphere_counts(builder)

    # Expect that valid spheres will be added (including duplicates).
    assert builder.add_collision_spheres('root', [world_only_sphere,
                                                  self_only_sphere,
                                                  world_only_sphere,
                                                  dual_sphere])

    # Expect the built robot to carry only the spheres of the successful call, each attached to
    # "root" with an effective radius accounting for its `world_collision_buffer_distance` or
    # `self_collision_buffer_distance`.
    robot = builder.build()
    inspector = cumotion.create_robot_world_inspector(robot, None)
    assert 3 == inspector.num_world_collision_spheres()
    assert 2 == inspector.num_self_collision_spheres()
    assert [1.1, 1.5, 1.5] == pytest.approx(sorted(inspector.world_collision_sphere_radii()))
    assert [1.2, 1.2] == pytest.approx(sorted(inspector.self_collision_sphere_radii()))
    assert (Counter({'root': 3}), Counter({'root': 2})) == _collision_spheres_per_frame(inspector)

    # Expect that valid spheres are appended to those already attached to the frame.
    assert builder.add_collision_spheres('root', [dual_sphere])
    assert (4, 3) == _collision_sphere_counts(builder)

    # Expect the appended sphere not to be reflected in the previously built `robot`, which is
    # fully independent of the builder that produced it.
    assert (3, 2) == _inspector_collision_sphere_counts(
        cumotion.create_robot_world_inspector(robot, None))


def test_add_self_collision_mask():
    """Test valid and invalid calls to `RobotDescriptionBuilder.add_self_collision_mask()`."""
    # Create a robot with four frames connected by revolute joints. Every fixed transform is the
    # identity, so the origins of all four frames are coincident.
    builder = _create_builder_with_frames(['frame_a', 'frame_b', 'frame_c'])
    assert _add_limited_revolute_joint(builder, 'joint_0', 'root', 'frame_a')
    assert _add_limited_revolute_joint(builder, 'joint_1', 'frame_a', 'frame_b')
    assert _add_limited_revolute_joint(builder, 'joint_2', 'frame_b', 'frame_c')
    assert builder.add_cspace_coord('coord_0', ['joint_0'], 1.0, 1.0)
    assert builder.add_cspace_coord('coord_1', ['joint_1'], 1.0, 1.0)
    assert builder.add_cspace_coord('coord_2', ['joint_2'], 1.0, 1.0)

    # Attach a sphere active only for self collision to the first, third, and fourth frames, and a
    # sphere active only for world collision to the second. The self-collision spheres are
    # coincident, so every pair of frames carrying one is in collision unless masked.
    sphere_type = cumotion.RobotDescriptionBuilder.CollisionSphere
    self_sphere = sphere_type(np.zeros(3), 1.0, None, 1.0)
    world_sphere = sphere_type(np.zeros(3), 1.0, 1.0, None)
    assert builder.add_collision_spheres('root', [self_sphere])
    assert builder.add_collision_spheres('frame_b', [self_sphere])
    assert builder.add_collision_spheres('frame_c', [self_sphere])
    assert builder.add_collision_spheres('frame_a', [world_sphere])

    root_and_b = frozenset(['root', 'frame_b'])
    root_and_c = frozenset(['root', 'frame_c'])
    b_and_c = frozenset(['frame_b', 'frame_c'])

    with errors_disabled:
        # Expect that a self-collision mask cannot be added for a nonexistent frame.
        assert not builder.add_self_collision_mask('not_the_root', ['root', 'frame_a'])

        # Expect that a self-collision mask cannot include a nonexistent frame.
        assert not builder.add_self_collision_mask('root', ['not_a_frame', 'frame_a'])

    # Every joint is revolute with an identity fixed transform, so no joint value moves a frame
    # origin off the root origin, and every self-collision sphere is centered on its frame origin.
    # The colliding pairs are therefore the same at every c-space position, and the zero position
    # is used throughout only because it is the simplest one within every joint's limits.
    #
    # The unmasked expectation is therefore all three pairs among the self-collision frames
    # "root", "frame_b", and "frame_c". "frame_a" never appears in a pair, because its only sphere
    # is enabled for world collision alone.

    # Expect the failed calls above to have added no mask.
    assert {root_and_b, root_and_c, b_and_c} == _colliding_frame_pairs(builder, np.zeros(3))

    # Expect that masking a frame against itself is a valid no-op.
    assert builder.add_self_collision_mask('root', ['root'])
    assert {root_and_b, root_and_c, b_and_c} == _colliding_frame_pairs(builder, np.zeros(3))

    # Expect that an empty self-collision mask is a valid no-op.
    assert builder.add_self_collision_mask('root', [])
    assert {root_and_b, root_and_c, b_and_c} == _colliding_frame_pairs(builder, np.zeros(3))

    # Expect that it is valid to mask a frame that has no self-collision spheres (as they may be
    # added subsequently). This mask, however, is discarded by `build()` with a warning, which
    # every `build()` from here on repeats.
    assert builder.add_self_collision_mask('root', ['frame_a'])
    with warnings_disabled:
        assert {root_and_b, root_and_c, b_and_c} == _colliding_frame_pairs(builder, np.zeros(3))

        # Expect a valid self-collision mask to eliminate the collision between "root" and
        # "frame_b".
        # NOTE: Masking is symmetric, so the pair is masked in one direction only.
        assert builder.add_self_collision_mask('root', ['frame_b'])
        assert {root_and_c, b_and_c} == _colliding_frame_pairs(builder, np.zeros(3))

        # Expect a further mask for "root" to be appended to the mask already recorded for it.
        assert builder.add_self_collision_mask('root', ['frame_c'])
        assert {b_and_c} == _colliding_frame_pairs(builder, np.zeros(3))

        # Expect that it is valid (but not required) to specify masking in both directions.
        assert builder.add_self_collision_mask('frame_c', ['root', 'frame_b'])
        assert set() == _colliding_frame_pairs(builder, np.zeros(3))


def test_add_self_collision_mask_symmetry():
    """Test that a mask added by `RobotDescriptionBuilder.add_self_collision_mask()` is symmetric.

    Masking is documented as symmetric, so a pair of frames only needs to be masked in one
    direction.
    """
    def in_self_collision(mask_frame_name, masked_frame_names):
        """Build a two-frame robot with the given mask and report whether it self-collides.

        The frames named in `masked_frame_names` are masked against the frame named
        `mask_frame_name`.
        """
        builder = _create_builder_with_frames(['frame_a'])
        assert _add_limited_revolute_joint(builder, 'joint', 'root', 'frame_a')
        assert builder.add_cspace_coord('coord', ['joint'], 1.0, 1.0)

        # Each sphere is centered on the origin of the frame carrying it, and the fixed transform
        # of "joint" is the identity, so the two spheres are coincident at every c-space position.
        sphere = cumotion.RobotDescriptionBuilder.CollisionSphere(np.zeros(3), 1.0, None, 0.0)
        assert builder.add_collision_spheres('root', [sphere])
        assert builder.add_collision_spheres('frame_a', [sphere])
        assert builder.add_self_collision_mask(mask_frame_name, masked_frame_names)

        inspector = cumotion.create_robot_world_inspector(builder.build(), None)
        return inspector.in_self_collision(np.zeros(1))

    # Expect the coincident spheres of "root" and "frame_a" to be in self-collision when the pair
    # is not masked.
    assert in_self_collision('root', [])

    # Expect the self-collision to be eliminated by a mask recorded on "frame_a".
    assert not in_self_collision('frame_a', ['root'])

    # Expect the self-collision to be eliminated by the same mask recorded on "root", because
    # masking is symmetric.
    assert not in_self_collision('root', ['frame_a'])


def test_build_culled_frames_simple():
    """Test that frames disconnected from the root frame are discarded by `build()`."""
    # Create a builder with four frames connected by revolute joints into two independent
    # kinematic structures:
    #   1. "root" -> "frame_a", and
    #   2. "frame_b" -> "frame_c".
    builder = cumotion.create_robot_description_builder('root')
    assert builder is not None
    assert builder.add_frame('frame_a', True)
    assert builder.add_frame('frame_b', True)
    assert builder.add_frame('frame_c', True)
    assert _add_limited_revolute_joint(builder, 'joint_0', 'root', 'frame_a')
    assert _add_limited_revolute_joint(builder, 'joint_1', 'frame_b', 'frame_c')
    assert builder.add_cspace_coord('coord', ['joint_0'], 1.0, 1.0)

    # Attach a collision sphere to each frame that is active for world- and self-collision.
    for frame_name in ['root', 'frame_a', 'frame_b', 'frame_c']:
        assert builder.add_collision_spheres(frame_name, [_dual_collision_sphere(1.0)])

    # Mask self-collision between all frames, so that the coincident spheres of the frames that
    # are retained are not in collision.
    assert builder.add_self_collision_mask('root', ['frame_a', 'frame_b', 'frame_c'])
    assert builder.add_self_collision_mask('frame_a', ['frame_b', 'frame_c'])
    assert builder.add_self_collision_mask('frame_b', ['frame_c'])

    # Expect the disconnected frames and joint to be discarded, along with the collision geometry
    # attached to them. The discarded tool frames are reported as nonfatal errors.
    with errors_disabled:
        robot = builder.build()
    assert robot is not None
    assert ['root', 'frame_a'] == robot.kinematics().frame_names()
    assert ['frame_a'] == robot.tool_frame_names()
    assert 1 == robot.num_cspace_coords()

    # Expect the retained frames to be the only ones carrying collision spheres, and the masks
    # involving the discarded frames to have been discarded with them.
    inspector = cumotion.create_robot_world_inspector(robot, None)
    assert 2 == inspector.num_world_collision_spheres()
    assert 2 == inspector.num_self_collision_spheres()
    expected_counts = Counter({'root': 1, 'frame_a': 1})
    assert (expected_counts, expected_counts) == _collision_spheres_per_frame(inspector)
    assert not inspector.in_self_collision(np.zeros(1))

    # Expect no more culling once the disjoint kinematic structures are joined.
    assert builder.add_fixed_joint('fixed', 'frame_a', 'frame_b', cumotion.Pose3.identity())
    joined_robot = builder.build()
    assert joined_robot is not None
    assert ['root', 'frame_a', 'frame_b', 'frame_c'] == joined_robot.kinematics().frame_names()
    assert ['frame_a', 'frame_b', 'frame_c'] == joined_robot.tool_frame_names()
    assert 1 == joined_robot.num_cspace_coords()

    joined_inspector = cumotion.create_robot_world_inspector(joined_robot, None)
    assert 4 == joined_inspector.num_world_collision_spheres()
    assert 4 == joined_inspector.num_self_collision_spheres()
    expected_joined_counts = Counter({'root': 1, 'frame_a': 1, 'frame_b': 1, 'frame_c': 1})
    assert (expected_joined_counts,
            expected_joined_counts) == _collision_spheres_per_frame(joined_inspector)
    assert not joined_inspector.in_self_collision(np.zeros(1))


def test_build_culled_frames_cspace():
    """Test that `build()` reports a fatal error if culling leaves a c-space coord uncontrolled."""
    # Create a builder with six frames connected by revolute joints into two independent kinematic
    # structures:
    #   1. "root" -> "frame_a" -> "frame_b", and
    #   2. "frame_c" -> "frame_d" -> "frame_e".
    # Every joint has position and velocity limits, making each of them suitable for control by a
    # c-space ("standard" or auxiliary) coordinate.
    builder = _create_builder_with_frames(['frame_a', 'frame_b', 'frame_c', 'frame_d', 'frame_e'])
    assert _add_limited_revolute_joint(builder, 'joint_0', 'root', 'frame_a')
    assert _add_limited_revolute_joint(builder, 'joint_1', 'frame_a', 'frame_b')
    assert _add_limited_revolute_joint(builder, 'culled_joint_0', 'frame_c', 'frame_d')
    assert _add_limited_revolute_joint(builder, 'culled_joint_1', 'frame_d', 'frame_e')

    # Add a c-space coord and an auxiliary c-space coord that each control one joint that will be
    # culled and one joint that will not.
    assert builder.add_cspace_coord('coord', ['joint_0', 'culled_joint_0'], 1.0, 1.0)
    assert builder.add_aux_cspace_coord('aux_coord', ['joint_1', 'culled_joint_1'])

    # Expect that the `RobotDescription` can be built, because each coord retains control of at
    # least one joint.
    with warnings_disabled:
        robot = builder.build()
    assert robot is not None
    assert 1 == robot.num_cspace_coords()
    assert 'coord' == robot.cspace_coord_name(0)

    # Create a builder in which "frame_a" -> "frame_b" is disconnected from the root frame, so
    # that the joint connecting them is culled by `build()`.
    culled_cspace_builder = _create_builder_with_frames(['frame_a', 'frame_b'])
    assert _add_limited_revolute_joint(culled_cspace_builder, 'culled_joint', 'frame_a', 'frame_b')
    assert culled_cspace_builder.add_cspace_coord('coord', ['culled_joint'], 1.0, 1.0)

    # Expect a fatal error, because *every* joint controlled by "coord" is culled, leaving it
    # controlling no joints at all.
    # NOTE: The fatal error is raised before any frame or joint is discarded, so this `build()`
    #       logs nothing and needs no suppression.
    with pytest.raises(RuntimeError):
        culled_cspace_builder.build()

    # Expect the same fatal error for an auxiliary c-space coord whose every controlled joint is
    # culled.
    culled_aux_builder = _create_builder_with_frames(['frame_a', 'frame_b'])
    assert _add_limited_revolute_joint(culled_aux_builder, 'culled_joint', 'frame_a', 'frame_b')
    assert culled_aux_builder.add_aux_cspace_coord('aux_coord', ['culled_joint'])
    with pytest.raises(RuntimeError):
        culled_aux_builder.build()


def test_build_held_joint_positions():
    """Test that `build()` holds joints not driven by standard c-space as documented."""
    builder = _create_builder_with_frames(
        ['driven_frame', 'held_frame', 'mapped_held_frame', 'aux_frame', 'mapped_aux_frame'])
    identity = cumotion.Pose3.identity()
    limits_type = cumotion.Kinematics.Limits
    joint_map = cumotion.RobotDescriptionBuilder.JointMap(2.0, 0.3)

    assert builder.add_revolute_joint(
        'driven_joint', 'root', 'driven_frame', identity, JOINT_AXIS,
        limits_type(-1.0, 1.0), 1.0)
    assert builder.add_prismatic_joint(
        'held', 'root', 'held_frame', identity, JOINT_AXIS, limits_type(0.2, 0.5), 1.0)
    assert builder.add_prismatic_joint(
        'held_1', 'root', 'mapped_held_frame', identity, JOINT_AXIS,
        limits_type(-1.0, 1.0), 1.0, joint_map)
    assert builder.add_prismatic_joint(
        'aux_joint', 'root', 'aux_frame', identity, JOINT_AXIS,
        limits_type(-0.5, -0.2), 1.0)
    assert builder.add_prismatic_joint(
        'mapped_aux_joint', 'root', 'mapped_aux_frame', identity, JOINT_AXIS,
        limits_type(-1.0, 1.0), 1.0, joint_map)

    # Name the c-space coord after the joint "held", which no coord controls, to check that
    # `build()` handles overlapping coord and joint names, including the joint named "held_1".
    assert builder.add_cspace_coord('held', ['driven_joint'], 1.0, 1.0)
    assert builder.add_aux_cspace_coord('aux', ['aux_joint'])
    assert builder.add_aux_cspace_coord('mapped_aux', ['mapped_aux_joint'], 0.0)

    robot = builder.build()
    assert robot is not None
    kinematics = robot.kinematics()
    cspace_position = np.zeros(1)

    # Expect a joint with no c-space coord to be held at zero clipped to its position limits.
    assert 0.2 == pytest.approx(kinematics.position(cspace_position, 'held_frame')[0])

    # Expect a joint with no c-space coord to be held at a joint position of zero, regardless of its
    # joint map.
    assert 0.0 == pytest.approx(kinematics.position(cspace_position, 'mapped_held_frame')[0])

    # Expect an auxiliary c-space coord with no default to be held at zero clipped to its limits.
    assert -0.2 == pytest.approx(kinematics.position(cspace_position, 'aux_frame')[0])

    # Expect a joint controlled by an auxiliary c-space coord with a default position of zero to be
    # held at the joint position that its joint map assigns to zero.
    assert 0.3 == pytest.approx(kinematics.position(cspace_position, 'mapped_aux_frame')[0])


def test_built_robot_description():
    """Test that a built `RobotDescription` reflects the specification given to the builder."""
    # Create a two-coordinate robot whose tool frame is offset along the local z-axis of the
    # second link.
    builder = _create_builder_with_frames(['link_1', 'link_2', 'tool'])
    identity = cumotion.Pose3.identity()
    assert builder.add_revolute_joint('shoulder_joint', 'root', 'link_1', identity, JOINT_AXIS,
                                      cumotion.Kinematics.Limits(-1.0, 1.0), 2.0)
    assert builder.add_prismatic_joint(
        'slide_joint', 'link_1', 'link_2', identity, JOINT_AXIS,
        cumotion.Kinematics.Limits(0.0, 1.0), 0.5,
        cumotion.RobotDescriptionBuilder.JointMap(0.5, 0.1))
    assert builder.add_fixed_joint('tool_joint', 'link_2', 'tool',
                                   cumotion.Pose3.from_translation(np.array([0.0, 0.0, 0.5])))
    assert builder.add_frame('tool_tip', is_tool_frame=True)
    assert builder.add_fixed_joint('tool_tip_joint', 'tool', 'tool_tip', identity)
    assert builder.add_cspace_coord('shoulder', ['shoulder_joint'], 5.0, 10.0,
                                    default_position=0.25)
    assert builder.add_cspace_coord('slide', ['slide_joint'], 3.0, 6.0)

    robot = builder.build()
    assert robot is not None

    # Expect the c-space coordinates to be reported in the order in which they were added.
    assert 2 == robot.num_cspace_coords()
    assert 'shoulder' == robot.cspace_coord_name(0)
    assert 'slide' == robot.cspace_coord_name(1)
    assert ['tool_tip'] == robot.tool_frame_names()

    # Expect the specified `default_position` for "shoulder", and a default of zero clipped to the
    # position limits for "slide".
    assert np.array([0.25, 0.0]) == pytest.approx(robot.default_cspace_configuration())

    kinematics = robot.kinematics()
    assert ['root', 'link_1', 'link_2', 'tool', 'tool_tip'] == kinematics.frame_names()
    assert 'root' == kinematics.base_frame_name()

    # Expect the position limits of "shoulder" to be those of the joint it controls, and those of
    # "slide" to be the limits of its joint expressed in c-space through its `joint_map` (i.e.,
    # `cspace_position = (joint_position - cspace_offset) / cspace_multiplier`).
    assert -1.0 == pytest.approx(kinematics.cspace_coord_limits(0).lower)
    assert 1.0 == pytest.approx(kinematics.cspace_coord_limits(0).upper)
    assert -0.2 == pytest.approx(kinematics.cspace_coord_limits(1).lower)
    assert 1.8 == pytest.approx(kinematics.cspace_coord_limits(1).upper)

    # Expect the acceleration and jerk limits specified for each c-space coordinate.
    assert 5.0 == kinematics.cspace_coord_acceleration_limit(0)
    assert 10.0 == kinematics.cspace_coord_jerk_limit(0)
    assert 3.0 == kinematics.cspace_coord_acceleration_limit(1)
    assert 6.0 == kinematics.cspace_coord_jerk_limit(1)

    # At the zero c-space position, "slide_joint" is at the `cspace_offset` of its `joint_map`, so
    # the tool tip is offset along local x by that amount and along local z by the fixed
    # transform of "tool_joint".
    zero_pose = kinematics.pose(np.zeros(2), 'tool_tip')
    assert 1.0 == pytest.approx(zero_pose.rotation.w())
    assert np.array([0.1, 0.0, 0.5]) == pytest.approx(zero_pose.translation)

    # Rotating "shoulder_joint" about the x-axis rotates the whole offset of the tool tip.
    angle = 1.0
    rotated_pose = kinematics.pose(np.array([angle, 0.0]), 'tool_tip')
    assert np.cos(0.5 * angle) == pytest.approx(rotated_pose.rotation.w())
    assert np.sin(0.5 * angle) == pytest.approx(rotated_pose.rotation.x())
    assert 0.0 == pytest.approx(rotated_pose.rotation.y())
    assert 0.0 == pytest.approx(rotated_pose.rotation.z())
    expected_translation = np.array([0.1, -0.5 * np.sin(angle), 0.5 * np.cos(angle)])
    assert expected_translation == pytest.approx(rotated_pose.translation)


def test_collision_sphere_and_joint_map_members():
    """Test the members of `CollisionSphere` and `JointMap` exposed by the bindings."""
    sphere = cumotion.RobotDescriptionBuilder.CollisionSphere(
        center=np.array([1.0, 2.0, 3.0]), radius=0.25,
        world_collision_buffer_distance=0.1, self_collision_buffer_distance=None)
    assert np.array([1.0, 2.0, 3.0]) == pytest.approx(sphere.center)
    assert 0.25 == sphere.radius
    assert 0.1 == sphere.world_collision_buffer_distance
    assert sphere.self_collision_buffer_distance is None

    # Expect every member to be writable, including populating and clearing the optional buffer
    # distances.
    sphere.center = np.zeros(3)
    sphere.radius = 0.5
    sphere.world_collision_buffer_distance = None
    sphere.self_collision_buffer_distance = -0.1
    assert np.zeros(3) == pytest.approx(sphere.center)
    assert 0.5 == sphere.radius
    assert sphere.world_collision_buffer_distance is None
    assert -0.1 == sphere.self_collision_buffer_distance

    # Expect both buffer distances to be required, so that a sphere enabled for neither world- nor
    # self-collision cannot be created by omission.
    with pytest.raises(TypeError):
        cumotion.RobotDescriptionBuilder.CollisionSphere(np.zeros(3), 0.25)

    joint_map = cumotion.RobotDescriptionBuilder.JointMap(cspace_multiplier=2.0,
                                                          cspace_offset=0.5)
    assert 2.0 == joint_map.cspace_multiplier
    assert 0.5 == joint_map.cspace_offset

    joint_map.cspace_multiplier = -1.0
    joint_map.cspace_offset = 0.0
    assert -1.0 == joint_map.cspace_multiplier
    assert 0.0 == joint_map.cspace_offset


def test_joint_limit_arguments():
    """Test the required and optional limit arguments of the joint methods."""
    builder = _create_builder_with_frames(['frame_a', 'frame_b', 'frame_c'])
    identity = cumotion.Pose3.identity()

    # Expect `position_limits` and `velocity_limit` to be required, matching the C++ signature.
    with pytest.raises(TypeError):
        builder.add_revolute_joint('joint', 'root', 'frame_a', identity, JOINT_AXIS)
    with pytest.raises(TypeError):
        builder.add_prismatic_joint('joint', 'root', 'frame_a', identity, JOINT_AXIS,
                                    cumotion.Kinematics.Limits(-1.0, 1.0))

    # Expect `joint_map` to default to no mapping between joint and c-space position.
    assert builder.add_revolute_joint('joint', 'root', 'frame_a', identity, JOINT_AXIS, None, None)

    # Expect a valid `joint_map` to be accepted by a revolute joint as well as a prismatic one.
    assert builder.add_revolute_joint('elbow', 'frame_a', 'frame_b', identity, JOINT_AXIS,
                                      cumotion.Kinematics.Limits(-1.0, 1.0), 1.0,
                                      cumotion.RobotDescriptionBuilder.JointMap(2.0, 0.1))

    # Expect every argument to be accepted by keyword.
    assert builder.add_prismatic_joint(
        name='slide',
        parent_frame_name='frame_b',
        child_frame_name='frame_c',
        fixed_transform=identity,
        joint_axis=JOINT_AXIS,
        position_limits=cumotion.Kinematics.Limits(0.0, 1.0),
        velocity_limit=1.0,
        joint_map=cumotion.RobotDescriptionBuilder.JointMap(1.0, 0.0))

    robot = builder.build()
    assert robot is not None
