# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Unit tests for trajectory_optimizer_python.h."""

# Standard Library
import os

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion

# Local Folder
from ._test_helper import CUMOTION_ROOT_DIR, errors_disabled


def test_trajectory_optimizer_config_set_param():
    """Test `TrajectoryOptimizerConfig.set_param()`."""
    # UR10 is arbitrarily chosen to load a robot description for testing.
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')

    # Load robot description and create world/view.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()

    # Create `TrajectoryOptimizerConfig` using default parameters.
    tool_frame_name = robot_description.tool_frame_names()[0]
    config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                 tool_frame_name,
                                                                 world_view)
    assert config is not None

    # Set known parameters to valid values, expecting success.
    # Integer parameters (positive).
    assert config.set_param("ik/num_seeds", 10)
    assert config.set_param("trajopt/num_seeds", 10)
    # Integer parameter (greater than or equal to 4).
    assert config.set_param("trajopt/num_knots_per_trajectory", 4)
    # Integer parameter (non-negative).
    assert config.set_param("ik/max_reattempts", 5)
    # Boolean parameters.
    assert config.set_param("enable_self_collision", True)
    assert config.set_param("enable_world_collision", False)
    # Double parameters (positive).
    assert config.set_param("task_space_terminal_position_tolerance", 0.001)
    assert config.set_param("task_space_terminal_orientation_tolerance", 0.005)

    with errors_disabled:
        # Set known parameters to invalid values, expecting parameters to *not* be updated.
        # Invalid positive integer parameters (must be > 0).
        assert not config.set_param("ik/num_seeds", 0)
        assert not config.set_param("ik/num_seeds", -3)
        assert not config.set_param("trajopt/num_seeds", 0)
        assert not config.set_param("trajopt/num_seeds", -3)
        # Invalid num_knots_per_trajectory (must be >= 4).
        assert not config.set_param("trajopt/num_knots_per_trajectory", 3)
        assert not config.set_param("trajopt/num_knots_per_trajectory", 0)
        assert not config.set_param("trajopt/num_knots_per_trajectory", -3)
        # Invalid non-negative integer parameter (must be >= 0).
        assert not config.set_param("ik/max_reattempts", -1)
        # Invalid double parameters (must be > 0).
        assert not config.set_param("task_space_terminal_position_tolerance", 0.0)
        assert not config.set_param("task_space_terminal_position_tolerance", -0.001)

        # Set unknown parameter, expecting parameter to *not* be updated.
        assert not config.set_param("unknown_parameter", 0)

        # Set known parameters with invalid types, expecting parameters to *not* be updated.
        assert not config.set_param("ik/num_seeds", 12.3)  # int parameter with double value.
        assert not config.set_param("enable_self_collision", 1)  # bool parameter with int value.
        assert not config.set_param("task_space_terminal_position_tolerance", False)  # double-bool.


def test_ur10_target_none():
    """Plan to a translation-only target for UR10 and expect success."""
    # Set absolute path to the XRDF and URDF for the UR10.
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')

    # Load robot description and create world/view.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()

    # Create optimizer using default parameters.
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    # Create task: translation target at (0.6, -0.5, 0.3) with no orientation constraint.
    position_target = np.array([0.6, -0.5, 0.3])
    translation_target = cumotion.TrajectoryOptimizer.TranslationConstraint.target(position_target)
    orientation_target = cumotion.TrajectoryOptimizer.OrientationConstraint.none()
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_target,
        orientation_target,
    )

    # Solve from default configuration.
    initial_configuration = robot_description.default_cspace_configuration()
    results = optimizer.plan_to_task_space_target(initial_configuration, task)

    # Check solution.
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.5) < 0.2


def test_ur10_target_constant():
    """Plan to a translation target with constant orientation for UR10 and expect success."""
    # Set absolute path to the XRDF and URDF for the UR10.
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')

    # Load robot description and create world/view.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()

    # Create optimizer using default parameters.
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    # Create task: translation target at (0.6, -0.5, 0.3) with constant orientation constraint.
    position_target = np.array([0.6, -0.5, 0.3])
    translation_target = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.target(
            position_target
        )
    )
    orientation_target = (
        cumotion.TrajectoryOptimizer.OrientationConstraint.constant()
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_target,
        orientation_target,
    )

    # Solve.
    initial_configuration = robot_description.default_cspace_configuration()
    results = optimizer.plan_to_task_space_target(initial_configuration, task)

    # Check solution.
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.5) < 0.2


def test_ur10_target_terminal_target():
    """Plan to a translation target with terminal orientation target for UR10 and expect success."""
    # Set absolute path to the XRDF and URDF for the UR10.
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')

    # Load robot description and create world/view.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()

    # Create optimizer using default parameters.
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    # Create task: translation target at (0.6, 0.5, 0.3) with terminal orientation = identity.
    position_target = np.array([0.6, 0.5, 0.3])
    translation_target = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.target(
            position_target
        )
    )
    orientation_target = (
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target(
            cumotion.Rotation3.identity()
        )
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_target,
        orientation_target,
    )

    # Solve.
    initial_configuration = robot_description.default_cspace_configuration()
    results = optimizer.plan_to_task_space_target(initial_configuration, task)

    # Check solution.
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.3) < 0.2


def test_ur10_target_terminal_and_path_target():
    """Plan to a translation target with terminal-and-path orientation target for UR10."""
    # Set absolute path to the XRDF and URDF for the UR10.
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')

    # Load robot description and create world/view.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()

    # Create optimizer using default parameters.
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    # Get initial tool pose and compute orientation target = R0 * RotX(0.4).
    initial_configuration = robot_description.default_cspace_configuration()
    kinematics = robot_description.kinematics()
    initial_tool_pose = kinematics.pose(initial_configuration, tool_frame_name)
    orientation_delta = cumotion.Rotation3.from_scaled_axis(np.array([0.4, 0.0, 0.0]))
    orientation_target = initial_tool_pose.rotation * orientation_delta

    # Create task: translation target at (0.6, -0.5, 0.3) with terminal and path orientation target.
    position_target = np.array([0.6, -0.5, 0.3])
    translation_constraint = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.target(
            position_target
        )
    )
    orientation_constraint = (
        cumotion.TrajectoryOptimizer.OrientationConstraint.
        terminal_and_path_target(
            orientation_target
        )
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_constraint,
        orientation_constraint,
    )

    # Solve.
    results = optimizer.plan_to_task_space_target(initial_configuration, task)

    # Check solution.
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.5) < 0.2


def test_ur10_target_terminal_axis():
    """Plan to a translation target with terminal axis orientation constraint for UR10."""
    # Set absolute path to the XRDF and URDF for the UR10.
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')

    # Load robot description and create world/view.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()

    # Create optimizer using default parameters.
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    # Create task: translation target at (0.6, -0.5, 0.3) and terminal axis constraint X->Z.
    position_target = np.array([0.6, -0.5, 0.3])
    translation_target = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.target(
            position_target
        )
    )
    orientation_target = (
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_axis(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_target,
        orientation_target,
    )

    # Solve.
    initial_configuration = robot_description.default_cspace_configuration()
    results = optimizer.plan_to_task_space_target(initial_configuration, task)

    # Check solution.
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.5) < 0.2


def test_ur10_target_terminal_and_path_axis():
    """Plan to a translation target with terminal-and-path axis constraint for UR10."""
    # Set absolute path to the XRDF and URDF for the UR10.
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')

    # Load robot description and create world/view.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()

    # Create optimizer using default parameters.
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    # Create task: translation target at (0.6, -0.5, 0.3) and axis constraint: tool Z aligns to
    # rotated -Z.
    position_target = np.array([0.6, -0.5, 0.3])
    translation_target = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.target(
            position_target
        )
    )

    # world target axis = R_y(0.1) * (-Z)
    world_target_axis = (
        cumotion.Rotation3.from_scaled_axis(np.array([0.0, 0.1, 0.0]))
        * np.array([0.0, 0.0, -1.0])
    )

    orientation_target = (
        cumotion.TrajectoryOptimizer.OrientationConstraint.
        terminal_and_path_axis(
            np.array([0.0, 0.0, 1.0]),
            world_target_axis,
        )
    )

    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_target,
        orientation_target,
    )

    # Solve.
    initial_configuration = robot_description.default_cspace_configuration()
    results = optimizer.plan_to_task_space_target(initial_configuration, task)

    # Check solution.
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.5) < 0.2


def test_ur10_target_terminal_target_and_path_axis():
    """Plan to a translation target with terminal target and path axis constraint for UR10."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    initial_configuration = robot_description.default_cspace_configuration()
    kinematics = robot_description.kinematics()
    initial_tool_pose = kinematics.pose(initial_configuration, tool_frame_name)
    orientation_target = (
        initial_tool_pose.rotation
        * cumotion.Rotation3.from_scaled_axis(np.array([2.0, 0.0, 0.0]))
    )

    position_target = np.array([-0.6, -0.5, 0.3])
    translation_constraint = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.target(
            position_target
        )
    )
    orientation_constraint = (
        cumotion.TrajectoryOptimizer.OrientationConstraint.
        terminal_target_and_path_axis(
            orientation_target,
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0])
        )
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_constraint,
        orientation_constraint,
    )

    results = optimizer.plan_to_task_space_target(initial_configuration, task)
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.8) < 0.2


def test_ur10_linear_none():
    """Linear path translation constraint with no orientation constraint."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    position_target = np.array([0.9, -0.5, 0.3])
    translation_target = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.
        linear_path_constraint(
            position_target
        )
    )
    orientation_target = cumotion.TrajectoryOptimizer.OrientationConstraint.none()
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_target,
        orientation_target,
    )

    initial_configuration = robot_description.default_cspace_configuration()
    results = optimizer.plan_to_task_space_target(initial_configuration, task)
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.8) < 0.2


def test_ur10_linear_constant():
    """Linear path translation with constant orientation."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    position_target = np.array([0.9, -0.5, 0.3])
    translation_target = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.
        linear_path_constraint(
            position_target
        )
    )
    orientation_target = cumotion.TrajectoryOptimizer.OrientationConstraint.constant()
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_target,
        orientation_target,
    )

    initial_configuration = robot_description.default_cspace_configuration()
    results = optimizer.plan_to_task_space_target(initial_configuration, task)
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.6) < 0.2


def test_ur10_linear_terminal_target():
    """Linear path translation with terminal orientation target."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    position_target = np.array([0.9, -0.5, 0.3])
    translation_target = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.
        linear_path_constraint(
            position_target
        )
    )
    orientation_target = cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target(
        cumotion.Rotation3.from_scaled_axis(np.array([0.0, 0.5, 0.0]))
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_target,
        orientation_target,
    )

    initial_configuration = robot_description.default_cspace_configuration()
    results = optimizer.plan_to_task_space_target(initial_configuration, task)
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.8) < 0.2


def test_ur10_linear_terminal_and_path_target():
    """Linear path translation with terminal-and-path orientation target."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    initial_configuration = robot_description.default_cspace_configuration()
    kinematics = robot_description.kinematics()
    initial_tool_pose = kinematics.pose(initial_configuration, tool_frame_name)

    position_target = np.array([0.8, -0.5, 0.1])
    translation_target = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.
        linear_path_constraint(
            position_target
        )
    )
    orientation_target = (
        cumotion.TrajectoryOptimizer.OrientationConstraint.
        terminal_and_path_target(
            initial_tool_pose.rotation
            * cumotion.Rotation3.from_scaled_axis(np.array([0.0, 0.3, 0.0])),
        )
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_target,
        orientation_target,
    )

    results = optimizer.plan_to_task_space_target(initial_configuration, task)
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.7) < 0.2


def test_ur10_linear_terminal_and_path_axis():
    """Linear path translation with terminal-and-path axis orientation constraint."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    initial_configuration = robot_description.default_cspace_configuration()

    position_target = np.array([0.8, -0.5, 0.1])
    translation_target = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.
        linear_path_constraint(
            position_target
        )
    )
    orientation_target = cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0])
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_target,
        orientation_target,
    )

    results = optimizer.plan_to_task_space_target(initial_configuration, task)
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.5) < 0.2


def test_ur10_linear_terminal_target_and_path_axis():
    """Linear path translation with terminal target and path axis constraint."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    initial_configuration = robot_description.default_cspace_configuration()
    kinematics = robot_description.kinematics()
    initial_tool_pose = kinematics.pose(initial_configuration, tool_frame_name)

    position_target = np.array([0.8, -0.5, 0.1])
    translation_target = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.
        linear_path_constraint(
            position_target
        )
    )
    orientation_target = (
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
            initial_tool_pose.rotation
            * cumotion.Rotation3.from_scaled_axis(np.array([-1.0, 0.0, 0.0])),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0])
        )
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_target,
        orientation_target,
    )

    results = optimizer.plan_to_task_space_target(initial_configuration, task)
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.7) < 0.2


def test_ur10_linear_terminal_axis():
    """Linear path translation with terminal axis constraint."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    position_target = np.array([-0.8, -0.5, 0.1])
    translation_target = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.
        linear_path_constraint(
            position_target
        )
    )

    world_target_axis = (
        cumotion.Rotation3.from_scaled_axis(np.array([np.pi, 0.0, 0.0]))
        * np.array([1.0, 0.0, 0.0])
    )
    orientation_target = (
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_axis(
            np.array([0.0, 0.0, 1.0]),
            world_target_axis,
        )
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_target,
        orientation_target,
    )

    initial_configuration = robot_description.default_cspace_configuration()
    results = optimizer.plan_to_task_space_target(initial_configuration, task)
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 2.2) < 0.2


def test_ur10_goalset_none():
    """Goalset planning: translation targets with no orientation constraints."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    position_targets = [np.array([0.3, 0.6, 0.3]), np.array([-0.4, -0.5, 0.3])]
    translation_goalset = (
        cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.target(
            position_targets
        )
    )
    orientation_goalset = (
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.none()
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTargetGoalset(
        translation_goalset,
        orientation_goalset,
    )

    initial_configuration = np.zeros(robot_description.num_cspace_coords())
    results = optimizer.plan_to_task_space_goalset(initial_configuration, task)
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.0) < 0.2


def test_ur10_goalset_target():
    """Goalset planning: translation targets with terminal orientation targets."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    position_targets = [np.array([0.3, 0.6, 0.3]), np.array([-0.4, -0.5, 0.3])]
    orientation_targets = [
        cumotion.Rotation3.from_scaled_axis(np.array([1.5, 0.0, 0.0])),
        cumotion.Rotation3.from_scaled_axis(np.array([0.0, 0.0, 2.0])),
    ]
    translation_goalset = (
        cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.target(
            position_targets
        )
    )
    orientation_goalset = (
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.
        terminal_target(
            orientation_targets
        )
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTargetGoalset(
        translation_goalset,
        orientation_goalset,
    )

    initial_configuration = np.zeros(robot_description.num_cspace_coords())
    results = optimizer.plan_to_task_space_goalset(initial_configuration, task)
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.2) < 0.2


def test_ur10_goalset_terminal_and_path_axis():
    """Goalset planning: translation targets with terminal-and-path axis constraints."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    position_targets = [np.array([0.8, 0.6, 0.3]), np.array([-0.2, 0.6, 0.3])]
    translation_goalset = (
        cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.target(
            position_targets
        )
    )
    tool_frame_axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    world_target_axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    orientation_goalset = (
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.
        terminal_and_path_axis(
            tool_frame_axes,
            world_target_axes,
        )
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTargetGoalset(
        translation_goalset,
        orientation_goalset,
    )

    initial_configuration = robot_description.default_cspace_configuration()
    results = optimizer.plan_to_task_space_goalset(initial_configuration, task)
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 0.7) < 0.2


def test_ur10_goalset_linear_none():
    """Goalset planning: linear path translation constraints with no orientation constraints."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()
    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    position_targets = [np.array([0.3, 0.6, 0.3]), np.array([-0.4, -0.5, 0.3])]
    translation_goalset = (
        cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.
        linear_path_constraint(
            position_targets
        )
    )
    orientation_goalset = (
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.none()
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTargetGoalset(
        translation_goalset,
        orientation_goalset,
    )

    initial_configuration = np.zeros(robot_description.num_cspace_coords())
    results = optimizer.plan_to_task_space_goalset(initial_configuration, task)
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.0) < 0.2


def _populate_sdf_from_world_view(
    world_view: cumotion.WorldViewHandle,
    min_bound: np.array,
    voxel_size: float,
    sdf_grid_values: np.array,
):
    """Populate SDF grid values from world view.

    Args:
        world_view: World view to populate SDF grid values from.
        min_bound: Minimum bound of the SDF grid.
        voxel_size: Size of the voxels in the SDF grid.
        sdf_grid_values: SDF grid values to populate.
    """
    world_inspector = cumotion.create_world_inspector(world_view)
    for x in range(sdf_grid_values.shape[0]):
        for y in range(sdf_grid_values.shape[1]):
            for z in range(sdf_grid_values.shape[2]):
                query_point = (
                    min_bound + voxel_size * np.array([x, y, z]) + voxel_size / 2.0
                )
                distance = world_inspector.min_distance(query_point)
                sdf_grid_values[x, y, z] = distance


def test_ur10_axis_target_obstacles():
    """Plan with axis orientation target and obstacles present in the world."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()

    # Define two cuboid obstacles.
    obstacle1 = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
    obstacle1.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, np.array([1.4, 1.0, 0.2]))
    handle1 = world.add_obstacle(obstacle1)
    world.set_pose(
        handle1,
        cumotion.Pose3(
            cumotion.Rotation3.identity(),
            np.array([0.0, 0.0, 1.0]),
        ),
    )

    obstacle2 = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
    obstacle2.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, np.array([0.2, 1.0, 1.0]))
    handle2 = world.add_obstacle(obstacle2)
    world.set_pose(
        handle2,
        cumotion.Pose3(
            cumotion.Rotation3.identity(),
            np.array([0.6, 0.2, 0.25]),
        ),
    )

    world_view = world.add_world_view()

    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = (
        cumotion.create_default_trajectory_optimizer_config(
            robot_description,
            tool_frame_name,
            world_view,
        )
    )
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    position_target = np.array([0.6, -0.5, 0.3])
    translation_target = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.target(
            position_target
        )
    )
    orientation_target = (
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_axis(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_target,
        orientation_target,
    )

    initial_configuration = robot_description.default_cspace_configuration()
    results = optimizer.plan_to_task_space_target(initial_configuration, task)
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert abs(results.trajectory().domain().span() - 1.53) < 0.2
    print("Trajectory time (Cuboid): ", results.trajectory().domain().span(), " seconds.")

    # Create an SDF grid that contains both obstacles, using 'world_view' as the distance function.
    sdf_world = cumotion.create_world()

    min_bound = np.array([-1.0, -1.0, -1.0])
    voxel_size = 0.015
    num_voxels_x = 150
    num_voxels_y = 150
    num_voxels_z = 150
    sdf_grid_values = np.zeros((num_voxels_x, num_voxels_y, num_voxels_z))
    _populate_sdf_from_world_view(world_view, min_bound, voxel_size, sdf_grid_values)

    sdf_grid = cumotion.Obstacle.Grid(
        num_voxels_x,
        num_voxels_y,
        num_voxels_z,
        voxel_size,
        cumotion.Obstacle.GridPrecision.FLOAT,
        cumotion.Obstacle.GridPrecision.FLOAT,
    )

    sdf_pose = cumotion.Pose3(
        cumotion.Rotation3.identity(),
        min_bound,
    )

    # Create an obstacle from the SDF grid.
    obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.SDF)
    obstacle.set_attribute(cumotion.Obstacle.Attribute.GRID, sdf_grid)
    sdf_obstacle_handle = sdf_world.add_obstacle(obstacle, sdf_pose)

    sdf_world.set_sdf_grid_values_from_host(sdf_obstacle_handle, sdf_grid_values)

    sdf_world_view = sdf_world.add_world_view()

    # Solve again with the SDF obstacle.
    optimizer_config = (
        cumotion.create_default_trajectory_optimizer_config(
            robot_description,
            tool_frame_name,
            sdf_world_view,
        )
    )

    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    initial_configuration = robot_description.default_cspace_configuration()
    results_sdf = optimizer.plan_to_task_space_target(initial_configuration, task)
    assert results_sdf is not None
    assert results_sdf.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results_sdf.trajectory() is not None
    obstacle_time = results.trajectory().domain().span()
    sdf_time = results_sdf.trajectory().domain().span()
    assert abs(sdf_time - obstacle_time) < 0.3
    print("Trajectory time (SDF): ", results.trajectory().domain().span(), " seconds.")


def test_plan_to_cspace_target_invalid_target_cspace_position():
    """Test that `plan_to_cspace_target()` returns `INVALID_TARGET_CSPACE_POSITION`."""
    # UR10 is arbitrarily chosen to load a robot description for testing.
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')

    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    kinematics = robot_description.kinematics()
    num_coords = robot_description.num_cspace_coords()
    world = cumotion.create_world()
    world_view = world.add_world_view()

    tool_frame_name = robot_description.tool_frame_names()[0]
    config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                 tool_frame_name,
                                                                 world_view)
    optimizer = cumotion.create_trajectory_optimizer(config)
    initial_cspace_position = robot_description.default_cspace_configuration()

    Status = cumotion.TrajectoryOptimizer.Results.Status

    with errors_disabled:
        # Target outside c-space position limits.
        upper_limits = np.array([kinematics.cspace_coord_limits(i).upper
                                 for i in range(num_coords)])
        target_above_limits = upper_limits + 0.1
        cspace_target = cumotion.TrajectoryOptimizer.CSpaceTarget(target_above_limits)
        results = optimizer.plan_to_cspace_target(initial_cspace_position, cspace_target)
        assert results.status() == Status.INVALID_TARGET_CSPACE_POSITION
        assert results.trajectory() is None

        # Target with incorrect number of c-space coordinates.
        target_wrong_size = np.zeros(num_coords + 1)
        cspace_target = cumotion.TrajectoryOptimizer.CSpaceTarget(target_wrong_size)
        results = optimizer.plan_to_cspace_target(initial_cspace_position, cspace_target)
        assert results.status() == Status.INVALID_TARGET_CSPACE_POSITION
        assert results.trajectory() is None


def _tool_translation(kinematics, tool_frame_name, trajectory):
    """Compute the translation of `tool_frame_name` from the start to the end of `trajectory`."""
    domain = trajectory.domain()
    q0 = trajectory.eval(domain.lower)
    qf = trajectory.eval(domain.upper)
    x0 = kinematics.position(q0, tool_frame_name)
    xf = kinematics.position(qf, tool_frame_name)
    return np.linalg.norm(x0 - xf)


def test_ur10_short_move_cspace():
    """Plan a small c-space movement for UR10 and expect success."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')

    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()

    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    # Create task: small c-space movement (0.05 rad on joint 1).
    initial_configuration = robot_description.default_cspace_configuration()
    cspace_position_change = np.zeros(initial_configuration.shape[0])
    cspace_position_change[1] = 0.05
    target_cspace_position = initial_configuration + cspace_position_change

    translation_path_constraint = (
        cumotion.TrajectoryOptimizer.CSpaceTarget.TranslationPathConstraint.none()
    )
    orientation_path_constraint = (
        cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.none()
    )
    task = cumotion.TrajectoryOptimizer.CSpaceTarget(
        target_cspace_position, translation_path_constraint, orientation_path_constraint,
    )

    # Solve.
    results = optimizer.plan_to_cspace_target(initial_configuration, task)

    # Check solution.
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert results.trajectory().domain().span() < 0.8


@pytest.mark.skip(reason="Disabled in C++ (DISABLED_ShortMove_TaskSpace)")
def test_ur10_short_move_task_space():
    """Plan a small task-space movement for UR10 and expect success."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')

    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    kinematics = robot_description.kinematics()
    world = cumotion.create_world()
    world_view = world.add_world_view()

    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    # Create task: small task-space movement (50mm in x and y) with constant orientation.
    initial_configuration = robot_description.default_cspace_configuration()
    x0 = kinematics.position(initial_configuration, tool_frame_name)
    dx = np.array([50.0, 50.0, 0.0]) * 1e-3
    xf = x0 + dx

    translation_constraint = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.linear_path_constraint(xf)
    )
    orientation_constraint = cumotion.TrajectoryOptimizer.OrientationConstraint.constant()
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_constraint, orientation_constraint,
    )

    # Solve.
    results = optimizer.plan_to_task_space_target(initial_configuration, task)

    # Check solution.
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert results.trajectory().domain().span() < 0.3


def test_ur10_null_move_cspace():
    """Plan a null c-space movement (same start and end) for UR10 and expect success."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')

    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    kinematics = robot_description.kinematics()
    world = cumotion.create_world()
    world_view = world.add_world_view()

    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    assert optimizer_config.set_param("trajopt/pbo/enabled", False)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    # Create task: same start and end c-space position with path constraints.
    initial_configuration = robot_description.default_cspace_configuration()
    target_cspace_position = initial_configuration.copy()

    translation_path_constraint = (
        cumotion.TrajectoryOptimizer.CSpaceTarget.TranslationPathConstraint.linear()
    )
    orientation_path_constraint = (
        cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.axis(
            np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
        )
    )
    task = cumotion.TrajectoryOptimizer.CSpaceTarget(
        target_cspace_position, translation_path_constraint, orientation_path_constraint,
    )

    # Solve.
    results = optimizer.plan_to_cspace_target(initial_configuration, task)

    # Check solution.
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert results.trajectory().domain().span() < 0.1
    assert _tool_translation(kinematics, tool_frame_name, results.trajectory()) < 1e-4
    assert results.trajectory().domain().span() < 0.05


def test_ur10_null_move_task_space():
    """Plan a null task-space movement (same start and end) for UR10 and expect success."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')

    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    kinematics = robot_description.kinematics()
    world = cumotion.create_world()
    world_view = world.add_world_view()

    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    assert optimizer_config.set_param("trajopt/pbo/enabled", False)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    # Create task: target is the same as the initial tool position.
    initial_configuration = robot_description.default_cspace_configuration()
    x0 = kinematics.position(initial_configuration, tool_frame_name)

    translation_constraint = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.linear_path_constraint(x0)
    )
    orientation_constraint = cumotion.TrajectoryOptimizer.OrientationConstraint.constant()
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_constraint, orientation_constraint,
    )

    # Solve.
    results = optimizer.plan_to_task_space_target(initial_configuration, task)

    # Check solution.
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert results.trajectory().domain().span() < 0.05
    assert _tool_translation(kinematics, tool_frame_name, results.trajectory()) < 1e-4


def test_ur10_rotation_only_task_space():
    """Plan a pure rotation (no translation) for UR10 and expect success."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')

    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    kinematics = robot_description.kinematics()
    world = cumotion.create_world()
    world_view = world.add_world_view()

    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    assert optimizer_config.set_param("trajopt/pbo/enabled", False)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    # Create task: rotate 0.5 rad around X axis while keeping translation constant.
    initial_configuration = robot_description.default_cspace_configuration()
    initial_tool_pose = kinematics.pose(initial_configuration, tool_frame_name)
    change_in_rotation = cumotion.Rotation3.from_scaled_axis(np.array([0.5, 0.0, 0.0]))
    target_rotation = initial_tool_pose.rotation * change_in_rotation

    translation_constraint = (
        cumotion.TrajectoryOptimizer.TranslationConstraint.linear_path_constraint(
            initial_tool_pose.translation,
        )
    )
    orientation_constraint = (
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target(target_rotation)
    )
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
        translation_constraint, orientation_constraint,
    )

    # Solve.
    results = optimizer.plan_to_task_space_target(initial_configuration, task)

    # Check solution.
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None
    assert results.trajectory().domain().span() < 0.6
    assert _tool_translation(kinematics, tool_frame_name, results.trajectory()) < 1e-4

    # Check that the tool frame position is nearly constant during the rotation.
    timestep = 0.01
    domain = results.trajectory().domain()
    x0 = kinematics.position(initial_configuration, tool_frame_name)
    time = domain.lower
    while time <= domain.upper:
        xf = kinematics.position(results.trajectory().eval(time), tool_frame_name)
        assert np.linalg.norm(x0 - xf) < 6e-3  # Doesn't deviate more than 6mm.
        time += timestep


def test_franka_target_none():
    """Plan to a translation-only target for Franka and expect success."""
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'franka.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'franka.urdf')

    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    kinematics = robot_description.kinematics()
    world = cumotion.create_world()
    world_view = world.add_world_view()

    tool_frame_name = robot_description.tool_frame_names()[0]
    optimizer_config = cumotion.create_default_trajectory_optimizer_config(robot_description,
                                                                           tool_frame_name,
                                                                           world_view)
    optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
    assert optimizer is not None

    # Create task: translation target at the tool position for a known c-space configuration.
    q_target = np.array([0.8782496796035466, 1.0471975511965976, 0.0, -1.0471975511965976,
                         0.0, 2.0943951023931953, 0.5235987755982988])
    position_target = kinematics.position(q_target, tool_frame_name)
    translation_target = cumotion.TrajectoryOptimizer.TranslationConstraint.target(position_target)
    orientation_target = cumotion.TrajectoryOptimizer.OrientationConstraint.none()
    task = cumotion.TrajectoryOptimizer.TaskSpaceTarget(translation_target, orientation_target)

    # Solve.
    initial_configuration = robot_description.default_cspace_configuration()
    results = optimizer.plan_to_task_space_target(initial_configuration, task)

    # Check solution.
    assert results is not None
    assert results.status() == cumotion.TrajectoryOptimizer.Results.Status.SUCCESS
    assert results.trajectory() is not None

    # Sanity check that the final tool position aligns with the target.
    qf = results.trajectory().eval(results.trajectory().domain().upper)
    final_tool_position = kinematics.position(qf, tool_frame_name)
    assert np.linalg.norm(final_tool_position - position_target) < 1e-6
    assert abs(results.trajectory().domain().span() - 1.5) < 0.2
