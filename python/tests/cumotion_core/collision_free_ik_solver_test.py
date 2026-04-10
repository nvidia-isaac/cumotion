# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Unit tests for collision_free_ik_solver_python.h."""

# Standard Library
import os
from collections import Counter

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion

# Local directory
from ._test_helper import CUMOTION_ROOT_DIR, errors_disabled


# Test constants
CSPACE_POSITION_TOLERANCE = 1e-12
POSITION_TOLERANCE = 1e-3  # 1mm
ORIENTATION_TOLERANCE = 5e-3  # ~0.29 degrees
NUM_SEEDS = 400  # Number of IK seeds to use


@pytest.fixture
def configure_franka_robot_description():
    """Test fixture to configure Franka robot description object."""
    def _configure_franka_robot_description():
        # Set directory for robot description YAML files.
        config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')

        # Set absolute path to the XRDF for Franka robot.
        xrdf_path = os.path.join(config_path, 'franka.xrdf')

        # Set absolute path to URDF for Franka robot.
        urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka',
                                 'franka.urdf')

        # Load and return robot description.
        return cumotion.load_robot_from_file(xrdf_path, urdf_path)

    return _configure_franka_robot_description


@pytest.fixture
def configure_franka_with_obstacles(configure_franka_robot_description):
    """Test fixture to configure Franka with a world containing obstacles."""
    def _configure_franka_with_obstacles():
        robot_description = configure_franka_robot_description()

        # Create world and add obstacles (similar to C++ test)
        world = cumotion.create_world()

        # Add two cuboid obstacles
        obstacle_configs = [
            (np.array([0.4, 0.0, 0.4]), np.array([0.2, 0.6, 0.2])),
            (np.array([0.0, 0.0, 0.8]), np.array([0.2, 0.2, 0.2]))
        ]

        for position, side_lengths in obstacle_configs:
            obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
            obstacle.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, side_lengths)
            obstacle_pose = cumotion.Pose3.from_translation(position)
            world.add_obstacle(obstacle, obstacle_pose)

        world_view = world.add_world_view()

        return robot_description, world, world_view

    return _configure_franka_with_obstacles


def compute_mode(indices):
    """Find the mode (most frequent value) in a list of indices."""
    counter = Counter(indices)
    return counter.most_common(1)[0][0]


def sample_reachable_poses(kinematics, frame_name, num_samples, seed=42):
    """Sample reachable poses by randomly sampling c-space and running FK."""
    np.random.seed(seed)

    num_coords = kinematics.num_cspace_coords()

    # Get lower and upper limits for each coordinate
    lower_limits = np.array([kinematics.cspace_coord_limits(i).lower for i in range(num_coords)])
    upper_limits = np.array([kinematics.cspace_coord_limits(i).upper for i in range(num_coords)])

    cspace_samples = []
    poses = []

    for _ in range(num_samples):
        # Sample c-space position within limits
        q = np.random.uniform(lower_limits, upper_limits)
        cspace_samples.append(q)

        # Compute forward kinematics
        pose = kinematics.pose(q, frame_name)
        poses.append(pose)

    return cspace_samples, poses


def validate_ik_solution(kinematics, inspector, solution, frame_name,
                         target_position, target_orientation=None,
                         is_axis_constraint=False,
                         position_tolerance=POSITION_TOLERANCE,
                         orientation_tolerance=ORIENTATION_TOLERANCE):
    """Validate an IK solution.

    Validate that an IK solution is:
    - Collision-free
    - Within c-space limits
    - Meets position and orientation targets within tolerances
    """
    # Check collision-free and within limits
    assert not inspector.in_self_collision(solution), "Solution is in self-collision"
    assert not inspector.in_collision_with_obstacle(solution), "Solution collides with obstacle"
    assert kinematics.within_cspace_limits(solution, False), "Solution violates c-space limits"

    # Check position target
    pose = kinematics.pose(solution, frame_name)
    position_error = np.linalg.norm(pose.translation - target_position)
    assert position_error < position_tolerance, \
        f"Position error {position_error} exceeds tolerance {position_tolerance}"

    if target_orientation is not None:
        if is_axis_constraint:
            # For axis constraints, check that the Z axis aligns
            # Extract Z axis from both orientations
            solution_z_axis = pose.rotation.matrix()[:, 2]
            target_z_axis = target_orientation.matrix()[:, 2]

            # Compute angle between axes
            dot_product = np.clip(np.dot(solution_z_axis, target_z_axis), -1.0, 1.0)
            axis_angle_error = np.arccos(dot_product)

            assert axis_angle_error < orientation_tolerance, \
                f"Axis angle error {axis_angle_error} exceeds tolerance {orientation_tolerance}"
        else:
            # For full orientation constraints, check rotation distance
            rotation_error = cumotion.Rotation3.distance(target_orientation, pose.rotation)

            assert rotation_error < orientation_tolerance, \
                f"Rotation error {rotation_error} exceeds tolerance {orientation_tolerance}"


def expect_unique_solutions(solutions, tolerance=1e-8):
    """Verify that no pair of c-space positions are within the given tolerance of each other."""
    for i in range(len(solutions)):
        for j in range(i):
            squared_error = np.sum((solutions[i] - solutions[j]) ** 2)
            assert squared_error > tolerance, \
                f"Solutions {i} and {j} are not unique (squared error: {squared_error})"


def test_collision_free_ik_solver_config(configure_franka_robot_description):
    """Test CollisionFreeIkSolverConfig parameter setting."""
    # Load robot description
    robot_description = configure_franka_robot_description()

    # Create world and world view
    world = cumotion.create_world()
    world_view = world.add_world_view()

    # Get tool frame name
    tool_frame_name = robot_description.tool_frame_names()[0]

    # Create a config
    config = cumotion.create_default_collision_free_ik_solver_config(
        robot_description, tool_frame_name, world_view)

    assert config is not None

    # Test setting "task_space_position_tolerance".
    assert config.set_param("task_space_position_tolerance", 0.002)

    with errors_disabled:
        assert not config.set_param("task_space_position_tolerance", 2)  # Wrong type

        assert not config.set_param("task_space_position_tolerance", -0.001)  # Negative
        assert not config.set_param("task_space_position_tolerance", 0.0)  # Zero

    # Test setting "task_space_orientation_tolerance".
    assert config.set_param("task_space_orientation_tolerance", 0.01)

    with errors_disabled:
        assert not config.set_param("task_space_orientation_tolerance", 2)  # Wrong type

        assert not config.set_param("task_space_orientation_tolerance", -0.001)  # Negative
        assert not config.set_param("task_space_orientation_tolerance", 0.0)  # Zero

    # Test setting "num_seeds".
    assert config.set_param("num_seeds", 50)

    with errors_disabled:
        assert not config.set_param("num_seeds", 2.0)  # Wrong type

        assert not config.set_param("num_seeds", -10)  # Negative
        assert not config.set_param("num_seeds", 0)  # Zero

    # Test setting "max_reattempts".
    assert config.set_param("max_reattempts", 0)  # Zero is allowed
    assert config.set_param("max_reattempts", 5)

    with errors_disabled:
        assert not config.set_param("max_reattempts", 2.0)  # Wrong type
        assert not config.set_param("max_reattempts", -1)  # Negative

    # Test setting an invalid parameter name
    with errors_disabled:
        assert not config.set_param("invalid_param_name", 1.0)


def test_franka_position_only_constraints(configure_franka_with_obstacles):
    """Test IK solver with position-only constraints."""
    robot_description, world, world_view = configure_franka_with_obstacles()

    # Use panda_rightfinger as tool frame (depends on auxiliary c-space coordinate)
    tool_frame_name = "panda_rightfinger"
    kinematics = robot_description.kinematics()

    # Create config and set parameters
    config = cumotion.create_default_collision_free_ik_solver_config(
        robot_description, tool_frame_name, world_view)
    config.set_param("task_space_position_tolerance", POSITION_TOLERANCE)
    config.set_param("num_seeds", NUM_SEEDS)

    # Create IK solver
    ik_solver = cumotion.create_collision_free_ik_solver(config)

    # Create a solver with `max_reattempts = 0` to compare `solve_array()` and `solve()` results.
    # NOTE: Currently, batch mode (enabled via `solve_array()`) uses `max_reattempts = 0`, which
    # affects Halton generator capacity. Using the same `max_reattempts` for the serial baseline
    # ensures identical initial seeds, and identical results (within numerical tolerance).
    config_zero_reattempts = cumotion.create_default_collision_free_ik_solver_config(
        robot_description, tool_frame_name, world_view)
    config_zero_reattempts.set_param("task_space_position_tolerance", POSITION_TOLERANCE)
    config_zero_reattempts.set_param("num_seeds", NUM_SEEDS)
    config_zero_reattempts.set_param("max_reattempts", 0)
    ik_solver_zero_reattempts = cumotion.create_collision_free_ik_solver(config_zero_reattempts)

    # Create robot world inspector
    inspector = cumotion.create_robot_world_inspector(robot_description, world_view)

    # Sample reachable poses
    num_targets = 5
    cspace_samples, poses = sample_reachable_poses(
        kinematics, tool_frame_name, num_targets, seed=41252)

    # Extract translation targets
    translation_targets = [pose.translation for pose in poses]

    # Test goalset with position-only constraints using `solve_array()`.
    translation_goalset_array = cumotion.CollisionFreeIkSolver.TranslationConstraintArray.target(
        [translation_targets])
    orientation_goalset_array = cumotion.CollisionFreeIkSolver.OrientationConstraintArray.none()
    task_space_target_goalsest_array = cumotion.CollisionFreeIkSolver.TaskSpaceTargetArray(
        translation_goalset_array, orientation_goalset_array)

    results_goalset_array = ik_solver.solve_array(task_space_target_goalsest_array)

    assert results_goalset_array.num_problems() == 1
    assert results_goalset_array.num_successes() == 1
    first_result_array = results_goalset_array.problem(0)
    assert first_result_array.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    solutions_array = first_result_array.cspace_positions()
    assert len(solutions_array) > 0

    goalset_array_target_indices = first_result_array.target_indices()

    for solution, target_idx in zip(solutions_array, goalset_array_target_indices):
        validate_ik_solution(kinematics, inspector, solution, tool_frame_name,
                             translation_targets[target_idx], None, False,
                             POSITION_TOLERANCE, ORIENTATION_TOLERANCE)
    expect_unique_solutions(solutions_array)

    # DEPRECATED section. --------------------------------------------------------------------------
    # Test goalset with position-only constraints using `solve_goalset()`.
    translation_goalset = cumotion.CollisionFreeIkSolver.TranslationConstraintGoalset.target(
        translation_targets)
    orientation_goalset = cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.none()
    task_space_target_goalset = cumotion.CollisionFreeIkSolver.TaskSpaceTargetGoalset(
        translation_goalset, orientation_goalset)

    goalset_results = ik_solver.solve_goalset(task_space_target_goalset)

    assert goalset_results.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    goalset_solutions = goalset_results.cspace_positions()
    assert len(goalset_solutions) > 0

    goalset_target_indices = goalset_results.target_indices()

    # Validate each solution
    for solution, target_idx in zip(goalset_solutions, goalset_target_indices):
        validate_ik_solution(kinematics, inspector, solution, tool_frame_name,
                             translation_targets[target_idx], None, False,
                             POSITION_TOLERANCE, ORIENTATION_TOLERANCE)

    # Verify solutions are unique
    expect_unique_solutions(goalset_solutions)

    # Validate that the goalset results match the goalset array results.
    assert goalset_results.status() == first_result_array.status()
    assert len(goalset_solutions) == len(solutions_array)
    assert list(goalset_target_indices) == list(goalset_array_target_indices)
    # End of DEPRECATED section.  ------------------------------------------------------------------

    # Test single target
    reached_target_idx = compute_mode(goalset_array_target_indices)
    reached_pose = poses[reached_target_idx]

    translation_target = cumotion.CollisionFreeIkSolver.TranslationConstraint.target(
        reached_pose.translation)
    orientation_target = cumotion.CollisionFreeIkSolver.OrientationConstraint.none()
    task_space_target = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
        translation_target, orientation_target)

    results = ik_solver.solve(task_space_target)

    assert results.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    solutions = results.cspace_positions()
    assert len(solutions) > 0

    # Validate each solution
    for solution in solutions:
        validate_ik_solution(kinematics, inspector, solution, tool_frame_name,
                             reached_pose.translation, None, False,
                             POSITION_TOLERANCE, ORIENTATION_TOLERANCE)

    # Verify solutions are unique
    expect_unique_solutions(solutions)

    # Test batch of single targets using `solve_array()`.

    # Slightly modify the translation of the `reached_pose`.
    modified_pose = cumotion.Pose3.from_translation(
        reached_pose.translation + np.array([0.01, -0.01, 0.01]))

    # Solve the modified target using `solve()`.
    modified_translation_target = cumotion.CollisionFreeIkSolver.TranslationConstraint.target(
        modified_pose.translation)
    modified_task_space_target = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
        modified_translation_target, orientation_target)
    modified_results = ik_solver.solve(modified_task_space_target)
    assert modified_results.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    modified_solutions = modified_results.cspace_positions()
    assert len(modified_solutions) > 0

    # Validate each modified solution
    for solution in modified_solutions:
        validate_ik_solution(kinematics, inspector, solution, tool_frame_name,
                             modified_pose.translation, None, False,
                             POSITION_TOLERANCE, ORIENTATION_TOLERANCE)

    # Verify modified solutions are unique
    expect_unique_solutions(modified_solutions)

    # Solve both targets serially using `ik_solver_zero_reattempts` (matching batch config).
    results_serial_0 = ik_solver_zero_reattempts.solve(task_space_target)
    assert results_serial_0.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    results_serial_1 = ik_solver_zero_reattempts.solve(modified_task_space_target)
    assert results_serial_1.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS

    # Create array of results from `solve()` for comparison with `solve_array()`.
    results_solve = [results_serial_0, results_serial_1]

    # Create batch translation targets with the original and modified targets.
    translation_targets_batch_array = [[reached_pose.translation], [modified_pose.translation]]
    translation_batch_array = cumotion.CollisionFreeIkSolver.TranslationConstraintArray.target(
        translation_targets_batch_array)
    orientation_batch_array = cumotion.CollisionFreeIkSolver.OrientationConstraintArray.none()
    task_space_target_array = cumotion.CollisionFreeIkSolver.TaskSpaceTargetArray(
        translation_batch_array, orientation_batch_array)

    results_batch_array = ik_solver.solve_array(task_space_target_array)
    num_problems_batch = 2
    assert results_batch_array.num_problems() == num_problems_batch
    assert results_batch_array.num_successes() == num_problems_batch

    # Verify that each problem's results match exactly the corresponding single-target `solve()`.
    for problem_index in range(num_problems_batch):
        result_batch_array = results_batch_array.problem(problem_index)
        result_solve = results_solve[problem_index]
        assert result_batch_array.status() == result_solve.status()
        assert list(result_batch_array.target_indices()) == list(result_solve.target_indices())
        problem_solutions_array = result_batch_array.cspace_positions()
        assert len(problem_solutions_array) == len(result_solve.cspace_positions())
        for solution_solve, solution_array in zip(result_solve.cspace_positions(),
                                                  problem_solutions_array):
            assert np.allclose(solution_solve, solution_array, atol=CSPACE_POSITION_TOLERANCE)


def test_franka_axis_alignment_constraints(configure_franka_with_obstacles):
    """Test IK solver with axis alignment constraints."""
    robot_description, world, world_view = configure_franka_with_obstacles()

    tool_frame_name = "panda_rightfinger"
    kinematics = robot_description.kinematics()

    # Create config and set parameters
    config = cumotion.create_default_collision_free_ik_solver_config(
        robot_description, tool_frame_name, world_view)
    config.set_param("task_space_position_tolerance", POSITION_TOLERANCE)
    config.set_param("task_space_orientation_tolerance", ORIENTATION_TOLERANCE)
    config.set_param("num_seeds", NUM_SEEDS)

    # Create IK solver
    ik_solver = cumotion.create_collision_free_ik_solver(config)

    # Create a solver with `max_reattempts = 0` to compare `solve_array()` and `solve()` results.
    config_zero_reattempts = cumotion.create_default_collision_free_ik_solver_config(
        robot_description, tool_frame_name, world_view)
    config_zero_reattempts.set_param("task_space_position_tolerance", POSITION_TOLERANCE)
    config_zero_reattempts.set_param("task_space_orientation_tolerance", ORIENTATION_TOLERANCE)
    config_zero_reattempts.set_param("num_seeds", NUM_SEEDS)
    config_zero_reattempts.set_param("max_reattempts", 0)
    ik_solver_zero_reattempts = cumotion.create_collision_free_ik_solver(config_zero_reattempts)

    # Create robot world inspector
    inspector = cumotion.create_robot_world_inspector(robot_description, world_view)

    # Sample reachable poses
    num_targets = 5
    cspace_samples, poses = sample_reachable_poses(
        kinematics, tool_frame_name, num_targets, seed=41252)

    # Extract translation and rotation targets
    translation_targets = [pose.translation for pose in poses]
    rotation_targets = [pose.rotation for pose in poses]

    # Create axis alignment constraints (Z axis)
    tool_frame_axes = [np.array([0.0, 0.0, 1.0]) for _ in range(num_targets)]
    world_target_axes = [rotation.matrix()[:, 2] for rotation in rotation_targets]

    # Test goalset with axis alignment constraints using `solve_array()`.
    translation_goalset_array = cumotion.CollisionFreeIkSolver.TranslationConstraintArray.target(
        [translation_targets])
    orientation_goalset_array = cumotion.CollisionFreeIkSolver.OrientationConstraintArray.axis(
        [tool_frame_axes], [world_target_axes])
    task_space_target_goalset_array = cumotion.CollisionFreeIkSolver.TaskSpaceTargetArray(
        translation_goalset_array, orientation_goalset_array)

    results_goalset_array = ik_solver.solve_array(task_space_target_goalset_array)

    assert results_goalset_array.num_problems() == 1
    assert results_goalset_array.num_successes() == 1
    first_result_array = results_goalset_array.problem(0)
    assert first_result_array.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    solutions_array = first_result_array.cspace_positions()
    assert len(solutions_array) > 0

    goalset_array_target_indices = first_result_array.target_indices()

    for solution, target_idx in zip(solutions_array, goalset_array_target_indices):
        validate_ik_solution(kinematics, inspector, solution, tool_frame_name,
                             translation_targets[target_idx], rotation_targets[target_idx],
                             True, POSITION_TOLERANCE, ORIENTATION_TOLERANCE)
    expect_unique_solutions(solutions_array)

    # DEPRECATED section. --------------------------------------------------------------------------
    # Test goalset with axis alignment constraints using `solve_goalset()`.
    translation_goalset = cumotion.CollisionFreeIkSolver.TranslationConstraintGoalset.target(
        translation_targets)
    orientation_goalset = cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.axis(
        tool_frame_axes, world_target_axes)
    task_space_target_goalset = cumotion.CollisionFreeIkSolver.TaskSpaceTargetGoalset(
        translation_goalset, orientation_goalset)

    goalset_results = ik_solver.solve_goalset(task_space_target_goalset)

    assert goalset_results.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    goalset_solutions = goalset_results.cspace_positions()
    assert len(goalset_solutions) > 0

    goalset_target_indices = goalset_results.target_indices()

    # Validate each solution
    for solution, target_idx in zip(goalset_solutions, goalset_target_indices):
        validate_ik_solution(kinematics, inspector, solution, tool_frame_name,
                             translation_targets[target_idx], rotation_targets[target_idx],
                             True, POSITION_TOLERANCE, ORIENTATION_TOLERANCE)

    # Verify solutions are unique
    expect_unique_solutions(goalset_solutions)

    # Validate that the goalset results match the goalset array results.
    assert goalset_results.status() == first_result_array.status()
    assert len(goalset_solutions) == len(solutions_array)
    assert list(goalset_target_indices) == list(goalset_array_target_indices)
    # End of DEPRECATED section.  ------------------------------------------------------------------

    # Test single target
    reached_target_idx = compute_mode(goalset_array_target_indices)
    reached_pose = poses[reached_target_idx]

    translation_target = cumotion.CollisionFreeIkSolver.TranslationConstraint.target(
        reached_pose.translation)
    world_target_axis = reached_pose.rotation.matrix()[:, 2]
    orientation_target = cumotion.CollisionFreeIkSolver.OrientationConstraint.axis(
        np.array([0.0, 0.0, 1.0]), world_target_axis)
    task_space_target = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
        translation_target, orientation_target)

    results = ik_solver.solve(task_space_target)

    assert results.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    solutions = results.cspace_positions()
    assert len(solutions) > 0

    # Validate each solution
    for solution in solutions:
        validate_ik_solution(kinematics, inspector, solution, tool_frame_name,
                             reached_pose.translation, reached_pose.rotation,
                             True, POSITION_TOLERANCE, ORIENTATION_TOLERANCE)

    # Verify solutions are unique
    expect_unique_solutions(solutions)

    # Test batch of single targets using `solve_array()`.

    # Slightly modify the translation and axis of the `reached_pose`.
    modified_pose = cumotion.Pose3.from_translation(
        reached_pose.translation + np.array([0.01, -0.01, 0.01]))
    # Rotate the world target axis slightly.
    small_rotation = cumotion.Rotation3.from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.1)
    modified_world_axis = small_rotation.matrix() @ world_target_axis

    # Solve the modified target using `solve()`.
    modified_translation_target = cumotion.CollisionFreeIkSolver.TranslationConstraint.target(
        modified_pose.translation)
    modified_orientation_target = cumotion.CollisionFreeIkSolver.OrientationConstraint.axis(
        np.array([0.0, 0.0, 1.0]), modified_world_axis)
    modified_task_space_target = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
        modified_translation_target, modified_orientation_target)
    modified_results = ik_solver.solve(modified_task_space_target)
    assert modified_results.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    modified_solutions = modified_results.cspace_positions()
    assert len(modified_solutions) > 0

    # Verify modified solutions are unique
    expect_unique_solutions(modified_solutions)

    # Solve both targets serially using `ik_solver_zero_reattempts` (matching batch config).
    results_serial_0 = ik_solver_zero_reattempts.solve(task_space_target)
    assert results_serial_0.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    results_serial_1 = ik_solver_zero_reattempts.solve(modified_task_space_target)
    assert results_serial_1.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS

    # Create array of results from `solve()` for comparison with `solve_array()`.
    results_solve = [results_serial_0, results_serial_1]

    # Create batch targets with the original and modified targets.
    translation_targets_batch_array = [[reached_pose.translation], [modified_pose.translation]]
    tool_frame_axes_batch = [[np.array([0.0, 0.0, 1.0])], [np.array([0.0, 0.0, 1.0])]]
    world_frame_axes_batch = [[world_target_axis], [modified_world_axis]]

    translation_batch_array = cumotion.CollisionFreeIkSolver.TranslationConstraintArray.target(
        translation_targets_batch_array)
    orientation_batch_array = cumotion.CollisionFreeIkSolver.OrientationConstraintArray.axis(
        tool_frame_axes_batch, world_frame_axes_batch)
    task_space_target_array = cumotion.CollisionFreeIkSolver.TaskSpaceTargetArray(
        translation_batch_array, orientation_batch_array)

    results_batch_array = ik_solver.solve_array(task_space_target_array)
    num_problems_batch = 2
    assert results_batch_array.num_problems() == num_problems_batch
    assert results_batch_array.num_successes() == num_problems_batch

    # Verify that each problem's results match exactly the corresponding single-target `solve()`.
    for problem_index in range(num_problems_batch):
        result_batch_array = results_batch_array.problem(problem_index)
        result_solve = results_solve[problem_index]
        assert result_batch_array.status() == result_solve.status()
        assert list(result_batch_array.target_indices()) == list(result_solve.target_indices())
        problem_solutions_array = result_batch_array.cspace_positions()
        assert len(problem_solutions_array) == len(result_solve.cspace_positions())
        for solution_solve, solution_array in zip(result_solve.cspace_positions(),
                                                  problem_solutions_array):
            assert np.allclose(solution_solve, solution_array, atol=CSPACE_POSITION_TOLERANCE)


def test_franka_full_orientation_constraints(configure_franka_with_obstacles):
    """Test IK solver with full orientation constraints."""
    robot_description, world, world_view = configure_franka_with_obstacles()

    tool_frame_name = "panda_rightfinger"
    kinematics = robot_description.kinematics()

    # Create config and set parameters
    config = cumotion.create_default_collision_free_ik_solver_config(
        robot_description, tool_frame_name, world_view)
    config.set_param("task_space_position_tolerance", POSITION_TOLERANCE)
    config.set_param("task_space_orientation_tolerance", ORIENTATION_TOLERANCE)
    config.set_param("num_seeds", NUM_SEEDS)

    # Create IK solver
    ik_solver = cumotion.create_collision_free_ik_solver(config)

    # Create a solver with `max_reattempts = 0` to compare `solve_array()` and `solve()` results.
    config_zero_reattempts = cumotion.create_default_collision_free_ik_solver_config(
        robot_description, tool_frame_name, world_view)
    config_zero_reattempts.set_param("task_space_position_tolerance", POSITION_TOLERANCE)
    config_zero_reattempts.set_param("task_space_orientation_tolerance", ORIENTATION_TOLERANCE)
    config_zero_reattempts.set_param("num_seeds", NUM_SEEDS)
    config_zero_reattempts.set_param("max_reattempts", 0)
    ik_solver_zero_reattempts = cumotion.create_collision_free_ik_solver(config_zero_reattempts)

    # Create robot world inspector
    inspector = cumotion.create_robot_world_inspector(robot_description, world_view)

    # Sample reachable poses
    num_targets = 5
    cspace_samples, poses = sample_reachable_poses(
        kinematics, tool_frame_name, num_targets, seed=41252)

    # Extract translation and rotation targets
    translation_targets = [pose.translation for pose in poses]
    rotation_targets = [pose.rotation for pose in poses]

    # Test goalset with full orientation constraints using `solve_array()`.
    translation_goalset_array = cumotion.CollisionFreeIkSolver.TranslationConstraintArray.target(
        [translation_targets])
    orientation_goalset_array = cumotion.CollisionFreeIkSolver.OrientationConstraintArray.target(
        [rotation_targets])
    task_space_target_goalset_array = cumotion.CollisionFreeIkSolver.TaskSpaceTargetArray(
        translation_goalset_array, orientation_goalset_array)

    results_goalset_array = ik_solver.solve_array(task_space_target_goalset_array)

    assert results_goalset_array.num_problems() == 1
    assert results_goalset_array.num_successes() == 1
    first_result_array = results_goalset_array.problem(0)
    assert first_result_array.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    solutions_array = first_result_array.cspace_positions()
    assert len(solutions_array) > 0

    goalset_array_target_indices = first_result_array.target_indices()

    for solution, target_idx in zip(solutions_array, goalset_array_target_indices):
        validate_ik_solution(kinematics, inspector, solution, tool_frame_name,
                             translation_targets[target_idx], rotation_targets[target_idx],
                             False, POSITION_TOLERANCE, ORIENTATION_TOLERANCE)
    expect_unique_solutions(solutions_array)

    # DEPRECATED section. --------------------------------------------------------------------------
    # Test goalset with full orientation constraints using `solve_goalset()`.
    translation_goalset = cumotion.CollisionFreeIkSolver.TranslationConstraintGoalset.target(
        translation_targets)
    orientation_goalset = cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.target(
        rotation_targets)
    task_space_target_goalset = cumotion.CollisionFreeIkSolver.TaskSpaceTargetGoalset(
        translation_goalset, orientation_goalset)

    goalset_results = ik_solver.solve_goalset(task_space_target_goalset)

    assert goalset_results.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    goalset_solutions = goalset_results.cspace_positions()
    assert len(goalset_solutions) > 0

    goalset_target_indices = goalset_results.target_indices()

    # Validate each solution
    for solution, target_idx in zip(goalset_solutions, goalset_target_indices):
        validate_ik_solution(kinematics, inspector, solution, tool_frame_name,
                             translation_targets[target_idx], rotation_targets[target_idx],
                             False, POSITION_TOLERANCE, ORIENTATION_TOLERANCE)

    # Verify solutions are unique
    expect_unique_solutions(goalset_solutions)

    # Validate that the goalset results match the goalset array results.
    assert goalset_results.status() == first_result_array.status()
    assert len(goalset_solutions) == len(solutions_array)
    assert list(goalset_target_indices) == list(goalset_array_target_indices)
    # End of DEPRECATED section.  ------------------------------------------------------------------

    # Test single target
    reached_target_idx = compute_mode(goalset_array_target_indices)
    reached_pose = poses[reached_target_idx]

    translation_target = cumotion.CollisionFreeIkSolver.TranslationConstraint.target(
        reached_pose.translation)
    orientation_target = cumotion.CollisionFreeIkSolver.OrientationConstraint.target(
        reached_pose.rotation)
    task_space_target = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
        translation_target, orientation_target)

    results = ik_solver.solve(task_space_target)

    assert results.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    solutions = results.cspace_positions()
    assert len(solutions) > 0

    # Validate each solution
    for solution in solutions:
        validate_ik_solution(kinematics, inspector, solution, tool_frame_name,
                             reached_pose.translation, reached_pose.rotation,
                             False, POSITION_TOLERANCE, ORIENTATION_TOLERANCE)

    # Verify solutions are unique
    expect_unique_solutions(solutions)

    # Test batch of single targets using `solve_array()`.

    # Slightly modify the translation and orientation of the `reached_pose`.
    modified_pose = cumotion.Pose3.from_translation(
        reached_pose.translation + np.array([0.01, -0.01, 0.01]))
    # Rotate the orientation slightly.
    small_rotation = cumotion.Rotation3.from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.1)
    modified_rotation = cumotion.Rotation3.from_matrix(
        small_rotation.matrix() @ reached_pose.rotation.matrix())

    # Solve the modified target using `solve()`.
    modified_translation_target = cumotion.CollisionFreeIkSolver.TranslationConstraint.target(
        modified_pose.translation)
    modified_orientation_target = cumotion.CollisionFreeIkSolver.OrientationConstraint.target(
        modified_rotation)
    modified_task_space_target = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
        modified_translation_target, modified_orientation_target)
    modified_results = ik_solver.solve(modified_task_space_target)
    assert modified_results.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    modified_solutions = modified_results.cspace_positions()
    assert len(modified_solutions) > 0

    # Verify modified solutions are unique
    expect_unique_solutions(modified_solutions)

    # Solve both targets serially using `ik_solver_zero_reattempts` (matching batch config).
    results_serial_0 = ik_solver_zero_reattempts.solve(task_space_target)
    assert results_serial_0.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    results_serial_1 = ik_solver_zero_reattempts.solve(modified_task_space_target)
    assert results_serial_1.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS

    # Create array of results from `solve()` for comparison with `solve_array()`.
    results_solve = [results_serial_0, results_serial_1]

    # Create batch targets with the original and modified targets.
    translation_targets_batch_array = [[reached_pose.translation], [modified_pose.translation]]
    orientation_targets_batch_array = [[reached_pose.rotation], [modified_rotation]]

    translation_batch_array = cumotion.CollisionFreeIkSolver.TranslationConstraintArray.target(
        translation_targets_batch_array)
    orientation_batch_array = cumotion.CollisionFreeIkSolver.OrientationConstraintArray.target(
        orientation_targets_batch_array)
    task_space_target_array = cumotion.CollisionFreeIkSolver.TaskSpaceTargetArray(
        translation_batch_array, orientation_batch_array)

    results_batch_array = ik_solver.solve_array(task_space_target_array)
    num_problems_batch = 2
    assert results_batch_array.num_problems() == num_problems_batch
    assert results_batch_array.num_successes() == num_problems_batch

    # Verify that each problem's results match exactly the corresponding single-target `solve()`.
    for problem_index in range(num_problems_batch):
        result_batch_array = results_batch_array.problem(problem_index)
        result_solve = results_solve[problem_index]
        assert result_batch_array.status() == result_solve.status()
        assert list(result_batch_array.target_indices()) == list(result_solve.target_indices())
        problem_solutions_array = result_batch_array.cspace_positions()
        assert len(problem_solutions_array) == len(result_solve.cspace_positions())
        for solution_solve, solution_array in zip(result_solve.cspace_positions(),
                                                  problem_solutions_array):
            assert np.allclose(solution_solve, solution_array, atol=CSPACE_POSITION_TOLERANCE)


def test_franka_custom_seeds(configure_franka_with_obstacles):
    """Test IK solver with custom seeds."""
    robot_description, world, world_view = configure_franka_with_obstacles()

    tool_frame_name = "panda_rightfinger"
    kinematics = robot_description.kinematics()
    num_cspace_coords = robot_description.num_cspace_coords()

    # Create config and set parameters
    config = cumotion.create_default_collision_free_ik_solver_config(
        robot_description, tool_frame_name, world_view)
    config.set_param("task_space_position_tolerance", POSITION_TOLERANCE)
    config.set_param("task_space_orientation_tolerance", ORIENTATION_TOLERANCE)
    config.set_param("num_seeds", NUM_SEEDS)

    # Create IK solver
    ik_solver = cumotion.create_collision_free_ik_solver(config)

    # Sample reachable poses
    num_targets = 5
    cspace_samples, poses = sample_reachable_poses(
        kinematics, tool_frame_name, num_targets, seed=34125)

    # Test with invalid custom seeds (wrong size)
    translation_target = cumotion.CollisionFreeIkSolver.TranslationConstraint.target(
        poses[0].translation)
    orientation_target = cumotion.CollisionFreeIkSolver.OrientationConstraint.target(
        poses[0].rotation)
    task_space_target = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
        translation_target, orientation_target)

    custom_seeds_invalid = [
        np.zeros(num_cspace_coords),
        np.zeros(num_cspace_coords - 1)  # Wrong size
    ]

    # Assert that calling `ik_solver.solve()` with invalid custom seeds raises an Exception.
    with pytest.raises(Exception):
        ik_solver.solve(task_space_target, custom_seeds_invalid)

    # Test with valid custom seeds
    num_custom_seeds = NUM_SEEDS // 2
    custom_seeds = [cspace_samples[0] for _ in range(num_custom_seeds)]

    # Solve without custom seeds
    results_no_seeds = ik_solver.solve(task_space_target)
    assert results_no_seeds.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS

    # Solve with custom seeds
    results_with_seeds = ik_solver.solve(task_space_target, custom_seeds)
    assert results_with_seeds.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS

    # Custom seeds affect the optimization, so we expect different results
    # (The number of solutions can be higher or lower depending on the seed quality).
    assert len(results_with_seeds.cspace_positions()) != len(results_no_seeds.cspace_positions())

    translation_targets = [pose.translation for pose in poses]
    rotation_targets = [pose.rotation for pose in poses]

    # Test with custom seeds using `solve_array()`.
    translation_targets_goalset_array = [translation_targets]
    rotation_targets_goalset_array = [rotation_targets]
    translation_constraint_goalset_array = \
        cumotion.CollisionFreeIkSolver.TranslationConstraintArray.target(
            translation_targets_goalset_array)
    orientation_constraint_goalset_array = \
        cumotion.CollisionFreeIkSolver.OrientationConstraintArray.target(
            rotation_targets_goalset_array)
    task_space_target_goalset_array = cumotion.CollisionFreeIkSolver.TaskSpaceTargetArray(
        translation_constraint_goalset_array, orientation_constraint_goalset_array)
    results_array = ik_solver.solve_array(task_space_target_goalset_array)
    assert results_array.num_problems() == 1
    assert results_array.num_successes() == 1
    first_result_array = results_array.problem(0)
    assert first_result_array.status() == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    solutions_array = first_result_array.cspace_positions()
    assert len(solutions_array) > 0
    target_indices_goalset_array = first_result_array.target_indices()

    # Find the easiest target and create custom seeds from it
    easiest_target_idx_array = compute_mode(target_indices_goalset_array)
    custom_seeds_array = [cspace_samples[easiest_target_idx_array] for _ in range(num_custom_seeds)]

    # Solve with custom seeds
    results_with_seeds_array = ik_solver.solve_array(task_space_target_goalset_array,
                                                     custom_seeds_array)
    assert results_with_seeds_array.num_problems() == 1
    assert results_with_seeds_array.num_successes() == 1
    first_result_with_seeds_array = results_with_seeds_array.problem(0)
    assert first_result_with_seeds_array.status() == \
        cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS

    # Custom seeds affect the optimization, so we expect different results
    # (The number of solutions can be higher or lower depending on the seed quality)
    assert len(first_result_with_seeds_array.cspace_positions()) != \
        len(first_result_array.cspace_positions())

    # DEPRECATED section. --------------------------------------------------------------------------
    # Test goalset with custom seeds using `solve_goalset()`.
    translation_goalset = cumotion.CollisionFreeIkSolver.TranslationConstraintGoalset.target(
        translation_targets)
    orientation_goalset = cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.target(
        rotation_targets)
    task_space_target_goalset = cumotion.CollisionFreeIkSolver.TaskSpaceTargetGoalset(
        translation_goalset, orientation_goalset)

    # Solve without custom seeds
    goalset_results_no_seeds = ik_solver.solve_goalset(task_space_target_goalset)
    assert goalset_results_no_seeds.status() == \
        cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS

    # Find the easiest target and create custom seeds from it
    easiest_target_idx = compute_mode(goalset_results_no_seeds.target_indices())
    custom_seeds = [cspace_samples[easiest_target_idx] for _ in range(num_custom_seeds)]

    # Solve with custom seeds
    goalset_results_with_seeds = ik_solver.solve_goalset(task_space_target_goalset, custom_seeds)
    assert goalset_results_with_seeds.status() == \
        cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS

    # Custom seeds affect the optimization, so we expect different results
    # (The number of solutions can be higher or lower depending on the seed quality)
    assert len(goalset_results_with_seeds.cspace_positions()) != \
        len(goalset_results_no_seeds.cspace_positions())

    # Validate that the goalset results match the goalset array results.
    assert goalset_results_with_seeds.status() == \
        first_result_with_seeds_array.status()
    assert len(goalset_results_with_seeds.cspace_positions()) == \
        len(first_result_with_seeds_array.cspace_positions())
    assert list(goalset_results_with_seeds.target_indices()) == \
        list(first_result_with_seeds_array.target_indices())
    # End of DEPRECATED section.  ------------------------------------------------------------------


def test_franka_with_deviation_limits(configure_franka_with_obstacles):
    """Test IK solver with deviation limits."""
    robot_description, world, world_view = configure_franka_with_obstacles()

    tool_frame_name = "panda_rightfinger"
    kinematics = robot_description.kinematics()

    # Sample reachable poses
    num_targets = 5
    cspace_samples, poses = sample_reachable_poses(
        kinematics, tool_frame_name, num_targets, seed=41252)

    # Extract translation and rotation targets
    translation_targets = [pose.translation for pose in poses]
    rotation_targets = [pose.rotation for pose in poses]

    # Define deviation limits
    position_deviation_limit = 0.10  # 10cm
    orientation_deviation_limit = 0.2  # ~11.5 degrees

    # Create config and IK solver
    config = cumotion.create_default_collision_free_ik_solver_config(
        robot_description, tool_frame_name, world_view)
    config.set_param("num_seeds", NUM_SEEDS)
    ik_solver = cumotion.create_collision_free_ik_solver(config)

    # Test goalset with position-only constraints using `solve_array()`.
    # 1a. No deviation limit
    translation_goalset_no_dev_array = \
        cumotion.CollisionFreeIkSolver.TranslationConstraintArray.target([translation_targets])
    orientation_goalset_none_array = \
        cumotion.CollisionFreeIkSolver.OrientationConstraintArray.none()
    task_space_goalset_no_dev_array = cumotion.CollisionFreeIkSolver.TaskSpaceTargetArray(
        translation_goalset_no_dev_array, orientation_goalset_none_array)

    results_goalset_no_dev_array = ik_solver.solve_array(task_space_goalset_no_dev_array)
    first_result_goalset_no_dev_array = results_goalset_no_dev_array.problem(0)
    assert first_result_goalset_no_dev_array.status() == \
        cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    num_solutions_goalset_no_dev_array = len(first_result_goalset_no_dev_array.cspace_positions())

    # 1b. With position deviation limit
    translation_goalset_with_dev_array = \
        cumotion.CollisionFreeIkSolver.TranslationConstraintArray.target(
            [translation_targets], position_deviation_limit)
    task_space_goalset_with_dev_array = cumotion.CollisionFreeIkSolver.TaskSpaceTargetArray(
        translation_goalset_with_dev_array, orientation_goalset_none_array)

    results_goalset_with_dev_array = ik_solver.solve_array(task_space_goalset_with_dev_array)
    first_result_goalset_with_dev_array = results_goalset_with_dev_array.problem(0)
    assert first_result_goalset_with_dev_array.status() == \
        cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    num_solutions_goalset_with_dev_array = len(
        first_result_goalset_with_dev_array.cspace_positions())

    # Expect more or equal solutions with deviation limits
    assert num_solutions_goalset_with_dev_array > num_solutions_goalset_no_dev_array

    # Test goalset with full orientation constraints using `solve_array()`.
    # 2a. No deviation limit
    orientation_goalset_no_dev_array = \
        cumotion.CollisionFreeIkSolver.OrientationConstraintArray.target([rotation_targets])
    task_space_goalset_orient_no_dev_array = cumotion.CollisionFreeIkSolver.TaskSpaceTargetArray(
        translation_goalset_no_dev_array, orientation_goalset_no_dev_array)

    results_goalset_orient_no_dev_array = \
        ik_solver.solve_array(task_space_goalset_orient_no_dev_array)
    first_result_goalset_orient_no_dev_array = results_goalset_orient_no_dev_array.problem(0)
    assert first_result_goalset_orient_no_dev_array.status() == \
        cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    num_solutions_goalset_orient_no_dev_array = len(
        first_result_goalset_orient_no_dev_array.cspace_positions())

    # 2b. With orientation deviation limit only
    orientation_goalset_with_dev_array = \
        cumotion.CollisionFreeIkSolver.OrientationConstraintArray.target(
            [rotation_targets], orientation_deviation_limit)
    task_space_goalset_orient_with_dev_array = cumotion.CollisionFreeIkSolver.TaskSpaceTargetArray(
        translation_goalset_no_dev_array, orientation_goalset_with_dev_array)

    results_goalset_orient_with_dev_array = ik_solver.solve_array(
        task_space_goalset_orient_with_dev_array)
    first_result_goalset_orient_with_dev_array = results_goalset_orient_with_dev_array.problem(0)
    assert first_result_goalset_orient_with_dev_array.status() == \
        cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    num_solutions_goalset_orient_with_dev_array = len(
        first_result_goalset_orient_with_dev_array.cspace_positions())

    # Expect more or equal solutions with deviation limits
    assert num_solutions_goalset_orient_with_dev_array > num_solutions_goalset_orient_no_dev_array

    # 2c. With both position and orientation deviation limits
    task_space_goalset_both_dev_array = cumotion.CollisionFreeIkSolver.TaskSpaceTargetArray(
        translation_goalset_with_dev_array, orientation_goalset_with_dev_array)

    results_goalset_both_dev_array = ik_solver.solve_array(task_space_goalset_both_dev_array)
    first_result_goalset_both_dev_array = results_goalset_both_dev_array.problem(0)
    assert first_result_goalset_both_dev_array.status() == \
        cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
    num_solutions_goalset_both_dev_array = len(
        first_result_goalset_both_dev_array.cspace_positions())

    # Expect more or equal solutions with both deviation limits
    assert num_solutions_goalset_both_dev_array > num_solutions_goalset_orient_no_dev_array

    # DEPRECATED section. --------------------------------------------------------------------------
    # Test goalset with position-only constraints using `solve_goalset()`.
    # 1a. No deviation limit
    translation_goalset_no_dev = cumotion.CollisionFreeIkSolver.TranslationConstraintGoalset.target(
        translation_targets)
    orientation_goalset_none = cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.none()
    task_space_goalset_no_dev = cumotion.CollisionFreeIkSolver.TaskSpaceTargetGoalset(
        translation_goalset_no_dev, orientation_goalset_none)

    results_goalset_no_dev = ik_solver.solve_goalset(task_space_goalset_no_dev)
    num_solutions_goalset_no_dev = len(results_goalset_no_dev.cspace_positions())

    # 1b. With position deviation limit
    translation_goalset_with_dev = \
        cumotion.CollisionFreeIkSolver.TranslationConstraintGoalset.target(
            translation_targets, position_deviation_limit)
    task_space_goalset_with_dev = cumotion.CollisionFreeIkSolver.TaskSpaceTargetGoalset(
        translation_goalset_with_dev, orientation_goalset_none)

    results_goalset_with_dev = ik_solver.solve_goalset(task_space_goalset_with_dev)
    num_solutions_goalset_with_dev = len(results_goalset_with_dev.cspace_positions())

    # Expect more or equal solutions with deviation limits
    assert num_solutions_goalset_with_dev > num_solutions_goalset_no_dev

    # Test goalset with full orientation constraints
    # 2a. No deviation limit
    orientation_goalset_no_dev = cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.target(
        rotation_targets)
    task_space_goalset_orient_no_dev = cumotion.CollisionFreeIkSolver.TaskSpaceTargetGoalset(
        translation_goalset_no_dev, orientation_goalset_no_dev)

    results_goalset_orient_no_dev = ik_solver.solve_goalset(task_space_goalset_orient_no_dev)
    num_solutions_goalset_orient_no_dev = len(results_goalset_orient_no_dev.cspace_positions())

    # 2b. With orientation deviation limit only
    orientation_goalset_with_dev = \
        cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.target(
            rotation_targets, orientation_deviation_limit)
    task_space_goalset_orient_with_dev = cumotion.CollisionFreeIkSolver.TaskSpaceTargetGoalset(
        translation_goalset_no_dev, orientation_goalset_with_dev)

    results_goalset_orient_with_dev = ik_solver.solve_goalset(task_space_goalset_orient_with_dev)
    num_solutions_goalset_orient_with_dev = len(results_goalset_orient_with_dev.cspace_positions())

    # Expect more or equal solutions with deviation limits
    assert num_solutions_goalset_orient_with_dev > num_solutions_goalset_orient_no_dev

    # 2c. With both position and orientation deviation limits
    task_space_goalset_both_dev = cumotion.CollisionFreeIkSolver.TaskSpaceTargetGoalset(
        translation_goalset_with_dev, orientation_goalset_with_dev)

    results_goalset_both_dev = ik_solver.solve_goalset(task_space_goalset_both_dev)
    num_solutions_goalset_both_dev = len(results_goalset_both_dev.cspace_positions())

    # Expect more or equal solutions with both deviation limits
    assert num_solutions_goalset_both_dev > num_solutions_goalset_orient_no_dev
    # End of DEPRECATED section.  ------------------------------------------------------------------

    # Test single target with deviation limits
    reached_target_idx = compute_mode(first_result_goalset_orient_no_dev_array.target_indices())
    reached_pose = poses[reached_target_idx]

    # 3a. No deviation limit
    translation_target = cumotion.CollisionFreeIkSolver.TranslationConstraint.target(
        reached_pose.translation)
    orientation_target = cumotion.CollisionFreeIkSolver.OrientationConstraint.target(
        reached_pose.rotation)
    task_space_target_no_dev = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
        translation_target, orientation_target)

    results_single_no_dev = ik_solver.solve(task_space_target_no_dev)
    num_solutions_single_no_dev = len(results_single_no_dev.cspace_positions())

    # 3b. With position deviation limit only
    translation_target_with_dev = cumotion.CollisionFreeIkSolver.TranslationConstraint.target(
        reached_pose.translation, position_deviation_limit)
    task_space_target_pos_dev = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
        translation_target_with_dev, orientation_target)

    results_single_pos_dev = ik_solver.solve(task_space_target_pos_dev)
    num_solutions_single_pos_dev = len(results_single_pos_dev.cspace_positions())

    # Expect more or equal solutions with deviation limits
    assert num_solutions_single_pos_dev > num_solutions_single_no_dev

    # 3c. With orientation deviation limit only
    orientation_target_with_dev = cumotion.CollisionFreeIkSolver.OrientationConstraint.target(
        reached_pose.rotation, orientation_deviation_limit)
    task_space_target_orient_dev = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
        translation_target, orientation_target_with_dev)

    results_single_orient_dev = ik_solver.solve(task_space_target_orient_dev)
    num_solutions_single_orient_dev = len(results_single_orient_dev.cspace_positions())

    # Expect more or equal solutions with deviation limits
    assert num_solutions_single_orient_dev >= num_solutions_single_no_dev

    # 3d. With both position and orientation deviation limits
    task_space_target_both_dev = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
        translation_target_with_dev, orientation_target_with_dev)

    results_single_both_dev = ik_solver.solve(task_space_target_both_dev)
    num_solutions_single_both_dev = len(results_single_both_dev.cspace_positions())

    # Expect more or equal solutions with both deviation limits
    assert num_solutions_single_both_dev > num_solutions_single_no_dev
