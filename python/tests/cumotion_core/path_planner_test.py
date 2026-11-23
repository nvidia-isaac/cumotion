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

"""Unit tests for path_planner_python.h."""

# Standard Library
import os

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion

# Local Folder
from ._test_helper import CUMOTION_ROOT_DIR, errors_disabled

# Cuboids forming the UR10 table environment, given as (center position, side lengths) pairs.
TABLE_CUBOIDS = (
    (np.array([0.75, 0.0, 0.6]), np.array([0.4, 1.5, 0.02])),
    (np.array([0.55, -0.5, 0.3]), np.array([0.02, 0.35, 0.6])),
    (np.array([0.55, 0.5, 0.3]), np.array([0.02, 0.35, 0.6])),
    (np.array([0.75, 0.0, 0.0]), np.array([0.4, 1.5, 0.02])),
    (np.array([0.9, 0.0, 0.15]), np.array([0.02, 0.35, 0.3])),
)

# Start and target c-space positions for the table environment.
START_CSPACE_POSITION = np.array([-1.8, 0.0, 0.0, 0.0, 0.0, 0.0])
TABLE_CSPACE_TARGET = np.array([0.0267, -0.372, 0.214, 3.252, 5.100, 3.994])

# C-space offset applied to the start to produce a target that the planner can reach with a direct
# connection, because the arm points away from the table throughout the connecting segment.
DIRECTLY_CONNECTED_TARGET_OFFSET = np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0])

DISTANCE_NEAR_ZERO_TOLERANCE = 1e-6


class Ur10TableScenario:
    """UR10 in the table environment from `path_planner_test.cpp`."""

    def __init__(self):
        """Load the UR10, populate the table environment, and create a robot-world inspector."""
        xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')
        urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                                 'ur10', 'ur10_robot.urdf')
        self.robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

        # The world must outlive the world view, so it is retained as a member.
        self.world = cumotion.create_world()
        for position, side_lengths in TABLE_CUBOIDS:
            obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
            obstacle.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, side_lengths)
            self.world.add_obstacle(obstacle, cumotion.Pose3.from_translation(position))
        self.world_view = self.world.add_world_view()

        self.inspector = cumotion.create_robot_world_inspector(self.robot_description,
                                                               self.world_view)

    def create_planner(self, random_seed=12345):
        """Create a `PathPlanner` from the default config, seeded with `random_seed`."""
        config = cumotion.create_default_path_planner_config(self.robot_description,
                                                             self.world_view)
        assert config is not None
        assert config.set_param("random_seed", random_seed)
        planner = cumotion.create_path_planner(config)
        assert planner is not None
        return planner

    def assert_collision_free(self, cspace_position):
        """Assert that the robot is free of self-collision and obstacle collision."""
        assert not self.inspector.in_self_collision(cspace_position)
        assert self.inspector.min_distance_to_obstacle(cspace_position) > 0.0

    def assert_interpolated_path_is_valid(self, solution, start, target):
        """Assert that `solution` interpolates from `start` to `target` without collision."""
        path = solution.path()
        assert np.allclose(path[0], start)
        assert np.allclose(path[-1], target)

        for cspace_position in solution.interpolated_path():
            self.assert_collision_free(cspace_position)


@pytest.fixture
def ur10_table():
    """Create the UR10 table scenario shared by the path planning tests."""
    return Ur10TableScenario()


def test_path_planner_config_set_param(ur10_table):
    """Test `PathPlannerConfig.set_param()`."""
    config = cumotion.create_default_path_planner_config(ur10_table.robot_description,
                                                         ur10_table.world_view)
    assert config is not None

    # Set every known parameter to a valid value, expecting success.
    # `random_seed` accepts any integer value.
    assert config.set_param("random_seed", 1234)
    assert config.set_param("random_seed", -1234)
    # Positive integer parameters.
    assert config.set_param("max_num_segments_per_forest", 10000)
    assert config.set_param("max_iterations", 250)
    assert config.set_param("num_validation_checks_per_extension", 100)
    assert config.set_param("num_spheres_per_validation_iteration", 125000)
    # Non-negative integer parameters.
    assert config.set_param("num_shortcut_iterations", 0)
    assert config.set_param("post_initial_solve_maximum_iterations", 0)
    # Positive double parameters.  `min_extension_length` must not exceed
    # `num_validation_checks_per_extension * step_size`, which is `100 * 0.02 = 2.0` here.
    assert config.set_param("step_size", 0.02)
    assert config.set_param("min_extension_length", 2.0)

    with errors_disabled:
        # Set known parameters to invalid values, expecting parameters to *not* be updated.
        assert not config.set_param("max_num_segments_per_forest", 0)
        assert not config.set_param("max_num_segments_per_forest", -1)
        assert not config.set_param("max_iterations", 0)
        assert not config.set_param("num_validation_checks_per_extension", 0)
        assert not config.set_param("num_spheres_per_validation_iteration", 0)
        assert not config.set_param("num_shortcut_iterations", -1)
        assert not config.set_param("post_initial_solve_maximum_iterations", -1)
        assert not config.set_param("step_size", 0.0)
        assert not config.set_param("min_extension_length", 0.0)
        # `min_extension_length` exceeds `num_validation_checks_per_extension * step_size`.
        assert not config.set_param("min_extension_length", 2.5)

        # Set an unknown parameter, expecting the parameter to *not* be updated.
        assert not config.set_param("unknown_parameter", 0)

        # Set known parameters with invalid types, expecting parameters to *not* be updated.
        assert not config.set_param("max_iterations", 250.5)  # int parameter with double value.
        assert not config.set_param("step_size", 1)  # double parameter with int value.


def test_create_path_planner_config_from_file(ur10_table, tmp_path):
    """Test `create_path_planner_config_from_file()`."""
    config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared',
                               'jetson_thor_path_planner_config.yaml')
    config = cumotion.create_path_planner_config_from_file(config_path,
                                                           ur10_table.robot_description,
                                                           ur10_table.world_view)
    assert config is not None

    # The loaded parameters produce a planner that solves the directly connected problem.
    planner = cumotion.create_path_planner(config)
    assert planner is not None
    target = START_CSPACE_POSITION + DIRECTLY_CONNECTED_TARGET_OFFSET
    results = planner.plan_to_cspace_targets(START_CSPACE_POSITION, [target])
    assert results.path_found()

    versioned_config_path = tmp_path / 'path_planner_config.yaml'

    def write_versioned_config(format_version, config_format='cumotion_path_planner'):
        versioned_config_path.write_text(
            f'format: {config_format}\nformat_version: {format_version}\n',
            encoding='utf-8',
        )

    def create_versioned_config():
        return cumotion.create_path_planner_config_from_file(
            str(versioned_config_path), ur10_table.robot_description, ur10_table.world_view)

    # Begin with a valid input, then change only `format` or `format_version` per invalid case.
    write_versioned_config('1.0')
    assert create_versioned_config() is not None

    with errors_disabled:
        write_versioned_config('1.0', config_format='unsupported_format')
        assert create_versioned_config() is None
        write_versioned_config('2.0')
        assert create_versioned_config() is None
        write_versioned_config('0.0.1')
        assert create_versioned_config() is None
        # An empty path or a file that cannot be loaded yields no config.
        assert cumotion.create_path_planner_config_from_file(
            '', ur10_table.robot_description, ur10_table.world_view) is None
        assert cumotion.create_path_planner_config_from_file(
            os.path.join(CUMOTION_ROOT_DIR, 'does_not_exist.yaml'),
            ur10_table.robot_description,
            ur10_table.world_view) is None


def test_ur10_table_plan_to_cspace_target(ur10_table):
    """Plan multiple collision-free paths through the UR10 table environment."""
    ur10_table.assert_collision_free(START_CSPACE_POSITION)
    ur10_table.assert_collision_free(TABLE_CSPACE_TARGET)

    planner = ur10_table.create_planner()

    compute_interpolated_path = True

    num_paths_to_return = 3
    results = planner.plan_to_cspace_targets(START_CSPACE_POSITION,
                                             [TABLE_CSPACE_TARGET],
                                             num_paths_to_return,
                                             compute_interpolated_path)
    assert results.path_found()
    assert results.num_solutions() == num_paths_to_return

    solutions = [results.solution(index) for index in range(results.num_solutions())]
    for solution in solutions:
        # Every solution reaches the only target.
        assert solution.target_index() == 0
        # Reaching the target requires navigating around the table, so no solution is a direct
        # connection between the start and the target.
        assert len(solution.path()) > 2
        assert solution.path_length() > 0.0
        ur10_table.assert_interpolated_path_is_valid(solution,
                                                     START_CSPACE_POSITION,
                                                     TABLE_CSPACE_TARGET)

    # Solutions are sorted shortest-first.
    path_lengths = [solution.path_length() for solution in solutions]
    assert path_lengths == sorted(path_lengths)

    # `interpolated_path()` is only populated when interpolation is requested.
    results_without_interpolation = planner.plan_to_cspace_targets(START_CSPACE_POSITION,
                                                                   [TABLE_CSPACE_TARGET])
    assert results_without_interpolation.path_found()
    assert results_without_interpolation.num_solutions() == 1
    assert len(results_without_interpolation.solution(0).path()) > 2
    assert len(results_without_interpolation.solution(0).interpolated_path()) == 0


def test_ur10_table_plan_to_multiple_cspace_targets(ur10_table):
    """Plan to a pair of targets where only one of them is directly connected to the start."""
    directly_connected_target = START_CSPACE_POSITION + DIRECTLY_CONNECTED_TARGET_OFFSET
    ur10_table.assert_collision_free(directly_connected_target)

    planner = ur10_table.create_planner()
    compute_interpolated_path = True
    results = planner.plan_to_cspace_targets(START_CSPACE_POSITION,
                                             [TABLE_CSPACE_TARGET, directly_connected_target],
                                             1,
                                             compute_interpolated_path)
    assert results.path_found()
    assert results.num_solutions() == 1

    # Only direct connection paths are returned when at least one target can be reached directly,
    # so the returned path is the single segment to `directly_connected_target`.
    solution = results.solution(0)
    assert solution.target_index() == 1
    assert len(solution.path()) == 2
    expected_path_length = np.linalg.norm(DIRECTLY_CONNECTED_TARGET_OFFSET)
    assert abs(solution.path_length() - expected_path_length) < DISTANCE_NEAR_ZERO_TOLERANCE
    ur10_table.assert_interpolated_path_is_valid(solution,
                                                 START_CSPACE_POSITION,
                                                 directly_connected_target)


def test_reset(ur10_table):
    """Test that `reset()` reproduces the paths found immediately after construction."""
    planner = ur10_table.create_planner()
    compute_interpolated_path = True
    reference = planner.plan_to_cspace_targets(START_CSPACE_POSITION,
                                               [TABLE_CSPACE_TARGET],
                                               1,
                                               compute_interpolated_path)
    assert reference.path_found()

    planner.reset()
    results = planner.plan_to_cspace_targets(START_CSPACE_POSITION,
                                             [TABLE_CSPACE_TARGET],
                                             1,
                                             compute_interpolated_path)
    assert results.path_found()
    assert results.num_solutions() == reference.num_solutions()

    reference_solution = reference.solution(0)
    solution = results.solution(0)
    assert solution.path_length() == reference_solution.path_length()
    assert len(solution.path()) == len(reference_solution.path())
    for reference_waypoint, waypoint in zip(reference_solution.path(), solution.path()):
        np.testing.assert_array_equal(reference_waypoint, waypoint)
    assert len(solution.interpolated_path()) == len(reference_solution.interpolated_path())
    for reference_waypoint, waypoint in zip(reference_solution.interpolated_path(),
                                            solution.interpolated_path()):
        np.testing.assert_array_equal(reference_waypoint, waypoint)


def test_plan_to_cspace_targets_input_validation(ur10_table):
    """Test input validation for `PathPlanner.plan_to_cspace_targets()`."""
    planner = ur10_table.create_planner()

    start = START_CSPACE_POSITION
    target = START_CSPACE_POSITION + DIRECTLY_CONNECTED_TARGET_OFFSET
    num_coords = ur10_table.robot_description.num_cspace_coords()

    # Start from a set of valid inputs that does not raise an exception.
    results = planner.plan_to_cspace_targets(start, [target])
    assert results.path_found()

    # `solution_index` must be in the range [0, `num_solutions()`).
    assert results.solution(0) is not None
    with pytest.raises(Exception):
        results.solution(-1)
    with pytest.raises(Exception):
        results.solution(results.num_solutions())

    # The number of targets must be in the range [1, 1024].
    with pytest.raises(Exception):
        planner.plan_to_cspace_targets(start, [])
    with pytest.raises(Exception):
        planner.plan_to_cspace_targets(start, [target] * 1025)

    # The start and every target must have as many coordinates as the robot description.
    with pytest.raises(Exception):
        planner.plan_to_cspace_targets(np.zeros(num_coords + 1), [target])
    with pytest.raises(Exception):
        planner.plan_to_cspace_targets(start, [np.zeros(num_coords + 1)])

    # The number of paths to return must be positive.
    with pytest.raises(Exception):
        planner.plan_to_cspace_targets(start, [target], 0)
    with pytest.raises(Exception):
        planner.plan_to_cspace_targets(start, [target], -1)
