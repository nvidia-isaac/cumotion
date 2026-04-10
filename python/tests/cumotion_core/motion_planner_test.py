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

"""Unit tests for motion_planner_python.h."""

# Standard Library
import os

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion

# Local Folder
from ._test_helper import CUMOTION_ROOT_DIR, errors_disabled, warnings_disabled


class ThreeLinkArmMotionPlannerFixtureData:
    """Class to store data from pytest fixture for testing motion planner with three-link arm."""

    pass


@pytest.fixture
def configure_three_link_arm_motion_planner():
    """Test fixture to configure robot description and Motion Planner objects."""
    def _configure_three_link_arm_motion_planner(enable_self_collision_checking=False):
        # `data` will be returned at end of function and used for writing tests.
        data = ThreeLinkArmMotionPlannerFixtureData

        # Set directory for RMPflow configuration and robot description YAML files.
        config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')

        # Set absolute path to MotionPlanner configuration for three-link arm.
        motion_planner_config_path = os.path.join(config_path,
                                                  'three_link_arm_planner_config.yaml')

        # Set absolute path to the XRDF for the three-link arm.
        xrdf_path = os.path.join(config_path, 'three_link_arm.xrdf')

        # Set absolute path to URDF file for three-link arm.
        urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'robots',
                                 'three_link_arm.urdf')

        # Load robot description.
        robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

        # Create world for obstacles.
        data.world = cumotion.create_world()

        data.tool_frame_name = "end_effector"

        # Add spherical obstacles to world.
        obstacle_radius = 0.3
        obstacle_centers = [np.array([0.0, 0.8, 0.0]),
                            np.array([0.0, 1.2, 0.0]),
                            np.array([0.0, 1.6, 0.0]),
                            np.array([0.0, -1.4, 0.0]),
                            np.array([-1.6, 0.0, 0.0]),
                            np.array([-1.6, 0.4, 0.0]),
                            np.array([-1.6, 0.8, 0.0])]
        sphere = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
        sphere.set_attribute(cumotion.Obstacle.Attribute.RADIUS, obstacle_radius)
        for center in obstacle_centers:
            pose = cumotion.Pose3.from_translation(center)
            data.world.add_obstacle(sphere, pose)

        # Create motion planner.
        data.motion_planner_config = \
            cumotion.create_motion_planner_config_from_file(motion_planner_config_path,
                                                            robot_description,
                                                            data.tool_frame_name,
                                                            data.world.add_world_view())
        # Set final tolerance for task space search.
        data.translation_target_final_tolerance = 1e-4
        data.motion_planner_config.set_param(
            "task_space_planning_params/translation_target_final_tolerance",
            data.translation_target_final_tolerance)

        data.orientation_target_final_tolerance = 0.005
        data.motion_planner_config.set_param(
            "task_space_planning_params/orientation_target_final_tolerance",
            data.orientation_target_final_tolerance)

        # Default to disabled for the legacy fixture.  Individual tests can override this behavior.
        data.motion_planner_config.set_param("enable_self_collision_checking",
                                             enable_self_collision_checking)

        # RRT returning a valid path is dependent on the random seed used.
        data.motion_planner_config.set_param("seed", 1234)

        data.motion_planner = cumotion.create_motion_planner(data.motion_planner_config)

        # Define initial and target c-space configurations
        data.q0 = np.array([0.0, 1.55, -1.55])
        data.q_target = np.array([-3.0, -1.5, -0.5])

        # Define task space target configuration.
        data.kinematics = robot_description.kinematics()
        data.pose_target = data.kinematics.pose(data.q_target, data.tool_frame_name)

        # Return data needed for testing.
        return data

    return _configure_three_link_arm_motion_planner


def test_default_motion_planner_config():
    """Test that default motion planner config is created correctly."""
    # Set directory for RMPflow configuration and robot description YAML files.
    config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')

    # Set absolute path to the XRDF for the three-link arm.
    xrdf_path = os.path.join(config_path, 'three_link_arm.xrdf')

    # Set absolute path to URDF file for three-link arm.
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'robots',
                             'three_link_arm.urdf')

    # Create empty world.
    world = cumotion.create_world()

    tool_frame_name = "end_effector"

    # Load robot description.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

    # Create default motion planner config.
    config = cumotion.create_default_motion_planner_config(robot_description,
                                                           tool_frame_name,
                                                           world.add_world_view())

    # Check that default motion planner config is not None.
    assert config is not None

    # Create motion planner.
    motion_planner = cumotion.create_motion_planner(config)

    # Plan path to translation target.
    results = motion_planner.plan_to_translation_target(np.array([0.0, 0.0, 0.0]),
                                                        np.array([2.0, 0.0, 0.0]))

    # Check that planner found a path.
    assert results.path_found


def test_cspace_target(configure_three_link_arm_motion_planner):
    """Test `MotionPlanner` interface with three-link arm and c-space target."""
    data = configure_three_link_arm_motion_planner()

    # Plan path to c-space target.
    results = data.motion_planner.plan_to_cspace_target(data.q0, data.q_target, True)

    # Check that planner found a path.
    assert results.path_found

    # Check that planned path has the desired initial and final configurations, and has the expected
    # length.
    assert np.allclose(data.q0, results.path[0])
    assert np.allclose(data.q_target, results.path[-1])
    assert len(results.path) == 5

    # Check that interpolation of the planned path has the desired initial and final configurations,
    # and has the expected length.
    assert np.allclose(data.q0, results.interpolated_path[0])
    assert np.allclose(data.q_target, results.interpolated_path[-1])
    assert len(results.interpolated_path) == 149


def test_translation_target(configure_three_link_arm_motion_planner):
    """Test `MotionPlanner` interface with three-link arm and task space translation target."""
    data = configure_three_link_arm_motion_planner()

    # Plan path to task space target.
    results = data.motion_planner.plan_to_translation_target(data.q0,
                                                             data.pose_target.translation,
                                                             True)

    # Check that planner found a path.
    assert results.path_found

    # Check that planned path has the desired initial and final configurations, and has the expected
    # length.
    assert np.allclose(data.q0, results.path[0])
    path_x_final_pose = data.kinematics.pose(results.path[-1], data.tool_frame_name)
    assert np.allclose(data.pose_target.translation,
                       path_x_final_pose.translation,
                       data.translation_target_final_tolerance)
    assert len(results.path) == 5

    # Check that planned path has the desired initial and final configurations, and has the expected
    # length.
    assert np.allclose(data.q0, results.interpolated_path[0])
    interpolated_path_x_final_pose = data.kinematics.pose(results.interpolated_path[-1],
                                                          data.tool_frame_name)
    assert np.allclose(data.pose_target.translation, interpolated_path_x_final_pose.translation,
                       data.translation_target_final_tolerance)
    # A range of golden values is accepted due to numerical variation between platforms
    # (e.g., x86_64 vs. aarch64 or GCC vs. MSVC).
    assert len(results.interpolated_path) in range(150, 154)


def test_pose_target(configure_three_link_arm_motion_planner):
    """Test `MotionPlanner` interface with three-link arm and task space pose target."""
    data = configure_three_link_arm_motion_planner()

    # Plan path to task space target.
    results = data.motion_planner.plan_to_pose_target(data.q0,
                                                      data.pose_target,
                                                      True)

    # Check that planner found a path.
    assert results.path_found

    # Check that planned path has the desired initial and final configurations, and has the expected
    # length.
    assert np.allclose(data.q0, results.path[0])
    path_final_pose = data.kinematics.pose(results.path[-1], data.tool_frame_name)
    assert np.allclose(data.pose_target.translation,
                       path_final_pose.translation,
                       data.translation_target_final_tolerance)
    rotation_distance = cumotion.Rotation3.distance(data.pose_target.rotation,
                                                    path_final_pose.rotation)
    assert (rotation_distance < data.orientation_target_final_tolerance)
    assert len(results.path) == 4

    # Check that planned path has the desired initial and final configurations, and has the expected
    # length.
    assert np.allclose(data.q0, results.interpolated_path[0])
    interpolated_path_final_pose = data.kinematics.pose(results.interpolated_path[-1],
                                                        data.tool_frame_name)
    assert np.allclose(data.pose_target.translation,
                       interpolated_path_final_pose.translation,
                       data.translation_target_final_tolerance)
    rotation_distance = cumotion.Rotation3.distance(data.pose_target.rotation,
                                                    interpolated_path_final_pose.rotation)
    assert (rotation_distance < data.orientation_target_final_tolerance)
    assert len(results.interpolated_path) == 158


def test_cspace_target_self_collision_enabled(configure_three_link_arm_motion_planner):
    """Test c-space planning with self-collision checking enabled."""
    data = configure_three_link_arm_motion_planner(enable_self_collision_checking=True)

    results = data.motion_planner.plan_to_cspace_target(data.q0, data.q_target, True)

    assert results.path_found
    assert len(results.path) > 0
    assert np.allclose(data.q0, results.path[0])
    assert np.allclose(data.q_target, results.path[-1])
    assert len(results.path) == 5
    assert len(results.interpolated_path) > 0
    assert np.allclose(data.q0, results.interpolated_path[0])
    assert np.allclose(data.q_target, results.interpolated_path[-1])
    assert len(results.interpolated_path) == 160


def test_translation_target_self_collision_enabled(configure_three_link_arm_motion_planner):
    """Test translation planning with self-collision checking enabled."""
    data = configure_three_link_arm_motion_planner(enable_self_collision_checking=True)

    results = data.motion_planner.plan_to_translation_target(data.q0,
                                                             data.pose_target.translation,
                                                             True)

    assert results.path_found
    assert len(results.path) > 0
    assert np.allclose(data.q0, results.path[0])
    path_x_final_pose = data.kinematics.pose(results.path[-1], data.tool_frame_name)
    assert np.allclose(data.pose_target.translation,
                       path_x_final_pose.translation,
                       data.translation_target_final_tolerance)
    assert len(results.path) == 5

    assert len(results.interpolated_path) > 0
    assert np.allclose(data.q0, results.interpolated_path[0])
    interpolated_path_x_final_pose = data.kinematics.pose(results.interpolated_path[-1],
                                                          data.tool_frame_name)
    assert np.allclose(data.pose_target.translation,
                       interpolated_path_x_final_pose.translation,
                       data.translation_target_final_tolerance)
    # A range of golden values is accepted due to numerical variation between platforms
    # (e.g., x86_64 vs. aarch64 or GCC vs. MSVC).
    assert len(results.interpolated_path) in range(150, 154)


def test_pose_target_self_collision_enabled(configure_three_link_arm_motion_planner):
    """Test pose planning with self-collision checking enabled."""
    data = configure_three_link_arm_motion_planner(enable_self_collision_checking=True)

    results = data.motion_planner.plan_to_pose_target(data.q0,
                                                      data.pose_target,
                                                      True)

    assert results.path_found
    assert len(results.path) > 0
    assert np.allclose(data.q0, results.path[0])
    path_final_pose = data.kinematics.pose(results.path[-1], data.tool_frame_name)
    assert np.allclose(data.pose_target.translation,
                       path_final_pose.translation,
                       data.translation_target_final_tolerance)
    rotation_distance = cumotion.Rotation3.distance(data.pose_target.rotation,
                                                    path_final_pose.rotation)
    assert rotation_distance < data.orientation_target_final_tolerance
    assert len(results.path) == 5

    assert len(results.interpolated_path) > 0
    assert np.allclose(data.q0, results.interpolated_path[0])
    interpolated_path_final_pose = data.kinematics.pose(results.interpolated_path[-1],
                                                        data.tool_frame_name)
    assert np.allclose(data.pose_target.translation,
                       interpolated_path_final_pose.translation,
                       data.translation_target_final_tolerance)
    rotation_distance = cumotion.Rotation3.distance(data.pose_target.rotation,
                                                    interpolated_path_final_pose.rotation)
    assert rotation_distance < data.orientation_target_final_tolerance
    assert len(results.interpolated_path) == 161


def test_set_param_enable_self_collision_checking(configure_three_link_arm_motion_planner):
    """Test `MotionPlanner::setParam` interface for setting enable self-collision checking."""
    data = configure_three_link_arm_motion_planner()

    assert data.motion_planner_config.set_param("enable_self_collision_checking", True)
    assert data.motion_planner_config.set_param("enable_self_collision_checking", False)


def test_set_param_seed(configure_three_link_arm_motion_planner):
    """Test `MotionPlanner::setParam` interface for setting seed."""
    data = configure_three_link_arm_motion_planner()

    assert data.motion_planner_config.set_param("seed", 456)
    with errors_disabled:
        assert not data.motion_planner_config.set_param("seed", -456)


def test_set_param_step_size(configure_three_link_arm_motion_planner):
    """Test `MotionPlanner::setParam` interface for setting step size."""
    data = configure_three_link_arm_motion_planner()

    assert data.motion_planner_config.set_param("step_size", 0.25)
    with errors_disabled:
        assert not data.motion_planner_config.set_param("step_size", -0.25)


def test_set_param_max_iterations(configure_three_link_arm_motion_planner):
    """Test `MotionPlanner::setParam` interface for setting maximum iterations."""
    data = configure_three_link_arm_motion_planner()

    assert data.motion_planner_config.set_param("max_iterations", 23)
    with errors_disabled:
        assert not data.motion_planner_config.set_param("max_iterations", -23)


def test_set_param_max_sampling(configure_three_link_arm_motion_planner):
    """Test `MotionPlanner::setParam` interface for setting maximum sampling."""
    data = configure_three_link_arm_motion_planner()

    assert data.motion_planner_config.set_param("max_sampling", 37)
    with errors_disabled:
        assert not data.motion_planner_config.set_param("max_sampling", -37)


def test_set_param_distance_metric_weights(configure_three_link_arm_motion_planner):
    """Test `MotionPlanner::setParam` interface for setting distance metric weights."""
    data = configure_three_link_arm_motion_planner()

    assert data.motion_planner_config.set_param("distance_metric_weights", [2.3, 4.6, 7.2])
    with errors_disabled:
        assert not data.motion_planner_config.set_param("distance_metric_weights", [2.3, -4.6, 7.2])
        assert not data.motion_planner_config.set_param("distance_metric_weights",
                                                        [2.3, 4.6, 7.2, 1.2])


def test_set_param_task_space_limits(configure_three_link_arm_motion_planner):
    """Test `MotionPlanner::setParam` interface for setting task space limits."""
    data = configure_three_link_arm_motion_planner()

    limits = [cumotion.MotionPlannerConfig.Limit(-3.7, 3.4),
              cumotion.MotionPlannerConfig.Limit(-2.9, 1.8),
              cumotion.MotionPlannerConfig.Limit(-6.7, 5.4)]
    assert data.motion_planner_config.set_param("task_space_limits", limits)

    invalid_limits = [cumotion.MotionPlannerConfig.Limit(-3.7, 3.4),
                      cumotion.MotionPlannerConfig.Limit(2.9, -1.8),
                      cumotion.MotionPlannerConfig.Limit(-6.7, 5.4)]
    with errors_disabled:
        assert not data.motion_planner_config.set_param("task_space_limits", invalid_limits)

    too_many_limits = [cumotion.MotionPlannerConfig.Limit(-3.7, 3.4),
                       cumotion.MotionPlannerConfig.Limit(-2.9, 1.8),
                       cumotion.MotionPlannerConfig.Limit(-6.7, 5.4),
                       cumotion.MotionPlannerConfig.Limit(-9.8, 2.3)]
    with errors_disabled:
        assert not data.motion_planner_config.set_param("task_space_limits", too_many_limits)


def test_set_param_cuda_tree_params(configure_three_link_arm_motion_planner):
    """Test `MotionPlanner::setParam` interface for CUDA tree params."""
    data = configure_three_link_arm_motion_planner()

    # Deprecated `max_num_nodes` is accepted but ignored (logs warning).
    with warnings_disabled:
        assert data.motion_planner_config.set_param("cuda_tree_params/max_num_nodes", 5000)
    # Test with valid, positive values.
    assert data.motion_planner_config.set_param("cuda_tree_params/max_buffer_size", 12)
    assert data.motion_planner_config.set_param("cuda_tree_params/num_nodes_cpu_gpu_crossover", 12)

    # Deprecated `max_num_nodes` is accepted and ignored regardless of value.
    with warnings_disabled:
        assert data.motion_planner_config.set_param("cuda_tree_params/max_num_nodes", 3)
    # `num_nodes_cpu_gpu_crossover` can be any positive value.
    assert data.motion_planner_config.set_param(
        "cuda_tree_params/num_nodes_cpu_gpu_crossover", 150000)

    # Deprecated `max_num_nodes` is accepted and ignored; other params still validate
    # that negative values are not accepted.
    assert data.motion_planner_config.set_param("cuda_tree_params/max_num_nodes", -12)
    with errors_disabled:
        assert not data.motion_planner_config.set_param("cuda_tree_params/max_buffer_size", -12)
        assert not data.motion_planner_config.set_param(
            "cuda_tree_params/num_nodes_cpu_gpu_crossover", -12)

    # CUDA tree can be disabled.
    assert data.motion_planner_config.set_param("enable_cuda_tree", False)

    # CUDA tree params can't be updated while CUDA tree is disabled.
    with errors_disabled:
        assert not data.motion_planner_config.set_param("cuda_tree_params/max_buffer_size", 12)
        assert not data.motion_planner_config.set_param(
            "cuda_tree_params/num_nodes_cpu_gpu_crossover", 12)
    # Deprecated `max_num_nodes` is accepted and ignored; other params still validate
    # that CUDA is enabled.
    with warnings_disabled:
        assert data.motion_planner_config.set_param("cuda_tree_params/max_num_nodes", 12)


def test_set_param_cspace(configure_three_link_arm_motion_planner):
    """Test `MotionPlanner::setParam` interface for setting c-space params."""
    data = configure_three_link_arm_motion_planner()

    assert data.motion_planner_config.set_param("cspace_planning_params/exploration_fraction", 0.7)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(
            "cspace_planning_params/exploration_fraction", 1.2)
        assert not data.motion_planner_config.set_param(
            "cspace_planning_params/exploration_fraction", -0.3)


def test_set_param_task_space(configure_three_link_arm_motion_planner):
    """Test `MotionPlanner::setParam` interface for setting task space params."""
    data = configure_three_link_arm_motion_planner()

    translation_zone_tol = "task_space_planning_params/translation_target_zone_tolerance"
    assert data.motion_planner_config.set_param(translation_zone_tol, 0.05)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(translation_zone_tol, -0.05)

    orientation_zone_tol = "task_space_planning_params/orientation_target_zone_tolerance"
    assert data.motion_planner_config.set_param(orientation_zone_tol, 0.05)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(orientation_zone_tol, -0.05)

    translation_final_tol = "task_space_planning_params/translation_target_final_tolerance"
    assert data.motion_planner_config.set_param(translation_final_tol, 0.3)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(translation_final_tol, -0.3)

    orientation_final_tol = "task_space_planning_params/orientation_target_final_tolerance"
    assert data.motion_planner_config.set_param(orientation_final_tol, 0.3)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(orientation_final_tol, -0.3)

    translation_gradient_weight = "task_space_planning_params/translation_gradient_weight"
    assert data.motion_planner_config.set_param(translation_gradient_weight, 0.3)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(translation_gradient_weight, -0.3)

    orientation_gradient_weight = "task_space_planning_params/orientation_gradient_weight"
    assert data.motion_planner_config.set_param(orientation_gradient_weight, 0.3)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(orientation_gradient_weight, -0.3)

    nn_translation_distance_weight = "task_space_planning_params/nn_translation_distance_weight"
    assert data.motion_planner_config.set_param(nn_translation_distance_weight, 0.3)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(nn_translation_distance_weight, -0.3)

    nn_orientation_distance_weight = "task_space_planning_params/nn_orientation_distance_weight"
    assert data.motion_planner_config.set_param(nn_orientation_distance_weight, 0.3)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(nn_orientation_distance_weight, -0.3)

    exploitation = "task_space_planning_params/task_space_exploitation_fraction"
    assert data.motion_planner_config.set_param(exploitation, 0.3)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(exploitation, -0.3)
        assert not data.motion_planner_config.set_param(exploitation, 1.3)
        assert not data.motion_planner_config.set_param(exploitation, 0.95)

    exploration = "task_space_planning_params/task_space_exploration_fraction"
    assert data.motion_planner_config.set_param(exploration, 0.3)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(exploration, -0.3)
        assert not data.motion_planner_config.set_param(exploration, 1.3)
        assert not data.motion_planner_config.set_param(exploration, 0.95)

    extension_substeps_away = "task_space_planning_params/max_extension_substeps_away_from_target"
    assert data.motion_planner_config.set_param(extension_substeps_away, 30)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(extension_substeps_away, -1)

    extension_substeps_near = "task_space_planning_params/max_extension_substeps_near_target"
    assert data.motion_planner_config.set_param(extension_substeps_near, 6)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(extension_substeps_near, -1)

    scale_factor = "task_space_planning_params/extension_substep_target_region_scale_factor"
    assert data.motion_planner_config.set_param(scale_factor, 3.0)
    assert data.motion_planner_config.set_param(scale_factor, 1.0)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(scale_factor, 0.0)
        assert not data.motion_planner_config.set_param(scale_factor, -1.0)

    culling_scalar = "task_space_planning_params/unexploited_nodes_culling_scalar"
    assert data.motion_planner_config.set_param(culling_scalar, 1.0)
    assert data.motion_planner_config.set_param(culling_scalar, 0.0)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(culling_scalar, -1.0)

    gradient_substep_size = "task_space_planning_params/gradient_substep_size"
    assert data.motion_planner_config.set_param(gradient_substep_size, 0.1)
    with errors_disabled:
        assert not data.motion_planner_config.set_param(gradient_substep_size, -1.0)
        assert not data.motion_planner_config.set_param(gradient_substep_size, 0.0)


def test_reset_cspace_target(configure_three_link_arm_motion_planner):
    """Test that `reset()` produces identical c-space plans across multiple resets."""
    data = configure_three_link_arm_motion_planner()

    data.motion_planner.reset()
    reference = data.motion_planner.plan_to_cspace_target(data.q0, data.q_target, True)
    assert reference.path_found

    for _ in range(5):
        data.motion_planner.reset()
        results = data.motion_planner.plan_to_cspace_target(data.q0, data.q_target, True)
        assert results.path_found
        assert len(results.path) == len(reference.path)
        for ref_q, res_q in zip(reference.path, results.path):
            np.testing.assert_array_equal(ref_q, res_q)
        assert len(results.interpolated_path) == len(reference.interpolated_path)
        for ref_q, res_q in zip(reference.interpolated_path, results.interpolated_path):
            np.testing.assert_array_equal(ref_q, res_q)


def test_reset_translation_target(configure_three_link_arm_motion_planner):
    """Test that `reset()` produces identical translation plans across multiple resets."""
    data = configure_three_link_arm_motion_planner()

    data.motion_planner.reset()
    reference = data.motion_planner.plan_to_translation_target(
        data.q0, data.pose_target.translation, True)
    assert reference.path_found

    for _ in range(5):
        data.motion_planner.reset()
        results = data.motion_planner.plan_to_translation_target(
            data.q0, data.pose_target.translation, True)
        assert results.path_found
        assert len(results.path) == len(reference.path)
        for ref_q, res_q in zip(reference.path, results.path):
            np.testing.assert_array_equal(ref_q, res_q)
        assert len(results.interpolated_path) == len(reference.interpolated_path)
        for ref_q, res_q in zip(reference.interpolated_path, results.interpolated_path):
            np.testing.assert_array_equal(ref_q, res_q)


def test_reset_pose_target(configure_three_link_arm_motion_planner):
    """Test that `reset()` produces identical pose plans across multiple resets."""
    data = configure_three_link_arm_motion_planner()

    data.motion_planner.reset()
    reference = data.motion_planner.plan_to_pose_target(data.q0, data.pose_target, True)
    assert reference.path_found

    for _ in range(5):
        data.motion_planner.reset()
        results = data.motion_planner.plan_to_pose_target(data.q0, data.pose_target, True)
        assert results.path_found
        assert len(results.path) == len(reference.path)
        for ref_q, res_q in zip(reference.path, results.path):
            np.testing.assert_array_equal(ref_q, res_q)
        assert len(results.interpolated_path) == len(reference.interpolated_path)
        for ref_q, res_q in zip(reference.interpolated_path, results.interpolated_path):
            np.testing.assert_array_equal(ref_q, res_q)
