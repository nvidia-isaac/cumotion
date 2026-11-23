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

"""Unit tests for rmpflow_python.h."""

# Standard Library
import os

# Third Party
import numpy as np
import pytest
import yaml

# cuMotion
import cumotion

# Local directory
from ._test_helper import CUMOTION_ROOT_DIR, warnings_disabled


@pytest.fixture
def configure_franka_rmpflow_test():
    """Test fixture to configure robot description and RMPFlow objects."""
    def _configure_franka_rmpflow_test():
        # Set directory for RMPflow configuration and robot description YAML files.
        config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')

        # Set absolute path to RMPflow configuration for Franka.
        rmpflow_config_path = os.path.join(
            config_path, 'franka_rmpflow_config_without_point_cloud.yaml')

        # Set absolute path to the XRDF for Franka.
        xrdf_path = os.path.join(config_path, "franka.xrdf")

        # Set absolute path to URDF file for Franka.
        urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka',
                                 'franka.urdf')

        # Load robot description.
        robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

        # Create world for obstacles.
        world = cumotion.create_world()
        world_view = world.add_world_view()

        # Create RMPflow configuration.
        rmpflow_config = cumotion.create_rmpflow_config_from_file(rmpflow_config_path,
                                                                  robot_description,
                                                                  world_view)

        # Create RMPflow policy.
        rmpflow = cumotion.create_rmpflow(rmpflow_config)

        # Set end effector frame for Franka and add a task-space target to `rmpflow`.
        end_effector_frame_name = 'right_gripper'
        rmpflow.add_target_frame(end_effector_frame_name)

        return (robot_description, end_effector_frame_name, world, world_view, rmpflow)

    return _configure_franka_rmpflow_test


def test_franka_rmpflow(configure_franka_rmpflow_test):
    """Test the Franka RMPFlow interface."""
    robot_description, end_effector_frame_name, world, world_view, rmpflow = (
        configure_franka_rmpflow_test()
    )

    # Expect that c-space values of all zeros exceed c-space limits (due to "panda_joint4").
    zero = np.zeros(7)
    with warnings_disabled:
        assert robot_description.kinematics().within_cspace_limits(zero, True) is False

    # Test that valid c-space positions are within bounds.
    valid_position = np.zeros(7)
    valid_position[3] = -1.5
    assert robot_description.kinematics().within_cspace_limits(valid_position, True)

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

    # Update world view after adding obstacles.
    world_view.update()

    # Set default configuration.
    default_cspace_config = np.array([0.00, -1.3, 0.00, -2.87, 0.00, 2.00, 0.75])

    # Set end effector position target.
    end_effector_position_target = np.array([0.8, 0.0, 0.35])
    rmpflow.set_position_target(end_effector_frame_name, end_effector_position_target)
    rmpflow.set_cspace_attractor(default_cspace_config)

    # Set c-space positions and c-space velocities.
    cspace_position = default_cspace_config
    cspace_velocity = np.zeros(robot_description.num_cspace_coords())

    # Initialize c-space acceleration.
    cspace_accel = np.zeros(robot_description.num_cspace_coords())

    # Evaluate acceleration from c-space state.
    rmpflow.eval_accel(cspace_position, cspace_velocity, cspace_accel)

    # Test acceleration values.
    cspace_accel_expected = np.array([-0.149459,
                                      -24.22777,
                                      -0.167684,
                                      -40.3788,
                                      0.0096364,
                                      -16.8079,
                                      0.0494])

    for idx, (expected, actual) in enumerate(zip(cspace_accel_expected, cspace_accel)):
        assert expected == pytest.approx(actual, abs=1e-3), 'cspace_accel[{}]'.format(idx)

    # Evaluate force and metric from c-space state.
    cspace_force, cspace_metric = rmpflow.eval_force_and_metric(cspace_position, cspace_velocity)

    # Test force values.
    cspace_force_expected = np.array([-61992.3972,
                                      -7534638.4889,
                                      -69745.5143,
                                      -13006502.376,
                                      4135.511,
                                      -5397947.9038,
                                      20835.2446])

    for expected, actual in zip(cspace_force_expected, cspace_force):
        assert expected == pytest.approx(actual, abs=1e-3)

    # Test metric values.
    cspace_metric_expected = np.array([[792.72954074,
                                        -15.74551121,
                                        704.7645975,
                                        339.52047539,
                                        105.18664579,
                                        126.15884042,
                                        -161.38733742],
                                       [-15.74551121,
                                        3237.08883909,
                                        -20.87428034,
                                        -1261.31699487,
                                        3.8195338,
                                        -44.59886505,
                                        -10.645875],
                                       [704.7645975,
                                        -20.87428034,
                                        776.30003264,
                                        387.57909383,
                                        39.61201339,
                                        143.76427375,
                                        -181.56109254],
                                       [339.52047539,
                                        -1261.31699487,
                                        387.57909383,
                                        11304.60179998,
                                        -27.35418726,
                                        4019.64865898,
                                        -111.43922314],
                                       [105.18664579,
                                        3.8195338,
                                        39.61201339,
                                        -27.35418726,
                                        138.07045058,
                                        -9.82603732,
                                        11.82765213],
                                       [126.15884042,
                                        -44.59886505,
                                        143.76427375,
                                        4019.64865898,
                                        -9.82603732,
                                        1673.22680204,
                                        -42.63080524],
                                       [-161.38733742,
                                        -10.645875,
                                        -181.56109254,
                                        -111.43922314,
                                        11.82765213,
                                        -42.63080524,
                                        107.10692904]])

    for expected_vector, actual_vector in zip(cspace_metric_expected, cspace_metric):
        for expected, actual in zip(expected_vector, actual_vector):
            assert expected == pytest.approx(actual, abs=1e-3)


@pytest.fixture
def franka_rmpflow_setup():
    """Fixture providing Franka setup for RMPflow tests."""
    config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')
    rmpflow_config_path = os.path.join(
        config_path, 'franka_rmpflow_config_without_point_cloud.yaml')
    xrdf_path = os.path.join(config_path, "franka.xrdf")
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka',
                             'franka.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()
    rmpflow_config = cumotion.create_rmpflow_config_from_file(
        rmpflow_config_path, robot_description, world_view)
    rmpflow = cumotion.create_rmpflow(rmpflow_config)
    return robot_description, world, world_view, rmpflow


def test_invalid_add_target_frame(franka_rmpflow_setup):
    """Test error handling for `add_target_frame()`."""
    robot_description, _, _, rmpflow = franka_rmpflow_setup

    # Expect there to be no task-space targets.
    assert rmpflow.num_target_frames() == 0
    assert len(rmpflow.target_frame_names()) == 0

    # Expect a fatal error when adding a target for a frame that does not exist in the
    # `RobotDescription`.
    with pytest.raises(Exception):
        rmpflow.add_target_frame("not_a_frame")

    # Expect a fatal error when adding a target with an invalid target config.
    target_config = cumotion.RmpFlow.TargetRmpConfig()
    target_config.position_config.accel_p_gain = -0.3
    with pytest.raises(Exception):
        rmpflow.add_target_frame(robot_description.tool_frame_names()[0], target_config)

    # Expect success when adding a target with a valid target config.
    target_config.position_config.accel_p_gain = 0.3
    rmpflow.add_target_frame(robot_description.tool_frame_names()[0], target_config)
    assert rmpflow.num_target_frames() == 1
    assert rmpflow.target_frame_names() == robot_description.tool_frame_names()

    # Expect a fatal error when adding the same target frame twice.
    with pytest.raises(Exception):
        rmpflow.add_target_frame(robot_description.tool_frame_names()[0])


def test_invalid_remove_target_frame(franka_rmpflow_setup):
    """Test error handling for `remove_target_frame()`."""
    robot_description, _, _, rmpflow = franka_rmpflow_setup

    # Expect there to be no task-space targets.
    assert rmpflow.num_target_frames() == 0
    assert len(rmpflow.target_frame_names()) == 0

    # Expect a fatal error when removing a target for a frame that has not been added.
    with pytest.raises(Exception):
        rmpflow.remove_target_frame(robot_description.tool_frame_names()[0])

    # Expect success when adding a target with a valid target config.
    rmpflow.add_target_frame(robot_description.tool_frame_names()[0])
    assert rmpflow.num_target_frames() == 1
    assert rmpflow.target_frame_names() == robot_description.tool_frame_names()

    # Expect success when removing a valid target frame.
    rmpflow.remove_target_frame(robot_description.tool_frame_names()[0])
    assert rmpflow.num_target_frames() == 0
    assert len(rmpflow.target_frame_names()) == 0

    # Expect a fatal error when removing the same target frame twice.
    with pytest.raises(Exception):
        rmpflow.remove_target_frame(robot_description.tool_frame_names()[0])


def test_invalid_target_setting_and_clearing(franka_rmpflow_setup):
    """Test error handling for target setting and clearing functions."""
    robot_description, _, _, rmpflow = franka_rmpflow_setup
    tool_frame = robot_description.tool_frame_names()[0]

    # Expect there to be no task-space targets.
    assert rmpflow.num_target_frames() == 0
    assert len(rmpflow.target_frame_names()) == 0

    # Expect a fatal error when setting or clearing a target for a frame that has not been added.
    with pytest.raises(Exception):
        rmpflow.set_pose_target(tool_frame, cumotion.Pose3.identity())
    with pytest.raises(Exception):
        rmpflow.clear_pose_target(tool_frame)
    with pytest.raises(Exception):
        rmpflow.set_position_target(tool_frame, np.zeros(3))
    with pytest.raises(Exception):
        rmpflow.clear_position_target(tool_frame)
    with pytest.raises(Exception):
        rmpflow.set_orientation_target(tool_frame, cumotion.Rotation3.identity())
    with pytest.raises(Exception):
        rmpflow.clear_orientation_target(tool_frame)

    # Expect success when adding a target with a valid target config.
    rmpflow.add_target_frame(tool_frame)
    assert rmpflow.num_target_frames() == 1
    assert rmpflow.target_frame_names() == robot_description.tool_frame_names()

    # Expect success when setting or clearing a target for a frame that has been added.
    rmpflow.set_pose_target(tool_frame, cumotion.Pose3.identity())
    rmpflow.clear_pose_target(tool_frame)
    rmpflow.set_position_target(tool_frame, np.zeros(3))
    rmpflow.clear_position_target(tool_frame)
    rmpflow.set_orientation_target(tool_frame, cumotion.Rotation3.identity())
    rmpflow.clear_orientation_target(tool_frame)

    # Expect success when removing a valid target frame.
    rmpflow.remove_target_frame(tool_frame)
    assert rmpflow.num_target_frames() == 0
    assert len(rmpflow.target_frame_names()) == 0

    # Expect a fatal error when setting or clearing a target for a frame that has been removed.
    with pytest.raises(Exception):
        rmpflow.set_pose_target(tool_frame, cumotion.Pose3.identity())
    with pytest.raises(Exception):
        rmpflow.clear_pose_target(tool_frame)
    with pytest.raises(Exception):
        rmpflow.set_position_target(tool_frame, np.zeros(3))
    with pytest.raises(Exception):
        rmpflow.clear_position_target(tool_frame)
    with pytest.raises(Exception):
        rmpflow.set_orientation_target(tool_frame, cumotion.Rotation3.identity())
    with pytest.raises(Exception):
        rmpflow.clear_orientation_target(tool_frame)


def load_franka_rmpflow_config_yaml():
    """Load the RMPflow configuration for Franka as a dictionary."""
    rmpflow_config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared',
                                       'franka_rmpflow_config_without_point_cloud.yaml')
    with open(rmpflow_config_path, 'r') as file:
        return yaml.safe_load(file)


def test_invalid_rmpflow_config(franka_rmpflow_setup):
    """Test error handling for creating an `RmpFlowConfig` from invalid input."""
    robot_description, _, world_view, _ = franka_rmpflow_setup

    # Expect a fatal error when the RMPflow configuration file does not exist.
    with pytest.raises(RuntimeError):
        cumotion.create_rmpflow_config_from_file(
            os.path.join(CUMOTION_ROOT_DIR, 'missing_rmpflow_config.yaml'),
            robot_description,
            world_view)

    # Expect a fatal error when the RMPflow configuration is not valid YAML.
    with pytest.raises(RuntimeError):
        cumotion.create_rmpflow_config_from_memory('rmp_params: [', robot_description, world_view)

    # Expect a fatal error when an RMP parameter value cannot be parsed.
    with pytest.raises(RuntimeError):
        cumotion.create_rmpflow_config_from_memory('rmp_params: {damping_rmp: {inertia: x}}',
                                                   robot_description,
                                                   world_view)


def test_invalid_rmpflow_params(franka_rmpflow_setup):
    """Test error handling for RMPflow parameters that are missing or cannot be parsed."""
    robot_description, _, world_view, _ = franka_rmpflow_setup

    # Expect fatal errors when creating `RmpFlow` or `TargetRmpConfig` without required parameters.
    empty_config = cumotion.create_rmpflow_config_from_memory('rmp_params: {}',
                                                              robot_description,
                                                              world_view)
    with pytest.raises(RuntimeError):
        cumotion.create_rmpflow(empty_config)
    with pytest.raises(RuntimeError):
        cumotion.RmpFlow.TargetRmpConfig(empty_config)

    # Expect a fatal error when creating `TargetRmpConfig` from target RMP parameters that cannot be
    # parsed.
    scalar_target_config = cumotion.create_rmpflow_config_from_memory(
        'rmp_params: {target_rmp: 3}', robot_description, world_view)
    with pytest.raises(RuntimeError):
        cumotion.RmpFlow.TargetRmpConfig(scalar_target_config)

    # Expect a fatal error when creating `RmpFlow` with joint limit buffers that cannot be parsed.
    config_yaml = load_franka_rmpflow_config_yaml()
    config_yaml['joint_limit_buffers'] = 'wide'
    unparsable_buffers_config = cumotion.create_rmpflow_config_from_memory(
        yaml.dump(config_yaml), robot_description, world_view)
    with pytest.raises(RuntimeError):
        cumotion.create_rmpflow(unparsable_buffers_config)


def test_invalid_default_target_rmp_params(franka_rmpflow_setup):
    """Test error handling for `add_target_frame()` without default target RMP parameters."""
    robot_description, _, world_view, _ = franka_rmpflow_setup
    tool_frame = robot_description.tool_frame_names()[0]

    # Create `RmpFlow` from a configuration without target RMP parameters.
    config_yaml = load_franka_rmpflow_config_yaml()
    del config_yaml['rmp_params']['target_rmp']
    missing_target_config = cumotion.create_rmpflow_config_from_memory(
        yaml.dump(config_yaml), robot_description, world_view)
    rmpflow = cumotion.create_rmpflow(missing_target_config)

    # Expect a fatal error when adding a target that relies on the missing default parameters.
    with pytest.raises(RuntimeError):
        rmpflow.add_target_frame(tool_frame)

    # Expect success when adding a target with an explicit `TargetRmpConfig`.
    rmpflow.add_target_frame(tool_frame, cumotion.RmpFlow.TargetRmpConfig())
    assert rmpflow.num_target_frames() == 1


def test_no_task_space_target(franka_rmpflow_setup):
    """Test RMPflow evaluation with no task-space target."""
    robot_description, _, _, rmpflow = franka_rmpflow_setup

    # Expect there to be no task-space targets.
    assert rmpflow.num_target_frames() == 0
    assert len(rmpflow.target_frame_names()) == 0

    # Expect setting or clearing task-space targets to fail.
    with pytest.raises(Exception):
        rmpflow.set_end_effector_position_attractor(np.zeros(3))
    with pytest.raises(Exception):
        rmpflow.clear_end_effector_position_attractor()
    with pytest.raises(Exception):
        rmpflow.set_end_effector_orientation_attractor(cumotion.Rotation3.identity())
    with pytest.raises(Exception):
        rmpflow.clear_end_effector_orientation_attractor()

    # Expect evaluation of RMPflow to be successful with no task-space target.
    cspace_position = robot_description.default_cspace_configuration()
    cspace_velocity = np.zeros(7)
    cspace_acceleration = np.zeros(7)
    rmpflow.eval_accel(cspace_position, cspace_velocity, cspace_acceleration)
    assert np.linalg.norm(cspace_acceleration) == pytest.approx(0.65281677, abs=1e-5)


def test_get_all_params():
    """Test that `get_all_params()` fills its lists with all parameter names and values."""
    config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')
    rmpflow_config_path = os.path.join(
        config_path, 'franka_rmpflow_config_without_point_cloud.yaml')
    xrdf_path = os.path.join(config_path, 'franka.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka',
                             'franka.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    world = cumotion.create_world()
    world_view = world.add_world_view()
    rmpflow_config = cumotion.create_rmpflow_config_from_file(
        rmpflow_config_path, robot_description, world_view)

    # Expect the lists to be overwritten with a value for each parameter name, matching
    # `get_param()`.
    names = ['not_a_param']
    values = [0.0]
    rmpflow_config.get_all_params(names, values)
    assert 'not_a_param' not in names
    assert len(names) > 0
    assert len(values) == len(names)
    for name, value in zip(names, values):
        assert rmpflow_config.get_param(name) == value

    # Expect the returned lists to be accepted by `set_all_params()`.
    values[0] += 1.0
    rmpflow_config.set_all_params(names, values)
    assert rmpflow_config.get_param(names[0]) == values[0]


def test_multiple_task_space_targets():
    """Test RMPflow with multiple task-space targets on Unitree G1."""
    config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')
    rmpflow_config_path = os.path.join(
        config_path, 'g1_29dof_with_hand_rmpflow_config.yaml')
    xrdf_path = os.path.join(config_path, 'g1_29dof_with_hand.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'unitree',
                             'g1_29dof_with_hand',
                             'g1_29dof_with_hand_rev_1_0.urdf')

    # Load robot description and extract kinematics.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    kinematics = robot_description.kinematics()

    # Create world.
    world = cumotion.create_world()
    world_view = world.add_world_view()

    # Create `RmpFlow` with no task-space target set.
    config = cumotion.create_rmpflow_config_from_file(
        rmpflow_config_path, robot_description, world_view)
    rmpflow = cumotion.create_rmpflow(config)

    # Add targets for the left and right hands.
    # NOTE: An "extra" target is first added (arbitrarily) for the head link. This is used later
    #       to ensure that internal indexing is updated correctly when a target frame is removed.
    head_frame_name = "head_link"
    left_frame_name = "left_hand_palm_link"
    right_frame_name = "right_hand_palm_link"
    rmpflow.add_target_frame(head_frame_name)
    rmpflow.add_target_frame(left_frame_name)
    rmpflow.add_target_frame(right_frame_name)
    assert rmpflow.num_target_frames() == 3
    assert len(rmpflow.target_frame_names()) == 3

    # Compute the default position for each hand.
    default_cspace_config = robot_description.default_cspace_configuration()
    left_initial_position = kinematics.position(default_cspace_config, left_frame_name)
    right_initial_position = kinematics.position(default_cspace_config, right_frame_name)

    # Set targets for each hand, offset along the x-axis.
    position_offset = 0.2
    left_target_position = left_initial_position + position_offset * np.array([1.0, 0.0, 0.0])
    right_target_position = right_initial_position + position_offset * np.array([1.0, 0.0, 0.0])
    rmpflow.set_position_target(left_frame_name, left_target_position)
    rmpflow.set_position_target(right_frame_name, right_target_position)

    # Initialize c-space state to the default configuration at rest.
    num_coords = kinematics.num_cspace_coords()
    cspace_position = default_cspace_config.copy()
    cspace_velocity = np.zeros(num_coords)
    cspace_accel = np.zeros(num_coords)

    # Evaluate RMPflow for sufficient iterations to converge at both task-space targets.
    num_iterations = 100
    dt = 0.01
    for _ in range(num_iterations):
        rmpflow.eval_accel(cspace_position, cspace_velocity, cspace_accel)
        cspace_position += dt * cspace_velocity
        cspace_velocity += dt * cspace_accel

    # Compute final position for each hand.
    left_final_position = kinematics.position(cspace_position, left_frame_name)
    right_final_position = kinematics.position(cspace_position, right_frame_name)

    # Expect each hand to have (nearly) reached the target.
    position_tolerance = 0.005
    np.testing.assert_allclose(left_final_position, left_target_position, atol=position_tolerance)
    np.testing.assert_allclose(right_final_position, right_target_position, atol=position_tolerance)

    # Record the final c-space position.
    final_cspace_position = cspace_position.copy()

    # Remove the target frame for the head link.
    rmpflow.remove_target_frame(head_frame_name)

    # Reinitialize c-space state.
    cspace_position = default_cspace_config.copy()
    cspace_velocity = np.zeros(num_coords)
    cspace_accel = np.zeros(num_coords)

    # Roll out RMPflow policy a second time, expecting the hands to still reach their targets.
    # This indicates that internal indexing into the `KinematicAggregator` has been appropriately
    # updated after removing the head link as a task-space target.
    for _ in range(num_iterations):
        rmpflow.eval_accel(cspace_position, cspace_velocity, cspace_accel)
        cspace_position += dt * cspace_velocity
        cspace_velocity += dt * cspace_accel

    # Expect each hand to have, once again, (nearly) reached the target.
    left_final_position = kinematics.position(cspace_position, left_frame_name)
    right_final_position = kinematics.position(cspace_position, right_frame_name)
    np.testing.assert_allclose(left_final_position, left_target_position, atol=position_tolerance)
    np.testing.assert_allclose(right_final_position, right_target_position, atol=position_tolerance)

    # We do *not* expect the same c-space position after removing the head link. This is because
    # task-space targets with no active position and/or orientation targets set still add damping
    # RMPs by default.
    assert np.linalg.norm(final_cspace_position - cspace_position) == pytest.approx(
        0.314, abs=5e-3)
