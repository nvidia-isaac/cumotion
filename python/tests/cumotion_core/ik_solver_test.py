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

"""Unit tests for ik_solver_python.h."""

# Standard Library
import os

# Third Party
import numpy as np
import pytest
import yaml

# cuMotion
import cumotion

# Local Folder
from ._test_helper import CUMOTION_ROOT_DIR


class AcculumulatedCcdResults():
    """Helper class to accumulate results from IK and provide averages."""

    def __init__(self):
        """Initialize all totals to zero."""
        self._num_poses = 0
        self._total_position_error = 0.0
        self._total_x_axis_orientation_error = 0.0
        self._total_y_axis_orientation_error = 0.0
        self._total_z_axis_orientation_error = 0.0
        self._total_num_descents = 0

    def accumulate(self, results):
        """Add `results` to accumulated totals."""
        self._num_poses += 1
        self._total_position_error += results.position_error
        self._total_x_axis_orientation_error += results.x_axis_orientation_error
        self._total_y_axis_orientation_error += results.y_axis_orientation_error
        self._total_z_axis_orientation_error += results.z_axis_orientation_error
        self._total_num_descents += results.num_descents

    def num_poses(self):
        """Return number of poses accumulated."""
        return self._num_poses

    def mean_position_error(self):
        """Return mean position error."""
        return self._total_position_error / self._num_poses

    def mean_x_axis_orientation_error(self):
        """Return mean x-axis orientation error."""
        return self._total_x_axis_orientation_error / self._num_poses

    def mean_y_axis_orientation_error(self):
        """Return mean y-axis orientation error."""
        return self._total_y_axis_orientation_error / self._num_poses

    def mean_z_axis_orientation_error(self):
        """Return mean z-axis orientation error."""
        return self._total_z_axis_orientation_error / self._num_poses

    def mean_num_descents(self):
        """Return mean number of descents."""
        return float(self._total_num_descents) / float(self._num_poses)


@pytest.fixture
def configure_franka_robot_description():
    """Test fixture to configure Franka robot description object."""
    def _configure_franka_robot_description():
        # Set directory for robot description YAML files.
        config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')

        # Set absolute path to the XRDF for three link robot.
        xrdf_path = os.path.join(config_path, 'franka.xrdf')

        # Set absolute path to URDF file for three link robot.
        urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka',
                                 'franka.urdf')

        # Load and return robot description.
        return cumotion.load_robot_from_file(xrdf_path, urdf_path)

    return _configure_franka_robot_description


def load_franka_right_gripper_poses():
    """Load poses that are known to reachable for Franka's "right_gripper" frame."""
    # Load YAML from file.
    data_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'tests', 'kinematics',
                             'shared', 'reachable_franka_right_gripper_targets.yaml')
    with open(data_path, 'r') as stream:
        poses_yaml = yaml.safe_load(stream)

    # Convert each YAML pose to `cumotion.Pose3`.
    poses = []
    for pose_yaml in poses_yaml:
        position = np.array([pose_yaml['position'][0],
                             pose_yaml['position'][1],
                             pose_yaml['position'][2]])
        rotation = cumotion.Rotation3(pose_yaml['orientation']['w'],
                                      pose_yaml['orientation']['xyz'][0],
                                      pose_yaml['orientation']['xyz'][1],
                                      pose_yaml['orientation']['xyz'][2])
        poses.append(cumotion.Pose3(rotation, position))

    assert 1238 == len(poses)
    return poses


def test_franka_inverse_kinematics(configure_franka_robot_description):
    """Test inverse kinematics for CartPole to show that prismatic joints work as expected."""
    # Init kinematics from test fixture
    kinematics = configure_franka_robot_description().kinematics()

    # Set end effector frame.
    end_effector_frame = "right_gripper"

    # Load target poses from file (where all targets are known to be reachable by
    # `end_effector_frame`.
    target_poses = load_franka_right_gripper_poses()

    # Accumulate separate results for successes and failures.
    successes = AcculumulatedCcdResults()
    failures = AcculumulatedCcdResults()

    # Create config with all default parameters.
    config = cumotion.IkConfig()

    # For each `target_pose` attempt inverse kinematics and record results as success or failure.
    for target_pose in target_poses:
        results = cumotion.solve_ik(kinematics, target_pose, end_effector_frame, config)

        if results.success:
            successes.accumulate(results)
        else:
            failures.accumulate(results)

    # Check that success metrics are as expected.
    assert 1227 == successes.num_poses()
    kEps = 1e-6
    assert 6.148908509e-06 == pytest.approx(successes.mean_position_error(), abs=kEps)
    assert 0.000868862 == pytest.approx(successes.mean_x_axis_orientation_error(), abs=kEps)
    assert 0.000872612 == pytest.approx(successes.mean_y_axis_orientation_error(), abs=kEps)
    assert 0.000932250 == pytest.approx(successes.mean_z_axis_orientation_error(), abs=kEps)
    assert 2.545232273 == pytest.approx(successes.mean_num_descents(), abs=kEps)

    # Check that failure metrics are as expected.
    assert 11 == failures.num_poses()
    # NOTE: The tolerances for `failures.mean_position_error()`,
    #       `failures.mean_x_axis_orientation_error()`, `failures.mean_y_axis_orientation_error()`,
    #       and `failures.mean_x_axis_orientation_error()` are loosened so this test will pass on
    #       Jetson Orin (tested with JetPack 6.2.1).
    assert 0.0035532 == pytest.approx(failures.mean_position_error(), abs=5e-5)
    assert 0.0238811 == pytest.approx(failures.mean_x_axis_orientation_error(), abs=5e-5)
    assert 0.0176478 == pytest.approx(failures.mean_y_axis_orientation_error(), abs=5e-5)
    assert 0.0260801 == pytest.approx(failures.mean_z_axis_orientation_error(), abs=5e-5)
    assert config.max_num_descents == pytest.approx(failures.mean_num_descents(), abs=kEps)


@pytest.fixture
def configure_cart_pole_robot_description():
    """Test fixture to configure cart pole robot description object."""
    def _configure_cart_pole_robot_description():
        # Set directory for robot description YAML files.
        config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')

        # Set absolute path to the XRDF for the "cart pole" robot.
        xrdf_path = os.path.join(config_path, 'cart_pole.xrdf')

        # Set absolute path to URDF file for the "cart pole" robot.
        urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'robots',
                                 'cart_pole.urdf')

        # Load and return robot description.
        return cumotion.load_robot_from_file(xrdf_path, urdf_path)

    return _configure_cart_pole_robot_description


def load_cart_pole_tip_poses():
    """Load poses that are known to be reachable positions for CartPoles's pole_tip frame."""
    poses = []

    # Set grid of XZ points for position targets for cart pole  (keeping y = 0).
    # Orientation targets are all set to Identity with high tolerance to accepting orientations
    # (since cart pole can only achieve, in general, two orientations for a given end effector
    # positions).
    x = -0.5
    z = -0.5
    spacing = 0.1
    points_per_side = 11
    for i in range(points_per_side):
        for j in range(points_per_side):
            position = np.array([x + i * spacing, 0.0, z + j * spacing])
            poses.append(cumotion.Pose3.from_translation(position))

    return poses


def test_cart_pole_inverse_kinematics(configure_cart_pole_robot_description):
    """Test inverse kinematics for CartPole to show that prismatic joints work as expected."""
    # Init kinematics from test fixture
    kinematics = configure_cart_pole_robot_description().kinematics()

    # Set end effector frame.
    end_effector_frame = "pole_tip"

    # Generate target poses with reachable positions.
    target_poses = load_cart_pole_tip_poses()

    # Accumulate results (expect only successes).
    accumulated_results = AcculumulatedCcdResults()

    # Create config for IK.
    # It should be noted that setting `orientation_weight` to zero and using arbitrarily
    # high tolerances for orientation will incur unnecessary overhead as compared to implementing a
    # position-only IK solver. CCD could be extended to efficiently handle this use-case as needed.
    config = cumotion.IkConfig()
    config.ccd_orientation_weight = 0.0
    config.bfgs_orientation_weight = 0.0
    config.orientation_tolerance = 10.0

    # For each `target_pose` attempt inverse kinematics and record results as success or failure.
    for target_pose in target_poses:
        results = cumotion.solve_ik(kinematics, target_pose, end_effector_frame, config)

        if results.success:
            accumulated_results.accumulate(results)
        else:
            # Expect all IK attempts to be successful!
            assert False

    # Check that results metrics are as expected.
    assert len(target_poses) == accumulated_results.num_poses()

    # Expect near perfect position results.
    assert 0.0 == pytest.approx(accumulated_results.mean_position_error(), abs=1e-6)

    # Orientation error is high (as expected) since cart-pole can't control orientation independent
    # of position.
    # NOTE: The y-axis is an exception since `pole_tip` is always aligned with y-axis
    assert 1.39542562 == pytest.approx(accumulated_results.mean_x_axis_orientation_error())
    assert 0.0 == pytest.approx(accumulated_results.mean_y_axis_orientation_error())
    assert 1.39542562 == pytest.approx(accumulated_results.mean_z_axis_orientation_error())

    # Expect very few descents needed, usually only 1.
    assert 1.87603305 == pytest.approx(accumulated_results.mean_num_descents())


def test_cspace_seeds_edits(configure_franka_robot_description):
    """Test that in-place edits to `IkConfig.cspace_seeds` change the config."""
    config = cumotion.IkConfig()
    seed_0 = np.zeros(7)
    seed_1 = np.ones(7)
    seed_2 = np.full(7, 2.0)

    # Expect each in-place edit to change the seeds of the config.
    config.cspace_seeds.append(seed_0)
    config.cspace_seeds.extend([seed_1, seed_2])
    config.cspace_seeds.insert(0, seed_2)
    config.cspace_seeds[1] = seed_1
    del config.cspace_seeds[2]
    assert np.array_equal(seed_2, config.cspace_seeds.pop())
    config.cspace_seeds += [seed_0]
    assert np.array_equal([seed_2, seed_1, seed_0], np.array(config.cspace_seeds))
    config.cspace_seeds.clear()
    assert 0 == len(config.cspace_seeds)

    # Expect each seed to be a read-only copy that later edits leave unchanged.
    config.cspace_seeds = [seed_0, seed_1]
    seed = config.cspace_seeds[-1]
    with pytest.raises(ValueError):
        seed[0] = 5.0
    config.cspace_seeds = []
    assert np.array_equal(seed_1, seed)

    # Expect an out-of-range index to raise `IndexError`.
    with pytest.raises(IndexError):
        _ = config.cspace_seeds[0]

    # Expect a view to keep its config alive.
    seeds = cumotion.IkConfig().cspace_seeds
    seeds.append(seed_0)
    assert 1 == len(seeds)

    # Expect the solver to use a seed added in place. A single descent from a seed that already
    # reaches the target returns that seed.
    robot_description = configure_franka_robot_description()
    kinematics = robot_description.kinematics()
    default_position = robot_description.default_cspace_configuration()
    target_pose = kinematics.pose(default_position, "right_gripper")
    config.max_num_descents = 1
    config.cspace_seeds.append(default_position)
    results = cumotion.solve_ik(kinematics, target_pose, "right_gripper", config)
    assert results.success
    assert np.allclose(default_position, results.cspace_position, atol=1e-6)
