# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Unit tests for kinematics_python.h."""

# Standard Library
import os

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion

# Local directory
from ._test_helper import CUMOTION_ROOT_DIR, warnings_disabled


@pytest.fixture
def configure_three_link_arm_kinematics():
    """Test fixture to configure three link arm robot kinematics object."""
    def _configure_three_link_arm_kinematics():
        # Set directory for robot description YAML files.
        config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')

        # Set absolute path to the XRDF for the three-link robot.
        xrdf_path = os.path.join(config_path, 'three_link_arm.xrdf')

        # Set absolute path to URDF file for three link robot.
        urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia',
                                 'robots', 'three_link_arm.urdf')

        # Load and return robot description.
        return cumotion.load_robot_from_file(xrdf_path, urdf_path).kinematics()

    return _configure_three_link_arm_kinematics


def test_three_link_arm_kinematics(configure_three_link_arm_kinematics):
    """Test cumotion::Kinematics Python wrapper."""
    # Init kinematics from test fixture
    kinematics = configure_three_link_arm_kinematics()

    # Check number of C-space coordinates
    assert 3 == kinematics.num_cspace_coords()

    # Check C-space limits
    for i in range(kinematics.num_cspace_coords()):
        limits = kinematics.cspace_coord_limits(i)
        assert -3.0 == limits.lower
        assert 3.0 == limits.upper

    # Check C-space velocity, acceleration, and jerk limits
    for i in range(kinematics.num_cspace_coords()):
        assert 100.0 == kinematics.cspace_coord_velocity_limit(i)
        assert 100.0 == kinematics.cspace_coord_acceleration_limit(i)
        assert 1000.0 == kinematics.cspace_coord_jerk_limit(i)

    # Check invalid joint position.
    invalid_position = np.array([-4.5, 6.7, 1.2])
    with warnings_disabled:
        assert kinematics.within_cspace_limits(invalid_position, True) is False

    # Check valid joint position.
    valid_position = np.array([1.2, -0.2, 2.3])
    assert kinematics.within_cspace_limits(valid_position, True)

    # Check C-space coordinate names
    assert 'joint0' == kinematics.cspace_coord_name(0)
    assert 'joint1' == kinematics.cspace_coord_name(1)
    assert 'joint2' == kinematics.cspace_coord_name(2)

    # Check frame names
    assert np.all(np.array(['base', 'link0', 'link1', 'link2',
                            'end_effector']) == kinematics.frame_names())

    # Check base frame
    assert 'base' == kinematics.base_frame_name()

    # Check pose, position and orientation
    zero_position = np.zeros(3)

    pose = kinematics.pose(zero_position, "end_effector", "base")
    assert 1 == pytest.approx(pose.rotation.w())
    assert 0 == pytest.approx(pose.rotation.x())
    assert 0 == pytest.approx(pose.rotation.y())
    assert 0 == pytest.approx(pose.rotation.z())
    assert np.array([3, 0, 0]) == pytest.approx(pose.translation)
    position = kinematics.position(zero_position, "end_effector", "base")
    assert position == pytest.approx(pose.translation)
    orientation = kinematics.orientation(zero_position, "end_effector", "base")
    assert orientation.w() == pose.rotation.w()
    assert orientation.x() == pose.rotation.x()
    assert orientation.y() == pose.rotation.y()
    assert orientation.z() == pose.rotation.z()

    pose = kinematics.pose(zero_position, "end_effector", "link0")
    assert 1 == pytest.approx(pose.rotation.w())
    assert 0 == pytest.approx(pose.rotation.x())
    assert 0 == pytest.approx(pose.rotation.y())
    assert 0 == pytest.approx(pose.rotation.z())
    assert np.array([3, 0, 0]) == pytest.approx(pose.translation)
    position = kinematics.position(zero_position, "end_effector", "link0")
    assert position == pytest.approx(pose.translation)
    orientation = kinematics.orientation(zero_position, "end_effector", "link0")
    assert orientation.w() == pose.rotation.w()
    assert orientation.x() == pose.rotation.x()
    assert orientation.y() == pose.rotation.y()
    assert orientation.z() == pose.rotation.z()

    pose = kinematics.pose(zero_position, "end_effector", "link1")
    assert 1 == pytest.approx(pose.rotation.w())
    assert 0 == pytest.approx(pose.rotation.x())
    assert 0 == pytest.approx(pose.rotation.y())
    assert 0 == pytest.approx(pose.rotation.z())
    assert np.array([2, 0, 0]) == pytest.approx(pose.translation)
    position = kinematics.position(zero_position, "end_effector", "link1")
    assert position == pytest.approx(pose.translation)
    orientation = kinematics.orientation(zero_position, "end_effector", "link1")
    assert orientation.w() == pose.rotation.w()
    assert orientation.x() == pose.rotation.x()
    assert orientation.y() == pose.rotation.y()
    assert orientation.z() == pose.rotation.z()

    pose = kinematics.pose(zero_position, "end_effector", "link2")
    assert 1 == pytest.approx(pose.rotation.w())
    assert 0 == pytest.approx(pose.rotation.x())
    assert 0 == pytest.approx(pose.rotation.y())
    assert 0 == pytest.approx(pose.rotation.z())
    assert np.array([1, 0, 0]) == pytest.approx(pose.translation)
    position = kinematics.position(zero_position, "end_effector", "link2")
    assert position == pytest.approx(pose.translation)
    orientation = kinematics.orientation(zero_position, "end_effector", "link2")
    assert orientation.w() == pose.rotation.w()
    assert orientation.x() == pose.rotation.x()
    assert orientation.y() == pose.rotation.y()
    assert orientation.z() == pose.rotation.z()

    pose = kinematics.pose(zero_position, "end_effector", "end_effector")
    assert 1 == pytest.approx(pose.rotation.w())
    assert 0 == pytest.approx(pose.rotation.x())
    assert 0 == pytest.approx(pose.rotation.y())
    assert 0 == pytest.approx(pose.rotation.z())
    assert np.array([0, 0, 0]) == pytest.approx(pose.translation)
    position = kinematics.position(zero_position, "end_effector", "end_effector")
    assert position == pytest.approx(pose.translation)
    orientation = kinematics.orientation(zero_position, "end_effector", "end_effector")
    assert orientation.w() == pose.rotation.w()
    assert orientation.x() == pose.rotation.x()
    assert orientation.y() == pose.rotation.y()
    assert orientation.z() == pose.rotation.z()

    pose = kinematics.pose(zero_position, "end_effector")
    assert 1 == pytest.approx(pose.rotation.w())
    assert 0 == pytest.approx(pose.rotation.x())
    assert 0 == pytest.approx(pose.rotation.y())
    assert 0 == pytest.approx(pose.rotation.z())
    assert np.array([3, 0, 0]) == pytest.approx(pose.translation)
    position = kinematics.position(zero_position, "end_effector")
    assert position == pytest.approx(pose.translation)
    orientation = kinematics.orientation(zero_position, "end_effector")
    assert orientation.w() == pose.rotation.w()
    assert orientation.x() == pose.rotation.x()
    assert orientation.y() == pose.rotation.y()
    assert orientation.z() == pose.rotation.z()

    ziq_zag_position = np.array([np.pi / 2, -np.pi / 2, np.pi / 2])
    pose = kinematics.pose(ziq_zag_position, "end_effector")
    assert 1 / np.sqrt(2) == pytest.approx(pose.rotation.w())
    assert 0 == pytest.approx(pose.rotation.x())
    assert 0 == pytest.approx(pose.rotation.y())
    assert 1 / np.sqrt(2) == pytest.approx(pose.rotation.z())
    assert np.array([1, 2, 0]) == pytest.approx(pose.translation)
    position = kinematics.position(ziq_zag_position, "end_effector")
    assert position == pytest.approx(pose.translation)
    orientation = kinematics.orientation(ziq_zag_position, "end_effector")
    assert orientation.w() == pose.rotation.w()
    assert orientation.x() == pose.rotation.x()
    assert orientation.y() == pose.rotation.y()
    assert orientation.z() == pose.rotation.z()

    # Check jacobian (including position- and orientation-only jacobians)
    jacobian_actual = kinematics.jacobian(zero_position, "end_effector")
    jacobian_expected = np.array([
        [0., 0., 0.],
        [3., 2., 1.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [1., 1., 1.]
    ])
    assert jacobian_actual == pytest.approx(jacobian_expected)
    position_jacobian = kinematics.position_jacobian(zero_position, "end_effector")
    assert position_jacobian == pytest.approx(jacobian_actual[0:3, :])
    orientation_jacobian = kinematics.orientation_jacobian(zero_position, "end_effector")
    assert orientation_jacobian == pytest.approx(jacobian_actual[3:6, :])

    jacobian_actual = kinematics.jacobian(np.array([np.pi / 2, 0, 0]), "end_effector")
    jacobian_expected = np.array([
        [-3., -2., -1.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [1., 1., 1.]
    ])
    assert jacobian_actual == pytest.approx(jacobian_expected)
    position_jacobian = kinematics.position_jacobian(np.array([np.pi / 2, 0, 0]), "end_effector")
    assert position_jacobian == pytest.approx(jacobian_actual[0:3, :])
    orientation_jacobian = kinematics.orientation_jacobian(
        np.array([np.pi / 2, 0, 0]), "end_effector")
    assert orientation_jacobian == pytest.approx(jacobian_actual[3:6, :])

    jacobian_actual = kinematics.jacobian(np.array([0, np.pi / 2, 0]), "end_effector")
    jacobian_expected = np.array([
        [-2., -2., -1.],
        [1., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [1., 1., 1.]
    ])
    assert jacobian_actual == pytest.approx(jacobian_expected)
    position_jacobian = kinematics.position_jacobian(np.array([0, np.pi / 2, 0]), "end_effector")
    assert position_jacobian == pytest.approx(jacobian_actual[0:3, :])
    orientation_jacobian = kinematics.orientation_jacobian(
        np.array([0, np.pi / 2, 0]), "end_effector")
    assert orientation_jacobian == pytest.approx(jacobian_actual[3:6, :])

    # Check string representation
    assert "cumotion.Kinematics" == str(kinematics)


@pytest.fixture
def configure_franka_kinematics():
    """Test fixture to configure Fraka kinematics object."""
    def _configure_franka_kinematics():
        # Set directory for robot description YAML files.
        config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')

        # Set absolute path to the XRDF for Franka robot.
        xrdf_path = os.path.join(config_path, 'franka.xrdf')

        # Set absolute path to URDF file for Franka robot.
        urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka',
                                 'franka.urdf')

        # Load and return kinematics.
        return cumotion.load_robot_from_file(xrdf_path, urdf_path).kinematics()

    return _configure_franka_kinematics


def test_franka_kinematics(configure_franka_kinematics):
    """Test cumotion::Kinematics Python wrapper."""
    # Init kinematics from test fixture
    kinematics = configure_franka_kinematics()

    # Check number of C-space coordinates
    assert 7 == kinematics.num_cspace_coords()

    # Check C-space position limits
    assert -2.8973 == kinematics.cspace_coord_limits(0).lower
    assert 2.8973 == kinematics.cspace_coord_limits(0).upper
    assert -1.7628 == kinematics.cspace_coord_limits(1).lower
    assert 1.7628 == kinematics.cspace_coord_limits(1).upper
    assert -2.8973 == kinematics.cspace_coord_limits(2).lower
    assert 2.8973 == kinematics.cspace_coord_limits(2).upper
    assert -3.0718 == kinematics.cspace_coord_limits(3).lower
    assert -0.0698 == kinematics.cspace_coord_limits(3).upper
    assert -2.8973 == kinematics.cspace_coord_limits(4).lower
    assert 2.8973 == kinematics.cspace_coord_limits(4).upper
    assert -0.0175 == kinematics.cspace_coord_limits(5).lower
    assert 3.7525 == kinematics.cspace_coord_limits(5).upper
    assert -2.8973 == kinematics.cspace_coord_limits(6).lower
    assert 2.8973 == kinematics.cspace_coord_limits(6).upper

    # Check C-space velocity limits
    assert 2.175 == kinematics.cspace_coord_velocity_limit(0)
    assert 2.175 == kinematics.cspace_coord_velocity_limit(1)
    assert 2.175 == kinematics.cspace_coord_velocity_limit(2)
    assert 2.175 == kinematics.cspace_coord_velocity_limit(3)
    assert 2.610 == kinematics.cspace_coord_velocity_limit(4)
    assert 2.610 == kinematics.cspace_coord_velocity_limit(5)
    assert 2.610 == kinematics.cspace_coord_velocity_limit(6)

    # Check C-space acceleration limits
    assert 15.0 == kinematics.cspace_coord_acceleration_limit(0)
    assert 7.50 == kinematics.cspace_coord_acceleration_limit(1)
    assert 10.0 == kinematics.cspace_coord_acceleration_limit(2)
    assert 12.5 == kinematics.cspace_coord_acceleration_limit(3)
    assert 15.0 == kinematics.cspace_coord_acceleration_limit(4)
    assert 20.0 == kinematics.cspace_coord_acceleration_limit(5)
    assert 20.0 == kinematics.cspace_coord_acceleration_limit(6)

    # Check C-space jerk limits
    assert 7500.0 == kinematics.cspace_coord_jerk_limit(0)
    assert 3750.0 == kinematics.cspace_coord_jerk_limit(1)
    assert 5000.0 == kinematics.cspace_coord_jerk_limit(2)
    assert 6250.0 == kinematics.cspace_coord_jerk_limit(3)
    assert 7500.0 == kinematics.cspace_coord_jerk_limit(4)
    assert 10000.0 == kinematics.cspace_coord_jerk_limit(5)
    assert 10000.0 == kinematics.cspace_coord_jerk_limit(6)

    # Check C-space coordinate names
    assert 'panda_joint1' == kinematics.cspace_coord_name(0)
    assert 'panda_joint2' == kinematics.cspace_coord_name(1)
    assert 'panda_joint3' == kinematics.cspace_coord_name(2)
    assert 'panda_joint4' == kinematics.cspace_coord_name(3)
    assert 'panda_joint5' == kinematics.cspace_coord_name(4)
    assert 'panda_joint6' == kinematics.cspace_coord_name(5)
    assert 'panda_joint7' == kinematics.cspace_coord_name(6)

    # Check frame names
    assert np.all(np.array(['base_link',
                            'panda_link0',
                            'panda_link1',
                            'panda_link2',
                            'panda_link3',
                            'panda_link4',
                            'panda_forearm_end_pt',
                            'panda_forearm_mid_pt',
                            'panda_link5',
                            'panda_forearm_distal',
                            'panda_forearm_mid_pt_shifted',
                            'panda_link6',
                            'panda_link7',
                            'panda_link8',
                            'panda_hand',
                            'camera_bottom_screw_frame',
                            'camera_link',
                            'camera_depth_frame',
                            'camera_color_frame',
                            'camera_color_optical_frame',
                            'camera_depth_optical_frame',
                            'camera_left_ir_frame',
                            'camera_left_ir_optical_frame',
                            'camera_right_ir_frame',
                            'camera_right_ir_optical_frame',
                            'panda_face_back_left',
                            'panda_face_back_right',
                            'panda_face_left',
                            'panda_face_right',
                            'panda_leftfinger',
                            'panda_leftfingertip',
                            'panda_rightfinger',
                            'panda_rightfingertip',
                            'right_gripper',
                            'panda_wrist_end_pt']) == kinematics.frame_names())

    # Check base frame
    assert 'base_link' == kinematics.base_frame_name()
