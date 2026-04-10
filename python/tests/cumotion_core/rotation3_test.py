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

"""Unit tests for rotation3_python.h."""

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion


def test_rotation3():
    """Test the cumotion::Rotation3 Python wrapper."""
    # WXYZ constructor
    rot = cumotion.Rotation3(1, 2, 3, 4)
    assert 1 / np.sqrt(30) == pytest.approx(rot.w())
    assert 2 / np.sqrt(30) == pytest.approx(rot.x())
    assert 3 / np.sqrt(30) == pytest.approx(rot.y())
    assert 4 / np.sqrt(30) == pytest.approx(rot.z())

    # Angle axis constructor
    rot = cumotion.Rotation3.from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
    assert 1 / np.sqrt(2) == pytest.approx(rot.w())
    assert 0 == pytest.approx(rot.x())
    assert 0 == pytest.approx(rot.y())
    assert 1 / np.sqrt(2) == pytest.approx(rot.z())

    # Rotation matrix constructor
    mat = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0]])
    rot = cumotion.Rotation3.from_matrix(mat)

    assert 0 == pytest.approx(rot.w())
    assert 0 == pytest.approx(rot.x())
    assert 1 == pytest.approx(rot.y())
    assert 0 == pytest.approx(rot.z())

    # From scaled axis and getter
    scaled_axis = np.array([0, 0, np.pi / 2])
    rot = cumotion.Rotation3.from_scaled_axis(scaled_axis)
    assert 1 / np.sqrt(2) == pytest.approx(rot.w())
    assert 0 == pytest.approx(rot.x())
    assert 0 == pytest.approx(rot.y())
    assert 1 / np.sqrt(2) == pytest.approx(rot.z())

    assert scaled_axis == pytest.approx(rot.scaled_axis())

    # Identity
    rot = cumotion.Rotation3.identity()
    assert 1 == pytest.approx(rot.w())
    assert 0 == pytest.approx(rot.x())
    assert 0 == pytest.approx(rot.y())
    assert 0 == pytest.approx(rot.z())

    # Quaternion and matrix
    w = 0.73
    x = 0.183
    y = 0.365
    z = 0.548
    rot = cumotion.Rotation3(w, x, y, z)
    assert w == pytest.approx(rot.w(), rel=5e-5)
    assert x == pytest.approx(rot.x(), rel=5e-5)
    assert y == pytest.approx(rot.y(), rel=5e-5)
    assert z == pytest.approx(rot.z(), rel=5e-5)

    # Inverse and rotation multiplication overload
    rot = cumotion.Rotation3(w, x, y, z)
    assert np.identity(3) == pytest.approx((rot * rot.inverse()).matrix())
    assert np.identity(3) == pytest.approx((rot.inverse() * rot).matrix())

    # Vector multiplication overload
    rot = cumotion.Rotation3.from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
    vec = np.array([1, 0, 0])
    assert np.array([0, 1, 0]) == pytest.approx(rot * vec)

    # String representation
    rot = cumotion.Rotation3.identity()
    assert "cumotion.Rotation3(w=1, x=0, y=0, z=0)" == repr(rot)

    # Distance between rotations
    angle = 1.45
    rotation_a = cumotion.Rotation3.identity()
    rotation_b = cumotion.Rotation3.from_axis_angle(np.array([1.0, 0.0, 0.0]), angle)
    # Expect the same angle regardless of order in which rotations are passed in.
    assert angle == pytest.approx(cumotion.Rotation3.distance(rotation_a, rotation_b))
    assert angle == pytest.approx(cumotion.Rotation3.distance(rotation_b, rotation_a))
    # Expect the angle between a rotations and itself to be zero.
    assert 0.0 == cumotion.Rotation3.distance(rotation_a, rotation_a)
    assert 0.0 == cumotion.Rotation3.distance(rotation_b, rotation_b)

    # Slerp between rotations
    slerped_rotation = cumotion.Rotation3.slerp(rotation_a, rotation_b, 0.0)
    assert slerped_rotation.matrix() == pytest.approx(rotation_a.matrix())
    slerped_rotation = cumotion.Rotation3.slerp(rotation_a, rotation_b, 1.0)
    assert slerped_rotation.matrix() == pytest.approx(rotation_b.matrix())
    t = 0.3
    slerped_rotation = cumotion.Rotation3.slerp(rotation_a, rotation_b, t)
    assert (t * angle) == pytest.approx(cumotion.Rotation3.distance(rotation_a, slerped_rotation))
    assert ((1.0 - t) * angle) == \
           pytest.approx(cumotion.Rotation3.distance(rotation_b, slerped_rotation))
