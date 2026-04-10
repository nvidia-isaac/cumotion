# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Unit tests for collision_sphere_generator_python.h."""

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion

# Local directory
from ._test_helper import warnings_disabled, errors_disabled


def test_collision_sphere_generator_set_param():
    """Test setting params for collision sphere generation."""
    # Create a simple, planar mesh with two triangles.
    vertices = [np.array([0.0, 0.0, 0.0]),  # Index 0
                np.array([1.0, 0.0, 0.0]),  # Index 1
                np.array([0.0, 1.0, 0.0]),  # Index 2
                np.array([1.0, 1.0, 1.0])]  # Index 3
    indices = [np.array([0, 1, 2]), np.array([1, 3, 2])]

    # Create sphere generator
    generator = cumotion.create_collision_sphere_generator(vertices, indices)
    assert 2 == generator.num_triangles()

    # Test setting valid value for `num_medial_sphere_samples`.
    assert generator.set_param("num_medial_sphere_samples", 34)
    # Test that setting a non-positive value (i.e., an invalid value) will not update the
    # `num_medial_sphere_samples`.
    with warnings_disabled:
        assert not generator.set_param("num_medial_sphere_samples", 0)
        assert not generator.set_param("num_medial_sphere_samples", -3)
    # Test that the wrong value type (e.g., floating point or boolean) will throw an error.
    with pytest.raises(Exception):
        generator.set_param("num_medial_sphere_samples", 23.0)
    with pytest.raises(Exception):
        generator.set_param("num_medial_sphere_samples", True)

    # Test setting valid value for `flip_normals`.
    assert generator.set_param("flip_normals", True)
    assert generator.set_param("flip_normals", False)

    # Test setting valid value for `min_sphere_radius`.
    assert generator.set_param("min_sphere_radius", 23.0)
    # Test that setting a non-positive value (i.e., an invalid value) will not update the
    # `min_sphere_radius`.
    with warnings_disabled:
        assert not generator.set_param("min_sphere_radius", 0.0)
        assert not generator.set_param("min_sphere_radius", -3.0)
    # Test that the wrong value type (e.g., integer or boolean) will throw an error.
    with pytest.raises(Exception):
        generator.set_param("min_sphere_radius", 23)
    with pytest.raises(Exception):
        generator.set_param("min_sphere_radius", True)

    # Test setting valid value for `seed`.
    assert generator.set_param("seed", 34)
    # Test that setting a non-positive value (i.e., an invalid value) will not update the
    # `seed`.
    with warnings_disabled:
        assert not generator.set_param("seed", 0)
        assert not generator.set_param("seed", -3)
    # Test that the wrong value type (e.g., floating point or boolean) will throw an error.
    with pytest.raises(Exception):
        generator.set_param("seed", 23.0)
    with pytest.raises(Exception):
        generator.set_param("seed", True)

    # Test setting valid value for `max_iterations`.
    assert generator.set_param("max_iterations", 34)
    # Test that setting a non-positive value (i.e., an invalid value) will not update the
    # `max_iterations`.
    with warnings_disabled:
        assert not generator.set_param("max_iterations", 0)
        assert not generator.set_param("max_iterations", -3)
    # Test that the wrong value type (e.g., floating point or boolean) will throw an error.
    with pytest.raises(Exception):
        generator.set_param("max_iterations", 23.0)
    with pytest.raises(Exception):
        generator.set_param("max_iterations", True)

    # Test setting valid value for `convergence_radius_tol`.
    assert generator.set_param("convergence_radius_tol", 23.0)
    # Test that setting a non-positive value (i.e., an invalid value) will not update the
    # `convergence_radius_tol`.
    with warnings_disabled:
        assert not generator.set_param("convergence_radius_tol", 0.0)
        assert not generator.set_param("convergence_radius_tol", -3.0)
    # Test that the wrong value type (e.g., integer or boolean) will throw an error.
    with pytest.raises(Exception):
        generator.set_param("convergence_radius_tol", 23)
    with pytest.raises(Exception):
        generator.set_param("convergence_radius_tol", True)

    # Test setting valid value for `surface_offset`.
    assert generator.set_param("surface_offset", 23.0)
    # Test that setting a non-positive value (i.e., an invalid value) will not update the
    # `surface_offset`.
    with warnings_disabled:
        assert not generator.set_param("surface_offset", 0.0)
        assert not generator.set_param("surface_offset", -3.0)
    # Test that the wrong value type (e.g., integer or boolean) will throw an error.
    with pytest.raises(Exception):
        generator.set_param("surface_offset", 23)
    with pytest.raises(Exception):
        generator.set_param("surface_offset", True)

    # Test setting valid value for `min_triangle_area`.
    assert generator.set_param("min_triangle_area", 23.0)
    # Test that setting a non-positive value (i.e., an invalid value) will not update the
    # `min_triangle_area`.
    with warnings_disabled:
        assert not generator.set_param("min_triangle_area", 0.0)
        assert not generator.set_param("min_triangle_area", -3.0)
    # Test that the wrong value type (e.g., integer or boolean) will throw an error.
    with pytest.raises(Exception):
        generator.set_param("min_triangle_area", 23)
    with pytest.raises(Exception):
        generator.set_param("min_triangle_area", True)

    # Test setting valid value for `num_voxels`.
    assert generator.set_param("num_voxels", 34)
    # Test that setting a non-positive value (i.e., an invalid value) will not update the
    # `num_voxels`.
    with warnings_disabled:
        assert not generator.set_param("num_voxels", 0)
        assert not generator.set_param("num_voxels", -3)
    # Test that the wrong value type (e.g., floating point or boolean) will throw an error.
    with pytest.raises(Exception):
        generator.set_param("num_voxels", 23.0)
    with pytest.raises(Exception):
        generator.set_param("num_voxels", True)


def compute_bounds(spheres):
    """Compute bounds for a list of spheres."""
    min = np.array([np.inf, np.inf, np.inf])
    max = np.array([-np.inf, -np.inf, -np.inf])

    for sphere in spheres:
        this_min = sphere.center - (sphere.radius * np.ones(3))
        this_max = sphere.center + (sphere.radius * np.ones(3))

        min = np.minimum(min, this_min)
        max = np.maximum(max, this_max)

    return min, max


def create_cube_mesh():
    """Create a simple, cubic mesh with twelve triangles."""
    vertices = [np.array([0.0, 0.0, 0.0]),  # Index 0
                np.array([1.0, 0.0, 0.0]),  # Index 1
                np.array([0.0, 1.0, 0.0]),  # Index 2
                np.array([1.0, 1.0, 0.0]),  # Index 3
                np.array([0.0, 0.0, 1.0]),  # Index 4
                np.array([1.0, 0.0, 1.0]),  # Index 5
                np.array([0.0, 1.0, 1.0]),  # Index 6
                np.array([1.0, 1.0, 1.0])]  # Index 7
    indices = [np.array([0, 2, 1]), np.array([1, 2, 3]),  # Bottom
               np.array([4, 5, 6]), np.array([5, 7, 6]),  # Top
               np.array([0, 1, 4]), np.array([1, 5, 4]),  # Front
               np.array([2, 6, 3]), np.array([3, 6, 7]),  # Back
               np.array([0, 6, 2]), np.array([0, 4, 6]),  # Left
               np.array([1, 3, 7]), np.array([1, 7, 5])]  # Right

    return vertices, indices


def test_collision_sphere_generator():
    """Test collision sphere generation for simple "cube" mesh."""
    # Create mesh
    vertices, indices = create_cube_mesh()

    # Create sphere generator
    generator = cumotion.create_collision_sphere_generator(vertices, indices)
    assert 12 == generator.num_triangles()

    # Inspect sampled spheres.
    samples = generator.get_sampled_spheres()

    # Expect number of sampled spheres to approximately equal the default number of samples (500).
    assert 497 == len(samples)

    # Expect sampled spheres to nearly span the original cube.
    sample_min, sample_max = compute_bounds(samples)
    assert np.zeros(3) == pytest.approx(sample_min, abs=5e-5)
    assert np.ones(3) == pytest.approx(sample_max, abs=2e-5)

    # Select a smaller set of spheres from the medial samples.
    num_spheres = 10
    selected = generator.generate_spheres(num_spheres, 0.0)

    # Expect number of selected spheres to be as specified.
    assert num_spheres == len(selected)

    # Expect sampled spheres to nearly span the original cube (but not as closely as the full set of
    # sampled spheres).
    selected_min, selected_max = compute_bounds(selected)
    assert np.zeros(3) == pytest.approx(selected_min, abs=5e-4)
    assert np.ones(3) == pytest.approx(selected_max, abs=6e-4)

    # Select a set of spheres, with their final radii expanded by `radius_offset`.
    radius_offset = 0.1
    inflated = generator.generate_spheres(num_spheres, radius_offset)

    # Expect the center positions of `inflated` to be identical to `selected`, and radius larger by
    # `radius_offset`.
    assert len(selected) == len(inflated)
    for selected_sphere, inflated_sphere in zip(selected, inflated):
        assert selected_sphere.center == pytest.approx(inflated_sphere.center)
        assert selected_sphere.radius + radius_offset == inflated_sphere.radius

    # Expect sampled spheres to extend the original cube by approximately `radius_offset`.
    inflated_min, inflated_max = compute_bounds(inflated)
    assert -radius_offset * np.ones(3) == pytest.approx(inflated_min, abs=5e-4)
    assert (1.0 + radius_offset) * np.ones(3) == pytest.approx(inflated_max, abs=6e-4)

    # Select a set of spheres, with their final radii decreased by `radius_offset`.
    shrunk = generator.generate_spheres(num_spheres, -radius_offset)

    # Expect the center positions of `shrunk` to be identical to `selected`, and radius smaller by
    # `radius_offset`.
    assert len(selected) == len(shrunk)
    for selected_sphere, shrunk_sphere in zip(selected, shrunk):
        assert selected_sphere.center == pytest.approx(shrunk_sphere.center)
        assert selected_sphere.radius - radius_offset == shrunk_sphere.radius

    # Expect sampled spheres to be retracted within the original cube by approximately
    # `radius_offset`.
    shrunk_min, shrunk_max = compute_bounds(shrunk)
    assert radius_offset * np.ones(3) == pytest.approx(shrunk_min, abs=5e-4)
    assert (1.0 - radius_offset) * np.ones(3) == pytest.approx(shrunk_max, abs=6e-4)

    # Select a set of spheres, with their final radii decreased by a really large number.
    big_offset = 1.0
    extra_shrunk = generator.generate_spheres(num_spheres, -big_offset)

    # Expect the center positions of `extra_shrunk` to be identical to `selected`, and radii to all
    # be equal to the default minimum sphere radius.
    assert len(selected) == len(extra_shrunk)
    for selected_sphere, extra_shrunk_sphere in zip(selected, extra_shrunk):
        assert selected_sphere.center == pytest.approx(extra_shrunk_sphere.center)
        assert 1e-3 == extra_shrunk_sphere.radius

    # Expect that if normals are flipped, no spheres will be sampled, and therefore no spheres will
    # be selected.
    generator.set_param("flip_normals", True)
    with errors_disabled:
        assert 0 == len(generator.get_sampled_spheres())
        assert 0 == len(generator.generate_spheres(10, 0))

    # Create a set of mostly invalid indices, expecting 8 of them to be discarded.
    bad_indices = [np.array([0, 2, 1]), np.array([1, 2, 2]),  # Bottom
                   np.array([4, 5, 6]), np.array([5, 6, 6]),  # Top
                   np.array([0, 1, 1]), np.array([1, 4, 4]),  # Front
                   np.array([2, 6, 3]), np.array([3, 3, 7]),  # Back
                   np.array([0, 6, 0]), np.array([0, 6, 6]),  # Left
                   np.array([1, 3, 7]), np.array([5, 7, 5])]  # Right
    with errors_disabled:
        bad_generator = cumotion.create_collision_sphere_generator(vertices, bad_indices)
    assert 4 == bad_generator.num_triangles()


def test_generate_collision_spheres():
    """Test collision sphere generation for simple "cube" mesh."""
    # Create mesh
    vertices, indices = create_cube_mesh()

    # Create collision spheres
    max_overshoot = 0.1
    spheres = cumotion.generate_collision_spheres(vertices, indices, max_overshoot)
    # The number of generated spheres is slightly different with MSVC on Windows than with GCC
    # on linux, presumably due to differences in floating-point rounding.
    assert len(spheres) == pytest.approx(58, abs=2)

    # Expect spheres to overshoot the original cube by approximately `max_overshoot`.
    spheres_min, spheres_max = compute_bounds(spheres)
    assert -max_overshoot * np.ones(3) == pytest.approx(spheres_min, abs=1e-12)
    assert (1.0 + max_overshoot) * np.ones(3) == pytest.approx(spheres_max, abs=1e-12)

    # Generate spheres using a smaller maximum overshoot
    smaller_max_overshoot = 0.01
    spheres2 = cumotion.generate_collision_spheres(vertices, indices, smaller_max_overshoot)

    # Expect more spheres to be needed to cover the surface with less overshoot.
    assert len(spheres2) == pytest.approx(828, abs=8)

    # Expect new spheres to overshoot the original cube by approximately `smaller_max_overshoot`.
    spheres2_min, spheres2_max = compute_bounds(spheres2)
    assert -smaller_max_overshoot * np.ones(3) == pytest.approx(spheres2_min, abs=1e-12)
    assert (1.0 + smaller_max_overshoot) * np.ones(3) == pytest.approx(spheres2_max, abs=1e-12)
