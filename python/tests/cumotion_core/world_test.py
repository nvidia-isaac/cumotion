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

"""Unit tests for obstacle_python.h."""

# Standard Library
import math

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion


def test_world():
    """Test the `cumotion::World` and `cumotion::WorldInspector` Python wrappers."""
    # Create world.
    world = cumotion.create_world()

    # Create sphere obstacle.
    sphere = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere_radius = 1.5
    sphere.set_attribute(cumotion.Obstacle.Attribute.RADIUS, sphere_radius)

    # Add sphere obstacle to world.
    initial_x = 2.3
    initial_sphere_pose = cumotion.Pose3.from_translation(np.array([initial_x, 0.0, 0.0]))
    sphere_handle = world.add_obstacle(sphere, initial_sphere_pose)

    # Create two world views of `world`.
    world_view_a = world.add_world_view()
    world_view_b = world.add_world_view()

    # Create world inspectors for collision tests and distance queries.
    world_inspector_a = cumotion.create_world_inspector(world_view_a)
    world_inspector_b = cumotion.create_world_inspector(world_view_b)

    # Test that the inspector returns the obstacle pose and enabled state.
    assert world_inspector_a.is_enabled(sphere_handle)
    assert world_inspector_b.is_enabled(sphere_handle)
    sphere_pose_a = world_inspector_a.pose(sphere_handle)
    sphere_pose_b = world_inspector_b.pose(sphere_handle)
    np.testing.assert_array_almost_equal(
        initial_sphere_pose.translation, sphere_pose_a.translation)
    np.testing.assert_array_almost_equal(
        initial_sphere_pose.translation, sphere_pose_b.translation)
    np.testing.assert_array_almost_equal(
        initial_sphere_pose.rotation.matrix(), sphere_pose_a.rotation.matrix())
    np.testing.assert_array_almost_equal(
        initial_sphere_pose.rotation.matrix(), sphere_pose_b.rotation.matrix())

    # Test distance from a point at the origin to the sphere obstacle.
    sphere_distance_a = world_inspector_a.distance_to(sphere_handle, np.zeros(3))
    sphere_distance_b = world_inspector_b.distance_to(sphere_handle, np.zeros(3))
    assert initial_x - sphere_radius == sphere_distance_a
    assert initial_x - sphere_radius == sphere_distance_b

    # Update obstacle pose
    updated_x = 4.7
    updated_pose = cumotion.Pose3.from_translation(np.array([updated_x, 0.0, 0.0]))
    world.set_pose(sphere_handle, updated_pose)

    # Test that world views have not yet been updated.
    sphere_distance_a = world_inspector_a.distance_to(sphere_handle, np.zeros(3))
    sphere_distance_b = world_inspector_b.distance_to(sphere_handle, np.zeros(3))
    assert initial_x - sphere_radius == sphere_distance_a
    assert initial_x - sphere_radius == sphere_distance_b

    # Update world view A and test that its distance to sphere is updated while world view B is
    # unchanged.
    world_view_a.update()
    sphere_distance_a = world_inspector_a.distance_to(sphere_handle, np.zeros(3))
    sphere_distance_b = world_inspector_b.distance_to(sphere_handle, np.zeros(3))
    assert updated_x - sphere_radius == sphere_distance_a
    assert initial_x - sphere_radius == sphere_distance_b

    # Update world view B and test that its distance to sphere is updated while world view A is
    # unchanged.
    world_view_b.update()
    sphere_distance_a = world_inspector_a.distance_to(sphere_handle, np.zeros(3))
    sphere_distance_b = world_inspector_b.distance_to(sphere_handle, np.zeros(3))
    assert updated_x - sphere_radius == sphere_distance_a
    assert updated_x - sphere_radius == sphere_distance_b

    # Create cuboid obstacle.
    cuboid = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
    cuboid_size = 2.6
    cuboid.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, cuboid_size * np.ones(3))

    # Add cuboid obstacle to world.
    initial_y = 9.8
    initial_cuboid_pose = cumotion.Pose3.from_translation(np.array([0.0, initial_y, 0.0]))
    cuboid_handle = world.add_obstacle(cuboid, initial_cuboid_pose)

    # Update world views to enable cuboid obstacle.
    world_view_a.update()
    world_view_b.update()

    # Test that the inspector returns the cuboid pose and enabled state.
    assert world_inspector_a.is_enabled(cuboid_handle)
    assert world_inspector_b.is_enabled(cuboid_handle)
    cuboid_pose_a = world_inspector_a.pose(cuboid_handle)
    cuboid_pose_b = world_inspector_b.pose(cuboid_handle)
    np.testing.assert_array_almost_equal(
        initial_cuboid_pose.translation, cuboid_pose_a.translation)
    np.testing.assert_array_almost_equal(
        initial_cuboid_pose.translation, cuboid_pose_b.translation)
    np.testing.assert_array_almost_equal(
        initial_cuboid_pose.rotation.matrix(), cuboid_pose_a.rotation.matrix())
    np.testing.assert_array_almost_equal(
        initial_cuboid_pose.rotation.matrix(), cuboid_pose_b.rotation.matrix())

    # Test distance from a point at the origin to the cuboid obstacle.
    cuboid_distance_a = world_inspector_a.distance_to(cuboid_handle, np.zeros(3))
    cuboid_distance_b = world_inspector_b.distance_to(cuboid_handle, np.zeros(3))
    assert initial_y - (0.5 * cuboid_size) == cuboid_distance_a
    assert initial_y - (0.5 * cuboid_size) == cuboid_distance_b

    # Test that `distances_to` provides the same output as `distance_to`.
    [distances_a, gradients_a] = world_inspector_a.distances_to(np.zeros(3), False)
    [distances_b, gradients_b] = world_inspector_b.distances_to(np.zeros(3), False)
    expected_distance_vector = np.array([sphere_distance_a, cuboid_distance_a])
    assert 2 == len(distances_a)
    assert 2 == len(distances_b)
    assert gradients_a is None
    assert gradients_b is None
    assert (expected_distance_vector == distances_a).all()
    assert (expected_distance_vector == distances_b).all()

    # Disable the original sphere obstacle from world.
    world.disable_obstacle(sphere_handle)

    # Expect that both world views still have two enabled obstacles (since they have not been
    # updated).
    assert 2 == world_inspector_a.num_enabled_obstacles()
    assert 2 == world_inspector_b.num_enabled_obstacles()

    # Distances should still be the same as before for both world views.
    [distances_a, gradients_a] = world_inspector_a.distances_to(np.zeros(3), False)
    [distances_b, gradients_b] = world_inspector_b.distances_to(np.zeros(3), False)
    assert 2 == len(distances_a)
    assert 2 == len(distances_b)
    assert gradients_a is None
    assert gradients_b is None
    assert (expected_distance_vector == distances_a).all()
    assert (expected_distance_vector == distances_b).all()

    # Update both world views and expect the number of obstacles to go down to one (since sphere is
    # no longer enabled).
    world_view_a.update()
    world_view_b.update()
    assert 1 == world_inspector_a.num_enabled_obstacles()
    assert 1 == world_inspector_b.num_enabled_obstacles()
    assert not world_inspector_a.is_enabled(sphere_handle)
    assert not world_inspector_b.is_enabled(sphere_handle)
    assert world_inspector_a.is_enabled(cuboid_handle)
    assert world_inspector_b.is_enabled(cuboid_handle)

    # The distance output should now only only include the cuboid.
    [distances_a, gradients_a] = world_inspector_a.distances_to(np.zeros(3), False)
    [distances_b, gradients_b] = world_inspector_b.distances_to(np.zeros(3), False)
    assert 1 == len(distances_a)
    assert 1 == len(distances_b)
    assert gradients_a is None
    assert gradients_b is None
    assert cuboid_distance_a == distances_a[0]
    assert cuboid_distance_a == distances_b[0]

    # Re-enable sphere for collision tests.
    world.enable_obstacle(sphere_handle)
    world_view_a.update()
    world_view_b.update()
    assert world_inspector_a.is_enabled(sphere_handle)
    assert world_inspector_b.is_enabled(sphere_handle)

    # Define a test sphere known to be in collision with the cuboid obstacle, but no the sphere
    # obstacle.
    test_center = np.array([0.0, initial_y + 3.0, 0.0])
    test_radius = 4.0

    # Expect collision with cuboid.
    assert world_inspector_a.in_collision(cuboid_handle, test_center, test_radius)
    assert world_inspector_b.in_collision(cuboid_handle, test_center, test_radius)

    # Expect no collision with sphere.
    assert not world_inspector_a.in_collision(sphere_handle, test_center, test_radius)
    assert not world_inspector_b.in_collision(sphere_handle, test_center, test_radius)

    # Expect collision with world (i.e., any enabled obstacle).
    assert world_inspector_a.in_collision(test_center, test_radius)
    assert world_inspector_b.in_collision(test_center, test_radius)

    # Disable cuboid obstacle and expect no collision with the world (i.e., any enabled obstacle).
    world.disable_obstacle(cuboid_handle)
    world_view_a.update()
    world_view_b.update()
    assert not world_inspector_a.in_collision(test_center, test_radius)
    assert not world_inspector_b.in_collision(test_center, test_radius)

    # Re-enable cuboid obstacle and expect collision with the world again.
    world.enable_obstacle(cuboid_handle)
    world_view_a.update()
    world_view_b.update()
    assert world_inspector_a.in_collision(test_center, test_radius)
    assert world_inspector_b.in_collision(test_center, test_radius)

    # Remove all obstacles from world.
    world.remove_obstacle(cuboid_handle)
    world.remove_obstacle(sphere_handle)
    world_view_a.update()
    world_view_b.update()

    # Expect no collisions since there are no obstacles enabled.
    assert not world_inspector_a.in_collision(test_center, test_radius)
    assert not world_inspector_b.in_collision(test_center, test_radius)

    # Additionally, the number of enabled obstacles for each view should be reported as zero.
    assert 0 == world_inspector_a.num_enabled_obstacles()
    assert 0 == world_inspector_b.num_enabled_obstacles()

    # Finally, there should be zero distances reported for both world views.
    [distances_a, gradients_a] = world_inspector_a.distances_to(np.zeros(3), False)
    [distances_b, gradients_b] = world_inspector_b.distances_to(np.zeros(3), False)
    assert 0 == len(distances_a)
    assert 0 == len(distances_b)
    assert gradients_a is None
    assert gradients_b is None


def test_world_inspector_pose_and_is_enabled():
    """Test `pose()` and `is_enabled()` return correct values and only update after view sync.

    WorldInspector reflects the `WorldView` state; values do not change until
    `world_view.update()` is called.
    """
    world = cumotion.create_world()

    # Add obstacle with default (Identity) pose.
    sphere = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere.set_attribute(cumotion.Obstacle.Attribute.RADIUS, 1.0)
    handle = world.add_obstacle(sphere)
    world_view = world.add_world_view()
    inspector = cumotion.create_world_inspector(world_view)

    assert inspector.is_enabled(handle)
    identity_pose = inspector.pose(handle)
    np.testing.assert_array_almost_equal(np.zeros(3), identity_pose.translation)
    np.testing.assert_array_almost_equal(np.eye(3), identity_pose.rotation.matrix())

    # Set a translation-only pose. The inspector does not update until `world_view` is synced.
    translation = np.array([1.0, 2.0, 3.0])
    pose_translation = cumotion.Pose3.from_translation(translation)
    world.set_pose(handle, pose_translation)

    pose_before_view_update = inspector.pose(handle)
    np.testing.assert_array_almost_equal(np.zeros(3), pose_before_view_update.translation)
    np.testing.assert_array_almost_equal(np.eye(3), pose_before_view_update.rotation.matrix())

    world_view.update()
    pose_after_update = inspector.pose(handle)
    np.testing.assert_array_almost_equal(translation, pose_after_update.translation)
    np.testing.assert_array_almost_equal(np.eye(3), pose_after_update.rotation.matrix())

    # Set a pose with rotation. Again, the inspector does not update until `world_view` is synced.
    quarter_pi = math.pi / 4.0
    rotation = cumotion.Rotation3.from_scaled_axis(np.array([0.0, 0.0, quarter_pi]))
    new_translation = np.array([4.0, 5.0, 6.0])
    pose_with_rotation = cumotion.Pose3(rotation, new_translation)
    world.set_pose(handle, pose_with_rotation)

    pose_before_second_view_update = inspector.pose(handle)
    np.testing.assert_array_almost_equal(translation, pose_before_second_view_update.translation)
    np.testing.assert_array_almost_equal(
        np.eye(3), pose_before_second_view_update.rotation.matrix())

    world_view.update()
    pose_with_rot = inspector.pose(handle)
    np.testing.assert_array_almost_equal(new_translation, pose_with_rot.translation)
    np.testing.assert_array_almost_equal(rotation.matrix(), pose_with_rot.rotation.matrix())

    # `is_enabled()` also reflects the world view; disabling in world does not
    # affect the inspector until `world_view.update()` is called.
    world.disable_obstacle(handle)
    assert inspector.is_enabled(handle)
    world_view.update()
    assert not inspector.is_enabled(handle)


def test_min_distance():
    """Test that `min_distance()` returns the minimum distance from `distances_to()`."""
    # Create world.
    world = cumotion.create_world()

    # Add randomly generated cuboids to the world.
    num_cuboids = 20
    task_space_bounds = 10.0 * np.ones(3)
    for _ in range(num_cuboids):
        cuboid = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
        cuboid_size = np.random.uniform(0.1 * np.ones(3), np.ones(3))
        cuboid.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, cuboid_size)
        random_position = np.random.uniform(-task_space_bounds, task_space_bounds)
        pose = cumotion.Pose3.from_translation(random_position)
        world.add_obstacle(cuboid, pose)

    # Create a world view, checking that number of obstacles is as expected.
    world_view = world.add_world_view()
    world_inspector = cumotion.create_world_inspector(world_view)
    assert num_cuboids == world_inspector.num_enabled_obstacles()

    # For a set of randomly generated test points, check that `min_distance()` is equivalent to the
    # minimum distance returned by `distances_to()`.
    num_test_points = 50
    for _ in range(num_test_points):
        point = np.random.uniform(-task_space_bounds, task_space_bounds)
        print("point: ", point)

        # Compute distances to all obstacles.
        distances, gradients = world_inspector.distances_to(point)

        # Compute the minimum distance and corresponding gradient using `min_distance()`
        min_gradient = np.zeros(3)
        min_distance = world_inspector.min_distance(point, min_gradient)

        # Check that the `min_distance` corresponds to the smallest value in `distances`
        assert min_distance == min(distances)

        # Check that the corresponding gradient is correct (i.e., corresponds to the minimum
        # distance)
        min_index = distances.index(min_distance)
        assert (gradients[min_index] == min_gradient).all()


def test_worldview_distance_gradients():
    """Test `WorldInspector.distance_to()` and `distances_to()` gradient calculations."""
    # Create world.
    world = cumotion.create_world()

    # Create sphere obstacle.
    sphere = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere_radius = 3.0
    sphere.set_attribute(cumotion.Obstacle.Attribute.RADIUS, sphere_radius)

    # Add four sphere obstacles to world.
    offset = 10.0
    pose1 = cumotion.Pose3.from_translation(np.array([offset, 0.0, 0.0]))
    pose2 = cumotion.Pose3.from_translation(np.array([0.0, offset, 0.0]))
    pose3 = cumotion.Pose3.from_translation(np.array([offset, offset, 0.0]))
    handle0 = world.add_obstacle(sphere)  # Default pose at origin
    handle1 = world.add_obstacle(sphere, pose1)
    handle2 = world.add_obstacle(sphere, pose2)
    handle3 = world.add_obstacle(sphere, pose3)

    # Create a view of `world` and a world inspector using that view.
    world_view = world.add_world_view()
    world_inspector = cumotion.create_world_inspector(world_view)

    # Test distance to each obstacle from a point in the center of obstacles.
    point = np.array([0.5 * offset, 0.5 * offset, 0.0])
    expected_distance = offset * 0.5 * math.sqrt(2.0) - sphere_radius
    assert expected_distance == world_inspector.distance_to(handle0, point)
    assert expected_distance == world_inspector.distance_to(handle1, point)
    assert expected_distance == world_inspector.distance_to(handle2, point)
    assert expected_distance == world_inspector.distance_to(handle3, point)

    # Test distance from all enabled obstacles at once.
    expected_distances = np.array(expected_distance * np.ones(4))
    [distances, gradients] = world_inspector.distances_to(point, False)
    assert 4 == len(distances)
    assert gradients is None
    assert (expected_distances == distances).all()

    # Test distance gradient from each obstacles.
    component = 0.5 * math.sqrt(2.0)

    # Obstacle 0
    gradient0 = np.zeros(3)
    world_inspector.distance_to(handle0, point, gradient0)
    assert np.array([component, component, 0.0]) == pytest.approx(gradient0)

    # Obstacle 1
    gradient1 = np.zeros(3)
    world_inspector.distance_to(handle1, point, gradient1)
    assert np.array([-component, component, 0.0]) == pytest.approx(gradient1)

    # Obstacle 2
    gradient2 = np.zeros(3)
    world_inspector.distance_to(handle2, point, gradient2)
    assert np.array([component, -component, 0.0]) == pytest.approx(gradient2)

    # Obstacle 3
    gradient3 = np.zeros(3)
    world_inspector.distance_to(handle3, point, gradient3)
    assert np.array([-component, -component, 0.0]) == pytest.approx(gradient3)

    # Test distance gradient from all obstacles.
    [all_distances, all_gradients] = world_inspector.distances_to(point)
    assert 4 == len(all_distances)
    assert 4 == len(all_gradients)
    assert (expected_distances == all_distances).all()
    assert gradient0 == pytest.approx(all_gradients[0])
    assert gradient1 == pytest.approx(all_gradients[1])
    assert gradient2 == pytest.approx(all_gradients[2])
    assert gradient3 == pytest.approx(all_gradients[3])


def test_sdf_obstacles():
    """Test SDF obstacles."""
    # Create world.
    sdf_world = cumotion.create_world()

    num_voxels_x = 30
    num_voxels_y = 32
    num_voxels_z = 33
    voxel_size = 0.015
    host_precision = cumotion.Obstacle.GridPrecision.FLOAT
    device_precision = cumotion.Obstacle.GridPrecision.FLOAT

    # Test that an SDF authored using a sphere as a distance function matches a sphere obstacle.

    # Create SDF obstacle.
    sdf = cumotion.create_obstacle(cumotion.Obstacle.Type.SDF)
    sdf.set_attribute(cumotion.Obstacle.Attribute.GRID,
                      cumotion.Obstacle.Grid(num_voxels_x, num_voxels_y, num_voxels_z,
                                             voxel_size, host_precision, device_precision))
    sdf_handle = sdf_world.add_obstacle(sdf)

    # Populate an SDF grid using a sphere as a distance function.
    def populate_sdf_grid_host(sdf_data, voxel_size, sphere_position, sphere_radius):
        for i in range(sdf_data.shape[0]):
            for j in range(sdf_data.shape[1]):
                for k in range(sdf_data.shape[2]):
                    voxel_position = np.array([i, j, k]) * voxel_size + voxel_size * 0.5
                    sdf_data[i, j, k] = \
                        np.linalg.norm(voxel_position - sphere_position) - sphere_radius

    sphere_position = np.array([0.3, 0.31, 0.25])
    sphere_radius = 0.1
    sdf_data_fp64 = np.empty((num_voxels_x, num_voxels_y, num_voxels_z), dtype=np.float64)
    populate_sdf_grid_host(sdf_data_fp64, voxel_size, sphere_position, sphere_radius)

    sdf_world.set_sdf_grid_values_from_host(sdf_handle, sdf_data_fp64)
    sdf_world_view = sdf_world.add_world_view()
    sdf_world_inspector = cumotion.create_world_inspector(sdf_world_view)

    sphere_world = cumotion.create_world()
    sphere_obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere_obstacle.set_attribute(cumotion.Obstacle.Attribute.RADIUS, sphere_radius)
    sphere_handle = sphere_world.add_obstacle(sphere_obstacle,
                                              cumotion.Pose3.from_translation(sphere_position))

    sphere_world_view = sphere_world.add_world_view()
    sphere_world_view.update()
    sphere_world_inspector = cumotion.create_world_inspector(sphere_world_view)

    # Test that the distance to the SDF obstacle matches the distance to the sphere obstacle for
    # points sampled within the bounds of the SDF.
    np.random.seed(20)
    num_test_points = 1000

    # Maximum expected error between distance lookups in the SDF and the analytical distance
    # function.
    max_expected_error = voxel_size * math.sqrt(3.0)
    for _ in range(num_test_points):
        # Sample a random point within the bounds of the SDF.
        point = np.random.uniform(0.0,
                                  min(num_voxels_x, num_voxels_y, num_voxels_z) * voxel_size, 3)
        distance_at_query_poiunt = sdf_world_inspector.distance_to(sdf_handle, point)
        sphere_distance = sphere_world_inspector.distance_to(sphere_handle, point)

        assert abs(distance_at_query_poiunt - sphere_distance) < max_expected_error

    # Validate matching behavior with `float` precision.
    sdf_world_fp32 = cumotion.create_world()
    sdf_obstacle_fp32 = cumotion.create_obstacle(cumotion.Obstacle.Type.SDF)
    sdf_obstacle_fp32.set_attribute(cumotion.Obstacle.Attribute.GRID,
                                    cumotion.Obstacle.AttributeValue(
                                        cumotion.Obstacle.Grid(num_voxels_x, num_voxels_y,
                                                               num_voxels_z, voxel_size,
                                                               host_precision, device_precision)))
    sdf_handle_fp32 = sdf_world_fp32.add_obstacle(sdf_obstacle_fp32)
    sdf_data_fp32 = np.empty((num_voxels_x, num_voxels_y, num_voxels_z), dtype=np.float32)
    populate_sdf_grid_host(sdf_data_fp32, voxel_size, sphere_position, sphere_radius)
    sdf_world_fp32.set_sdf_grid_values_from_host(sdf_handle_fp32, sdf_data_fp32)
    sdf_world_view_fp32 = sdf_world_fp32.add_world_view()
    sdf_world_inspector_fp32 = cumotion.create_world_inspector(sdf_world_view_fp32)

    # Test that SDF queries match for the obstacles populated with `float` and `double` precision.
    expected_floating_point_error = 1e-6
    for _ in range(num_test_points):
        # Sample points both inside and outside the SDF.
        point = np.random.uniform(-1.0, 4.0, 3)
        distance_at_query_poiunt = sdf_world_inspector.distance_to(sdf_handle, point)
        sdf_distance_fp32 = sdf_world_inspector_fp32.distance_to(sdf_handle_fp32, point)
        assert abs(distance_at_query_poiunt - sdf_distance_fp32) < expected_floating_point_error

    # Test removing SDF obstacle from the world.
    query_point = np.array([0.3, 0.2, 0.1])
    sdf_world.remove_obstacle(sdf_handle)

    # Expect that the obstacle still exists in the outdated world view.
    sdf_distances = sdf_world_inspector.distances_to(query_point)[0]
    assert len(sdf_distances) == 1
    distance_at_query_point = sdf_distances[0]
    assert abs(distance_at_query_point - np.linalg.norm(query_point - sphere_position)
               + sphere_radius) < max_expected_error

    # Expect that the obstacle is no longer in the updated world view.
    sdf_world_view.update()
    assert len(sdf_world_inspector.distances_to(query_point)[0]) == 0

    # Test setting the pose of an SDF obstacle.
    new_translation = np.array([0.1, 0.2, 0.3])
    new_pose = cumotion.Pose3.from_translation(new_translation)
    sdf_world_fp32.set_pose(sdf_handle_fp32, new_pose)
    sdf_world_view_fp32.update()
    transformed_query_point = query_point + new_translation
    distance_at_transformed_query_point = sdf_world_inspector_fp32.distance_to(
        sdf_handle_fp32, transformed_query_point)
    assert abs(distance_at_transformed_query_point - distance_at_query_point) < \
        expected_floating_point_error

    # Test disabling sdf obstacles.
    sdf_world_fp32.disable_obstacle(sdf_handle_fp32)
    sdf_world_view_fp32.update()
    assert len(sdf_world_inspector_fp32.distances_to(transformed_query_point)[0]) == 0


def test_inspect_sdf():
    """Test `World.inspect_sdf()` function."""
    # Create world.
    world = cumotion.create_world()

    num_voxels_x = 30
    num_voxels_y = 32
    num_voxels_z = 33
    voxel_size = 0.015
    host_precision = cumotion.Obstacle.GridPrecision.FLOAT
    device_precision = cumotion.Obstacle.GridPrecision.FLOAT

    # Create SDF obstacle.
    sdf = cumotion.create_obstacle(cumotion.Obstacle.Type.SDF)
    sdf.set_attribute(cumotion.Obstacle.Attribute.GRID,
                      cumotion.Obstacle.Grid(num_voxels_x, num_voxels_y, num_voxels_z,
                                             voxel_size, host_precision, device_precision))
    sdf_handle = world.add_obstacle(sdf)

    # Populate an SDF grid using a sphere as a distance function.
    def populate_sdf_grid_host(sdf_data, voxel_size, sphere_position, sphere_radius):
        for i in range(sdf_data.shape[0]):
            for j in range(sdf_data.shape[1]):
                for k in range(sdf_data.shape[2]):
                    voxel_position = np.array([i, j, k]) * voxel_size + voxel_size * 0.5
                    sdf_data[i, j, k] = \
                        np.linalg.norm(voxel_position - sphere_position) - sphere_radius

    sphere_position = np.array([0.2, 0.2, 0.2])
    sphere_radius = 0.1
    sdf_data = np.empty((num_voxels_x, num_voxels_y, num_voxels_z), dtype=np.float32)
    populate_sdf_grid_host(sdf_data, voxel_size, sphere_position, sphere_radius)

    # Populate the SDF with valid data.
    world.set_sdf_grid_values_from_host(sdf_handle, sdf_data)

    # Inspect SDF with default tolerances - expect no errors since it was populated from an
    # analytically correct distance function.
    results = world.inspect_sdf(sdf_handle)
    assert results.num_errors() == 0
    assert results.num_voxels_matching_all_neighbors == 0
    assert results.num_voxels_too_far_from_neighbors == 0

    # Inspect SDF with explicit default tolerances.
    tolerances = cumotion.World.SdfInspectionTolerances()
    results = world.inspect_sdf(sdf_handle, tolerances)
    assert results.num_errors() == 0

    # Introduce an outlier by modifying a single voxel value.
    outlier_coord = (num_voxels_x // 2, num_voxels_y // 3, num_voxels_z // 4)
    outlier_error = voxel_size * 2.0
    sdf_data_with_outlier = sdf_data.copy()
    sdf_data_with_outlier[outlier_coord] += outlier_error

    world.set_sdf_grid_values_from_host(sdf_handle, sdf_data_with_outlier)

    # Inspect SDF - expect errors due to the outlier voxel and its neighbors.
    results = world.inspect_sdf(sdf_handle)
    assert results.num_voxels_too_far_from_neighbors == 7
    assert results.num_voxels_matching_all_neighbors == 0
    assert results.num_errors() == 7

    # Set tolerance to the introduced error - expect no errors.
    tolerances = cumotion.World.SdfInspectionTolerances(0.0, outlier_error)
    results = world.inspect_sdf(sdf_handle, tolerances)
    assert results.num_voxels_too_far_from_neighbors == 0
    assert results.num_voxels_matching_all_neighbors == 0
    assert results.num_errors() == 0

    # Create SDF data with voxels matching all neighbors (a homogeneous region).
    sdf_data_homogeneous = sdf_data.copy()
    homogeneous_value = -1.0
    center_coord = (num_voxels_x // 2, num_voxels_y // 2, num_voxels_z // 2)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                coord = (center_coord[0] + i, center_coord[1] + j, center_coord[2] + k)
                sdf_data_homogeneous[coord] = homogeneous_value

    world.set_sdf_grid_values_from_host(sdf_handle, sdf_data_homogeneous)

    # Inspect SDF - expect the center voxel to match all neighbors.
    results = world.inspect_sdf(sdf_handle)
    assert results.num_voxels_matching_all_neighbors == 1
    # The homogeneous region will also cause "too far from neighbor" errors at its boundaries.
    assert results.num_errors() >= 1


def test_inspect_sdf_error_cases():
    """Test `World.inspect_sdf()` error cases."""
    world = cumotion.create_world()

    # Error case 1: Attempting to inspect a non-SDF obstacle should raise an exception.
    sphere = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere.set_attribute(cumotion.Obstacle.Attribute.RADIUS, 1.0)
    sphere_handle = world.add_obstacle(sphere)

    with pytest.raises(RuntimeError):
        world.inspect_sdf(sphere_handle)

    # Error case 2: Attempting to inspect an unpopulated SDF should raise an exception.
    num_voxels_x = 10
    num_voxels_y = 10
    num_voxels_z = 10
    voxel_size = 0.01
    host_precision = cumotion.Obstacle.GridPrecision.FLOAT
    device_precision = cumotion.Obstacle.GridPrecision.FLOAT

    sdf = cumotion.create_obstacle(cumotion.Obstacle.Type.SDF)
    sdf.set_attribute(cumotion.Obstacle.Attribute.GRID,
                      cumotion.Obstacle.Grid(num_voxels_x, num_voxels_y, num_voxels_z,
                                             voxel_size, host_precision, device_precision))
    sdf_handle = world.add_obstacle(sdf)

    with pytest.raises(RuntimeError):
        world.inspect_sdf(sdf_handle)

    # Populate the SDF so we can test the removed obstacle case.
    sdf_data = np.zeros((num_voxels_x, num_voxels_y, num_voxels_z), dtype=np.float32)
    world.set_sdf_grid_values_from_host(sdf_handle, sdf_data)

    # Verify inspection works after populating.
    results = world.inspect_sdf(sdf_handle)
    assert results is not None

    # Error case 3: Attempting to inspect a removed obstacle should raise an exception.
    world.remove_obstacle(sdf_handle)

    with pytest.raises(RuntimeError):
        world.inspect_sdf(sdf_handle)
