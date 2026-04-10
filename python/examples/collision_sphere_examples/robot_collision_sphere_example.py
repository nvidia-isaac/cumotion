#!/usr/bin/env python3

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

"""This example demonstrates how to generate collision spheres for a robot."""

# Standard Library
import os

# Third Party
import colorsys
import numpy as np
import yaml

try:
    import cumotion_vis  # noqa: F401 import is unused.
except ImportError:
    print("'cumotion_vis' not installed. Cannot run robot collision sphere example.")
    print("CUMOTION EXAMPLE SKIPPED")
    exit(0)

import open3d as o3d  # Guaranteed to be installed if `cumotion_vis` is instealled.

# cuMotion
import cumotion

# Set cuMotion root directory
CUMOTION_ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def generate_spheres_from_mesh_file(mesh_filename, num_spheres, radius_offset):
    """Create a set of collision spheres from a mesh.

    :param mesh_filename: Absolute path to mesh file
    :param num_spheres: Desired number of spheres to approximate volume
    :param radius_offset: Offset to add to generated sphere. If set to zero, the spheres will be
                          tangent to the mesh surface. Positive values will cause spheres to extend
                          beyond the mesh and negative values will shrink the spheres into the mesh.
    :return: Set of spheres
    """
    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    generator = cumotion.create_collision_sphere_generator(mesh.vertices, mesh.triangles)
    return generator.generate_spheres(num_spheres, radius_offset)


# Write sphere positions and radii to `output`.
def write_sphere_data(output, spheres):
    """Write sphere positions and radii to file.

    :param output: Filestream to which sphere data should be written
    :param spheres: List of spheres with center positions and radii
    """
    for sphere in spheres:
        x = sphere.center[0]
        y = sphere.center[1]
        z = sphere.center[2]
        output.write("    - center: [" + str(x) + ", " + str(y) + ", " + str(z) + "]\n")
        output.write("      radius: " + str(sphere.radius) + "\n")


def generate_spheres_from_config(config_filename, output_filename, enable_visualization):
    """Generate collision spheres from YAML specification.

    :param config_filename: Absolute filepath to YAML configuration.
    :param output_filename: Filepath to which collision spheres will be written.
    :param enable_visualization: Optionally, visualize spheres generated for each mesh.
    """
    # Load collision sphere specification from YAML file.
    with open(config_filename, "r") as stream:
        data = yaml.safe_load(stream)

    # Setup output file
    output = open(output_filename, "w")
    output.write("collision_spheres:\n")

    mesh_directory = data["mesh_directory"]
    sphere_spec = data["collision_spheres"]
    for link in sphere_spec:
        # Generate spheres
        filename = os.path.join(CUMOTION_ROOT_DIR, mesh_directory, link["mesh"])
        spheres = generate_spheres_from_mesh_file(filename,
                                                  link["num_spheres"],
                                                  link["radius_offset"])

        # Write sphere data to `output`.
        output.write("  - " + link["name"] + ":\n")
        write_sphere_data(output, spheres)

        # Optionally visualize each set of collision spheres.
        if enable_visualization:
            print("Visualizing frame: " + link["name"])
            visualize(filename, spheres)


def visualize(mesh_filename, spheres):
    """Visualize mesh as wireframe with collision spheres inside.

    Visualization window will stay open until manually closed.

    :param mesh_filename: Absolute file path to mesh (will be shown as wireframe)
    :param spheres: Set of spheres (will be shown as solids)
    """
    # Convert mesh to wireframe representation
    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    mesh_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_wireframe.paint_uniform_color(np.array([0.463, 0.725, 0.000]))  # NVIDIA Green

    # Initialize list of renderables with wireframe of mesh.
    renderables = [mesh_wireframe]

    # Add solid spheres to list of renderables
    for i, sphere in enumerate(spheres):
        # Create sphere geometry.
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(sphere.radius)
        sphere_mesh.translate(sphere.center)

        # Add color in "rainbow" order.
        hue = i / len(spheres)
        sphere_mesh.paint_uniform_color(colorsys.hsv_to_rgb(hue, 1.0, 1.0))

        # Compute normals for better visualization
        sphere_mesh.compute_triangle_normals()
        sphere_mesh.compute_vertex_normals()

        # Add to list of renderables
        renderables.append(sphere_mesh)

    o3d.visualization.draw_geometries(renderables)


if __name__ == '__main__':
    cumotion.set_log_level(cumotion.LogLevel.ERROR)

    enable_visualization = True

    # Generate collision spheres for Fanuc M20-iA
    if True:
        fanuc_spec = 'content/nvidia/collision_sphere_generation/fanuc_m20ia_collision_spec.yaml'
        generate_spheres_from_config(os.path.join(CUMOTION_ROOT_DIR, fanuc_spec),
                                     "fanuc_m20ia_collision_spheres.yaml",
                                     enable_visualization)

    # Generate collision spheres for Fanuc M900ib-700
    if True:
        fanuc_spec = 'content/nvidia/collision_sphere_generation/' \
                     'fanuc_m900ib700_collision_spec.yaml'
        generate_spheres_from_config(os.path.join(CUMOTION_ROOT_DIR, fanuc_spec),
                                     "fanuc_m900ib700_collision_spheres.yaml",
                                     enable_visualization)

    # Generate collision spheres for Franka
    if True:
        franka_spec = 'content/nvidia/collision_sphere_generation/franka_collision_spec.yaml'
        generate_spheres_from_config(os.path.join(CUMOTION_ROOT_DIR, franka_spec),
                                     "franka_collision_spheres.yaml",
                                     enable_visualization)

    print("CUMOTION EXAMPLE COMPLETED SUCCESSFULLY")
