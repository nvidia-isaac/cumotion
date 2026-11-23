#!/usr/bin/env python3

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

"""Demonstrate visualizing the same robot in multiple positions at once with different colors."""

# Standard Library
import argparse
import os
from time import sleep

# Third Party
import numpy as np

# cuMotion
import cumotion
try:
    from cumotion_vis.visualizer import (
        FanucM20iAVisualization,
        FrankaVisualization,
        UR5Visualization,
        UR10Visualization,
        Visualizer,
    )
except ImportError:
    print("Visualizer not installed. Cannot run visualization example.")
    print("CUMOTION EXAMPLE SKIPPED")
    exit(0)

CUMOTION_ROOT_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)

ROBOT_CONFIGS = {
    'fanuc_m20ia': {
        'visualization': FanucM20iAVisualization,
        'xrdf': os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared',
                             'fanuc_m20ia.xrdf'),
        'urdf': os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'fanuc', 'm20ia',
                             'm20ia.urdf'),
        'mesh_folder': os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'fanuc', 'm20ia',
                                    'meshes', 'visual'),
    },
    'franka': {
        'visualization': FrankaVisualization,
        'xrdf': os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'franka.xrdf'),
        'urdf': os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka',
                             'franka.urdf'),
        'mesh_folder': os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka',
                                    'meshes'),
    },
    'ur5': {
        'visualization': UR5Visualization,
        'xrdf': os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur5.xrdf'),
        'urdf': os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur5', 'ur5.urdf'),
        'mesh_folder': os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party',
                                    'universal_robots', 'ur5', 'meshes', 'visual'),
    },
    'ur10': {
        'visualization': UR10Visualization,
        'xrdf': os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf'),
        'urdf': os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf'),
        'mesh_folder': os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party',
                                    'universal_robots', 'ur10', 'meshes', 'visual'),
    },
}


def ParseArguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot',
                        choices=sorted(ROBOT_CONFIGS),
                        default='franka',
                        help='Robot visualization class to test.')
    return parser.parse_args()


if __name__ == '__main__':
    args = ParseArguments()
    robot_config = ROBOT_CONFIGS[args.robot]

    robot_description = cumotion.load_robot_from_file(robot_config['xrdf'], robot_config['urdf'])

    q0 = robot_description.default_cspace_configuration()
    q1 = np.array(q0, copy=True)
    q1[0] += 0.5
    if len(q1) > 1:
        q1[1] -= 0.5

    visualizer = Visualizer()
    robot_config['visualization'](robot_description,
                                  robot_config['mesh_folder'],
                                  visualizer,
                                  q0,
                                  name=args.robot + "_red_",
                                  color=(1.0, 0.1, 0.1))
    robot_config['visualization'](robot_description,
                                  robot_config['mesh_folder'],
                                  visualizer,
                                  q1,
                                  name=args.robot + "_blue_",
                                  color=(0.1, 0.3, 1.0))
    visualizer.fit_camera_to_scene()

    print("Two robot instances should be visible: one red and one blue.")
    while visualizer.is_active():
        visualizer.update()
        sleep(0.02)

    visualizer.close()

    print("CUMOTION EXAMPLE COMPLETED SUCCESSFULLY")
