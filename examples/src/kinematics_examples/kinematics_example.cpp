// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cassert>
#include <filesystem>
#include <iostream>
#include <vector>

#include "cumotion/kinematics.h"
#include "cumotion/robot_description.h"

#include "utils/cumotion_examples_utils.h"

int main() {
  // Get the paths to the XRDF and URDF files.
  const std::filesystem::path content_dir(CONTENT_DIR);
  std::cout << "content_dir: " << content_dir << "\n";
  const std::filesystem::path xrdf_file = content_dir / "nvidia" / "shared" / "franka.xrdf";
  const std::filesystem::path urdf_file = content_dir / "third_party" / "franka" / "franka.urdf";
  assert(std::filesystem::is_regular_file(xrdf_file));
  assert(std::filesystem::is_regular_file(urdf_file));

  // Load the robot from file.
  std::unique_ptr<cumotion::RobotDescription> robot_description =
      cumotion::LoadRobotFromFile(xrdf_file, urdf_file);
  assert(robot_description != nullptr);

  const int num_coords = robot_description->numCSpaceCoords();
  std::vector<std::string> coord_names(num_coords);
  for (int coord_index = 0; coord_index < num_coords; ++coord_index) {
    coord_names[coord_index] = robot_description->cSpaceCoordName(coord_index);
  }
  for (int coord_index = 0; coord_index < num_coords; ++coord_index) {
    std::cout << "Coordinate " << coord_index << " name: " << coord_names[coord_index] << "\n";
  }

  bool success = num_coords == 7;
  PrintExampleStatus(success);

  return 0;
}
