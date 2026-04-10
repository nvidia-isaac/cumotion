// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//                         All rights reserved.
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
#include <memory>
#include <optional>
#include <string>
#include <vector>

// cuMotion public headers.
#include "cumotion/cumotion.h"
#include "cumotion/collision_free_ik_solver.h"
#include "cumotion/robot_description.h"

#include "utils/cumotion_examples_utils.h"

int main() {
  // Set log level (optional, since `WARNING` is already the default).
  cumotion::SetLogLevel(cumotion::LogLevel::WARNING);

  // Set content directory (for loading URDF and XRDF files).
  const std::filesystem::path content_dir(CONTENT_DIR);

  // Set absolute path to URDF file for Franka.
  const std::filesystem::path urdf_path = content_dir / "third_party" / "franka" / "franka.urdf";

  // Set absolute path to the XRDF for Franka.
  //
  // XRDF extends the URDF with additional information such as semantic labeling of configuration
  // space, acceleration limits, jerk limits, and collision spheres. For additional details,
  // see: https://nvidia-isaac-ros.github.io/concepts/manipulation/xrdf.html
  const std::filesystem::path xrdf_path = content_dir / "nvidia" / "shared" / "franka.xrdf";

  // Load robot description.
  std::unique_ptr<cumotion::RobotDescription> robot_description =
      cumotion::LoadRobotFromFile(xrdf_path, urdf_path);

  // Set end effector frame for Franka.
  // NOTE: The Franka XRDF includes the optional "tool_frames" field.
  assert(!robot_description->toolFrameNames().empty());
  const std::string end_effector_frame_name = robot_description->toolFrameNames().front();

  // The `CreateWorld()` function creates an empty world that will be populated with obstacles.
  // A `World` represents a collection of obstacles that the robot must avoid.
  auto world = cumotion::CreateWorld();

  // Calling `addWorldView()` creates a view into the world that can be used for collision checks
  // and distance evaluations. Each `WorldView` maintains a static snapshot of the world until
  // it is updated (via a call to `WorldViewHandle::update()`).
  auto world_view = world->addWorldView();

  // Create configuration for collision-free inverse kinematics (IK).
  std::unique_ptr<cumotion::CollisionFreeIkSolverConfig> config =
      cumotion::CreateDefaultCollisionFreeIkSolverConfig(*robot_description,
                                                         end_effector_frame_name,
                                                         world_view);

  // Create a collision-free IK solver.
  std::unique_ptr<cumotion::CollisionFreeIkSolver> ik_solver =
      cumotion::CreateCollisionFreeIkSolver(*config);

  // Define a target pose for the end effector.
  const Eigen::Vector3d target_translation(0.5, 0.2, 0.6);
  const auto target_orientation = cumotion::Rotation3::FromAxisAngle(Eigen::Vector3d::UnitY(),
                                                                     0.5 * M_PI);

  // Use the target pose to created task-space constraints for the IK solver.
  const auto translation_constraint =
      cumotion::CollisionFreeIkSolver::TranslationConstraint::Target(target_translation);
  const auto orientation_constraint =
      cumotion::CollisionFreeIkSolver::OrientationConstraint::Target(target_orientation);
  const auto task_space_target =
      cumotion::CollisionFreeIkSolver::TaskSpaceTarget(translation_constraint,
                                                       orientation_constraint);

  // Solve IK, expecting multiple distinct c-space solutions.
  auto results = ik_solver->solve(task_space_target);
  bool success = results->status() == cumotion::CollisionFreeIkSolver::Results::Status::SUCCESS;
  int num_solutions = results->cSpacePositions().size();
  std::cout << "Found " << num_solutions << " distinct IK solutions." << std::endl;
  success = success && (num_solutions >= 8);

  // Add a cuboid obstacle to the `world`.
  Eigen::Vector3d cuboid_side_lengths(0.5, 0.35, 0.8);
  Eigen::Vector3d cuboid_position(0.4, -0.2, 0.4);
  const auto cuboid = cumotion::CreateObstacle(cumotion::Obstacle::Type::CUBOID);
  cuboid->setAttribute(cumotion::Obstacle::Attribute::SIDE_LENGTHS, cuboid_side_lengths);
  world->addObstacle(*cuboid, cumotion::Pose3::FromTranslation(cuboid_position));

  // Updating `world_view` updates the snapshot of the `world` used by `ik_solver`. With this
  // update, `ik_solver` will actively try to avoid the newly added cuboid obstacle.
  world_view.update();

  // Solve IK (with cuboid added), expecting multiple distinct c-space solutions.
  auto results_with_cuboid = ik_solver->solve(task_space_target);
  success = success && results_with_cuboid->status() ==
         cumotion::CollisionFreeIkSolver::Results::Status::SUCCESS;
  num_solutions = results_with_cuboid->cSpacePositions().size();
  std::cout << "Found " << num_solutions
            << " distinct IK solutions that avoid the cuboid obstacle." << std::endl;
  success = success && (num_solutions >= 8);

  PrintExampleStatus(success);

  return 0;
}
