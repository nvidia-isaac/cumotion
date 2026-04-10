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
#include <string>

// cuMotion public headers.
#include "cumotion/cumotion.h"
#include "cumotion/kinematics.h"
#include "cumotion/robot_description.h"
#include "cumotion/trajectory_optimizer.h"

#include "utils/cumotion_examples_utils.h"

// This example demonstrates a complete workflow for trajectory optimization:
//  1. Load robot description and kinematics.
//  2. Create world environment.
//  3. Configure trajectory optimizer.
//  4. Generate trajectory to target position.
int main() {
  // Set log level (optional, since `WARNING` is already the default).
  cumotion::SetLogLevel(cumotion::LogLevel::WARNING);

  // ===============================================================================================
  // Load Robot Description
  // ===============================================================================================

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

  // Load kinematics.
  auto kinematics = robot_description->kinematics();

  // ===============================================================================================
  // Create cuMotion World
  // ===============================================================================================

  // The `CreateWorld()` function creates an empty world with no obstacles. A `World` represents
  // a collection of obstacles that the robot must avoid during motion planning. This provides
  // the environment context for collision-aware trajectory optimization.
  auto world = cumotion::CreateWorld();

  // Calling `addWorldView()` creates a view into the world that can be used for collision checks
  // and distance evaluations. Each `WorldView` maintains a static snapshot of the world until
  // it is updated (via a call to `WorldViewHandle::update()`), enabling efficient queries during
  // trajectory optimization.
  auto world_view = world->addWorldView();

  // ===============================================================================================
  // Create Trajectory Optimizer
  // ===============================================================================================

  // Set end effector frame for Franka.
  // NOTE: The Franka XRDF includes the optional "tool_frames" field.
  assert(!robot_description->toolFrameNames().empty());
  const std::string tool_frame_name = robot_description->toolFrameNames().front();

  // Create a default trajectory optimizer config for this example. This function combines default
  // optimization parameters with the robot description, tool frame, and world view to create a
  // complete configuration for trajectory optimization. The default parameters include settings
  // for collision checking, optimization convergence, and trajectory resolution.
  auto trajectory_optimizer_config = cumotion::CreateDefaultTrajectoryOptimizerConfig(
      *robot_description,
      tool_frame_name,
      world_view);

  // Create the trajectory optimizer using the configuration. This instantiates the numerical
  // optimization engine that will generate collision-free trajectories using a combination
  // of particle-based optimization (PBO) and L-BFGS methods.
  auto trajectory_optimizer = cumotion::CreateTrajectoryOptimizer(*trajectory_optimizer_config);

  // ===============================================================================================
  // Define initial c-space position
  // ===============================================================================================

  // Use the robot's default configuration as the initial c-space (i.e., "configuration space")
  // position. This retrieves the default c-space position defined in the robot
  // description, which specifies the joint angles for all actively controlled joints.
  const auto q_initial = robot_description->defaultCSpaceConfiguration();

  // ===============================================================================================
  // Create task-space constraints
  // ===============================================================================================

  // Set the target position in world coordinates (i.e., the base frame of the robot). This defines
  // the desired position for the origin of the tool frame. The coordinates are in meters.
  const Eigen::Vector3d target_position(0.3, 0.1, 0.1);

  // Create a translation constraint that requires the tool frame origin to reach the specified
  // target position at the end of the trajectory. The constraint is active at termination but not
  // along the path.
  const auto translation_constraint =
      cumotion::TrajectoryOptimizer::TranslationConstraint::Target(target_position);

  // Create an orientation constraint that does not restrict the orientation of the tool frame along
  // the path or at termination.
  const auto orientation_constraint = cumotion::TrajectoryOptimizer::OrientationConstraint::None();

  // Create a task-space target combining translation and orientation constraints.
  const auto task_space_target = cumotion::TrajectoryOptimizer::TaskSpaceTarget(
      translation_constraint,
      orientation_constraint);

  // ===============================================================================================
  // Plan Trajectory
  // ===============================================================================================

  // Plan trajectory from the initial c-space position to the task-space target. This is implemented
  // with GPU-accelerated numerical optimization algorithms.
  const auto results = trajectory_optimizer->planToTaskSpaceTarget(q_initial, task_space_target);

  // ===============================================================================================
  // Check Results and Visualize
  // ===============================================================================================

  // Check if trajectory optimization was successful. The status indicates whether a valid
  // collision-free trajectory was found that satisfies all constraints and limits.
  bool success = false;
  if (results->status() == cumotion::TrajectoryOptimizer::Results::Status::SUCCESS) {
    // Extract the optimized trajectory. This is a time-parameterized path through c-space.
    const auto trajectory = results->trajectory();

    // Get the time domain of the trajectory. The domain specifies the valid time range
    // [lower, upper] over which the trajectory is defined and can be evaluated. The lower bound of
    // the domain will always be set to zero; thus, the upper bound of the domain represents the
    // time span of the trajectory.
    const auto domain = trajectory->domain();

    std::cout << "Trajectory generation successful" << std::endl;
    std::cout << "Trajectory duration: " << domain.span() << " seconds" << std::endl;

    // Evaluate trajectory at start and end times to show c-space positions. The `eval()`
    // function computes the c-space position at any time within the domain.
    std::cout << "Trajectory start position: " << trajectory->eval(domain.lower).transpose()
              << std::endl;
    std::cout << "Trajectory end position: " << trajectory->eval(domain.upper).transpose()
              << std::endl;

    // Check final position error by computing forward kinematics. This verifies that
    // the final c-space position actually places the tool frame at the target.
    const auto q_final = trajectory->eval(domain.upper);
    const auto tool_frame = kinematics->frame(tool_frame_name);
    const auto final_position = kinematics->position(q_final, tool_frame);
    const auto position_error = (final_position - target_position).norm();
    std::cout << "Final task-space position error: " << position_error * 1000.0 << " mm"
              << std::endl;
    success = position_error < 1e-3;
  } else {
    // Report trajectory optimization failure.
    std::cout << "Failed to find trajectory." << std::endl;
  }

  PrintExampleStatus(success);

  return 0;
}
