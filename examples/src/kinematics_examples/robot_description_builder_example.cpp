// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

//! @file
//! @brief Demonstrate how to build a robot with the public `RobotDescriptionBuilder` API.

#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "cumotion/kinematics.h"
#include "cumotion/pose3.h"
#include "cumotion/robot_description.h"
#include "cumotion/robot_description_builder.h"
#include "cumotion/robot_world_inspector.h"

#include "utils/cumotion_examples_utils.h"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kPositionTolerance = 1e-9;
constexpr cumotion::Kinematics::Limits kJointLimits{-kPi, kPi};

// Return `true` if the c-space coordinates of `robot` correspond to `expected_cspace_coord_names`.
//
// The names must be in the same order to be considered equivalent.
bool HasExpectedCSpaceCoords(const cumotion::RobotDescription &robot,
                             const std::vector<std::string> &expected_cspace_coord_names) {
  if (robot.numCSpaceCoords() != static_cast<int>(expected_cspace_coord_names.size())) {
    return false;
  }
  for (int i = 0; i < robot.numCSpaceCoords(); ++i) {
    if (robot.cSpaceCoordName(i) != expected_cspace_coord_names[static_cast<size_t>(i)]) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  // Build a three-link planar arm. Each revolute joint rotates around the z-axis, and each
  // unit-length link extends along its local x-axis.
  const auto builder = cumotion::CreateRobotDescriptionBuilder("base");
  const cumotion::Pose3 link_offset =
      cumotion::Pose3::FromTranslation(Eigen::Vector3d::UnitX());
  constexpr double kVelocityLimit = 2.0;
  constexpr double kAccelerationLimit = 4.0;
  constexpr double kJerkLimit = 8.0;

  // Procedural construction is expected to succeed as all frames, joints, and c-space coordinates
  // are added.
  //
  // The generally recommended flow of:
  //   1. adding all frames,
  //   2. adding all joints, and
  //   3. adding all c-space coordinates
  // is used.
  //
  // This ordering is not required, but each component must be added prior to being referenced.
  // For example, adding "joint_1" requires that the frames "base" and "link_1" have been added.
  // Following the frames -> joint -> c-space ordering is a simple way to ensure these dependencies
  // are met.
  //
  // A c-space coordinate is intentionally omitted for "joint_3" at this time. This is to first show
  // the expected behavior of joints not controlled by a c-space coordinate.
  constexpr bool kIsToolFrame = true;
  if (builder == nullptr ||
      !builder->addFrame("link_1") ||
      !builder->addFrame("link_2") || !builder->addFrame("link_3") ||
      !builder->addFrame("end_effector", kIsToolFrame) ||
      !builder->addRevoluteJoint("joint_1", "base", "link_1", cumotion::Pose3::Identity(),
                                 Eigen::Vector3d::UnitZ(), kJointLimits, kVelocityLimit) ||
      !builder->addRevoluteJoint("joint_2", "link_1", "link_2", link_offset,
                                 Eigen::Vector3d::UnitZ(), kJointLimits, kVelocityLimit) ||
      !builder->addRevoluteJoint("joint_3", "link_2", "link_3", link_offset,
                                 Eigen::Vector3d::UnitZ(), kJointLimits, kVelocityLimit) ||
      !builder->addFixedJoint("end_effector_joint", "link_3", "end_effector", link_offset) ||
      !builder->addCSpaceCoord("joint_1", {"joint_1"}, kAccelerationLimit, kJerkLimit) ||
      !builder->addCSpaceCoord("joint_2", {"joint_2"}, kAccelerationLimit, kJerkLimit)) {
    PrintExampleStatus(false);
    return 1;
  }

  // First build the arm with only two c-space coordinates. `joint_3` has no c-space control, so
  // the resulting robot holds it at zero.
  //
  // Note: the default joint position for an uncontrolled joint will be "zero clipped to the
  // position limits"; in this case no clipping is needed. If a different default is preferred *and*
  // it is desired that the joint not be controlled by standard c-space, then an auxiliary c-space
  // coordinate can be used (via `addAuxCSpaceCoord()`).
  const auto two_coord_robot = builder->build();
  const std::vector<std::string> expected_two_coord_names = {"joint_1", "joint_2"};
  if (two_coord_robot == nullptr ||
      !HasExpectedCSpaceCoords(*two_coord_robot, expected_two_coord_names) ||
      two_coord_robot->toolFrameNames() != std::vector<std::string>{"end_effector"}) {
    PrintExampleStatus(false);
    return 1;
  }

  // Extract a kinematic structure for the `two_coord_robot`, expecting the number of c-space
  // coordinates to be 2.
  const auto two_coord_kinematics = two_coord_robot->kinematics();
  if (two_coord_kinematics == nullptr || two_coord_kinematics->numCSpaceCoords() != 2) {
    PrintExampleStatus(false);
    return 1;
  }
  const auto two_coord_end_effector = two_coord_kinematics->frame("end_effector");
  const Eigen::Vector3d two_coord_position = two_coord_kinematics->position(
      0.5 * kPi * Eigen::VectorXd::Ones(2), two_coord_end_effector);
  const bool two_coord_success =
      two_coord_kinematics->frameName(two_coord_end_effector) == "end_effector" &&
      two_coord_position.isApprox(Eigen::Vector3d(-2.0, 1.0, 0.0), kPositionTolerance);
  std::cout << "Two-coordinate end-effector position (`joint_3` at zero): "
            << two_coord_position.transpose() << "\n";

  // Add control of `joint_3` and build an independent three-coordinate robot description.
  if (!builder->addCSpaceCoord("joint_3", {"joint_3"}, kAccelerationLimit, kJerkLimit)) {
    PrintExampleStatus(false);
    return 1;
  }
  const auto three_coord_robot = builder->build();
  const std::vector<std::string> expected_three_coord_names = {"joint_1", "joint_2", "joint_3"};
  if (three_coord_robot == nullptr ||
      !HasExpectedCSpaceCoords(*three_coord_robot, expected_three_coord_names) ||
      two_coord_robot->numCSpaceCoords() != 2) {
    PrintExampleStatus(false);
    return 1;
  }

  // Extract a kinematic structure for the `three_coord_robot`, expecting the number of c-space
  // coordinates to be 3.
  const auto three_coord_kinematics = three_coord_robot->kinematics();
  if (three_coord_kinematics == nullptr || three_coord_kinematics->numCSpaceCoords() != 3) {
    PrintExampleStatus(false);
    return 1;
  }
  const auto three_coord_end_effector = three_coord_kinematics->frame("end_effector");
  const Eigen::VectorXd three_coord_configuration = 0.5 * kPi * Eigen::VectorXd::Ones(3);
  const Eigen::Vector3d three_coord_position =
      three_coord_kinematics->position(three_coord_configuration, three_coord_end_effector);
  const bool three_coord_success =
      three_coord_kinematics->frameName(three_coord_end_effector) == "end_effector" &&
      three_coord_position.isApprox(Eigen::Vector3d(-1.0, 0.0, 0.0), kPositionTolerance);
  std::cout << "Three-coordinate end-effector position (`joint_3` at pi / 2): "
            << three_coord_position.transpose() << "\n";

  // Attach a sphere to the frame "link_3". The sphere center is expressed in the local coordinates
  // of that frame. Populating both optional buffer distances enables the sphere for world- and
  // self-collision.
  constexpr double kCollisionSphereRadius = 0.1;
  constexpr double kCollisionBufferDistance = 0.0;
  const cumotion::RobotDescriptionBuilder::CollisionSphere collision_sphere{
      0.5 * Eigen::Vector3d::UnitX(),
      kCollisionSphereRadius,
      kCollisionBufferDistance,
      kCollisionBufferDistance};
  if (!builder->addCollisionSpheres("link_3", {collision_sphere})) {
    PrintExampleStatus(false);
    return 1;
  }

  // Build a new robot containing the collision geometry. Previously built robot descriptions
  // remain unchanged.
  const auto collision_robot = builder->build();
  if (collision_robot == nullptr) {
    PrintExampleStatus(false);
    return 1;
  }
  const auto collision_inspector =
      cumotion::CreateRobotWorldInspector(*collision_robot, std::nullopt);
  if (collision_inspector == nullptr) {
    PrintExampleStatus(false);
    return 1;
  }
  const bool collision_sphere_success =
      collision_inspector->numWorldCollisionSpheres() == 1 &&
      collision_inspector->numSelfCollisionSpheres() == 1;
  std::cout << "Collision geometry: " << collision_inspector->numWorldCollisionSpheres()
            << " world-collision sphere and "
            << collision_inspector->numSelfCollisionSpheres() << " self-collision sphere\n";

  const bool success = two_coord_success && three_coord_success && collision_sphere_success;
  PrintExampleStatus(success);
  return success ? 0 : 1;
}
