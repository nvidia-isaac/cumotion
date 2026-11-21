// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//                         All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

//! @file
//! @brief Public interface for building a robot description

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "cumotion/cumotion_export.h"
#include "cumotion/kinematics.h"
#include "cumotion/pose3.h"
#include "cumotion/robot_description.h"

namespace cumotion {

//! Interface to build a `RobotDescription` programmatically.
//!
//! This interface provides an alternative to loading a `RobotDescription` from URDF/XRDF files
//! (see `LoadRobotFromFile()` in `robot_description.h`).
//!
//! A builder is initialized with a root frame by `CreateRobotDescriptionBuilder()`. The robot is
//! then specified by adding frames, the joints that connect them, the c-space coordinates that
//! control those joints, and any collision geometry. A `RobotDescription` is produced by `build()`.
//!
//! Each `add*()` method validates its input and reports an invalid specification by logging a
//! nonfatal error and returning `false`, leaving the state of the builder unchanged. This means a
//! failed addition can be corrected and retried, and that a `RobotDescription` is only ever built
//! from additions that have each been validated.
//!
//! Frames, joints and c-space coordinates occupy independent namespaces, so a joint and the
//! c-space coordinate that controls it may share a name. Within a namespace, each name must be
//! unique: a frame name is unique among frames, a joint name is unique among joints of *ALL* types,
//! and a c-space coord name is unique across standard *AND* auxiliary c-space coordinates. Every
//! name must be non-empty.
//!
//! A revolute or prismatic joint need not be controlled by a c-space coordinate (standard or
//! auxiliary). Such a joint is not driven by the resulting `RobotDescription`, and is held at a
//! position of zero clipped to its position limits. Controlling a joint by a c-space coordinate
//! enables overriding its default position.
class CUMO_EXPORT RobotDescriptionBuilder {
 public:
  virtual ~RobotDescriptionBuilder() = default;

  //! Representation of a sphere for use as collision geometry.
  //!
  //! A collision sphere may be enabled for world-collision, self-collision, or both, and must be
  //! enabled for at least one of the two.
  struct CUMO_EXPORT CollisionSphere {
    //! Center position of the sphere.
    //!
    //! This position is expressed in the local coordinates of the frame to which the sphere is
    //! attached.
    Eigen::Vector3d center;

    //! Radius of the sphere.
    //!
    //! This is the "base" radius, shared by the world- and self-collision spheres prior to adding
    //! the respective buffer distances below.
    //!
    //! The radius must be positive.
    double radius;

    //! If populated, enable world-collision with an effective sphere radius of
    //! `radius + world_collision_buffer_distance.value()`.
    //!
    //! The buffer distance may be negative, but the effective radius must remain positive.
    std::optional<double> world_collision_buffer_distance;

    //! If populated, enable self-collision with an effective sphere radius of
    //! `radius + self_collision_buffer_distance.value()`.
    //!
    //! The buffer distance may be negative, but the effective radius must remain positive.
    std::optional<double> self_collision_buffer_distance;
  };

  //! Map from c-space position to joint position.
  //!
  //! Revolute and prismatic joints (optionally) support a linear map of the form:
  //!   `joint_position = cspace_multiplier * cspace_position + cspace_offset`
  //!
  //! A c-space position is dimensionless, and this map converts it to a joint position in meters
  //! (for a prismatic joint) or radians (for a revolute joint).
  struct CUMO_EXPORT JointMap {
    //! Scale factor from c-space position to joint position.
    //!
    //! A `cspace_multiplier` that is (nearly) zero is invalid, as it would decouple the joint from
    //! the c-space coordinate that controls it.
    double cspace_multiplier;

    //! Joint position corresponding to a c-space position of zero.
    double cspace_offset;
  };

  //! Add a frame named `name` to the robot description.
  //!
  //! A valid frame specification will add the frame and return `true`. An invalid frame
  //! specification will log a nonfatal error, make no changes to the state of this builder, and
  //! return `false`.
  //!
  //! Frame specification is invalid if:
  //!   1. The `name` is a duplicate of an existing frame name (including the `root_frame_name`
  //!      used to initialize the `RobotDescriptionBuilder`), *OR*
  //!   2. The `name` is an empty string.
  //!
  //! A frame added with `is_tool_frame` set is reported by `RobotDescription::toolFrameNames()`,
  //! identifying it as a frame of interest for tasks such as grasping. A frame is otherwise
  //! unaffected by `is_tool_frame`.
  //!
  //! Tool frames are reported in the order in which they were added, and their interpretation is
  //! otherwise application-defined.
  [[nodiscard]] virtual bool addFrame(const std::string &name, bool is_tool_frame = false) = 0;

  //! Add a fixed joint named `name` to the robot description.
  //!
  //! A valid joint specification will add the joint and return `true`. An invalid joint
  //! specification will log a nonfatal error, make no changes to the state of this builder, and
  //! return `false`.
  //!
  //! Each joint `name` must be non-empty and unique (as compared to existing joint names of *ALL*
  //! joint types). Failure to provide a non-empty, unique `name` constitutes an invalid joint
  //! specification.
  //!
  //! The joint connects frames with names `parent_frame_name` and `child_frame_name`. Any of the
  //! below conditions constitute an invalid joint specification:
  //!   1. Frames with these names have *NOT* already been added to the robot description. The
  //!      joint must connect existing frames.
  //!   2. The frame corresponding to `child_frame_name` has been previously added as the child
  //!      frame of another joint. This is because each frame can only be directly controlled by
  //!      a single joint.
  //!      NOTE: Multiple joints can be descended from a given frame, so the frame corresponding
  //!      to `parent_frame_name` *can* have been previously added as the parent frame of another
  //!      joint.
  //!   3. The `child_frame_name` corresponds to the root frame. The root frame (by definition)
  //!      is not "driven" by any joint.
  //!   4. The `parent_frame_name` and `child_frame_name` are the same.
  //!
  //! NOTE: The above constraints ensure that the kinematic structure being built is a tree.
  //!
  //! The `fixed_transform` represents a fixed mapping from the parent frame to the child frame.
  [[nodiscard]] virtual bool addFixedJoint(const std::string &name,
                                           const std::string &parent_frame_name,
                                           const std::string &child_frame_name,
                                           const Pose3 &fixed_transform) = 0;

  //! Add a prismatic joint named `name` to the robot description.
  //!
  //! A valid joint specification will add the joint and return `true`. An invalid joint
  //! specification will log a nonfatal error, make no changes to the state of this builder, and
  //! return `false`.
  //!
  //! Each joint `name` must be non-empty and unique (as compared to existing joint names of *ALL*
  //! joint types). Failure to provide a non-empty, unique `name` constitutes an invalid joint
  //! specification.
  //!
  //! The joint connects frames with names `parent_frame_name` and `child_frame_name`. Any of the
  //! below conditions constitute an invalid joint specification:
  //!   1. Frames with these names have *NOT* already been added to the robot description. The
  //!      joint must connect existing frames.
  //!   2. The frame corresponding to `child_frame_name` has been previously added as the child
  //!      frame of another joint. This is because each frame can only be directly controlled by
  //!      a single joint.
  //!      NOTE: Multiple joints can be descended from a given frame, so the frame corresponding
  //!      to `parent_frame_name` *can* have been previously added as the parent frame of another
  //!      joint.
  //!   3. The `child_frame_name` corresponds to the root frame. The root frame (by definition)
  //!      is not "driven" by any joint.
  //!   4. The `parent_frame_name` and `child_frame_name` are the same.
  //!
  //! NOTE: The above constraints ensure that the kinematic structure being built is a tree.
  //!
  //! The `fixed_transform` represents a fixed mapping from the parent frame to the joint frame.
  //! This joint frame is an intermediate frame between the parent frame and child frame that has no
  //! specified name and cannot be queried in the resulting kinematic structure.
  //!
  //! The `joint_axis` is specified in the joint frame, where the joint frame is defined as the
  //! parent frame transformed by the preceding `fixed_transform`. The `joint_axis` need not be unit
  //! length, as it is normalized internally, but it is an invalid joint specification if the
  //! `joint_axis` is (nearly) zero.
  //!
  //! The (optional) `position_limits` represent the range of valid motion for the joint
  //! (in meters). It is an invalid joint specification if the upper limit (i.e.,
  //! `position_limits.upper`) is not greater than the lower limit (i.e., `position_limits.lower`).
  //! If no `position_limits` are specified, the motion of the joint is unconstrained.
  //!
  //! The (optional) `velocity_limit` represents the maximum valid velocity magnitude for the
  //! joint (in meters/second). It is an invalid joint specification if the `velocity_limit` is not
  //! positive. If no `velocity_limit` is specified, the velocity of the joint is unlimited.
  //!
  //! The (optional) `joint_map` provides a linear map of the form:
  //!   `joint_position = cspace_multiplier * cspace_position + cspace_offset`
  //! It is an invalid joint specification if the `cspace_multiplier` is (nearly) zero. If no
  //! `joint_map` is provided, the joint position is the c-space position (i.e., a
  //! `cspace_multiplier` of `1` and a `cspace_offset` of `0`).
  [[nodiscard]] virtual bool addPrismaticJoint(
      const std::string &name,
      const std::string &parent_frame_name,
      const std::string &child_frame_name,
      const Pose3 &fixed_transform,
      const Eigen::Vector3d &joint_axis,
      const std::optional<Kinematics::Limits> &position_limits,
      std::optional<double> velocity_limit,
      std::optional<JointMap> joint_map = std::nullopt) = 0;

  //! Add a revolute joint named `name` to the robot description.
  //!
  //! A valid joint specification will add the joint and return `true`. An invalid joint
  //! specification will log a nonfatal error, make no changes to the state of this builder, and
  //! return `false`.
  //!
  //! Each joint `name` must be non-empty and unique (as compared to existing joint names of *ALL*
  //! joint types). Failure to provide a non-empty, unique `name` constitutes an invalid joint
  //! specification.
  //!
  //! The joint connects frames with names `parent_frame_name` and `child_frame_name`. Any of the
  //! below conditions constitute an invalid joint specification:
  //!   1. Frames with these names have *NOT* already been added to the robot description. The
  //!      joint must connect existing frames.
  //!   2. The frame corresponding to `child_frame_name` has been previously added as the child
  //!      frame of another joint. This is because each frame can only be directly controlled by
  //!      a single joint.
  //!      NOTE: Multiple joints can be descended from a given frame, so the frame corresponding
  //!      to `parent_frame_name` *can* have been previously added as the parent frame of another
  //!      joint.
  //!   3. The `child_frame_name` corresponds to the root frame. The root frame (by definition)
  //!      is not "driven" by any joint.
  //!   4. The `parent_frame_name` and `child_frame_name` are the same.
  //!
  //! NOTE: The above constraints ensure that the kinematic structure being built is a tree.
  //!
  //! The `fixed_transform` represents a fixed mapping from the parent frame to the joint frame.
  //! This joint frame is an intermediate frame between the parent frame and child frame that has no
  //! specified name and cannot be queried in the resulting kinematic structure.
  //!
  //! The `joint_axis` is specified in the joint frame, where the joint frame is defined as the
  //! parent frame transformed by the preceding `fixed_transform`. The `joint_axis` need not be unit
  //! length, as it is normalized internally, but it is an invalid joint specification if the
  //! `joint_axis` is (nearly) zero.
  //!
  //! The (optional) `position_limits` represent the range of valid motion for the joint
  //! (in radians). It is an invalid joint specification if the upper limit (i.e.,
  //! `position_limits.upper`) is not greater than the lower limit (i.e., `position_limits.lower`).
  //! If no `position_limits` are specified, the joint rotates continuously.
  //!
  //! The (optional) `velocity_limit` represents the maximum valid velocity magnitude for the
  //! joint (in radians/second). It is an invalid joint specification if the `velocity_limit` is not
  //! positive. If no `velocity_limit` is specified, the velocity of the joint is unlimited.
  //!
  //! The (optional) `joint_map` provides a linear map of the form:
  //!   `joint_position = cspace_multiplier * cspace_position + cspace_offset`
  //! It is an invalid joint specification if the `cspace_multiplier` is (nearly) zero. If no
  //! `joint_map` is provided, the joint position is the c-space position (i.e., a
  //! `cspace_multiplier` of `1` and a `cspace_offset` of `0`).
  [[nodiscard]] virtual bool addRevoluteJoint(
      const std::string &name,
      const std::string &parent_frame_name,
      const std::string &child_frame_name,
      const Pose3 &fixed_transform,
      const Eigen::Vector3d &joint_axis,
      const std::optional<Kinematics::Limits> &position_limits,
      std::optional<double> velocity_limit,
      std::optional<JointMap> joint_map = std::nullopt) = 0;

  //! Add a c-space coordinate named `name` to the robot description.
  //!
  //! These "standard" c-space coordinates are used for motion generation and generally referred to
  //! simply as "c-space coordinates". The full configuration space of the robot also includes
  //! "auxiliary" c-space coordinates. See `addAuxCSpaceCoord()` below for details.
  //!
  //! A valid c-space coord specification will add the c-space coord and return `true`. An invalid
  //! c-space coord specification will log a nonfatal error, make no changes to the state of this
  //! builder, and return `false`.
  //!
  //! Each c-space coordinate will be assigned an index according to the order in which it is
  //! added to the robot description (i.e., the first c-space coordinate added will have index `0`,
  //! the second will have index `1`, etc.). This index corresponds to the name order reported by
  //! `RobotDescription::cSpaceCoordName()`.
  //!
  //! Each c-space coord `name` must be non-empty and unique (as compared to standard *AND*
  //! auxiliary c-space coordinate names). Failure to provide a non-empty, unique `name`
  //! constitutes an invalid c-space coord specification.
  //!
  //! Each c-space coord will control one or more joints indicated by `joint_names`. Any of the
  //! below conditions constitute an invalid c-space coord specification:
  //!   1. A joint name is included in `joint_names` for a joint that has not yet been added to
  //!      the robot description.
  //!   2. A joint name is included more than once in `joint_names`.
  //!   3. A joint is included in `joint_names` that is already controlled by another c-space
  //!      coordinate *OR* auxiliary c-space coordinate.
  //!   4. One of the names in `joint_names` corresponds to a fixed joint (i.e., a joint that, by
  //!      definition, cannot be controlled).
  //!   5. No joint names are provided in `joint_names`. Each c-space coord must control at least
  //!      one joint.
  //!   6. No joint in `joint_names` has position limits. At least one controlled joint must have
  //!      position limits specified.
  //!   7. No joint in `joint_names` has a velocity limit. At least one controlled joint must have
  //!      a velocity limit specified.
  //!   8. The position limits of the joints in `joint_names` do not intersect to form a valid
  //!      (non-zero) range.
  //!
  //! A c-space position is dimensionless, and is converted to the position of each controlled joint
  //! by the `joint_map` of that joint. A single c-space coord may therefore control a mix of
  //! prismatic and revolute joints.
  //!
  //! An `acceleration_limit` and a `jerk_limit` are required for every c-space coord, and are
  //! expressed in c-space units per second squared and per second cubed, respectively. It is an
  //! invalid c-space coord specification if either limit is not positive.
  //!
  //! The position limits of a c-space coord are the intersection of the position limits of the
  //! joints it controls, expressed in c-space (i.e., accounting for the `joint_map` of each
  //! controlled joint).
  //!
  //! The (optional) `default_position` is a c-space position reported by
  //! `RobotDescription::defaultCSpaceConfiguration()`. If no `default_position` is specified, the
  //! default position is zero clipped to the position limits of the c-space coord. It is an
  //! invalid c-space coord specification if a `default_position` violates those position limits.
  [[nodiscard]] virtual bool addCSpaceCoord(
      const std::string &name,
      const std::vector<std::string> &joint_names,
      double acceleration_limit,
      double jerk_limit,
      std::optional<double> default_position = std::nullopt) = 0;

  //! Add an auxiliary c-space coordinate named `name` to the robot description.
  //!
  //! An auxiliary c-space coordinate controls joints that are excluded from the c-space used for
  //! motion generation (e.g., a gripper may be controlled by an auxiliary c-space coordinate).
  //! Such joints are held at a fixed position.
  //!
  //! Unlike a standard c-space coordinate (i.e., a c-space coordinate added via
  //! `addCSpaceCoord()`), an auxiliary c-space coord requires no velocity, acceleration, or jerk
  //! limits. Any velocity limits set for the joints it controls are ignored.
  //!
  //! A valid auxiliary c-space coord specification will add the auxiliary c-space coord and return
  //! `true`. An invalid auxiliary c-space coord specification will log a nonfatal error, make no
  //! changes to the state of this builder, and return `false`.
  //!
  //! Each auxiliary c-space coordinate will be assigned an index according to the order in which
  //! it is added to the robot description (i.e., the first auxiliary c-space coordinate added will
  //! have index `0`, the second will have index `1`, etc.).
  //!
  //! Each auxiliary c-space coord `name` must be non-empty and unique (as compared to standard
  //! *AND* auxiliary c-space coordinate names). Failure to provide a non-empty, unique `name`
  //! constitutes an invalid auxiliary c-space coord specification.
  //!
  //! Each auxiliary c-space coord will control one or more joints indicated by `joint_names`.
  //! Any of the below conditions constitute an invalid auxiliary c-space coord specification:
  //!   1. A joint name is included in `joint_names` for a joint that has not yet been added to
  //!      the robot description.
  //!   2. A joint name is included more than once in `joint_names`.
  //!   3. A joint is included in `joint_names` that is already controlled by another c-space
  //!      coordinate *OR* auxiliary c-space coordinate.
  //!   4. One of the names in `joint_names` corresponds to a fixed joint (i.e., a joint that, by
  //!      definition, cannot be controlled).
  //!   5. No joint names are provided in `joint_names`. Each auxiliary c-space coord must control
  //!      at least one joint.
  //!   6. The position limits of the joints in `joint_names` do not overlap. Joints with no
  //!      position limits are permitted, but any limits that are present must intersect to form a
  //!      valid (non-zero) range.
  //!
  //! The position limits of an auxiliary c-space coord are the intersection of the position limits
  //! of the joints it controls that have them, expressed in c-space (i.e., accounting for the
  //! `joint_map` of each controlled joint). An auxiliary c-space coord whose controlled joints all
  //! lack position limits is unconstrained.
  //!
  //! The (optional) `default_position` is the c-space position at which the controlled joints are
  //! held. If no `default_position` is specified, the default position is zero clipped to the
  //! position limits of the auxiliary c-space coord. It is an invalid auxiliary c-space coord
  //! specification if a `default_position` violates those position limits.
  [[nodiscard]] virtual bool addAuxCSpaceCoord(
      const std::string &name,
      const std::vector<std::string> &joint_names,
      std::optional<double> default_position = std::nullopt) = 0;

  //! Attach `collision_spheres` to the frame named `frame_name`.
  //!
  //! The `collision_spheres` are specified with centers in the local coordinates of `frame_name`,
  //! and are appended to any spheres previously attached to that frame.
  //!
  //! A valid collision sphere specification will attach the spheres and return `true`. An invalid
  //! collision sphere specification will log a nonfatal error, make no changes to the state of this
  //! builder, and return `false`. Any of the below conditions constitute an invalid collision
  //! sphere specification:
  //!   1. The `frame_name` has *NOT* already been added to the robot description,
  //!   2. A sphere in `collision_spheres` has a non-positive `radius`,
  //!   3. A sphere in `collision_spheres` is enabled for neither world-collision nor
  //!      self-collision, *OR*
  //!   4. A sphere in `collision_spheres` has a non-positive effective world- or self-collision
  //!      radius (i.e., a buffer distance that is not greater than the negated `radius`).
  //!
  //! NOTE: The spheres are attached only if *every* sphere in `collision_spheres` is valid.
  //!
  //! An empty `collision_spheres` makes no change to this builder and returns `true`.
  [[nodiscard]] virtual bool addCollisionSpheres(
      const std::string &frame_name,
      const std::vector<CollisionSphere> &collision_spheres) = 0;

  //! Mask self-collision between the frame named `frame_name` and each frame in
  //! `masked_frame_names`.
  //!
  //! Intersection between the self-collision spheres of a masked pair of frames does *NOT*
  //! constitute a self-collision. This is typically used for pairs of frames that are:
  //!   - always in contact (e.g., adjacent links), or
  //!   - never in contact (i.e., frames for which collision is impossible).
  //!
  //! NOTE: Masking is symmetric, so a pair only needs to be masked in one direction.
  //!
  //! Masked frame names are appended to any previously specified for `frame_name`, and duplicates
  //! are ignored. The `frame_name` itself is also ignored if it is included in
  //! `masked_frame_names`, since self-collision is never evaluated between the spheres of a single
  //! frame.
  //!
  //! It is an invalid self-collision mask specification if `frame_name`, or any name in
  //! `masked_frame_names`, has not already been added to the robot description, in which case a
  //! nonfatal error is logged, no changes are made to the state of this builder, and `false` is
  //! returned.
  //!
  //! A `masked_frame_names` that names no frame other than `frame_name`, including an empty
  //! `masked_frame_names`, makes no change to this builder and returns `true`.
  //!
  //! NOTE: A mask is only meaningful between frames that have self-collision spheres. A mask
  //!       involving a frame with no self-collision spheres is discarded (with a warning) by
  //!       `build()`.
  [[nodiscard]] virtual bool addSelfCollisionMask(
      const std::string &frame_name,
      const std::vector<std::string> &masked_frame_names) = 0;

  //! Create a `RobotDescription` from the specified frames, joints, c-space coordinates, and
  //! collision geometry.
  //!
  //! Any frames not connected to the root frame will be discarded from the returned
  //! `RobotDescription`, as will collision spheres attached to those frames and any self-collision
  //! masks involving those frames.
  //!
  //! Each returned `RobotDescription` is fully independent from other built robot descriptions.
  //! Likewise, any additions to the builder after calling `build()` will not be reflected in
  //! previously built robot descriptions.
  //!
  //! A nonfatal error will be logged if a tool frame is not connected to the root frame, in which
  //! case it is omitted from `RobotDescription::toolFrameNames()`.
  //!
  //! A warning will be logged for each frame that is not connected to the root frame, and for each
  //! joint that drives such a frame, identifying what has been discarded.
  //!
  //! A warning will also be logged, and the affected geometry discarded, if:
  //!   1. A self-collision mask involves a frame with no self-collision spheres, *OR*
  //!   2. A frame with attached collision spheres is not connected to the root frame.
  //!
  //! A fatal error will be logged if a c-space (standard or auxiliary) coordinate would:
  //!   1. Control no joints because every joint it controls is discarded with a frame that is not
  //!      connected to the root frame, *OR*
  //!   2. Have insufficient position and/or velocity limit specification among the joints that
  //!      remain after such discarding.
  [[nodiscard]] virtual std::unique_ptr<RobotDescription> build() const = 0;
};

//! Initialize a `RobotDescriptionBuilder` with the name of the root frame.
//!
//! The root frame is added to the robot description by this function, so it must *NOT* be added
//! again via `RobotDescriptionBuilder::addFrame()`.
//!
//! The root frame is the base frame of each built `RobotDescription`, so the poses of all other
//! frames are expressed relative to it.
//!
//! A nonfatal error is logged and `nullptr` returned if `root_frame_name` is empty.
[[nodiscard]] CUMO_EXPORT std::unique_ptr<RobotDescriptionBuilder> CreateRobotDescriptionBuilder(
    const std::string &root_frame_name);

}  // namespace cumotion
