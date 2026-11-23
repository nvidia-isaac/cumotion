// SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES.
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
//! @brief Public interface for representing a pose in 3D.

#pragma once

#include <vector>

#include "Eigen/Core"

#include "cumotion/cumotion_export.h"
#include "cumotion/rotation3.h"

namespace cumotion {

//! Class representing a 3D pose.
struct CUMO_EXPORT Pose3 {
  //! Construct an identity pose with `rotation` set to identity and `translation` set to zero.
  Pose3();

  //! Construct pose from a `rotation` and `translation`.
  Pose3(const Rotation3 &rotation, const Eigen::Vector3d &translation);

  //! Construct pose from a 4 x 4 homogeneous transform matrix.
  //!
  //! NOTE: The upper left 3 x 3 submatrix of the input matrix is assumed to be SO(3), but it is not
  //! explicitly checked.
  //!
  //! NOTE: The bottom row of the input matrix is assumed to be `[0, 0, 0, 1]`, but is not
  //! explicitly checked.
  //!
  //! WARNING: Failure to meet input expectations results in undefined behavior.
  explicit Pose3(const Eigen::Matrix<double, 4, 4> &matrix);

  //! Create a pure rotational pose.
  static Pose3 FromRotation(const Rotation3 &rotation);

  //! Create a pure translational pose.
  static Pose3 FromTranslation(const Eigen::Vector3d &translation);

  //! Create identity pose.
  static Pose3 Identity();

  //! Return matrix representation of the pose.
  [[nodiscard]] Eigen::Matrix4d matrix() const;

  //! Return the inverse pose.
  [[nodiscard]] Pose3 inverse() const;

  //! Transform a collection of 3D `vectors` by this pose.
  [[nodiscard]]
  std::vector<Eigen::Vector3d> transformVectors(const std::vector<Eigen::Vector3d> &vectors) const;

  //! Transform a collection of 3D `vectors` by this pose.
  //!
  //! Each column of `vectors` represents a 3D vector to be transformed.
  [[nodiscard]] Eigen::Matrix<double, 3, Eigen::Dynamic> transformVectors(
      const Eigen::Matrix<double, 3, Eigen::Dynamic> &vectors) const;

  //! Compose poses via the `*` operator.
  friend CUMO_EXPORT Pose3 operator*(const Pose3 &lhs, const Pose3 &rhs);

  //! Transform a vector by the given pose.
  friend CUMO_EXPORT Eigen::Vector3d operator*(const Pose3 &pose, const Eigen::Vector3d &vector);

  //! Rotation component of pose.
  Rotation3 rotation;

  //! Translation component of pose.
  Eigen::Vector3d translation;
};

}  // namespace cumotion
