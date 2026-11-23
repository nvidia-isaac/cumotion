// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//                         All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"

#include "cumotion/cumotion_export.h"
#include "cumotion/robot_description.h"
#include "cumotion/rotation3.h"
#include "cumotion/trajectory.h"
#include "cumotion/world.h"

namespace cumotion {

//! Configuration parameters for a `TrajectoryOptimizer`.
class CUMO_EXPORT TrajectoryOptimizerConfig {
 public:
  virtual ~TrajectoryOptimizerConfig() = default;

  //! Specify the value for a given parameter.
  //!
  //! The required `ParamValue` constructor for each param is detailed in the
  //! documentation for `setParam()`.
  struct CUMO_EXPORT ParamValue {
    //! Create `ParamValue` from `int`.
    ParamValue(int value);  // NOLINT Allow implicit conversion
    //! Create `ParamValue` from `double`.
    ParamValue(double value);  // NOLINT Allow implicit conversion
    //! Create `ParamValue` from `bool`.
    ParamValue(bool value);  // NOLINT Allow implicit conversion
    //! Create `ParamValue` from `std::vector<double>`.
    ParamValue(const std::vector<double> &value);  // NOLINT Allow implicit conversion

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Set the value of the parameter.
  //!
  //! `setParam()` returns `true` if the parameter has been successfully updated and `false` if an
  //! error has occurred (e.g., invalid parameter).
  //!
  //! The following parameters can be set for `TrajectoryOptimizer`:
  //!
  //! `enable_self_collision` [`bool`]
  //!   - Enable or disable self-collision avoidance between robot spheres during optimization.
  //!   - When `true`, both inverse kinematics (IK) and trajectory optimization will include
  //!     collision costs to prevent robot spheres from colliding with each other.
  //!   - When `false`, self-collision costs are disabled, which may improve solve time but allows
  //!     self-intersecting trajectories.
  //!   - Default value is `true` (self-collision avoidance enabled).
  //!   - DEPRECATED: A deprecation warning is logged when `enable_self_collision` is set via
  //!     `setParam()` or when present in a configuration file. The value is still applied in this
  //!     release.  This parameter will be removed in a future release.
  //!
  //! `enable_world_collision` [`bool`]
  //!   - Enable or disable collision avoidance with world obstacles during optimization.
  //!   - When `true`, both IK and trajectory optimization will include collision costs to avoid
  //!     obstacles in the environment.
  //!   - When `false`, world collision costs are disabled, which may improve performance but allows
  //!     trajectories that intersect with obstacles.
  //!   - Default value is `true` (world collision avoidance enabled).
  //!   - DEPRECATED: A deprecation warning is logged when `enable_world_collision` is set via
  //!     `setParam()` or when present in a configuration file. The value is still applied in this
  //!     release.  This parameter will be removed in a future release.
  //!
  //! `task_space_terminal_position_tolerance` [`double`]
  //!   - Maximum allowable violation of the user-specified constraint on the position of the tool
  //!     frame.
  //!   - Used both for IK and trajectory optimization problems.
  //!     It determines when a solution is considered to have satisfied position targets.
  //!     Smaller values require more precise solutions but may be harder to achieve.
  //!   - Units are in meters.
  //!   - A default value of 1e-3 (1mm) is recommended for most use-cases.
  //!   - Must be positive.
  //!
  //! `task_space_terminal_orientation_tolerance` [`double`]
  //!   - Maximum allowable violation of the user-specified constraint on the orientation of the
  //!     tool frame.
  //!   - Used both for IK and trajectory optimization problems.
  //!     It determines when a solution is considered to have satisfied orientation targets.
  //!     Smaller values require more precise solutions but may be harder to achieve.
  //!   - Units are in radians.
  //!   - A default value of 5e-3 (approximately 0.29 degrees) is recommended for most use-cases.
  //!   - Must be positive.
  //!
  //! `ik/num_seeds` [`int`]
  //!   - Number of seeds used to solve the inverse kinematics (IK) problem.
  //!   - The trajectory optimizer generates multiple pseudo-random c-space configurations and
  //!     optimizes them to find diverse collision-free configurations for the desired tool pose.
  //!     Higher values increase the likelihood of finding valid solutions but increase
  //!     computational cost.
  //!   - A default value of 32 is recommended for most use-cases.
  //!   - Must be positive.
  //!
  //! `ik/max_reattempts` [`int`]
  //!   - Maximum number of times to restart the IK problem with different random seeds, in case of
  //!     failure, before giving up.
  //!   - Higher values increase the likelihood of finding valid IK solutions but increase
  //!     memory usage and the maximum possible time to find a solution. A value of 0 means no
  //!     retries (i.e. only perform the initial attempt).
  //!   - A default value of 10 is recommended for most use-cases.
  //!   - Must be in the range [0, 50].
  //!
  //! `ik/desired_cspace_position_weight` [`double`]
  //!   - Weight used on the distance to a desired c-space configuration when ranking IK solutions.
  //!   - A default value of 1.0 provides balanced behavior for most use-cases.
  //!   - Must be positive.
  //!
  //! `ik/pbo/cost/tool_frame_position_error_penalty/weight` [`double`]
  //!   - Penalty weight applied in task-space position error violations at tool-frame constraints
  //!     during PBO IK optimization.
  //!   - Higher values more strongly enforce exact position matching at goal targets.
  //!   - A default value of `10000.0` is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/pbo/cost/tool_frame_position_error_penalty/activation_distance` [`double`]
  //!   - Distance from the deviation limit at which task-space position error penalties activate.
  //!     This value is subtracted from the deviation limit specified when creating the position
  //!     constraint (e.g., `TranslationConstraint::Target()`) during PBO IK
  //!     optimization.
  //!   - If the activation distance is greater than the deviation limit (which is often zero), all
  //!     non-zero position errors will be penalized, encouraging solutions that have as little
  //!     error as possible.
  //!   - If the activation distance is less than the deviation limit, any solution with an error
  //!     less than `deviation_limit - activation_distance` will have equal cost. Most solutions
  //!     will have a final error of `deviation_limit - activation_distance`, instead of an error
  //!     near zero.
  //!   - Units are in meters.
  //!   - A default value of `0.01` is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/pbo/cost/tool_frame_orientation_error_penalty/weight` [`double`]
  //!   - Penalty weight applied in task-space orientation error violations at tool-frame
  //!     constraints during PBO IK optimization.
  //!   - Higher values more strongly enforce exact orientation matching at goal targets.
  //!   - A default value of 200.0 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/pbo/cost/tool_frame_orientation_error_penalty/activation_distance` [`double`]
  //!   - Angular distance from the deviation limit at which task-space orientation error penalties
  //!     activate. This value is subtracted from the deviation limit specified when creating the
  //!     orientation constraint (e.g., `OrientationConstraint::TerminalTarget()`) during
  //!     PBO IK optimization.
  //!   - If the activation distance is greater than the deviation limit (which is often zero), all
  //!     non-zero orientation errors will be penalized, encouraging solutions that have as little
  //!     angular error as possible.
  //!   - If the activation distance is less than the deviation limit, any solution with an error
  //!     less than `deviation_limit - activation_distance` will have equal cost. Most solutions
  //!     will have a final angular error of `deviation_limit - activation_distance`, instead of an
  //!     error near zero.
  //!   - Units are in radians.
  //!   - A default value of `0.01` is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/pbo/cost/cspace_position_limit_penalty/weight` [`double`]
  //!   - Penalty weight applied when IK solutions violate c-space position limits, minus the
  //!     activation distance, during Particle-Based Optimizer (PBO) IK optimization.
  //!   - Higher values more strongly discourage solutions near joint limits.
  //!   - A default value of 500.0 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/pbo/cost/cspace_position_limit_penalty/activation_distance` [`double`]
  //!   - Distance threshold from c-space position limits at which the position limit penalty
  //!     activates, during PBO IK optimization.
  //!   - Units correspond to rad for revolute joints, m for prismatic joints.
  //!   - A default value of 0.02 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/pbo/cost/enable_cspace_position_limit` [`bool`]
  //!   - Enable or disable c-space position limit penalties during PBO IK optimization.
  //!   - When `true`, PBO IK will penalize solutions that violate or approach c-space
  //!     position limits.
  //!   - When `false`, position limit costs are ignored (not recommended).
  //!   - Default value is `true` (position limit penalties enabled).
  //!
  //! `ik/pbo/cost/self_collision_penalty/weight` [`double`]
  //!   - Penalty weight applied in self-colliding robot configurations during PBO IK optimization.
  //!   - Higher values more strongly discourage self-colliding configurations.
  //!   - This parameter is only used when `enable_self_collision` is `true`.
  //!   - A default value of 1000.0 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/pbo/cost/self_collision_penalty/activation_distance` [`double`]
  //!   - Distance threshold at which the self-collision penalty activates during PBO IK
  //!     optimization.
  //!   - The penalty becomes non-zero when robot collision geometries are within this distance of
  //!     each other.
  //!   - Units are in meters.
  //!   - A default value of 0.01 m (1 cm) is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/pbo/cost/world_collision_penalty/weight` [`double`]
  //!   - Penalty weight applied in world-colliding robot configurations during PBO IK optimization.
  //!   - Higher values more strongly discourage configurations that collide with obstacles.
  //!   - This parameter is only used when `enable_world_collision` is `true`.
  //!   - A default value of 3000.0 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/pbo/cost/world_collision_penalty/activation_distance` [`double`]
  //!   - Distance threshold at which the world collision penalty activates during PBO IK
  //!     optimization.
  //!   - The penalty becomes non-zero when robot collision geometries are within this distance of
  //!     obstacles.
  //!   - Units are in meters.
  //!   - A default value of 0.01 m is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/lbfgs/cost/tool_frame_position_error_penalty/weight` [`double`]
  //!   - Penalty weight applied in task-space position error violations at tool-frame constraints
  //!     during L-BFGS IK optimization.
  //!   - Higher values more strongly enforce exact position matching at goal targets.
  //!   - A default value of 40000.0 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/lbfgs/cost/tool_frame_position_error_penalty/activation_distance` [`double`]
  //!   - Distance from the deviation limit at which task-space position error penalties activate.
  //!     This value is subtracted from the deviation limit specified when creating the position
  //!     constraint (e.g., `TranslationConstraint::Target()`) during L-BFGS IK
  //!     optimization.
  //!   - If the activation distance is greater than the deviation limit (which is often zero), all
  //!     non-zero position errors will be penalized, encouraging solutions that have as little
  //!     error as possible.
  //!   - If the activation distance is less than the deviation limit, any solution with an error
  //!     less than `deviation_limit - activation_distance` will have equal cost. Most solutions
  //!     will have a final error of `deviation_limit - activation_distance`, instead of an error
  //!     near zero.
  //!   - Units are in meters.
  //!   - A default value of `0.01` is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/lbfgs/cost/tool_frame_orientation_error_penalty/weight` [`double`]
  //!   - Penalty weight applied in task-space orientation error violations at tool-frame
  //!     constraints during L-BFGS IK optimization.
  //!   - Higher values more strongly enforce exact orientation matching at goal targets.
  //!   - A default value of 200.0 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/lbfgs/cost/tool_frame_orientation_error_penalty/activation_distance` [`double`]
  //!   - Angular distance from the deviation limit at which task-space orientation error penalties
  //!     activate. This value is subtracted from the deviation limit specified when creating the
  //!     orientation constraint (e.g., `OrientationConstraint::TerminalTarget()`) during L-BFGS IK
  //!     optimization.
  //!   - If the activation distance is greater than the deviation limit (which is often zero), all
  //!     non-zero orientation errors will be penalized, encouraging solutions that have as little
  //!     angular error as possible.
  //!   - If the activation distance is less than the deviation limit, any solution with an error
  //!     less than `deviation_limit - activation_distance` will have equal cost. Most solutions
  //!     will have a final angular error of `deviation_limit - activation_distance`, instead of an
  //!     error near zero.
  //!   - Units are in radians.
  //!   - A default value of `0.01` is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/lbfgs/cost/cspace_position_limit_penalty/weight` [`double`]
  //!   - Penalty weight applied when IK solutions violate c-space position limits, minus the
  //!     activation distance, during L-BFGS IK optimization.
  //!   - Higher values more strongly discourage solutions near joint limits.
  //!   - A default value of 500.0 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/lbfgs/cost/cspace_position_limit_penalty/activation_distance` [`double`]
  //!   - Distance threshold from c-space position limits at which the position limit penalty
  //!     activates, during L-BFGS IK optimization.
  //!   - Units correspond to rad for revolute joints, m for prismatic joints.
  //!   - A default value of 0.02 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/lbfgs/cost/enable_cspace_position_limit` [`bool`]
  //!   - Enable or disable c-space position limit penalties during L-BFGS IK optimization.
  //!   - When `true`, L-BFGS IK will penalize solutions that violate or approach c-space
  //!     position limits.
  //!   - When `false`, position limit costs are ignored (not recommended).
  //!   - Default value is `true` (position limit penalties enabled).
  //!
  //! `ik/lbfgs/cost/self_collision_penalty/weight` [`double`]
  //!   - Penalty weight applied in self-colliding robot configurations during L-BFGS IK
  //!     optimization.
  //!   - Higher values more strongly discourage self-colliding configurations.
  //!   - This parameter is only used when `enable_self_collision` is `true`.
  //!   - A default value of 1000.0 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/lbfgs/cost/self_collision_penalty/activation_distance` [`double`]
  //!   - Distance threshold at which the self-collision penalty activates during L-BFGS IK
  //!     optimization.
  //!   - The penalty becomes non-zero when robot collision geometries are within this distance of
  //!     each other.
  //!   - Units are in meters.
  //!   - A default value of 0.01 m (1 cm) is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/lbfgs/cost/world_collision_penalty/weight` [`double`]
  //!   - Penalty weight applied in world-colliding robot configurations during L-BFGS IK
  //!     optimization.
  //!   - Higher values more strongly discourage configurations that collide with obstacles.
  //!   - This parameter is only used when `enable_world_collision` is `true`.
  //!   - A default value of 3000.0 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/lbfgs/cost/world_collision_penalty/activation_distance` [`double`]
  //!   - Distance threshold at which the world collision penalty activates during L-BFGS IK
  //!     optimization.
  //!   - The penalty becomes non-zero when robot collision geometries are within this distance of
  //!     obstacles.
  //!   - Units are in meters.
  //!   - A default value of 0.0001 m (0.1 mm) is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `ik/lbfgs/max_iterations` [`int`]
  //!   - Maximum number of L-BFGS iterations for the inverse kinematics solver.
  //!   - Higher values allow the IK solver to converge more precisely but increase computational
  //!     cost per IK attempt.
  //!   - A default value of 150 is recommended for most use-cases.
  //!   - Must be positive.
  //!
  //! `trajopt/num_seeds` [`int`]
  //!   - Number of seeds used to solve the trajectory optimization problem.
  //!   - The trajectory optimizer generates initial trajectories from the initial configuration to
  //!     the c-space target or to the IK solutions, one per seed, and then optimizes each
  //!     trajectory. Higher values provide more trajectory candidates but increase computational
  //!     cost.
  //!   - A default value of 4 is recommended for most use-cases.
  //!   - Must be positive.
  //!
  //! `trajopt/num_spans` [`int`]
  //!   - Number of uniform spans in each cubic B-spline trajectory.
  //!   - Higher values provide more free control points and therefore more optimization freedom,
  //!     but increase memory use, computational cost, and the number of iterations required to
  //!     converge.
  //!   - A default value of 25 is recommended for most use-cases.
  //!   - Must be greater than or equal to 6, and must satisfy
  //!     `RobotDescription::numCSpaceCoords() * (trajopt/num_spans - 5) <= 1024`.
  //!
  //! `trajopt/num_knots_per_trajectory` [`int`]
  //!   - REMOVED: Use `trajopt/num_spans` to configure the trajectory discretization instead.
  //!     Trajectories are no longer represented by a dense sequence of knot points, so this
  //!     parameter is not mapped to `trajopt/num_spans` or any other parameter.
  //!   - Setting this parameter via `setParam()` will result in a non-fatal error and a `false`
  //!     value returned.
  //!   - Setting this parameter via a configuration file will result in a non-fatal error and a
  //!     `nullptr` returned by `CreateTrajectoryOptimizerConfigFromFile()`.
  //!
  //! `trajopt/num_geometric_planning_attempts` [`int`]
  //!   - DEPRECATED: This parameter no longer has any effect. A deprecation warning is logged when
  //!     `trajopt/num_geometric_planning_attempts` is set via `setParam()` or when present in a
  //!     configuration file.  Setting this parameter will always succeed for integer inputs, but
  //!     its value will be ignored.
  //!   - It is recommended to remove this parameter from `TrajectoryOptimizerConfig` settings.
  //!   - Support for this parameter will be removed in a future release.
  //!
  //! `trajopt/geometric_planning/max_iterations` [`int`]
  //!   - DEPRECATED: Use `path_planner/max_iterations` instead.
  //!   - For compatibility, `path_planner/max_iterations` is set by dividing the provided
  //!     `trajopt/geometric_planning/max_iterations` value by `10` to achieve roughly analogous
  //!     behavior.
  //!   - Values less than `10` cannot be converted to a valid `path_planner/max_iterations` value
  //!     and are rejected with a non-fatal error.
  //!   - When both this parameter and `path_planner/max_iterations` are present in a configuration
  //!     file, `path_planner/max_iterations` takes precedence and this parameter is ignored without
  //!     being converted, so any integer value is accepted for it.
  //!   - Support for this parameter will be removed in a future release.
  //!
  //! `trajopt/pbo/enabled` [`bool`]
  //!   - Enable or disable Particle-Based Optimizer (PBO) for trajectory optimization problems with
  //!     active path constraints.
  //!   - PBO is an algorithm that uses a set of "particles" to explore a non-linear design space.
  //!     If enabled, PBO is used to find collision-free trajectories before starting the
  //!     gradient-based L-BFGS optimization algorithm.  PBO is helpful for complex scenarios where
  //!     a linearly-interpolated path between the initial configuration and the IK solutions is not
  //!     sufficient to find a collision-free trajectory.
  //!   - When set to `false`, PBO is disabled, and L-BFGS is initialized with a
  //!     linearly-interpolated path.
  //!   - When set to `true`, PBO is enabled and the `trajopt/pbo` parameters control its behavior.
  //!   - This parameter has no effect when no path constraints are active. Those problems are
  //!     warm-started by the `PathPlanner` instead (see the `path_planner` parameters below).
  //!   - Default value is `true` (PBO enabled).
  //!
  //! `trajopt/pbo/num_particles_per_problem` [`int`]
  //!   - Number of particles used by PBO in the trajectory optimization problem, when enabled.
  //!   - This parameter controls the population size of particles in the PBO algorithm. Higher
  //!     values increase the likelihood of finding better solutions but increase solve time and
  //!     memory usage.
  //!   - This parameter is only used when `trajopt/pbo/enabled` is set to `true` and path
  //!     constraints are active.
  //!   - A default value of 25 is recommended for most use-cases.
  //!   - Must be positive.
  //!
  //! `trajopt/pbo/num_iterations` [`int`]
  //!   - Number of iterations to run PBO in the trajectory optimization problem, when enabled.
  //!   - This parameter controls how many optimization iterations PBO performs.
  //!     Higher values allow for more thorough optimization but increase computational cost.
  //!   - This parameter is only used when `trajopt/pbo/enabled` is set to `true` and path
  //!     constraints are active.
  //!   - A default value of 5 is recommended for most use-cases.
  //!   - Must be positive.
  //!
  //! `trajopt/pbo/cost/path_position_error_penalty/weight` [`double`]
  //!   - Penalty weight applied in task-space position error violations at tool-frame path
  //!     constraints during PBO trajectory optimization for non-terminal sampled positions.
  //!   - Higher values more strongly enforce exact position matching at goal targets.
  //!   - A default value of `5000.0` is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/path_position_error_penalty/activation_distance` [`double`]
  //!   - Distance from the deviation limit at which task-space path position error penalties
  //!     activate.  This value is subtracted from the deviation limit specified when creating the
  //!     path position constraint (e.g., `TranslationConstraint::LinearPathConstraint()`) during
  //!     PBO trajectory optimization.
  //!   - If the activation distance is greater than the deviation limit (which is often zero), all
  //!     non-zero position errors will be penalized, encouraging solutions that have as little
  //!     error as possible.
  //!   - If the activation distance is less than the deviation limit, any solution with an error
  //!     less than `deviation_limit - activation_distance` will have equal cost. Most solutions
  //!     will have a final error of `deviation_limit - activation_distance`, instead of an error
  //!     near zero.
  //!   - Units are in meters.
  //!   - A default value of `3e-3` is recommended for most use-cases.  Additionally, it is strongly
  //!     recommended that this value is less than or equal to the parameter
  //!     `trajopt/task_space_path_constraint_position_tolerance`.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/path_orientation_error_penalty/weight` [`double`]
  //!   - Penalty weight applied in task-space path orientation error violations at tool-frame path
  //!     constraints during PBO trajectory optimization.
  //!   - Higher values more strongly enforce exact orientation matching at goal targets.
  //!   - A default value of `500.0` is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/path_orientation_error_penalty/activation_distance` [`double`]
  //!   - Angular distance from the deviation limit at which task-space path orientation error
  //!     penalties activate. This value is subtracted from the deviation limit specified when
  //!     creating the path orientation constraint (e.g.,
  //!     `OrientationConstraint::TerminalAndPathTarget()`) during PBO trajectory optimization.
  //!   - If the activation distance is greater than the deviation limit (which is often zero), all
  //!     non-zero orientation errors will be penalized, encouraging solutions that have as little
  //!     angular error as possible.
  //!   - If the activation distance is less than the deviation limit, any solution with an error
  //!     less than `deviation_limit - activation_distance` will have equal cost. Most solutions
  //!     will have a final angular error of `deviation_limit - activation_distance`, instead of an
  //!     error near zero.
  //!   - Units are in radians.
  //!   - A default value of `1e-2` is recommended for most use-cases.  Additionally, it is strongly
  //!     recommended that this value is less than or equal to the parameter
  //!     `trajopt/task_space_path_constraint_orientation_tolerance`.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/cspace_position_limit_penalty/weight` [`double`]
  //!   - Penalty weight applied in c-space position limit violations during PBO trajectory
  //!     optimization.
  //!   - It discourages trajectories that violate joint limits.
  //!   - A default value of 10.0 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/cspace_position_limit_penalty/activation_distance` [`double`]
  //!   - Distance threshold from c-space position limits for PBO trajectory optimization penalties.
  //!   - Units correspond to rad for revolute joints, m for prismatic joints.
  //!   - A default value of 0.0 (no activation distance) is used.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/cspace_velocity_limit_penalty/weight` [`double`]
  //!   - Penalty weight applied in c-space velocity limit violations during PBO trajectory
  //!     optimization.
  //!   - It discourages trajectories with velocities that exceed joint limits.
  //!   - A default value of 0.0 (disabled) is used.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/cspace_velocity_limit_penalty/activation_distance` [`double`]
  //!   - DEPRECATED: This parameter no longer has any effect. A deprecation warning is logged when
  //!     this parameter is set via `setParam()` or when present in a configuration file. Setting
  //!     this parameter will succeed for non-negative inputs, but its value will be ignored.
  //!   - Use the `relative_activation_distance` parameter of the same penalty instead.
  //!   - Support for this parameter will be removed in a future release.
  //!
  //! `trajopt/pbo/cost/cspace_velocity_limit_penalty/relative_activation_distance` [`double`]
  //!   - Distance threshold, relative to the velocity limits, for PBO trajectory optimization
  //!     penalties.
  //!   - The penalty is non-zero wherever shortening the trajectory duration by the fraction
  //!     `relative_activation_distance` would exceed a velocity limit, i.e., where the magnitude of
  //!     a c-space velocity exceeds `(1 - relative_activation_distance) * limit`.
  //!   - A default value of 0.0 is used.
  //!   - Must be in the range `[0, 0.95]`.
  //!
  //! `trajopt/pbo/cost/cspace_acceleration_limit_penalty/weight` [`double`]
  //!   - Penalty weight applied in c-space acceleration limit violations during PBO trajectory
  //!     optimization.
  //!   - It discourages trajectories with accelerations that exceed joint limits.
  //!   - A default value of 0.0 (disabled) is used.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/cspace_acceleration_limit_penalty/activation_distance` [`double`]
  //!   - DEPRECATED: This parameter no longer has any effect. A deprecation warning is logged when
  //!     this parameter is set via `setParam()` or when present in a configuration file. Setting
  //!     this parameter will succeed for non-negative inputs, but its value will be ignored.
  //!   - Use the `relative_activation_distance` parameter of the same penalty instead.
  //!   - Support for this parameter will be removed in a future release.
  //!
  //! `trajopt/pbo/cost/cspace_acceleration_limit_penalty/relative_activation_distance` [`double`]
  //!   - Distance threshold, relative to the acceleration limits, for PBO trajectory optimization
  //!     penalties.
  //!   - The penalty is non-zero wherever shortening the trajectory duration by the fraction
  //!     `relative_activation_distance` would exceed an acceleration limit, i.e., where the
  //!     magnitude of a c-space acceleration exceeds
  //!     `(1 - relative_activation_distance)^2 * limit`.
  //!   - A default value of 0.0 is used.
  //!   - Must be in the range `[0, 0.95]`.
  //!
  //! `trajopt/pbo/cost/cspace_jerk_limit_penalty/weight` [`double`]
  //!   - Penalty weight applied in c-space jerk limit violations during PBO trajectory
  //!     optimization.
  //!   - It discourages trajectories with jerks that exceed joint limits.
  //!   - A default value of 0.0 (disabled) is used.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/cspace_jerk_limit_penalty/activation_distance` [`double`]
  //!   - DEPRECATED: This parameter no longer has any effect. A deprecation warning is logged when
  //!     this parameter is set via `setParam()` or when present in a configuration file. Setting
  //!     this parameter will succeed for non-negative inputs, but its value will be ignored.
  //!   - Use the `relative_activation_distance` parameter of the same penalty instead.
  //!   - Support for this parameter will be removed in a future release.
  //!
  //! `trajopt/pbo/cost/cspace_jerk_limit_penalty/relative_activation_distance` [`double`]
  //!   - Distance threshold, relative to the jerk limits, for PBO trajectory optimization
  //!     penalties.
  //!   - The penalty is non-zero wherever shortening the trajectory duration by the fraction
  //!     `relative_activation_distance` would exceed a jerk limit, i.e., where the magnitude of a
  //!     c-space jerk exceeds `(1 - relative_activation_distance)^3 * limit`.
  //!   - A default value of 0.0 is used.
  //!   - Must be in the range `[0, 0.95]`.
  //!
  //! `trajopt/pbo/cost/cspace_acceleration_smoothing_cost` [`double`]
  //!   - Weight used to encourage smooth trajectories during PBO trajectory optimization by
  //!     penalizing c-space acceleration.
  //!   - Higher values encourage smoother trajectories by penalizing non-zero accelerations.
  //!   - A default value of 10.0 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/cspace_jerk_smoothing_cost` [`double`]
  //!   - Weight applied to minimize c-space jerk (smoothness regularization) during PBO trajectory
  //!     optimization.
  //!   - Higher values encourage smoother trajectories by penalizing non-zero jerk values.
  //!   - A default value of 0.0 (disabled) is used.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/self_collision_penalty/weight` [`double`]
  //!   - Penalty weight applied in self-colliding robot configurations during PBO trajectory
  //!     optimization.
  //!   - This parameter is only used when `enable_self_collision` is `true`.
  //!   - A default value of 100000.0 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/self_collision_penalty/activation_distance` [`double`]
  //!   - Distance threshold at which the self-collision penalty activates during PBO trajectory
  //!     optimization.
  //!   - The penalty becomes non-zero when robot collision geometries are within this distance of
  //!     each other.
  //!   - Units are in meters.
  //!   - A default value of 0.01 m (1 cm) is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/world_collision_penalty/weight` [`double`]
  //!   - Penalty weight applied in world-colliding robot configurations during PBO trajectory
  //!     optimization.
  //!   - This parameter is only used when `enable_world_collision` is `true`.
  //!   - A default value of 5000.0 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/world_collision_penalty/activation_distance` [`double`]
  //!   - Distance threshold at which the world collision penalty activates during PBO trajectory
  //!     optimization.
  //!   - The penalty becomes non-zero when robot collision geometries are within this distance of
  //!     obstacles.
  //!   - Units are in meters.
  //!   - A default value of 0.025 m is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/pbo/cost/world_collision_max_sweep_steps` [`int`]
  //!   - Maximum number of discrete sampling steps when "sweeping" collision spheres between
  //!     adjacent sampled positions during PBO optimization. "Sweeping" samples sphere positions
  //!     along each interval to detect collisions that might be missed if only checking the
  //!     sampled positions themselves.
  //!   - Higher values provide more accurate collision detection but increase computational cost.
  //!   - This parameter is only used when `enable_world_collision` is `true`.
  //!   - A default value of 5 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/path_position_error_penalty/weight` [`double`]
  //!   - Penalty weight applied in task-space position error violations at tool-frame path
  //!     constraints during L-BFGS trajectory optimization for non-terminal sampled positions.
  //!   - Higher values more strongly enforce exact position matching at goal targets.
  //!   - A default value of `2.8e3` is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/path_position_error_penalty/activation_distance` [`double`]
  //!   - Distance from the deviation limit at which task-space path position error penalties
  //!     activate.  This value is subtracted from the deviation limit specified when creating the
  //!     path position constraint (e.g., `TranslationConstraint::LinearPathConstraint()`) during
  //!     L-BFGS trajectory optimization.
  //!   - If the activation distance is greater than the deviation limit (which is often zero), all
  //!     non-zero position errors will be penalized, encouraging solutions that have as little
  //!     error as possible.
  //!   - If the activation distance is less than the deviation limit, any solution with an error
  //!     less than `deviation_limit - activation_distance` will have equal cost. Most solutions
  //!     will have a final error of `deviation_limit - activation_distance`, instead of an error
  //!     near zero.
  //!   - Units are in meters.
  //!   - A default value of `3e-3` is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/path_orientation_error_penalty/weight` [`double`]
  //!   - Penalty weight applied in task-space path orientation error violations at tool-frame path
  //!     constraints during L-BFGS trajectory optimization.
  //!   - Higher values more strongly enforce exact orientation matching at goal targets.
  //!   - A default value of `7e2` is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/path_orientation_error_penalty/activation_distance` [`double`]
  //!   - Angular distance from the deviation limit at which task-space path orientation error
  //!     penalties activate. This value is subtracted from the deviation limit specified when
  //!     creating the path orientation constraint (e.g.,
  //!     `OrientationConstraint::TerminalAndPathTarget()`) during L-BFGS trajectory optimization.
  //!   - If the activation distance is greater than the deviation limit (which is often zero), all
  //!     non-zero orientation errors will be penalized, encouraging solutions that have as little
  //!     angular error as possible.
  //!   - If the activation distance is less than the deviation limit, any solution with an error
  //!     less than `deviation_limit - activation_distance` will have equal cost. Most solutions
  //!     will have a final angular error of `deviation_limit - activation_distance`, instead of an
  //!     error near zero.
  //!   - Units are in radians.
  //!   - A default value of `1e-2` is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/cspace_position_limit_penalty/weight` [`double`]
  //!   - Penalty weight applied in c-space position limit violations during L-BFGS trajectory
  //!     optimization.
  //!   - Higher values more strongly discourage trajectories that violate joint limits.
  //!   - A default value of 5e2 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/cspace_position_limit_penalty/activation_distance` [`double`]
  //!   - Distance threshold from c-space position limits for L-BFGS trajectory optimization
  //!     penalties.
  //!   - Units correspond to rad for revolute joints, m for prismatic joints.
  //!   - A default value equivalent to 5 degrees is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/cspace_velocity_limit_penalty/weight` [`double`]
  //!   - Penalty weight applied in c-space velocity limit violations during L-BFGS trajectory
  //!     optimization.
  //!   - Higher values more strongly discourage trajectories with excessive joint velocities.
  //!   - A default value of 1e5 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/cspace_velocity_limit_penalty/activation_distance` [`double`]
  //!   - DEPRECATED: This parameter no longer has any effect. A deprecation warning is logged when
  //!     this parameter is set via `setParam()` or when present in a configuration file. Setting
  //!     this parameter will succeed for non-negative inputs, but its value will be ignored.
  //!   - Use the `relative_activation_distance` parameter of the same penalty instead.
  //!   - Support for this parameter will be removed in a future release.
  //!
  //! `trajopt/lbfgs/cost/cspace_velocity_limit_penalty/relative_activation_distance` [`double`]
  //!   - Distance threshold, relative to the velocity limits, for L-BFGS trajectory optimization
  //!     penalties.
  //!   - The penalty is non-zero wherever shortening the trajectory duration by the fraction
  //!     `relative_activation_distance` would exceed a velocity limit, i.e., where the magnitude of
  //!     a c-space velocity exceeds `(1 - relative_activation_distance) * limit`.
  //!   - A default value of 0.2 is recommended for most use-cases.
  //!   - Must be in the range `[0, 0.95]`.
  //!
  //! `trajopt/lbfgs/cost/cspace_acceleration_limit_penalty/weight` [`double`]
  //!   - Penalty weight applied in c-space acceleration limit violations during L-BFGS trajectory
  //!     optimization.
  //!   - Higher values more strongly discourage trajectories with excessive joint accelerations.
  //!   - A default value of 7.5e3 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/cspace_acceleration_limit_penalty/activation_distance` [`double`]
  //!   - DEPRECATED: This parameter no longer has any effect. A deprecation warning is logged when
  //!     this parameter is set via `setParam()` or when present in a configuration file. Setting
  //!     this parameter will succeed for non-negative inputs, but its value will be ignored.
  //!   - Use the `relative_activation_distance` parameter of the same penalty instead.
  //!   - Support for this parameter will be removed in a future release.
  //!
  //! `trajopt/lbfgs/cost/cspace_acceleration_limit_penalty/relative_activation_distance` [`double`]
  //!   - Distance threshold, relative to the acceleration limits, for L-BFGS trajectory
  //!     optimization penalties.
  //!   - The penalty is non-zero wherever shortening the trajectory duration by the fraction
  //!     `relative_activation_distance` would exceed an acceleration limit, i.e., where the
  //!     magnitude of a c-space acceleration exceeds
  //!     `(1 - relative_activation_distance)^2 * limit`.
  //!   - A default value of 0.2 is recommended for most use-cases.
  //!   - Must be in the range `[0, 0.95]`.
  //!
  //! `trajopt/lbfgs/cost/cspace_jerk_limit_penalty/weight` [`double`]
  //!   - Penalty weight applied in c-space jerk limit violations during L-BFGS trajectory
  //!     optimization.
  //!   - Higher values more strongly discourage trajectories with excessive joint jerk.
  //!   - A default value of 3.5e3 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/cspace_jerk_limit_penalty/activation_distance` [`double`]
  //!   - DEPRECATED: This parameter no longer has any effect. A deprecation warning is logged when
  //!     this parameter is set via `setParam()` or when present in a configuration file. Setting
  //!     this parameter will succeed for non-negative inputs, but its value will be ignored.
  //!   - Use the `relative_activation_distance` parameter of the same penalty instead.
  //!   - Support for this parameter will be removed in a future release.
  //!
  //! `trajopt/lbfgs/cost/cspace_jerk_limit_penalty/relative_activation_distance` [`double`]
  //!   - Distance threshold, relative to the jerk limits, for L-BFGS trajectory optimization
  //!     penalties.
  //!   - The penalty is non-zero wherever shortening the trajectory duration by the fraction
  //!     `relative_activation_distance` would exceed a jerk limit, i.e., where the magnitude of a
  //!     c-space jerk exceeds `(1 - relative_activation_distance)^3 * limit`.
  //!   - A default value of 0.2 is recommended for most use-cases.
  //!   - Must be in the range `[0, 0.95]`.
  //!
  //! `trajopt/lbfgs/cost/cspace_acceleration_smoothing_cost` [`double`]
  //!   - Weight applied to minimize c-space acceleration (smoothness regularization) during L-BFGS
  //!     trajectory optimization.
  //!   - Higher values encourage smoother trajectories by penalizing large accelerations.
  //!   - A default value of 7e-6 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/cspace_jerk_smoothing_cost` [`double`]
  //!   - Weight applied to minimize c-space jerk (smoothness regularization) during L-BFGS
  //!     trajectory optimization.
  //!   - Higher values encourage smoother trajectories by penalizing large jerk values.
  //!   - A default value of 2.5e-8 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/self_collision_penalty/weight` [`double`]
  //!   - Penalty weight applied in self-colliding robot configurations during L-BFGS trajectory
  //!     optimization.
  //!   - This parameter is only used when `enable_self_collision` is `true`.
  //!   - A default value of 5.5e4 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/self_collision_penalty/activation_distance` [`double`]
  //!   - Distance threshold at which the self-collision penalty activates during L-BFGS trajectory
  //!     optimization.
  //!   - The penalty becomes non-zero when robot collision geometries are within this distance of
  //!     each other.
  //!   - Units are in meters.
  //!   - A default value of 0.011 m (1.1 cm) is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/world_collision_penalty/weight` [`double`]
  //!   - Penalty weight applied in world-colliding robot configurations during L-BFGS trajectory
  //!     optimization.
  //!   - This parameter is only used when `enable_world_collision` is `true`.
  //!   - A default value of 8.5e2 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/world_collision_penalty/activation_distance` [`double`]
  //!   - Distance threshold at which the world-collision penalty activates during L-BFGS trajectory
  //!     optimization.
  //!   - The penalty becomes non-zero when robot collision geometries are within this distance of
  //!     obstacles.
  //!   - Units are in meters.
  //!   - A default value of 0.01 m (1 cm) is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/cost/world_collision_max_sweep_steps` [`int`]
  //!   - Maximum number of discrete sampling steps when "sweeping" collision spheres between
  //!     adjacent sampled positions during L-BFGS optimization. "Sweeping" samples sphere
  //!     positions along each interval to detect collisions that might be missed if only checking
  //!     the sampled positions themselves.
  //!   - Higher values provide more accurate collision detection but increase computational cost.
  //!   - This parameter is only used when `enable_world_collision` is `true`.
  //!   - A default value of 5 is recommended for most use-cases.
  //!   - Must be non-negative.
  //!
  //! `trajopt/lbfgs/history_length` [`int`]
  //!   - Number of past iterations stored by the L-BFGS algorithm to approximate the inverse
  //!     Hessian during trajectory optimization.
  //!   - Larger values may improve convergence quality but increase memory usage and per-iteration
  //!     computational cost.
  //!   - A default value of 5 is recommended for most use-cases.
  //!   - Must be less than or equal to the number of optimization variables in the trajectory
  //!     optimization problem. For the current implementation, this is calculated as
  //!     `RobotDescription::numCSpaceCoords() * (trajopt/num_spans - 5)`.
  //!     This value is automatically clamped to not exceed the number of optimization variables, if
  //!     setting any other parameter (e.g., `trajopt/num_spans`) reduces the upper bound.
  //!   - Must be positive.
  //!
  //! `trajopt/lbfgs/initial_iterations` [`int`]
  //!   - Number of L-BFGS iterations used to solve the initial trajectory optimization problem.
  //!   - A smaller number of iterations will result in a more coarse solution used when retiming
  //!     the trajectory.
  //!   - The default value is a reasonable starting point, but performance improvements may be
  //!     found by tuning this parameter.
  //!   - Default value: 150
  //!   - Must be positive.
  //!
  //! `trajopt/lbfgs/retiming_initial_iterations` [`int`]
  //!   - Number of L-BFGS iterations used to solve the 2nd trajectory optimization problem, after
  //!     the initial solutions are retimed using `trajopt/timestep_scaling_factor` to minimize the
  //!     trajectory time by saturating the kinematic limits.
  //!   - A smaller number of iterations will result in a more coarse solution used when retiming
  //!     the trajectory.
  //!   - The default value is a reasonable starting point, but performance improvements may be
  //!     found by tuning this parameter.
  //!   - Default value: 50
  //!   - Must be positive.
  //!
  //! `trajopt/lbfgs/retiming_iterations` [`int`]
  //!   - Number of L-BFGS iterations used to solve the trajectory optimization problem after the
  //!     initial solve and first retiming attempt, after the previous solution is scaled by
  //!     `trajopt/timestep_scaling_factor` to minimize trajectory time by saturating the
  //!     kinematic limits.
  //!   - A smaller number of iterations will result in a more coarse final solution, including the
  //!     possibility of failing to find a valid solution where one would have been found with a
  //!     greater number of iterations.
  //!   - A larger number of iterations will require more compute time, but may provide a
  //!     higher-quality final trajectory (typically meaning smoother kinematic derivatives).
  //!   - The default value is a reasonable starting point, but performance improvements may be
  //!     found by tuning this parameter.
  //!   - Default value: 50
  //!   - Must be positive.
  //!
  //! `trajopt/initial_timestep_scaling_factor` [`double`]
  //!   - Scales the saturating timestep calculated on the initial trajectory seeds.
  //!   - Smaller values result in shorter initial trajectories where the derivative limits are
  //!     violated for a greater percentage of the trajectory.
  //!   - A value of 1.0 or more will not violate kinematic derivative limits at the sampled
  //!     positions unless the configured maximum timestep prevents it.
  //!   - The default value is a reasonable starting point, but performance improvements may be
  //!     found by tuning this parameter.
  //!   - Default value: 1.24
  //!   - Must be positive.
  //!
  //! `trajopt/timestep_scaling_factor` [`double`]
  //!   - Scales the saturating timestep calculated on the retimed trajectory seeds (after the
  //!     initial solve).
  //!   - Smaller values result in shorter retimed trajectories where the derivative limits are
  //!     violated for a greater percentage of the trajectory.
  //!   - A value of 1.0 or more will not violate kinematic derivative limits at the sampled
  //!     positions unless the configured maximum timestep prevents it.
  //!   - The default value is a reasonable starting point, but performance improvements may be
  //!     found by tuning this parameter.
  //!   - Default value: 1.0
  //!   - Must be positive.
  //!
  //! `trajopt/num_retiming_iterations` [`int`]
  //!   - Number of times the trajectory is retimed after the initial solve.
  //!   - A smaller number of iterations may result in a shorter planning time and a longer final
  //!     trajectory time. A higher number of iterations may result in a shorter final trajectory
  //!     time.
  //!   - The default value is a reasonable starting point, but performance improvements may be
  //!     found by tuning this parameter.
  //!   - A value of 0 will disable retiming and the trajectory will use the timestep calculated
  //!     by saturating the kinematic limits of the initial solution.
  //!   - Default value: 5
  //!   - Must be non-negative.
  //!
  //! `trajopt/task_space_path_constraint_position_tolerance` [`double`]
  //!   - Maximum allowable violation of the user-specified path constraint on the position of the
  //!     tool frame. A position path constraint is considered satisfied if the maximum deviation
  //!     of the tool frame from the path is less than the sum of this tolerance and the
  //!     user-specified deviation limit.
  //!   - Units are in meters.
  //!   - A default value of `5e-3` meters is recommended for most use-cases. If tighter tolerances
  //!     are required, the caller may find that more trajectory seeds are needed to consistently
  //!     find a solution.
  //!   - Must be positive.
  //!
  //! `trajopt/task_space_path_constraint_orientation_tolerance` [`double`]
  //!   - Maximum allowable violation of the user-specified path constraint on the orientation of
  //!     the tool frame. An orientation path constraint is considered satisfied if the
  //!     maximum deviation of the tool frame from the desired orientation constraint is less than
  //!     the sum of this tolerance and the user-specified deviation limit.
  //!   - Units are in radians.
  //!   - A default value of `0.02` radians is recommended for most use-cases. If tighter tolerances
  //!     are required, the caller may find that more trajectory seeds are needed to consistently
  //!     find a solution.
  //!   - Must be positive.
  //!
  //! The `path_planner` namespace configures the `PathPlanner`, which generates feasible
  //! warm-starts for the trajectory optimization when no path constraints are present.  A
  //! parameter named `path_planner/<name>` sets the parameter `<name>` of the `PathPlanner`, and is
  //! accepted if and only if `<name>` is accepted by `PathPlannerConfig::setParam()`.  See
  //! `path_planner.h` for a list of supported parameters and tuning recommendations.
  //!
  //! NOTE: Some `PathPlanner` parameters have different tuning recommendations across different
  //!       GPU architectures.  Alternative tunings for different architectures are provided in the
  //!       `content/nvidia/shared/` directory.
  [[nodiscard]] virtual bool setParam(const std::string &param_name, ParamValue value) = 0;
};

//! Load a set of `TrajectoryOptimizer` configuration parameters from file.
//!
//! These parameters are combined with `robot_description`, `tool_frame_name`, and `world_view`
//! to create a configuration for trajectory optimization.
//!
//! Default values will be used for any parameters not specified in
//! `trajectory_optimizer_config_file`.
//!
//! A configuration will *NOT* be created if:
//!   1. `trajectory_optimizer_config_file` path is invalid or empty,
//!   2. `trajectory_optimizer_config_file` points to an invalid YAML file,
//!   3. `trajectory_optimizer_config_file` does not specify the supported format
//!      (`format: cumotion_trajopt` and `format_version: 2.0`),
//!   4. `trajectory_optimizer_config_file` contains invalid parameters,
//!   5. `trajectory_optimizer_config_file` contains duplicate keys,
//!   6. `robot_description` has zero c-space coordinates, *OR*
//!   7. `tool_frame_name` is not a valid frame in `robot_description`.
//!
//! In any of these cases, a non-fatal error will be logged and a `nullptr` will be returned.
[[nodiscard]] CUMO_EXPORT std::unique_ptr<TrajectoryOptimizerConfig>
CreateTrajectoryOptimizerConfigFromFile(
    const std::filesystem::path &trajectory_optimizer_config_file,
    const RobotDescription &robot_description,
    const std::string &tool_frame_name,
    const WorldViewHandle &world_view);

//! Use default parameters to create a configuration for trajectory optimization.
//!
//! These default parameters are combined with `robot_description`, `tool_frame_name`, and
//! `world_view`.
//!
//! A configuration will *NOT* be created if:
//!   1. `robot_description` has zero c-space coordinates, *OR*
//!   2. `tool_frame_name` is not a valid frame in `robot_description`.
//!
//! In the case of failure, a non-fatal error will be logged and a `nullptr` will be returned.
[[nodiscard]] CUMO_EXPORT std::unique_ptr<TrajectoryOptimizerConfig>
CreateDefaultTrajectoryOptimizerConfig(const RobotDescription &robot_description,
                                       const std::string &tool_frame_name,
                                       const WorldViewHandle &world_view);

//! Interface for using numerical optimization to generate collision-free trajectories for a robot.
//!
//! \rst
//! See documentation for corresponding :py:class:`Python class <cumotion.TrajectoryOptimizer>`.
//! \endrst
class CUMO_EXPORT TrajectoryOptimizer {
 public:
  virtual ~TrajectoryOptimizer() = default;

  //! Translation constraints restrict the position of the origin of a tool frame.
  //!
  //! These constraints are always active at the terminal point of the trajectory; partial
  //! constraints may be active along the path.
  class CUMO_EXPORT TranslationConstraint {
   public:
    //! Create a `TranslationConstraint` such that a `translation_target` is specified at
    //! termination, but no translation constraints are active along the path.
    //!
    //! The optional parameter `terminal_deviation_limit` can be used to allow deviation from the
    //! `translation_target` at termination. This limit specifies the maximum allowable deviation
    //! from the desired position.
    //!
    //! If `terminal_deviation_limit` is not input, then the deviation limit is assumed to be zero
    //! (i.e., it is desired that terminal translation be exactly `translation_target`).
    //!
    //! A fatal error will be logged if:
    //!   1. `terminal_deviation_limit` is negative.
    //!
    //! NOTE: The `translation_target` is specified in world frame coordinates (i.e., relative to
    //! the base of the robot).
    static TranslationConstraint Target(const Eigen::Vector3d &translation_target,
                                        const double *terminal_deviation_limit = nullptr);

    //! Create a `TranslationConstraint` such that a `translation_target` is specified at
    //! termination *AND* a linear translation constraint is active along the path.
    //!
    //! The linear path constraint is defined by the line **segment** between `translation_target`
    //! and the origin of the tool frame at the initial c-space position. This path constraint is
    //! satisfied if the origin of the tool frame is on this line segment between the initial and
    //! final tool positions at all points in the trajectory. It is considered a constraint
    //! violation if the origin of the tool frame "overshoots" either the initial or final
    //! positions.
    //!
    //! If the distance between the initial and final tool frames are nearly the same, the path
    //! constraint becomes a constraint on the distance from the initial and final positions in any
    //! direction.
    //!
    //! The optional parameters `path_deviation_limit` and `terminal_deviation_limit` can be used
    //! to allow deviations from the specified line along the path and at the end of the path,
    //! respectively.
    //!
    //! If both limits are input, the `path_deviation_limit` must be greater than or equal to
    //! (i.e., no more restrictive than) the `terminal_deviation_limit`.
    //!
    //! If neither is input, then both deviation limits are assumed to be zero, indicating that
    //! translation should always be exactly coincident with the specified line.
    //!
    //! It is valid to input a `path_deviation_limit` without a `terminal_deviation_limit`
    //! (implicitly setting the latter to zero) but not to input a `terminal_deviation_limit`
    //! without a `path_deviation_limit`. This ensures that the terminal constraint is more
    //! restrictive than the corresponding path constraint.
    //!
    //! A fatal error will be logged if:
    //!   1. `path_deviation_limit` is negative,
    //!   2. `terminal_deviation_limit` is negative,
    //!   3. both `path_deviation_limit` and `terminal_deviation_limit` are defined, but
    //!      `path_deviation_limit` < `terminal_deviation_limit`, *OR*
    //!   4. `terminal_deviation_limit` is defined without defining a `path_deviation_limit`.
    //!
    //! NOTE: The `translation_target` is specified in world frame coordinates (i.e., relative to
    //! the base of the robot).
    static TranslationConstraint LinearPathConstraint(
        const Eigen::Vector3d &translation_target,
        const double *path_deviation_limit = nullptr,
        const double *terminal_deviation_limit = nullptr);

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Variant of `TranslationConstraint` for "goalset" planning.
  //!
  //! For goalset planning, a set of `TranslationConstraint`s are considered concurrently. Each
  //! `TranslationConstraint` in the goalset must have the same mode (e.g., "terminal target
  //! with linear path constraint") but may have different data for each `TranslationConstraint`.
  class CUMO_EXPORT TranslationConstraintGoalset {
   public:
    //! Create a `TranslationConstraintGoalset` such that `translation_targets` are specified at
    //! termination, but no translation constraints are active along the path.
    //!
    //! See `TranslationConstraint::Target()` for details.
    //!
    //! A fatal error will be logged if:
    //!   1. Any condition of `TranslationConstraint::Target()` is not met, *OR*
    //!   2. `translation_targets` is empty.
    static TranslationConstraintGoalset Target(
        const std::vector<Eigen::Vector3d> &translation_targets,
        const double *terminal_deviation_limit = nullptr);

    //! Create a `TranslationConstraintGoalset` such that `translation_targets` are specified at
    //! termination *AND* linear translation constraints are active along the path.
    //!
    //! See `TranslationConstraint::LinearPathConstraint()` for details.
    //!
    //! A fatal error will be logged if:
    //!   1. Any condition of `TranslationConstraint::LinearPathConstraint()` is not met, *OR*
    //!   2. `translation_targets` is empty.
    static TranslationConstraintGoalset LinearPathConstraint(
        const std::vector<Eigen::Vector3d> &translation_targets,
        const double *path_deviation_limit = nullptr,
        const double *terminal_deviation_limit = nullptr);

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Orientation constraints restrict the orientation of a tool frame.
  //!
  //! These constraints may be active at the terminal point of the trajectory and/or along the path.
  //! Each constraint may fully or partially constrain the orientation.
  class CUMO_EXPORT OrientationConstraint {
   public:
    //! Create an `OrientationConstraint` such that no tool frame orientation constraints are
    //! active along the path *OR* at termination.
    static OrientationConstraint None();

    //! Create an `OrientationConstraint` such that the tool frame orientation along the path *AND*
    //! at termination are constrained to the tool frame orientation at the initial
    //! c-space position.
    //!
    //! The optional parameters `path_deviation_limit` and `terminal_deviation_limit` can be used
    //! to allow deviations from this constant orientation along the path and at termination
    //! respectively. If input, the limits are expressed in radians. These limits specify the
    //! maximum allowable deviation from the desired orientation.
    //!
    //! If both limits are input, the `path_deviation_limit` must be greater than or equal to
    //! (i.e., no more restrictive than) the `terminal_deviation_limit`.
    //!
    //! If neither is input, then both deviation limits are assumed to be zero, indicating that
    //! orientation should be exactly constant.
    //!
    //! It is valid to input a `path_deviation_limit` without a `terminal_deviation_limit`
    //! (implicitly setting the latter to zero) but not to input a `terminal_deviation_limit`
    //! without a `path_deviation_limit`. This ensures that the terminal constraint is more
    //! restrictive than the corresponding path constraint.
    //!
    //! In general, it is suggested that angular deviation limits are set to values less than pi.
    //! A limit that is near or greater than pi essentially disables the constraint (without culling
    //! the computation). Non-fatal warnings will be logged if:
    //!   1. `path_deviation_limit` is near or greater than pi, *OR*
    //!   2. `terminal_deviation_limit` is near or greater than pi.
    //!
    //! A fatal error will be logged if:
    //!   1. `path_deviation_limit` is negative,
    //!   2. `terminal_deviation_limit` is negative,
    //!   3. both `path_deviation_limit` and `terminal_deviation_limit` are defined, but
    //!      `path_deviation_limit` < `terminal_deviation_limit`, *OR*
    //!   4. `terminal_deviation_limit` is defined without defining a `path_deviation_limit`.
    static OrientationConstraint Constant(const double *path_deviation_limit = nullptr,
                                          const double *terminal_deviation_limit = nullptr);

    //! Create an `OrientationConstraint` such that a tool frame `orientation_target` is specified
    //! at termination, but no orientation constraints are active along the path.
    //!
    //! The optional parameter `terminal_deviation_limit` can be used to allow deviation from the
    //! `orientation_target` at termination. If input, `terminal_deviation_limit` is expressed in
    //! radians. This limit specifies the maximum allowable deviation from the desired
    //! orientation.
    //!
    //! If `terminal_deviation_limit` is not input, then the deviation limit is assumed to be zero
    //! (i.e., it is desired that terminal orientation be exactly `orientation_target`).
    //!
    //! In general, it is suggested that angular deviation limits are set to values less than pi.
    //! A limit that is near or greater than pi essentially disables the constraint (without culling
    //! the computation). Non-fatal warnings will be logged if:
    //!   - `terminal_deviation_limit` is near or greater than pi.
    //!
    //! A fatal error will be logged if:
    //!   - `terminal_deviation_limit` is negative.
    //!
    //! NOTE: The `orientation_target` is specified in world frame coordinates (i.e., relative to
    //! the base of the robot).
    static OrientationConstraint TerminalTarget(const Rotation3 &orientation_target,
                                                const double *terminal_deviation_limit = nullptr);

    //! Create an `OrientationConstraint` such that a tool frame `orientation_target` is specified
    //! along the path *AND* at termination.
    //!
    //! The optional parameters `path_deviation_limit` and `terminal_deviation_limit` can be used
    //! to allow deviations from the `orientation_target` along the path and at termination
    //! respectively. If input, the limits are expressed in radians. These limits specify the
    //! maximum allowable deviation from the desired orientation.
    //!
    //! If both limits are input, the `path_deviation_limit` must be greater than or equal to
    //! (i.e., no more restrictive than) the `terminal_deviation_limit`.
    //!
    //! If the `path_deviation_limit` is *NOT* input, its value is set to the most restrictive
    //! feasible value. This feasibility is determined by the tool frame orientation at the initial
    //! c-space position. If the initial orientation exactly matches `orientation_target`, then
    //! the effective `path_deviation_limit` will be zero. Otherwise, the `path_deviation_limit`
    //! will be set to the angular distance between the initial orientation and
    //! `orientation_target`. If the `path_deviation_limit` is input, it must be greater than or
    //! equal to this angular distance.
    //!
    //! If the `terminal_deviation_limit` is *NOT* input, its value is set to zero (i.e., the most
    //! restrictive feasible value). If `terminal_deviation_limit` *AND* `path_deviation_limit` are
    //! input, then `terminal_deviation_limit` must be less than or equal to `path_deviation_limit`.
    //! If `terminal_deviation_limit` is specified but `path_deviation_limit` is auto-computed, then
    //! `terminal_deviation_limit` will be clamped to the minimum of the input value and the
    //! auto-computed value of `path_deviation_limit`.
    //!
    //! In general, it is suggested that angular deviation limits are set to values less than pi.
    //! A limit that is near or greater than pi essentially disables the constraint (without culling
    //! the computation). Non-fatal warnings will be logged if:
    //!   1. `path_deviation_limit` is near or greater than pi, *OR*
    //!   2. `terminal_deviation_limit` is near or greater than pi.
    //!
    //! A fatal error will be logged if:
    //!   1. `path_deviation_limit` is negative,
    //!   2. `terminal_deviation_limit` is negative, *OR*
    //!   3. `terminal_deviation_limit` is greater than `path_deviation_limit`.
    //!
    //! A non-fatal error will be logged if `path_deviation_limit` results in an infeasible initial
    //! c-space position. In this case, `Results::status()` will return
    //! `INVALID_TARGET_SPECIFICATION`.
    //!
    //! NOTE: This condition will only be validated when the resulting `OrientationConstraint` is
    //!       used for trajectory optimization (i.e., it is used as input to
    //!       `planToTaskSpaceTarget()`).
    //!
    //! NOTE: The `orientation_target` is specified in world frame coordinates (i.e., relative to
    //!       the base of the robot).
    static OrientationConstraint TerminalAndPathTarget(
        const Rotation3 &orientation_target,
        const double *path_deviation_limit = nullptr,
        const double *terminal_deviation_limit = nullptr);

    //! Create an `OrientationConstraint` such that the terminal tool frame orientation is
    //! constrained to rotate about a "free axis", but no orientation constraints are active along
    //! the path.
    //!
    //! The "free axis" for rotation is defined by a `tool_frame_axis` (specified in the tool frame
    //! coordinates) and a corresponding `world_target_axis` (specified in world frame coordinates).
    //!
    //! The optional `terminal_axis_deviation_limit` can be used to allow deviation from the
    //! desired axis alignment at termination. If input, `terminal_axis_deviation_limit` is
    //! expressed in radians and the limit specifies the maximum allowable deviation of the
    //! rotation axis at termination. If `terminal_axis_deviation_limit` is not input, then the
    //! deviation limit is assumed to be zero (i.e., it is desired that the tool frame axis be
    //! exactly aligned with `world_target_axis`).
    //!
    //! In general, it is suggested that angular deviation limits are set to values less than pi.
    //! A limit that is near or greater than pi essentially disables the constraint (without culling
    //! the computation). Non-fatal warnings will be logged if:
    //!   - `terminal_axis_deviation_limit` is near or greater than pi.
    //!
    //! A fatal error will be logged if:
    //!   1. `terminal_axis_deviation_limit` is negative,
    //!   2. `tool_frame_axis` is (nearly) zero, *OR*
    //!   3. `world_target_axis` is (nearly) zero.
    //!
    //! NOTE: `tool_frame_axis` and `world_target_axis` inputs will be normalized.
    static OrientationConstraint TerminalAxis(
        const Eigen::Vector3d &tool_frame_axis,
        const Eigen::Vector3d &world_target_axis,
        const double *terminal_axis_deviation_limit = nullptr);

    //! Create an `OrientationConstraint` such that the tool frame orientation is constrained to
    //! rotate about a "free axis" along the path *AND* at termination.
    //!
    //! The "free axis" for rotation is defined by a `tool_frame_axis` (specified in the tool frame
    //! coordinates) and a corresponding `world_target_axis` (specified in world frame coordinates).
    //!
    //! The optional parameters `path_axis_deviation_limit` and `terminal_axis_deviation_limit`
    //! can be used to allow deviation from the desired axis alignment along the path and at
    //! termination respectively. If input, the limits are expressed in radians.
    //!
    //! If both limits are input, the `path_axis_deviation_limit` must be greater than or equal to
    //! (i.e., no more restrictive than) the `terminal_axis_deviation_limit`.
    //!
    //! If the `path_axis_deviation_limit` is *NOT* input, its value is set to the most restrictive
    //! feasible value. This feasibility is determined by the tool frame orientation at the initial
    //! c-space position. If the initial orientation exactly satisfies the axis constraint,
    //! then the effective `path_axis_deviation_limit` will be zero. Otherwise, the
    //! `path_axis_deviation_limit` will be set to the angular distance between the initial tool
    //! frame rotation axis and the desired tool frame rotation axis. If the
    //! `path_axis_deviation_limit` is input, it must be greater than or equal to this angular
    //! distance.
    //!
    //! If the `terminal_axis_deviation_limit` is *NOT* input, its value is set to zero (i.e., the
    //! most restrictive feasible value). If `terminal_axis_deviation_limit` *AND*
    //! `path_axis_deviation_limit` are input, then `terminal_axis_deviation_limit` must be less
    //! than or equal to `path_axis_deviation_limit`. If `terminal_axis_deviation_limit` is
    //! specified but `path_axis_deviation_limit` is auto-computed, then
    //! `terminal_axis_deviation_limit` will be clamped to the minimum of the input value and the
    //! auto-computed value of `path_axis_deviation_limit`.
    //!
    //! In general, it is suggested that angular deviation limits are set to values less than pi.
    //! A limit that is near or greater than pi essentially disables the constraint (without culling
    //! the computation). Non-fatal warnings will be logged if:
    //!   1. `path_axis_deviation_limit` is near or greater than pi, *OR*
    //!   2. `terminal_axis_deviation_limit` is near or greater than pi.
    //!
    //! A fatal error will be logged if:
    //!   1. `path_axis_deviation_limit` is negative,
    //!   2. `terminal_axis_deviation_limit` is negative,
    //!   3. `terminal_axis_deviation_limit` is greater than `path_axis_deviation_limit`,
    //!   4. `tool_frame_axis` is (nearly) zero, *OR*
    //!   5. `world_target_axis` is (nearly) zero.
    //!
    //! A non-fatal error will be logged if `path_axis_deviation_limit` results in an infeasible
    //! initial c-space position. In this case, `Results::status()` will return
    //! `INVALID_TARGET_SPECIFICATION`.
    //!
    //! NOTE: This condition will only be validated when the resulting `OrientationConstraint` is
    //!       used for trajectory optimization (i.e., it is used as input to
    //!       `planToTaskSpaceTarget()`).
    //!
    //! NOTE: `tool_frame_axis` and `world_target_axis` inputs will be normalized.
    static OrientationConstraint TerminalAndPathAxis(
        const Eigen::Vector3d &tool_frame_axis,
        const Eigen::Vector3d &world_target_axis,
        const double *path_axis_deviation_limit = nullptr,
        const double *terminal_axis_deviation_limit = nullptr);

    //! Create an `OrientationConstraint` such that a tool frame `terminal_orientation_target` is
    //! specified at termination, *AND* the tool frame orientation is constrained to rotate
    //! about a "free axis" along the path.
    //!
    //! The optional parameter `terminal_deviation_limit` can be used to allow deviation from the
    //! `terminal_orientation_target` at termination. If input, `terminal_deviation_limit` is
    //! expressed in radians. This limit specifies the maximum allowable deviation from the desired
    //! orientation.
    //!
    //! If `terminal_deviation_limit` is not input, then the deviation limit is assumed to be zero
    //! (i.e., it is desired that terminal orientation be exactly `terminal_orientation_target`).
    //!
    //! The "free axis" for rotation along the path is defined by a `tool_frame_axis` (specified
    //! in the tool frame coordinates) and a corresponding `world_target_axis` (specified in
    //! world frame coordinates).
    //!
    //! The optional `path_axis_deviation_limit` can be used to allow deviation from the
    //! desired axis alignment along the path. If input, `path_axis_deviation_limit` is
    //! expressed in radians. This limit specifies the maximum allowable deviation of the
    //! rotation axis along the path.
    //!
    //! If the `path_axis_deviation_limit` is *NOT* input, its value is set to the most restrictive
    //! feasible value. This feasibility is determined by the tool frame orientation at the initial
    //! c-space position *AND* by the `terminal_orientation_target`. The `path_axis_deviation_limit`
    //! will be set to the maximum of:
    //!   1. The angular distance between the initial tool frame rotation axis and the desired tool
    //!      frame rotation axis, *AND*
    //!   2. The angular distance between the tool frame rotation axis at
    //!      `terminal_orientation_target` and the desired tool frame rotation axis, reduced by
    //!      `terminal_deviation_limit` (but not below zero).
    //!
    //! If the `path_axis_deviation_limit` is input, it must be greater than or equal to this
    //! minimum feasible deviation limit.
    //!
    //! In general, it is suggested that angular deviation limits are set to values less than pi.
    //! A limit that is near or greater than pi essentially disables the constraint (without culling
    //! the computation). Non-fatal warnings will be logged if:
    //!   1. `path_axis_deviation_limit` is near or greater than pi, *OR*
    //!   2. `terminal_deviation_limit` is near or greater than pi.
    //!
    //! A fatal error will be logged if:
    //!   1. `terminal_deviation_limit` is negative,
    //!   2. `path_axis_deviation_limit` is negative,
    //!   3. `tool_frame_axis` is (nearly) zero, *OR*
    //!   4. `world_target_axis` is (nearly) zero.
    //!
    //! A non-fatal error will be logged if `path_axis_deviation_limit` is infeasible for the
    //! initial c-space position and `terminal_orientation_target`. In this case,
    //! `Results::status()` will return `INVALID_TARGET_SPECIFICATION`.
    //!
    //! NOTE: This condition will only be validated when the resulting `OrientationConstraint` is
    //!       used for trajectory optimization (i.e., it is used as input to
    //!       `planToTaskSpaceTarget()`).
    //!
    //! NOTE: `tool_frame_axis` and `world_target_axis` inputs will be normalized.
    //!
    //! NOTE: The `terminal_orientation_target` is specified in world frame coordinates
    //!       (i.e., relative to the base of the robot).
    static OrientationConstraint TerminalTargetAndPathAxis(
        const Rotation3 &terminal_orientation_target,
        const Eigen::Vector3d &tool_frame_axis,
        const Eigen::Vector3d &world_target_axis,
        const double *terminal_deviation_limit = nullptr,
        const double *path_axis_deviation_limit = nullptr);

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Variant of `OrientationConstraint` for "goalset" planning.
  //!
  //! For goalset planning, a set of `OrientationConstraint`s are considered concurrently. Each
  //! `OrientationConstraint` in the goalset must have the same mode (e.g., "full terminal target
  //! with free axis path constraint"), but may have different data for each
  //! `OrientationConstraint`.
  class CUMO_EXPORT OrientationConstraintGoalset {
   public:
    //! Create an `OrientationConstraintGoalset` such that no tool frame orientation constraints
    //! are active along the path *OR* at termination.
    static OrientationConstraintGoalset None();

    //! Create an `OrientationConstraintGoalset` such that the tool frame orientation along the
    //! path *AND* at termination are constrained to the tool frame orientation at the initial
    //! c-space position.
    //!
    //! See `OrientationConstraint::Constant()` for details.
    //!
    //! A fatal error will be logged if:
    //!   1. Any condition of `OrientationConstraint::Constant()` is not met.
    static OrientationConstraintGoalset Constant(const double *path_deviation_limit = nullptr,
                                                 const double *terminal_deviation_limit = nullptr);

    //! Create an `OrientationConstraintGoalset` such that tool frame `orientation_targets` are
    //! specified at termination, but no orientation constraints are active along the path.
    //!
    //! See `OrientationConstraint::TerminalTarget()` for details.
    //!
    //! A fatal error will be logged if:
    //!   1. Any condition of `OrientationConstraint::TerminalTarget()` is not met, *OR*
    //!   2. `orientation_targets` is empty.
    static OrientationConstraintGoalset TerminalTarget(
        const std::vector<Rotation3> &orientation_targets,
        const double *terminal_deviation_limit = nullptr);

    //! Create an `OrientationConstraintGoalset` such that tool frame `orientation_targets` are
    //! specified along the path *AND* at termination.
    //!
    //! See `OrientationConstraint::TerminalAndPathTarget()` for details.
    //!
    //! A fatal error will be logged if:
    //!   1. Any condition of `OrientationConstraint::TerminalAndPathTarget()` is not met, *OR*
    //!   2. `orientation_targets` is empty.
    //!
    //! NOTE: The feasibility of `path_deviation_limit` will only be validated when the resulting
    //!       `OrientationConstraintGoalset` is used for trajectory optimization (i.e., it is used
    //!       as input to `planToTaskSpaceGoalset()`).
    static OrientationConstraintGoalset TerminalAndPathTarget(
        const std::vector<Rotation3> &orientation_targets,
        const double *path_deviation_limit = nullptr,
        const double *terminal_deviation_limit = nullptr);

    //! Create an `OrientationConstraintGoalset` such that the terminal tool frame orientation is
    //! constrained to rotate about a "free axis", but no orientation constraints are active along
    //! the path.
    //!
    //! See `OrientationConstraint::TerminalAxis()` for details.
    //!
    //! A fatal error will be logged if:
    //!   1. Any condition of `OrientationConstraint::TerminalAxis()` is not met,
    //!   2. `tool_frame_axes` and `world_target_axes` do not have the same number of elements, *OR*
    //!   3. `tool_frame_axes` and `world_target_axes` are empty.
    static OrientationConstraintGoalset TerminalAxis(
        const std::vector<Eigen::Vector3d> &tool_frame_axes,
        const std::vector<Eigen::Vector3d> &world_target_axes,
        const double *terminal_axis_deviation_limit = nullptr);

    //! Create an `OrientationConstraintGoalset` such that the tool frame orientation is
    //! constrained to rotate about a "free axis" along the path *AND* at termination.
    //!
    //! See `OrientationConstraint::TerminalAndPathAxis()` for details.
    //!
    //! A fatal error will be logged if:
    //!   1. Any condition of `OrientationConstraint::TerminalAndPathAxis()` is not met,
    //!   2. `tool_frame_axes` and `world_target_axes` do not have the same number of elements, *OR*
    //!   3. `tool_frame_axes` and `world_target_axes` are empty.
    //!
    //! NOTE: The feasibility of `path_axis_deviation_limit` will only be validated when the
    //!       resulting `OrientationConstraintGoalset` is used for trajectory optimization (i.e., it
    //!       is used as input to `planToTaskSpaceGoalset()`).
    static OrientationConstraintGoalset TerminalAndPathAxis(
        const std::vector<Eigen::Vector3d> &tool_frame_axes,
        const std::vector<Eigen::Vector3d> &world_target_axes,
        const double *path_axis_deviation_limit = nullptr,
        const double *terminal_axis_deviation_limit = nullptr);

    //! Create an `OrientationConstraintGoalset` such that tool frame `terminal_orientation_targets`
    //! are specified at termination, *AND* the tool frame orientation is constrained to rotate
    //! about a "free axis" along the path.
    //!
    //! See `OrientationConstraint::TerminalTargetAndPathAxis()` for details.
    //!
    //! A fatal error will be logged if:
    //!   1. Any condition of `OrientationConstraint::TerminalTargetAndPathAxis()` is not met,
    //!   2. `terminal_orientation_targets`, `tool_frame_axes`, and `world_target_axes` do not have
    //!      the same number of elements, *OR*
    //!   3. `terminal_orientation_targets`, `tool_frame_axes`, and `world_target_axes` are empty.
    //!
    //! NOTE: The feasibility of `path_axis_deviation_limit` will only be validated when the
    //!       resulting `OrientationConstraintGoalset` is used for trajectory optimization (i.e., it
    //!       is used as input to `planToTaskSpaceGoalset()`).
    static OrientationConstraintGoalset TerminalTargetAndPathAxis(
        const std::vector<Rotation3> &terminal_orientation_targets,
        const std::vector<Eigen::Vector3d> &tool_frame_axes,
        const std::vector<Eigen::Vector3d> &world_target_axes,
        const double *terminal_deviation_limit = nullptr,
        const double *path_axis_deviation_limit = nullptr);

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Task-space targets restrict the position and (optionally) orientation of the tool frame at the
  //! termination of a trajectory and (optionally) along the path.
  struct CUMO_EXPORT TaskSpaceTarget {
    //! Create a task-space target.
    explicit TaskSpaceTarget(
        const TranslationConstraint &translation_constraint,
        const OrientationConstraint &orientation_constraint = OrientationConstraint::None());

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Variant of `TaskSpaceTarget` for "goalset" planning.
  //!
  //! For goalset planning, a set of pose constraints are considered concurrently. Each pose
  //! constraint in the goalset must have the same mode (e.g., "terminal target
  //! with linear path constraint and no orientation constraints") but may have different data for
  //! each constraint.
  struct CUMO_EXPORT TaskSpaceTargetGoalset {
    //! Create a task-space target goalset.
    //!
    //! A fatal error will be logged if:
    //!   1. The number of `translation_constraints` does not match the number of
    //!      `orientation_constraints`.
    explicit TaskSpaceTargetGoalset(const TranslationConstraintGoalset &translation_constraints,
                                    const OrientationConstraintGoalset &orientation_constraints =
                                        OrientationConstraintGoalset::None());

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! C-space targets fully restrict the c-space configuration at the termination of the trajectory
  //! while allowing (optional) task-space constraints along the path.
  class CUMO_EXPORT CSpaceTarget {
   public:
    //! Translation path constraints restrict the position of the origin of a tool frame along the
    //! path.
    class CUMO_EXPORT TranslationPathConstraint {
     public:
      //! Create a `TranslationPathConstraint` such that the position of the tool-frame is not
      //! restricted along the path.
      static TranslationPathConstraint None();

      //! Create a `TranslationPathConstraint` such that a linear translation constraint is active
      //! along the path.
      //!
      //! The linear path constraint is defined by the line **segment** between the origin of the
      //! tool frame at the initial c-space position and the origin of the tool frame at the
      //! terminal c-space target. This path constraint is satisfied if the origin of the tool frame
      //! is on this line segment between the initial and final tool positions at all points in the
      //! trajectory. It is considered a constraint violation if the origin of the tool frame
      //! "overshoots" either the initial or final tool positions.
      //!
      //! If the distance between the initial and final tool frames are nearly the same, the path
      //! constraint becomes a constraint on the distance from the initial and final positions in
      //! any direction.
      //!
      //! The optional parameter `path_deviation_limit` can be used to allow deviation from the
      //! nominal task-space line along the path. This limit specifies the maximum allowable
      //! deviation from the desired linear path.
      //!
      //! If `path_deviation_limit` is not input, then the deviation limit is assumed to be zero
      //! (i.e., it is desired that translation is always exactly coincident with the specified
      //! line).
      //!
      //! A fatal error will be logged if:
      //!   1. `path_deviation_limit` is negative.
      static TranslationPathConstraint Linear(const double *path_deviation_limit = nullptr);

      struct Impl;
      std::shared_ptr<Impl> impl;
    };

    //! Orientation path constraints restrict the orientation of a tool frame along the path.
    class CUMO_EXPORT OrientationPathConstraint {
     public:
      //! Create a `OrientationPathConstraint` such that the orientation of the tool frame is not
      //! restricted along the path.
      static OrientationPathConstraint None();

      //! Create an `OrientationPathConstraint` such that the tool frame orientation along the path
      //! is constant.
      //!
      //! If `use_terminal_orientation` is set to `true`, then the tool frame orientation
      //! corresponding to the terminal c-space target will be used to define the orientation
      //! target along the path. Otherwise, the tool frame orientation corresponding to the initial
      //! c-space position will be used.
      //!
      //! The optional parameter `path_deviation_limit` can be used to allow deviations from this
      //! constant orientation along the path. If input, the units are radians. These limits
      //! specify the maximum allowable deviation from the desired orientation.
      //!
      //! If the `path_deviation_limit` is *NOT* input, its value is set to the most restrictive
      //! feasible value. This feasible limit is determined by the angular distance between the tool
      //! frame orientations at the initial and terminal c-space positions. If the
      //! `path_deviation_limit` is input, it must be greater than or equal to this angular
      //! distance.
      //!
      //! In general, it is suggested that angular deviation limits are set to values less than pi.
      //! A limit that is near or greater than pi essentially disables the constraint (without
      //! culling the computation). Non-fatal warnings will be logged if:
      //!   - `path_deviation_limit` is near or greater than pi.
      //!
      //! A fatal error will be logged if `path_deviation_limit` is negative.
      //!
      //! A non-fatal error will be logged if `path_deviation_limit` is infeasible for the given
      //! initial c-space position and terminal c-space target. In this case, `Results::status()`
      //! will return `INVALID_TARGET_SPECIFICATION`.
      //!
      //! NOTE: This condition will only be validated when the resulting `OrientationPathConstraint`
      //!       is used for trajectory optimization (i.e., it is used as input to
      //!       `planToCSpaceTarget()`).
      static OrientationPathConstraint Constant(const double *path_deviation_limit = nullptr,
                                                bool use_terminal_orientation = true);

      //! Create an `OrientationPathConstraint` such that the tool frame orientation is constrained
      //! to rotate about a "free axis" along the path.
      //!
      //! The "free axis" for rotation is defined by a `tool_frame_axis` (specified in the tool
      //! frame coordinates) and a corresponding `world_target_axis` (specified in world frame
      //! coordinates).
      //!
      //! The optional parameter `path_axis_deviation_limit` can be used to allow deviation from
      //! the desired axis alignment along the path. If input, the units are radians.
      //!
      //! If the `path_axis_deviation_limit` is *NOT* input, its value is set to the most
      //! restrictive feasible value. This feasibility is determined by the tool frame orientations
      //! at the initial and terminal c-space positions. For each orientation, the angular distance
      //! (if any) from satisfying the constraint will be computed. The `path_axis_deviation_limit`
      //! will be set to the maximum of these angular distances. If the `path_axis_deviation_limit`
      //! is input, it must be greater than or equal to this minimum feasible deviation limit.
      //!
      //! In general, it is suggested that angular deviation limits are set to values less than pi.
      //! A limit that is near or greater than pi essentially disables the constraint (without
      //! culling the computation). Non-fatal warnings will be logged if:
      //!   - `path_axis_deviation_limit` near or greater than pi.
      //!
      //! A fatal error will be logged if:
      //!   1. `path_axis_deviation_limit` is negative,
      //!   2. `tool_frame_axis` is (nearly) zero, *OR*
      //!   3. `world_target_axis` is (nearly) zero.
      //!
      //! A non-fatal error will be logged if `path_axis_deviation_limit` is infeasible for the
      //! given initial c-space position and terminal c-space target. In this case,
      //! `Results::status()` will return `INVALID_TARGET_SPECIFICATION`.
      //!
      //! NOTE: This condition will only be validated when the resulting `OrientationPathConstraint`
      //!       is used for trajectory optimization (i.e., it is used as input to
      //!       `planToCSpaceTarget()`).
      //!
      //! NOTE: `tool_frame_axis` and `world_target_axis` inputs will be normalized.
      static OrientationPathConstraint Axis(const Eigen::Vector3d &tool_frame_axis,
                                            const Eigen::Vector3d &world_target_axis,
                                            const double *path_axis_deviation_limit = nullptr);

      struct Impl;
      std::shared_ptr<Impl> impl;
    };

    //! Create a c-space target.
    explicit CSpaceTarget(const Eigen::VectorXd &cspace_position_terminal_target,
                          const TranslationPathConstraint &translation_path_constraint =
                              TranslationPathConstraint::None(),
                          const OrientationPathConstraint &orientation_path_constraint =
                              OrientationPathConstraint::None());

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Results from a trajectory optimization.
  class CUMO_EXPORT Results {
   public:
    //! Indicate the success or failure of the trajectory optimization.
    enum class Status {
      //! Valid trajectory found.
      SUCCESS,

      //! Invalid initial c-space position.
      //!
      //! NOTE: This invalid state may be due to an incorrect number of c-space coordinates,
      //!       world-collision, self-collision, or c-space position limit violations. The
      //!       `RobotWorldInspector` can be used to check for world-collision and self-collision,
      //!       and `Kinematics::withinCSpaceLimits()` can be used to check c-space position limits.
      INVALID_INITIAL_CSPACE_POSITION,

      //! Invalid target c-space position.
      //!
      //! Only applicable when planning to c-space targets.
      //!
      //! NOTE: This invalid state may be due to an incorrect number of c-space coordinates,
      //!       world-collision, self-collision, or c-space position limit violations. The
      //!       `RobotWorldInspector` can be used to check for world-collision and self-collision,
      //!       and `Kinematics::withinCSpaceLimits()` can be used to check c-space position limits.
      INVALID_TARGET_CSPACE_POSITION,

      //! The c-space or task-space target specification is invalid.
      INVALID_TARGET_SPECIFICATION,

      //! The inverse kinematics solver failed to find a valid solution.
      INVERSE_KINEMATICS_FAILURE,

      //! Geometric path planning found no collision-free path.
      //!
      //! This failure mode can only occur when no path constraints are active.
      GEOMETRIC_PLANNING_FAILURE,

      //! Trajectory optimization attempts failed to produce a valid trajectory.
      TRAJECTORY_OPTIMIZATION_FAILURE
    };

    virtual ~Results() = default;

    //! Return the `Status` from the trajectory optimization.
    [[nodiscard]] virtual Status status() const = 0;

    //! If `status()` returns `SUCCESS`, then `trajectory()` returns a valid `Trajectory`.
    //! Otherwise, `nullptr` will be returned.
    [[nodiscard]] virtual std::unique_ptr<Trajectory> trajectory() const = 0;

    //! Return the index of the target selected for the trajectory's terminal constraint.
    //!
    //! For goalset planning (e.g., `planToTaskSpaceGoalset()`) this represents an index into the
    //! constraint vectors used to create the goalset (e.g., `TaskSpaceTargetGoalset`).
    //!
    //! For single target planning (e.g., `planToTaskSpaceTarget()` and `planToCSpaceTarget()`),
    //! zero will be returned for successful trajectory optimization.
    //!
    //! In all cases, -1 will be returned if trajectory optimization is unsuccessful.
    [[nodiscard]] virtual int targetIndex() const = 0;
  };

  //! Attempt to find a trajectory from `initial_cspace_position` to `task_space_target`.
  [[nodiscard]] virtual std::unique_ptr<Results> planToTaskSpaceTarget(
      const Eigen::VectorXd &initial_cspace_position,
      const TaskSpaceTarget &task_space_target) const = 0;

  //! Attempt to find a trajectory from `initial_cspace_position` to `task_space_target_goalset`.
  [[nodiscard]] virtual std::unique_ptr<Results> planToTaskSpaceGoalset(
      const Eigen::VectorXd &initial_cspace_position,
      const TaskSpaceTargetGoalset &task_space_target_goalset) const = 0;

  //! Attempt to find a trajectory from `initial_cspace_position` to `cspace_target`.
  [[nodiscard]] virtual std::unique_ptr<Results> planToCSpaceTarget(
      const Eigen::VectorXd &initial_cspace_position,
      const CSpaceTarget &cspace_target) const = 0;
};

//! Create a `TrajectoryOptimizer` with the given `config`.
CUMO_EXPORT std::unique_ptr<TrajectoryOptimizer> CreateTrajectoryOptimizer(
    const TrajectoryOptimizerConfig &config);

}  // namespace cumotion
