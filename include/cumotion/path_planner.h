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
//! @brief Public interface for global path planning.

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"

#include "cumotion/cumotion_export.h"
#include "cumotion/robot_description.h"
#include "cumotion/world.h"

namespace cumotion {

//! Configuration parameters for a path planner.
class CUMO_EXPORT PathPlannerConfig {
 public:
  virtual ~PathPlannerConfig() = default;

  //! Specify the value for a given parameter.
  //!
  //! The required `ParamValue` constructor for each param is detailed in the
  //! documentation for `setParam()`.
  struct CUMO_EXPORT ParamValue {
    //! Create `ParamValue` from `int`.
    ParamValue(int value);  // NOLINT Allow implicit conversion
    //! Create `ParamValue` from `double`.
    ParamValue(double value);  // NOLINT Allow implicit conversion

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Set the value of the parameter.
  //!
  //! `param_name` identifies the parameter to update, and `value` supplies its new value.
  //!
  //! Return `true` if the parameter is updated successfully. Return `false` and log a non-fatal
  //! error if `param_name` is unknown, `value` does not hold the documented type for `param_name`,
  //! or `value` violates a parameter constraint.
  //!
  //! Each call validates the full set of parameters, so a constraint that relates parameters is
  //! checked against the current values of the others. For example, with the default parameters,
  //! `step_size` cannot be lowered to 0.001 until `min_extension_length` is lowered to 0.2 or less.
  //!
  //! The following parameters can be set for `PathPlanner`:
  //!
  //! `random_seed` [`int`]
  //!   - Used to initialize random sampling.
  //!   - Any value is accepted.
  //!   - Default: 12345.
  //!
  //! `max_num_segments_per_forest` [`int`]
  //!   - Maximum number of segments in a single planner forest.  Internally, this parameter value
  //!     may be inflated to ensure that at least one iteration of the planner is always possible.
  //!   - Must be positive.
  //!   - The default value of 20000 is recommended for RTX 5090 GPUs, and alternative tunings for
  //!     different architectures are provided in the `content/nvidia/shared/` directory.
  //!
  //! `num_validation_checks_per_extension` [`int`]
  //!   - Number of validation checks per extension.  Increasing this parameter will result in
  //!     launching a smaller number of extensions in parallel that are individually longer.
  //!   - In practice, increasing this parameter typically results in fewer iterations to solve, at
  //!     the cost of longer output paths.
  //!   - The length of each extension is up to `step_size * num_validation_checks_per_extension`.
  //!   - Must be positive, and large enough that `num_validation_checks_per_extension * step_size`
  //!     is at least `min_extension_length`.
  //!   - The default value of 200 (paired with the default `step_size` of `0.01`) is a good
  //!     starting point for most use-cases.
  //!
  //! `step_size` [`double`]
  //!   - Step size between validated c-space configurations in the returned path.  It is assumed
  //!     that `step_size` is sufficiently small such that in a contiguous sequence of
  //!     valid nodes with the distance between neighbors no larger than `step_size`, the continuous
  //!     path connecting the nodes is also valid.
  //!   - The length of each extension is up to `step_size * num_validation_checks_per_extension`.
  //!   - Must be positive, and large enough that `num_validation_checks_per_extension * step_size`
  //!     is at least `min_extension_length`.
  //!   - The default value of 0.01 is recommended for most use-cases.
  //!
  //! `num_spheres_per_validation_iteration` [`int`]
  //!   - Number of sphere collision checks per iteration.  This value is used to determine the
  //!     total amount of validation work (in terms of individual robot sphere collision checks)
  //!     that will be performed in each planner iteration.  I.e., robots with more collision
  //!     spheres have a smaller number of parallel extension validations.
  //!   - Internally, a minimum number of parallel extensions is required for efficient path
  //!     planning. This minimum creates an effective lower bound on
  //!     `num_spheres_per_validation_iteration`; decreasing `num_spheres_per_validation_iteration`
  //!     below this lower bound will have no effect.
  //!   - NOTE: This parameter is intended to account for the difference in available resources
  //!     between different GPU architectures.
  //!   - Must be positive.
  //!   - The default value of 600'000 is recommended for RTX 5090 GPUs, and alternative tunings for
  //!     different architectures are provided in the `content/nvidia/shared/` directory.
  //!
  //! `min_extension_length` [`double`]
  //!   - Minimum length in c-space for an extension to be retained.
  //!   - It is observed in practice that allowing short extensions (on the order of 1 to 10 times
  //!     `step_size`) corresponds to an increase in the number of total iterations required to find
  //!     a solution.
  //!   - Must be positive, and less than or equal to
  //!     `num_validation_checks_per_extension * step_size`.
  //!   - The default value of 0.3 is recommended for most use-cases, paired with the default
  //!     extension length of `step_size * num_validation_checks_per_extension = 0.01 * 200 = 2.0`.
  //!
  //! `num_shortcut_iterations` [`int`]
  //!   - Number of shortcutting iterations per path.  The number of shortcutting iterations hits
  //!     diminishing returns based on the difficulty of the environment.  I.e., in environments
  //!     with long multi-segment paths, iterative shortcutting will yield worthwhile incremental
  //!     improvements for more iterations.
  //!   - As the number of paths to return increases, the shortcutting stage will scale linearly
  //!     in time, while the time to find the requested number of solutions will scale sublinearly.
  //!     I.e., when returning upwards of 50 paths, the shortcutting stage will dominate solve time.
  //!   - A value of 0 will disable shortcutting.
  //!   - Must be non-negative.
  //!   - The default value of 10 is recommended for most use-cases.
  //!
  //! `max_iterations` [`int`]
  //!   - Maximum number of iterations executed by the planner.  Planning always stops once this
  //!     many iterations have been executed, whether or not the requested number of paths have
  //!     been found.
  //!   - If the planner stops due to this parameter without finding any path, `Results` will report
  //!     that `pathFound()` is `false` and `numSolutions()` is zero.
  //!   - If the planner stops due to this parameter having found fewer paths than requested, it
  //!     will re-use the paths that it found in order to return the requested number of paths.
  //!   - Must be positive.
  //!   - The default value of 2000 is recommended for RTX 5090 GPUs, where the per-iteration time
  //!     for robots with less than 100 collision spheres is typically around 0.5 ms.  It has been
  //!     observed in practice that only planning problems which require navigating tight spaces
  //!     require more than 10 iterations to solve.
  //!
  //! `post_initial_solve_maximum_iterations` [`int`]
  //!   - Maximum number of iterations executed after the first solution is found.  Once the
  //!     first valid path is found, planning runs at most this many additional iterations before
  //!     stopping, unless the requested number of paths are collected sooner.
  //!   - If the planner stops early due to this parameter, it will re-use solutions multiple times
  //!     with different shortcutting seeds in order to return the requested number of paths.
  //!   - Must be non-negative.
  //!   - The default value of 20 is recommended for most use-cases.
  [[nodiscard]] virtual bool setParam(const std::string &param_name, ParamValue value) = 0;
};

//! Load a set of `PathPlanner` configuration parameters from file.
//!
//! These parameters are combined with `robot_description` and `world_view` to create a
//! configuration for path planning.
//!
//! `path_planner_config_file` specifies the YAML file to load, `robot_description` supplies the
//! robot model, and `world_view` selects the world used for collision checking.
//!
//! The YAML file is a map providing partial overrides of the default parameter values, keyed by
//! the parameter names documented in `PathPlannerConfig::setParam()`.  Parameters not provided in
//! the YAML file retain their default values.
//!
//! A configuration will *NOT* be created if:
//!   1. `path_planner_config_file` path is invalid or empty,
//!   2. `path_planner_config_file` points to an invalid YAML file,
//!   3. `path_planner_config_file` does not specify the supported format
//!      (`format: cumotion_path_planner` and `format_version: 1.0`),
//!   4. `path_planner_config_file` contains invalid parameters,
//!   5. `path_planner_config_file` contains duplicate keys, *OR*
//!   6. `robot_description` has zero c-space coordinates.
//!
//! In any of these cases, a non-fatal error will be logged and a `nullptr` will be returned.
CUMO_EXPORT std::unique_ptr<PathPlannerConfig> CreatePathPlannerConfigFromFile(
    const std::filesystem::path &path_planner_config_file,
    const RobotDescription &robot_description,
    const WorldViewHandle &world_view);

//! Use default parameters to create a configuration for path planning.
//!
//! `robot_description` supplies the robot model, and `world_view` selects the world used for
//! collision checking.
//!
//! A non-fatal error will be logged, and `nullptr` returned, if `robot_description` has zero
//! c-space coordinates.
//!
//! See `PathPlannerConfig::setParam()` for the default parameter values.
//!
//! NOTE: The default `PathPlanner` params are tuned for an RTX 5090 GPU.  Some parameters are
//!       intended to be adjusted for different GPU architectures.  Alternative tunings for other
//!       architectures may be found in the `content/nvidia/shared/` directory.
CUMO_EXPORT std::unique_ptr<PathPlannerConfig> CreateDefaultPathPlannerConfig(
    const RobotDescription &robot_description,
    const WorldViewHandle &world_view);

//! Interface for using the path planner to generate collision-free paths for a robot.
class CUMO_EXPORT PathPlanner {
 public:
  virtual ~PathPlanner() = default;

  //! Collision-free path composed of linear connections between waypoints in c-space.
  class CUMO_EXPORT Solution {
   public:
    virtual ~Solution() = default;

    //! Return a minimal representation of a collision-free path.
    //!
    //! Each vector represents a waypoint in c-space, where successive waypoints can be linearly
    //! interpolated in c-space to generate a collision-free path.
    [[nodiscard]] virtual std::vector<Eigen::VectorXd> path() const = 0;

    //! Return an interpolation of `path()` such that successive waypoints are (in general) one
    //! `step_size` apart from each other (where distance is defined as L2-norm in c-space).
    //!
    //! NOTE: Each straight segment from `path()` is interpolated individually, where the distance
    //!       between the last two returned waypoints for each segment will, in general, be less
    //!       than one `step_size` from each other.
    //!
    //! An empty vector will be returned unless `generate_interpolated_path` is set to `true` when
    //! generating a path plan.
    [[nodiscard]] virtual std::vector<Eigen::VectorXd> interpolatedPath() const = 0;

    //! Return the c-space L2-norm length of `path()`.
    [[nodiscard]] virtual double pathLength() const = 0;

    //! Return the index into the `cspace_targets` passed to `planToCSpaceTargets()` of the target
    //! reached by this solution.
    //!
    //! I.e., the final waypoint of `path()` is the c-space target `cspace_targets[targetIndex()]`.
    [[nodiscard]] virtual int targetIndex() const = 0;
  };

  //! Path planning results, combining a set of valid solutions.
  class CUMO_EXPORT Results {
   public:
    virtual ~Results() = default;

    //! Indicate whether a collision-free path was found.
    //!
    //! If `false`, `numSolutions()` will be zero.
    [[nodiscard]] virtual bool pathFound() const = 0;

    //! Return the number of collision-free solutions.
    [[nodiscard]] virtual int numSolutions() const = 0;

    //! Return the collision-free solution corresponding to `solution_index`.
    //!
    //! Solutions are sorted shortest-first, i.e., in ascending order of `Solution::pathLength()`.
    //!
    //! A fatal error will be logged if:
    //!   1. `solution_index` < 0, *OR*
    //!   2. `solution_index` >= `numSolutions()`.
    [[nodiscard]] virtual std::shared_ptr<const Solution> solution(int solution_index) const = 0;
  };

  //! Compute a set of collision-free paths from `initial_cspace_position` to at least one of the
  //! targets in `cspace_targets` and return them in ascending order of path length.
  //!
  //! `num_paths_to_return` is the number of solutions to return.  The number of returned paths will
  //! always be either `num_paths_to_return` exactly, or zero if no paths are found.  In general,
  //! the returned paths will be mutually distinct from each other.  There are two exceptions:
  //!   1. The planner finds direct collision-free connections between `initial_cspace_position`
  //!      and one or more of the targets in `cspace_targets`.  In this case, only direct connection
  //!      paths will be returned, duplicating them as needed to fill the return slots.
  //!   2. The planner stops before `num_paths_to_return` paths are found, for example because
  //!      `max_iterations` is reached.  In this case, the paths that were found are re-used to fill
  //!      the remaining return slots.  Re-used paths are shortcut with distinct seeds, so they are
  //!      exact duplicates of each other only when shortcutting is disabled.
  //!
  //! If `generate_interpolated_path` is set to `true`, `Solution::interpolatedPath()` will be
  //! populated for each returned `PathPlanner::Solution`.
  //!
  //! A non-fatal error will be logged and a planning failure (i.e., `Results` for which
  //! `pathFound()` is `false`) returned, if `initial_cspace_position` or *ALL* elements of
  //! `cspace_targets` are invalid c-space positions by:
  //!   1. Violating c-space position limits, *OR*
  //!   2. Being in world- or self- collision.
  //! Each invalid element of `cspace_targets` will be removed prior to planning with a warning.
  //!
  //! A fatal error will be logged if:
  //!   1. `cspace_targets.size()` is outside the range `[1, 1024]`, *OR*
  //!   2. `initial_cspace_position` or any element of `cspace_targets` has a size different from
  //!      the robot description used to create `PathPlannerConfig`, *OR*
  //!   3. `num_paths_to_return` is not positive.
  [[nodiscard]] virtual std::unique_ptr<Results> planToCSpaceTargets(
      const Eigen::VectorXd &initial_cspace_position,
      const std::vector<Eigen::VectorXd> &cspace_targets,
      int num_paths_to_return = 1,
      bool generate_interpolated_path = false) = 0;

  //! Reset all internal state so that subsequent planning calls produce the same results as they
  //! did immediately after construction, given the same inputs.
  virtual void reset() = 0;
};

//! Create a `PathPlanner` with the given `config`.
CUMO_EXPORT std::unique_ptr<PathPlanner> CreatePathPlanner(const PathPlannerConfig &config);

}  // namespace cumotion
