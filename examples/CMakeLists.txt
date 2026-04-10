# SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES.
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

cmake_minimum_required(VERSION 3.21)

project(cumotion_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA.
find_package(CUDAToolkit 12.2 REQUIRED)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH
  "Filepath to nvcc compiler, default installation path provided")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Enable Blackwell if supported by CUDA toolkit.
  set(ENABLE_BLACKWELL OFF)
  if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 12.8)
    set(ENABLE_BLACKWELL ON)
  endif()
  message(STATUS "CUDA Toolkit VERSION: ${CUDAToolkit_VERSION}. Supports Blackwell? ${ENABLE_BLACKWELL}")
  if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin
    if(${ENABLE_BLACKWELL})
      list(APPEND CMAKE_CUDA_ARCHITECTURES 110)  # Jetson Thor (Blackwell)
      list(APPEND CMAKE_CUDA_ARCHITECTURES 121)  # DGX Spark (GB10)
    endif()
  else()
    set(CMAKE_CUDA_ARCHITECTURES "75;86;89")  # Turing (TU10x), Ampere (GA10x), and Ada (AD10x)
    if(${ENABLE_BLACKWELL})
      list(APPEND CMAKE_CUDA_ARCHITECTURES 120)  # Blackwell (GB20x)
    endif()
  endif()
endif()
enable_language(CUDA)

if(MSVC)
  # Required to use math constants (e.g., `M_PI`) on Windows.
  add_compile_definitions(_USE_MATH_DEFINES)
endif()

# Find dependencies.
find_package(Eigen3 REQUIRED)

# Find cumotion if this project is the top-level project.
if(PROJECT_IS_TOP_LEVEL)
  find_package(cumotion REQUIRED)
  if(cumotion_FOUND)
    message(STATUS "cuMotion found at ${cumotion_DIR}")
  else()
    message(STATUS "cuMotion not found!")
  endif()
endif()

# Get the location of the content/ directory.
if(NOT DEFINED CONTENT_DIR)
  # Find
  #   content/
  # relative to
  #   lib/cmake/cumotion/cumotionConfig.cmake
  set(CONTENT_DIR "${cumotion_DIR}/../../../content")
endif()
cmake_path(ABSOLUTE_PATH CONTENT_DIR NORMALIZE)
message(STATUS "cuMotion content directory: ${CONTENT_DIR}")

# Build the examples.
macro(add_example TARGET SRC)
  add_executable(${TARGET} ${SRC})
  target_link_libraries(${TARGET} PRIVATE cumotion::cumotion)
  target_compile_definitions(${TARGET} PRIVATE CONTENT_DIR="${CONTENT_DIR}")
  target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  if(WIN32)
    # On Windows, it's necessary to copy the cuMotion DLL to the same directory as the executable.
    add_custom_command(TARGET ${TARGET} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:${TARGET}>
        $<TARGET_FILE_DIR:${TARGET}>
      COMMAND_EXPAND_LISTS
    )
  endif()
endmacro()

# Kinematics example
add_example(kinematics_example src/kinematics_example.cpp)

# Franka collision-free IK example
add_example(franka_collision_free_ik_solver_example src/trajectory_optimizer_examples/franka_collision_free_ik_solver_example.cpp)

# Franka trajectory optimization example (with no obstacles)
add_example(franka_trajectory_optimizer_example src/trajectory_optimizer_examples/franka_trajectory_optimizer_example.cpp)

# Franka collision-free trajectory optimization example (with cuboid obstacles)
add_example(franka_trajectory_optimizer_obstacle_example src/trajectory_optimizer_examples/franka_trajectory_optimizer_obstacle_example.cpp)
