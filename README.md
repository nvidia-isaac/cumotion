<p align="left">
<img src="https://media.githubusercontent.com/media/nvidia-isaac/cumotion/084b9bd853dd8c5bea2ac5839cf09e94e19f9dd9/_static/images/m48-robot-manufacturing-256px-grn.png" width="192" /></p>

# cuMotion: GPU-Accelerated Motion Generation for Robotics

<h4>
    <a href="https://nvidia-isaac.github.io/cumotion/">Documentation</a> [
    <a href="https://nvidia-isaac.github.io/cumotion/getting_started.html">Getting Started</a> |
    <a href="https://nvidia-isaac.github.io/cumotion/tutorials.html">Tutorials</a> |
    <a href="https://nvidia-isaac.github.io/cumotion/api/cpp_api.html">C++ API</a> |
    <a href="https://nvidia-isaac.github.io/cumotion/api/python_api.html">Python API</a> |
    <a href="https://nvidia-isaac.github.io/cumotion/release_notes.html">Release Notes</a> ]
</h4>

NVIDIA cuMotion is a high-performance motion generation library for robotics, focused mainly on manipulation.
GPU acceleration is leveraged throughout where beneficial for performance.

Capabilities include:

* Kinematics and inverse kinematics (IK)
* Collision-aware IK
* Collision-aware graph-based path planning
* Collision-aware trajectory optimization and end-to-end motion generation
* Reactive control via the mathematical framework of [RMPflow](https://arxiv.org/abs/1811.07049)
* Time-optimal (collision-unaware) trajectory generation for paths given by any number of moves in configuration
  space (joint space) and/or task space
* Collision sphere generation, for approximating a closed mesh by a set of spheres
* Robot segmentation, for removing the contribution of a robot arm from one or more depth image streams

Robots with any number of degrees of freedom are supported.

cuMotion is implemented in C++ and provides a complete set of Python bindings.  Although developed as a
standalone library, cuMotion will also soon serve as the motion generation backend for
[Isaac ROS cuMotion](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/index.html)
and the manipulation
[reference workflows](https://nvidia-isaac-ros.github.io/reference_workflows/isaac_for_manipulation/index.html)
in [Isaac ROS](https://nvidia-isaac-ros.github.io/index.html).

cuMotion is descended from two libraries:

1. [Lula](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.robot_motion.lula/docs/index.html)
   was released for a number of years as part of Isaac Sim.  It has been entirely subsumed by cuMotion,
   and Python users of Lula will find that ``import cumotion as lula`` will suffice for running most client code with
   only a few modifications.
2. [cuRobo](https://curobo.org) was developed by NVIDIA Research and continues to be maintained as a library for
   robotics researchers.  cuMotion incorporates optimized and hardened implementations of algorithms first introduced
   in cuRobo, including those for collision-aware IK and trajectory optimization.

See the [documentation](https://nvidia-isaac.github.io/cumotion/) for a full overview of features.

# System Requirements

The current release of the cuMotion library is provided as a set of shared object files and Python wheel files
for both Linux and Windows on x86-based computers, as well Jetson Orin, Jetson Thor, and DGX Spark.

* Under Linux, the library is compatible with GCC 11.4 and later (corresponding to `libstdc++` symbol version
  `GLIBCXX_3.4.30`).  It has been tested on Ubuntu 22.04 and 24.04.
* Under Windows, the library is compatible with the Visual Studio 2019 toolchain and later.
* On both platforms, Python wheel files are provided for all
  [actively-supported Python versions](https://devguide.python.org/versions/) at the time of release, currently
  consisting of Python 3.10 through 3.14, inclusive.

The software has been tested on Jetson AGX Orin and Jetson AGX Thor but is expected to work on other Jetson
Orin configurations (e.g., Jetson Orin NX).  The Jetson Thor (CUDA 13.0) package has also been optimized for
and tested on DGX Spark.

On x86-based computers, an NVIDIA GPU of the Turing generation (e.g., GeForce RTX 2080) or later is required.
For optimal performance on Blackwell GPUs (e.g., RTX PRO 6000, GeForce RTX 5080), the CUDA 13 variant of the
cuMotion package is recommended.  See the
[CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/#cuda-driver)
for details on the minimum GPU driver version required for a given CUDA version.

## Linux

| Platform                              | CUDA | Tested OS                | Download |
|---------------------------------------|------|--------------------------|----------|
| Jetson Orin                           | 12.6 | JetPack 6.2.1            | [Latest Release](https://github.com/nvidia-isaac/cumotion/releases/download/v1.1.0/cumotion-1.1.0-cuda12.6-aarch64.tar.gz) |
| Jetson Thor <br>DGX Spark             | 13.0 | JetPack 7.0 <br>DGX OS 7 | [Latest Release](https://github.com/nvidia-isaac/cumotion/releases/download/v1.1.0/cumotion-1.1.0-cuda13.0-aarch64.tar.gz) |
| x86_64 + NVIDIA GPU (Turing or later) | 12.6 | Ubuntu 22.04             | [Latest Release](https://github.com/nvidia-isaac/cumotion/releases/download/v1.1.0/cumotion-1.1.0-cuda12.6-x86_64.tar.gz) |
| x86_64 + NVIDIA GPU (Turing or later) | 13.0 | Ubuntu 24.04             | [Latest Release](https://github.com/nvidia-isaac/cumotion/releases/download/v1.1.0/cumotion-1.1.0-cuda13.0-x86_64.tar.gz) |

## Microsoft Windows

| Platform                              | CUDA | Tested OS                | Download |
|---------------------------------------|------|--------------------------|----------|
| x86_64 + NVIDIA GPU (Turing or later) | 13.0 | Microsoft Windows 10/11  | [Latest Release](https://github.com/nvidia-isaac/cumotion/releases/download/v1.1.0/cumotion-1.1.0-cuda13.0-windows-x86_64.zip) |

# Installation

Download the cuMotion package for your desired platform from
[Releases](https://github.com/nvidia-isaac/cumotion/releases) and extract the contents,
for example on Linux,

```
tar xzvf cumotion-<version>-<cuda_version>-<arch>.tar.gz && \
    cd cumotion-<version>-<python_version>-<cuda_version>-<arch>
```

See [Getting Started](https://nvidia-isaac.github.io/cumotion/getting_started.html) for more details.

___
Copyright &copy; 2019-2026 NVIDIA Corporation & Affiliates
