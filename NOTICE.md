PyNeurale
Copyright (c) 2026 Xingchen RAN

PyNeurale is licensed under the MIT License. See LICENSE for its terms.
Third-party components retain their own terms. Their bundled license and
notice texts are collected in LICENSES_bundled.txt.

The native bindings use pybind11. With experiment presentation enabled,
neurale._native links GLFW, FreeType, and HarfBuzz. These dependencies may
come from system packages or the fallback source versions identified in
LICENSES_bundled.txt. Portions of this software are based on work by the
FreeType Team.

An MKL-enabled build statically links Intel oneMKL. Release wheels also
bundle the Intel OpenMP runtime.

CUDA-enabled Linux wheels bundle the CUDA runtime (`libcudart`), governed by
the NVIDIA CUDA Toolkit license. The NVIDIA driver is not bundled.
https://docs.nvidia.com/cuda/eula/

Linux wheel repair may bundle the GLVND OpenGL dispatcher libraries; Windows
wheel repair may bundle Microsoft Visual C++ runtime libraries. Their
redistribution terms remain applicable to any libraries actually bundled.
https://github.com/NVIDIA/libglvnd/blob/master/README.md
https://learn.microsoft.com/en-us/visualstudio/releases/2026/redistribution
