PyNeurale
Copyright (c) 2026 Xingchen RAN

PyNeurale is licensed under the MIT License. See LICENSE for its terms.
Third-party components retain their own copyright and license terms.

## Optional native dependencies

When experiment presentation is enabled, neurale._native links GLFW,
FreeType, and HarfBuzz. The build may use system packages or build the
fallback versions below from source; system package versions may differ.

- GLFW (fallback version 3.4): zlib/libpng license.
  https://github.com/glfw/glfw/blob/3.4/LICENSE.md
- FreeType (fallback version 2.13.3): used under the FreeType License (FTL).
  Portions of this software are based on work by the FreeType Team.
  https://github.com/freetype/freetype/blob/VER-2-13-3/docs/FTL.TXT
- HarfBuzz (fallback version 14.3.1): primarily the Old MIT license;
  see its COPYING file and any applicable subdirectory notices.
  https://github.com/harfbuzz/harfbuzz/blob/14.3.1/COPYING

An MKL-enabled native build statically links Intel oneMKL under the Intel
Simplified Software License. Release wheels also bundle the Intel OpenMP runtime.
https://www.intel.com/content/www/us/en/developer/articles/tool/onemkl-license-faq.html

CUDA-enabled Linux wheels bundle the CUDA runtime (`libcudart`), governed by
the NVIDIA CUDA Toolkit license. The NVIDIA driver is not bundled.
https://docs.nvidia.com/cuda/eula/

Linux wheel repair may bundle the GLVND OpenGL dispatcher libraries; Windows
wheel repair may bundle Microsoft Visual C++ runtime libraries. These retain
their respective third-party terms.
https://github.com/NVIDIA/libglvnd/blob/master/README.md
https://learn.microsoft.com/en-us/visualstudio/releases/2026/redistribution
