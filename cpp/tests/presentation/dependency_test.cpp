// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "dependency_versions.h"

int main()
{
    const auto versions = neurale::experiment_presentation::dependency_versions();
    if (versions.glfw_major <= 0 || versions.freetype_major <= 0 || versions.harfbuzz_major <= 0)
    {
        return 1;
    }
    if (versions.minimum_opengl_major != 3 || versions.minimum_opengl_minor != 3)
    {
        return 2;
    }
    return 0;
}
