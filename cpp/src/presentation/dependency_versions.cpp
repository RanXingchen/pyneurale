// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <ft2build.h>
#include FT_FREETYPE_H
#include <hb.h>

#include "dependency_versions.h"

namespace neurale::experiment_presentation
{

DependencyVersions dependency_versions() noexcept
{
    return DependencyVersions{
        .glfw_major = GLFW_VERSION_MAJOR,
        .glfw_minor = GLFW_VERSION_MINOR,
        .glfw_revision = GLFW_VERSION_REVISION,
        .freetype_major = FREETYPE_MAJOR,
        .freetype_minor = FREETYPE_MINOR,
        .freetype_patch = FREETYPE_PATCH,
        .harfbuzz_major = HB_VERSION_MAJOR,
        .harfbuzz_minor = HB_VERSION_MINOR,
        .harfbuzz_micro = HB_VERSION_MICRO,
    };
}

} // namespace neurale::experiment_presentation
