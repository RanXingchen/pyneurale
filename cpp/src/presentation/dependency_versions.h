// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#pragma once

namespace neurale::experiment_presentation
{

struct DependencyVersions
{
    int glfw_major{};
    int glfw_minor{};
    int glfw_revision{};
    int freetype_major{};
    int freetype_minor{};
    int freetype_patch{};
    int harfbuzz_major{};
    int harfbuzz_minor{};
    int harfbuzz_micro{};
    int minimum_opengl_major{3};
    int minimum_opengl_minor{3};
};

[[nodiscard]] DependencyVersions dependency_versions() noexcept;

} // namespace neurale::experiment_presentation
