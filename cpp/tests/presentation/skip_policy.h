/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// Whether a display-dependent presentation test may skip.
///
/// The `--window` cases need a real GLFW/OpenGL context and, for text, a font
/// on disk. When one is missing they return 77, which CTest turns into a SKIP.
/// That is right for a developer machine with no display; it is wrong for CI,
/// where a green run would then prove only that the environment was unusable --
/// every rendering, timing, and glyph assertion silently not executed, and the
/// job passing anyway.
///
/// Setting NEURALE_PRESENTATION_REQUIRE_DISPLAY makes the skip a failure, so a
/// CI job that is *supposed* to have xvfb and fonts fails loudly when it does
/// not, instead of reporting success it did not earn.

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace neurale::presentation_test
{

/// Whether the environment demands that display-dependent cases actually run.
[[nodiscard]] inline bool display_required() noexcept
{
    const auto* value = std::getenv("NEURALE_PRESENTATION_REQUIRE_DISPLAY");
    if (value == nullptr)
    {
        return false;
    }
    const std::string_view text(value);
    return !text.empty() && text != "0";
}

/// The process exit code for a display-dependent case that cannot run.
///
/// 77 (CTest SKIP) normally; 1, with an explanation, when the environment said
/// the case had to run. @p reason names what was missing, so a failing CI job
/// says which of display, GL, or font was absent rather than only that
/// something was.
[[nodiscard]] inline int skip_or_fail(std::string_view reason) noexcept
{
    if (!display_required())
    {
        std::cerr << "skipping display-dependent presentation test: " << reason << '\n';
        return 77;
    }
    std::cerr << "NEURALE_PRESENTATION_REQUIRE_DISPLAY is set, but this display-dependent "
                 "presentation test cannot run: "
              << reason << '\n';
    return 1;
}

} // namespace neurale::presentation_test
