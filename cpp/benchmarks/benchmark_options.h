/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// Command-line value parsing shared by the native benchmarks.
///
/// Which options a benchmark takes is its own -- the names, the defaults, and
/// the cross-checks between them all describe one benchmark's shape, and a
/// shared option table would only make three unlike benchmarks answer to one.
/// Turning an argument into a count is not: every one of them wants a decimal
/// that consumed the whole argument and did not overflow, and refuses anything
/// else before it starts measuring.

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace neurale::benchmark
{

/// Parse @p value as a decimal count, or throw `std::invalid_argument`.
///
/// @param value      The argument text; must be consumed in full.
/// @param message    What to say when it is not a count -- the caller's own
///                   wording, so the error names the option that was wrong.
/// @param allow_zero Whether zero is a count this option accepts. Most mean
///                   "how many", where zero would measure nothing.
[[nodiscard]] inline std::size_t parse_size(const char* value, const char* message,
                                            bool allow_zero = false)
{
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || (!allow_zero && parsed == 0) ||
        parsed > (std::numeric_limits<std::size_t>::max)())
    {
        throw std::invalid_argument(message);
    }
    return static_cast<std::size_t>(parsed);
}

} // namespace neurale::benchmark
