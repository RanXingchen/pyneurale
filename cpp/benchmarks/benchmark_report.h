/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The reporting every native benchmark does identically.
///
/// A benchmark's subject is its own -- what it drives, what it times, which
/// regimes it sweeps. What it does with the numbers afterwards is not: they all
/// print one JSON Lines record per case, and they all reduce a vector of
/// nanosecond durations to the same six-number summary. Those two things are
/// properties of the output format that `docs/development/` publishes and that
/// `benchmarks/_regression_policy.py` parses, not of any one benchmark, and a
/// copy per benchmark is a chance for one of them to print a record the policy
/// reads differently.
///
/// Nothing here is called from a timed region: a benchmark collects durations
/// first and reports after, so these may allocate and may touch `std::cout`.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <span>
#include <string_view>
#include <vector>

namespace neurale::benchmark
{

/// Write @p value as a JSON string, escaping what RFC 8259 requires.
///
/// Control characters below 0x20 become `\u00XX` rather than being dropped or
/// passed through: a record carrying one is still a record a parser must be
/// able to read, and the fields here include a reconstructed command line.
inline void emit_json_string(std::ostream& stream, std::string_view value)
{
    stream << '"';
    for (const unsigned char character : value)
    {
        switch (character)
        {
        case '"':
            stream << "\\\"";
            break;
        case '\\':
            stream << "\\\\";
            break;
        case '\n':
            stream << "\\n";
            break;
        case '\r':
            stream << "\\r";
            break;
        case '\t':
            stream << "\\t";
            break;
        default:
            if (character < 0x20U)
            {
                constexpr char digits[] = "0123456789abcdef";
                stream << "\\u00" << digits[(character >> 4U) & 0xFU] << digits[character & 0xFU];
            }
            else
            {
                stream << static_cast<char>(character);
            }
        }
    }
    stream << '"';
}

/// Write `,"name":"value"` to @p stream. The leading comma continues a record
/// a caller has already opened.
inline void text_field(std::ostream& stream, std::string_view name, std::string_view value)
{
    stream << ',';
    emit_json_string(stream, name);
    stream << ':';
    emit_json_string(stream, value);
}

/// Write `,"name":true|false` to @p stream.
inline void bool_field(std::ostream& stream, std::string_view name, bool value)
{
    stream << ',';
    emit_json_string(stream, name);
    stream << ':' << (value ? "true" : "false");
}

/// Write `,"name":"value"` to `std::cout`, which is where every benchmark's
/// JSON Lines output goes.
inline void text_field(std::string_view name, std::string_view value)
{
    text_field(std::cout, name, value);
}

/// Write `,"name":true|false` to `std::cout`.
inline void bool_field(std::string_view name, bool value)
{
    bool_field(std::cout, name, value);
}

/// Write @p value as a JSON string to `std::cout`.
inline void emit_json_string(std::string_view value)
{
    emit_json_string(std::cout, value);
}

/// The platform a record was produced on, as the regression policy spells it.
[[nodiscard]] constexpr std::string_view platform_name() noexcept
{
#ifdef _WIN32
    return "windows";
#elif defined(__linux__)
    return "linux";
#elif defined(__APPLE__)
    return "macos";
#else
    return "unknown";
#endif
}

/// The six numbers every latency record carries.
struct Summary
{
    std::size_t samples{};
    std::uint64_t minimum{};
    double median{};
    double p95{};
    double p99{};
    std::uint64_t maximum{};
};

/// Linear-interpolated percentile of an already sorted, non-empty range.
[[nodiscard]] inline double percentile(std::span<const std::uint64_t> sorted, double fraction)
{
    if (sorted.size() == 1)
    {
        return static_cast<double>(sorted.front());
    }
    const auto pos = fraction * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(pos);
    const auto upper = (std::min)(lower + 1, sorted.size() - 1);
    const auto weight = pos - static_cast<double>(lower);
    return static_cast<double>(sorted[lower]) * (1.0 - weight) +
           static_cast<double>(sorted[upper]) * weight;
}

/// Reduce @p values to a ::Summary. An empty input summarises to all zeros
/// rather than reading `front()` off an empty range.
[[nodiscard]] inline Summary summarize(std::span<const std::uint64_t> values)
{
    if (values.empty())
    {
        return {};
    }
    std::vector<std::uint64_t> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    const auto middle = sorted.size() / 2;
    const auto median =
        sorted.size() % 2 == 0
            ? (static_cast<double>(sorted[middle - 1]) + static_cast<double>(sorted[middle])) / 2.0
            : static_cast<double>(sorted[middle]);
    return Summary{
        sorted.size(), sorted.front(), median, percentile(sorted, 0.95), percentile(sorted, 0.99),
        sorted.back()};
}

} // namespace neurale::benchmark
