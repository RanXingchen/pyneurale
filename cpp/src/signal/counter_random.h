/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cmath>
#include <cstdint>

namespace neurale::signal::simulation::detail
{

inline constexpr double kInverse53 = 1.0 / 9007199254740992.0;
inline constexpr double kTwoPi = 6.283185307179586476925286766559;

[[nodiscard]] inline std::uint64_t splitmix64(std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

[[nodiscard]] inline std::uint64_t counter(std::uint64_t seed, std::uint64_t sample,
                                           std::uint64_t stream) noexcept
{
    return seed ^ (sample * 0xd2b74407b1ce6e93ULL) ^ (stream * 0xca5a826395121157ULL);
}

[[nodiscard]] inline double uniform01(std::uint64_t seed, std::uint64_t sample,
                                      std::uint64_t stream) noexcept
{
    return static_cast<double>(splitmix64(counter(seed, sample, stream)) >> 11U) * kInverse53;
}

[[nodiscard]] inline double uniform_open(std::uint64_t seed, std::uint64_t sample,
                                         std::uint64_t stream) noexcept
{
    return (static_cast<double>(splitmix64(counter(seed, sample, stream)) >> 11U) + 0.5) *
           kInverse53;
}

[[nodiscard]] inline double gaussian(std::uint64_t seed, std::uint64_t sample,
                                     std::uint64_t stream) noexcept
{
    const auto first = uniform_open(seed, sample, stream * 2U);
    const auto second = uniform_open(seed, sample, stream * 2U + 1U);
    return std::sqrt(-2.0 * std::log(first)) * std::cos(kTwoPi * second);
}

} // namespace neurale::signal::simulation::detail
