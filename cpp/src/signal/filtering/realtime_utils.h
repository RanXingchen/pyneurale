/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>

namespace neurale::signal::detail
{

inline constexpr std::size_t kMklRealtimeMinChannels = 128;

inline bool use_mkl_realtime_blas(std::size_t n_channels) noexcept
{
    return n_channels >= kMklRealtimeMinChannels;
}

} // namespace neurale::signal::detail
