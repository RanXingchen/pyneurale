/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <span>

namespace neurale::signal::detail
{

inline void update_fir_state(std::span<const double> x, std::size_t n_samples,
                             std::size_t n_channels, std::size_t history_length,
                             std::span<const double> state, std::span<double> final_state)
{
    if (history_length == 0)
    {
        return;
    }

    const auto history_size = history_length * n_channels;
    if (n_samples >= history_length)
    {
        const auto first_sample = n_samples - history_length;
        std::copy_n(x.begin() + static_cast<std::ptrdiff_t>(first_sample * n_channels),
                    history_size, final_state.begin());
        return;
    }

    const auto retained_history = history_length - n_samples;
    const auto retained_size = retained_history * n_channels;
    std::copy_n(state.begin() + static_cast<std::ptrdiff_t>(n_samples * n_channels), retained_size,
                final_state.begin());
    std::copy(x.begin(), x.end(), final_state.begin() + static_cast<std::ptrdiff_t>(retained_size));
}

} // namespace neurale::signal::detail
