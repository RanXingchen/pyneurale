/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "fir_builtin.h"
#include "fir_state.h"

#include <cstddef>
#include <stdexcept>

namespace neurale::signal::detail
{

void fir_filter_builtin(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                        std::span<const double> taps, std::span<const double> state,
                        std::span<double> output, std::span<double> final_state)
{
    if (taps.empty())
    {
        throw std::invalid_argument("FIR taps must be nonempty");
    }

    const std::size_t history_length = taps.size() - 1;
    if (x.size() != n_samples * n_channels || output.size() != x.size() ||
        state.size() != history_length * n_channels || final_state.size() != state.size())
    {
        throw std::invalid_argument("invalid FIR input or state shape");
    }

    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        const auto input_taps = std::min(sample + 1, taps.size());
        for (std::size_t channel = 0; channel < n_channels; ++channel)
        {
            double value = 0.0;

            for (std::size_t tap = 0; tap < input_taps; ++tap)
            {
                value += taps[tap] * x[(sample - tap) * n_channels + channel];
            }
            for (std::size_t tap = input_taps; tap < taps.size(); ++tap)
            {
                const auto history_sample = history_length + sample - tap;
                value += taps[tap] * state[history_sample * n_channels + channel];
            }

            output[sample * n_channels + channel] = value;
        }
    }

    update_fir_state(x, n_samples, n_channels, history_length, state, final_state);
}

} // namespace neurale::signal::detail
