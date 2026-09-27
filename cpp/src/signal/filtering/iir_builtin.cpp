/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "iir_builtin.h"

#include <algorithm>

namespace neurale::signal::detail
{

void iir_filter_builtin(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                        std::span<const double> b, std::span<const double> a,
                        std::span<const double> state, std::span<double> output,
                        std::span<double> final_state)
{
    const auto order = b.size() - 1;
    if (state.data() != final_state.data())
    {
        std::copy(state.begin(), state.end(), final_state.begin());
    }
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        for (std::size_t channel = 0; channel < n_channels; ++channel)
        {
            const auto sample_value = x[sample * n_channels + channel];
            auto y = b[0] * sample_value;
            if (order > 0)
                y += final_state[channel];
            output[sample * n_channels + channel] = y;
            for (std::size_t i = 0; i + 1 < order; ++i)
            {
                final_state[i * n_channels + channel] = b[i + 1] * sample_value - a[i + 1] * y +
                                                        final_state[(i + 1) * n_channels + channel];
            }
            if (order > 0)
            {
                final_state[(order - 1) * n_channels + channel] =
                    b[order] * sample_value - a[order] * y;
            }
        }
    }
}

void sos_filter_builtin(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                        std::span<const double> sos, std::size_t n_sections,
                        std::span<const double> state, std::span<double> output,
                        std::span<double> final_state)
{
    if (state.data() != final_state.data())
    {
        std::copy(state.begin(), state.end(), final_state.begin());
    }
    if (x.data() != output.data())
    {
        std::copy(x.begin(), x.end(), output.begin());
    }
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        auto* row = output.data() + sample * n_channels;
        for (std::size_t section = 0; section < n_sections; ++section)
        {
            const auto* coefs = sos.data() + section * 6;
            auto* section_state = final_state.data() + section * 2 * n_channels;
            for (std::size_t channel = 0; channel < n_channels; ++channel)
            {
                const auto input_value = row[channel];
                const auto result = coefs[0] * input_value + section_state[channel];
                section_state[channel] = coefs[1] * input_value - coefs[4] * result +
                                         section_state[n_channels + channel];
                section_state[n_channels + channel] = coefs[2] * input_value - coefs[5] * result;
                row[channel] = result;
            }
        }
    }
}

} // namespace neurale::signal::detail
