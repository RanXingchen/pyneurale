/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/iir.h>

#include "iir_dispatch.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace neurale::signal
{
namespace detail
{
namespace
{

void validate_zero_phase_sos(std::span<const double> x, std::size_t n_samples,
                             std::size_t n_channels, std::span<const double> sos,
                             std::size_t n_sections, std::span<double> output, bool check_finite)
{
    if (n_channels == 0 || n_sections == 0 || sos.size() != n_sections * 6)
    {
        throw std::invalid_argument("invalid SOS dimensions");
    }
    if (x.size() != n_samples * n_channels || output.size() != x.size())
    {
        throw std::invalid_argument("invalid SOS buffer shape");
    }
    for (std::size_t section = 0; section < n_sections; ++section)
    {
        if (sos[section * 6 + 3] != 1.0)
        {
            throw std::invalid_argument("SOS denominator must be normalized");
        }
    }
    if (!check_finite)
        return;
    if (!std::all_of(x.begin(), x.end(), [](double value) { return std::isfinite(value); }))
    {
        throw std::invalid_argument("input must be finite");
    }
    if (!std::all_of(sos.begin(), sos.end(), [](double value) { return std::isfinite(value); }))
    {
        throw std::invalid_argument("sos must be finite");
    }
}

std::size_t default_edge(std::span<const double> sos, std::size_t n_sections)
{
    std::size_t zeros_at_origin = 0;
    std::size_t poles_at_origin = 0;
    for (std::size_t section = 0; section < n_sections; ++section)
    {
        zeros_at_origin += sos[section * 6 + 2] == 0.0;
        poles_at_origin += sos[section * 6 + 5] == 0.0;
    }
    return 3 * (2 * n_sections + 1 - std::min(zeros_at_origin, poles_at_origin));
}

std::vector<double> steady_state(std::span<const double> sos, std::size_t n_sections)
{
    std::vector<double> state(n_sections * 2);
    double scale = 1.0;
    for (std::size_t section = 0; section < n_sections; ++section)
    {
        const auto* c = sos.data() + section * 6;
        const double den = 1.0 + c[4] + c[5];
        if (den == 0.0)
        {
            throw std::invalid_argument("SOS steady-state initial condition is singular");
        }
        const double gain = (c[0] + c[1] + c[2]) / den;
        state[section * 2] = scale * (gain - c[0]);
        state[section * 2 + 1] = scale * (c[2] - c[5] * gain);
        scale *= gain;
    }
    return state;
}

void fill_state(std::span<const double> steady, std::span<const double> endpoint,
                std::size_t n_sections, std::size_t n_channels, std::span<double> state)
{
    for (std::size_t section = 0; section < n_sections; ++section)
    {
        for (std::size_t row = 0; row < 2; ++row)
        {
            const double scale = steady[section * 2 + row];
            auto* target = state.data() + (section * 2 + row) * n_channels;
            for (std::size_t channel = 0; channel < n_channels; ++channel)
            {
                target[channel] = scale * endpoint[channel];
            }
        }
    }
}

void odd_extend(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                std::size_t edge, std::span<double> extended)
{
    const auto* first = x.data();
    const auto* last = x.data() + (n_samples - 1) * n_channels;
    for (std::size_t i = 0; i < edge; ++i)
    {
        const auto* left = x.data() + (edge - i) * n_channels;
        const auto* right = x.data() + (n_samples - 2 - i) * n_channels;
        auto* left_output = extended.data() + i * n_channels;
        auto* right_output = extended.data() + (edge + n_samples + i) * n_channels;
        for (std::size_t channel = 0; channel < n_channels; ++channel)
        {
            left_output[channel] = 2.0 * first[channel] - left[channel];
            right_output[channel] = 2.0 * last[channel] - right[channel];
        }
    }
    std::copy(x.begin(), x.end(),
              extended.begin() + static_cast<std::ptrdiff_t>(edge * n_channels));
}

void reverse_samples(std::span<double> values, std::size_t n_samples, std::size_t n_channels)
{
    for (std::size_t sample = 0; sample < n_samples / 2; ++sample)
    {
        auto* first = values.data() + sample * n_channels;
        auto* last = values.data() + (n_samples - 1 - sample) * n_channels;
        std::swap_ranges(first, first + n_channels, last);
    }
}

} // namespace

void sos_filtfilt_with_kernel(std::span<const double> x, std::size_t n_samples,
                              std::size_t n_channels, std::span<const double> sos,
                              std::size_t n_sections, std::span<double> output, IirKernel kernel,
                              bool check_finite)
{
    validate_zero_phase_sos(x, n_samples, n_channels, sos, n_sections, output, check_finite);

    const std::size_t edge = default_edge(sos, n_sections);
    if (n_samples <= edge)
    {
        throw std::invalid_argument("input length must be greater than the padding length " +
                                    std::to_string(edge));
    }

    const std::size_t extended_samples = n_samples + 2 * edge;
    std::vector<double> workspace(extended_samples * n_channels);
    odd_extend(x, n_samples, n_channels, edge, workspace);

    const auto initial = steady_state(sos, n_sections);
    std::vector<double> state(n_sections * 2 * n_channels);
    const IirKernel selected = kernel == IirKernel::automatic
                                   ? select_sos_kernel(extended_samples, n_channels, n_sections)
                                   : kernel;

    fill_state(initial, {workspace.data(), n_channels}, n_sections, n_channels, state);
    sos_filter_with_kernel(workspace, extended_samples, n_channels, sos, n_sections, state,
                           workspace, state, selected, false);

    reverse_samples(workspace, extended_samples, n_channels);
    fill_state(initial, {workspace.data(), n_channels}, n_sections, n_channels, state);
    sos_filter_with_kernel(workspace, extended_samples, n_channels, sos, n_sections, state,
                           workspace, state, selected, false);

    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        const auto* source = workspace.data() + (n_samples + edge - 1 - sample) * n_channels;
        std::copy_n(source, n_channels, output.data() + sample * n_channels);
    }
}

} // namespace detail

void sos_filtfilt(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                  std::span<const double> sos, std::size_t n_sections, std::span<double> output)
{
    detail::sos_filtfilt_with_kernel(x, n_samples, n_channels, sos, n_sections, output,
                                     detail::IirKernel::automatic);
}

} // namespace neurale::signal
