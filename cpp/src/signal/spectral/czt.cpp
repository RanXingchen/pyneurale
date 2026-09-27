/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/spectral.h>

#include "../numerics.h"
#include "fft_dispatch.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace neurale::signal
{

void czt(std::span<const std::complex<double>> x, std::size_t n_samples, std::size_t n_channels,
         std::size_t output_length, std::complex<double> ratio, std::complex<double> start,
         std::span<std::complex<double>> output)
{
    if (n_samples == 0 || n_channels == 0 || output_length == 0)
    {
        throw std::invalid_argument("CZT dimensions must be positive");
    }
    if (x.size() != n_samples * n_channels || output.size() != output_length * n_channels)
    {
        throw std::invalid_argument("CZT buffer shape is incorrect");
    }
    if (start == std::complex<double>{0.0, 0.0} || ratio == std::complex<double>{0.0, 0.0})
    {
        throw std::invalid_argument("CZT start and ratio must be nonzero");
    }

    const auto chirp_length = n_samples + output_length - 1;
    const auto fft_length = detail::next_power_of_two(chirp_length);
    std::vector<std::complex<double>> chirp(chirp_length);
    for (std::size_t i = 0; i < chirp_length; ++i)
    {
        const auto signed_idx = static_cast<double>(i) - static_cast<double>(n_samples - 1);
        chirp[i] = std::pow(ratio, 0.5 * signed_idx * signed_idx);
    }

    std::vector<std::complex<double>> convolution(fft_length, 0.0);
    for (std::size_t i = 0; i < chirp_length; ++i)
    {
        convolution[i] = 1.0 / chirp[i];
    }
    detail::fft_inplace(convolution, false);

    std::vector<std::complex<double>> transformed(fft_length);
    for (std::size_t channel = 0; channel < n_channels; ++channel)
    {
        std::fill(transformed.begin(), transformed.end(), 0.0);
        for (std::size_t sample = 0; sample < n_samples; ++sample)
        {
            const auto sample_idx = static_cast<double>(sample);
            transformed[sample] = x[sample * n_channels + channel] * std::pow(start, -sample_idx) *
                                  std::pow(ratio, 0.5 * sample_idx * sample_idx);
        }
        detail::fft_inplace(transformed, false);
        for (std::size_t i = 0; i < fft_length; ++i)
        {
            transformed[i] *= convolution[i];
        }
        detail::fft_inplace(transformed, true);
        for (std::size_t point = 0; point < output_length; ++point)
        {
            output[point * n_channels + channel] =
                transformed[n_samples - 1 + point] * chirp[n_samples - 1 + point];
        }
    }
}

} // namespace neurale::signal
