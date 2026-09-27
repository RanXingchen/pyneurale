/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <complex>
#include <cstddef>
#include <span>

namespace neurale::signal
{

/**
 * @brief Integrate spectra in the frequency domain.
 *
 * @param x Complex spectra with `n_samples * n_channels` elements.
 * @param n_samples Number of frequency bins per channel.
 * @param n_channels Number of interleaved channels.
 * @param time_step Positive sampling interval in seconds.
 * @param order Positive integration order.
 * @param output Caller-owned complex output buffer matching the `x` size.
 * @throws std::invalid_argument If dimensions, sampling interval, order, or
 *         buffer sizes are invalid.
 */
void fft_integrate(std::span<const std::complex<double>> x, std::size_t n_samples,
                   std::size_t n_channels, double time_step, std::size_t order,
                   std::span<std::complex<double>> output);

/**
 * @brief Integrate real spectra in the frequency domain.
 *
 * @param x Real spectra with `n_samples * n_channels` elements.
 * @param n_samples Number of frequency bins per channel.
 * @param n_channels Number of interleaved channels.
 * @param time_step Positive sampling interval in seconds.
 * @param order Positive integration order.
 * @param output Caller-owned real output buffer matching the `x` size.
 * @throws std::invalid_argument If dimensions, sampling interval, order, or
 *         buffer sizes are invalid.
 */
void fft_integrate(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                   double time_step, std::size_t order, std::span<double> output);

} // namespace neurale::signal
