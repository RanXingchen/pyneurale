/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>

namespace neurale::signal
{

// Filter design

/**
 * @brief Design a windowed linear-phase FIR filter.
 *
 * Cutoff frequencies are normalized to Nyquist in `(0, 1)`.
 *
 * @param order Filter order. The output contains `order + 1` coefficients.
 * @param cutoff One cutoff for low/high-pass or two for band-pass/stop.
 * @param band Response type: `lowpass`, `highpass`, `bandpass`, or `bandstop`.
 * @param window Design window: `hann`, `hamming`, `blackman`, or `flattop`.
 * @param scale Normalize the response at the center of the first passband.
 * @param output Caller-owned coefficient buffer of length `order + 1`.
 * @throws std::invalid_argument If parameters or output size are invalid.
 */
void firwin(std::size_t order, std::span<const double> cutoff, std::string_view band,
            std::string_view window, bool scale, std::span<double> output);

/**
 * @brief Design a linear-phase FIR filter by least squares.
 *
 * Frequencies are normalized to Nyquist in the closed interval `[0, 1]`.
 *
 * @param order Filter order. The output contains `order + 1` coefficients.
 * @param bands Monotonic band-edge frequencies in adjacent pairs.
 * @param desired Desired gain at each band edge.
 * @param weights Positive weight for each frequency band.
 * @param output Caller-owned coefficient buffer of length `order + 1`.
 * @throws std::invalid_argument If order, array lengths, frequencies, gains,
 *         weights, or output size are invalid.
 */
void firls(std::size_t order, std::span<const double> bands, std::span<const double> desired,
           std::span<const double> weights, std::span<double> output);

// Filtering

/**
 * @brief Apply a causal FIR filter to sample-major data.
 *
 * Input, output, state, and final state use interleaved sample-major layout.
 * State stores the preceding `taps.size() - 1` samples for every channel.
 *
 * @param x Input samples with `n_samples * n_channels` elements.
 * @param n_samples Number of input samples per channel.
 * @param n_channels Number of interleaved channels.
 * @param taps FIR coefficients ordered from newest to oldest contribution.
 * @param state Initial filter state with
 *        `(taps.size() - 1) * n_channels` elements.
 * @param output Caller-owned output buffer matching the `x` size.
 * @param final_state Caller-owned final-state buffer matching `state`.
 * @throws std::invalid_argument If dimensions or buffer sizes are invalid.
 */
void fir_filter(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                std::span<const double> taps, std::span<const double> state,
                std::span<double> output, std::span<double> final_state);

} // namespace neurale::signal
