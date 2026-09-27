/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/**
 * @file
 * @brief Public native IIR filter-design and filtering functions.
 */

#pragma once

#include <neurale/signal/representations.h>

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace neurale::signal
{

// Filter design

/**
 * @brief Design a second-order digital notch filter in ZPK form.
 *
 * Frequencies are normalized to Nyquist.
 *
 * @param freq Notch center frequency in the open interval `(0, 1)`.
 * @param bandwidth Positive normalized notch bandwidth.
 * @return Digital zeros, poles, and gain.
 * @throws std::invalid_argument If frequency or bandwidth is invalid.
 */
Zpk notch_zpk(double freq, double bandwidth);

/**
 * @brief Design a classical digital IIR filter in ZPK form.
 *
 * @param kind Prototype name: `butterworth`, `bessel`, or `elliptic`.
 * @param order Positive prototype order.
 * @param cutoff One cutoff for low/high-pass or two for band-pass/stop.
 * @param band Response type: `lowpass`, `highpass`, `bandpass`, or
 *        `bandstop`.
 * @param fs Positive sampling rate.
 * @param rp Positive passband ripple used by elliptic designs.
 * @param rs Positive stopband attenuation used by elliptic designs.
 * @return Digital zeros, poles, and gain.
 * @throws std::invalid_argument If design parameters are invalid.
 * @throws std::runtime_error If the numerical design fails.
 */
Zpk iir_zpk(std::string_view kind, std::size_t order, std::span<const double> cutoff,
            std::string_view band, double fs, double rp, double rs);

/**
 * @brief Design a classical digital IIR filter in transfer-function form.
 *
 * Parameters follow iir_zpk().
 *
 * @return Numerator and denominator coefficients.
 * @throws std::invalid_argument If design parameters are invalid.
 * @throws std::runtime_error If the numerical design fails.
 */
TransferFunction iir_tf(std::string_view kind, std::size_t order, std::span<const double> cutoff,
                        std::string_view band, double fs, double rp, double rs);

/**
 * @brief Design a classical digital IIR filter as second-order sections.
 *
 * Parameters follow iir_zpk().
 *
 * @return Flattened `(n_sections, 6)` SOS coefficients in
 *         `[b0, b1, b2, a0, a1, a2]` order.
 * @throws std::invalid_argument If design parameters are invalid.
 * @throws std::runtime_error If the numerical design fails.
 */
std::vector<double> iir_sos(std::string_view kind, std::size_t order,
                            std::span<const double> cutoff, std::string_view band, double fs,
                            double rp, double rs);

/**
 * @brief Design a classical digital IIR filter in state-space form.
 *
 * The cutoff frequencies follow SciPy semantics: when fs is 2,
 * cutoffs are normalized to Nyquist in (0, 1); otherwise cutoffs use the same
 * units as fs. The returned state coordinates are internally
 * normalized to Nyquist for numerical stability.
 *
 * The implementation constructs a canonical analog state-space realization
 * from prototype zeros, poles, and gain, applies the requested analog lowpass
 * transformation, and then performs a state-space bilinear transform. It does
 * not form a high-order transfer polynomial.
 */
StateSpace iir_ss(std::string_view kind, std::size_t order, std::span<const double> cutoff,
                  std::string_view band, double fs, double rp, double rs);

// Filtering

/**
 * @brief Apply a causal direct-form IIR filter to sample-major data.
 *
 * @param x Input with `n_samples * n_channels` elements.
 * @param n_samples Number of samples per channel.
 * @param n_channels Number of interleaved channels.
 * @param b Numerator coefficients.
 * @param a Denominator coefficients with a non-zero leading value.
 * @param state Initial transposed direct-form II state with
 *        `max(b.size(), a.size()) - 1` values per channel.
 * @param output Caller-owned output buffer matching `x`.
 * @param final_state Caller-owned state buffer matching `state`.
 * @throws std::invalid_argument If coefficients, dimensions, state, or buffer
 *         sizes are invalid.
 */
void iir_filter(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                std::span<const double> b, std::span<const double> a, std::span<const double> state,
                std::span<double> output, std::span<double> final_state);

/**
 * @brief Apply cascaded second-order sections to sample-major data.
 *
 * @param x Input with `n_samples * n_channels` elements.
 * @param n_samples Number of samples per channel.
 * @param n_channels Number of interleaved channels.
 * @param sos Flattened `(n_sections, 6)` section coefficients in
 *        `[b0, b1, b2, a0, a1, a2]` order.
 * @param n_sections Number of second-order sections.
 * @param state Initial state in `(n_sections, 2, n_channels)` order.
 * @param output Caller-owned output buffer matching `x`.
 * @param final_state Caller-owned state buffer matching `state`.
 * @throws std::invalid_argument If sections, dimensions, state, or buffer
 *         sizes are invalid.
 */
void sos_filter(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                std::span<const double> sos, std::size_t n_sections, std::span<const double> state,
                std::span<double> output, std::span<double> final_state);

/**
 * @brief Apply a zero-phase forward-backward SOS filter.
 *
 * The signal is extended at both ends using odd reflection. The extension
 * length follows the default SciPy sosfiltfilt convention.
 *
 * @param x Input with `n_samples * n_channels` elements.
 * @param n_samples Number of samples per channel.
 * @param n_channels Number of interleaved channels.
 * @param sos Flattened normalized `(n_sections, 6)` SOS coefficients.
 * @param n_sections Number of second-order sections.
 * @param output Caller-owned output buffer matching `x`.
 * @throws std::invalid_argument If dimensions, coefficients, or signal length
 *         are invalid.
 */
void sos_filtfilt(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                  std::span<const double> sos, std::size_t n_sections, std::span<double> output);

} // namespace neurale::signal
