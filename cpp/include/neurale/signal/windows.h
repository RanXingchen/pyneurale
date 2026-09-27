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

/**
 * @brief Write a Hann, Hamming, Blackman, or flat-top window.
 *
 * @param kind Window name: `hann`, `hamming`, `blackman`, or `flattop`.
 * @param output Caller-owned output buffer.
 * @param symmetric Generate a filter-design window when `true`, or a periodic
 *        spectral-analysis window when `false`.
 * @throws std::invalid_argument If `kind` is unknown.
 */
void cosine_window(std::string_view kind, std::span<double> output, bool symmetric = true);

/**
 * @brief Write a Kaiser window.
 *
 * @param output Caller-owned output buffer.
 * @param beta Kaiser shape parameter.
 * @param symmetric Generate a filter-design window when `true`, or a periodic
 *        spectral-analysis window when `false`.
 * @throws std::invalid_argument If `beta` is not finite.
 */
void kaiser_window(std::span<double> output, double beta, bool symmetric = true);

/**
 * @brief Compute discrete prolate spheroidal sequences.
 *
 * This implements the Lees/Park multitaper calculation. It solves the
 * symmetric tridiagonal DPSS eigenproblem by Sturm
 * bisection and inverse iteration, then computes exact concentration ratios.
 *
 * @param length Number of samples in each taper.
 * @param nw Time-half-bandwidth product in `(0, length / 2)`.
 * @param n_tapers Number of requested DPSS tapers.
 * @param tapers Caller-owned sample-major `(length, n_tapers)` output.
 * @param concentration_ratios Caller-owned output with `n_tapers` elements.
 * @throws std::invalid_argument If dimensions or parameters are invalid.
 * @throws std::runtime_error If an eigenvector fails to converge.
 */
void multitap(std::size_t length, double nw, std::size_t n_tapers, std::span<double> tapers,
              std::span<double> concentration_ratios);

} // namespace neurale::signal
