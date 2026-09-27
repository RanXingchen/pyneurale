/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/**
 * @file
 * @brief Public native spatial filtering functions.
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>

namespace neurale::signal
{

/**
 * @brief Subtract a sample-wise common channel reference.
 *
 * @param x Finite sample-major data with
 *        `n_samples * n_channels` elements.
 * @param n_samples Number of samples per channel.
 * @param n_channels Number of interleaved channels.
 * @param reference_channels Non-empty, unique channel positions used to
 *        estimate the reference.
 * @param method Reference statistic: `mean` or `median`.
 * @param output Caller-owned referenced data buffer matching `x`.
 * @param reference Caller-owned vector containing one reference value per
 *        sample.
 * @throws std::invalid_argument If dimensions, channel positions, method, or
 *         x values are invalid.
 */
void common_reference(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                      std::span<const std::size_t> reference_channels, std::string_view method,
                      std::span<double> output, std::span<double> reference);

} // namespace neurale::signal
