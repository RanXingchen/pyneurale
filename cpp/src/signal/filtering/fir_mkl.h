/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <span>

namespace neurale::signal::detail
{

enum class MklFirMode
{
    direct,
    fft,
};

void fir_filter_mkl(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                    std::span<const double> taps, std::span<const double> state,
                    std::span<double> output, std::span<double> final_state, MklFirMode mode);

} // namespace neurale::signal::detail
