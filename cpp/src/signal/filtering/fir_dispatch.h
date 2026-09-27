/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>

namespace neurale::signal::detail
{

enum class FirKernel
{
    automatic,
    builtin_direct,
    mkl_vsl_direct,
    mkl_vsl_fft,
};

void fir_filter_with_kernel(std::span<const double> x, std::size_t n_samples,
                            std::size_t n_channels, std::span<const double> taps,
                            std::span<const double> state, std::span<double> output,
                            std::span<double> final_state, FirKernel kernel);

FirKernel select_fir_kernel(std::size_t n_samples, std::size_t n_channels, std::size_t n_taps);

bool fir_block_path_useful(std::size_t n_samples, std::size_t n_taps) noexcept;

std::string_view fir_kernel_name(FirKernel kernel) noexcept;

bool fir_kernel_available(FirKernel kernel) noexcept;

} // namespace neurale::signal::detail
