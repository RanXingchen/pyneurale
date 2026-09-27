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

enum class IirKernel
{
    automatic,
    builtin_df2t,
    mkl_blas_df2t,
};

void iir_filter_with_kernel(std::span<const double> x, std::size_t n_samples,
                            std::size_t n_channels, std::span<const double> b,
                            std::span<const double> a, std::span<const double> state,
                            std::span<double> output, std::span<double> final_state,
                            IirKernel kernel, bool check_finite = true);

void sos_filter_with_kernel(std::span<const double> x, std::size_t n_samples,
                            std::size_t n_channels, std::span<const double> sos,
                            std::size_t n_sections, std::span<const double> state,
                            std::span<double> output, std::span<double> final_state,
                            IirKernel kernel, bool check_finite = true);

void sos_filtfilt_with_kernel(std::span<const double> x, std::size_t n_samples,
                              std::size_t n_channels, std::span<const double> sos,
                              std::size_t n_sections, std::span<double> output, IirKernel kernel,
                              bool check_finite = true);

IirKernel select_iir_kernel(std::size_t n_samples, std::size_t n_channels, std::size_t order);

IirKernel select_sos_kernel(std::size_t n_samples, std::size_t n_channels, std::size_t n_sections);

std::string_view iir_kernel_name(IirKernel kernel) noexcept;

bool iir_kernel_available(IirKernel kernel) noexcept;

} // namespace neurale::signal::detail
