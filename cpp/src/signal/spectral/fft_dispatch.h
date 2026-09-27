/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <complex>
#include <span>
#include <string_view>

namespace neurale::signal::detail
{

enum class FftKernel
{
    automatic,
    builtin,
    mkl_dfti,
};

void fft_inplace(std::span<std::complex<double>> values, bool inverse);

void fft_with_kernel(std::span<std::complex<double>> values, bool inverse, FftKernel kernel);

FftKernel select_fft_kernel(std::size_t length);

std::string_view fft_kernel_name(FftKernel kernel) noexcept;

bool fft_kernel_available(FftKernel kernel) noexcept;

} // namespace neurale::signal::detail
