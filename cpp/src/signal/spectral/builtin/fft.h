/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <complex>
#include <span>

namespace neurale::signal::detail
{

void fft_builtin_inplace(std::span<std::complex<double>> values, bool inverse);

} // namespace neurale::signal::detail
