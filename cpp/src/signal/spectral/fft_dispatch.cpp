/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "fft_dispatch.h"

#include "builtin/fft.h"

#ifdef NEURALE_SIGNAL_WITH_MKL
#include "mkl/fft.h"
#endif

#include <stdexcept>

namespace neurale::signal::detail
{

void fft_with_kernel(std::span<std::complex<double>> values, bool inverse, FftKernel kernel)
{
    const auto selected =
        kernel == FftKernel::automatic ? select_fft_kernel(values.size()) : kernel;
    switch (selected)
    {
    case FftKernel::builtin:
        fft_builtin_inplace(values, inverse);
        return;
#ifdef NEURALE_SIGNAL_WITH_MKL
    case FftKernel::mkl_dfti:
        fft_mkl_inplace(values, inverse);
        return;
#else
    case FftKernel::mkl_dfti:
        throw std::runtime_error("MKL DFTI is unavailable in this build");
#endif
    case FftKernel::automatic:
        break;
    }
    throw std::invalid_argument("invalid FFT kernel");
}

void fft_inplace(std::span<std::complex<double>> values, bool inverse)
{
    fft_with_kernel(values, inverse, FftKernel::automatic);
}

FftKernel select_fft_kernel(std::size_t length)
{
#ifdef NEURALE_SIGNAL_WITH_MKL
    return length >= 64 ? FftKernel::mkl_dfti : FftKernel::builtin;
#else
    (void)length;
    return FftKernel::builtin;
#endif
}

std::string_view fft_kernel_name(FftKernel kernel) noexcept
{
    switch (kernel)
    {
    case FftKernel::automatic:
        return "auto";
    case FftKernel::builtin:
        return "builtin-fft";
    case FftKernel::mkl_dfti:
        return "mkl-dfti";
    }
    return "unknown";
}

bool fft_kernel_available(FftKernel kernel) noexcept
{
    switch (kernel)
    {
    case FftKernel::automatic:
    case FftKernel::builtin:
        return true;
    case FftKernel::mkl_dfti:
#ifdef NEURALE_SIGNAL_WITH_MKL
        return true;
#else
        return false;
#endif
    }
    return false;
}

} // namespace neurale::signal::detail
