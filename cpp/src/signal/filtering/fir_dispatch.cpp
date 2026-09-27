/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/fir.h>

#include "fir_builtin.h"
#include "fir_dispatch.h"

#ifdef NEURALE_SIGNAL_WITH_MKL
#include "fir_mkl.h"
#endif

#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace neurale::signal
{
namespace detail
{
namespace
{

#ifdef NEURALE_SIGNAL_WITH_MKL
constexpr std::size_t kFftMinSamples = 16;
constexpr std::size_t kFftMinTaps = 257;
constexpr std::size_t kMklDirectMinTaps = 32;
constexpr std::size_t kMklDirectMinChannels = 16;
#endif
constexpr std::size_t kDirectMaxTaps = 64;
constexpr std::size_t kBlockDirectMinSamples = 64;

#ifdef NEURALE_SIGNAL_WITH_MKL
bool fft_is_useful(std::size_t n_samples, std::size_t n_taps) noexcept
{
    return n_samples >= kFftMinSamples && n_taps >= kFftMinTaps;
}

bool mkl_direct_is_useful(std::size_t n_channels, std::size_t n_taps) noexcept
{
    return n_taps >= kMklDirectMinTaps || n_channels >= kMklDirectMinChannels;
}
#endif

void execute_kernel(FirKernel kernel, std::span<const double> x, std::size_t n_samples,
                    std::size_t n_channels, std::span<const double> taps,
                    std::span<const double> state, std::span<double> output,
                    std::span<double> final_state)
{
    switch (kernel)
    {
    case FirKernel::builtin_direct:
        fir_filter_builtin(x, n_samples, n_channels, taps, state, output, final_state);
        return;
#ifdef NEURALE_SIGNAL_WITH_MKL
    case FirKernel::mkl_vsl_direct:
        fir_filter_mkl(x, n_samples, n_channels, taps, state, output, final_state,
                       MklFirMode::direct);
        return;
    case FirKernel::mkl_vsl_fft:
        fir_filter_mkl(x, n_samples, n_channels, taps, state, output, final_state, MklFirMode::fft);
        return;
#else
    case FirKernel::mkl_vsl_direct:
    case FirKernel::mkl_vsl_fft:
        throw std::runtime_error("requested FIR kernel is unavailable in this build");
#endif
    case FirKernel::automatic:
        break;
    }
    throw std::invalid_argument("invalid FIR kernel");
}

} // namespace

FirKernel select_fir_kernel(std::size_t n_samples, std::size_t n_channels, std::size_t n_taps)
{
#ifdef NEURALE_SIGNAL_WITH_MKL
    if (n_channels == 0 || n_taps == 0 || n_taps <= 8)
    {
        return FirKernel::builtin_direct;
    }
    if (fft_is_useful(n_samples, n_taps))
    {
        return FirKernel::mkl_vsl_fft;
    }
    if (mkl_direct_is_useful(n_channels, n_taps))
    {
        return FirKernel::mkl_vsl_direct;
    }
#else
    (void)n_samples;
    (void)n_channels;
    (void)n_taps;
#endif
    return FirKernel::builtin_direct;
}

bool fir_block_path_useful(std::size_t n_samples, std::size_t n_taps) noexcept
{
    if (n_samples <= 1 || n_taps <= kDirectMaxTaps)
    {
        return false;
    }
#ifdef NEURALE_SIGNAL_WITH_MKL
    if (n_taps >= kFftMinTaps)
    {
        return n_samples >= kFftMinSamples;
    }
#endif
    return n_samples >= kBlockDirectMinSamples;
}

void fir_filter_with_kernel(std::span<const double> x, std::size_t n_samples,
                            std::size_t n_channels, std::span<const double> taps,
                            std::span<const double> state, std::span<double> output,
                            std::span<double> final_state, FirKernel kernel)
{
    if (n_channels == 0 || taps.empty())
    {
        throw std::invalid_argument("FIR filtering requires channels and coefficients");
    }
    execute_kernel(kernel == FirKernel::automatic
                       ? select_fir_kernel(n_samples, n_channels, taps.size())
                       : kernel,
                   x, n_samples, n_channels, taps, state, output, final_state);
}

std::string_view fir_kernel_name(FirKernel kernel) noexcept
{
    switch (kernel)
    {
    case FirKernel::automatic:
        return "auto";
    case FirKernel::builtin_direct:
        return "builtin-direct";
    case FirKernel::mkl_vsl_direct:
        return "mkl-vsl-direct";
    case FirKernel::mkl_vsl_fft:
        return "mkl-vsl-fft";
    }
    return "unknown";
}

bool fir_kernel_available(FirKernel kernel) noexcept
{
    switch (kernel)
    {
    case FirKernel::automatic:
    case FirKernel::builtin_direct:
        return true;
    case FirKernel::mkl_vsl_direct:
    case FirKernel::mkl_vsl_fft:
#ifdef NEURALE_SIGNAL_WITH_MKL
        return true;
#else
        return false;
#endif
    }
    return false;
}

} // namespace detail

void fir_filter(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                std::span<const double> taps, std::span<const double> state,
                std::span<double> output, std::span<double> final_state)
{
    detail::fir_filter_with_kernel(x, n_samples, n_channels, taps, state, output, final_state,
                                   detail::FirKernel::automatic);
}

} // namespace neurale::signal
