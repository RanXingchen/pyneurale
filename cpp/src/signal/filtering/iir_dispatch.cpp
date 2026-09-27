/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/iir.h>

#include "iir_builtin.h"
#include "iir_dispatch.h"

#ifdef NEURALE_SIGNAL_WITH_MKL
#include "iir_mkl.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace neurale::signal
{
namespace detail
{
namespace
{

constexpr std::size_t kMklMinChannels = 128;

void validate_finite(std::span<const double> values, const char* name)
{
    if (!std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); }))
    {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

void validate_iir(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                  std::span<const double> b, std::span<const double> a,
                  std::span<const double> state, std::span<double> output,
                  std::span<double> final_state, bool check_finite)
{
    if (n_channels == 0 || b.empty() || b.size() != a.size())
    {
        throw std::invalid_argument("invalid IIR dimensions");
    }
    const auto order = b.size() - 1;
    if (x.size() != n_samples * n_channels || output.size() != x.size() ||
        state.size() != order * n_channels || final_state.size() != state.size())
    {
        throw std::invalid_argument("invalid IIR buffer shape");
    }
    if (a[0] != 1.0)
    {
        throw std::invalid_argument("IIR denominator must be normalized");
    }
    if (check_finite)
    {
        validate_finite(x, "input");
        validate_finite(b, "b");
        validate_finite(a, "a");
        validate_finite(state, "state");
    }
}

void validate_sos(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                  std::span<const double> sos, std::size_t n_sections,
                  std::span<const double> state, std::span<double> output,
                  std::span<double> final_state, bool check_finite)
{
    if (n_channels == 0 || n_sections == 0 || sos.size() != n_sections * 6)
    {
        throw std::invalid_argument("invalid SOS dimensions");
    }
    if (x.size() != n_samples * n_channels || output.size() != x.size() ||
        state.size() != n_sections * 2 * n_channels || final_state.size() != state.size())
    {
        throw std::invalid_argument("invalid SOS buffer shape");
    }
    if (check_finite)
    {
        validate_finite(x, "input");
        validate_finite(sos, "sos");
        validate_finite(state, "state");
    }
    for (std::size_t section = 0; section < n_sections; ++section)
    {
        if (sos[section * 6 + 3] != 1.0)
        {
            throw std::invalid_argument("SOS denominator must be normalized");
        }
    }
}

void execute_iir(IirKernel kernel, std::span<const double> x, std::size_t n_samples,
                 std::size_t n_channels, std::span<const double> b, std::span<const double> a,
                 std::span<const double> state, std::span<double> output,
                 std::span<double> final_state)
{
    switch (kernel)
    {
    case IirKernel::builtin_df2t:
        iir_filter_builtin(x, n_samples, n_channels, b, a, state, output, final_state);
        return;
#ifdef NEURALE_SIGNAL_WITH_MKL
    case IirKernel::mkl_blas_df2t:
        iir_filter_mkl_blas(x, n_samples, n_channels, b, a, state, output, final_state);
        return;
#else
    case IirKernel::mkl_blas_df2t:
        throw std::runtime_error("requested IIR kernel is unavailable in this build");
#endif
    case IirKernel::automatic:
        break;
    }
    throw std::invalid_argument("invalid IIR kernel");
}

void execute_sos(IirKernel kernel, std::span<const double> x, std::size_t n_samples,
                 std::size_t n_channels, std::span<const double> sos, std::size_t n_sections,
                 std::span<const double> state, std::span<double> output,
                 std::span<double> final_state)
{
    switch (kernel)
    {
    case IirKernel::builtin_df2t:
        sos_filter_builtin(x, n_samples, n_channels, sos, n_sections, state, output, final_state);
        return;
#ifdef NEURALE_SIGNAL_WITH_MKL
    case IirKernel::mkl_blas_df2t:
        sos_filter_mkl_blas(x, n_samples, n_channels, sos, n_sections, state, output, final_state);
        return;
#else
    case IirKernel::mkl_blas_df2t:
        throw std::runtime_error("requested SOS kernel is unavailable in this build");
#endif
    case IirKernel::automatic:
        break;
    }
    throw std::invalid_argument("invalid SOS kernel");
}

} // namespace

IirKernel select_iir_kernel(std::size_t n_samples, std::size_t n_channels, std::size_t order)
{
#ifdef NEURALE_SIGNAL_WITH_MKL
    (void)order;
    if (n_samples > 0 && n_channels >= kMklMinChannels)
    {
        return IirKernel::mkl_blas_df2t;
    }
#else
    (void)n_samples;
    (void)n_channels;
    (void)order;
#endif
    return IirKernel::builtin_df2t;
}

IirKernel select_sos_kernel(std::size_t n_samples, std::size_t n_channels, std::size_t n_sections)
{
#ifdef NEURALE_SIGNAL_WITH_MKL
    (void)n_sections;
    if (n_samples > 0 && n_channels >= kMklMinChannels)
    {
        return IirKernel::mkl_blas_df2t;
    }
#else
    (void)n_samples;
    (void)n_channels;
    (void)n_sections;
#endif
    return IirKernel::builtin_df2t;
}

void iir_filter_with_kernel(std::span<const double> x, std::size_t n_samples,
                            std::size_t n_channels, std::span<const double> b,
                            std::span<const double> a, std::span<const double> state,
                            std::span<double> output, std::span<double> final_state,
                            IirKernel kernel, bool check_finite)
{
    validate_iir(x, n_samples, n_channels, b, a, state, output, final_state, check_finite);
    execute_iir(kernel == IirKernel::automatic
                    ? select_iir_kernel(n_samples, n_channels, b.size() - 1)
                    : kernel,
                x, n_samples, n_channels, b, a, state, output, final_state);
}

void sos_filter_with_kernel(std::span<const double> x, std::size_t n_samples,
                            std::size_t n_channels, std::span<const double> sos,
                            std::size_t n_sections, std::span<const double> state,
                            std::span<double> output, std::span<double> final_state,
                            IirKernel kernel, bool check_finite)
{
    validate_sos(x, n_samples, n_channels, sos, n_sections, state, output, final_state,
                 check_finite);
    execute_sos(kernel == IirKernel::automatic
                    ? select_sos_kernel(n_samples, n_channels, n_sections)
                    : kernel,
                x, n_samples, n_channels, sos, n_sections, state, output, final_state);
}

std::string_view iir_kernel_name(IirKernel kernel) noexcept
{
    switch (kernel)
    {
    case IirKernel::automatic:
        return "auto";
    case IirKernel::builtin_df2t:
        return "builtin-df2t";
    case IirKernel::mkl_blas_df2t:
        return "mkl-blas-df2t";
    }
    return "unknown";
}

bool iir_kernel_available(IirKernel kernel) noexcept
{
    switch (kernel)
    {
    case IirKernel::automatic:
    case IirKernel::builtin_df2t:
        return true;
    case IirKernel::mkl_blas_df2t:
#ifdef NEURALE_SIGNAL_WITH_MKL
        return true;
#else
        return false;
#endif
    }
    return false;
}

} // namespace detail

void iir_filter(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                std::span<const double> b, std::span<const double> a, std::span<const double> state,
                std::span<double> output, std::span<double> final_state)
{
    detail::iir_filter_with_kernel(x, n_samples, n_channels, b, a, state, output, final_state,
                                   detail::IirKernel::automatic);
}

void sos_filter(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                std::span<const double> sos, std::size_t n_sections, std::span<const double> state,
                std::span<double> output, std::span<double> final_state)
{
    detail::sos_filter_with_kernel(x, n_samples, n_channels, sos, n_sections, state, output,
                                   final_state, detail::IirKernel::automatic);
}

} // namespace neurale::signal
