/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "iir_mkl.h"
#include "mkl_utils.h"

#include <mkl_cblas.h>

#include <algorithm>
#include <vector>

namespace neurale::signal::detail
{
namespace
{

void affine_vector(MKL_INT size, double first_scale, const double* first, double second_scale,
                   const double* second, double* output)
{
    cblas_dcopy(size, first, 1, output, 1);
    cblas_dscal(size, first_scale, output, 1);
    cblas_daxpy(size, second_scale, second, 1, output, 1);
}

} // namespace

void iir_filter_mkl_blas(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                         std::span<const double> b, std::span<const double> a,
                         std::span<const double> state, std::span<double> output,
                         std::span<double> final_state)
{
    const auto order = b.size() - 1;
    if (state.data() != final_state.data())
    {
        std::copy(state.begin(), state.end(), final_state.begin());
    }
    if (n_samples == 0)
        return;

    const MKL_INT channels = mkl::checked_mkl_int(n_channels, "channel count");
    mkl::LocalThreadLimit thread_limit;
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        const double* sample_x = x.data() + sample * n_channels;
        double* y = output.data() + sample * n_channels;
        if (order == 0)
        {
            cblas_dcopy(channels, sample_x, 1, y, 1);
            cblas_dscal(channels, b[0], y, 1);
            continue;
        }
        affine_vector(channels, b[0], sample_x, 1.0, final_state.data(), y);
        for (std::size_t i = 0; i + 1 < order; ++i)
        {
            double* current = final_state.data() + i * n_channels;
            const double* next = current + n_channels;
            cblas_dcopy(channels, next, 1, current, 1);
            cblas_daxpy(channels, b[i + 1], sample_x, 1, current, 1);
            cblas_daxpy(channels, -a[i + 1], y, 1, current, 1);
        }
        double* last = final_state.data() + (order - 1) * n_channels;
        affine_vector(channels, b[order], sample_x, -a[order], y, last);
    }
}

void sos_filter_mkl_blas(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                         std::span<const double> sos, std::size_t n_sections,
                         std::span<const double> state, std::span<double> output,
                         std::span<double> final_state)
{
    if (state.data() != final_state.data())
    {
        std::copy(state.begin(), state.end(), final_state.begin());
    }
    if (x.data() != output.data())
    {
        std::copy(x.begin(), x.end(), output.begin());
    }
    if (n_samples == 0)
        return;

    const MKL_INT channels = mkl::checked_mkl_int(n_channels, "channel count");
    mkl::LocalThreadLimit thread_limit;
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        double* value = output.data() + sample * n_channels;
        for (std::size_t section = 0; section < n_sections; ++section)
        {
            const double* coefs = sos.data() + section * 6;
            double* state0 = final_state.data() + section * 2 * n_channels;
            double* state1 = state0 + n_channels;

            thread_local std::vector<double> result;
            result.resize(n_channels);
            affine_vector(channels, coefs[0], value, 1.0, state0, result.data());
            cblas_dcopy(channels, state1, 1, state0, 1);
            cblas_daxpy(channels, coefs[1], value, 1, state0, 1);
            cblas_daxpy(channels, -coefs[4], result.data(), 1, state0, 1);
            affine_vector(channels, coefs[2], value, -coefs[5], result.data(), state1);
            cblas_dcopy(channels, result.data(), 1, value, 1);
        }
    }
}

} // namespace neurale::signal::detail
