/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "iir_realtime.h"
#include "realtime_utils.h"

#ifdef NEURALE_SIGNAL_WITH_MKL
#include "mkl_utils.h"
#include <mkl_cblas.h>
#endif

#include <algorithm>
#include <cstring>

namespace neurale::signal::detail
{

IirRealtimeProcessor::IirRealtimeProcessor(std::span<const double> b, std::span<const double> a,
                                           std::size_t n_channels, std::span<const double> state)
    : n_channels_(n_channels), b_(b.begin(), b.end()), a_(a.begin(), a.end())
{
#ifdef NEURALE_SIGNAL_WITH_MKL
    if (order() > 0 && use_mkl_realtime_blas(n_channels_))
    {
        kernel_ = Kernel::mkl_blas_df2t;
    }
#endif
    const auto rows = b_.size() - 1 + (kernel_ == Kernel::mkl_blas_df2t ? 1 : 0);
    state_.assign(rows * n_channels_, 0.0);
    copy_public_state(state);
}

void IirRealtimeProcessor::process(std::span<double> frame, std::size_t n_samples)
{
    if (kernel_ == Kernel::mkl_blas_df2t)
    {
        process_mkl(frame, n_samples);
        return;
    }
    process_scalar(frame, n_samples);
}

void IirRealtimeProcessor::reset()
{
    std::fill(state_.begin(), state_.end(), 0.0);
}

void IirRealtimeProcessor::set_state(std::span<const double> state)
{
    copy_public_state(state);
}

void IirRealtimeProcessor::state(std::span<double> output) const
{
    std::copy_n(state_.data(), output.size(), output.data());
}

std::size_t IirRealtimeProcessor::order() const noexcept
{
    return b_.size() - 1;
}

std::size_t IirRealtimeProcessor::n_channels() const noexcept
{
    return n_channels_;
}

std::string_view IirRealtimeProcessor::kernel_name() const noexcept
{
    return kernel_ == Kernel::mkl_blas_df2t ? "mkl-blas-df2t" : "builtin-df2t";
}

void IirRealtimeProcessor::process_scalar(std::span<double> frame, std::size_t n_samples)
{
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        auto* row = frame.data() + sample * n_channels_;
        for (std::size_t channel = 0; channel < n_channels_; ++channel)
        {
            row[channel] = process_channel(row[channel], state_.data() + channel);
        }
    }
}

void IirRealtimeProcessor::process_mkl(std::span<double> frame, std::size_t n_samples)
{
#ifdef NEURALE_SIGNAL_WITH_MKL
    const auto order_value = order();
    const auto rows = mkl::checked_mkl_int(order_value + 1, "IIR state rows");
    const auto order_rows = mkl::checked_mkl_int(order_value, "IIR order");
    const auto channels = mkl::checked_mkl_int(n_channels_, "channel count");
    mkl::LocalThreadLimit thread_limit;

    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        auto* row = frame.data() + sample * n_channels_;
        cblas_dger(CblasRowMajor, rows, channels, 1.0, b_.data(), 1, row, 1, state_.data(),
                   channels);
        std::copy_n(state_.data(), n_channels_, row);
        cblas_dger(CblasRowMajor, order_rows, channels, -1.0, a_.data() + 1, 1, row, 1,
                   state_.data() + n_channels_, channels);
        std::memmove(state_.data(), state_.data() + n_channels_,
                     sizeof(double) * order_value * n_channels_);
        std::fill_n(state_.data() + order_value * n_channels_, n_channels_, 0.0);
    }
#else
    process_scalar(frame, n_samples);
#endif
}

double IirRealtimeProcessor::process_channel(double x, double* state)
{
    const auto order_value = order();
    double output = b_[0] * x;
    if (order_value > 0)
    {
        output += state[0];
    }
    for (std::size_t i = 0; i + 1 < order_value; ++i)
    {
        state[i * n_channels_] = b_[i + 1] * x - a_[i + 1] * output + state[(i + 1) * n_channels_];
    }
    if (order_value > 0)
    {
        state[(order_value - 1) * n_channels_] = b_[order_value] * x - a_[order_value] * output;
    }
    return output;
}

void IirRealtimeProcessor::copy_public_state(std::span<const double> state)
{
    std::fill(state_.begin(), state_.end(), 0.0);
    std::copy_n(state.data(), state.size(), state_.data());
}

} // namespace neurale::signal::detail
