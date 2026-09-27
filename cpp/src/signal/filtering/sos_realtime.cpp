/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "sos_realtime.h"
#include "realtime_utils.h"

#ifdef NEURALE_SIGNAL_WITH_MKL
#include "mkl_utils.h"
#include <mkl_cblas.h>
#endif

#include <algorithm>
#include <cstring>

namespace neurale::signal::detail
{

SosRealtimeProcessor::SosRealtimeProcessor(std::span<const double> sos, std::size_t n_channels,
                                           std::span<const double> state)
    : n_sections_(sos.size() / 6), n_channels_(n_channels), sos_(sos.begin(), sos.end())
{
#ifdef NEURALE_SIGNAL_WITH_MKL
    if (use_mkl_realtime_blas(n_channels_))
    {
        kernel_ = Kernel::mkl_blas_df2t;
    }
#endif
    state_.assign(n_sections_ * state_rows() * n_channels_, 0.0);
    copy_public_state(state);
}

void SosRealtimeProcessor::process(std::span<double> frame, std::size_t n_samples)
{
    if (kernel_ == Kernel::mkl_blas_df2t)
    {
        process_mkl(frame, n_samples);
        return;
    }
    process_scalar(frame, n_samples);
}

void SosRealtimeProcessor::reset()
{
    std::fill(state_.begin(), state_.end(), 0.0);
}

void SosRealtimeProcessor::set_state(std::span<const double> state)
{
    copy_public_state(state);
}

void SosRealtimeProcessor::state(std::span<double> output) const
{
    if (kernel_ != Kernel::mkl_blas_df2t)
    {
        std::copy(state_.begin(), state_.end(), output.begin());
        return;
    }
    for (std::size_t section = 0; section < n_sections_; ++section)
    {
        const auto* source = section_state(section);
        auto* target = output.data() + section * 2 * n_channels_;
        std::copy_n(source, 2 * n_channels_, target);
    }
}

std::size_t SosRealtimeProcessor::n_sections() const noexcept
{
    return n_sections_;
}

std::size_t SosRealtimeProcessor::n_channels() const noexcept
{
    return n_channels_;
}

std::string_view SosRealtimeProcessor::kernel_name() const noexcept
{
    return kernel_ == Kernel::mkl_blas_df2t ? "mkl-blas-df2t" : "builtin-df2t";
}

void SosRealtimeProcessor::process_scalar(std::span<double> frame, std::size_t n_samples)
{
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        auto* row = frame.data() + sample * n_channels_;
        for (std::size_t section = 0; section < n_sections_; ++section)
        {
            const auto* coefs = sos_.data() + section * 6;
            auto* section_state = this->section_state(section);
            for (std::size_t channel = 0; channel < n_channels_; ++channel)
            {
                process_channel(row[channel], coefs, section_state + channel);
            }
        }
    }
}

void SosRealtimeProcessor::process_mkl(std::span<double> frame, std::size_t n_samples)
{
#ifdef NEURALE_SIGNAL_WITH_MKL
    const auto channels = mkl::checked_mkl_int(n_channels_, "channel count");
    mkl::LocalThreadLimit thread_limit;

    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        auto* row = frame.data() + sample * n_channels_;
        for (std::size_t section = 0; section < n_sections_; ++section)
        {
            const auto* coefs = sos_.data() + section * 6;
            auto* state = section_state(section);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, channels, 1, 1.0, coefs, 1,
                        row, channels, 1.0, state, channels);
            std::copy_n(state, n_channels_, row);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, channels, 1, -1.0, coefs + 4,
                        1, row, channels, 1.0, state + n_channels_, channels);
            std::memmove(state, state + n_channels_, sizeof(double) * 2 * n_channels_);
            std::fill_n(state + 2 * n_channels_, n_channels_, 0.0);
        }
    }
#else
    process_scalar(frame, n_samples);
#endif
}

void SosRealtimeProcessor::process_channel(double& value, const double* coefs, double* state)
{
    const auto x = value;
    const auto output = coefs[0] * x + state[0];
    state[0] = coefs[1] * x - coefs[4] * output + state[n_channels_];
    state[n_channels_] = coefs[2] * x - coefs[5] * output;
    value = output;
}

void SosRealtimeProcessor::copy_public_state(std::span<const double> state)
{
    std::fill(state_.begin(), state_.end(), 0.0);
    if (state.empty())
    {
        // No external state supplied -- state_ is already zero-initialised by
        // the fill above, which is the correct quiescent initial state. The
        // mkl_blas_df2t branch below reads a fixed 2 * n_channels_ per section
        // from state.data() and would dereference null for an empty span, so
        // bail out before touching the pointer.
        return;
    }
    if (kernel_ != Kernel::mkl_blas_df2t)
    {
        std::copy(state.begin(), state.end(), state_.begin());
        return;
    }
    for (std::size_t section = 0; section < n_sections_; ++section)
    {
        const auto* source = state.data() + section * 2 * n_channels_;
        auto* target = section_state(section);
        std::copy_n(source, 2 * n_channels_, target);
    }
}

std::size_t SosRealtimeProcessor::state_rows() const noexcept
{
    return kernel_ == Kernel::mkl_blas_df2t ? 3 : 2;
}

double* SosRealtimeProcessor::section_state(std::size_t section) noexcept
{
    return state_.data() + section * state_rows() * n_channels_;
}

const double* SosRealtimeProcessor::section_state(std::size_t section) const noexcept
{
    return state_.data() + section * state_rows() * n_channels_;
}

} // namespace neurale::signal::detail
