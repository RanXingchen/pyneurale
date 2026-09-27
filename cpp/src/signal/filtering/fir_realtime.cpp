/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "fir_realtime.h"
#include "fir_dispatch.h"

#ifdef NEURALE_SIGNAL_WITH_MKL
#include "mkl_utils.h"
#include <mkl_cblas.h>
#endif

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace neurale::signal::detail
{
namespace
{

constexpr std::size_t kMklMinChannels = 5;
constexpr std::size_t kMklMinSingleChannelTaps = 32;
constexpr std::size_t kMklChannelDotMaxChannels = 4;

} // namespace

FirRealtimeProcessor::FirRealtimeProcessor(std::span<const double> taps, std::size_t n_channels,
                                           std::span<const double> state)
    : n_channels_(n_channels), taps_(taps.begin(), taps.end())
{
    if (taps_.empty() || n_channels_ == 0 || state.size() != order() * n_channels_)
    {
        throw std::invalid_argument("FIR realtime coefficient/state mismatch");
    }
#ifdef NEURALE_SIGNAL_WITH_MKL
    if (taps_.size() > 1 && n_channels_ <= kMklChannelDotMaxChannels &&
        taps_.size() >= kMklMinSingleChannelTaps)
    {
        kernel_ = Kernel::mkl_channel_dot_ring;
    }
    else if (taps_.size() >= kMklMinSingleChannelTaps && n_channels_ >= kMklMinChannels)
    {
        kernel_ = Kernel::mkl_blas_ring;
    }
#endif
    ring_.assign(2 * taps_.size() * n_channels_, 0.0);
    if (kernel_ == Kernel::mkl_channel_dot_ring)
    {
        channel_ring_.assign(2 * taps_.size() * n_channels_, 0.0);
    }
#ifdef NEURALE_SIGNAL_WITH_MKL
    mkl_taps_ = mkl::checked_mkl_int(taps_.size(), "tap count");
    mkl_channels_ = mkl::checked_mkl_int(n_channels_, "channel count");
#endif
    copy_public_state(state);
}

void FirRealtimeProcessor::process(std::span<double> frame, std::size_t n_samples)
{
    if (frame.size() != n_samples * n_channels_)
    {
        throw std::invalid_argument("FIR realtime frame shape mismatch");
    }
    if (should_process_block(n_samples))
    {
        process_block(frame, n_samples);
        return;
    }
    if (kernel_ == Kernel::mkl_channel_dot_ring)
    {
        process_mkl_channel_dot(frame, n_samples);
    }
    else if (kernel_ == Kernel::mkl_blas_ring)
    {
        process_mkl(frame, n_samples);
    }
    else
    {
        process_builtin(frame, n_samples);
    }
}

void FirRealtimeProcessor::reset()
{
    std::fill(ring_.begin(), ring_.end(), 0.0);
    std::fill(channel_ring_.begin(), channel_ring_.end(), 0.0);
    position_ = 0;
}

void FirRealtimeProcessor::set_state(std::span<const double> state)
{
    copy_public_state(state);
}

void FirRealtimeProcessor::state(std::span<double> output) const
{
    copy_current_state(output);
}

std::size_t FirRealtimeProcessor::order() const noexcept
{
    return taps_.size() - 1;
}

std::size_t FirRealtimeProcessor::n_channels() const noexcept
{
    return n_channels_;
}

std::string_view FirRealtimeProcessor::kernel_name() const noexcept
{
    switch (kernel_)
    {
    case Kernel::builtin_ring:
        return "builtin-ring";
    case Kernel::mkl_channel_dot_ring:
        return "mkl-channel-dot-ring";
    case Kernel::mkl_blas_ring:
        return "mkl-blas-ring";
    }
    return "unknown";
}

void FirRealtimeProcessor::process_builtin(std::span<double> frame, std::size_t n_samples)
{
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        process_builtin_sample(frame.data() + sample * n_channels_);
    }
}

void FirRealtimeProcessor::process_mkl(std::span<double> frame, std::size_t n_samples)
{
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        process_mkl_sample(frame.data() + sample * n_channels_);
    }
}

void FirRealtimeProcessor::process_mkl_channel_dot(std::span<double> frame, std::size_t n_samples)
{
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        process_mkl_channel_dot_sample(frame.data() + sample * n_channels_);
    }
}

void FirRealtimeProcessor::process_block(std::span<double> frame, std::size_t n_samples)
{
    const auto kernel = select_fir_kernel(n_samples, n_channels_, taps_.size());
    std::span<const double> x = frame;
    if (kernel == FirKernel::builtin_direct)
    {
        block_input_.resize(frame.size());
        std::copy(frame.begin(), frame.end(), block_input_.begin());
        x = block_input_;
    }

    const auto state_size = order() * n_channels_;
    block_state_.resize(state_size);
    block_final_state_.resize(state_size);
    copy_current_state(block_state_);

    fir_filter_with_kernel(x, n_samples, n_channels_, taps_, block_state_, frame,
                           block_final_state_, kernel);
    copy_public_state(block_final_state_);
}

void FirRealtimeProcessor::process_builtin_sample(double* row)
{
    auto* current = ring_.data() + position_ * n_channels_;
    std::copy_n(row, n_channels_, current);
    std::copy_n(row, n_channels_, current + taps_.size() * n_channels_);

    if (n_channels_ == 1)
    {
        row[0] = std::inner_product(taps_.begin(), taps_.end(),
                                    ring_.begin() + static_cast<std::ptrdiff_t>(position_), 0.0);
        advance();
        return;
    }

    std::fill_n(row, n_channels_, 0.0);
    const auto* history = ring_.data() + position_ * n_channels_;
    for (std::size_t tap = 0; tap < taps_.size(); ++tap)
    {
        const double coef = taps_[tap];
        const auto* values = history + tap * n_channels_;
        for (std::size_t channel = 0; channel < n_channels_; ++channel)
        {
            row[channel] += coef * values[channel];
        }
    }
    advance();
}

void FirRealtimeProcessor::process_mkl_channel_dot_sample(double* row)
{
#ifdef NEURALE_SIGNAL_WITH_MKL
    const auto stride = 2 * taps_.size();
    for (std::size_t channel = 0; channel < n_channels_; ++channel)
    {
        auto* current = channel_ring_.data() + channel * stride + position_;
        current[0] = row[channel];
        current[taps_.size()] = row[channel];
        row[channel] = cblas_ddot(mkl_taps_, current, 1, taps_.data(), 1);
    }
    advance();
#else
    process_builtin_sample(row);
#endif
}

void FirRealtimeProcessor::process_mkl_sample(double* row)
{
#ifdef NEURALE_SIGNAL_WITH_MKL
    auto* current = ring_.data() + position_ * n_channels_;
    std::copy_n(row, n_channels_, current);
    std::copy_n(row, n_channels_, current + taps_.size() * n_channels_);

    if (n_channels_ == 1)
    {
        row[0] = cblas_ddot(mkl_taps_, taps_.data(), 1, ring_.data() + position_, 1);
        advance();
        return;
    }
    cblas_dgemv(CblasRowMajor, CblasTrans, mkl_taps_, mkl_channels_, 1.0,
                ring_.data() + position_ * n_channels_, mkl_channels_, taps_.data(), 1, 0.0, row,
                1);
    advance();
#else
    process_builtin_sample(row);
#endif
}

bool FirRealtimeProcessor::should_process_block(std::size_t n_samples) const noexcept
{
    return fir_block_path_useful(n_samples, taps_.size());
}

void FirRealtimeProcessor::advance() noexcept
{
    position_ = position_ == 0 ? taps_.size() - 1 : position_ - 1;
}

void FirRealtimeProcessor::copy_public_state(std::span<const double> state)
{
    if (state.size() != order() * n_channels_)
    {
        throw std::invalid_argument("FIR realtime state shape mismatch");
    }
    reset();
    const auto history = order();
    if (kernel_ == Kernel::mkl_channel_dot_ring)
    {
        const auto stride = 2 * taps_.size();
        for (std::size_t row = 0; row < history; ++row)
        {
            const auto offset = history - row;
            const auto* source = state.data() + row * n_channels_;
            for (std::size_t channel = 0; channel < n_channels_; ++channel)
            {
                auto* target = channel_ring_.data() + channel * stride + offset;
                target[0] = source[channel];
                target[taps_.size()] = source[channel];
            }
        }
    }
    else
    {
        for (std::size_t row = 0; row < history; ++row)
        {
            const auto offset = history - row;
            auto* target = ring_.data() + offset * n_channels_;
            const auto* source = state.data() + row * n_channels_;
            std::copy_n(source, n_channels_, target);
            std::copy_n(source, n_channels_, target + taps_.size() * n_channels_);
        }
    }
}

void FirRealtimeProcessor::copy_current_state(std::span<double> output) const
{
    if (output.size() != order() * n_channels_)
    {
        throw std::invalid_argument("FIR realtime state output mismatch");
    }
    if (kernel_ == Kernel::mkl_channel_dot_ring)
    {
        copy_channel_major_state(output);
    }
    else
    {
        copy_sample_major_state(output);
    }
}

void FirRealtimeProcessor::copy_sample_major_state(std::span<double> output) const
{
    const auto history = order();
    for (std::size_t row = 0; row < history; ++row)
    {
        const auto offset = history - row;
        const auto* source = ring_.data() + (position_ + offset) * n_channels_;
        std::copy_n(source, n_channels_, output.data() + row * n_channels_);
    }
}

void FirRealtimeProcessor::copy_channel_major_state(std::span<double> output) const
{
    const auto history = order();
    const auto stride = 2 * taps_.size();
    for (std::size_t row = 0; row < history; ++row)
    {
        const auto offset = history - row;
        auto* target = output.data() + row * n_channels_;
        for (std::size_t channel = 0; channel < n_channels_; ++channel)
        {
            target[channel] = channel_ring_[channel * stride + position_ + offset];
        }
    }
}

} // namespace neurale::signal::detail
