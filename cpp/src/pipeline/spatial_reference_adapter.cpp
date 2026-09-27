/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "spatial_reference_adapter.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include <neurale/streaming/frame_validation.h>

#include "adapter_support.h"
#include "inplace_adapter_stream.h"
#include "spatial_realtime.h"

namespace neurale::pipeline
{
namespace
{

constexpr adapter_support::Checked kChecked{"common reference"};

[[nodiscard]] signal::detail::ReferenceStatistic
kernel_statistic(SpatialReferenceStatistic statistic) noexcept
{
    switch (statistic)
    {
    case SpatialReferenceStatistic::median:
        return signal::detail::ReferenceStatistic::median;
    case SpatialReferenceStatistic::mean:
        break;
    }
    return signal::detail::ReferenceStatistic::mean;
}

} // namespace

struct SpatialReferenceAdapter::Impl : adapter_support::InplaceStreamState
{
    Impl(std::span<const std::size_t> channels, SpatialReferenceStatistic requested)
        : reference_channels(channels.begin(), channels.end()), statistic(requested)
    {
    }

    std::vector<std::size_t> reference_channels;
    SpatialReferenceStatistic statistic{SpatialReferenceStatistic::mean};
    std::unique_ptr<signal::detail::CommonReferenceRealtimeProcessor> processor;
};

SpatialReferenceAdapter::SpatialReferenceAdapter(std::span<const std::size_t> reference_channels,
                                                 SpatialReferenceStatistic statistic)
{
    // The upper bound belongs to prepare(), which is the first point that knows
    // the channel count; duplicates are wrong regardless of geometry.
    for (std::size_t i = 0; i < reference_channels.size(); ++i)
    {
        if (std::find(reference_channels.begin(), reference_channels.begin() + i,
                      reference_channels[i]) != reference_channels.begin() + i)
        {
            throw std::invalid_argument(
                "common reference adapter channels must not contain duplicates");
        }
    }
    impl_ = std::make_unique<Impl>(reference_channels, statistic);
}

SpatialReferenceAdapter::~SpatialReferenceAdapter() = default;

streaming::PreparedProcessorContract
SpatialReferenceAdapter::prepare(const streaming::ProcessorPrepareContext& context)
{
    const auto& signal_schema =
        adapter_support::sampled_inplace_input(kChecked, context, impl_->prepared_schema.get());
    // Not shared: an empty channel set means "reference against every channel"
    // in the kernel, so a zero-channel signal is the one geometry that leaves it
    // with nothing to average, and only this adapter carries channel indices
    // that the signal has to be wide enough for.
    if (signal_schema.n_channels == 0)
    {
        throw std::invalid_argument("common reference adapter requires at least one channel");
    }
    for (const auto channel : impl_->reference_channels)
    {
        if (channel >= signal_schema.n_channels)
        {
            throw std::invalid_argument(
                "common reference adapter channel index exceeds the signal channel count");
        }
    }

    if (impl_->processor == nullptr)
    {
        impl_->processor = std::make_unique<signal::detail::CommonReferenceRealtimeProcessor>(
            signal_schema.n_channels, impl_->reference_channels,
            kernel_statistic(impl_->statistic));
        impl_->validator = std::make_unique<streaming::FrameValidator>(context.input_schema);
        impl_->prepared_schema =
            std::make_unique<streaming::StreamSchema>(context.input_schema.clone());
        impl_->n_channels = signal_schema.n_channels;
    }
    else
    {
        impl_->processor->reset();
    }

    auto workspace_bytes =
        kChecked.checked_multiply(impl_->reference_channels.size(), sizeof(std::size_t));
    if (impl_->statistic == SpatialReferenceStatistic::median)
    {
        // n_reference_channels() rather than reference_channels.size(): a
        // whole-set median keeps no index vector but still sizes a workspace of
        // one double per channel.
        workspace_bytes = kChecked.checked_add(
            workspace_bytes,
            kChecked.checked_multiply(impl_->processor->n_reference_channels(), sizeof(double)));
    }
    return adapter_support::forwarding_contract(context.input_schema, workspace_bytes);
}

streaming::StreamStatus SpatialReferenceAdapter::process(streaming::FrameBorrow& frame,
                                                         streaming::FrameEmitter& emitter) noexcept
{
    return adapter_support::process_inplace_frame(
        *impl_, impl_->processor != nullptr, frame, emitter,
        [this](std::span<double> samples, std::size_t n_samples)
        { impl_->processor->process(samples, n_samples); });
}

streaming::StreamStatus
SpatialReferenceAdapter::handle_discontinuity(const streaming::Discontinuity&) noexcept
{
    if (impl_->processor == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->processor->reset();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus SpatialReferenceAdapter::flush(streaming::FrameEmitter&) noexcept
{
    return impl_->processor == nullptr ? streaming::StreamStatus::invalid_state
                                       : streaming::StreamStatus::ok;
}

streaming::StreamStatus SpatialReferenceAdapter::reset() noexcept
{
    if (impl_->processor == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->processor->reset();
    return streaming::StreamStatus::ok;
}

} // namespace neurale::pipeline
