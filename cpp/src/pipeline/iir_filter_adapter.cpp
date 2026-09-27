/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "iir_filter_adapter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <neurale/streaming/frame_validation.h>

#include "adapter_support.h"
#include "iir_realtime.h"
#include "inplace_adapter_stream.h"

namespace neurale::pipeline
{
namespace
{

constexpr adapter_support::Checked kChecked{"IIR"};

[[nodiscard]] std::size_t checked_workspace_bytes(std::size_t n_coefs, std::size_t n_channels)
{
    if (n_coefs > std::numeric_limits<std::size_t>::max() / 2 + 1)
    {
        throw std::overflow_error("IIR adapter workspace size overflows size_t");
    }
    const auto rows = n_coefs + n_coefs - 1;
    if ((n_channels != 0 && rows > std::numeric_limits<std::size_t>::max() / n_channels) ||
        rows * n_channels > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        throw std::overflow_error("IIR adapter workspace size overflows size_t");
    }
    return rows * n_channels * sizeof(double);
}

} // namespace

struct IirFilterAdapter::Impl : adapter_support::InplaceStreamState
{
    Impl(std::span<const double> numerator, std::span<const double> denominator)
        : b(numerator.begin(), numerator.end()), a(denominator.begin(), denominator.end())
    {
    }

    std::vector<double> b;
    std::vector<double> a;
    std::unique_ptr<signal::detail::IirRealtimeProcessor> processor;
};

IirFilterAdapter::IirFilterAdapter(std::span<const double> b, std::span<const double> a)
{
    const auto finite = [](std::span<const double> values)
    {
        return std::all_of(values.begin(), values.end(),
                           [](double value) { return std::isfinite(value); });
    };
    if (b.empty() || b.size() != a.size() || !finite(b) || !finite(a))
    {
        throw std::invalid_argument(
            "IIR adapter b and a must be finite, nonempty, and equal-sized");
    }
    if (a.front() != 1.0)
    {
        throw std::invalid_argument("IIR adapter denominator must be normalized");
    }
    impl_ = std::make_unique<Impl>(b, a);
}

IirFilterAdapter::~IirFilterAdapter() = default;

streaming::PreparedProcessorContract
IirFilterAdapter::prepare(const streaming::ProcessorPrepareContext& context)
{
    const auto& signal_schema =
        adapter_support::sampled_inplace_input(kChecked, context, impl_->prepared_schema.get());
    const auto workspace_bytes = checked_workspace_bytes(impl_->b.size(), signal_schema.n_channels);

    if (impl_->processor == nullptr)
    {
        const auto state_size = (impl_->b.size() - 1) * signal_schema.n_channels;
        std::vector<double> initial_state(state_size, 0.0);
        impl_->processor = std::make_unique<signal::detail::IirRealtimeProcessor>(
            impl_->b, impl_->a, signal_schema.n_channels, initial_state);
        impl_->validator = std::make_unique<streaming::FrameValidator>(context.input_schema);
        impl_->prepared_schema =
            std::make_unique<streaming::StreamSchema>(context.input_schema.clone());
        impl_->n_channels = signal_schema.n_channels;
    }
    else
    {
        impl_->processor->reset();
    }

    return adapter_support::forwarding_contract(context.input_schema, workspace_bytes);
}

streaming::StreamStatus IirFilterAdapter::process(streaming::FrameBorrow& frame,
                                                  streaming::FrameEmitter& emitter) noexcept
{
    return adapter_support::process_inplace_frame(
        *impl_, impl_->processor != nullptr, frame, emitter,
        [this](std::span<double> samples, std::size_t n_samples)
        { impl_->processor->process(samples, n_samples); });
}

streaming::StreamStatus
IirFilterAdapter::handle_discontinuity(const streaming::Discontinuity&) noexcept
{
    return reset();
}

streaming::StreamStatus IirFilterAdapter::flush(streaming::FrameEmitter&) noexcept
{
    return impl_->processor == nullptr ? streaming::StreamStatus::invalid_state
                                       : streaming::StreamStatus::ok;
}

streaming::StreamStatus IirFilterAdapter::reset() noexcept
{
    if (impl_->processor == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    try
    {
        impl_->processor->reset();
    }
    catch (...)
    {
        return streaming::StreamStatus::processor_failure;
    }
    return streaming::StreamStatus::ok;
}

} // namespace neurale::pipeline
