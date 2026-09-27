/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "fir_filter_adapter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <neurale/streaming/frame_validation.h>

#include "adapter_support.h"
#include "fir_realtime.h"
#include "inplace_adapter_stream.h"

namespace neurale::pipeline
{
namespace
{

constexpr adapter_support::Checked kChecked{"FIR"};

[[nodiscard]] std::size_t checked_workspace_bytes(std::size_t n_taps, std::size_t n_channels)
{
    if (n_channels > (std::numeric_limits<std::size_t>::max() - 1) / 4 ||
        n_taps > std::numeric_limits<std::size_t>::max() / (1 + 4 * n_channels) ||
        n_taps * (1 + 4 * n_channels) > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        throw std::overflow_error("FIR adapter workspace size overflows size_t");
    }
    return n_taps * (1 + 4 * n_channels) * sizeof(double);
}

} // namespace

struct FirFilterAdapter::Impl : adapter_support::InplaceStreamState
{
    explicit Impl(std::span<const double> coefs) : taps(coefs.begin(), coefs.end()) {}

    std::vector<double> taps;
    std::unique_ptr<signal::detail::FirRealtimeProcessor> processor;
};

FirFilterAdapter::FirFilterAdapter(std::span<const double> taps)
{
    if (taps.empty() ||
        !std::all_of(taps.begin(), taps.end(), [](double value) { return std::isfinite(value); }))
    {
        throw std::invalid_argument("FIR adapter coefficients must be nonempty and finite");
    }
    impl_ = std::make_unique<Impl>(taps);
}

FirFilterAdapter::~FirFilterAdapter() = default;

streaming::PreparedProcessorContract
FirFilterAdapter::prepare(const streaming::ProcessorPrepareContext& context)
{
    const auto& signal_schema =
        adapter_support::sampled_inplace_input(kChecked, context, impl_->prepared_schema.get());
    const auto workspace_bytes =
        checked_workspace_bytes(impl_->taps.size(), signal_schema.n_channels);

    if (impl_->processor == nullptr)
    {
        const auto state_size = (impl_->taps.size() - 1) * signal_schema.n_channels;
        std::vector<double> initial_state(state_size, 0.0);
        impl_->processor = std::make_unique<signal::detail::FirRealtimeProcessor>(
            impl_->taps, signal_schema.n_channels, initial_state);
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

streaming::StreamStatus FirFilterAdapter::process(streaming::FrameBorrow& frame,
                                                  streaming::FrameEmitter& emitter) noexcept
{
    return adapter_support::process_inplace_frame(
        *impl_, impl_->processor != nullptr, frame, emitter,
        [this](std::span<double> samples, std::size_t n_samples)
        {
            // One sample per call, deliberately, and not an unoptimised way of
            // spelling a block call. FirRealtimeProcessor::process() switches to
            // its block path once `should_process_block(n_samples)` holds, and
            // that path sizes `block_input_`, `block_state_` and
            // `block_final_state_` with resize() inside process() -- a heap
            // allocation on the real-time thread, which the strict-realtime
            // allocation gate forbids. `should_process_block(1)` is false for
            // every tap count, so calling per sample is what keeps the block
            // path unreachable from here.
            //
            // Switching to a block call requires giving the kernel a
            // prepare-time bounded workspace first; until then this loop is
            // load-bearing.
            for (std::size_t sample = 0; sample < n_samples; ++sample)
            {
                impl_->processor->process(
                    {samples.data() + sample * impl_->n_channels, impl_->n_channels}, 1);
            }
        });
}

streaming::StreamStatus
FirFilterAdapter::handle_discontinuity(const streaming::Discontinuity&) noexcept
{
    return reset();
}

streaming::StreamStatus FirFilterAdapter::flush(streaming::FrameEmitter&) noexcept
{
    return impl_->processor == nullptr ? streaming::StreamStatus::invalid_state
                                       : streaming::StreamStatus::ok;
}

streaming::StreamStatus FirFilterAdapter::reset() noexcept
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
