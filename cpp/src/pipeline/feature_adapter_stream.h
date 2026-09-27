/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "adapter_support.h"

#include <neurale/streaming/frame.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/schema.h>

/// The windowed-feature adapters' shared data plane.
///
/// LMP, Hilbert-envelope and multitaper-bandpower differ in what they compute
/// and in nothing else on the data plane: each validates the frame, anchors
/// stream time on the first block, asks its processor how many observations the
/// block yields, runs it into a workspace, projects the input header onto one
/// output block and publishes.
///
/// This header is separate from `adapter_support.h` so the boundary that matters
/// most here is a file boundary: **everything in `adapter_support.h` is
/// prepare-time and may throw; everything below runs on the real-time thread and
/// may not.** An audit of what executes on the data plane opens this file and
/// stops.
///
/// The contract that follows from that:
///
/// * `process_feature_frame` is a template, not a virtual call. It is
///   instantiated once per adapter, so the kernel it drives is resolved at
///   compile time -- no vtable, no type erasure and no `std::function` is
///   introduced by sharing this code. At `-O2` each adapter's `process` is eight
///   instructions that `jmp` into its own instantiation: a tail call, no stack
///   frame.
///
///   The two indirect calls that remain in the compiled body are
///   `FrameEmitter::try_acquire` and `publish_acquired`, which are pure virtual
///   on the caller-supplied emitter (`streaming/frame_emitter.h`); nothing here
///   adds a level of indirection beyond that.
/// * It allocates nothing. Every buffer it touches was sized in `prepare()`;
///   the output frame comes from the emitter's pool. This is covered by
///   `neurale_pipeline_strict_realtime_contract_test --allocation-only`.
/// * It is `noexcept` and reports every ordinary outcome as a `StreamStatus`.
///   The one place a processor may throw is wrapped, because a kernel's failure
///   is a result here, not an error path.
///
/// :class:`FeatureStreamState` is a plain aggregate with no virtual functions.
/// Each adapter's `Impl` inherits it so that prepare-time code keeps naming the
/// fields directly; inheritance here is field reuse, not dispatch.
namespace neurale::pipeline::adapter_support
{

/// The per-stream state a windowed-feature adapter's data plane reads and advances.
///
/// Holds no processor: the kernel type differs per adapter and is passed to
/// `process_feature_frame` separately, which is what keeps this struct free of
/// any template parameter or virtual function.
struct FeatureStreamState
{
    std::unique_ptr<streaming::FrameValidator> validator;
    std::unique_ptr<streaming::StreamSchema> input_schema;
    std::unique_ptr<streaming::StreamSchema> output_schema;
    std::vector<double> output_workspace;
    std::size_t max_input_samples{};
    std::size_t max_output_observations{};
    std::size_t n_channels{};
    /// Scalars per observation. Equal to `channel_count` for a per-channel
    /// feature such as LMP, and to channels times bands where a band axis exists.
    std::size_t n_features{};
    std::uint64_t window_center_ns{};
    std::uint64_t shift_ns{};
    streaming::HostTimeNs stream_time_anchor_ns{};
    std::uint64_t emitted_observations{};
    bool timing_anchored{};

    /// Drop the stream-time anchor and the observation counter.
    ///
    /// Separate from each adapter's `clear_stream_state`, which additionally
    /// resets its own typed processor.
    void clear_timing() noexcept
    {
        timing_anchored = false;
        stream_time_anchor_ns = 0;
        emitted_observations = 0;
    }
};

/// Run one input frame through *processor* and publish the observations it yields.
///
/// *processor* needs two members: `output_count(n_samples)` and
/// `process(input_span, n_samples, output_span)`. Both are called inside the
/// one guarded region, because a kernel that refuses a block is an ordinary
/// result on this thread and must not escape as an exception.
///
/// Returns `ok` without publishing when the block yields no observation, which is
/// the normal state of a window that has not filled yet.
template <typename Processor>
[[nodiscard]] streaming::StreamStatus
process_feature_frame(FeatureStreamState& state, Processor* processor,
                      streaming::SchemaId output_schema_id, streaming::SignalId output_signal_id,
                      streaming::FrameBorrow& frame, streaming::FrameEmitter& emitter) noexcept
{
    if (processor == nullptr || state.validator == nullptr || state.output_schema == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    if (state.validator->validate(frame.view()) != streaming::FrameValidationError::none)
    {
        return streaming::StreamStatus::invalid_frame;
    }
    const auto& block = frame.blocks().front();
    if (block.n_samples > state.max_input_samples)
    {
        return streaming::StreamStatus::invalid_frame;
    }
    if (!state.timing_anchored)
    {
        if (!device_tick_host_time(block, state.stream_time_anchor_ns))
        {
            return streaming::StreamStatus::invalid_frame;
        }
        state.timing_anchored = true;
    }

    std::size_t n_observations{};
    try
    {
        n_observations = processor->output_count(block.n_samples);
        if (n_observations > state.max_output_observations)
        {
            return streaming::StreamStatus::output_limit;
        }
        const auto* input =
            reinterpret_cast<const double*>(frame.payload().data() + block.payload_offset);
        processor->process({input, static_cast<std::size_t>(block.n_samples) * state.n_channels},
                           block.n_samples,
                           {state.output_workspace.data(), n_observations * state.n_features});
    }
    catch (...)
    {
        return streaming::StreamStatus::processor_failure;
    }
    if (n_observations == 0)
    {
        return streaming::StreamStatus::ok;
    }

    std::uint64_t observation_offset_ns{};
    if (state.emitted_observations != 0 &&
        state.shift_ns > std::numeric_limits<std::uint64_t>::max() / state.emitted_observations)
    {
        return streaming::StreamStatus::invalid_frame;
    }
    observation_offset_ns = state.emitted_observations * state.shift_ns;
    if (state.window_center_ns >
            std::numeric_limits<std::uint64_t>::max() - observation_offset_ns ||
        state.stream_time_anchor_ns > std::numeric_limits<std::uint64_t>::max() -
                                          (observation_offset_ns + state.window_center_ns) ||
        n_observations > std::numeric_limits<std::uint64_t>::max() - state.emitted_observations)
    {
        return streaming::StreamStatus::invalid_frame;
    }

    streaming::FrameBorrow output{};
    auto status = emitter.try_acquire(output);
    if (status != streaming::StreamStatus::ok)
    {
        return status;
    }
    output->header() = frame.header();
    output->header().schema_id = output_schema_id;
    const auto n_scalars = n_observations * state.n_features;
    const auto payload_bytes = n_scalars * sizeof(double);
    output->block_storage()[0] = block;
    output->block_storage()[0].sample_idx_start = state.emitted_observations;
    output->block_storage()[0].device_tick_start = 0;
    output->block_storage()[0].observation_time_start_ns =
        state.stream_time_anchor_ns + observation_offset_ns + state.window_center_ns;
    output->block_storage()[0].payload_offset = 0;
    output->block_storage()[0].payload_byte_count = payload_bytes;
    output->block_storage()[0].signal_id = output_signal_id;
    output->block_storage()[0].n_samples = static_cast<std::uint32_t>(n_observations);
    std::memcpy(output->payload_storage().data(), state.output_workspace.data(), payload_bytes);
    status = output->set_used_sizes(1, payload_bytes);
    if (status != streaming::StreamStatus::ok)
    {
        return status;
    }
    status = emitter.publish_acquired();
    if (status == streaming::StreamStatus::ok)
    {
        state.emitted_observations += n_observations;
    }
    return status;
}

} // namespace neurale::pipeline::adapter_support
