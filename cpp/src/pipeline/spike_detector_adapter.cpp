/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "spike_detector_adapter.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "adapter_support.h"

#include <neurale/streaming/frame_validation.h>

namespace neurale::pipeline
{
namespace
{

using adapter_support::scaled_duration;

[[nodiscard]] bool block_time_seconds(const streaming::SignalBlockHeader& block,
                                      double& result) noexcept
{
    const auto& sync = block.clock_sync;
    if (!streaming::has_flag(sync.flags, streaming::ClockSyncFlags::synchronized) ||
        sync.generation == 0 || sync.device_tick_rate.numerator == 0 ||
        sync.device_tick_rate.denominator == 0)
        return false;
    std::uint64_t duration{};
    std::uint64_t host_ns{};
    if (block.device_tick_start >= sync.device_tick_reference)
    {
        if (!scaled_duration(block.device_tick_start - sync.device_tick_reference,
                             sync.device_tick_rate, false, duration) ||
            duration > std::numeric_limits<std::uint64_t>::max() - sync.host_time_reference_ns)
            return false;
        host_ns = sync.host_time_reference_ns + duration;
    }
    else
    {
        if (!scaled_duration(sync.device_tick_reference - block.device_tick_start,
                             sync.device_tick_rate, true, duration) ||
            duration > sync.host_time_reference_ns)
            return false;
        host_ns = sync.host_time_reference_ns - duration;
    }
    result = static_cast<double>(host_ns) * 1e-9;
    return std::isfinite(result);
}

} // namespace

struct SpikeDetectorAdapter::Impl
{
    explicit Impl(SpikeDetectorAdapterConfig value) : config(std::move(value)) {}
    SpikeDetectorAdapterConfig config;
    std::unique_ptr<sorting::SpikeBlockLayout> layout;
    std::unique_ptr<sorting::OnlineThresholdDetector> detector;
    std::unique_ptr<streaming::FrameValidator> validator;
    std::unique_ptr<streaming::StreamSchema> input_schema;
    std::unique_ptr<streaming::StreamSchema> output_schema;
    streaming::RationalRate fs{};
    std::size_t n_channels{};
    std::uint64_t segment_id{};
    streaming::FrameHeader last_header{};
    streaming::SignalBlockHeader last_block{};
    std::uint64_t next_output_sequence{};
    bool has_last_metadata{};

    // Control-plane reset of the cached output metadata. Called from
    // reset/handle_discontinuity/flush after the detector itself is reset.
    void reset_metadata() noexcept
    {
        last_header = {};
        last_block = {};
        next_output_sequence = 0;
        has_last_metadata = false;
    }
};

SpikeDetectorAdapter::SpikeDetectorAdapter(SpikeDetectorAdapterConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
    if (impl_->config.output_schema_id == 0 || impl_->config.output_signal_id == 0 ||
        impl_->config.block_capacity == 0 || impl_->config.channel_centers.empty() ||
        impl_->config.channel_centers.size() != impl_->config.channel_thresholds.size() ||
        impl_->config.post_samples == std::numeric_limits<std::size_t>::max() ||
        impl_->config.pre_samples >
            std::numeric_limits<std::size_t>::max() - impl_->config.post_samples - 1)
        throw std::invalid_argument("invalid online spike detector adapter configuration");
}

SpikeDetectorAdapter::~SpikeDetectorAdapter() = default;

streaming::PreparedProcessorContract
SpikeDetectorAdapter::prepare(const streaming::ProcessorPrepareContext& context)
{
    const auto signals = context.input_schema.signals();
    if (signals.size() != 1)
        throw std::invalid_argument("spike detector requires one input");
    const auto& input = signals.front();
    if (input.kind != streaming::SignalKind::sampled ||
        input.dtype != streaming::SignalDType::float64 ||
        input.layout != streaming::SignalLayout::sample_major ||
        input.n_channels != impl_->config.channel_centers.size())
        throw std::invalid_argument(
            "spike detector requires matching sample-major float64 input channels");
    if (context.max_process_outputs < 1 || context.max_flush_outputs < 1 ||
        context.available_frame_pool_leases < 1)
        throw std::invalid_argument(
            "spike detector requires one process output, one flush output, and one frame lease");
    if (impl_->input_schema && !impl_->input_schema->equivalent(context.input_schema))
        throw std::invalid_argument("spike detector schema cannot change after prepare");

    const auto waveform_samples = impl_->config.pre_samples + 1 + impl_->config.post_samples;
    auto layout = std::make_unique<sorting::SpikeBlockLayout>(impl_->config.block_capacity,
                                                              waveform_samples, input.n_channels);
    const auto output_signal = streaming::SignalSchema{
        impl_->config.output_signal_id,
        streaming::SignalDType::float64,
        input.n_channels,
        static_cast<std::uint32_t>(impl_->config.block_capacity),
        static_cast<std::uint32_t>(impl_->config.block_capacity),
        {0, 1},
        input.clock_domain,
        streaming::SignalLayout::sample_major,
        streaming::DeviceTickTracking::unavailable,
        input.physical_unit,
        input.channel_set_id,
        input.calibration_id,
        input.reference_id,
        streaming::SignalKind::spike,
        0,
        streaming::ObservationTiming::not_applicable,
        layout->payload_bytes(),
    };
    const std::array output_signals{output_signal};
    auto declared_output = streaming::StreamSchema{impl_->config.output_schema_id, output_signals};
    if (!impl_->detector)
    {
        impl_->detector = std::make_unique<sorting::OnlineThresholdDetector>(
            sorting::OnlineThresholdDetectorConfig{
                .max_input_samples = input.max_block_samples,
                .block_capacity = impl_->config.block_capacity,
                .refractory_samples = impl_->config.refractory_samples,
                .alignment_search_radius = impl_->config.alignment_search_radius,
                .pre_samples = impl_->config.pre_samples,
                .post_samples = impl_->config.post_samples,
                .polarity = impl_->config.polarity,
                .boundary_behavior = impl_->config.boundary_behavior,
                .overflow_policy = impl_->config.overflow_policy,
                .channel_centers = impl_->config.channel_centers,
                .channel_thresholds = impl_->config.channel_thresholds,
                .electrode_groups = impl_->config.electrode_groups,
            });
        impl_->layout = std::move(layout);
        impl_->validator = std::make_unique<streaming::FrameValidator>(context.input_schema);
        impl_->input_schema =
            std::make_unique<streaming::StreamSchema>(context.input_schema.clone());
        impl_->output_schema = std::make_unique<streaming::StreamSchema>(declared_output.clone());
        impl_->fs = input.fs;
        impl_->n_channels = input.n_channels;
    }
    else if (!impl_->output_schema->equivalent(declared_output))
        throw std::invalid_argument("spike detector output schema changed after prepare");
    impl_->segment_id = 0;
    impl_->last_header = {};
    impl_->last_block = {};
    impl_->next_output_sequence = 0;
    impl_->has_last_metadata = false;
    impl_->detector->reset(0);
    // The detector owns its full fixed workspace (input buffer, pending/complete
    // event buffers, waveform scratch, per-channel/per-group state, and the
    // configuration vectors). Ask it for that reservation directly instead of
    // re-deriving a partial buffer+waveform estimate here.
    return {
        .accepted_input_schema = context.input_schema.clone(),
        .output_schema = declared_output.clone(),
        .max_process_outputs_per_input = 1,
        .max_flush_outputs = 1,
        .can_forward_input = false,
        .required_resources =
            streaming::ProcessorResourceBounds{
                .workspace_bytes = impl_->detector->workspace_bytes(),
                .frame_pool_leases = 1,
            },
    };
}

streaming::StreamStatus SpikeDetectorAdapter::process(streaming::FrameBorrow& frame,
                                                      streaming::FrameEmitter& emitter) noexcept
{
    if (!impl_->detector || !impl_->validator || !impl_->layout || !impl_->output_schema)
        return streaming::StreamStatus::invalid_state;
    if (impl_->validator->validate(frame.view()) != streaming::FrameValidationError::none)
        return streaming::StreamStatus::invalid_frame;
    const auto& input_block = frame.blocks().front();
    double time_start{};
    if (!block_time_seconds(input_block, time_start))
        return streaming::StreamStatus::invalid_frame;
    if (input_block.sample_idx_start >
        static_cast<streaming::SampleIndex>(std::numeric_limits<std::int64_t>::max()))
        return streaming::StreamStatus::invalid_frame;
    streaming::FrameBorrow output{};
    auto status = emitter.try_acquire(output);
    if (status != streaming::StreamStatus::ok)
        return status;
    if (output.payload_storage().size() < impl_->layout->payload_bytes())
        return streaming::StreamStatus::buffer_exhausted;
    try
    {
        sorting::FixedCapacitySpikeBlock block{
            output.payload_storage().first(impl_->layout->payload_bytes()),
            *impl_->layout,
            impl_->config.pre_samples,
            impl_->config.post_samples,
            impl_->segment_id,
            impl_->config.polarity};
        const auto* values =
            reinterpret_cast<const double*>(frame.payload().data() + input_block.payload_offset);
        const auto detector_status = impl_->detector->process(
            {values, static_cast<std::size_t>(input_block.n_samples) * impl_->n_channels},
            input_block.n_samples, static_cast<std::int64_t>(input_block.sample_idx_start),
            time_start,
            static_cast<double>(impl_->fs.numerator) / static_cast<double>(impl_->fs.denominator),
            impl_->segment_id, block);
        if (detector_status == sorting::OnlineDetectionStatus::overflow)
            return streaming::StreamStatus::output_limit;
        if (detector_status == sorting::OnlineDetectionStatus::invalid_input)
            return streaming::StreamStatus::invalid_frame;
        if (detector_status == sorting::OnlineDetectionStatus::boundary_error)
            return streaming::StreamStatus::processor_failure;

        if (frame.header().sequence == std::numeric_limits<std::uint64_t>::max())
            return streaming::StreamStatus::invalid_frame;
        impl_->last_header = frame.header();
        impl_->last_block = input_block;
        impl_->next_output_sequence = frame.header().sequence + 1;
        impl_->has_last_metadata = true;

        output.header() = frame.header();
        output.header().schema_id = impl_->output_schema->id();
        output.block_storage()[0] = streaming::SignalBlockHeader{
            .sample_idx_start = block.header().n_valid != 0
                                    ? static_cast<streaming::SampleIndex>(block.sample_indices()[0])
                                    : input_block.sample_idx_start,
            .device_tick_start = input_block.device_tick_start,
            .payload_offset = 0,
            .payload_byte_count = impl_->layout->payload_bytes(),
            .signal_id = impl_->config.output_signal_id,
            .n_samples = block.header().n_valid,
            .clock_sync = input_block.clock_sync,
            .last_sample_idx = block.header().n_valid != 0 ? static_cast<streaming::SampleIndex>(
                                                                 block.sample_indices().back())
                                                           : 0,
        };
        status = output.set_used_sizes(1, impl_->layout->payload_bytes());
        if (status != streaming::StreamStatus::ok)
            return status;
        return emitter.publish_acquired();
    }
    catch (...)
    {
        return streaming::StreamStatus::processor_failure;
    }
}

streaming::StreamStatus
SpikeDetectorAdapter::handle_discontinuity(const streaming::Discontinuity&) noexcept
{
    if (!impl_->detector)
        return streaming::StreamStatus::invalid_state;
    if (impl_->segment_id == std::numeric_limits<std::uint64_t>::max())
        return streaming::StreamStatus::processor_failure;
    ++impl_->segment_id;
    impl_->detector->reset(impl_->segment_id);
    impl_->reset_metadata();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus SpikeDetectorAdapter::flush(streaming::FrameEmitter& emitter) noexcept
{
    if (!impl_->detector)
        return streaming::StreamStatus::invalid_state;
    if (!impl_->has_last_metadata)
    {
        impl_->detector->reset(impl_->segment_id);
        return streaming::StreamStatus::ok;
    }
    if (impl_->detector->ready_count() == 0)
    {
        const auto finish_status = impl_->detector->finish();
        impl_->detector->reset(impl_->segment_id);
        impl_->reset_metadata();
        return finish_status == sorting::OnlineDetectionStatus::boundary_error
                   ? streaming::StreamStatus::processor_failure
                   : streaming::StreamStatus::ok;
    }

    streaming::FrameBorrow output{};
    auto status = emitter.try_acquire(output);
    if (status != streaming::StreamStatus::ok)
        return status;
    if (output.payload_storage().size() < impl_->layout->payload_bytes())
        return streaming::StreamStatus::buffer_exhausted;
    sorting::OnlineDetectionStatus finish_status{};
    try
    {
        sorting::FixedCapacitySpikeBlock block{
            output.payload_storage().first(impl_->layout->payload_bytes()),
            *impl_->layout,
            impl_->config.pre_samples,
            impl_->config.post_samples,
            impl_->segment_id,
            impl_->config.polarity};
        finish_status = impl_->detector->finish(block);
        if (finish_status == sorting::OnlineDetectionStatus::overflow)
        {
            impl_->detector->reset(impl_->segment_id);
            return streaming::StreamStatus::output_limit;
        }
        if (block.header().n_valid != 0 || block.header().overflow_count != 0)
        {
            output.header() = impl_->last_header;
            output.header().schema_id = impl_->output_schema->id();
            output.header().sequence = impl_->next_output_sequence;
            output.block_storage()[0] = impl_->last_block;
            output.block_storage()[0].sample_idx_start =
                block.header().n_valid != 0
                    ? static_cast<streaming::SampleIndex>(block.sample_indices()[0])
                    : impl_->last_block.sample_idx_start;
            output.block_storage()[0].last_sample_idx =
                block.header().n_valid != 0
                    ? static_cast<streaming::SampleIndex>(block.sample_indices().back())
                    : 0;
            output.block_storage()[0].payload_offset = 0;
            output.block_storage()[0].payload_byte_count = impl_->layout->payload_bytes();
            output.block_storage()[0].signal_id = impl_->config.output_signal_id;
            output.block_storage()[0].n_samples = block.header().n_valid;
            status = output.set_used_sizes(1, impl_->layout->payload_bytes());
            if (status == streaming::StreamStatus::ok)
                status = emitter.publish_acquired();
            if (status != streaming::StreamStatus::ok)
            {
                impl_->detector->reset(impl_->segment_id);
                return status;
            }
        }
    }
    catch (...)
    {
        impl_->detector->reset(impl_->segment_id);
        return streaming::StreamStatus::processor_failure;
    }
    impl_->detector->reset(impl_->segment_id);
    impl_->reset_metadata();
    return finish_status == sorting::OnlineDetectionStatus::boundary_error
               ? streaming::StreamStatus::processor_failure
               : streaming::StreamStatus::ok;
}

streaming::StreamStatus SpikeDetectorAdapter::reset() noexcept
{
    if (!impl_->detector)
        return streaming::StreamStatus::invalid_state;
    impl_->segment_id = 0;
    impl_->detector->reset(0);
    impl_->reset_metadata();
    return streaming::StreamStatus::ok;
}

} // namespace neurale::pipeline
