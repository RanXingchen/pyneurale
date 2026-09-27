/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "lmp_feature_adapter.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "adapter_support.h"
#include "feature_adapter_stream.h"

#include <neurale/features/online.h>
#include <neurale/streaming/frame_validation.h>

namespace neurale::pipeline
{
namespace
{

using adapter_support::ceil_div;
using adapter_support::equal_unit;

constexpr adapter_support::Checked kChecked{"LMP"};

} // namespace

struct LmpFeatureAdapter::Impl : adapter_support::FeatureStreamState
{
    explicit Impl(LmpFeatureAdapterConfig adapter_config) : config(std::move(adapter_config)) {}

    void clear_stream_state() noexcept
    {
        if (processor != nullptr)
        {
            processor->reset();
        }
        clear_timing();
    }

    LmpFeatureAdapterConfig config;
    std::unique_ptr<features::LmpProcessor> processor;
};

LmpFeatureAdapter::LmpFeatureAdapter(LmpFeatureAdapterConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
    const auto& value = impl_->config;
    if (value.output_schema_id == 0 || value.output_signal_id == 0 || value.feature_set_id == 0 ||
        value.feature_unit.id == 0 || value.feature_unit.symbol.empty() ||
        value.feature_names.empty() || value.source_stream.empty() ||
        value.algorithm_version.empty() || value.window_samples == 0 || value.shift_samples == 0 ||
        value.shift_samples > value.window_samples || value.sos_sections == 0 ||
        value.sos.size() != kChecked.checked_multiply(value.sos_sections, 6))
    {
        throw std::invalid_argument("invalid LMP feature adapter configuration");
    }
}

LmpFeatureAdapter::~LmpFeatureAdapter() = default;

streaming::PreparedProcessorContract
LmpFeatureAdapter::prepare(const streaming::ProcessorPrepareContext& context)
{
    const auto signals = context.input_schema.signals();
    if (signals.size() != 1)
    {
        throw std::invalid_argument("LMP adapter requires exactly one input signal");
    }
    const auto& input = signals.front();
    if (input.kind != streaming::SignalKind::sampled ||
        input.dtype != streaming::SignalDType::float64 ||
        input.layout != streaming::SignalLayout::sample_major)
    {
        throw std::invalid_argument("LMP adapter requires a sample-major float64 sampled signal");
    }
    if (impl_->config.output_schema_id == context.input_schema.id() ||
        impl_->config.output_signal_id == input.id)
    {
        throw std::invalid_argument("LMP output requires distinct schema and signal identifiers");
    }
    if (impl_->config.feature_names.size() != input.n_channels)
    {
        throw std::invalid_argument("LMP feature names must match the input channel count");
    }
    if (impl_->input_schema != nullptr && !impl_->input_schema->equivalent(context.input_schema))
    {
        throw std::invalid_argument("LMP adapter input schema cannot change after prepare");
    }

    const auto window_ns = kChecked.exact_duration_ns(impl_->config.window_samples, input.fs);
    if (window_ns % 2 != 0)
    {
        throw std::invalid_argument("LMP window center must resolve to integral nanoseconds");
    }
    impl_->window_center_ns = window_ns / 2;
    impl_->shift_ns = kChecked.exact_duration_ns(impl_->config.shift_samples, input.fs);

    std::vector<streaming::UnitDescriptor> units{context.input_schema.units().units().begin(),
                                                 context.input_schema.units().units().end()};
    const auto existing_unit = std::find_if(units.begin(), units.end(), [&](const auto& unit)
                                            { return unit.id == impl_->config.feature_unit.id; });
    if (existing_unit == units.end())
    {
        units.push_back(impl_->config.feature_unit);
    }
    else if (!equal_unit(*existing_unit, impl_->config.feature_unit))
    {
        throw std::invalid_argument("LMP feature unit id conflicts with registry");
    }

    std::vector<streaming::FeatureSetDescriptor> descriptors{
        context.input_schema.feature_sets().descriptors().begin(),
        context.input_schema.feature_sets().descriptors().end()};
    if (std::any_of(descriptors.begin(), descriptors.end(), [&](const auto& descriptor)
                    { return descriptor.id == impl_->config.feature_set_id; }))
    {
        throw std::invalid_argument("LMP feature-set id is already registered");
    }
    descriptors.push_back(streaming::FeatureSetDescriptor{
        .id = impl_->config.feature_set_id,
        .feature_names = impl_->config.feature_names,
        .unit_ids = std::vector<streaming::UnitId>(input.n_channels, impl_->config.feature_unit.id),
        .source_stream_id = input.id,
        .source_stream = impl_->config.source_stream,
        .algorithm_name = "lmp",
        .algorithm_version = impl_->config.algorithm_version,
        .window_length_ns = window_ns,
        .shift_ns = impl_->shift_ns,
        .timestamp_reference = streaming::FeatureTimestampReference::window_center,
    });

    impl_->max_input_samples = input.max_block_samples;
    const auto output = adapter_support::feature_output(
        kChecked, input, impl_->config.output_signal_id, impl_->config.feature_set_id,
        input.n_channels, impl_->config.shift_samples);
    impl_->max_output_observations = output.max_observations;
    const std::array output_signals{output.signal};
    auto declared_output_schema = streaming::StreamSchema{
        impl_->config.output_schema_id,
        output_signals,
        descriptors,
        units,
    };
    if (context.max_process_outputs < 1 || context.available_frame_pool_leases < 1)
    {
        throw std::invalid_argument("LMP adapter requires one process output and one frame lease");
    }

    if (impl_->processor == nullptr)
    {
        impl_->processor = std::make_unique<features::LmpProcessor>(
            impl_->config.window_samples, impl_->config.shift_samples, input.n_channels,
            impl_->config.sos, impl_->config.sos_sections);
        impl_->validator = std::make_unique<streaming::FrameValidator>(context.input_schema);
        impl_->input_schema =
            std::make_unique<streaming::StreamSchema>(context.input_schema.clone());
        impl_->output_schema =
            std::make_unique<streaming::StreamSchema>(declared_output_schema.clone());
        impl_->n_channels = input.n_channels;
        // LMP emits one scalar per channel, and the feature-name check above
        // already refused any configuration where those two counts differ.
        impl_->n_features = input.n_channels;
        impl_->output_workspace.resize(
            kChecked.checked_multiply(impl_->max_output_observations, impl_->n_channels));
    }
    else if (!impl_->output_schema->equivalent(declared_output_schema))
    {
        throw std::invalid_argument("LMP adapter output schema changed after reset");
    }
    impl_->clear_stream_state();

    const auto retained_scalars = kChecked.checked_add(
        kChecked.checked_multiply(impl_->config.window_samples, impl_->n_channels),
        kChecked.checked_add(
            kChecked.checked_multiply(impl_->config.sos_sections, 5),
            kChecked.checked_add(
                kChecked.checked_multiply(kChecked.checked_multiply(impl_->config.sos_sections, 2),
                                          impl_->n_channels),
                kChecked.checked_multiply(2, impl_->n_channels))));
    const auto workspace_bytes = kChecked.checked_multiply(
        kChecked.checked_add(retained_scalars, impl_->output_workspace.size()), sizeof(double));
    return {
        .accepted_input_schema = context.input_schema.clone(),
        .output_schema = declared_output_schema.clone(),
        .max_process_outputs_per_input = 1,
        .max_flush_outputs = 0,
        .can_forward_input = false,
        .required_resources =
            streaming::ProcessorResourceBounds{
                .workspace_bytes = workspace_bytes,
                .frame_pool_leases = 1,
            },
    };
}

streaming::StreamStatus LmpFeatureAdapter::process(streaming::FrameBorrow& frame,
                                                   streaming::FrameEmitter& emitter) noexcept
{
    return adapter_support::process_feature_frame(*impl_, impl_->processor.get(),
                                                  impl_->config.output_schema_id,
                                                  impl_->config.output_signal_id, frame, emitter);
}

streaming::StreamStatus
LmpFeatureAdapter::handle_discontinuity(const streaming::Discontinuity&) noexcept
{
    if (impl_->processor == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus LmpFeatureAdapter::flush(streaming::FrameEmitter&) noexcept
{
    if (impl_->processor == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus LmpFeatureAdapter::reset() noexcept
{
    if (impl_->processor == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

} // namespace neurale::pipeline
