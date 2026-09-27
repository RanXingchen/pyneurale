/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "hilbert_envelope_adapter.h"

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

constexpr adapter_support::Checked kChecked{"Hilbert-envelope"};

[[nodiscard]] streaming::UnitDescriptor inherited_amplitude_unit(streaming::UnitId id,
                                                                 streaming::PhysicalUnit unit)
{
    switch (unit)
    {
    case streaming::PhysicalUnit::volts:
        return {id, "V", "volts"};
    case streaming::PhysicalUnit::amperes:
        return {id, "A", "amperes"};
    case streaming::PhysicalUnit::dimensionless:
        return {id, "1", "dimensionless"};
    case streaming::PhysicalUnit::unspecified:
        break;
    }
    throw std::invalid_argument("Hilbert-envelope input amplitude unit must be specified");
}

} // namespace

struct HilbertEnvelopeAdapter::Impl : adapter_support::FeatureStreamState
{
    explicit Impl(HilbertEnvelopeAdapterConfig adapter_config) : config(std::move(adapter_config))
    {
    }

    void clear_stream_state() noexcept
    {
        if (processor != nullptr)
        {
            processor->reset();
        }
        clear_timing();
    }

    HilbertEnvelopeAdapterConfig config;
    std::unique_ptr<features::HilbertEnvelopeProcessor> processor;
};

HilbertEnvelopeAdapter::HilbertEnvelopeAdapter(HilbertEnvelopeAdapterConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
    const auto& value = impl_->config;
    if (value.output_schema_id == 0 || value.output_signal_id == 0 || value.feature_set_id == 0 ||
        value.feature_unit_id == 0 || value.band_names.empty() || value.channel_names.empty() ||
        value.source_stream.empty() || value.algorithm_version.empty() ||
        value.window_samples == 0 || value.shift_samples == 0 ||
        value.shift_samples > value.window_samples || value.n_bands == 0 ||
        value.n_bands != value.band_names.size() || value.sos_sections == 0 ||
        value.fft_length < value.window_samples ||
        value.sos.size() != kChecked.checked_multiply(
                                kChecked.checked_multiply(value.n_bands, value.sos_sections), 6))
    {
        throw std::invalid_argument("invalid Hilbert-envelope feature adapter configuration");
    }
}

HilbertEnvelopeAdapter::~HilbertEnvelopeAdapter() = default;

streaming::PreparedProcessorContract
HilbertEnvelopeAdapter::prepare(const streaming::ProcessorPrepareContext& context)
{
    const auto signals = context.input_schema.signals();
    if (signals.size() != 1)
    {
        throw std::invalid_argument("Hilbert-envelope adapter requires exactly one input signal");
    }
    const auto& input = signals.front();
    if (input.kind != streaming::SignalKind::sampled ||
        input.dtype != streaming::SignalDType::float64 ||
        input.layout != streaming::SignalLayout::sample_major)
    {
        throw std::invalid_argument(
            "Hilbert-envelope adapter requires a sample-major float64 sampled signal");
    }
    if (impl_->config.output_schema_id == context.input_schema.id() ||
        impl_->config.output_signal_id == input.id)
    {
        throw std::invalid_argument(
            "Hilbert-envelope output requires distinct schema and signal identifiers");
    }
    if (impl_->config.channel_names.size() != input.n_channels)
    {
        throw std::invalid_argument("Hilbert-envelope channel names must match the input channels");
    }
    if (impl_->input_schema != nullptr && !impl_->input_schema->equivalent(context.input_schema))
    {
        throw std::invalid_argument(
            "Hilbert-envelope adapter input schema cannot change after prepare");
    }

    const auto window_ns = kChecked.exact_duration_ns(impl_->config.window_samples, input.fs);
    if (window_ns % 2 != 0)
    {
        throw std::invalid_argument(
            "Hilbert-envelope window center must resolve to integral nanoseconds");
    }
    impl_->window_center_ns = window_ns / 2;
    impl_->shift_ns = kChecked.exact_duration_ns(impl_->config.shift_samples, input.fs);

    std::vector<streaming::UnitDescriptor> units{context.input_schema.units().units().begin(),
                                                 context.input_schema.units().units().end()};
    const auto feature_unit =
        inherited_amplitude_unit(impl_->config.feature_unit_id, input.physical_unit);
    const auto existing_unit = std::find_if(units.begin(), units.end(), [&](const auto& unit)
                                            { return unit.id == feature_unit.id; });
    if (existing_unit == units.end())
    {
        const auto symbol_conflict = std::find_if(units.begin(), units.end(), [&](const auto& unit)
                                                  { return unit.symbol == feature_unit.symbol; });
        if (symbol_conflict != units.end())
        {
            throw std::invalid_argument("Hilbert-envelope amplitude unit symbol uses another id");
        }
        units.push_back(feature_unit);
    }
    else if (!equal_unit(*existing_unit, feature_unit))
    {
        throw std::invalid_argument("Hilbert-envelope feature unit id conflicts with registry");
    }

    std::vector<std::string> feature_names;
    feature_names.reserve(kChecked.checked_multiply(impl_->config.n_bands, input.n_channels));
    for (const auto& band : impl_->config.band_names)
    {
        for (const auto& channel : impl_->config.channel_names)
        {
            feature_names.push_back(band + ":" + channel);
        }
    }

    std::vector<streaming::FeatureSetDescriptor> descriptors{
        context.input_schema.feature_sets().descriptors().begin(),
        context.input_schema.feature_sets().descriptors().end()};
    if (std::any_of(descriptors.begin(), descriptors.end(), [&](const auto& descriptor)
                    { return descriptor.id == impl_->config.feature_set_id; }))
    {
        throw std::invalid_argument("Hilbert-envelope feature-set id is already registered");
    }
    descriptors.push_back(streaming::FeatureSetDescriptor{
        .id = impl_->config.feature_set_id,
        .feature_names = feature_names,
        .unit_ids =
            std::vector<streaming::UnitId>(feature_names.size(), impl_->config.feature_unit_id),
        .source_stream_id = input.id,
        .source_stream = impl_->config.source_stream,
        .algorithm_name = "hilbert-envelope",
        .algorithm_version = impl_->config.algorithm_version,
        .window_length_ns = window_ns,
        .shift_ns = impl_->shift_ns,
        .timestamp_reference = streaming::FeatureTimestampReference::window_center,
    });

    impl_->max_input_samples = input.max_block_samples;
    const auto output = adapter_support::feature_output(
        kChecked, input, impl_->config.output_signal_id, impl_->config.feature_set_id,
        feature_names.size(), impl_->config.shift_samples);
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
        throw std::invalid_argument(
            "Hilbert-envelope adapter requires one process output and one frame lease");
    }

    if (impl_->processor == nullptr)
    {
        impl_->processor = std::make_unique<features::HilbertEnvelopeProcessor>(
            impl_->config.window_samples, impl_->config.shift_samples, input.n_channels,
            impl_->config.sos, impl_->config.n_bands, impl_->config.sos_sections,
            impl_->config.fft_length, signal::SpectralBackend::automatic);
        impl_->validator = std::make_unique<streaming::FrameValidator>(context.input_schema);
        impl_->input_schema =
            std::make_unique<streaming::StreamSchema>(context.input_schema.clone());
        impl_->output_schema =
            std::make_unique<streaming::StreamSchema>(declared_output_schema.clone());
        impl_->n_channels = input.n_channels;
        impl_->n_features = feature_names.size();
        impl_->output_workspace.resize(
            kChecked.checked_multiply(impl_->max_output_observations, impl_->n_features));
    }
    else if (!impl_->output_schema->equivalent(declared_output_schema))
    {
        throw std::invalid_argument("Hilbert-envelope adapter output schema changed after reset");
    }
    impl_->clear_stream_state();

    const auto window_feature_scalars =
        kChecked.checked_multiply(impl_->config.window_samples, impl_->n_features);
    const auto window_channel_scalars =
        kChecked.checked_multiply(impl_->config.window_samples, impl_->n_channels);
    const auto filter_state_scalars = kChecked.checked_multiply(
        kChecked.checked_multiply(
            kChecked.checked_multiply(impl_->config.n_bands, impl_->config.sos_sections), 2),
        impl_->n_channels);
    const auto analytic_scalars = kChecked.checked_multiply(
        kChecked.checked_multiply(2, impl_->config.fft_length), impl_->n_channels);
    const auto retained_scalars = kChecked.checked_add(
        kChecked.checked_multiply(2, window_feature_scalars),
        kChecked.checked_add(
            window_channel_scalars,
            kChecked.checked_add(
                filter_state_scalars,
                kChecked.checked_add(impl_->config.sos.size(),
                                     kChecked.checked_add(impl_->n_features, analytic_scalars)))));
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

streaming::StreamStatus HilbertEnvelopeAdapter::process(streaming::FrameBorrow& frame,
                                                        streaming::FrameEmitter& emitter) noexcept
{
    return adapter_support::process_feature_frame(*impl_, impl_->processor.get(),
                                                  impl_->config.output_schema_id,
                                                  impl_->config.output_signal_id, frame, emitter);
}

streaming::StreamStatus
HilbertEnvelopeAdapter::handle_discontinuity(const streaming::Discontinuity&) noexcept
{
    if (impl_->processor == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus HilbertEnvelopeAdapter::flush(streaming::FrameEmitter&) noexcept
{
    if (impl_->processor == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus HilbertEnvelopeAdapter::reset() noexcept
{
    if (impl_->processor == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

} // namespace neurale::pipeline
