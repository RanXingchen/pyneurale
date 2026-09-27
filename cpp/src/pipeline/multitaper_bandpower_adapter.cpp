/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "multitaper_bandpower_adapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "adapter_support.h"
#include "feature_adapter_stream.h"

#include <neurale/features/online.h>
#include <neurale/signal/windows.h>
#include <neurale/streaming/frame_validation.h>

namespace neurale::pipeline
{
namespace
{

using adapter_support::ceil_div;
using adapter_support::equal_unit;

constexpr adapter_support::Checked kChecked{"multitaper bandpower"};

[[nodiscard]] bool is_power_of_two(std::size_t value) noexcept
{
    return value != 0 && (value & (value - 1)) == 0;
}

[[nodiscard]] streaming::UnitDescriptor integrated_power_unit(streaming::UnitId id,
                                                              streaming::PhysicalUnit unit)
{
    switch (unit)
    {
    case streaming::PhysicalUnit::volts:
        return {id, "V^2", "volts squared"};
    case streaming::PhysicalUnit::amperes:
        return {id, "A^2", "amperes squared"};
    case streaming::PhysicalUnit::dimensionless:
        return {id, "1", "dimensionless power"};
    case streaming::PhysicalUnit::unspecified:
        break;
    }
    throw std::invalid_argument("multitaper bandpower input amplitude unit must be specified");
}

[[nodiscard]] std::vector<std::size_t> make_band_bins(std::span<const MultitaperBand> bands,
                                                      double fs, std::size_t fft_length)
{
    const auto n_bins = fft_length / 2 + 1;
    std::vector<std::size_t> result;
    result.reserve(kChecked.checked_multiply(bands.size(), 2));
    for (const auto& band : bands)
    {
        auto begin = n_bins;
        auto end = n_bins;
        for (std::size_t bin = 0; bin < n_bins; ++bin)
        {
            const auto freq = static_cast<double>(bin) * fs / static_cast<double>(fft_length);
            if (begin == n_bins && freq >= band.low_hz && freq < band.high_hz)
            {
                begin = bin;
            }
            if (begin != n_bins && freq >= band.high_hz)
            {
                end = bin;
                break;
            }
        }
        if (begin == n_bins)
        {
            throw std::invalid_argument("multitaper bandpower band contains no FFT bins");
        }
        if (end == n_bins)
        {
            end = n_bins;
        }
        result.push_back(begin);
        result.push_back(end);
    }
    return result;
}

} // namespace

struct MultitaperBandpowerAdapter::Impl : adapter_support::FeatureStreamState
{
    explicit Impl(MultitaperBandpowerAdapterConfig adapter_config)
        : config(std::move(adapter_config))
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

    MultitaperBandpowerAdapterConfig config;
    std::unique_ptr<features::BandpowerProcessor> processor;
    std::vector<double> tapers;
    std::vector<double> concentration_ratios;
    std::vector<std::size_t> band_bins;
};

MultitaperBandpowerAdapter::MultitaperBandpowerAdapter(MultitaperBandpowerAdapterConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
    const auto& value = impl_->config;
    if (value.output_schema_id == 0 || value.output_signal_id == 0 || value.feature_set_id == 0 ||
        value.feature_unit_id == 0 || value.bands.empty() || value.channel_names.empty() ||
        value.source_stream.empty() || value.algorithm_version.empty() ||
        value.window_samples == 0 || value.shift_samples == 0 ||
        value.shift_samples > value.window_samples || !std::isfinite(value.time_bandwidth) ||
        value.time_bandwidth <= 0.0 ||
        value.time_bandwidth >= static_cast<double>(value.window_samples) / 2.0 ||
        value.n_tapers < 2 || value.n_tapers > value.window_samples ||
        value.fft_length < value.window_samples || !is_power_of_two(value.fft_length))
    {
        throw std::invalid_argument("invalid multitaper bandpower feature adapter configuration");
    }
    const auto maximum_tapers_value = std::floor(2.0 * value.time_bandwidth - 1.0);
    if (maximum_tapers_value < 2.0)
    {
        throw std::invalid_argument("multitaper bandpower time bandwidth must allow two tapers");
    }
    const auto maximum_tapers =
        std::min(value.window_samples, static_cast<std::size_t>(maximum_tapers_value));
    if (value.n_tapers > maximum_tapers)
    {
        throw std::invalid_argument("multitaper bandpower taper count exceeds the PMTM limit");
    }
    std::vector<std::string> band_names;
    band_names.reserve(value.bands.size());
    for (const auto& band : value.bands)
    {
        if (band.name.empty() || !std::isfinite(band.low_hz) || !std::isfinite(band.high_hz) ||
            band.low_hz < 0.0 || band.high_hz <= band.low_hz)
        {
            throw std::invalid_argument("invalid multitaper bandpower band configuration");
        }
        band_names.push_back(band.name);
    }
    if (std::any_of(value.channel_names.begin(), value.channel_names.end(),
                    [](const auto& name) { return name.empty(); }))
    {
        throw std::invalid_argument("multitaper bandpower channel names must not be empty");
    }
    auto channel_names = value.channel_names;
    std::sort(band_names.begin(), band_names.end());
    std::sort(channel_names.begin(), channel_names.end());
    if (std::adjacent_find(band_names.begin(), band_names.end()) != band_names.end() ||
        std::adjacent_find(channel_names.begin(), channel_names.end()) != channel_names.end())
    {
        throw std::invalid_argument("multitaper bandpower names must be unique");
    }
}

MultitaperBandpowerAdapter::~MultitaperBandpowerAdapter() = default;

streaming::PreparedProcessorContract
MultitaperBandpowerAdapter::prepare(const streaming::ProcessorPrepareContext& context)
{
    const auto signals = context.input_schema.signals();
    if (signals.size() != 1)
    {
        throw std::invalid_argument(
            "multitaper bandpower adapter requires exactly one input signal");
    }
    const auto& input = signals.front();
    if (input.kind != streaming::SignalKind::sampled ||
        input.dtype != streaming::SignalDType::float64 ||
        input.layout != streaming::SignalLayout::sample_major)
    {
        throw std::invalid_argument(
            "multitaper bandpower adapter requires a sample-major float64 sampled signal");
    }
    if (impl_->config.output_schema_id == context.input_schema.id() ||
        impl_->config.output_signal_id == input.id)
    {
        throw std::invalid_argument(
            "multitaper bandpower output requires distinct schema and signal identifiers");
    }
    if (impl_->config.channel_names.size() != input.n_channels)
    {
        throw std::invalid_argument(
            "multitaper bandpower channel names must match the input channel count");
    }
    if (impl_->input_schema != nullptr && !impl_->input_schema->equivalent(context.input_schema))
    {
        throw std::invalid_argument(
            "multitaper bandpower adapter input schema cannot change after prepare");
    }

    const auto window_ns = kChecked.exact_duration_ns(impl_->config.window_samples, input.fs);
    if (window_ns % 2 != 0)
    {
        throw std::invalid_argument(
            "multitaper bandpower window center must resolve to integral nanoseconds");
    }
    impl_->window_center_ns = window_ns / 2;
    impl_->shift_ns = kChecked.exact_duration_ns(impl_->config.shift_samples, input.fs);

    const auto fs =
        static_cast<double>(input.fs.numerator) / static_cast<double>(input.fs.denominator);
    const auto nyquist = fs / 2.0;
    if (!std::isfinite(fs) || fs <= 0.0 ||
        std::any_of(impl_->config.bands.begin(), impl_->config.bands.end(),
                    [nyquist](const auto& band) { return band.high_hz > nyquist; }))
    {
        throw std::invalid_argument("multitaper bandpower bands must not exceed Nyquist");
    }
    auto prepared_band_bins = make_band_bins(impl_->config.bands, fs, impl_->config.fft_length);

    std::vector<std::string> feature_names;
    feature_names.reserve(kChecked.checked_multiply(impl_->config.bands.size(), input.n_channels));
    for (const auto& band : impl_->config.bands)
    {
        for (const auto& channel : impl_->config.channel_names)
        {
            feature_names.push_back(band.name + ":" + channel);
        }
    }

    const auto feature_unit =
        integrated_power_unit(impl_->config.feature_unit_id, input.physical_unit);

    std::vector<streaming::UnitDescriptor> units{context.input_schema.units().units().begin(),
                                                 context.input_schema.units().units().end()};
    const auto existing_unit = std::find_if(units.begin(), units.end(), [&](const auto& unit)
                                            { return unit.id == feature_unit.id; });
    if (existing_unit == units.end())
    {
        const auto symbol_conflict = std::find_if(units.begin(), units.end(), [&](const auto& unit)
                                                  { return unit.symbol == feature_unit.symbol; });
        if (symbol_conflict != units.end())
        {
            throw std::invalid_argument("multitaper bandpower unit symbol uses another id");
        }
        units.push_back(feature_unit);
    }
    else if (!equal_unit(*existing_unit, feature_unit))
    {
        throw std::invalid_argument("multitaper bandpower feature unit id conflicts with registry");
    }

    std::vector<streaming::FeatureSetDescriptor> descriptors{
        context.input_schema.feature_sets().descriptors().begin(),
        context.input_schema.feature_sets().descriptors().end()};
    if (std::any_of(descriptors.begin(), descriptors.end(), [&](const auto& descriptor)
                    { return descriptor.id == impl_->config.feature_set_id; }))
    {
        throw std::invalid_argument("multitaper bandpower feature-set id is already registered");
    }
    descriptors.push_back(streaming::FeatureSetDescriptor{
        .id = impl_->config.feature_set_id,
        .feature_names = feature_names,
        .unit_ids =
            std::vector<streaming::UnitId>(feature_names.size(), impl_->config.feature_unit_id),
        .source_stream_id = input.id,
        .source_stream = impl_->config.source_stream,
        .algorithm_name = "multitaper-bandpower",
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
            "multitaper bandpower adapter requires one process output and one frame lease");
    }

    if (impl_->processor == nullptr)
    {
        impl_->tapers.resize(
            kChecked.checked_multiply(impl_->config.window_samples, impl_->config.n_tapers));
        impl_->concentration_ratios.resize(impl_->config.n_tapers);
        signal::multitap(impl_->config.window_samples, impl_->config.time_bandwidth,
                         impl_->config.n_tapers, impl_->tapers, impl_->concentration_ratios);
        impl_->band_bins = std::move(prepared_band_bins);
        impl_->processor = std::make_unique<features::BandpowerProcessor>(
            impl_->config.window_samples, impl_->config.shift_samples, input.n_channels,
            impl_->config.fft_length, fs, impl_->tapers, impl_->config.n_tapers,
            impl_->concentration_ratios, impl_->config.weighting, impl_->band_bins,
            fs / static_cast<double>(impl_->config.fft_length), features::Detrend::none,
            signal::SpectralBackend::automatic);
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
        throw std::invalid_argument(
            "multitaper bandpower adapter output schema changed after reset");
    }
    impl_->clear_stream_state();

    const auto spectrum_bins = impl_->config.fft_length / 2 + 1;
    const auto retained_scalars = kChecked.checked_multiply(
        4,
        kChecked.checked_add(
            kChecked.checked_multiply(impl_->config.window_samples, impl_->n_channels),
            kChecked.checked_add(
                kChecked.checked_multiply(impl_->config.window_samples, impl_->config.n_tapers),
                kChecked.checked_add(kChecked.checked_multiply(
                                         spectrum_bins, kChecked.checked_add(impl_->config.n_tapers,
                                                                             impl_->n_channels)),
                                     impl_->config.fft_length))));
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

streaming::StreamStatus
MultitaperBandpowerAdapter::process(streaming::FrameBorrow& frame,
                                    streaming::FrameEmitter& emitter) noexcept
{
    return adapter_support::process_feature_frame(*impl_, impl_->processor.get(),
                                                  impl_->config.output_schema_id,
                                                  impl_->config.output_signal_id, frame, emitter);
}

streaming::StreamStatus
MultitaperBandpowerAdapter::handle_discontinuity(const streaming::Discontinuity&) noexcept
{
    if (impl_->processor == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus MultitaperBandpowerAdapter::flush(streaming::FrameEmitter&) noexcept
{
    if (impl_->processor == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus MultitaperBandpowerAdapter::reset() noexcept
{
    if (impl_->processor == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

} // namespace neurale::pipeline
