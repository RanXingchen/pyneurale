/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "resampler_adapter.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "adapter_support.h"
#include <neurale/signal/fir.h>
#include <neurale/signal/resample.h>
#include <neurale/signal/windows.h>

#include <neurale/streaming/frame_validation.h>

namespace neurale::pipeline
{
namespace
{

using adapter_support::ceil_div;

constexpr adapter_support::Checked kChecked{"resampler"};
constexpr std::size_t kFilterNeighborTerms = 10;
constexpr std::size_t kMaxFilterTaps = 4097;
constexpr double kKaiserBeta = 5.0;

[[nodiscard]] std::vector<double> default_filter(std::size_t up, std::size_t down)
{
    if (up == 0 || down == 0)
    {
        throw std::invalid_argument("up and down must be positive");
    }
    const auto common = std::gcd(up, down);
    up /= common;
    down /= common;
    if (up == 1 && down == 1)
    {
        return {1.0};
    }
    const auto maximum_rate = std::max(up, down);
    if (maximum_rate > (kMaxFilterTaps - 1) / (2 * kFilterNeighborTerms))
    {
        throw std::invalid_argument(
            "resampling ratio requires too many anti-alias filter taps; use staged conversion");
    }
    const auto length = 2 * kFilterNeighborTerms * maximum_rate + 1;
    const auto cutoff = 1.0 / static_cast<double>(maximum_rate);
    const std::array bands{0.0, cutoff, cutoff, 1.0};
    const std::array desired{1.0, 1.0, 0.0, 0.0};
    const std::array weights{1.0, 1.0};
    std::vector<double> taps(length);
    signal::firls(length - 1, bands, desired, weights, taps);
    std::vector<double> window(length);
    signal::kaiser_window(window, kKaiserBeta);
    double total{};
    for (std::size_t i = 0; i < length; ++i)
    {
        taps[i] *= window[i];
        total += taps[i];
    }
    for (auto& tap : taps)
    {
        tap /= total;
    }
    return taps;
}

[[nodiscard]] std::uint64_t checked_multiply_u64(std::uint64_t left, std::uint64_t right)
{
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
    {
        throw std::overflow_error("resampled rate overflows uint64");
    }
    return left * right;
}

[[nodiscard]] streaming::RationalRate resampled_rate(streaming::RationalRate input, std::size_t up,
                                                     std::size_t down)
{
    const auto up64 = static_cast<std::uint64_t>(up);
    const auto down64 = static_cast<std::uint64_t>(down);
    const auto cancel_num = std::gcd(input.numerator, down64);
    const auto cancel_den = std::gcd(input.denominator, up64);
    auto num = checked_multiply_u64(input.numerator / cancel_num, up64 / cancel_den);
    auto den = checked_multiply_u64(input.denominator / cancel_den, down64 / cancel_num);
    const auto common = std::gcd(num, den);
    num /= common;
    den /= common;
    return {num, den};
}

[[nodiscard]] bool scaled_ceil(std::uint64_t value, std::size_t numerator, std::size_t denominator,
                               std::uint64_t& result) noexcept
{
    const auto numerator64 = static_cast<std::uint64_t>(numerator);
    const auto denominator64 = static_cast<std::uint64_t>(denominator);
    const auto quotient = value / denominator64;
    const auto remainder = value % denominator64;
    if (quotient != 0 && numerator64 > std::numeric_limits<std::uint64_t>::max() / quotient)
    {
        return false;
    }
    const auto whole = quotient * numerator64;
    if (remainder != 0 && numerator64 > std::numeric_limits<std::uint64_t>::max() / remainder)
    {
        return false;
    }
    const auto product = remainder * numerator64;
    const auto fractional = product == 0 ? std::uint64_t{0} : (product - 1) / denominator64 + 1;
    if (fractional > std::numeric_limits<std::uint64_t>::max() - whole)
    {
        return false;
    }
    result = whole + fractional;
    return true;
}

[[nodiscard]] bool scaled_floor(std::uint64_t value, std::size_t numerator, std::size_t denominator,
                                std::uint64_t& result) noexcept
{
    const auto numerator64 = static_cast<std::uint64_t>(numerator);
    const auto denominator64 = static_cast<std::uint64_t>(denominator);
    const auto quotient = value / denominator64;
    const auto remainder = value % denominator64;
    if (quotient != 0 && numerator64 > std::numeric_limits<std::uint64_t>::max() / quotient)
    {
        return false;
    }
    if (remainder != 0 && numerator64 > std::numeric_limits<std::uint64_t>::max() / remainder)
    {
        return false;
    }
    const auto whole = quotient * numerator64;
    const auto fractional = remainder * numerator64 / denominator64;
    if (fractional > std::numeric_limits<std::uint64_t>::max() - whole)
    {
        return false;
    }
    result = whole + fractional;
    return true;
}

[[nodiscard]] std::size_t inferred_coefficients_per_phase(std::size_t filter_size, std::size_t up,
                                                          std::size_t down)
{
    if (up == 1 && down == 1)
    {
        return 0;
    }
    const auto half_length = (filter_size - 1) / 2;
    const auto pre_padding = down - half_length % down;
    return ceil_div(kChecked.checked_add(pre_padding, filter_size), up);
}

} // namespace

struct ResamplerAdapter::Impl
{
    Impl(streaming::SchemaId schema_id, std::size_t up, std::size_t down,
         std::span<const double> filter)
        : output_schema_id(schema_id), filter_size(filter.size()), resampler(up, down, filter)
    {
    }

    Impl(streaming::SchemaId schema_id, std::size_t up, std::size_t down,
         std::vector<double> filter)
        : Impl(schema_id, up, down, std::span<const double>{filter})
    {
    }

    [[nodiscard]] bool output_device_tick(std::uint64_t output_sample,
                                          streaming::DeviceTick& tick) const noexcept
    {
        if (!timing_anchored || output_sample < output_anchor_sample)
        {
            return false;
        }
        std::uint64_t source_delta{};
        if (!scaled_floor(output_sample - output_anchor_sample, resampler.down(), resampler.up(),
                          source_delta) ||
            source_delta > std::numeric_limits<std::uint64_t>::max() - source_tick_anchor)
        {
            return false;
        }
        tick = source_tick_anchor + source_delta;
        return true;
    }

    void clear_timing() noexcept
    {
        timing_anchored = false;
        has_last_metadata = false;
        output_anchor_sample = 0;
        source_tick_anchor = 0;
        next_output_sample = 0;
        next_output_sequence = 0;
        last_header = {};
        last_block = {};
    }

    [[nodiscard]] streaming::StreamStatus emit(std::span<const double> samples,
                                               std::size_t n_samples,
                                               streaming::FrameEmitter& emitter) noexcept
    {
        std::size_t offset = 0;
        while (offset < n_samples)
        {
            const auto chunk = std::min(output_block_samples, n_samples - offset);
            streaming::DeviceTick output_tick{};
            if (!output_device_tick(next_output_sample, output_tick) ||
                chunk > std::numeric_limits<std::uint64_t>::max() - next_output_sample ||
                next_output_sequence == std::numeric_limits<std::uint64_t>::max())
            {
                return streaming::StreamStatus::invalid_frame;
            }
            streaming::FrameBorrow output{};
            auto status = emitter.try_acquire(output);
            if (status != streaming::StreamStatus::ok)
            {
                return status;
            }
            output->header() = last_header;
            output->header().schema_id = output_schema_id;
            output->header().sequence = next_output_sequence;
            if (streaming::has_flag(output->header().flags, streaming::FrameFlags::source_tick))
            {
                output->header().source_tick = output_tick;
            }

            const auto n_scalars = chunk * n_channels;
            const auto payload_bytes = n_scalars * sizeof(double);
            output->block_storage()[0] = last_block;
            output->block_storage()[0].sample_idx_start = next_output_sample;
            output->block_storage()[0].device_tick_start = output_tick;
            output->block_storage()[0].payload_offset = 0;
            output->block_storage()[0].payload_byte_count = payload_bytes;
            output->block_storage()[0].n_samples = static_cast<std::uint32_t>(chunk);
            std::memcpy(output->payload_storage().data(), samples.data() + offset * n_channels,
                        payload_bytes);
            status = output->set_used_sizes(1, payload_bytes);
            if (status != streaming::StreamStatus::ok)
            {
                return status;
            }
            status = emitter.publish_acquired();
            if (status != streaming::StreamStatus::ok)
            {
                return status;
            }
            next_output_sample += chunk;
            ++next_output_sequence;
            offset += chunk;
        }
        return streaming::StreamStatus::ok;
    }

    streaming::SchemaId output_schema_id{};
    std::size_t filter_size{};
    signal::Resampler resampler;
    std::unique_ptr<streaming::FrameValidator> validator;
    std::unique_ptr<streaming::StreamSchema> input_schema;
    std::unique_ptr<streaming::StreamSchema> output_schema;
    std::vector<double> output_workspace;
    std::size_t max_input_samples{};
    std::size_t output_block_samples{};
    std::size_t max_process_output_samples{};
    std::size_t max_flush_output_samples{};
    std::size_t max_process_outputs{};
    std::size_t max_flush_outputs{};
    std::size_t n_channels{};
    streaming::FrameHeader last_header{};
    streaming::SignalBlockHeader last_block{};
    std::uint64_t output_anchor_sample{};
    streaming::DeviceTick source_tick_anchor{};
    std::uint64_t next_output_sample{};
    std::uint64_t next_output_sequence{};
    bool timing_anchored{};
    bool has_last_metadata{};
};

ResamplerAdapter::ResamplerAdapter(streaming::SchemaId output_schema_id, std::size_t up,
                                   std::size_t down, std::span<const double> filter)
    : impl_(std::make_unique<Impl>(output_schema_id, up, down, filter))
{
}

ResamplerAdapter::ResamplerAdapter(streaming::SchemaId output_schema_id, std::size_t up,
                                   std::size_t down)
    : impl_(std::make_unique<Impl>(output_schema_id, up, down, default_filter(up, down)))
{
}

ResamplerAdapter::~ResamplerAdapter() = default;

streaming::PreparedProcessorContract
ResamplerAdapter::prepare(const streaming::ProcessorPrepareContext& context)
{
    const auto signals = context.input_schema.signals();
    if (signals.size() != 1)
    {
        throw std::invalid_argument("resampler adapter requires exactly one input signal");
    }
    const auto& input = signals.front();
    if (input.kind != streaming::SignalKind::sampled ||
        input.dtype != streaming::SignalDType::float64 ||
        input.layout != streaming::SignalLayout::sample_major)
    {
        throw std::invalid_argument(
            "resampler adapter requires a sample-major float64 sampled signal");
    }
    if (impl_->output_schema_id == context.input_schema.id())
    {
        throw std::invalid_argument("resampler output requires a distinct schema id");
    }
    if (impl_->input_schema != nullptr && !impl_->input_schema->equivalent(context.input_schema))
    {
        throw std::invalid_argument("resampler adapter input schema cannot change after prepare");
    }

    const auto output_nominal = impl_->resampler.output_length(input.nominal_block_samples);
    const auto output_block_samples = output_nominal;
    const auto output_signal = streaming::SignalSchema{
        input.id,
        input.dtype,
        input.n_channels,
        kChecked.checked_block_samples(output_nominal),
        kChecked.checked_block_samples(output_block_samples),
        resampled_rate(input.fs, impl_->resampler.up(), impl_->resampler.down()),
        input.clock_domain,
        input.layout,
        streaming::DeviceTickTracking::unavailable,
        input.physical_unit,
        input.channel_set_id,
        input.calibration_id,
        input.reference_id,
        input.kind,
        input.feature_set_id,
        input.observation_timing,
        input.fixed_block_bytes,
        input.channel_names,
        input.channel_impedances_ohm,
    };
    const std::array output_signals{output_signal};
    auto declared_output_schema = streaming::StreamSchema{
        impl_->output_schema_id,
        output_signals,
        context.input_schema.feature_sets().descriptors(),
        context.input_schema.units().units(),
    };

    impl_->max_input_samples = input.max_block_samples;
    impl_->output_block_samples = output_block_samples;
    impl_->max_process_output_samples = impl_->resampler.max_output_length(input.max_block_samples);
    impl_->max_flush_output_samples = impl_->resampler.max_flush_length();
    impl_->max_process_outputs =
        ceil_div(impl_->max_process_output_samples, impl_->output_block_samples);
    impl_->max_flush_outputs =
        ceil_div(impl_->max_flush_output_samples, impl_->output_block_samples);
    if (impl_->max_process_outputs > context.max_process_outputs ||
        impl_->max_flush_outputs > context.max_flush_outputs ||
        context.available_frame_pool_leases < 1)
    {
        throw std::invalid_argument("resampler adapter output bounds exceed runtime capacity");
    }

    const auto coef_rows = inferred_coefficients_per_phase(
        impl_->filter_size, impl_->resampler.up(), impl_->resampler.down());
    const auto flush_input_samples =
        coef_rows == 0 ? std::size_t{0}
                       : kChecked.checked_add(coef_rows - 1, impl_->resampler.down());
    const auto prepared_input_samples = std::max(impl_->max_input_samples, flush_input_samples);
    impl_->resampler.prepare(input.n_channels, prepared_input_samples);
    impl_->resampler.reset();
    impl_->n_channels = input.n_channels;
    const auto output_capacity_samples =
        std::max(impl_->max_process_output_samples, impl_->max_flush_output_samples);
    impl_->output_workspace.resize(
        kChecked.checked_multiply(output_capacity_samples, impl_->n_channels));

    if (impl_->input_schema == nullptr)
    {
        impl_->validator = std::make_unique<streaming::FrameValidator>(context.input_schema);
        impl_->input_schema =
            std::make_unique<streaming::StreamSchema>(context.input_schema.clone());
        impl_->output_schema =
            std::make_unique<streaming::StreamSchema>(declared_output_schema.clone());
    }
    else if (!impl_->output_schema->equivalent(declared_output_schema))
    {
        throw std::invalid_argument("resampler adapter output schema changed after reset");
    }
    impl_->clear_timing();

    const auto history_samples = coef_rows == 0 ? std::size_t{0} : coef_rows - 1;
    const auto retained_scalars = kChecked.checked_add(
        kChecked.checked_multiply(coef_rows, impl_->resampler.up()),
        kChecked.checked_add(
            kChecked.checked_multiply(history_samples, impl_->n_channels),
            kChecked.checked_multiply(kChecked.checked_add(history_samples, prepared_input_samples),
                                      impl_->n_channels)));
    const auto workspace_scalars =
        kChecked.checked_add(retained_scalars, impl_->output_workspace.size());
    const auto workspace_bytes = kChecked.checked_multiply(workspace_scalars, sizeof(double));
    return {
        .accepted_input_schema = context.input_schema.clone(),
        .output_schema = declared_output_schema.clone(),
        .max_process_outputs_per_input = impl_->max_process_outputs,
        .max_flush_outputs = impl_->max_flush_outputs,
        .can_forward_input = false,
        .required_resources =
            streaming::ProcessorResourceBounds{
                .workspace_bytes = workspace_bytes,
                .frame_pool_leases = 1,
            },
    };
}

streaming::StreamStatus ResamplerAdapter::process(streaming::FrameBorrow& frame,
                                                  streaming::FrameEmitter& emitter) noexcept
{
    if (impl_->validator == nullptr || impl_->output_schema == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    if (impl_->validator->validate(frame.view()) != streaming::FrameValidationError::none)
    {
        return streaming::StreamStatus::invalid_frame;
    }
    const auto& block = frame.blocks().front();
    if (block.n_samples > impl_->max_input_samples)
    {
        return streaming::StreamStatus::invalid_frame;
    }
    if (!impl_->timing_anchored)
    {
        if (!scaled_ceil(block.sample_idx_start, impl_->resampler.up(), impl_->resampler.down(),
                         impl_->next_output_sample))
        {
            return streaming::StreamStatus::invalid_frame;
        }
        impl_->output_anchor_sample = impl_->next_output_sample;
        impl_->source_tick_anchor = block.device_tick_start;
        impl_->next_output_sequence = frame.header().sequence;
        impl_->timing_anchored = true;
    }
    impl_->last_header = frame.header();
    impl_->last_block = block;
    impl_->has_last_metadata = true;

    const auto* input =
        reinterpret_cast<const double*>(frame.payload().data() + block.payload_offset);
    std::size_t written{};
    try
    {
        written = impl_->resampler.process(
            {input, static_cast<std::size_t>(block.n_samples) * impl_->n_channels}, block.n_samples,
            impl_->n_channels, impl_->output_workspace);
    }
    catch (...)
    {
        return streaming::StreamStatus::processor_failure;
    }
    if (written > impl_->max_process_output_samples)
    {
        return streaming::StreamStatus::output_limit;
    }
    return impl_->emit({impl_->output_workspace.data(), written * impl_->n_channels}, written,
                       emitter);
}

streaming::StreamStatus
ResamplerAdapter::handle_discontinuity(const streaming::Discontinuity&) noexcept
{
    if (impl_->validator == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->resampler.reset();
    impl_->clear_timing();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus ResamplerAdapter::flush(streaming::FrameEmitter& emitter) noexcept
{
    if (impl_->validator == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    if (!impl_->has_last_metadata)
    {
        impl_->resampler.reset();
        impl_->clear_timing();
        return streaming::StreamStatus::ok;
    }
    std::size_t written{};
    try
    {
        written = impl_->resampler.flush(impl_->output_workspace);
    }
    catch (...)
    {
        return streaming::StreamStatus::processor_failure;
    }
    if (written > impl_->max_flush_output_samples)
    {
        return streaming::StreamStatus::output_limit;
    }
    const auto status = impl_->emit({impl_->output_workspace.data(), written * impl_->n_channels},
                                    written, emitter);
    impl_->clear_timing();
    return status;
}

streaming::StreamStatus ResamplerAdapter::reset() noexcept
{
    if (impl_->validator == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->resampler.reset();
    impl_->clear_timing();
    return streaming::StreamStatus::ok;
}

} // namespace neurale::pipeline
