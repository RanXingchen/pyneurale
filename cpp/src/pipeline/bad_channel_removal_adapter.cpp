/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "bad_channel_removal_adapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <neurale/streaming/frame_validation.h>

#include "adapter_support.h"

namespace neurale::pipeline
{
namespace
{

constexpr adapter_support::Checked kChecked{"bad-channel removal"};

template <typename T> [[nodiscard]] T next_identifier(T value, const char* message)
{
    if (value == std::numeric_limits<T>::max())
    {
        throw std::invalid_argument(message);
    }
    return value + 1;
}

} // namespace

struct BadChannelRemovalAdapter::Impl
{
    Impl(std::span<const std::size_t> indices, std::span<const std::string> names,
         std::optional<double> minimum, std::optional<double> maximum)
        : bad_channel_indices(indices.begin(), indices.end()),
          bad_channel_names(names.begin(), names.end()), min_impedance_ohm(minimum),
          max_impedance_ohm(maximum)
    {
    }

    std::vector<std::size_t> bad_channel_indices;
    std::vector<std::string> bad_channel_names;
    std::optional<double> min_impedance_ohm;
    std::optional<double> max_impedance_ohm;
    std::vector<std::size_t> keep_channels;
    std::unique_ptr<streaming::FrameValidator> validator;
    std::unique_ptr<streaming::StreamSchema> input_schema;
    std::unique_ptr<streaming::StreamSchema> output_schema;
    streaming::SchemaId output_schema_id{};
    std::size_t input_channels{};
};

BadChannelRemovalAdapter::BadChannelRemovalAdapter(std::span<const std::size_t> bad_channel_indices,
                                                   std::span<const std::string> bad_channel_names,
                                                   std::optional<double> min_impedance_ohm,
                                                   std::optional<double> max_impedance_ohm)
{
    if (!bad_channel_indices.empty() && !bad_channel_names.empty())
    {
        throw std::invalid_argument("bad-channel removal cannot mix channel indices and names");
    }
    if (bad_channel_indices.empty() && bad_channel_names.empty() &&
        !min_impedance_ohm.has_value() && !max_impedance_ohm.has_value())
    {
        throw std::invalid_argument("bad-channel removal requires a channel or impedance rule");
    }
    for (const auto value : {min_impedance_ohm, max_impedance_ohm})
    {
        if (value.has_value() && (!std::isfinite(*value) || *value < 0.0))
        {
            throw std::invalid_argument(
                "bad-channel removal impedance limits must be finite and non-negative");
        }
    }
    if (min_impedance_ohm.has_value() && max_impedance_ohm.has_value() &&
        *min_impedance_ohm > *max_impedance_ohm)
    {
        throw std::invalid_argument(
            "bad-channel removal minimum impedance must not exceed maximum impedance");
    }
    for (std::size_t i = 0; i < bad_channel_indices.size(); ++i)
    {
        if (std::find(bad_channel_indices.begin(), bad_channel_indices.begin() + i,
                      bad_channel_indices[i]) != bad_channel_indices.begin() + i)
        {
            throw std::invalid_argument("bad-channel removal indices must not contain duplicates");
        }
    }
    for (std::size_t i = 0; i < bad_channel_names.size(); ++i)
    {
        if (bad_channel_names[i].empty() ||
            std::find(bad_channel_names.begin(), bad_channel_names.begin() + i,
                      bad_channel_names[i]) != bad_channel_names.begin() + i)
        {
            throw std::invalid_argument("bad-channel removal names must be nonempty and unique");
        }
    }
    impl_ = std::make_unique<Impl>(bad_channel_indices, bad_channel_names, min_impedance_ohm,
                                   max_impedance_ohm);
}

BadChannelRemovalAdapter::~BadChannelRemovalAdapter() = default;

streaming::PreparedProcessorContract
BadChannelRemovalAdapter::prepare(const streaming::ProcessorPrepareContext& context)
{
    const auto signals = context.input_schema.signals();
    if (signals.size() != 1)
    {
        throw std::invalid_argument("bad-channel removal requires exactly one input signal");
    }
    const auto& input = signals.front();
    if (input.kind != streaming::SignalKind::sampled ||
        input.dtype != streaming::SignalDType::float64 ||
        input.layout != streaming::SignalLayout::sample_major)
    {
        throw std::invalid_argument(
            "bad-channel removal requires a sample-major float64 sampled signal");
    }
    if (context.max_process_outputs < 1 || context.available_frame_pool_leases < 1)
    {
        throw std::invalid_argument(
            "bad-channel removal requires one process output and one frame lease");
    }
    if (impl_->input_schema != nullptr &&
        (!impl_->input_schema->equivalent(context.input_schema) ||
         impl_->input_schema->signals().front().channel_names != input.channel_names ||
         impl_->input_schema->signals().front().channel_impedances_ohm !=
             input.channel_impedances_ohm))
    {
        throw std::invalid_argument("bad-channel removal input schema cannot change after prepare");
    }
    std::vector<bool> remove(input.n_channels, false);
    if (!impl_->bad_channel_indices.empty())
    {
        for (const auto channel : impl_->bad_channel_indices)
        {
            if (channel >= input.n_channels)
            {
                throw std::invalid_argument(
                    "bad-channel removal index exceeds the input channel count");
            }
            remove[channel] = true;
        }
    }
    if (!impl_->bad_channel_names.empty())
    {
        if (input.channel_names.empty())
        {
            throw std::invalid_argument(
                "bad-channel removal by name requires channel names in the input schema");
        }
        for (const auto& name : impl_->bad_channel_names)
        {
            const auto found =
                std::find(input.channel_names.begin(), input.channel_names.end(), name);
            if (found == input.channel_names.end())
            {
                throw std::invalid_argument(
                    "bad-channel removal name is absent from input schema: " + name);
            }
            remove[static_cast<std::size_t>(found - input.channel_names.begin())] = true;
        }
    }
    if (impl_->min_impedance_ohm.has_value() || impl_->max_impedance_ohm.has_value())
    {
        if (input.channel_impedances_ohm.empty())
        {
            throw std::invalid_argument("bad-channel impedance filtering requires "
                                        "channel_impedances_ohm in the input schema");
        }
        for (std::size_t channel = 0; channel < input.n_channels; ++channel)
        {
            const auto impedance = input.channel_impedances_ohm[channel];
            if ((impl_->min_impedance_ohm.has_value() && impedance < *impl_->min_impedance_ohm) ||
                (impl_->max_impedance_ohm.has_value() && impedance > *impl_->max_impedance_ohm))
            {
                remove[channel] = true;
            }
        }
    }

    std::vector<std::size_t> keep_channels;
    keep_channels.reserve(input.n_channels);
    std::vector<std::string> output_channel_names;
    if (!input.channel_names.empty())
    {
        output_channel_names.reserve(input.n_channels);
    }
    std::vector<double> output_channel_impedances;
    if (!input.channel_impedances_ohm.empty())
    {
        output_channel_impedances.reserve(input.n_channels);
    }
    for (std::size_t channel = 0; channel < input.n_channels; ++channel)
    {
        if (!remove[channel])
        {
            keep_channels.push_back(channel);
            if (!input.channel_names.empty())
            {
                output_channel_names.push_back(input.channel_names[channel]);
            }
            if (!input.channel_impedances_ohm.empty())
            {
                output_channel_impedances.push_back(input.channel_impedances_ohm[channel]);
            }
        }
    }
    if (keep_channels.empty())
    {
        throw std::invalid_argument("bad-channel removal cannot remove every input channel");
    }

    const auto output_schema_id = next_identifier(
        context.input_schema.id(), "bad-channel removal cannot allocate an output schema id");
    const auto output_channel_set_id = next_identifier(
        input.channel_set_id, "bad-channel removal cannot allocate an output channel-set id");
    const auto output_signal = streaming::SignalSchema{
        input.id,
        input.dtype,
        kChecked.checked_block_samples(keep_channels.size()),
        input.nominal_block_samples,
        input.max_block_samples,
        input.fs,
        input.clock_domain,
        input.layout,
        input.device_tick_tracking,
        input.physical_unit,
        output_channel_set_id,
        input.calibration_id,
        input.reference_id,
        input.kind,
        input.feature_set_id,
        input.observation_timing,
        input.fixed_block_bytes,
        std::move(output_channel_names),
        std::move(output_channel_impedances),
    };
    const std::array output_signals{output_signal};
    auto declared_output_schema = streaming::StreamSchema{
        output_schema_id,
        output_signals,
        context.input_schema.feature_sets().descriptors(),
        context.input_schema.units().units(),
    };

    if (impl_->input_schema == nullptr)
    {
        impl_->validator = std::make_unique<streaming::FrameValidator>(context.input_schema);
        impl_->input_schema =
            std::make_unique<streaming::StreamSchema>(context.input_schema.clone());
        impl_->output_schema =
            std::make_unique<streaming::StreamSchema>(declared_output_schema.clone());
        impl_->keep_channels = std::move(keep_channels);
        impl_->output_schema_id = output_schema_id;
        impl_->input_channels = input.n_channels;
    }
    else if (!impl_->output_schema->equivalent(declared_output_schema))
    {
        throw std::invalid_argument("bad-channel removal output schema changed after prepare");
    }

    return {
        .accepted_input_schema = context.input_schema.clone(),
        .output_schema = declared_output_schema.clone(),
        .max_process_outputs_per_input = 1,
        .max_flush_outputs = 0,
        .can_forward_input = false,
        .required_resources =
            streaming::ProcessorResourceBounds{
                .workspace_bytes = impl_->keep_channels.size() * sizeof(std::size_t),
                .frame_pool_leases = 1,
            },
    };
}

streaming::StreamStatus BadChannelRemovalAdapter::process(streaming::FrameBorrow& frame,
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
    const auto& input_block = frame.blocks().front();
    const auto output_scalars =
        static_cast<std::size_t>(input_block.n_samples) * impl_->keep_channels.size();
    const auto output_bytes = output_scalars * sizeof(double);
    streaming::FrameBorrow output;
    auto status = emitter.try_acquire(output);
    if (status != streaming::StreamStatus::ok)
    {
        return status;
    }
    if (output->block_storage().empty() || output_bytes > output->payload_storage().size())
    {
        return streaming::StreamStatus::invalid_frame;
    }

    output->header() = frame.header();
    output->header().schema_id = impl_->output_schema_id;
    output->block_storage()[0] = input_block;
    output->block_storage()[0].payload_offset = 0;
    output->block_storage()[0].payload_byte_count = output_bytes;

    const auto* input =
        reinterpret_cast<const double*>(frame.payload().data() + input_block.payload_offset);
    auto* destination = reinterpret_cast<double*>(output->payload_storage().data());
    for (std::size_t sample = 0; sample < input_block.n_samples; ++sample)
    {
        for (std::size_t output_channel = 0; output_channel < impl_->keep_channels.size();
             ++output_channel)
        {
            destination[sample * impl_->keep_channels.size() + output_channel] =
                input[sample * impl_->input_channels + impl_->keep_channels[output_channel]];
        }
    }
    status = output->set_used_sizes(1, output_bytes);
    return status == streaming::StreamStatus::ok ? emitter.publish_acquired() : status;
}

streaming::StreamStatus
BadChannelRemovalAdapter::handle_discontinuity(const streaming::Discontinuity&) noexcept
{
    return impl_->validator == nullptr ? streaming::StreamStatus::invalid_state
                                       : streaming::StreamStatus::ok;
}

streaming::StreamStatus BadChannelRemovalAdapter::flush(streaming::FrameEmitter&) noexcept
{
    return impl_->validator == nullptr ? streaming::StreamStatus::invalid_state
                                       : streaming::StreamStatus::ok;
}

streaming::StreamStatus BadChannelRemovalAdapter::reset() noexcept
{
    return impl_->validator == nullptr ? streaming::StreamStatus::invalid_state
                                       : streaming::StreamStatus::ok;
}

} // namespace neurale::pipeline
