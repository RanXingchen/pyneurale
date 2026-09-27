/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "kalman_decoder_adapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "adapter_support.h"

#include <neurale/streaming/frame_validation.h>

namespace neurale::pipeline
{
namespace
{

constexpr adapter_support::Checked kChecked{"Kalman decoder"};

[[nodiscard]] std::uint32_t checked_block_samples(std::uint32_t value)
{
    if (value == 0)
    {
        throw std::invalid_argument("Kalman decoder adapter block sizes must be positive");
    }
    return value;
}

/// Classification of one selected observation row.
enum class RowState : std::uint8_t
{
    present,
    absent,
    invalid,
};

} // namespace

struct KalmanDecoderAdapter::Impl
{
    explicit Impl(KalmanDecoderAdapterConfig adapter_config) : config(std::move(adapter_config)) {}

    void clear_stream_state() noexcept
    {
        if (filter != nullptr)
        {
            filter->reset();
        }
        at_segment_start = true;
    }

    /// Gather one input row through the selection and classify what it holds.
    ///
    /// The finiteness rule is read off the raw feature values rather than the
    /// scaled ones, which is the same decision the Python decoder makes: an
    /// absent row has nothing to scale, and scaling cannot turn a finite value
    /// into a missing one.
    [[nodiscard]] RowState gather(const double* row) noexcept
    {
        std::size_t missing = 0;
        for (std::size_t i = 0; i < observation.size(); ++i)
        {
            const double value = row[selection[i]];
            observation[i] = value;
            if (std::isnan(value))
            {
                ++missing;
            }
            else if (!std::isfinite(value))
            {
                // An infinity is an upstream overflow, not a measurement that
                // was never taken; only nan marks a row as missing.
                return RowState::invalid;
            }
        }
        if (missing == 0)
        {
            return RowState::present;
        }
        if (missing != observation.size() || config.missing != KalmanMissingPolicy::predict)
        {
            return RowState::invalid;
        }
        return RowState::absent;
    }

    void scale() noexcept
    {
        apply_feature_scaling(config.scaling, observation, config.scaler_center,
                              config.scaler_scale);
    }

    KalmanDecoderAdapterConfig config;
    std::unique_ptr<models::PreparedKalmanFilter> filter;
    std::unique_ptr<streaming::FrameValidator> validator;
    std::unique_ptr<streaming::StreamSchema> input_schema;
    std::unique_ptr<streaming::StreamSchema> output_schema;
    /// Selection copied out of the config so the data plane reads one flat
    /// vector instead of walking the configuration object.
    std::vector<std::size_t> selection;
    std::vector<double> observation;
    std::vector<double> output_workspace;
    std::size_t n_features{};
    std::size_t n_states{};
    std::size_t max_observations{};
    bool at_segment_start{true};
    std::unique_ptr<DecoderTrainingCapture> training_capture;
    std::size_t training_capture_capacity{};
};

KalmanDecoderAdapter::KalmanDecoderAdapter(KalmanDecoderAdapterConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
    const auto& value = impl_->config;
    if (value.output_schema_id == 0 || value.output_signal_id == 0 || value.feature_set_id == 0 ||
        value.selection.empty() || value.selection.size() != value.selected_feature_names.size() ||
        value.selection.size() != value.model.observation_dim || value.model.state_dim == 0)
    {
        throw std::invalid_argument("invalid Kalman decoder adapter configuration");
    }
    for (const auto& name : value.selected_feature_names)
    {
        if (name.empty())
        {
            throw std::invalid_argument("Kalman decoder adapter feature names must be non-empty");
        }
    }
    auto sorted = value.selection;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    {
        throw std::invalid_argument("Kalman decoder adapter selection repeats a feature column");
    }
    validate_feature_scaler(value.scaling, value.selection.size(), value.scaler_center,
                            value.scaler_scale, "Kalman decoder");
    validate_fitted_contract(value.fitted_feature_contract, "Kalman decoder");
}

KalmanDecoderAdapter::~KalmanDecoderAdapter() = default;

streaming::PreparedProcessorContract
KalmanDecoderAdapter::prepare(const streaming::ProcessorPrepareContext& context)
{
    const auto signals = context.input_schema.signals();
    if (signals.size() != 1)
    {
        throw std::invalid_argument("Kalman decoder adapter requires exactly one input signal");
    }
    const auto& input = signals.front();
    if (input.kind != streaming::SignalKind::feature ||
        input.dtype != streaming::SignalDType::float64 ||
        input.layout != streaming::SignalLayout::sample_major ||
        input.observation_timing != streaming::ObservationTiming::regular)
    {
        throw std::invalid_argument(
            "Kalman decoder adapter requires an observation-major float64 regular feature signal");
    }
    if (input.feature_set_id != impl_->config.feature_set_id)
    {
        throw std::invalid_argument("Kalman decoder adapter input declares a different feature set "
                                    "than the decoder was fitted on");
    }
    const auto* descriptor = context.input_schema.feature_sets().find(input.feature_set_id);
    if (descriptor == nullptr)
    {
        throw std::invalid_argument("Kalman decoder adapter input feature set is not registered");
    }
    if (descriptor->feature_names.size() != input.n_channels ||
        descriptor->unit_ids.size() != input.n_channels)
    {
        throw std::invalid_argument("Kalman decoder adapter input feature-set descriptor does not "
                                    "describe the input signal");
    }
    match_fitted_contract(impl_->config.fitted_feature_contract, context.input_schema, *descriptor,
                          input, kChecked, "Kalman decoder");
    // Feature identity for the selected columns is checked on top of the full
    // contract: a selection of the right length pointing at the wrong columns
    // feeds the model different neural features under the names the decoder
    // goes on reporting, and every matrix still multiplies.
    for (std::size_t i = 0; i < impl_->config.selection.size(); ++i)
    {
        const auto column = impl_->config.selection[i];
        if (column >= input.n_channels)
        {
            throw std::invalid_argument("Kalman decoder adapter selects a feature column the input "
                                        "does not have");
        }
        if (descriptor->feature_names[column] != impl_->config.selected_feature_names[i])
        {
            throw std::invalid_argument("Kalman decoder adapter selection names a different "
                                        "feature than the input provides at that column");
        }
    }
    if (impl_->config.output_schema_id == context.input_schema.id() ||
        impl_->config.output_signal_id == input.id)
    {
        throw std::invalid_argument(
            "Kalman decoder output requires distinct schema and signal identifiers");
    }
    if (impl_->input_schema != nullptr && !impl_->input_schema->equivalent(context.input_schema))
    {
        throw std::invalid_argument("Kalman decoder adapter input schema cannot change after "
                                    "prepare");
    }
    if (context.max_process_outputs < 1 || context.available_frame_pool_leases < 1)
    {
        throw std::invalid_argument(
            "Kalman decoder adapter requires one process output and one frame lease");
    }
    if (impl_->config.model.state_dim > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::overflow_error("Kalman decoder state dimension exceeds uint32");
    }

    // One decoded sample per input observation, on the input's own clock and
    // observation rate: the decoded state is what the features measured, not a
    // resampling of them.
    const auto output_signal = streaming::SignalSchema{
        impl_->config.output_signal_id,
        streaming::SignalDType::float64,
        static_cast<std::uint32_t>(impl_->config.model.state_dim),
        checked_block_samples(input.nominal_block_samples),
        checked_block_samples(input.max_block_samples),
        input.fs,
        input.clock_domain,
        streaming::SignalLayout::sample_major,
        streaming::DeviceTickTracking::unavailable,
        impl_->config.output_physical_unit,
        impl_->config.output_channel_set_id,
        input.calibration_id,
        input.reference_id,
        streaming::SignalKind::sampled,
    };
    const std::array output_signals{output_signal};
    auto declared_output_schema = streaming::StreamSchema{
        impl_->config.output_schema_id,
        output_signals,
        context.input_schema.feature_sets().descriptors(),
        context.input_schema.units().units(),
    };

    if (impl_->filter == nullptr)
    {
        impl_->filter = std::make_unique<models::PreparedKalmanFilter>(
            impl_->config.model, impl_->config.innovation_jitter);
        impl_->validator = std::make_unique<streaming::FrameValidator>(context.input_schema);
        impl_->input_schema =
            std::make_unique<streaming::StreamSchema>(context.input_schema.clone());
        impl_->output_schema =
            std::make_unique<streaming::StreamSchema>(declared_output_schema.clone());
        impl_->selection = impl_->config.selection;
        impl_->n_features = input.n_channels;
        impl_->n_states = impl_->config.model.state_dim;
        impl_->max_observations = input.max_block_samples;
        impl_->observation.assign(impl_->config.model.observation_dim, 0.0);
        impl_->output_workspace.assign(
            kChecked.checked_multiply(impl_->max_observations, impl_->n_states), 0.0);
        if (impl_->training_capture_capacity != 0)
        {
            impl_->training_capture = std::make_unique<DecoderTrainingCapture>();
            impl_->training_capture->prepare(impl_->training_capture_capacity,
                                             impl_->observation.size());
        }
    }
    else if (!impl_->output_schema->equivalent(declared_output_schema))
    {
        throw std::invalid_argument("Kalman decoder adapter output schema changed after reset");
    }
    impl_->clear_stream_state();

    const auto workspace_bytes =
        impl_->filter->workspace_bytes() +
        kChecked.checked_multiply(impl_->observation.size() + impl_->output_workspace.size(),
                                  sizeof(double)) +
        kChecked.checked_multiply(impl_->selection.size(), sizeof(std::size_t));
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

streaming::StreamStatus KalmanDecoderAdapter::process(streaming::FrameBorrow& frame,
                                                      streaming::FrameEmitter& emitter) noexcept
{
    if (impl_->filter == nullptr || impl_->validator == nullptr || impl_->output_schema == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    if (impl_->validator->validate(frame.view()) != streaming::FrameValidationError::none)
    {
        return streaming::StreamStatus::invalid_frame;
    }
    const auto& block = frame.blocks().front();
    const auto observations = static_cast<std::size_t>(block.n_samples);
    if (observations > impl_->max_observations)
    {
        return streaming::StreamStatus::output_limit;
    }
    if (impl_->training_capture != nullptr && !impl_->training_capture->can_push(observations))
    {
        impl_->training_capture->note_dropped(observations);
        return streaming::StreamStatus::queue_overflow;
    }

    const auto* input =
        reinterpret_cast<const double*>(frame.payload().data() + block.payload_offset);
    for (std::size_t row = 0; row < observations; ++row)
    {
        const auto state = impl_->gather(input + row * impl_->n_features);
        if (state == RowState::invalid)
        {
            return streaming::StreamStatus::invalid_frame;
        }
        const bool observed = state == RowState::present;
        if (impl_->training_capture != nullptr)
        {
            const auto status =
                impl_->training_capture->try_push(block.sample_idx_start + row, impl_->observation);
            if (status != streaming::StreamStatus::ok)
                return status;
        }
        if (observed)
        {
            impl_->scale();
        }
        // The first row after a fit, a reset, or a discontinuity is corrected
        // where it stands: the prepared initial state estimates the state at
        // the start of a segment rather than the step before it. Every later
        // row costs a transition and a correction.
        const auto status =
            impl_->filter->step(impl_->observation, observed, !impl_->at_segment_start);
        if (status != models::KalmanStepStatus::ok)
        {
            return streaming::StreamStatus::processor_failure;
        }
        impl_->at_segment_start = false;
        const auto decoded = impl_->filter->state();
        std::copy(decoded.begin(), decoded.end(),
                  impl_->output_workspace.begin() +
                      static_cast<std::ptrdiff_t>(row * impl_->n_states));
    }

    streaming::FrameBorrow output{};
    auto status = emitter.try_acquire(output);
    if (status != streaming::StreamStatus::ok)
    {
        return status;
    }
    const auto payload_bytes = observations * impl_->n_states * sizeof(double);
    if (payload_bytes > output->payload_storage().size() || output->block_storage().empty())
    {
        return streaming::StreamStatus::output_limit;
    }
    output->header() = frame.header();
    output->header().schema_id = impl_->config.output_schema_id;
    // The decoded block keeps the observation index and the observation time of
    // the features it decoded, so the output timeline is the input timeline and
    // nothing depends on how the stream was chunked.
    auto& decoded_block = output->block_storage()[0];
    decoded_block = block;
    decoded_block.signal_id = impl_->config.output_signal_id;
    decoded_block.device_tick_start = 0;
    decoded_block.last_sample_idx = 0;
    decoded_block.payload_offset = 0;
    decoded_block.payload_byte_count = payload_bytes;
    decoded_block.n_samples = static_cast<std::uint32_t>(observations);
    std::memcpy(output->payload_storage().data(), impl_->output_workspace.data(), payload_bytes);
    status = output->set_used_sizes(1, payload_bytes);
    if (status != streaming::StreamStatus::ok)
    {
        return status;
    }
    return emitter.publish_acquired();
}

streaming::StreamStatus
KalmanDecoderAdapter::handle_discontinuity(const streaming::Discontinuity&) noexcept
{
    if (impl_->filter == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    // A gap ends the segment the filter was tracking: the next frame starts
    // from the prepared initial state instead of continuing a trajectory across
    // data that is not there.
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus KalmanDecoderAdapter::flush(streaming::FrameEmitter&) noexcept
{
    if (impl_->filter == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    // Nothing is buffered: every observation left as part of the frame that
    // carried it, so a flush has no output to produce and only ends the segment.
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus KalmanDecoderAdapter::reset() noexcept
{
    if (impl_->filter == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

void KalmanDecoderAdapter::enable_training_capture(std::size_t capacity)
{
    if (capacity == 0 || impl_->filter != nullptr || impl_->training_capture_capacity != 0)
        throw std::invalid_argument(
            "Kalman decoder training capture must be enabled once before prepare");
    impl_->training_capture_capacity = capacity;
}

DecoderTrainingCapture* KalmanDecoderAdapter::training_capture() noexcept
{
    return impl_->training_capture.get();
}

} // namespace neurale::pipeline
