/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "linear_decoder_adapter.h"

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

#include <neurale/models/classification.h>
#include <neurale/streaming/frame_validation.h>

namespace neurale::pipeline
{
namespace
{

constexpr adapter_support::Checked kChecked{"linear decoder"};

[[nodiscard]] std::uint32_t checked_block_samples(std::uint32_t value)
{
    if (value == 0)
    {
        throw std::invalid_argument("linear decoder adapter block sizes must be positive");
    }
    return value;
}

} // namespace

struct LinearDecoderAdapter::Impl
{
    explicit Impl(LinearDecoderAdapterConfig adapter_config) : config(std::move(adapter_config)) {}

    /// Gather one input row through the selection into @p destination.
    ///
    /// Any non-finite selected value is a fault. Unlike the Kalman adapter
    /// there is no "missing" classification to make: an affine predictor has no
    /// state to advance through an absent observation, so a row it cannot
    /// evaluate has no decoded value to stand in its place.
    [[nodiscard]] bool gather(const double* row, double* destination) noexcept
    {
        for (std::size_t i = 0; i < selection.size(); ++i)
        {
            const double value = row[selection[i]];
            if (!std::isfinite(value))
            {
                return false;
            }
            destination[i] = value;
        }
        return true;
    }

    LinearDecoderAdapterConfig config;
    std::unique_ptr<models::LinearModel> model;
    std::unique_ptr<models::LdaModel> classifier;
    std::unique_ptr<streaming::FrameValidator> validator;
    std::unique_ptr<streaming::StreamSchema> input_schema;
    std::unique_ptr<streaming::StreamSchema> output_schema;
    /// Selection copied out of the config so the data plane reads one flat
    /// vector instead of walking the configuration object.
    std::vector<std::size_t> selection;
    /// One whole block of gathered, scaled rows. The model is evaluated once
    /// per block rather than once per row: a single GEMM over every observation
    /// is what the BLAS kernel is shaped for, and it amortizes the call over
    /// the block instead of paying it per row.
    std::vector<double> design_workspace;
    std::vector<double> output_workspace;
    std::size_t n_model_features{};
    std::size_t n_features{};
    std::size_t n_outputs{};
    std::size_t max_observations{};
    std::unique_ptr<DecoderTrainingCapture> training_capture;
    std::size_t training_capture_capacity{};
};

LinearDecoderAdapter::LinearDecoderAdapter(LinearDecoderAdapterConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
    const auto& value = impl_->config;
    if (value.output_schema_id == 0 || value.output_signal_id == 0 || value.feature_set_id == 0 ||
        value.selection.empty() || value.selection.size() != value.selected_feature_names.size() ||
        value.selection.size() != value.model.n_features || value.model.n_outputs == 0)
    {
        throw std::invalid_argument("invalid linear decoder adapter configuration");
    }
    if (value.model.coef.size() != value.model.n_outputs * value.model.n_features ||
        value.model.intercept.size() != value.model.n_outputs)
    {
        throw std::invalid_argument("linear decoder adapter model shape is inconsistent");
    }
    for (const double coefficient : value.model.coef)
    {
        if (!std::isfinite(coefficient))
        {
            throw std::invalid_argument("linear decoder adapter coefficients must be finite");
        }
    }
    for (const double offset : value.model.intercept)
    {
        if (!std::isfinite(offset))
        {
            throw std::invalid_argument("linear decoder adapter intercept must be finite");
        }
    }
    for (const auto& name : value.selected_feature_names)
    {
        if (name.empty())
        {
            throw std::invalid_argument("linear decoder adapter feature names must be non-empty");
        }
    }
    auto sorted = value.selection;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    {
        throw std::invalid_argument("linear decoder adapter selection repeats a feature column");
    }
    validate_feature_scaler(value.scaling, value.selection.size(), value.scaler_center,
                            value.scaler_scale, "linear decoder");
    validate_fitted_contract(value.fitted_feature_contract, "linear decoder");
    if (!value.classes.empty())
    {
        if (value.classes.size() < 2 ||
            value.model.n_outputs != (value.classes.size() == 2 ? 1 : value.classes.size()))
            throw std::invalid_argument("LDA class count does not match model outputs");
        auto labels = value.classes;
        std::sort(labels.begin(), labels.end());
        if (std::adjacent_find(labels.begin(), labels.end()) != labels.end())
            throw std::invalid_argument("LDA classes must be unique");
        for (const auto label : labels)
            if (!std::isfinite(label) || std::trunc(label) != label ||
                std::abs(label) > 9007199254740991.0)
                throw std::invalid_argument("native class labels must be exact float64 integers");
    }
}

LinearDecoderAdapter::~LinearDecoderAdapter() = default;

streaming::PreparedProcessorContract
LinearDecoderAdapter::prepare(const streaming::ProcessorPrepareContext& context)
{
    const auto signals = context.input_schema.signals();
    if (signals.size() != 1)
    {
        throw std::invalid_argument("linear decoder adapter requires exactly one input signal");
    }
    const auto& input = signals.front();
    if (input.kind != streaming::SignalKind::feature ||
        input.dtype != streaming::SignalDType::float64 ||
        input.layout != streaming::SignalLayout::sample_major ||
        (input.observation_timing != streaming::ObservationTiming::regular &&
         input.observation_timing != streaming::ObservationTiming::irregular))
    {
        throw std::invalid_argument(
            "linear decoder adapter requires an observation-major float64 regular feature signal");
    }
    if (input.feature_set_id != impl_->config.feature_set_id)
    {
        throw std::invalid_argument("linear decoder adapter input declares a different feature set "
                                    "than the decoder was fitted on");
    }
    const auto* descriptor = context.input_schema.feature_sets().find(input.feature_set_id);
    if (descriptor == nullptr)
    {
        throw std::invalid_argument("linear decoder adapter input feature set is not registered");
    }
    if (descriptor->feature_names.size() != input.n_channels ||
        descriptor->unit_ids.size() != input.n_channels)
    {
        throw std::invalid_argument("linear decoder adapter input feature-set descriptor does not "
                                    "describe the input signal");
    }
    match_fitted_contract(impl_->config.fitted_feature_contract, context.input_schema, *descriptor,
                          input, kChecked, "linear decoder");
    // Feature identity for the selected columns is checked on top of the full
    // contract: a selection of the right length pointing at the wrong columns
    // feeds the model different neural features under the names the decoder
    // goes on reporting, and every matrix still multiplies.
    for (std::size_t i = 0; i < impl_->config.selection.size(); ++i)
    {
        const auto column = impl_->config.selection[i];
        if (column >= input.n_channels)
        {
            throw std::invalid_argument("linear decoder adapter selects a feature column the input "
                                        "does not have");
        }
        if (descriptor->feature_names[column] != impl_->config.selected_feature_names[i])
        {
            throw std::invalid_argument("linear decoder adapter selection names a different "
                                        "feature than the input provides at that column");
        }
    }
    if (impl_->config.output_schema_id == context.input_schema.id() ||
        impl_->config.output_signal_id == input.id)
    {
        throw std::invalid_argument(
            "linear decoder output requires distinct schema and signal identifiers");
    }
    if (impl_->input_schema != nullptr && !impl_->input_schema->equivalent(context.input_schema))
    {
        throw std::invalid_argument("linear decoder adapter input schema cannot change after "
                                    "prepare");
    }
    if (context.max_process_outputs < 1 || context.available_frame_pool_leases < 1)
    {
        throw std::invalid_argument(
            "linear decoder adapter requires one process output and one frame lease");
    }
    if (impl_->config.model.n_outputs > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::overflow_error("linear decoder output dimension exceeds uint32");
    }

    // One decoded sample per input observation, on the input's own clock and
    // observation rate: the decoded target is what the features measured, not a
    // resampling of them.
    const auto output_signal = streaming::SignalSchema{
        impl_->config.output_signal_id,
        streaming::SignalDType::float64,
        static_cast<std::uint32_t>(impl_->config.classes.empty() ? impl_->config.model.n_outputs
                                                                 : 1),
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
        input.observation_timing == streaming::ObservationTiming::irregular
            ? streaming::SignalKind::event
            : streaming::SignalKind::sampled,
    };
    const std::array output_signals{output_signal};
    auto declared_output_schema = streaming::StreamSchema{
        impl_->config.output_schema_id,
        output_signals,
        context.input_schema.feature_sets().descriptors(),
        context.input_schema.units().units(),
    };

    if (impl_->model == nullptr)
    {
        impl_->model = std::make_unique<models::LinearModel>(impl_->config.model);
        if (!impl_->config.classes.empty())
        {
            const auto& state = impl_->config.model;
            impl_->classifier = std::make_unique<models::LdaModel>(
                models::LdaModelState{state.n_features, impl_->config.classes.size(),
                                      state.n_outputs, state.coef, state.intercept});
        }
        impl_->validator = std::make_unique<streaming::FrameValidator>(context.input_schema);
        impl_->input_schema =
            std::make_unique<streaming::StreamSchema>(context.input_schema.clone());
        impl_->output_schema =
            std::make_unique<streaming::StreamSchema>(declared_output_schema.clone());
        impl_->selection = impl_->config.selection;
        impl_->n_features = input.n_channels;
        impl_->n_model_features = impl_->config.model.n_features;
        impl_->n_outputs = impl_->config.model.n_outputs;
        impl_->max_observations = input.max_block_samples;
        impl_->design_workspace.assign(
            kChecked.checked_multiply(impl_->max_observations, impl_->n_model_features), 0.0);
        impl_->output_workspace.assign(
            kChecked.checked_multiply(impl_->max_observations, impl_->n_outputs), 0.0);
        if (impl_->training_capture_capacity != 0)
        {
            impl_->training_capture = std::make_unique<DecoderTrainingCapture>();
            impl_->training_capture->prepare(impl_->training_capture_capacity,
                                             impl_->n_model_features);
        }
    }
    else if (!impl_->output_schema->equivalent(declared_output_schema))
    {
        throw std::invalid_argument("linear decoder adapter output schema changed after reset");
    }

    const auto workspace_bytes =
        kChecked.checked_multiply(impl_->design_workspace.size() + impl_->output_workspace.size(),
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

streaming::StreamStatus LinearDecoderAdapter::process(streaming::FrameBorrow& frame,
                                                      streaming::FrameEmitter& emitter) noexcept
{
    if (impl_->model == nullptr || impl_->validator == nullptr || impl_->output_schema == nullptr)
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
        double* const design_row = impl_->design_workspace.data() + row * impl_->n_model_features;
        if (!impl_->gather(input + row * impl_->n_features, design_row))
        {
            return streaming::StreamStatus::invalid_frame;
        }
        if (impl_->training_capture != nullptr)
        {
            const auto status = impl_->training_capture->try_push(
                block.sample_idx_start + row,
                std::span<const double>{design_row, impl_->n_model_features});
            if (status != streaming::StreamStatus::ok)
                return status;
        }
        apply_feature_scaling(impl_->config.scaling,
                              std::span<double>{design_row, impl_->n_model_features},
                              impl_->config.scaler_center, impl_->config.scaler_scale);
    }
    // One product for the whole block. predict() throws only for a shape it
    // cannot evaluate, and the spans here are sized from the prepared geometry,
    // so the catch is a guard against a programming error rather than a path
    // the data plane is expected to take -- constructing the exception would
    // itself allocate, which is why it must not be reachable in steady state.
    try
    {
        const auto design = std::span<const double>{impl_->design_workspace.data(),
                                                    observations * impl_->n_model_features};
        const auto scores =
            std::span<double>{impl_->output_workspace.data(), observations * impl_->n_outputs};
        if (impl_->classifier)
            impl_->classifier->decision_function(design, observations, scores);
        else
            impl_->model->predict(design, observations, scores);
    }
    catch (const std::exception&)
    {
        return streaming::StreamStatus::processor_failure;
    }

    streaming::FrameBorrow output{};
    auto status = emitter.try_acquire(output);
    if (status != streaming::StreamStatus::ok)
    {
        return status;
    }
    if (impl_->classifier)
    {
        for (std::size_t row = 0; row < observations; ++row)
        {
            const auto* scores = impl_->output_workspace.data() + row * impl_->n_outputs;
            if (!std::all_of(scores, scores + impl_->n_outputs,
                             [](double v) { return std::isfinite(v); }))
                return streaming::StreamStatus::processor_failure;
            const auto index = impl_->n_outputs == 1
                                   ? (scores[0] > 0 ? 1 : 0)
                                   : std::max_element(scores, scores + impl_->n_outputs) - scores;
            impl_->output_workspace[row] = impl_->config.classes[index];
        }
    }
    const auto payload_bytes =
        observations * (impl_->classifier ? 1 : impl_->n_outputs) * sizeof(double);
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

// Stateless: there is no trajectory a gap could invalidate, nothing buffered
// for a flush to emit, and nothing for a reset to restore. The three still
// report invalid_state before prepare, so an unprepared adapter answers like
// every other one.
streaming::StreamStatus
LinearDecoderAdapter::handle_discontinuity(const streaming::Discontinuity&) noexcept
{
    return impl_->model == nullptr ? streaming::StreamStatus::invalid_state
                                   : streaming::StreamStatus::ok;
}

streaming::StreamStatus LinearDecoderAdapter::flush(streaming::FrameEmitter&) noexcept
{
    return impl_->model == nullptr ? streaming::StreamStatus::invalid_state
                                   : streaming::StreamStatus::ok;
}

streaming::StreamStatus LinearDecoderAdapter::reset() noexcept
{
    return impl_->model == nullptr ? streaming::StreamStatus::invalid_state
                                   : streaming::StreamStatus::ok;
}

void LinearDecoderAdapter::enable_training_capture(std::size_t capacity)
{
    if (capacity == 0 || impl_->model != nullptr || impl_->training_capture_capacity != 0)
        throw std::invalid_argument(
            "linear decoder training capture must be enabled once before prepare");
    impl_->training_capture_capacity = capacity;
}

DecoderTrainingCapture* LinearDecoderAdapter::training_capture() noexcept
{
    return impl_->training_capture.get();
}

} // namespace neurale::pipeline
