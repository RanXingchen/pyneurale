/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "adaptive_center_out_actuator.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>

namespace neurale::execution
{

using namespace neurale::experiments;
namespace
{
using streaming::StreamStatus;

[[nodiscard]] bool valid_feature_schema(const streaming::StreamSchema& schema) noexcept
{
    const auto signals = schema.signals();
    return signals.size() == 1 && signals.front().kind == streaming::SignalKind::feature &&
           signals.front().dtype == streaming::SignalDType::float64 &&
           signals.front().layout == streaming::SignalLayout::sample_major &&
           signals.front().n_channels != 0 && signals.front().max_block_samples == 1;
}

[[nodiscard]] bool valid_decoded_schema(const streaming::StreamSchema& schema,
                                        streaming::SignalId signal_id) noexcept
{
    const auto signals = schema.signals();
    return signals.size() == 1 && signals.front().id == signal_id &&
           signals.front().dtype == streaming::SignalDType::float64 &&
           signals.front().layout == streaming::SignalLayout::sample_major &&
           signals.front().n_channels == 2 && signals.front().max_block_samples == 1;
}
} // namespace

class AdaptiveFeatureQueue final
{
    static_assert(std::atomic<std::size_t>::is_always_lock_free);

  public:
    void prepare(std::size_t capacity, std::size_t n_features)
    {
        if (capacity == 0 || n_features == 0)
            throw std::invalid_argument("adaptive feature queue dimensions must be positive");
        if (capacity > (std::numeric_limits<std::size_t>::max)() / n_features)
            throw std::length_error("adaptive feature queue size overflows size_t");
        const auto value_count = capacity * n_features;
        if (value_count > (std::numeric_limits<std::size_t>::max)() / sizeof(double))
            throw std::length_error("adaptive feature queue storage size overflows size_t");
        sample_indices_ = std::make_unique_for_overwrite<streaming::SampleIndex[]>(capacity);
        values_ = std::make_unique_for_overwrite<double[]>(value_count);
        capacity_ = capacity;
        n_features_ = n_features;
        reset();
    }

    [[nodiscard]] bool can_push() const noexcept
    {
        const auto produced = produced_.load(std::memory_order_acquire);
        const auto consumed = consumed_.load(std::memory_order_acquire);
        return produced - consumed < capacity_;
    }

    [[nodiscard]] StreamStatus try_push(streaming::SampleIndex sample_idx,
                                        std::span<const double> values) noexcept
    {
        if (values.size() != n_features_)
            return StreamStatus::invalid_frame;
        const auto produced = produced_.load(std::memory_order_relaxed);
        const auto consumed = consumed_.load(std::memory_order_acquire);
        if (produced - consumed == capacity_)
        {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return StreamStatus::queue_overflow;
        }
        const auto slot = produced % capacity_;
        sample_indices_[slot] = sample_idx;
        std::copy_n(values.data(), n_features_, values_.get() + slot * n_features_);
        produced_.store(produced + 1, std::memory_order_release);
        return StreamStatus::ok;
    }

    [[nodiscard]] StreamStatus try_pop(AdaptiveFeatureObservation& observation)
    {
        const auto consumed = consumed_.load(std::memory_order_relaxed);
        const auto produced = produced_.load(std::memory_order_acquire);
        if (consumed == produced)
            return StreamStatus::would_block;
        observation.values.resize(n_features_);
        const auto slot = consumed % capacity_;
        observation.sample_idx = sample_indices_[slot];
        std::copy_n(values_.get() + slot * n_features_, n_features_, observation.values.begin());
        consumed_.store(consumed + 1, std::memory_order_release);
        return StreamStatus::ok;
    }

    void note_dropped() noexcept
    {
        dropped_.fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t dropped() const noexcept
    {
        return dropped_.load(std::memory_order_relaxed);
    }

    void reset() noexcept
    {
        consumed_.store(0, std::memory_order_relaxed);
        produced_.store(0, std::memory_order_relaxed);
    }

  private:
    std::unique_ptr<streaming::SampleIndex[]> sample_indices_{};
    std::unique_ptr<double[]> values_{};
    std::size_t capacity_{};
    std::size_t n_features_{};
    alignas(64) std::atomic<std::size_t> produced_{};
    alignas(64) std::atomic<std::size_t> consumed_{};
    alignas(64) std::atomic<std::uint64_t> dropped_{};
};

class AdaptiveCenterOutActuator::ControllerEmitter final : public streaming::FrameEmitter
{
  public:
    ControllerEmitter(CenterOutController& controller, const streaming::StreamSchema& schema)
        : controller_(controller),
          pool_(1, static_cast<std::size_t>(schema.signals().front().max_block_bytes), 1)
    {
        pool_.prefault();
    }

    void begin() noexcept
    {
        published_ = 0;
        status_ = StreamStatus::ok;
    }

    [[nodiscard]] std::size_t published() const noexcept
    {
        return published_;
    }

    [[nodiscard]] StreamStatus status() const noexcept
    {
        return status_;
    }

    [[nodiscard]] StreamStatus publish_input() noexcept override
    {
        return StreamStatus::invalid_state;
    }

  private:
    [[nodiscard]] StreamStatus try_acquire_frame(streaming::MutableFrame*& frame) noexcept override
    {
        const auto status = pool_.try_acquire(lease_);
        frame = status == StreamStatus::ok ? &lease_.frame() : nullptr;
        return status;
    }

    [[nodiscard]] StreamStatus publish_acquired_frame() noexcept override
    {
        status_ = controller_.consume(lease_.view());
        ++published_;
        const auto released = lease_.reset();
        return status_ == StreamStatus::ok ? released : status_;
    }

    [[nodiscard]] StreamStatus publish_owned(streaming::FrameLease lease) noexcept override
    {
        status_ = controller_.consume(lease.view());
        ++published_;
        return status_;
    }

    CenterOutController& controller_;
    streaming::FramePool pool_;
    streaming::FrameLease lease_{};
    std::size_t published_{};
    StreamStatus status_{StreamStatus::ok};
};

AdaptiveCenterOutActuator::AdaptiveCenterOutActuator() noexcept = default;
AdaptiveCenterOutActuator::~AdaptiveCenterOutActuator() = default;

streaming::PreparedProcessorContract
AdaptiveCenterOutActuator::prepare_decoder(streaming::NativeFrameProcessor& decoder)
{
    return decoder.prepare(streaming::ProcessorPrepareContext{
        .input_schema = *feature_schema_,
        .max_process_outputs = 1,
        .max_flush_outputs = 0,
        .available_frame_pool_leases = 1,
    });
}

StreamStatus AdaptiveCenterOutActuator::prepare(const streaming::StreamSchema& feature_schema,
                                                streaming::NativeFrameProcessor& initial_decoder,
                                                std::uint64_t initial_version,
                                                const CenterOutControllerConfig& controller_config)
{
    if (prepared_ || initial_version == 0 || !valid_feature_schema(feature_schema))
        return StreamStatus::realtime_configuration_failed;
    feature_schema_ = std::make_unique<streaming::StreamSchema>(feature_schema.clone());
    const auto contract = prepare_decoder(initial_decoder);
    if (!contract.accepted_input_schema.equivalent(*feature_schema_) ||
        contract.max_process_outputs_per_input != 1 || contract.max_flush_outputs != 0 ||
        contract.can_forward_input ||
        !valid_decoded_schema(contract.output_schema, controller_config.decoded_signal_id))
        return StreamStatus::realtime_configuration_failed;
    decoded_schema_ = std::make_unique<streaming::StreamSchema>(contract.output_schema.clone());
    auto status = controller_.prepare(*decoded_schema_, controller_config);
    if (status != StreamStatus::ok)
        return status;

    validator_ = std::make_unique<streaming::FrameValidator>(*feature_schema_);
    input_pool_ = std::make_unique<streaming::FramePool>(
        1, static_cast<std::size_t>(feature_schema_->signals().front().max_block_bytes), 1);
    input_pool_->prefault();
    emitter_ = std::make_unique<ControllerEmitter>(controller_, *decoded_schema_);
    n_features_ = feature_schema_->signals().front().n_channels;
    training_capacity_ = controller_config.training_capture_capacity;
    if (training_capacity_ != 0)
    {
        features_ = std::make_unique<AdaptiveFeatureQueue>();
        features_->prepare(training_capacity_, n_features_);
    }
    initial_ = PreparedDecoderCandidate{&initial_decoder, initial_version};
    active_ = &initial_;
    active_version_.store(initial_version, std::memory_order_release);
    controller_.set_decoder_version(initial_version);
    prepared_ = true;
    return StreamStatus::ok;
}

StreamStatus AdaptiveCenterOutActuator::prepare_candidate(streaming::NativeFrameProcessor& decoder,
                                                          std::uint64_t version)
{
    if (!prepared_ || version <= active_version_.load(std::memory_order_acquire) ||
        pending_.load(std::memory_order_acquire) != nullptr)
        return StreamStatus::invalid_state;
    const auto contract = prepare_decoder(decoder);
    return contract.accepted_input_schema.equivalent(*feature_schema_) &&
                   contract.output_schema.equivalent(*decoded_schema_) &&
                   contract.max_process_outputs_per_input == 1 && contract.max_flush_outputs == 0 &&
                   !contract.can_forward_input
               ? StreamStatus::ok
               : StreamStatus::realtime_configuration_failed;
}

StreamStatus
AdaptiveCenterOutActuator::publish_candidate(PreparedDecoderCandidate& candidate) noexcept
{
    if (!prepared_ || candidate.processor == nullptr ||
        candidate.version <= active_version_.load(std::memory_order_acquire))
        return StreamStatus::invalid_state;
    PreparedDecoderCandidate* expected = nullptr;
    return pending_.compare_exchange_strong(expected, &candidate, std::memory_order_release,
                                            std::memory_order_relaxed)
               ? StreamStatus::ok
               : StreamStatus::invalid_state;
}

StreamStatus AdaptiveCenterOutActuator::copy_input(streaming::FrameView source,
                                                   streaming::MutableFrame& destination) noexcept
{
    if (source.blocks.size() > destination.block_storage().size() ||
        source.payload.size() > destination.payload_storage().size())
        return StreamStatus::output_limit;
    destination.header() = source.header;
    std::copy(source.blocks.begin(), source.blocks.end(), destination.block_storage().begin());
    std::memcpy(destination.payload_storage().data(), source.payload.data(), source.payload.size());
    return destination.set_used_sizes(source.blocks.size(), source.payload.size());
}

StreamStatus AdaptiveCenterOutActuator::consume(streaming::FrameView frame) noexcept
{
    if (!prepared_ || active_ == nullptr ||
        validator_->validate(frame) != streaming::FrameValidationError::none)
        return StreamStatus::invalid_frame;
    const auto& block = frame.blocks.front();
    if (block.n_samples != 1)
        return StreamStatus::realtime_configuration_failed;
    if (training_capacity_ != 0 && !features_->can_push())
    {
        features_->note_dropped();
        return StreamStatus::queue_overflow;
    }

    streaming::FrameLease input;
    auto status = input_pool_->try_acquire(input);
    if (status != StreamStatus::ok)
        return status;
    status = copy_input(frame, input.frame());
    if (status != StreamStatus::ok)
        return status;

    if (training_capacity_ != 0)
    {
        const auto* values = reinterpret_cast<const double*>(
            frame.payload.data() + static_cast<std::ptrdiff_t>(block.payload_offset));
        status = features_->try_push(block.sample_idx_start,
                                     std::span<const double>(values, n_features_));
        if (status != StreamStatus::ok)
            return status;
    }

    const auto trials_before = controller_.completed_trial_count();
    controller_.set_decoder_version(active_->version);
    emitter_->begin();
    status = active_->processor->process(input.frame(), *emitter_);
    if (status != StreamStatus::ok)
        return status;
    if (emitter_->status() != StreamStatus::ok)
        return emitter_->status();
    if (emitter_->published() != 1)
        return StreamStatus::invalid_frame;
    if (controller_.completed_trial_count() != trials_before)
        activate_after_trial_boundary();
    return StreamStatus::ok;
}

void AdaptiveCenterOutActuator::activate_after_trial_boundary() noexcept
{
    auto* candidate = pending_.exchange(nullptr, std::memory_order_acq_rel);
    if (candidate == nullptr)
        return;
    active_ = candidate;
    active_version_.store(candidate->version, std::memory_order_release);
    activated_.fetch_add(1, std::memory_order_relaxed);
    controller_.set_decoder_version(candidate->version);
}

StreamStatus AdaptiveCenterOutActuator::handle_discontinuity(
    const streaming::Discontinuity& discontinuity) noexcept
{
    if (!prepared_ || active_ == nullptr)
        return StreamStatus::invalid_state;
    const auto decoder = active_->processor->handle_discontinuity(discontinuity);
    return decoder == StreamStatus::ok ? controller_.handle_discontinuity(discontinuity) : decoder;
}

StreamStatus AdaptiveCenterOutActuator::flush() noexcept
{
    if (!prepared_ || active_ == nullptr)
        return StreamStatus::invalid_state;
    emitter_->begin();
    const auto decoder = active_->processor->flush(*emitter_);
    if (decoder != StreamStatus::ok)
        return decoder;
    if (emitter_->status() != StreamStatus::ok)
        return emitter_->status();
    return controller_.flush();
}

StreamStatus AdaptiveCenterOutActuator::reset() noexcept
{
    if (!prepared_ || active_ == nullptr)
        return StreamStatus::invalid_state;
    pending_.store(nullptr, std::memory_order_release);
    active_ = &initial_;
    active_version_.store(initial_.version, std::memory_order_release);
    activated_.store(0, std::memory_order_release);
    if (training_capacity_ != 0)
        features_->reset();
    const auto decoder = active_->processor->reset();
    controller_.set_decoder_version(initial_.version);
    return decoder == StreamStatus::ok ? controller_.reset() : decoder;
}

void AdaptiveCenterOutActuator::cancel() noexcept
{
    controller_.cancel();
}

CenterOutController& AdaptiveCenterOutActuator::controller() noexcept
{
    return controller_;
}

const CenterOutController& AdaptiveCenterOutActuator::controller() const noexcept
{
    return controller_;
}

std::uint64_t AdaptiveCenterOutActuator::active_decoder_version() const noexcept
{
    return active_version_.load(std::memory_order_acquire);
}

std::uint64_t AdaptiveCenterOutActuator::activated_decoder_count() const noexcept
{
    return activated_.load(std::memory_order_relaxed);
}

StreamStatus AdaptiveCenterOutActuator::try_pop_feature(AdaptiveFeatureObservation& observation)
{
    return features_ == nullptr ? StreamStatus::would_block : features_->try_pop(observation);
}

std::uint64_t AdaptiveCenterOutActuator::dropped_feature_count() const noexcept
{
    return features_ == nullptr ? 0 : features_->dropped();
}

} // namespace neurale::execution
