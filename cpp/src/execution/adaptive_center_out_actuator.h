/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/consumer.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/processor.h>

#include "center_out_controller.h"

namespace neurale::execution
{

using namespace neurale::experiments;

struct AdaptiveFeatureObservation
{
    streaming::SampleIndex sample_idx{};
    std::vector<double> values{};
};

class AdaptiveFeatureQueue;

struct PreparedDecoderCandidate
{
    streaming::NativeFrameProcessor* processor{};
    std::uint64_t version{};
};

/// Feature consumer that synchronously decodes and applies one Center-Out row.
///
/// The one-row input contract is deliberate: it makes a trial boundary also a
/// decoder boundary. A candidate prepared on a control thread is published as
/// a pointer and becomes active only after the task accepts ``trial_stop``.
class AdaptiveCenterOutActuator final : public streaming::NativeFrameConsumer
{
  public:
    AdaptiveCenterOutActuator() noexcept;
    ~AdaptiveCenterOutActuator() override;

    AdaptiveCenterOutActuator(const AdaptiveCenterOutActuator&) = delete;
    AdaptiveCenterOutActuator& operator=(const AdaptiveCenterOutActuator&) = delete;

    [[nodiscard]] streaming::StreamStatus
    prepare(const streaming::StreamSchema& feature_schema,
            streaming::NativeFrameProcessor& initial_decoder, std::uint64_t initial_version,
            const CenterOutControllerConfig& controller_config);

    [[nodiscard]] streaming::StreamStatus
    prepare_candidate(streaming::NativeFrameProcessor& decoder, std::uint64_t version);

    [[nodiscard]] streaming::StreamStatus
    publish_candidate(PreparedDecoderCandidate& candidate) noexcept;

    /// Discard an unactivated candidate after the runtime has joined.
    void discard_pending_candidate() noexcept
    {
        pending_.store(nullptr, std::memory_order_release);
    }

    [[nodiscard]] streaming::StreamStatus consume(streaming::FrameView frame) noexcept override;
    void bind_runtime_clock(streaming::NativeClock& clock) noexcept override
    {
        controller_.bind_runtime_clock(clock);
    }
    [[nodiscard]] streaming::StreamStatus
    handle_discontinuity(const streaming::Discontinuity& discontinuity) noexcept override;
    [[nodiscard]] streaming::StreamStatus flush() noexcept override;
    [[nodiscard]] streaming::StreamStatus reset() noexcept override;
    void cancel() noexcept override;

    [[nodiscard]] CenterOutController& controller() noexcept;
    [[nodiscard]] const CenterOutController& controller() const noexcept;
    [[nodiscard]] std::uint64_t active_decoder_version() const noexcept;
    [[nodiscard]] std::uint64_t activated_decoder_count() const noexcept;
    [[nodiscard]] streaming::StreamStatus try_pop_feature(AdaptiveFeatureObservation& observation);
    [[nodiscard]] std::uint64_t dropped_feature_count() const noexcept;

  private:
    class ControllerEmitter;

    [[nodiscard]] streaming::PreparedProcessorContract
    prepare_decoder(streaming::NativeFrameProcessor& decoder);
    [[nodiscard]] streaming::StreamStatus copy_input(streaming::FrameView source,
                                                     streaming::MutableFrame& destination) noexcept;
    void activate_after_trial_boundary() noexcept;

    CenterOutController controller_{};
    std::unique_ptr<streaming::FrameValidator> validator_{};
    std::unique_ptr<streaming::FramePool> input_pool_{};
    std::unique_ptr<ControllerEmitter> emitter_{};
    std::unique_ptr<AdaptiveFeatureQueue> features_{};
    std::unique_ptr<streaming::StreamSchema> feature_schema_{};
    std::unique_ptr<streaming::StreamSchema> decoded_schema_{};
    PreparedDecoderCandidate initial_{};
    PreparedDecoderCandidate* active_{};
    std::atomic<PreparedDecoderCandidate*> pending_{};
    std::atomic<std::uint64_t> active_version_{};
    std::atomic<std::uint64_t> activated_{};
    std::size_t n_features_{};
    std::size_t training_capacity_{};
    bool prepared_{};
};

} // namespace neurale::execution
