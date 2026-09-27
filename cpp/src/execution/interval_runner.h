// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#pragma once

#include "../pipeline/interval_mean_processor.h"
#include <atomic>
#include <memory>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/consumer.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/processor.h>
#include <neurale/streaming/spsc_ring.h>
#include <thread>
#include <vector>

namespace neurale::execution
{

struct IntervalResult
{
    pipeline::ObservationInterval interval;
    std::uint64_t windows{}, center_ns{};
    /// Optional decoder output. Release before destroying its IntervalRunner.
    streaming::FrameLease decoded;
};

struct IntervalFeatureValue
{
    std::uint64_t sample{}, center{}, column{};
    double value{};
};

/// Bounded, native timed processing chain: feature frames -> interval mean ->
/// optional compiled decoder -> result queue. No experiment or target labels.
/// The worker is the sole owner of processing state; schedule() and consume()
/// have separate SPSC ingress queues. publish_decoder() is control-plane only.
class IntervalRunner final : public streaming::NativeFrameConsumer
{
  public:
    IntervalRunner(const streaming::StreamSchema&, std::uint64_t duration_ns,
                   std::size_t intervals);
    ~IntervalRunner() override;
    const streaming::StreamSchema& feature_schema() const noexcept
    {
        return contract_.output_schema;
    }
    void publish_decoder(streaming::NativeFrameProcessor&);
    streaming::StreamStatus schedule(pipeline::ObservationInterval interval) noexcept;
    streaming::StreamStatus pop_result(IntervalResult& result) noexcept
    {
        return results_.try_pop(result);
    }
    streaming::StreamStatus pop_feature(IntervalFeatureValue& value) noexcept
    {
        return feature_trace_.try_pop(value);
    }
    std::span<const double> features(std::size_t index) const;
    pipeline::ObservationInterval interval(std::size_t index) const
    {
        return index < ready() ? completed_intervals_[index] : pipeline::ObservationInterval{};
    }
    std::uint64_t count(std::size_t index) const noexcept
    {
        return index < ready() ? counts_[index] : 0;
    }
    std::size_t ready() const noexcept
    {
        return ready_.load(std::memory_order_acquire);
    }
    streaming::StreamStatus status() const noexcept
    {
        return status_.load(std::memory_order_acquire);
    }
    void bind_runtime_clock(streaming::NativeClock& clock) noexcept override
    {
        clock_ = &clock;
    }
    std::uint64_t now_ns() const noexcept;
    streaming::StreamStatus start();
    void stop() noexcept;
    /// Single worker iteration, also used by deterministic clock tests.
    streaming::StreamStatus advance_time(std::uint64_t now) noexcept;
    streaming::StreamStatus consume(streaming::FrameView) noexcept override;
    streaming::StreamStatus handle_discontinuity(const streaming::Discontinuity&) noexcept override;
    streaming::StreamStatus flush() noexcept override
    {
        return status();
    }
    streaming::StreamStatus reset() noexcept override
    {
        return streaming::StreamStatus::invalid_state;
    }
    void cancel() noexcept override
    {
        stopping_.store(true, std::memory_order_release);
    }

  private:
    class MeanEmitter;
    class ResultEmitter;
    streaming::StreamSchema input_schema_;
    streaming::FrameValidator validator_;
    pipeline::IntervalMeanProcessor mean_;
    streaming::PreparedProcessorContract contract_;
    streaming::FramePool input_pool_;
    streaming::SpscRing<streaming::FrameLease> frames_{64};
    std::unique_ptr<ResultEmitter> result_emitter_;
    streaming::SpscRing<pipeline::ObservationInterval> intervals_{8};
    streaming::SpscRing<IntervalResult> results_{64};
    streaming::SpscRing<IntervalFeatureValue> feature_trace_;
    std::unique_ptr<MeanEmitter> mean_emitter_;
    std::atomic<streaming::NativeFrameProcessor*> decoder_{};
    std::vector<double> features_;
    std::vector<std::uint64_t> counts_;
    std::vector<pipeline::ObservationInterval> completed_intervals_;
    std::size_t scheduled_{};
    std::atomic<std::size_t> ready_{};
    std::atomic<streaming::StreamStatus> status_{streaming::StreamStatus::ok};
    std::atomic<bool> stopping_{};
    streaming::NativeClock* clock_{};
    std::thread worker_;
};
} // namespace neurale::execution
