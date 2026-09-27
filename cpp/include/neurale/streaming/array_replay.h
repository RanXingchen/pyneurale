/* SPDX-License-Identifier: MIT */
#pragma once

#include <atomic>
#include <span>
#include <vector>

#include <neurale/streaming/consumer.h>
#include <neurale/streaming/source.h>

namespace neurale::streaming
{
struct ReplayBlockTiming
{
    SampleIndex sample_idx_start{};
    HostTimeNs planned_ns{};
    HostTimeNs ready_ns{};
};

/// Owns preloaded float64 sample-major data. Pacing uses absolute block-end
/// deadlines on the host steady clock. Construction and snapshots are control
/// plane operations; reset requires a quiescent runtime.
class ArrayReplaySource final : public NativeFrameSource
{
  public:
    ArrayReplaySource(const StreamSchema& schema, std::span<const double> data, bool paced = true,
                      SessionId session_id = 1);
    StreamStatus read(MutableFrame& frame) noexcept override;
    void cancel() noexcept override;
    StreamStatus reset() noexcept override;
    [[nodiscard]] std::span<const ReplayBlockTiming> timings() const noexcept;

  private:
    SchemaId schema_id_;
    SignalSchema signal_;
    std::vector<double> data_;
    std::vector<ReplayBlockTiming> timings_;
    SteadyNativeClock clock_;
    bool paced_;
    SessionId session_id_;
    HostTimeNs period_ns_{};
    HostTimeNs epoch_ns_{};
    std::size_t next_{};
    std::atomic<std::size_t> published_{};
    std::atomic<bool> cancelled_{};
};

struct NativeResultTiming
{
    SampleIndex observation_index{};
    DeviceTick source_tick{};
    HostTimeNs source_received_ns{};
    HostTimeNs delivered_ns{};
};

/// Bounded result capture with no callbacks or steady-state allocation.
/// Full capacity faults explicitly. Snapshots may be taken during a run;
/// reset and clock binding require a quiescent runtime.
/// Nonzero window_samples/hop_samples select a null-path measurement mode:
/// validate contiguous sampled blocks, skip payload computation, and capture
/// output_channels zeros whenever a window ends (at most one per input block).
class NativeResultSink final : public NativeFrameConsumer
{
  public:
    NativeResultSink(const StreamSchema& schema, std::size_t capacity,
                     std::size_t window_samples = 0, std::size_t hop_samples = 0,
                     std::size_t output_channels = 5);
    void bind_runtime_clock(NativeClock& clock) noexcept override;
    StreamStatus consume(FrameView frame) noexcept override;
    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override;
    StreamStatus flush() noexcept override;
    StreamStatus reset() noexcept override;
    [[nodiscard]] std::span<const NativeResultTiming> timings() const noexcept;
    [[nodiscard]] std::span<const double> values(std::size_t count) const noexcept;
    [[nodiscard]] std::size_t n_channels() const noexcept
    {
        return output_channels_;
    }

  private:
    SchemaId schema_id_;
    SignalSchema signal_;
    std::size_t window_samples_{}, hop_samples_{}, output_channels_{}, samples_{};
    std::vector<double> values_;
    std::vector<NativeResultTiming> timings_;
    NativeClock* clock_{&default_native_clock()};
    std::size_t next_{};
    std::atomic<std::size_t> published_{};
};
} // namespace neurale::streaming
