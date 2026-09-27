// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/processor.h>

namespace neurale::pipeline
{

struct ObservationInterval
{
    std::uint64_t key{}, start_ns{}, end_ns{};
};

/// Reduce complete feature windows inside an externally supplied interval.
/// The caller advances time explicitly; closing an interval needs no new frame.
class IntervalMeanProcessor final : public streaming::NativeFrameProcessor
{
  public:
    using NativeFrameProcessor::process;
    explicit IntervalMeanProcessor(std::uint64_t duration_ns) : duration_ns_(duration_ns) {}
    streaming::PreparedProcessorContract
    prepare(const streaming::ProcessorPrepareContext&) override;
    streaming::StreamStatus begin(ObservationInterval interval) noexcept;
    streaming::StreamStatus advance_time(std::uint64_t now, streaming::FrameEmitter&) noexcept;
    streaming::StreamStatus process(streaming::FrameBorrow&,
                                    streaming::FrameEmitter&) noexcept override;
    streaming::StreamStatus handle_discontinuity(const streaming::Discontinuity&) noexcept override;
    streaming::StreamStatus flush(streaming::FrameEmitter&) noexcept override;
    streaming::StreamStatus reset() noexcept override;
    bool active() const noexcept
    {
        return active_;
    }
    std::uint64_t count() const noexcept
    {
        return count_;
    }
    ObservationInterval interval() const noexcept
    {
        return interval_;
    }

  private:
    std::uint64_t duration_ns_, window_ns_{}, count_{}, last_end_{}, last_key_{}, previous_end_{};
    bool active_{};
    ObservationInterval interval_;
    std::vector<double> mean_;
    streaming::FrameHeader last_header_{};
    std::unique_ptr<streaming::StreamSchema> output_schema_;
    std::unique_ptr<streaming::FrameValidator> validator_;
};

} // namespace neurale::pipeline
