/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <neurale/sorting/online_detection.h>
#include <neurale/streaming/processor.h>

namespace neurale::pipeline
{

struct SpikeDetectorAdapterConfig
{
    streaming::SchemaId output_schema_id{};
    streaming::SignalId output_signal_id{};
    std::size_t block_capacity{};
    std::size_t refractory_samples{};
    std::size_t alignment_search_radius{};
    std::size_t pre_samples{};
    std::size_t post_samples{};
    sorting::DetectionPolarity polarity{sorting::DetectionPolarity::Negative};
    sorting::BoundaryBehavior boundary_behavior{sorting::BoundaryBehavior::Drop};
    sorting::SpikeBlockOverflowPolicy overflow_policy{sorting::SpikeBlockOverflowPolicy::fault};
    std::vector<double> channel_centers;
    std::vector<double> channel_thresholds;
    std::vector<std::vector<std::size_t>> electrode_groups;
};

/// Internal sampled-signal to fixed-capacity sparse-spike adapter.
class SpikeDetectorAdapter final : public streaming::NativeFrameProcessor
{
  public:
    using NativeFrameProcessor::process;
    explicit SpikeDetectorAdapter(SpikeDetectorAdapterConfig config);
    ~SpikeDetectorAdapter() override;

    [[nodiscard]] streaming::PreparedProcessorContract
    prepare(const streaming::ProcessorPrepareContext& context) override;
    [[nodiscard]] streaming::StreamStatus
    process(streaming::FrameBorrow& frame, streaming::FrameEmitter& emitter) noexcept override;
    [[nodiscard]] streaming::StreamStatus
    handle_discontinuity(const streaming::Discontinuity& discontinuity) noexcept override;
    [[nodiscard]] streaming::StreamStatus flush(streaming::FrameEmitter& emitter) noexcept override;
    [[nodiscard]] streaming::StreamStatus reset() noexcept override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace neurale::pipeline
