/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <neurale/signal/spectral.h>
#include <neurale/streaming/processor.h>

namespace neurale::pipeline
{

struct MultitaperBand
{
    std::string name;
    double low_hz{};
    double high_hz{};
};

/// Control-plane metadata and fixed geometry for the private bandpower adapter.
struct MultitaperBandpowerAdapterConfig
{
    streaming::SchemaId output_schema_id{};
    streaming::SignalId output_signal_id{};
    streaming::FeatureSetId feature_set_id{};
    streaming::UnitId feature_unit_id{};
    std::vector<MultitaperBand> bands;
    std::vector<std::string> channel_names;
    std::string source_stream;
    std::string algorithm_version;
    std::size_t window_samples{};
    std::size_t shift_samples{};
    double time_bandwidth{};
    std::size_t n_tapers{};
    std::size_t fft_length{};
    signal::MultitaperWeighting weighting{signal::MultitaperWeighting::adaptive};
};

/// Internal fixed-window adapter for native online multitaper bandpower.
class MultitaperBandpowerAdapter final : public streaming::NativeFrameProcessor
{
  public:
    using NativeFrameProcessor::process;
    explicit MultitaperBandpowerAdapter(MultitaperBandpowerAdapterConfig config);
    ~MultitaperBandpowerAdapter() override;

    MultitaperBandpowerAdapter(const MultitaperBandpowerAdapter&) = delete;
    MultitaperBandpowerAdapter& operator=(const MultitaperBandpowerAdapter&) = delete;
    MultitaperBandpowerAdapter(MultitaperBandpowerAdapter&&) = delete;
    MultitaperBandpowerAdapter& operator=(MultitaperBandpowerAdapter&&) = delete;

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
