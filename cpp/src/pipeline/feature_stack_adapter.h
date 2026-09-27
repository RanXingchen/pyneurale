/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "multitaper_bandpower_adapter.h"

#include <neurale/features/online.h>
#include <neurale/signal/spectral.h>
#include <neurale/streaming/processor.h>

namespace neurale::pipeline
{

struct MultitaperBandpowerBranchConfig
{
    std::vector<MultitaperBand> bands;
    double time_bandwidth{2.5};
    std::size_t n_tapers{};
    signal::MultitaperWeighting weighting{signal::MultitaperWeighting::adaptive};
    features::Detrend detrend{features::Detrend::none};
    signal::SpectralBackend backend{signal::SpectralBackend::automatic};
};

struct HilbertEnvelopeBranchConfig
{
    std::vector<MultitaperBand> bands;
    std::size_t filter_order{4};
    std::string filter_kind{"butterworth"};
    double passband_ripple_db{};
    double stopband_attenuation_db{};
};

struct LmpBranchConfig
{
    double cutoff_hz{};
    std::size_t filter_order{4};
    std::string filter_kind{"butterworth"};
    double passband_ripple_db{};
    double stopband_attenuation_db{};
};

using FeatureStackBranchConfig =
    std::variant<MultitaperBandpowerBranchConfig, HilbertEnvelopeBranchConfig, LmpBranchConfig>;

struct FeatureStackAdapterConfig
{
    std::string algorithm_version;
    std::uint64_t window_ns{};
    std::uint64_t update_interval_ns{};
    std::vector<FeatureStackBranchConfig> branches;
};

/// Bounded same-input composition of the built-in windowed feature processors.
class FeatureStackAdapter final : public streaming::NativeFrameProcessor
{
  public:
    using NativeFrameProcessor::process;
    explicit FeatureStackAdapter(FeatureStackAdapterConfig config);
    ~FeatureStackAdapter() override;

    FeatureStackAdapter(const FeatureStackAdapter&) = delete;
    FeatureStackAdapter& operator=(const FeatureStackAdapter&) = delete;
    FeatureStackAdapter(FeatureStackAdapter&&) = delete;
    FeatureStackAdapter& operator=(FeatureStackAdapter&&) = delete;

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
