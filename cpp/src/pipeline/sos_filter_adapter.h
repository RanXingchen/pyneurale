/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include <neurale/streaming/processor.h>

namespace neurale::pipeline
{

struct FilterDesign
{
    std::string type;
    std::vector<double> cutoff_hz;
    std::size_t order{4};
    std::string kind{"butterworth"};
    double passband_ripple_db{};
    double stopband_attenuation_db{};
};

struct LineNoiseFilterDesign
{
    double frequency_hz{50.0};
    double bandwidth_hz{2.0};
    std::size_t harmonics{3};
    std::size_t order{2};
};

/// Internal fixed-schema adapter for the stateful native SOS kernel.
class SosFilterAdapter final : public streaming::NativeFrameProcessor
{
  public:
    using NativeFrameProcessor::process;
    explicit SosFilterAdapter(std::span<const double> normalized_sos);
    explicit SosFilterAdapter(FilterDesign design);
    explicit SosFilterAdapter(LineNoiseFilterDesign design);
    ~SosFilterAdapter() override;

    SosFilterAdapter(const SosFilterAdapter&) = delete;
    SosFilterAdapter& operator=(const SosFilterAdapter&) = delete;
    SosFilterAdapter(SosFilterAdapter&&) = delete;
    SosFilterAdapter& operator=(SosFilterAdapter&&) = delete;

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
