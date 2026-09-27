/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <neurale/streaming/processor.h>

namespace neurale::pipeline
{

/// Control-plane metadata and fixed geometry for the private LMP adapter.
struct LmpFeatureAdapterConfig
{
    streaming::SchemaId output_schema_id{};
    streaming::SignalId output_signal_id{};
    streaming::FeatureSetId feature_set_id{};
    streaming::UnitDescriptor feature_unit;
    std::vector<std::string> feature_names;
    std::string source_stream;
    std::string algorithm_version;
    std::size_t window_samples{};
    std::size_t shift_samples{};
    std::vector<double> sos;
    std::size_t sos_sections{};
};

/// Internal fixed-window adapter for the existing native LMP processor.
class LmpFeatureAdapter final : public streaming::NativeFrameProcessor
{
  public:
    using NativeFrameProcessor::process;
    explicit LmpFeatureAdapter(LmpFeatureAdapterConfig config);
    ~LmpFeatureAdapter() override;

    LmpFeatureAdapter(const LmpFeatureAdapter&) = delete;
    LmpFeatureAdapter& operator=(const LmpFeatureAdapter&) = delete;
    LmpFeatureAdapter(LmpFeatureAdapter&&) = delete;
    LmpFeatureAdapter& operator=(LmpFeatureAdapter&&) = delete;

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
