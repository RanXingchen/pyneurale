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

/// Control-plane metadata and fixed geometry for the private Hilbert-envelope adapter.
struct HilbertEnvelopeAdapterConfig
{
    streaming::SchemaId output_schema_id{};
    streaming::SignalId output_signal_id{};
    streaming::FeatureSetId feature_set_id{};
    streaming::UnitId feature_unit_id{};
    std::vector<std::string> band_names;
    std::vector<std::string> channel_names;
    std::string source_stream;
    std::string algorithm_version;
    std::size_t window_samples{};
    std::size_t shift_samples{};
    std::vector<double> sos;
    std::size_t n_bands{};
    std::size_t sos_sections{};
    std::size_t fft_length{};
};

/// Internal fixed-window adapter for the existing native Hilbert-envelope processor.
class HilbertEnvelopeAdapter final : public streaming::NativeFrameProcessor
{
  public:
    using NativeFrameProcessor::process;
    explicit HilbertEnvelopeAdapter(HilbertEnvelopeAdapterConfig config);
    ~HilbertEnvelopeAdapter() override;

    HilbertEnvelopeAdapter(const HilbertEnvelopeAdapter&) = delete;
    HilbertEnvelopeAdapter& operator=(const HilbertEnvelopeAdapter&) = delete;
    HilbertEnvelopeAdapter(HilbertEnvelopeAdapter&&) = delete;
    HilbertEnvelopeAdapter& operator=(HilbertEnvelopeAdapter&&) = delete;

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
