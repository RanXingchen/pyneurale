/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include <neurale/streaming/processor.h>

namespace neurale::pipeline
{

/// Internal fixed-schema adapter for the stateful native rational resampler.
class ResamplerAdapter final : public streaming::NativeFrameProcessor
{
  public:
    using NativeFrameProcessor::process;
    ResamplerAdapter(streaming::SchemaId output_schema_id, std::size_t up, std::size_t down);
    ResamplerAdapter(streaming::SchemaId output_schema_id, std::size_t up, std::size_t down,
                     std::span<const double> filter);
    ~ResamplerAdapter() override;

    ResamplerAdapter(const ResamplerAdapter&) = delete;
    ResamplerAdapter& operator=(const ResamplerAdapter&) = delete;
    ResamplerAdapter(ResamplerAdapter&&) = delete;
    ResamplerAdapter& operator=(ResamplerAdapter&&) = delete;

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
