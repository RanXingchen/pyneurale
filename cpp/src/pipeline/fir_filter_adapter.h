/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <memory>
#include <span>

#include <neurale/streaming/processor.h>

namespace neurale::pipeline
{

/// Internal fixed-schema adapter for the stateful native FIR processor.
class FirFilterAdapter final : public streaming::NativeFrameProcessor
{
  public:
    using NativeFrameProcessor::process;
    explicit FirFilterAdapter(std::span<const double> taps);
    ~FirFilterAdapter() override;

    FirFilterAdapter(const FirFilterAdapter&) = delete;
    FirFilterAdapter& operator=(const FirFilterAdapter&) = delete;
    FirFilterAdapter(FirFilterAdapter&&) = delete;
    FirFilterAdapter& operator=(FirFilterAdapter&&) = delete;

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
