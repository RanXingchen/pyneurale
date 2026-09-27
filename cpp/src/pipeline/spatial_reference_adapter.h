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

/// Reference statistic the adapter subtracts at each sample.
///
/// Mirrors `signal::detail::ReferenceStatistic` rather than reusing it: this
/// header stays free of the private signal headers, exactly as
/// `sos_filter_adapter.h` does not include `sos_realtime.h`. The two are mapped
/// in one switch in the translation unit.
enum class SpatialReferenceStatistic
{
    mean,
    median
};

/// Internal fixed-schema adapter for the stateless native common-reference kernel.
///
/// Subtracts a sample-wise common reference -- mean or median across the
/// configured reference channels -- from every channel, in place. The output schema is the input
/// schema: the referencing scheme a `SignalSchema` carries is the opaque
/// `reference_id` its source declared, and rewriting it here would forfeit
/// `can_forward_input` -- the chain requires identical schemas to forward a
/// frame without copying (linear_processor_chain.cpp).
class SpatialReferenceAdapter final : public streaming::NativeFrameProcessor
{
  public:
    using NativeFrameProcessor::process;
    /// An empty `reference_channels` references against every channel.
    explicit SpatialReferenceAdapter(
        std::span<const std::size_t> reference_channels,
        SpatialReferenceStatistic statistic = SpatialReferenceStatistic::mean);
    ~SpatialReferenceAdapter() override;

    SpatialReferenceAdapter(const SpatialReferenceAdapter&) = delete;
    SpatialReferenceAdapter& operator=(const SpatialReferenceAdapter&) = delete;
    SpatialReferenceAdapter(SpatialReferenceAdapter&&) = delete;
    SpatialReferenceAdapter& operator=(SpatialReferenceAdapter&&) = delete;

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
