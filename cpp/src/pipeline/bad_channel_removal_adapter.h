/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include <neurale/streaming/processor.h>

namespace neurale::pipeline
{

/// Internal fixed-schema adapter that removes known sampled channels.
class BadChannelRemovalAdapter final : public streaming::NativeFrameProcessor
{
  public:
    using NativeFrameProcessor::process;
    BadChannelRemovalAdapter(std::span<const std::size_t> bad_channel_indices,
                             std::span<const std::string> bad_channel_names,
                             std::optional<double> min_impedance_ohm = std::nullopt,
                             std::optional<double> max_impedance_ohm = std::nullopt);
    ~BadChannelRemovalAdapter() override;

    BadChannelRemovalAdapter(const BadChannelRemovalAdapter&) = delete;
    BadChannelRemovalAdapter& operator=(const BadChannelRemovalAdapter&) = delete;
    BadChannelRemovalAdapter(BadChannelRemovalAdapter&&) = delete;
    BadChannelRemovalAdapter& operator=(BadChannelRemovalAdapter&&) = delete;

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
