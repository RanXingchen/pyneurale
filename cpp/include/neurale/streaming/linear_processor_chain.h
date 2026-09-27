/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include <neurale/streaming/processor.h>

namespace neurale::streaming
{

/// Same-thread, depth-first composition of a fixed ordered processor list.
class LinearProcessorChain final : public NativeFrameProcessor
{
  public:
    using NativeFrameProcessor::process;

    explicit LinearProcessorChain(std::span<NativeFrameProcessor* const> stages);
    ~LinearProcessorChain() override;

    LinearProcessorChain(const LinearProcessorChain&) = delete;
    LinearProcessorChain& operator=(const LinearProcessorChain&) = delete;
    LinearProcessorChain(LinearProcessorChain&&) = delete;
    LinearProcessorChain& operator=(LinearProcessorChain&&) = delete;

    [[nodiscard]] PreparedProcessorContract
    prepare(const ProcessorPrepareContext& context) override;

    [[nodiscard]] StreamStatus process(FrameBorrow& frame, FrameEmitter& emitter) noexcept override;

    [[nodiscard]] StreamStatus
    handle_discontinuity(const Discontinuity& discontinuity) noexcept override;

    [[nodiscard]] StreamStatus flush(FrameEmitter& emitter) noexcept override;

    [[nodiscard]] StreamStatus reset() noexcept override;

    [[nodiscard]] std::size_t stage_count() const noexcept;

  private:
    class StageEmitter;
    struct Impl;

    [[nodiscard]] StreamStatus publish_owned(FrameEmitter& destination, FrameLease lease) noexcept;

    std::unique_ptr<Impl> impl_;
};

} // namespace neurale::streaming
