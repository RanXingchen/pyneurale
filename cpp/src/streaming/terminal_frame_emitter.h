/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <cstddef>

#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame_emitter.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/spsc_ring.h>
#include <neurale/streaming/stats.h>
#include <neurale/streaming/stream_message.h>

namespace neurale::streaming
{

class NativeStreamRunner;

/// Terminal emitter that publishes processor outputs to the critical edge.
class TerminalFrameEmitter final : public FrameEmitter
{
  public:
    ~TerminalFrameEmitter() override = default;

    [[nodiscard]] StreamStatus try_acquire_frame(MutableFrame*& frame) noexcept override;

    [[nodiscard]] StreamStatus publish_acquired_frame() noexcept override;

    [[nodiscard]] StreamStatus publish_input() noexcept override;

  private:
    friend class NativeStreamRunner;

    TerminalFrameEmitter(FramePool& pool, FrameValidator& validator,
                         SpscRing<StreamMessage>& critical_edge, RuntimeCounters& stats,
                         NativeClock& clock, std::atomic<std::uint64_t>& critical_epoch) noexcept;

    void begin(FrameLease input, std::size_t output_limit, bool input_forwarding,
               bool flushing) noexcept;
    [[nodiscard]] FrameBorrow& input() noexcept;
    [[nodiscard]] StreamStatus finish() noexcept;
    [[nodiscard]] StreamStatus publish(FrameLease& lease) noexcept;
    [[nodiscard]] StreamStatus publish_owned(FrameLease lease) noexcept override;
    void remember_failure(StreamStatus status, FaultCode code) noexcept;

    [[nodiscard]] std::size_t published_count() const noexcept
    {
        return published_count_;
    }

    FramePool& pool_;
    FrameValidator& validator_;
    SpscRing<StreamMessage>& critical_edge_;
    RuntimeCounters& stats_;
    NativeClock& clock_;
    std::atomic<std::uint64_t>& critical_epoch_;
    FrameLease input_{};
    FrameLease acquired_{};
    detail::FrameBorrowState input_borrow_state_{};
    FrameBorrow input_borrow_{};
    std::size_t output_limit_{};
    std::size_t published_count_{};
    StreamStatus failure_{StreamStatus::ok};
    FaultCode failure_code_{FaultCode::none};
    bool active_{};
    bool input_forwarding_{};
    bool flushing_{};
};

} // namespace neurale::streaming
