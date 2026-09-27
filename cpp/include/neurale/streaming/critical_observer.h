/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

#include <neurale/streaming/discontinuity.h>
#include <neurale/streaming/fault.h>
#include <neurale/streaming/frame.h>

namespace neurale::streaming
{

/// Identity and timestamp fixed when the runtime accepts one data message.
struct RuntimeAcceptance
{
    std::uint64_t data_message_ordinal{};
    HostTimeNs accepted_at_ns{};
};

enum class AcceptedMessageKind : std::uint8_t
{
    frame,
    discontinuity,
};

struct RejectedMessage
{
    AcceptedMessageKind kind{AcceptedMessageKind::frame};
    std::uint64_t frame_sequence{};
};

enum class RuntimeTerminalReason : std::uint8_t
{
    end_of_stream,
    stop,
    abort,
    fault,
};

/// Fixed-size terminal notice delivered after every accepted item was offered.
struct RuntimeTerminalNotice
{
    RuntimeTerminalReason reason{RuntimeTerminalReason::stop};
    HostTimeNs requested_at_ns{};
    std::uint64_t accepted_message_count{};
};

/// Format-independent lifecycle boundary for a lossless critical observer.
///
/// All callbacks are native, bounded, allocation-free, and noexcept. The
/// runtime owns frame leases; views and acceptance metadata are callback-scoped
/// borrows. `cancel()` must make an in-progress `drain()` return within the
/// implementation's documented bound and must be idempotent after termination.
class NativeCriticalObserver
{
  public:
    virtual ~NativeCriticalObserver() = default;

    [[nodiscard]] virtual StreamStatus ready_for_runtime() noexcept = 0;
    [[nodiscard]] virtual StreamStatus start_observing() noexcept = 0;
    [[nodiscard]] virtual StreamStatus health() const noexcept = 0;

    [[nodiscard]] virtual StreamStatus observe_accepted(FrameView frame,
                                                        RuntimeAcceptance acceptance) noexcept = 0;
    [[nodiscard]] virtual StreamStatus
    handle_accepted_discontinuity(const Discontinuity& discontinuity,
                                  RuntimeAcceptance acceptance) noexcept = 0;
    virtual void note_rejected_before_acceptance(RejectedMessage message) noexcept = 0;

    virtual void publish_primary_fault(const FaultRecord& fault) noexcept = 0;
    [[nodiscard]] virtual StreamStatus drain(RuntimeTerminalNotice terminal) noexcept = 0;
    virtual void cancel() noexcept = 0;
    [[nodiscard]] virtual StreamStatus reset() noexcept = 0;
};

} // namespace neurale::streaming
