/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>
#include <variant>

#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/critical_observer.h>
#include <neurale/streaming/discontinuity.h>
#include <neurale/streaming/discontinuity_pool.h>
#include <neurale/streaming/fault.h>
#include <neurale/streaming/frame.h>

namespace neurale::streaming
{

struct EndOfStreamMessage
{
};
struct ShutdownMessage
{
};
struct AbortMessage
{
    FaultRecord fault{};
};

enum class StreamMessageKind : std::uint8_t
{
    frame,
    discontinuity,
    end_of_stream,
    shutdown,
    abort,
};

/// Move-only, allocation-free message passed through native streaming edges.
class StreamMessage
{
  public:
    StreamMessage() noexcept : payload_(EndOfStreamMessage{}) {}

    StreamMessage(const StreamMessage&) = delete;
    StreamMessage& operator=(const StreamMessage&) = delete;
    StreamMessage(StreamMessage&&) noexcept = default;
    StreamMessage& operator=(StreamMessage&&) noexcept = default;
    ~StreamMessage() = default;

    [[nodiscard]] static StreamMessage from_frame(FrameLease frame,
                                                  HostTimeNs enqueued_at_ns = 0) noexcept
    {
        StreamMessage message{std::move(frame)};
        message.enqueued_at_ns_ = enqueued_at_ns;
        return message;
    }

    [[nodiscard]] static StreamMessage from_discontinuity(DiscontinuityLease discontinuity,
                                                          HostTimeNs enqueued_at_ns = 0) noexcept
    {
        StreamMessage message{std::move(discontinuity)};
        message.enqueued_at_ns_ = enqueued_at_ns;
        return message;
    }

    [[nodiscard]] static StreamMessage end_of_stream() noexcept
    {
        return StreamMessage{EndOfStreamMessage{}};
    }

    [[nodiscard]] static StreamMessage shutdown() noexcept
    {
        return StreamMessage{ShutdownMessage{}};
    }

    [[nodiscard]] static StreamMessage abort(FaultRecord fault) noexcept
    {
        return StreamMessage{AbortMessage{fault}};
    }

    [[nodiscard]] StreamMessageKind kind() const noexcept
    {
        if (std::holds_alternative<FrameLease>(payload_))
        {
            return StreamMessageKind::frame;
        }
        if (std::holds_alternative<DiscontinuityLease>(payload_))
        {
            return StreamMessageKind::discontinuity;
        }
        if (std::holds_alternative<EndOfStreamMessage>(payload_))
        {
            return StreamMessageKind::end_of_stream;
        }
        if (std::holds_alternative<ShutdownMessage>(payload_))
        {
            return StreamMessageKind::shutdown;
        }
        return StreamMessageKind::abort;
    }

    [[nodiscard]] FrameView frame() const noexcept
    {
        const auto* lease = std::get_if<FrameLease>(&payload_);
        return lease == nullptr ? FrameView{} : lease->view();
    }

    [[nodiscard]] FrameLease take_frame() noexcept
    {
        auto* lease = std::get_if<FrameLease>(&payload_);
        return lease == nullptr ? FrameLease{} : std::move(*lease);
    }

    [[nodiscard]] HostTimeNs enqueued_at_ns() const noexcept
    {
        return enqueued_at_ns_;
    }

    [[nodiscard]] Discontinuity discontinuity() const noexcept
    {
        const auto* lease = std::get_if<DiscontinuityLease>(&payload_);
        return lease == nullptr ? Discontinuity{} : lease->view();
    }

    [[nodiscard]] DiscontinuityLease take_discontinuity() noexcept
    {
        auto* lease = std::get_if<DiscontinuityLease>(&payload_);
        return lease == nullptr ? DiscontinuityLease{} : std::move(*lease);
    }

    [[nodiscard]] const FaultRecord* abort_fault() const noexcept
    {
        const auto* message = std::get_if<AbortMessage>(&payload_);
        return message == nullptr ? nullptr : &message->fault;
    }

    void set_runtime_acceptance(RuntimeAcceptance acceptance) noexcept
    {
        runtime_acceptance_ = acceptance;
        has_runtime_acceptance_ = true;
    }

    [[nodiscard]] bool has_runtime_acceptance() const noexcept
    {
        return has_runtime_acceptance_;
    }

    [[nodiscard]] RuntimeAcceptance runtime_acceptance() const noexcept
    {
        return runtime_acceptance_;
    }

  private:
    using Payload = std::variant<FrameLease, DiscontinuityLease, EndOfStreamMessage,
                                 ShutdownMessage, AbortMessage>;

    explicit StreamMessage(FrameLease frame) noexcept : payload_(std::move(frame)) {}
    explicit StreamMessage(DiscontinuityLease discontinuity) noexcept
        : payload_(std::move(discontinuity))
    {
    }
    explicit StreamMessage(EndOfStreamMessage message) noexcept : payload_(message) {}
    explicit StreamMessage(ShutdownMessage message) noexcept : payload_(message) {}
    explicit StreamMessage(AbortMessage message) noexcept : payload_(message) {}

    Payload payload_;
    HostTimeNs enqueued_at_ns_{};
    RuntimeAcceptance runtime_acceptance_{};
    bool has_runtime_acceptance_{};
};

static_assert(!std::is_copy_constructible_v<StreamMessage>);
static_assert(std::is_nothrow_move_constructible_v<StreamMessage>);
static_assert(std::is_nothrow_move_assignable_v<StreamMessage>);

} // namespace neurale::streaming
