/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <neurale/streaming/discontinuity_pool.h>
#include <neurale/streaming/fault.h>
#include <neurale/streaming/frame.h>

namespace neurale::streaming
{

/// Native source invoked by the realtime data-plane thread.
class NativeFrameSource
{
  public:
    virtual ~NativeFrameSource() = default;

    /// Fill a caller-owned frame. would_block reports a transient stall;
    /// end_of_stream is terminal for the current session.
    virtual StreamStatus read(MutableFrame& frame) noexcept = 0;

    /// Generic read: fill either the caller-owned frame or the caller-owned
    /// discontinuity slot. Returning StreamStatus::discontinuity means the
    /// discontinuity was filled and the frame was not; every other status has
    /// the meaning it has for read(). A source that produces both kinds of
    /// data message exists because a frame and a discontinuity are two kinds
    /// of one ordered item, and a source that could only return frames would
    /// have to fold a recorded gap into the frame that follows it.
    ///
    /// The default forwards to the frame-only read, so a source written
    /// against the original interface keeps its behaviour and its cost.
    [[nodiscard]] virtual StreamStatus read_message(MutableFrame& frame,
                                                    DiscontinuityLease& discontinuity) noexcept
    {
        static_cast<void>(discontinuity);
        return read(frame);
    }

    /// Whether read_message() may ever return StreamStatus::discontinuity.
    /// A source that answers false is never handed a discontinuity lease, so
    /// the generic path costs a frame-only source no pool slot and no branch
    /// it did not already have.
    [[nodiscard]] virtual bool produces_discontinuities() const noexcept
    {
        return false;
    }

    /// Request cancellation from another thread. Implementations must make a
    /// blocked read return promptly. This operation is idempotent and must be
    /// safe to call concurrently with read().
    virtual void cancel() noexcept = 0;

    virtual StreamStatus reset() noexcept = 0;
};

} // namespace neurale::streaming
