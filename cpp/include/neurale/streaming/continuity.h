/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include <neurale/streaming/discontinuity.h>
#include <neurale/streaming/discontinuity_pool.h>
#include <neurale/streaming/frame.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/schema.h>
#include <neurale/streaming/stream_message.h>

namespace neurale::streaming
{

enum class ContinuityStatus : std::uint8_t
{
    continuous,
    discontinuity,
    fatal,
};

enum class ContinuityError : std::uint8_t
{
    none,
    schema_changed,
    session_changed,
    duplicate_frame_sequence,
    frame_sequence_regressed,
    frame_sequence_overflow,
    block_count_mismatch,
    unknown_signal,
    duplicate_signal,
    invalid_sample_count,
    sample_idx_regressed,
    sample_idx_overflow,
    invalid_clock_sync,
    invalid_payload,
    insufficient_gap_capacity,
    discontinuity_pool_exhausted,
    invalid_frame_lease,
};

/// One ordered checker result: Frame, or Discontinuity followed by Frame.
struct ContinuityOutput
{
    ContinuityStatus status{ContinuityStatus::fatal};
    ContinuityError error{ContinuityError::none};
    std::uint8_t n_messages{};
    std::array<StreamMessage, 2> messages{};
};

/// Per-signal continuity state allocated and initialized before realtime startup.
class ContinuityChecker
{
  public:
    explicit ContinuityChecker(const StreamSchema& schema);
    ~ContinuityChecker();

    ContinuityChecker(const ContinuityChecker&) = delete;
    ContinuityChecker& operator=(const ContinuityChecker&) = delete;
    ContinuityChecker(ContinuityChecker&&) = delete;
    ContinuityChecker& operator=(ContinuityChecker&&) = delete;

    [[nodiscard]] ContinuityOutput check(FrameLease frame,
                                         DiscontinuityPool& discontinuity_pool) noexcept;

    /// Take an explicit discontinuity produced by the source as the account of
    /// this break, and stop expecting the frame after it to continue the one
    /// before it. Without this the checker would infer a second break for the
    /// same gap and the consumer would see the recorded discontinuity replaced
    /// by a derived one -- inference yields to an explicit record, never the
    /// other way round. The session identity is kept, so a session change
    /// across an explicit break stays the fatal error it is.
    void accept_discontinuity(const Discontinuity& discontinuity) noexcept;

    void reset() noexcept;

    [[nodiscard]] std::size_t signal_count() const noexcept
    {
        return n_signals_;
    }

  private:
    struct SignalState;
    struct GapSummary;

    void clear_states() noexcept;
    [[nodiscard]] SignalState* find_state(SignalId id) noexcept;
    [[nodiscard]] ContinuityError validate_sequence(FrameView frame) const noexcept;
    [[nodiscard]] GapSummary detect_gaps(FrameView frame) noexcept;
    void commit(FrameView frame) noexcept;

    std::size_t n_signals_{};
    FrameValidator frame_validator_;
    std::unique_ptr<SignalState[]> states_;
    std::unique_ptr<SignalGap[]> gap_scratch_;
    SessionId session_id_{};
    std::uint64_t previous_frame_sequence_{};
    bool anchored_{};
};

} // namespace neurale::streaming
