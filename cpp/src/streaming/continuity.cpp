/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/continuity.h>

#include <algorithm>
#include <limits>

namespace neurale::streaming
{
namespace
{

[[nodiscard]] ContinuityOutput fatal(ContinuityError error) noexcept
{
    return ContinuityOutput{
        .status = ContinuityStatus::fatal,
        .error = error,
    };
}

[[nodiscard]] constexpr int reason_priority(GapReason reason) noexcept
{
    switch (reason)
    {
    case GapReason::device_restart:
        return 4;
    case GapReason::sample_gap:
        return 3;
    case GapReason::device_tick_gap:
        return 2;
    case GapReason::frame_sequence_gap:
        return 1;
    case GapReason::source_gap:
    case GapReason::buffer_exhausted:
    case GapReason::queue_overflow:
        return 0;
    }
    return 0;
}

[[nodiscard]] constexpr GapReason stronger_reason(GapReason current, GapReason candidate) noexcept
{
    return reason_priority(candidate) > reason_priority(current) ? candidate : current;
}

[[nodiscard]] constexpr ContinuityError map_validation_error(FrameValidationError error) noexcept
{
    switch (error)
    {
    case FrameValidationError::none:
        return ContinuityError::none;
    case FrameValidationError::schema_changed:
        return ContinuityError::schema_changed;
    case FrameValidationError::block_count_mismatch:
        return ContinuityError::block_count_mismatch;
    case FrameValidationError::unknown_signal:
        return ContinuityError::unknown_signal;
    case FrameValidationError::duplicate_signal:
        return ContinuityError::duplicate_signal;
    case FrameValidationError::invalid_sample_count:
        return ContinuityError::invalid_sample_count;
    case FrameValidationError::sample_idx_overflow:
        return ContinuityError::sample_idx_overflow;
    case FrameValidationError::invalid_sparse_idx_range:
        return ContinuityError::invalid_payload;
    case FrameValidationError::invalid_clock_sync:
        return ContinuityError::invalid_clock_sync;
    case FrameValidationError::payload_size_mismatch:
    case FrameValidationError::payload_layout_invalid:
        return ContinuityError::invalid_payload;
    }
    return ContinuityError::invalid_payload;
}

} // namespace

struct ContinuityChecker::SignalState
{
    SignalId id{};
    DeviceTickTracking tick_tracking{DeviceTickTracking::unavailable};
    SignalKind kind{SignalKind::sampled};
    std::uint64_t expected_frame_sequence{};
    SampleIndex expected_sample_idx{};
    /// Last-event index of the most recent non-empty spike block. Spike blocks
    /// are sparse, so continuity compares the next first event against this
    /// inclusive boundary instead of deriving a dense sample span.
    SampleIndex last_event_idx{};
    DeviceTick expected_device_tick{};
    DeviceTick previous_device_tick_start{};
    SessionId session_id{};
    const SignalBlockHeader* validated_block{};
    bool anchored{};
    bool has_last_event{};
};

struct ContinuityChecker::GapSummary
{
    std::size_t count{};
    GapReason primary_reason{GapReason::frame_sequence_gap};
    ContinuityError error{ContinuityError::none};
};

ContinuityChecker::ContinuityChecker(const StreamSchema& schema)
    : n_signals_(schema.signals().size()), frame_validator_(schema),
      states_(std::make_unique<SignalState[]>(n_signals_)),
      gap_scratch_(std::make_unique_for_overwrite<SignalGap[]>(n_signals_))
{
    for (std::size_t i = 0; i < n_signals_; ++i)
    {
        const auto& signal = schema.signals()[i];
        states_[i].id = signal.id;
        states_[i].tick_tracking = signal.device_tick_tracking;
        states_[i].kind = signal.kind;
    }
    std::sort(states_.get(), states_.get() + n_signals_,
              [](const SignalState& left, const SignalState& right) { return left.id < right.id; });
}

ContinuityChecker::~ContinuityChecker() = default;

void ContinuityChecker::clear_states() noexcept
{
    for (std::size_t i = 0; i < n_signals_; ++i)
    {
        const auto id = states_[i].id;
        const auto tick_tracking = states_[i].tick_tracking;
        const auto kind = states_[i].kind;
        states_[i] = SignalState{
            .id = id,
            .tick_tracking = tick_tracking,
            .kind = kind,
        };
    }
    previous_frame_sequence_ = 0;
    anchored_ = false;
}

void ContinuityChecker::accept_discontinuity(const Discontinuity& discontinuity) noexcept
{
    static_cast<void>(discontinuity);
    clear_states();
}

void ContinuityChecker::reset() noexcept
{
    clear_states();
    session_id_ = 0;
}

ContinuityChecker::SignalState* ContinuityChecker::find_state(SignalId id) noexcept
{
    std::size_t first = 0;
    std::size_t last = n_signals_;
    while (first < last)
    {
        const auto middle = first + (last - first) / 2;
        if (states_[middle].id < id)
        {
            first = middle + 1;
        }
        else
        {
            last = middle;
        }
    }
    return first < n_signals_ && states_[first].id == id ? &states_[first] : nullptr;
}

ContinuityError ContinuityChecker::validate_sequence(FrameView frame) const noexcept
{
    if (anchored_)
    {
        if (frame.header.session_id != session_id_)
        {
            return ContinuityError::session_changed;
        }
        if (frame.header.sequence == previous_frame_sequence_)
        {
            return ContinuityError::duplicate_frame_sequence;
        }
        if (frame.header.sequence < previous_frame_sequence_)
        {
            return ContinuityError::frame_sequence_regressed;
        }
    }
    return frame.header.sequence == std::numeric_limits<std::uint64_t>::max()
               ? ContinuityError::frame_sequence_overflow
               : ContinuityError::none;
}

ContinuityChecker::GapSummary ContinuityChecker::detect_gaps(FrameView frame) noexcept
{
    GapSummary summary;
    for (std::size_t i = 0; i < n_signals_; ++i)
    {
        auto& state = states_[i];
        const auto* block = state.validated_block;
        if (block == nullptr || !state.anchored)
        {
            continue;
        }
        if (state.session_id != frame.header.session_id)
        {
            summary.error = ContinuityError::session_changed;
            return summary;
        }

        if (state.kind == SignalKind::spike)
        {
            // Spike blocks are sparse: ``n_samples`` is an event count and
            // ``sample_idx_start`` is the first event's absolute sample index.
            // They do not cover a contiguous sample region, so the dense
            // sample-span gap check must not run: a normal inter-event gap is
            // not a missing-sample discontinuity, and an empty spike block
            // (``n_samples == 0``) must not synthesize one either. Only the
            // frame-sequence gap and an event-index non-regression rule apply;
            // first/last event indices are only meaningful for non-empty
            // blocks, so empty blocks neither update nor check it.
            if (block->n_samples != 0 && state.has_last_event &&
                block->sample_idx_start < state.last_event_idx)
            {
                summary.error = ContinuityError::sample_idx_regressed;
                return summary;
            }
            const bool frame_gap = frame.header.sequence > state.expected_frame_sequence;
            if (frame_gap)
            {
                gap_scratch_[summary.count++] = SignalGap{
                    .expected_sample_idx =
                        state.has_last_event ? state.last_event_idx : block->sample_idx_start,
                    .actual_sample_idx = block->sample_idx_start,
                    .missing_samples = 0,
                    .expected_device_tick = block->device_tick_start,
                    .actual_device_tick = block->device_tick_start,
                    .signal_id = state.id,
                    .reason = GapReason::frame_sequence_gap,
                    .flags = SignalGapFlags::none,
                };
                summary.primary_reason =
                    summary.count == 1
                        ? GapReason::frame_sequence_gap
                        : stronger_reason(summary.primary_reason, GapReason::frame_sequence_gap);
            }
            continue;
        }

        if (block->sample_idx_start < state.expected_sample_idx)
        {
            summary.error = ContinuityError::sample_idx_regressed;
            return summary;
        }

        const bool frame_gap = frame.header.sequence > state.expected_frame_sequence;
        const auto missing_samples = block->sample_idx_start - state.expected_sample_idx;
        auto reason = frame_gap ? GapReason::frame_sequence_gap : GapReason::sample_gap;
        bool has_gap = frame_gap || missing_samples != 0;
        if (missing_samples != 0)
        {
            reason = GapReason::sample_gap;
        }

        auto flags = SignalGapFlags::missing_samples_known;
        if (state.tick_tracking == DeviceTickTracking::sample_counter)
        {
            flags = flags | SignalGapFlags::device_ticks_available;
            if (block->device_tick_start != state.expected_device_tick)
            {
                const bool expected_wrap =
                    state.expected_device_tick < state.previous_device_tick_start;
                const auto tick_reason =
                    !expected_wrap && block->device_tick_start < state.previous_device_tick_start
                        ? GapReason::device_restart
                        : GapReason::device_tick_gap;
                reason = has_gap ? stronger_reason(reason, tick_reason) : tick_reason;
                has_gap = true;
            }
        }

        if (has_gap)
        {
            gap_scratch_[summary.count++] = SignalGap{
                .expected_sample_idx = state.expected_sample_idx,
                .actual_sample_idx = block->sample_idx_start,
                .missing_samples = missing_samples,
                .expected_device_tick = state.expected_device_tick,
                .actual_device_tick = block->device_tick_start,
                .signal_id = state.id,
                .reason = reason,
                .flags = flags,
            };
            summary.primary_reason =
                summary.count == 1 ? reason : stronger_reason(summary.primary_reason, reason);
        }
    }
    return summary;
}

void ContinuityChecker::commit(FrameView frame) noexcept
{
    for (std::size_t i = 0; i < n_signals_; ++i)
    {
        auto& state = states_[i];
        const auto* block = state.validated_block;
        state.expected_frame_sequence = frame.header.sequence + 1;
        if (block == nullptr)
        {
            continue;
        }
        if (state.kind == SignalKind::spike)
        {
            // Only non-empty spike blocks carry first/last event indices; empty
            // blocks leave the previously anchored last-event index untouched.
            // The dense sample-span and device-tick bookkeeping does not apply
            // to sparse event streams.
            if (block->n_samples != 0)
            {
                state.last_event_idx = block->last_sample_idx;
                state.has_last_event = true;
            }
            state.session_id = frame.header.session_id;
            state.anchored = true;
            continue;
        }
        state.expected_sample_idx = block->sample_idx_start + block->n_samples;
        state.previous_device_tick_start = block->device_tick_start;
        if (state.tick_tracking == DeviceTickTracking::sample_counter)
        {
            state.expected_device_tick = block->device_tick_start + block->n_samples;
        }
        state.session_id = frame.header.session_id;
        state.anchored = true;
    }
    session_id_ = frame.header.session_id;
    previous_frame_sequence_ = frame.header.sequence;
    anchored_ = true;
}

ContinuityOutput ContinuityChecker::check(FrameLease frame_lease,
                                          DiscontinuityPool& discontinuity_pool) noexcept
{
    if (!frame_lease)
    {
        return fatal(ContinuityError::invalid_frame_lease);
    }
    const auto frame = frame_lease.view();
    const auto frame_error = frame_validator_.validate(frame);
    if (frame_error != FrameValidationError::none)
    {
        return fatal(map_validation_error(frame_error));
    }
    const auto sequence_error = validate_sequence(frame);
    if (sequence_error != ContinuityError::none)
    {
        return fatal(sequence_error);
    }
    for (std::size_t i = 0; i < n_signals_; ++i)
    {
        states_[i].validated_block = nullptr;
    }
    for (const auto& block : frame.blocks)
    {
        find_state(block.signal_id)->validated_block = &block;
    }

    const auto gaps = detect_gaps(frame);
    if (gaps.error != ContinuityError::none)
    {
        return fatal(gaps.error);
    }

    DiscontinuityLease discontinuity_lease;
    if (gaps.count != 0)
    {
        const auto status = discontinuity_pool.try_acquire(discontinuity_lease);
        if (status == StreamStatus::buffer_exhausted)
        {
            return fatal(ContinuityError::discontinuity_pool_exhausted);
        }
        if (status != StreamStatus::ok || discontinuity_lease.gap_capacity() < gaps.count ||
            discontinuity_lease.assign(frame.header.session_id, previous_frame_sequence_,
                                       frame.header.sequence, gaps.primary_reason,
                                       {gap_scratch_.get(), gaps.count}) != StreamStatus::ok)
        {
            return fatal(ContinuityError::insufficient_gap_capacity);
        }
    }

    commit(frame);
    ContinuityOutput output{
        .status = gaps.count == 0 ? ContinuityStatus::continuous : ContinuityStatus::discontinuity,
        .error = ContinuityError::none,
        .n_messages = static_cast<std::uint8_t>(gaps.count == 0 ? 1 : 2),
    };
    if (gaps.count != 0)
    {
        output.messages[0] = StreamMessage::from_discontinuity(std::move(discontinuity_lease));
        output.messages[1] = StreamMessage::from_frame(std::move(frame_lease));
    }
    else
    {
        output.messages[0] = StreamMessage::from_frame(std::move(frame_lease));
    }
    return output;
}

} // namespace neurale::streaming
