/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/continuity.h>
#include <neurale/streaming/spsc_ring.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <utility>

#include "check_returns.h"

namespace
{

using namespace neurale::streaming;

[[nodiscard]] SignalSchema
make_signal(SignalId id, std::uint64_t rate,
            DeviceTickTracking tick_tracking = DeviceTickTracking::unavailable)
{
    return SignalSchema{
        id,
        SignalDType::float32,
        4,
        8,
        64,
        {rate, 1},
        id + 100,
        SignalLayout::sample_major,
        tick_tracking,
    };
}

[[nodiscard]] SignalBlockHeader block(SignalId id, SampleIndex sample, std::uint32_t count,
                                      DeviceTick tick = 0) noexcept
{
    return SignalBlockHeader{
        .sample_idx_start = sample,
        .device_tick_start = tick,
        .payload_byte_count = static_cast<std::uint64_t>(count) * 16,
        .signal_id = id,
        .n_samples = count,
    };
}

// Spike signals are fixed-capacity sparse streams: ``n_samples`` is an event
// count and ``sample_idx_start`` is the first event's absolute sample index.
// Their payload is a fixed layout whose byte size is independent of the event
// count, so the block helper must report that fixed size rather than a
// per-sample byte product.
[[nodiscard]] SignalSchema make_spike_signal(SignalId id, std::uint64_t fixed_bytes)
{
    return SignalSchema{
        id,
        SignalDType::float64,
        4,
        8,
        64,
        {0, 1},
        id + 100,
        SignalLayout::sample_major,
        DeviceTickTracking::unavailable,
        PhysicalUnit::unspecified,
        0,
        0,
        0,
        SignalKind::spike,
        0,
        ObservationTiming::not_applicable,
        fixed_bytes,
    };
}

[[nodiscard]] SignalBlockHeader spike_block(SignalId id, SampleIndex first_event,
                                            SampleIndex last_event, std::uint32_t count,
                                            std::uint64_t fixed_bytes) noexcept
{
    return SignalBlockHeader{
        .sample_idx_start = first_event,
        .device_tick_start = 0,
        .payload_byte_count = fixed_bytes,
        .signal_id = id,
        .n_samples = count,
        .last_sample_idx = last_event,
    };
}

template <std::size_t N>
[[nodiscard]] FrameLease make_frame(FramePool& pool, SessionId session, std::uint64_t sequence,
                                    SchemaId schema, const std::array<SignalBlockHeader, N>& blocks)
{
    FrameLease lease;
    if (pool.try_acquire(lease) != StreamStatus::ok)
    {
        std::abort();
    }
    auto& mutable_frame = lease.frame();
    mutable_frame.header() = FrameHeader{
        .session_id = session,
        .sequence = sequence,
        .schema_id = schema,
        .signal_block_count = static_cast<std::uint32_t>(N),
    };
    auto storage = mutable_frame.block_storage();
    std::uint64_t payload_size = 0;
    for (std::size_t i = 0; i < N; ++i)
    {
        storage[i] = blocks[i];
        storage[i].payload_offset = payload_size;
        payload_size += storage[i].payload_byte_count;
    }
    if (mutable_frame.set_used_sizes(N, payload_size) != StreamStatus::ok)
    {
        std::abort();
    }
    return lease;
}

int test_sample_gap_and_reanchor()
{
    const std::array signals{make_signal(1, 1'000)};
    const StreamSchema schema{7, signals};
    ContinuityChecker checker{schema};
    FramePool frames{2, 256, 1};
    DiscontinuityPool gaps{1, 1};

    const std::array first{block(1, 0, 10)};
    auto output = checker.check(make_frame(frames, 3, 0, 7, first), gaps);
    CHECK(output.status == ContinuityStatus::continuous);
    CHECK(output.messages[0].frame().header.sequence == 0);

    const std::array skipped{block(1, 15, 10)};
    output = checker.check(make_frame(frames, 3, 1, 7, skipped), gaps);
    CHECK(output.status == ContinuityStatus::discontinuity);
    const auto discontinuity = output.messages[0].discontinuity();
    CHECK(discontinuity.reason == GapReason::sample_gap);
    CHECK(discontinuity.signal_gaps.size() == 1);
    CHECK(discontinuity.signal_gaps[0].expected_sample_idx == 10);
    CHECK(discontinuity.signal_gaps[0].actual_sample_idx == 15);
    CHECK(discontinuity.signal_gaps[0].missing_samples == 5);

    const std::array continued{block(1, 25, 10)};
    output = checker.check(make_frame(frames, 3, 2, 7, continued), gaps);
    CHECK(output.status == ContinuityStatus::continuous);
    return 0;
}

int test_multirate_signals_are_independent()
{
    const std::array signals{make_signal(10, 1'000), make_signal(20, 250)};
    const StreamSchema schema{4, signals};
    ContinuityChecker checker{schema};
    FramePool frames{2, 512, 2};
    DiscontinuityPool gaps{1, 2};

    const std::array first{block(20, 0, 2), block(10, 0, 10)};
    CHECK(checker.check(make_frame(frames, 8, 0, 4, first), gaps).status ==
          ContinuityStatus::continuous);
    const std::array second{block(10, 12, 10), block(20, 2, 2)};
    auto output = checker.check(make_frame(frames, 8, 1, 4, second), gaps);
    const auto discontinuity = output.messages[0].discontinuity();
    CHECK(discontinuity.signal_gaps.size() == 1);
    CHECK(discontinuity.signal_gaps[0].signal_id == 10);
    CHECK(discontinuity.signal_gaps[0].missing_samples == 2);

    const std::array third{block(20, 4, 2), block(10, 22, 10)};
    output = checker.check(make_frame(frames, 8, 2, 4, third), gaps);
    CHECK(output.status == ContinuityStatus::continuous);
    return 0;
}

int test_partial_multirate_frames_preserve_continuity()
{
    const std::array signals{make_signal(10, 1'000), make_signal(20, 250)};
    const StreamSchema schema{4, signals};
    ContinuityChecker checker{schema};
    FramePool frames{2, 512, 2};
    DiscontinuityPool gaps{1, 2};

    const std::array first{block(10, 0, 10), block(20, 0, 2)};
    CHECK(checker.check(make_frame(frames, 8, 0, 4, first), gaps).status ==
          ContinuityStatus::continuous);
    const std::array fast_only{block(10, 10, 10)};
    CHECK(checker.check(make_frame(frames, 8, 1, 4, fast_only), gaps).status ==
          ContinuityStatus::continuous);
    const std::array slow_only{block(20, 2, 2)};
    CHECK(checker.check(make_frame(frames, 8, 2, 4, slow_only), gaps).status ==
          ContinuityStatus::continuous);
    const std::array both{block(10, 20, 10), block(20, 4, 2)};
    CHECK(checker.check(make_frame(frames, 8, 3, 4, both), gaps).status ==
          ContinuityStatus::continuous);
    return 0;
}

int test_sequence_gap_and_device_counter()
{
    const std::array signals{make_signal(1, 1'000, DeviceTickTracking::sample_counter)};
    const StreamSchema schema{6, signals};
    FramePool frames{2, 256, 1};
    DiscontinuityPool gaps{1, 1};

    ContinuityChecker sequence_checker{schema};
    const std::array first{block(1, 0, 4, 100)};
    CHECK(sequence_checker.check(make_frame(frames, 2, 100, 6, first), gaps).status ==
          ContinuityStatus::continuous);
    const std::array sequence_gap{block(1, 4, 4, 104)};
    auto output = sequence_checker.check(make_frame(frames, 2, 104, 6, sequence_gap), gaps);
    auto discontinuity = output.messages[0].discontinuity();
    CHECK(discontinuity.signal_gaps[0].missing_samples == 0);
    CHECK(discontinuity.signal_gaps[0].reason == GapReason::frame_sequence_gap);

    ContinuityChecker restart_checker{schema};
    output = restart_checker.check(make_frame(frames, 2, 0, 6, first), gaps);
    const std::array restarted{block(1, 4, 4, 5)};
    output = restart_checker.check(make_frame(frames, 2, 1, 6, restarted), gaps);
    CHECK(output.messages[0].discontinuity().reason == GapReason::device_restart);

    ContinuityChecker wrap_checker{schema};
    constexpr auto max = std::numeric_limits<DeviceTick>::max();
    const std::array before_wrap{block(1, 0, 4, max - 3)};
    output = wrap_checker.check(make_frame(frames, 2, 0, 6, before_wrap), gaps);
    const std::array after_wrap{block(1, 4, 4, 0)};
    output = wrap_checker.check(make_frame(frames, 2, 1, 6, after_wrap), gaps);
    CHECK(output.status == ContinuityStatus::continuous);
    return 0;
}

int test_fatal_validation_is_transactional()
{
    const std::array signals{make_signal(1, 1'000)};
    const StreamSchema schema{9, signals};
    FramePool frames{2, 256, 1};
    DiscontinuityPool gaps{1, 1};
    ContinuityChecker checker{schema};
    const std::array first{block(1, 10, 10)};
    CHECK(checker.check(make_frame(frames, 4, 0, 9, first), gaps).status ==
          ContinuityStatus::continuous);

    const std::array rollback{block(1, 19, 10)};
    auto output = checker.check(make_frame(frames, 4, 1, 9, rollback), gaps);
    CHECK(output.error == ContinuityError::sample_idx_regressed);
    const std::array next{block(1, 20, 10)};
    output = checker.check(make_frame(frames, 4, 1, 9, next), gaps);
    CHECK(output.status == ContinuityStatus::continuous);

    output = checker.check(make_frame(frames, 4, 1, 9, next), gaps);
    CHECK(output.error == ContinuityError::duplicate_frame_sequence);
    output = checker.check(make_frame(frames, 5, 2, 9, next), gaps);
    CHECK(output.error == ContinuityError::session_changed);
    return 0;
}

int test_discontinuity_storage_survives_later_checks()
{
    const std::array signals{make_signal(1, 1'000)};
    const StreamSchema schema{3, signals};
    ContinuityChecker checker{schema};
    FramePool frames{3, 256, 1};
    DiscontinuityPool gaps{2, 1};

    const std::array first{block(1, 0, 10)};
    auto anchor = checker.check(make_frame(frames, 1, 0, 3, first), gaps);
    anchor = ContinuityOutput{};
    const std::array gap_one{block(1, 12, 10)};
    auto first_gap = checker.check(make_frame(frames, 1, 1, 3, gap_one), gaps);
    const auto* const first_gap_address = first_gap.messages[0].discontinuity().signal_gaps.data();

    const std::array gap_two{block(1, 25, 10)};
    auto second_gap = checker.check(make_frame(frames, 1, 2, 3, gap_two), gaps);
    const auto saved = first_gap.messages[0].discontinuity();
    CHECK(saved.signal_gaps.data() == first_gap_address);
    CHECK(saved.signal_gaps[0].expected_sample_idx == 10);
    CHECK(saved.signal_gaps[0].actual_sample_idx == 12);
    CHECK(saved.signal_gaps[0].missing_samples == 2);
    CHECK(second_gap.messages[0].discontinuity().signal_gaps.data() != first_gap_address);
    return 0;
}

int test_ordered_queue_and_pool_exhaustion()
{
    const std::array signals{make_signal(1, 1'000)};
    const StreamSchema schema{3, signals};
    ContinuityChecker checker{schema};
    FramePool frames{2, 256, 1};
    DiscontinuityPool gaps{1, 1};
    SpscRing<StreamMessage> ring{3};

    const std::array first{block(1, 0, 10)};
    auto output = checker.check(make_frame(frames, 1, 100, 3, first), gaps);
    CHECK(ring.try_push(std::move(output.messages[0])) == StreamStatus::ok);
    const std::array second{block(1, 10, 10)};
    output = checker.check(make_frame(frames, 1, 104, 3, second), gaps);
    CHECK(ring.try_push(std::move(output.messages[0])) == StreamStatus::ok);
    CHECK(ring.try_push(std::move(output.messages[1])) == StreamStatus::ok);

    StreamMessage message;
    CHECK(ring.try_pop(message) == StreamStatus::ok);
    CHECK(message.frame().header.sequence == 100);
    CHECK(ring.try_pop(message) == StreamStatus::ok);
    CHECK(message.kind() == StreamMessageKind::discontinuity);
    CHECK(ring.try_pop(message) == StreamStatus::ok);
    CHECK(message.frame().header.sequence == 104);
    message = StreamMessage::end_of_stream();

    ContinuityChecker exhausted_checker{schema};
    const std::array anchor_blocks{block(1, 0, 1)};
    output = exhausted_checker.check(make_frame(frames, 1, 0, 3, anchor_blocks), gaps);
    const std::array gap_blocks{block(1, 2, 1)};
    auto held_gap = exhausted_checker.check(make_frame(frames, 1, 1, 3, gap_blocks), gaps);
    output = ContinuityOutput{};
    const std::array another_gap{block(1, 4, 1)};
    output = exhausted_checker.check(make_frame(frames, 1, 2, 3, another_gap), gaps);
    CHECK(output.error == ContinuityError::discontinuity_pool_exhausted);
    held_gap = ContinuityOutput{};
    output = exhausted_checker.check(make_frame(frames, 1, 2, 3, another_gap), gaps);
    CHECK(output.status == ContinuityStatus::discontinuity);
    CHECK(output.messages[0].discontinuity().signal_gaps[0].expected_sample_idx == 3);
    return 0;
}

int test_spike_sparse_blocks_are_continuous()
{
    constexpr std::uint64_t fixed_bytes = 256;
    const std::array signals{make_spike_signal(1, fixed_bytes)};
    const StreamSchema schema{11, signals};
    ContinuityChecker checker{schema};
    FramePool frames{2, 512, 1};
    DiscontinuityPool gaps{1, 1};

    const std::array first{spike_block(1, 101, 105, 3, fixed_bytes)};
    CHECK(checker.check(make_frame(frames, 3, 0, 11, first), gaps).status ==
          ContinuityStatus::continuous);
    // Multiple spikes may share an aligned sample index across adjacent
    // blocks, so equality with the previous inclusive last event is valid.
    const std::array equal_boundary{spike_block(1, 105, 110, 2, fixed_bytes)};
    CHECK(checker.check(make_frame(frames, 3, 1, 11, equal_boundary), gaps).status ==
          ContinuityStatus::continuous);
    // A jump to a later event is a normal sparse inter-event interval, not a
    // dense missing-sample gap.
    const std::array second{spike_block(1, 200, 204, 2, fixed_bytes)};
    CHECK(checker.check(make_frame(frames, 3, 2, 11, second), gaps).status ==
          ContinuityStatus::continuous);
    return 0;
}

int test_spike_empty_blocks_are_continuous()
{
    constexpr std::uint64_t fixed_bytes = 256;
    const std::array signals{make_spike_signal(1, fixed_bytes)};
    const StreamSchema schema{12, signals};
    ContinuityChecker checker{schema};
    FramePool frames{2, 512, 1};
    DiscontinuityPool gaps{1, 1};

    const std::array first{spike_block(1, 0, 0, 0, fixed_bytes)};
    CHECK(checker.check(make_frame(frames, 3, 0, 12, first), gaps).status ==
          ContinuityStatus::continuous);
    // An empty spike block (``n_samples == 0``) must not synthesize a gap
    // from the absent sample span.
    const std::array second{spike_block(1, 64, 0, 0, fixed_bytes)};
    CHECK(checker.check(make_frame(frames, 3, 1, 12, second), gaps).status ==
          ContinuityStatus::continuous);
    return 0;
}

int test_spike_empty_to_nonempty_is_continuous()
{
    constexpr std::uint64_t fixed_bytes = 256;
    const std::array signals{make_spike_signal(1, fixed_bytes)};
    const StreamSchema schema{13, signals};
    ContinuityChecker checker{schema};
    FramePool frames{2, 512, 1};
    DiscontinuityPool gaps{1, 1};

    const std::array empty_block{spike_block(1, 0, 0, 0, fixed_bytes)};
    CHECK(checker.check(make_frame(frames, 3, 0, 13, empty_block), gaps).status ==
          ContinuityStatus::continuous);
    // The empty anchor recorded no first-event index, so the first non-empty
    // block must anchor without any regression check.
    const std::array nonempty{spike_block(1, 500, 501, 2, fixed_bytes)};
    CHECK(checker.check(make_frame(frames, 3, 1, 13, nonempty), gaps).status ==
          ContinuityStatus::continuous);
    return 0;
}

int test_spike_nonempty_to_empty_is_continuous()
{
    constexpr std::uint64_t fixed_bytes = 256;
    const std::array signals{make_spike_signal(1, fixed_bytes)};
    const StreamSchema schema{14, signals};
    ContinuityChecker checker{schema};
    FramePool frames{2, 512, 1};
    DiscontinuityPool gaps{1, 1};

    const std::array nonempty{spike_block(1, 101, 105, 3, fixed_bytes)};
    CHECK(checker.check(make_frame(frames, 3, 0, 14, nonempty), gaps).status ==
          ContinuityStatus::continuous);
    // An empty block following a non-empty block must not be treated as a
    // regression against the previous first-event index.
    const std::array empty_block{spike_block(1, 0, 0, 0, fixed_bytes)};
    CHECK(checker.check(make_frame(frames, 3, 1, 14, empty_block), gaps).status ==
          ContinuityStatus::continuous);
    return 0;
}

int test_spike_frame_sequence_gap_is_detected()
{
    constexpr std::uint64_t fixed_bytes = 256;
    const std::array signals{make_spike_signal(1, fixed_bytes)};
    const StreamSchema schema{15, signals};
    ContinuityChecker checker{schema};
    FramePool frames{2, 512, 1};
    DiscontinuityPool gaps{1, 1};

    const std::array first{spike_block(1, 101, 105, 3, fixed_bytes)};
    CHECK(checker.check(make_frame(frames, 3, 0, 15, first), gaps).status ==
          ContinuityStatus::continuous);
    // The frame sequence jumps from 0 to 5; the spike payload is a normal
    // sparse continuation, but the skipped frames must still surface as a
    // frame-sequence discontinuity with no missing samples, and the frame
    // must follow it so downstream stages can consume it.
    const std::array second{spike_block(1, 300, 301, 2, fixed_bytes)};
    auto output = checker.check(make_frame(frames, 3, 5, 15, second), gaps);
    CHECK(output.status == ContinuityStatus::discontinuity);
    const auto discontinuity = output.messages[0].discontinuity();
    CHECK(discontinuity.signal_gaps.size() == 1);
    CHECK(discontinuity.signal_gaps[0].signal_id == 1);
    CHECK(discontinuity.signal_gaps[0].missing_samples == 0);
    CHECK(discontinuity.signal_gaps[0].reason == GapReason::frame_sequence_gap);
    CHECK(discontinuity.reason == GapReason::frame_sequence_gap);
    CHECK(output.messages[1].frame().header.sequence == 5);
    return 0;
}

int test_spike_does_not_mask_dense_discontinuity()
{
    constexpr std::uint64_t fixed_bytes = 256;
    const std::array signals{make_signal(10, 1'000), make_spike_signal(20, fixed_bytes)};
    const StreamSchema schema{16, signals};
    ContinuityChecker checker{schema};
    FramePool frames{2, 768, 2};
    DiscontinuityPool gaps{1, 2};

    const std::array first{block(10, 0, 10), spike_block(20, 101, 105, 3, fixed_bytes)};
    CHECK(checker.check(make_frame(frames, 3, 0, 16, first), gaps).status ==
          ContinuityStatus::continuous);
    // The dense signal has a real 5-sample gap; the spike signal merely
    // advances to a sparse later event. The dense discontinuity must still be
    // detected and emitted for downstream propagation, while the sparse spike
    // interval contributes no gap.
    const std::array second{block(10, 15, 10), spike_block(20, 200, 204, 2, fixed_bytes)};
    auto output = checker.check(make_frame(frames, 3, 1, 16, second), gaps);
    CHECK(output.status == ContinuityStatus::discontinuity);
    const auto discontinuity = output.messages[0].discontinuity();
    CHECK(discontinuity.signal_gaps.size() == 1);
    CHECK(discontinuity.signal_gaps[0].signal_id == 10);
    CHECK(discontinuity.signal_gaps[0].missing_samples == 5);
    CHECK(discontinuity.signal_gaps[0].reason == GapReason::sample_gap);
    return 0;
}

int test_spike_cross_block_overlap_is_fatal()
{
    constexpr std::uint64_t fixed_bytes = 256;
    const std::array signals{make_spike_signal(1, fixed_bytes)};
    const StreamSchema schema{17, signals};
    ContinuityChecker checker{schema};
    FramePool frames{2, 512, 1};
    DiscontinuityPool gaps{1, 1};

    const std::array first{spike_block(1, 20, 45, 2, fixed_bytes)};
    CHECK(checker.check(make_frame(frames, 3, 0, 17, first), gaps).status ==
          ContinuityStatus::continuous);
    // Comparing only first events would accept 43 >= 20, despite the actual
    // cross-block sequence [20, 45, 43]. The previous last event is the
    // inclusive boundary and therefore makes this overlap fatal.
    const std::array regressed{spike_block(1, 43, 43, 1, fixed_bytes)};
    auto output = checker.check(make_frame(frames, 3, 1, 17, regressed), gaps);
    CHECK(output.status == ContinuityStatus::fatal);
    CHECK(output.error == ContinuityError::sample_idx_regressed);
    return 0;
}

} // namespace

int main()
{
    for (const auto test : {
             test_sample_gap_and_reanchor,
             test_multirate_signals_are_independent,
             test_partial_multirate_frames_preserve_continuity,
             test_sequence_gap_and_device_counter,
             test_fatal_validation_is_transactional,
             test_discontinuity_storage_survives_later_checks,
             test_ordered_queue_and_pool_exhaustion,
             test_spike_sparse_blocks_are_continuous,
             test_spike_empty_blocks_are_continuous,
             test_spike_empty_to_nonempty_is_continuous,
             test_spike_nonempty_to_empty_is_continuous,
             test_spike_frame_sequence_gap_is_detected,
             test_spike_does_not_mask_dense_discontinuity,
             test_spike_cross_block_overlap_is_fatal,
         })
    {
        if (const auto line = test(); line != 0)
        {
            return line;
        }
    }
    return 0;
}
