/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// `NativeReplaySource`: what a run does with a validated replay image.
///
/// The image decides what is emitted and in what order; this suite is about
/// everything a *run* decides -- pacing, permits, cancellation, the end of the
/// data, reset, and injected faults -- plus the two ways a run can refuse to
/// start (a schema that is not the image's, a frame that does not fit the
/// pool). Section numbers below are `docs/development/native_recording_replay.md`.
///
/// One executable, two entries: `--contract-only` is the behaviour, and
/// `--allocation-only` is the gate that proves a steady-state read allocates
/// nothing at all.

#include "allocation_tracker.h"
#include "check_returns.h"
#include "replay/source.h"
#include "replay_test_support.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/discontinuity_pool.h>
#include <neurale/streaming/schema.h>

namespace
{

using neurale::recording::NativeReplaySource;
using neurale::recording::ReplayConfigStatus;
using neurale::recording::ReplayFaultEffect;
using neurale::recording::ReplayFaultSpec;
using neurale::recording::ReplayFaultTarget;
using neurale::recording::ReplayImageStatus;
using neurale::recording::ReplayItemKind;
using neurale::recording::ReplayPacing;
using neurale::recording::ReplaySourceConfig;
using neurale::recording::ReplayTerminal;
using neurale::recording::test::BuiltBlock;
using neurale::recording::test::BuiltDiscontinuity;
using neurale::recording::test::BuiltFrame;
using neurale::recording::test::BuiltGap;
using neurale::recording::test::ReplayImageBuilder;
using neurale::recording::test::Scratch;
using neurale::streaming::DiscontinuityLease;
using neurale::streaming::DiscontinuityPool;
using neurale::streaming::FrameLease;
using neurale::streaming::FramePool;
using neurale::streaming::HostTimeNs;
using neurale::streaming::StreamStatus;

/// Virtual time. A wait moves the clock to the deadline, so a paced run is
/// deterministic and finishes at once; freezing it is how a test stages the
/// catch-up case, where the deadline has already passed.
class FakeClock final : public neurale::streaming::NativeClock
{
  public:
    HostTimeNs now_ns() noexcept override
    {
        return now_.load(std::memory_order_acquire);
    }

    void wait_until(HostTimeNs deadline_ns) noexcept override
    {
        waits_.fetch_add(1, std::memory_order_acq_rel);
        if (frozen_.load(std::memory_order_acquire))
        {
            // A frozen clock never reaches the deadline on its own. The short
            // real sleep is a test-double concession: it keeps a blocked read
            // from spinning while another thread cancels or grants a permit,
            // and it is the only real time this suite ever waits.
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            return;
        }
        auto current = now_.load(std::memory_order_acquire);
        if (deadline_ns > current)
        {
            now_.store(deadline_ns, std::memory_order_release);
        }
    }

    void wake() noexcept override
    {
        wakes_.fetch_add(1, std::memory_order_acq_rel);
    }

    void set(HostTimeNs value) noexcept
    {
        now_.store(value, std::memory_order_release);
    }
    void freeze(bool value) noexcept
    {
        frozen_.store(value, std::memory_order_release);
    }
    [[nodiscard]] std::uint64_t waits() const noexcept
    {
        return waits_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t wakes() const noexcept
    {
        return wakes_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<HostTimeNs> now_{1'000'000};
    std::atomic<std::uint64_t> waits_{};
    std::atomic<std::uint64_t> wakes_{};
    std::atomic<bool> frozen_{};
};

/// A consumer's side of one read: the pools a runtime would own, plus what the
/// last read produced.
struct Reader
{
    FramePool frames{4, 4096, 8};
    DiscontinuityPool discontinuities{4, 8};

    FrameLease frame;
    DiscontinuityLease discontinuity;
    StreamStatus status{StreamStatus::ok};

    StreamStatus next(NativeReplaySource& source) noexcept
    {
        static_cast<void>(frame.reset());
        static_cast<void>(discontinuity.reset());
        if (frames.try_acquire(frame) != StreamStatus::ok ||
            discontinuities.try_acquire(discontinuity) != StreamStatus::ok)
        {
            status = StreamStatus::buffer_exhausted;
            return status;
        }
        status = source.read_message(frame.frame(), discontinuity);
        return status;
    }
};

std::vector<std::byte> bytes_of(std::initializer_list<unsigned char> values)
{
    std::vector<std::byte> result;
    for (const auto value : values)
    {
        result.push_back(static_cast<std::byte>(value));
    }
    return result;
}

BuiltFrame make_frame(std::uint64_t ordinal, std::uint64_t replay_sequence,
                      std::uint64_t original_sequence, std::uint64_t timeline_ns,
                      std::initializer_list<unsigned char> payload)
{
    BuiltFrame frame;
    frame.data_message_ordinal = ordinal;
    frame.replay_sequence = replay_sequence;
    frame.original_sequence = original_sequence;
    frame.timeline_ns = timeline_ns;
    frame.original_host_received_ns = timeline_ns;
    frame.payload = bytes_of(payload);
    BuiltBlock block;
    block.payload_byte_count = frame.payload.size();
    block.n_samples = static_cast<std::uint32_t>(frame.payload.size());
    block.sample_idx_start = replay_sequence * 4;
    block.device_tick_start = replay_sequence * 8;
    block.source_block_ordinal = replay_sequence;
    frame.blocks.push_back(block);
    return frame;
}

/// Three frames with a recorded hole between the second and the third, and the
/// discontinuity that explains it: enough shape for every ordering question.
ReplayImageBuilder ledger_builder(bool exact)
{
    ReplayImageBuilder builder;
    builder.mode = exact ? "exact_frames" : "recorded_projection";
    builder.add_frame(make_frame(0, exact ? 41 : 0, 41, 0, {1, 2, 3, 4}));
    builder.add_frame(make_frame(1, exact ? 42 : 1, 42, 1'000, {5, 6, 7, 8}));
    BuiltDiscontinuity record;
    record.data_message_ordinal = 2;
    record.replay_sequence = exact ? 44 : 2;
    record.original_sequence = 44;
    record.previous_replay_sequence = exact ? 42 : 1;
    record.original_previous_sequence = 42;
    record.timeline_ns = 2'000;
    record.reason = "sample_gap";
    BuiltGap gap;
    gap.native_signal_id = 1;
    gap.expected_sample_idx = 8;
    gap.actual_sample_idx = 20;
    gap.missing_samples = 12;
    gap.signal_gap_ordinal = 0;
    record.gaps.push_back(gap);
    builder.add_discontinuity(std::move(record));
    builder.add_frame(make_frame(3, exact ? 44 : 2, 44, 3'000, {9, 10, 11, 12}));
    builder.fidelity.push_back({.stream = "stream", .native_signal_id = 1});
    builder.signals.push_back({});
    return builder;
}

[[nodiscard]] ReplayImageStatus write_and_open(ReplayImageBuilder& builder, const std::string& path,
                                               NativeReplaySource& source)
{
    if (!builder.write(path))
    {
        return ReplayImageStatus::unreadable;
    }
    return source.open(path.c_str());
}

// --- the contract suite ------------------------------------------------------

int check_image_semantics(const Scratch& scratch)
{
    for (const bool exact : {true, false})
    {
        auto builder = ledger_builder(exact);
        NativeReplaySource source;
        FakeClock clock;
        const auto path = scratch.path(exact ? "exact.nrimg" : "projection.nrimg");
        CHECK(write_and_open(builder, path, source) == ReplayImageStatus::ok);
        ReplaySourceConfig config;
        config.clock = &clock;
        config.replay_run_session_id = 4242;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        CHECK(source.session_id() == 4242);

        Reader reader;
        CHECK(reader.next(source) == StreamStatus::ok);
        const auto first = reader.frame.frame().view();
        // exact_frames emits the recorded sequence; a projection renumbers
        // from zero and keeps the original as provenance only (8.1, 8.2).
        CHECK(first.header.sequence == (exact ? 41U : 0U));
        // The run's identity, never the recording's (8.12).
        CHECK(first.header.session_id == 4242);
        // Current-run host time, not the recorded arrival time (8.6).
        CHECK(first.header.host_received_ns == clock.now_ns());
        CHECK(first.header.host_received_ns != 0);
        CHECK(!has_flag(first.header.flags, neurale::streaming::FrameFlags::valid_until));
        CHECK(first.blocks.size() == 1);
        CHECK(first.blocks[0].signal_id == 1);
        CHECK(first.blocks[0].payload_byte_count == 4);
        CHECK(first.payload.size() == 4);
        CHECK(first.payload[0] == std::byte{1});
        CHECK(first.payload[3] == std::byte{4});

        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(reader.frame.frame().view().header.sequence == (exact ? 42U : 1U));

        CHECK(reader.next(source) == StreamStatus::discontinuity);
        const auto discontinuity = reader.discontinuity.view();
        CHECK(discontinuity.session_id == 4242);
        CHECK(discontinuity.reason == neurale::streaming::GapReason::sample_gap);
        CHECK(discontinuity.previous_frame_sequence == (exact ? 42U : 1U));
        CHECK(discontinuity.actual_frame_sequence == (exact ? 44U : 2U));
        CHECK(discontinuity.signal_gaps.size() == 1);
        CHECK(discontinuity.signal_gaps[0].missing_samples == 12);
        CHECK(has_flag(discontinuity.signal_gaps[0].flags,
                       neurale::streaming::SignalGapFlags::missing_samples_known));
        CHECK(discontinuity.signal_gaps[0].actual_sample_idx == 20);

        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(reader.frame.frame().view().header.sequence == (exact ? 44U : 2U));
        CHECK(reader.frame.frame().view().payload[0] == std::byte{9});

        // End of data is a distinct terminal result, reported exactly once and
        // then repeated on every later read without ever re-emitting (8.9).
        CHECK(reader.next(source) == StreamStatus::end_of_stream);
        CHECK(source.terminal() == ReplayTerminal::end_of_data);
        CHECK(reader.next(source) == StreamStatus::end_of_stream);
        CHECK(source.stats().frames_emitted == 3);
        CHECK(source.stats().discontinuities_emitted == 1);
    }

    {
        // A synthesized image carries no clock-sync snapshot, and the reader
        // reports that absence rather than inventing one (8.6).
        ReplayImageBuilder builder;
        builder.mode = "stream_frames";
        builder.flags = 0;
        auto frame = make_frame(neurale::recording::test::kAbsentU64, 0, 0, 0, {7, 7});
        frame.blocks[0].clock_sync = false;
        frame.blocks[0].clock_reference_tick = 999;
        builder.add_frame(std::move(frame));
        builder.fidelity.push_back({.stream = "cursor", .native_signal_id = 1});

        NativeReplaySource source;
        FakeClock clock;
        const auto path = scratch.path("synth.nrimg");
        CHECK(write_and_open(builder, path, source) == ReplayImageStatus::ok);
        ReplaySourceConfig config;
        config.clock = &clock;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        CHECK(source.session_id() != 0);
        Reader reader;
        CHECK(reader.next(source) == StreamStatus::ok);
        const auto view = reader.frame.frame().view();
        CHECK(view.blocks[0].clock_sync.device_tick_reference == 0);
        CHECK(view.blocks[0].clock_sync.host_time_reference_ns == 0);
    }

    {
        // A recorded deadline is cleared by default and preserved only when
        // the caller asks for it explicitly (8.6).
        ReplayImageBuilder builder;
        auto frame = make_frame(0, 0, 0, 0, {1});
        frame.has_valid_until = true;
        frame.valid_until_ns = 555;
        frame.has_source_tick = true;
        frame.source_tick = 77;
        builder.add_frame(std::move(frame));
        const auto path = scratch.path("deadline.nrimg");

        for (const bool preserve : {false, true})
        {
            NativeReplaySource source;
            FakeClock clock;
            CHECK(write_and_open(builder, path, source) == ReplayImageStatus::ok);
            ReplaySourceConfig config;
            config.clock = &clock;
            config.preserve_recorded_valid_until = preserve;
            CHECK(source.prepare(config) == ReplayConfigStatus::ok);
            Reader reader;
            CHECK(reader.next(source) == StreamStatus::ok);
            const auto header = reader.frame.frame().view().header;
            CHECK(header.source_tick == 77);
            CHECK(has_flag(header.flags, neurale::streaming::FrameFlags::source_tick));
            CHECK(header.valid_until_ns == (preserve ? 555U : 0U));
            CHECK(has_flag(header.flags, neurale::streaming::FrameFlags::valid_until) == preserve);
        }
    }
    return 0;
}

int check_pacing(const Scratch& scratch)
{
    {
        // as_fast_as_possible waits for nothing at all.
        auto builder = ledger_builder(true);
        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("afap.nrimg"), source) == ReplayImageStatus::ok);
        ReplaySourceConfig config;
        config.clock = &clock;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        Reader reader;
        for (int i = 0; i < 4; ++i)
        {
            CHECK(reader.next(source) != StreamStatus::end_of_stream);
        }
        CHECK(reader.next(source) == StreamStatus::end_of_stream);
        CHECK(clock.waits() == 0);
    }

    for (const double factor : {1.0, 2.0})
    {
        // Recorded pacing places each item at its recorded offset from the
        // first emitted one, scaled -- and never by adding one sleep to the
        // last wake-up (8.7).
        auto builder = ledger_builder(true);
        NativeReplaySource source;
        FakeClock clock;
        clock.set(500'000);
        CHECK(write_and_open(builder, scratch.path("paced.nrimg"), source) ==
              ReplayImageStatus::ok);
        ReplaySourceConfig config;
        config.clock = &clock;
        config.pacing = ReplayPacing::recorded;
        config.speed_factor = factor;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);

        Reader reader;
        CHECK(reader.next(source) == StreamStatus::ok);
        const auto origin = clock.now_ns();
        CHECK(origin == 500'000); // no time is waited before the first item
        const std::uint64_t offsets[] = {1'000, 2'000, 3'000};
        for (const auto offset : offsets)
        {
            CHECK(reader.next(source) != StreamStatus::end_of_stream);
            const auto expected =
                origin + static_cast<std::uint64_t>(static_cast<double>(offset) / factor);
            CHECK(clock.now_ns() == expected);
        }
        CHECK(reader.next(source) == StreamStatus::end_of_stream);
        CHECK(source.stats().late_item_count == 0);
    }

    {
        // Catch-up: a deadline that has already passed fires at once, nothing
        // is skipped or compressed, and the lateness is reported (8.7).
        auto builder = ledger_builder(true);
        NativeReplaySource source;
        FakeClock clock;
        clock.set(1'000);
        CHECK(write_and_open(builder, scratch.path("late.nrimg"), source) == ReplayImageStatus::ok);
        ReplaySourceConfig config;
        config.clock = &clock;
        config.pacing = ReplayPacing::recorded;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);

        Reader reader;
        CHECK(reader.next(source) == StreamStatus::ok);
        clock.set(1'000 + 2'500); // past the second and third deadlines
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(clock.now_ns() == 3'500); // emitted immediately, no waiting back
        CHECK(reader.next(source) == StreamStatus::discontinuity);
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(reader.next(source) == StreamStatus::end_of_stream);
        const auto stats = source.stats();
        CHECK(stats.late_item_count == 2);
        CHECK(stats.max_lateness_ns == 1'500);
        CHECK(stats.frames_emitted == 3);
        CHECK(stats.discontinuities_emitted == 1);
    }

    {
        // Recorded host time that runs backwards is clamped to zero, counted,
        // and never waited for backwards (8.7).
        ReplayImageBuilder builder;
        builder.add_frame(make_frame(0, 0, 0, 5'000, {1}));
        builder.add_frame(make_frame(1, 1, 1, 4'000, {2}));
        builder.add_frame(make_frame(2, 2, 2, 6'000, {3}));
        NativeReplaySource source;
        FakeClock clock;
        clock.set(100);
        CHECK(write_and_open(builder, scratch.path("backwards.nrimg"), source) ==
              ReplayImageStatus::ok);
        ReplaySourceConfig config;
        config.clock = &clock;
        config.pacing = ReplayPacing::recorded;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        Reader reader;
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(clock.now_ns() == 100);
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(clock.now_ns() == 100 + 2'000);
        CHECK(source.stats().negative_delta_count == 1);
    }

    {
        // speed_factor belongs to recorded pacing alone, and is rejected --
        // not ignored -- everywhere else (8.7, 8.8).
        auto builder = ledger_builder(true);
        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("speed.nrimg"), source) ==
              ReplayImageStatus::ok);
        for (const auto pacing : {ReplayPacing::as_fast_as_possible, ReplayPacing::step})
        {
            ReplaySourceConfig config;
            config.clock = &clock;
            config.pacing = pacing;
            config.speed_factor = 2.0;
            CHECK(source.prepare(config) == ReplayConfigStatus::speed_factor_rejected);
        }
        for (const double factor : {0.0, -1.0, 1001.0})
        {
            ReplaySourceConfig config;
            config.clock = &clock;
            config.pacing = ReplayPacing::recorded;
            config.speed_factor = factor;
            CHECK(source.prepare(config) == ReplayConfigStatus::invalid_speed_factor);
        }
    }
    return 0;
}

int check_step(const Scratch& scratch)
{
    auto builder = ledger_builder(true);
    NativeReplaySource source;
    FakeClock clock;
    CHECK(write_and_open(builder, scratch.path("step.nrimg"), source) == ReplayImageStatus::ok);
    ReplaySourceConfig config;
    config.clock = &clock;
    config.pacing = ReplayPacing::step;
    config.blocking = false;
    CHECK(source.prepare(config) == ReplayConfigStatus::ok);

    Reader reader;
    // With zero permits the source produces nothing and does not spin.
    CHECK(reader.next(source) == StreamStatus::would_block);
    CHECK(reader.next(source) == StreamStatus::would_block);

    CHECK(source.advance(2) == 2);
    CHECK(reader.next(source) == StreamStatus::ok);
    CHECK(reader.next(source) == StreamStatus::ok);
    CHECK(source.permits() == 0);
    CHECK(reader.next(source) == StreamStatus::would_block);

    // A discontinuity costs a permit exactly as a frame does (8.8).
    CHECK(source.advance(1) == 1);
    CHECK(reader.next(source) == StreamStatus::discontinuity);
    CHECK(source.permits() == 0);
    CHECK(source.advance(1) == 1);
    CHECK(reader.next(source) == StreamStatus::ok);
    CHECK(source.advance(1) == 1);
    CHECK(reader.next(source) == StreamStatus::end_of_stream);
    CHECK(source.stats().permits_consumed == 4);
    CHECK(source.advance(0) == 0);

    {
        // The counter is uint64 and saturates; a call at the bound grants
        // nothing and is reported as saturated rather than wrapping (8.8).
        NativeReplaySource saturating;
        auto other = ledger_builder(true);
        CHECK(write_and_open(other, scratch.path("saturate.nrimg"), saturating) ==
              ReplayImageStatus::ok);
        ReplaySourceConfig step;
        step.clock = &clock;
        step.pacing = ReplayPacing::step;
        step.blocking = false;
        CHECK(saturating.prepare(step) == ReplayConfigStatus::ok);
        CHECK(saturating.advance(0xFFFFFFFFFFFFFFFFULL) == 0xFFFFFFFFFFFFFFFFULL);
        CHECK(!saturating.stats().permit_saturated);
        CHECK(saturating.advance(1) == 0);
        CHECK(saturating.stats().permit_saturated);
        CHECK(saturating.permits() == 0xFFFFFFFFFFFFFFFFULL);
    }

    {
        // A blocking step read waits, and a permit granted from another thread
        // wakes it -- neither form spins on an empty counter.
        NativeReplaySource blocking;
        auto other = ledger_builder(true);
        FakeClock frozen;
        frozen.freeze(true);
        CHECK(write_and_open(other, scratch.path("step-blocking.nrimg"), blocking) ==
              ReplayImageStatus::ok);
        ReplaySourceConfig waiting;
        waiting.clock = &frozen;
        waiting.pacing = ReplayPacing::step;
        CHECK(blocking.prepare(waiting) == ReplayConfigStatus::ok);

        Reader other_reader;
        std::atomic<StreamStatus> outcome{StreamStatus::invalid_state};
        std::thread worker([&] { outcome.store(other_reader.next(blocking)); });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        CHECK(blocking.advance(1) == 1);
        worker.join();
        CHECK(outcome.load() == StreamStatus::ok);
        CHECK(frozen.wakes() > 0);
    }
    return 0;
}

int check_cancel_and_reset(const Scratch& scratch)
{
    {
        // Cancellation wakes a waiting source promptly: the wait is bounded by
        // the declared wake latency, not by the remaining pacing interval.
        ReplayImageBuilder builder;
        builder.add_frame(make_frame(0, 0, 0, 0, {1}));
        builder.add_frame(make_frame(1, 1, 1, 30'000'000'000ULL, {2}));
        NativeReplaySource source;
        FakeClock frozen;
        CHECK(write_and_open(builder, scratch.path("cancel.nrimg"), source) ==
              ReplayImageStatus::ok);
        ReplaySourceConfig config;
        config.clock = &frozen;
        config.pacing = ReplayPacing::recorded;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);

        Reader reader;
        CHECK(reader.next(source) == StreamStatus::ok);
        frozen.freeze(true); // the 30-second gap will never elapse on its own
        std::atomic<StreamStatus> outcome{StreamStatus::invalid_state};
        std::thread worker([&] { outcome.store(reader.next(source)); });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        source.cancel();
        worker.join();
        CHECK(outcome.load() == StreamStatus::stopped);
        CHECK(source.terminal() == ReplayTerminal::cancelled);
        // Further cancellation is a no-op and the source emits nothing more.
        source.cancel();
        CHECK(source.terminal() == ReplayTerminal::cancelled);
        CHECK(reader.next(source) == StreamStatus::stopped);
        CHECK(source.stats().frames_emitted == 1);
    }

    {
        // Terminal states do not downgrade: cancelling after end_of_data
        // leaves end_of_data (8.9).
        auto builder = ledger_builder(true);
        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("sticky.nrimg"), source) ==
              ReplayImageStatus::ok);
        ReplaySourceConfig config;
        config.clock = &clock;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        Reader reader;
        while (reader.next(source) != StreamStatus::end_of_stream)
        {
        }
        source.cancel();
        CHECK(source.terminal() == ReplayTerminal::end_of_data);
        CHECK(reader.next(source) == StreamStatus::end_of_stream);
    }

    {
        // Reset returns to the start of the same run: the same items, the same
        // replay-run sequences, the same SessionId (8.10, 8.12).
        auto builder = ledger_builder(false);
        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("reset.nrimg"), source) ==
              ReplayImageStatus::ok);
        ReplaySourceConfig config;
        config.clock = &clock;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        const auto identity = source.session_id();

        std::vector<std::uint64_t> first_run;
        Reader reader;
        while (reader.next(source) != StreamStatus::end_of_stream)
        {
            first_run.push_back(reader.status == StreamStatus::discontinuity
                                    ? reader.discontinuity.view().actual_frame_sequence
                                    : reader.frame.frame().view().header.sequence);
        }
        CHECK(first_run.size() == 4);
        CHECK(source.reset() == StreamStatus::ok);
        CHECK(source.terminal() == ReplayTerminal::none);
        CHECK(source.session_id() == identity);
        CHECK(source.stats().frames_emitted == 0);

        std::vector<std::uint64_t> second_run;
        while (reader.next(source) != StreamStatus::end_of_stream)
        {
            second_run.push_back(reader.status == StreamStatus::discontinuity
                                     ? reader.discontinuity.view().actual_frame_sequence
                                     : reader.frame.frame().view().header.sequence);
        }
        CHECK(first_run == second_run);
        // Reset also clears a terminal state reached by cancellation.
        source.cancel();
        CHECK(source.reset() == StreamStatus::ok);
        CHECK(source.terminal() == ReplayTerminal::none);
    }

    {
        // A reset concurrent with an in-flight read is rejected, never raced.
        ReplayImageBuilder builder;
        builder.add_frame(make_frame(0, 0, 0, 0, {1}));
        builder.add_frame(make_frame(1, 1, 1, 30'000'000'000ULL, {2}));
        NativeReplaySource source;
        FakeClock frozen;
        CHECK(write_and_open(builder, scratch.path("reset-race.nrimg"), source) ==
              ReplayImageStatus::ok);
        ReplaySourceConfig config;
        config.clock = &frozen;
        config.pacing = ReplayPacing::recorded;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        Reader reader;
        CHECK(reader.next(source) == StreamStatus::ok);
        frozen.freeze(true);
        std::atomic<StreamStatus> outcome{StreamStatus::invalid_state};
        std::thread worker([&] { outcome.store(reader.next(source)); });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        CHECK(source.reset() == StreamStatus::invalid_state);
        source.cancel();
        worker.join();
        CHECK(outcome.load() == StreamStatus::stopped);
        CHECK(source.reset() == StreamStatus::ok);
    }
    return 0;
}

int check_faults(const Scratch& scratch)
{
    const auto ledger_target = [](std::uint64_t ordinal, ReplayItemKind kind)
    {
        ReplayFaultTarget target;
        target.ledger_based = true;
        target.kind = kind;
        target.data_message_ordinal = ordinal;
        return target;
    };

    {
        // sequence_gap: the frame is not emitted, so the consumer sees a
        // genuine jump in the replay-run frame sequence (8.11).
        auto builder = ledger_builder(false);
        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("gap.nrimg"), source) == ReplayImageStatus::ok);
        const ReplayFaultSpec faults[] = {
            {.target = ledger_target(1, ReplayItemKind::frame),
             .effect = ReplayFaultEffect::sequence_gap},
        };
        ReplaySourceConfig config;
        config.clock = &clock;
        config.faults = faults;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);

        Reader reader;
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(reader.frame.frame().view().header.sequence == 0);
        CHECK(reader.next(source) == StreamStatus::discontinuity);
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(reader.frame.frame().view().header.sequence == 2); // 1 is missing
        CHECK(reader.next(source) == StreamStatus::end_of_stream);
        CHECK(source.fault_count() == 1);
        CHECK(source.fault_report(0).fired);
        CHECK(source.fault_report(0).emitted_known);
        CHECK(!source.fault_report(0).emitted);
        CHECK(source.stats().frames_emitted == 2);

        // Every fault re-arms on reset and fires again at the same position.
        CHECK(source.reset() == StreamStatus::ok);
        CHECK(!source.fault_report(0).fired);
        while (reader.next(source) != StreamStatus::end_of_stream)
        {
        }
        CHECK(source.fault_report(0).fired);
    }

    {
        // read_failure ends the run in a terminal faulted state; the item is
        // not emitted, and v1 defines no recoverable form.
        auto builder = ledger_builder(false);
        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("failure.nrimg"), source) ==
              ReplayImageStatus::ok);
        const ReplayFaultSpec faults[] = {
            {.target = ledger_target(1, ReplayItemKind::frame),
             .effect = ReplayFaultEffect::read_failure},
        };
        ReplaySourceConfig config;
        config.clock = &clock;
        config.faults = faults;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        Reader reader;
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(reader.next(source) == StreamStatus::source_failure);
        CHECK(source.terminal() == ReplayTerminal::faulted);
        CHECK(reader.next(source) == StreamStatus::source_failure);
        CHECK(!source.fault_report(0).emitted);
    }

    {
        // abnormal_end ends the run with the abnormal terminal result of
        // section 8.5, never with end_of_data.
        auto builder = ledger_builder(false);
        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("abnormal.nrimg"), source) ==
              ReplayImageStatus::ok);
        const ReplayFaultSpec faults[] = {
            {.target = ledger_target(2, ReplayItemKind::discontinuity),
             .effect = ReplayFaultEffect::abnormal_end},
        };
        ReplaySourceConfig config;
        config.clock = &clock;
        config.faults = faults;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        Reader reader;
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(reader.next(source) == StreamStatus::source_failure);
        CHECK(source.terminal() == ReplayTerminal::abnormal_end);
    }

    {
        // stall delays the item by the configured duration and then the run
        // continues normally; the item is still emitted.
        auto builder = ledger_builder(false);
        NativeReplaySource source;
        FakeClock clock;
        clock.set(1'000);
        CHECK(write_and_open(builder, scratch.path("stall.nrimg"), source) ==
              ReplayImageStatus::ok);
        const ReplayFaultSpec faults[] = {
            {.target = ledger_target(1, ReplayItemKind::frame),
             .effect = ReplayFaultEffect::stall,
             .stall_ns = 750},
        };
        ReplaySourceConfig config;
        config.clock = &clock;
        config.faults = faults;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        Reader reader;
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(clock.now_ns() == 1'000);
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(clock.now_ns() == 1'750);
        CHECK(source.fault_report(0).fired);
        CHECK(source.fault_report(0).emitted);
        CHECK(source.stats().injected_delay_ns == 750);
        while (reader.next(source) != StreamStatus::end_of_stream)
        {
        }
        CHECK(source.stats().frames_emitted == 3);
    }

    {
        // A fault whose position is not in this image never fires and is
        // reported as unfired rather than silently dropped (8.11).
        auto builder = ledger_builder(false);
        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("unfired.nrimg"), source) ==
              ReplayImageStatus::ok);
        const ReplayFaultSpec faults[] = {
            {.target = ledger_target(4096, ReplayItemKind::frame),
             .effect = ReplayFaultEffect::read_failure},
        };
        ReplaySourceConfig config;
        config.clock = &clock;
        config.faults = faults;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        Reader reader;
        while (reader.next(source) != StreamStatus::end_of_stream)
        {
        }
        CHECK(source.fault_count() == 1);
        CHECK(!source.fault_report(0).fired);
        CHECK(!source.fault_report(0).emitted_known);
        CHECK(source.fault_report(0).item_idx == neurale::recording::kReplayAbsentU64);
    }

    {
        // The configuration refusals, all before the run starts.
        auto builder = ledger_builder(false);
        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("refusals.nrimg"), source) ==
              ReplayImageStatus::ok);
        ReplaySourceConfig config;
        config.clock = &clock;

        const ReplayFaultSpec at_discontinuity[] = {
            {.target = ledger_target(2, ReplayItemKind::discontinuity),
             .effect = ReplayFaultEffect::sequence_gap},
        };
        config.faults = at_discontinuity;
        CHECK(source.prepare(config) == ReplayConfigStatus::sequence_gap_targets_discontinuity);

        for (const std::uint64_t boundary : {0U, 3U})
        {
            const ReplayFaultSpec at_edge[] = {
                {.target = ledger_target(boundary, ReplayItemKind::frame),
                 .effect = ReplayFaultEffect::sequence_gap},
            };
            config.faults = at_edge;
            CHECK(source.prepare(config) == ReplayConfigStatus::sequence_gap_at_run_boundary);
        }

        const ReplayFaultSpec unbounded[] = {
            {.target = ledger_target(1, ReplayItemKind::frame),
             .effect = ReplayFaultEffect::stall,
             .stall_ns = 0},
        };
        config.faults = unbounded;
        CHECK(source.prepare(config) == ReplayConfigStatus::stall_bound_missing);

        const ReplayFaultSpec twice[] = {
            {.target = ledger_target(1, ReplayItemKind::frame),
             .effect = ReplayFaultEffect::stall,
             .stall_ns = 1},
            {.target = ledger_target(1, ReplayItemKind::frame),
             .effect = ReplayFaultEffect::read_failure},
        };
        config.faults = twice;
        CHECK(source.prepare(config) == ReplayConfigStatus::duplicate_fault_target);
    }

    {
        // Two faults on the same recorded target are a duplicate even when
        // that target is outside the current image/range: both resolve to
        // kReplayAbsentU64, but the identity is the recorded target, not the
        // resolved item index (8.11; matches Python ReplayConfig).
        auto builder = ledger_builder(false);
        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("dup-absent-ledger.nrimg"), source) ==
              ReplayImageStatus::ok);
        const ReplayFaultSpec twice_absent[] = {
            {.target = ledger_target(999, ReplayItemKind::frame),
             .effect = ReplayFaultEffect::stall,
             .stall_ns = 1},
            {.target = ledger_target(999, ReplayItemKind::frame),
             .effect = ReplayFaultEffect::read_failure},
        };
        ReplaySourceConfig config;
        config.clock = &clock;
        config.faults = twice_absent;
        CHECK(source.prepare(config) == ReplayConfigStatus::duplicate_fault_target);
    }

    {
        // A stream_frames fault is positioned by (stream, block ordinal) --
        // never by the run-local replay sequence (8.11).
        ReplayImageBuilder builder;
        builder.mode = "stream_frames";
        builder.flags = 0;
        for (std::uint64_t i = 0; i < 3; ++i)
        {
            auto frame = make_frame(neurale::recording::test::kAbsentU64, i, i, i * 1'000,
                                    {static_cast<unsigned char>(i)});
            frame.blocks[0].stream = "cursor";
            frame.blocks[0].source_block_ordinal = 10 + i;
            builder.add_frame(std::move(frame));
        }
        builder.fidelity.push_back({.stream = "cursor", .native_signal_id = 1});
        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("stream-fault.nrimg"), source) ==
              ReplayImageStatus::ok);
        ReplayFaultTarget target;
        target.ledger_based = false;
        target.kind = ReplayItemKind::frame;
        target.stream_id = "cursor";
        target.ordinal = 11;
        const ReplayFaultSpec faults[] = {
            {.target = target, .effect = ReplayFaultEffect::sequence_gap},
        };
        ReplaySourceConfig config;
        config.clock = &clock;
        config.faults = faults;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        CHECK(source.fault_report(0).item_idx == 1);

        Reader reader;
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(reader.frame.frame().view().header.sequence == 0);
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(reader.frame.frame().view().header.sequence == 2);
        CHECK(reader.next(source) == StreamStatus::end_of_stream);
    }

    {
        // stream_frames: two faults on the same recorded target outside the
        // image are a duplicate by identity, not by resolved index (8.11).
        ReplayImageBuilder builder;
        builder.mode = "stream_frames";
        builder.flags = 0;
        for (std::uint64_t i = 0; i < 3; ++i)
        {
            auto frame = make_frame(neurale::recording::test::kAbsentU64, i, i, i * 1'000,
                                    {static_cast<unsigned char>(i)});
            frame.blocks[0].stream = "cursor";
            frame.blocks[0].source_block_ordinal = 10 + i;
            builder.add_frame(std::move(frame));
        }
        builder.fidelity.push_back({.stream = "cursor", .native_signal_id = 1});
        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("dup-absent-stream.nrimg"), source) ==
              ReplayImageStatus::ok);
        ReplayFaultTarget target;
        target.ledger_based = false;
        target.kind = ReplayItemKind::frame;
        target.stream_id = "cursor";
        target.ordinal = 999;
        const ReplayFaultSpec twice_absent[] = {
            {.target = target, .effect = ReplayFaultEffect::stall, .stall_ns = 1},
            {.target = target, .effect = ReplayFaultEffect::read_failure},
        };
        ReplaySourceConfig config;
        config.clock = &clock;
        config.faults = twice_absent;
        CHECK(source.prepare(config) == ReplayConfigStatus::duplicate_fault_target);
    }

    {
        // A skipped item costs no permit -- advance(k) still yields k observed
        // items -- and leaves its recorded time in place (8.7, 8.8).
        ReplayImageBuilder builder;
        builder.add_frame(make_frame(0, 0, 0, 0, {1}));
        builder.add_frame(make_frame(1, 1, 1, 1'000, {2}));
        builder.add_frame(make_frame(2, 2, 2, 2'000, {3}));
        NativeReplaySource source;
        FakeClock clock;
        clock.set(10'000);
        CHECK(write_and_open(builder, scratch.path("skip-time.nrimg"), source) ==
              ReplayImageStatus::ok);
        const ReplayFaultSpec faults[] = {
            {.target = ledger_target(1, ReplayItemKind::frame),
             .effect = ReplayFaultEffect::sequence_gap},
        };
        ReplaySourceConfig config;
        config.clock = &clock;
        config.pacing = ReplayPacing::recorded;
        config.faults = faults;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        Reader reader;
        CHECK(reader.next(source) == StreamStatus::ok);
        CHECK(reader.next(source) == StreamStatus::ok);
        // The whole 2 microseconds of the original run, not just the surviving
        // interval: the skipped item left its time in place.
        CHECK(clock.now_ns() == 12'000);
        CHECK(reader.next(source) == StreamStatus::end_of_stream);
    }
    return 0;
}

int check_refusals(const Scratch& scratch)
{
    {
        // A frame larger than the pool slot is a capacity error the run cannot
        // paper over, and it is terminal rather than a retry loop.
        ReplayImageBuilder builder;
        BuiltFrame frame;
        frame.replay_sequence = 0;
        frame.payload.assign(64, std::byte{7});
        BuiltBlock block;
        block.payload_byte_count = 64;
        block.n_samples = 64;
        frame.blocks.push_back(block);
        builder.add_frame(std::move(frame));

        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("too-big.nrimg"), source) ==
              ReplayImageStatus::ok);
        ReplaySourceConfig config;
        config.clock = &clock;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);

        FramePool small{2, 16, 4};
        DiscontinuityPool discontinuities{2, 4};
        FrameLease lease;
        DiscontinuityLease gap_lease;
        CHECK(small.try_acquire(lease) == StreamStatus::ok);
        CHECK(discontinuities.try_acquire(gap_lease) == StreamStatus::ok);
        CHECK(source.read_message(lease.frame(), gap_lease) == StreamStatus::invalid_frame);
        CHECK(source.terminal() == ReplayTerminal::faulted);
    }

    {
        // A discontinuity with more gaps than the lease can hold is the same
        // kind of refusal on the discontinuity edge.
        ReplayImageBuilder builder;
        builder.add_frame(make_frame(0, 0, 0, 0, {1}));
        BuiltDiscontinuity record;
        record.replay_sequence = 1;
        record.previous_replay_sequence = 0;
        record.timeline_ns = 10;
        for (std::uint32_t i = 0; i < 6; ++i)
        {
            BuiltGap gap;
            gap.native_signal_id = i + 1;
            gap.gap_idx_in_message = i;
            record.gaps.push_back(gap);
        }
        builder.add_discontinuity(std::move(record));

        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("too-many-gaps.nrimg"), source) ==
              ReplayImageStatus::ok);
        ReplaySourceConfig config;
        config.clock = &clock;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);

        FramePool frames{2, 4096, 8};
        DiscontinuityPool narrow{2, 2};
        FrameLease lease;
        DiscontinuityLease gap_lease;
        CHECK(frames.try_acquire(lease) == StreamStatus::ok);
        CHECK(narrow.try_acquire(gap_lease) == StreamStatus::ok);
        CHECK(source.read_message(lease.frame(), gap_lease) == StreamStatus::ok);
        static_cast<void>(lease.reset());
        CHECK(frames.try_acquire(lease) == StreamStatus::ok);
        CHECK(source.read_message(lease.frame(), gap_lease) == StreamStatus::buffer_exhausted);
        CHECK(source.terminal() == ReplayTerminal::faulted);
    }

    {
        // The frame-only interface has nowhere to put a discontinuity, and
        // inferring one away is exactly what the contract forbids.
        auto builder = ledger_builder(true);
        NativeReplaySource source;
        FakeClock clock;
        CHECK(write_and_open(builder, scratch.path("frame-only.nrimg"), source) ==
              ReplayImageStatus::ok);
        ReplaySourceConfig config;
        config.clock = &clock;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        FramePool frames{2, 4096, 8};
        FrameLease lease;
        CHECK(frames.try_acquire(lease) == StreamStatus::ok);
        CHECK(source.read(lease.frame()) == StreamStatus::ok);
        static_cast<void>(lease.reset());
        CHECK(frames.try_acquire(lease) == StreamStatus::ok);
        CHECK(source.read(lease.frame()) == StreamStatus::ok);
        static_cast<void>(lease.reset());
        CHECK(frames.try_acquire(lease) == StreamStatus::ok);
        CHECK(source.read(lease.frame()) == StreamStatus::invalid_state);
        CHECK(source.terminal() == ReplayTerminal::faulted);
    }

    {
        // An unprepared source reads nothing, and prepare needs an image.
        NativeReplaySource source;
        FramePool frames{2, 64, 4};
        FrameLease lease;
        CHECK(frames.try_acquire(lease) == StreamStatus::ok);
        CHECK(source.read(lease.frame()) == StreamStatus::invalid_state);
        CHECK(source.reset() == StreamStatus::invalid_state);
        ReplaySourceConfig config;
        CHECK(source.prepare(config) == ReplayConfigStatus::image_not_open);
    }
    return 0;
}

int check_incomplete(const Scratch& scratch)
{
    // A truncated session replayed under allow-incomplete ends abnormally,
    // never as if the data ran out naturally (8.5).
    auto builder = ledger_builder(true);
    builder.flags |= (1U << 0U) | (1U << 1U); // allow_incomplete, abnormal_end_required
    NativeReplaySource source;
    FakeClock clock;
    CHECK(write_and_open(builder, scratch.path("incomplete.nrimg"), source) ==
          ReplayImageStatus::ok);
    ReplaySourceConfig config;
    config.clock = &clock;
    CHECK(source.prepare(config) == ReplayConfigStatus::ok);
    Reader reader;
    for (int i = 0; i < 4; ++i)
    {
        CHECK(reader.next(source) != StreamStatus::end_of_stream);
    }
    CHECK(reader.next(source) == StreamStatus::source_failure);
    CHECK(source.terminal() == ReplayTerminal::abnormal_end);
    CHECK(reader.next(source) == StreamStatus::source_failure);
    return 0;
}

/// The image's declared schema is checked *whole*.
///
/// A comparison that looks at a subset of the fields answers "compatible" for a
/// pair of schemas that disagree about sample layout, tick semantics, or the
/// identity of the channel set -- and every one of those changes what the
/// frames mean. So the image carries a signal whose every optional field is
/// deliberately non-default, and each variation below differs from it in
/// exactly one field.
int check_schema_is_whole(const Scratch& scratch)
{
    using neurale::streaming::DeviceTickTracking;
    using neurale::streaming::ObservationTiming;
    using neurale::streaming::PhysicalUnit;
    using neurale::streaming::SignalDType;
    using neurale::streaming::SignalKind;
    using neurale::streaming::SignalLayout;
    using neurale::streaming::SignalSchema;
    using neurale::streaming::StreamSchema;

    auto builder = ledger_builder(true);
    builder.signals.clear();
    neurale::recording::test::BuiltSignal declared;
    declared.clock_domain = 77;
    declared.layout = "CHANNEL_MAJOR";
    declared.device_tick_tracking = "SAMPLE_COUNTER";
    declared.physical_unit = "VOLTS";
    declared.channel_set_id = 9;
    declared.calibration_id = 8;
    declared.reference_id = 7;
    builder.signals.push_back(declared);

    NativeReplaySource source;
    FakeClock clock;
    CHECK(write_and_open(builder, scratch.path("schema.nrimg"), source) == ReplayImageStatus::ok);
    ReplaySourceConfig config;
    config.clock = &clock;
    CHECK(source.prepare(config) == ReplayConfigStatus::ok);

    // Exactly the image's signal.
    const auto matching = []
    {
        return SignalSchema{1,
                            SignalDType::float32,
                            1,
                            1,
                            1,
                            {1000, 1},
                            77,
                            SignalLayout::channel_major,
                            DeviceTickTracking::sample_counter,
                            PhysicalUnit::volts,
                            9,
                            8,
                            7};
    };

    const SignalSchema agreeing[] = {matching()};
    const StreamSchema good{1, agreeing};
    CHECK(source.check_schema(good) == ReplayConfigStatus::ok);

    // One field at a time. Every one of these answered `ok` when the check
    // only compared id, channel count, block limit, dtype, and rate.
    {
        auto signal = matching();
        signal.clock_domain = 0;
        const SignalSchema signals[] = {signal};
        const StreamSchema schema{1, signals};
        CHECK(source.check_schema(schema) == ReplayConfigStatus::schema_mismatch);
    }
    {
        auto signal = matching();
        signal.layout = SignalLayout::sample_major;
        const SignalSchema signals[] = {signal};
        const StreamSchema schema{1, signals};
        CHECK(source.check_schema(schema) == ReplayConfigStatus::schema_mismatch);
    }
    {
        auto signal = matching();
        signal.device_tick_tracking = DeviceTickTracking::unavailable;
        const SignalSchema signals[] = {signal};
        const StreamSchema schema{1, signals};
        CHECK(source.check_schema(schema) == ReplayConfigStatus::schema_mismatch);
    }
    {
        auto signal = matching();
        signal.physical_unit = PhysicalUnit::unspecified;
        const SignalSchema signals[] = {signal};
        const StreamSchema schema{1, signals};
        CHECK(source.check_schema(schema) == ReplayConfigStatus::schema_mismatch);
    }
    {
        auto signal = matching();
        signal.channel_set_id = 1;
        const SignalSchema signals[] = {signal};
        const StreamSchema schema{1, signals};
        CHECK(source.check_schema(schema) == ReplayConfigStatus::schema_mismatch);
    }
    {
        auto signal = matching();
        signal.calibration_id = 1;
        const SignalSchema signals[] = {signal};
        const StreamSchema schema{1, signals};
        CHECK(source.check_schema(schema) == ReplayConfigStatus::schema_mismatch);
    }
    {
        auto signal = matching();
        signal.reference_id = 1;
        const SignalSchema signals[] = {signal};
        const StreamSchema schema{1, signals};
        CHECK(source.check_schema(schema) == ReplayConfigStatus::schema_mismatch);
    }
    {
        auto signal = matching();
        signal.nominal_block_samples = 1;
        signal.max_block_samples = 2;
        signal.max_block_bytes = 8;
        const SignalSchema signals[] = {signal};
        const StreamSchema schema{1, signals};
        CHECK(source.check_schema(schema) == ReplayConfigStatus::schema_mismatch);
    }
    {
        // A different schema id, and a different signal count, are both still
        // caught.
        const SignalSchema signals[] = {matching()};
        const StreamSchema wrong_id{2, signals};
        CHECK(source.check_schema(wrong_id) == ReplayConfigStatus::schema_mismatch);
        const SignalSchema pair[] = {matching(),
                                     SignalSchema{2, SignalDType::float32, 1, 1, 1, {1000, 1}, 77}};
        const StreamSchema two{1, pair};
        CHECK(source.check_schema(two) == ReplayConfigStatus::schema_mismatch);
    }

    {
        // Feature-set descriptors and units are part of the schema, and the
        // image carries both. They were not decoded at all before this round,
        // so a runtime built for a different feature set went unnoticed.
        using neurale::streaming::FeatureSetDescriptor;
        using neurale::streaming::UnitDescriptor;

        auto featured = ledger_builder(true);
        featured.signals.clear();
        neurale::recording::test::BuiltSignal feature_signal;
        feature_signal.kind = "FEATURE";
        feature_signal.observation_timing = "REGULAR";
        feature_signal.feature_set_id = 1;
        featured.signals.push_back(feature_signal);
        featured.feature_sets.push_back({});
        featured.units.push_back({});

        NativeReplaySource featured_source;
        CHECK(write_and_open(featured, scratch.path("featured.nrimg"), featured_source) ==
              ReplayImageStatus::ok);

        const auto signal = SignalSchema{1,
                                         SignalDType::float32,
                                         1,
                                         1,
                                         1,
                                         {1000, 1},
                                         0,
                                         SignalLayout::sample_major,
                                         DeviceTickTracking::unavailable,
                                         PhysicalUnit::unspecified,
                                         0,
                                         0,
                                         0,
                                         SignalKind::feature,
                                         1,
                                         ObservationTiming::regular};
        const auto descriptor = []
        {
            FeatureSetDescriptor value;
            value.id = 1;
            value.feature_names = {"alpha"};
            value.unit_ids = {1};
            value.source_stream_id = 1;
            value.source_stream = "stream";
            value.algorithm_name = "bandpower";
            value.algorithm_version = "1";
            value.window_length_ns = 2'000'000;
            value.shift_ns = 1'000'000;
            return value;
        };
        const auto unit = []
        {
            UnitDescriptor value;
            value.id = 1;
            value.symbol = "uV";
            value.description = "microvolts";
            return value;
        };

        const SignalSchema signals[] = {signal};
        {
            const FeatureSetDescriptor descriptors[] = {descriptor()};
            const UnitDescriptor units[] = {unit()};
            const StreamSchema schema{1, signals, descriptors, units};
            CHECK(featured_source.check_schema(schema) == ReplayConfigStatus::ok);
        }
        {
            auto other = descriptor();
            other.algorithm_version = "2";
            const FeatureSetDescriptor descriptors[] = {other};
            const UnitDescriptor units[] = {unit()};
            const StreamSchema schema{1, signals, descriptors, units};
            CHECK(featured_source.check_schema(schema) == ReplayConfigStatus::schema_mismatch);
        }
        {
            auto other = unit();
            other.description = "millivolts";
            const FeatureSetDescriptor descriptors[] = {descriptor()};
            const UnitDescriptor units[] = {other};
            const StreamSchema schema{1, signals, descriptors, units};
            CHECK(featured_source.check_schema(schema) == ReplayConfigStatus::schema_mismatch);
        }
        {
            // A runtime that carries an extra unit is not the one this image
            // was built for. (A runtime with *no* feature metadata is not a
            // case: StreamSchema refuses to construct a feature signal whose
            // descriptor is missing, so it cannot reach this check.)
            const FeatureSetDescriptor descriptors[] = {descriptor()};
            auto second = unit();
            second.id = 2;
            second.symbol = "mV";
            const UnitDescriptor units[] = {unit(), second};
            const StreamSchema schema{1, signals, descriptors, units};
            CHECK(featured_source.check_schema(schema) == ReplayConfigStatus::schema_mismatch);
        }
    }

    {
        // The other direction: the image declares metadata the runtime does
        // not. Iterating the runtime's list would compare nothing at all, so
        // the counts are checked before any element is.
        auto extra = ledger_builder(true);
        extra.units.push_back({});
        NativeReplaySource extra_source;
        CHECK(write_and_open(extra, scratch.path("extra-unit.nrimg"), extra_source) ==
              ReplayImageStatus::ok);
        const neurale::streaming::SignalSchema plain[] = {
            neurale::streaming::SignalSchema{1, SignalDType::float32, 1, 1, 1, {1000, 1}, 0}};
        const StreamSchema bare{1, plain};
        CHECK(extra_source.check_schema(bare) == ReplayConfigStatus::schema_mismatch);
    }

    {
        // A value this build cannot name is not "different" -- it makes the two
        // schemas incomparable, and saying so is not the same answer.
        auto unreadable = ledger_builder(true);
        unreadable.signals.clear();
        neurale::recording::test::BuiltSignal alien;
        alien.dtype = "FLOAT128";
        unreadable.signals.push_back(alien);
        NativeReplaySource other;
        CHECK(write_and_open(unreadable, scratch.path("alien-schema.nrimg"), other) ==
              ReplayImageStatus::ok);
        const SignalSchema signals[] = {matching()};
        const StreamSchema schema{1, signals};
        CHECK(other.check_schema(schema) == ReplayConfigStatus::unsupported_schema_value);
    }
    return 0;
}

/// A replay run's identity is the run's, and never the recording's (8.12).
int check_session_identity(const Scratch& scratch)
{
    FakeClock clock;
    ReplaySourceConfig config;
    config.clock = &clock;

    // What the allocator would hand out next. Nothing else in this suite runs
    // concurrently, so the following identity is exactly one higher.
    auto probe_builder = ledger_builder(true);
    NativeReplaySource probe;
    CHECK(write_and_open(probe_builder, scratch.path("identity-probe.nrimg"), probe) ==
          ReplayImageStatus::ok);
    CHECK(probe.prepare(config) == ReplayConfigStatus::ok);
    const auto next = probe.session_id() + 1;
    CHECK(probe.session_id() != 0);

    // An image whose recording used precisely that identity. The default
    // allocation must step over it rather than hand back a run that is
    // indistinguishable from the original recording on the wire.
    auto builder = ledger_builder(true);
    builder.native_session_id = next;
    NativeReplaySource source;
    CHECK(write_and_open(builder, scratch.path("identity.nrimg"), source) == ReplayImageStatus::ok);
    CHECK(source.prepare(config) == ReplayConfigStatus::ok);
    CHECK(source.session_id() != 0);
    CHECK(source.session_id() != next);

    // Asking for it explicitly is refused, not quietly substituted: a caller
    // told "ok" would go on believing the frames carry the id it asked for.
    ReplaySourceConfig conflicting = config;
    conflicting.replay_run_session_id = next;
    CHECK(source.prepare(conflicting) == ReplayConfigStatus::replay_session_id_conflict);
    // ... and the refusal changed nothing.
    CHECK(source.session_id() != next);

    ReplaySourceConfig explicit_ok = config;
    explicit_ok.replay_run_session_id = next + 1000;
    CHECK(source.prepare(explicit_ok) == ReplayConfigStatus::ok);
    CHECK(source.session_id() == next + 1000);
    return 0;
}

/// A prepare that refuses changes nothing at all.
int check_prepare_is_transactional(const Scratch& scratch)
{
    auto builder = ledger_builder(true);
    NativeReplaySource source;
    FakeClock clock;
    CHECK(write_and_open(builder, scratch.path("transactional.nrimg"), source) ==
          ReplayImageStatus::ok);

    ReplayFaultSpec stall;
    stall.target.kind = ReplayItemKind::frame;
    stall.target.data_message_ordinal = 1;
    stall.effect = ReplayFaultEffect::stall;
    stall.stall_ns = 500;
    const ReplayFaultSpec faults[] = {stall};

    ReplaySourceConfig config;
    config.clock = &clock;
    config.pacing = ReplayPacing::recorded;
    config.speed_factor = 2.0;
    config.faults = faults;
    CHECK(source.prepare(config) == ReplayConfigStatus::ok);
    const auto identity = source.session_id();

    Reader reader;
    CHECK(reader.next(source) == StreamStatus::ok);
    CHECK(reader.frame.frame().view().header.sequence == 41U);

    // A refusal that happens after the speed factor was read but before
    // anything was committed.
    ReplaySourceConfig invalid = config;
    invalid.speed_factor = 0.0;
    CHECK(source.prepare(invalid) == ReplayConfigStatus::invalid_speed_factor);

    // A refusal that happens *after* part of the new fault vector was already
    // built: two of these three resolve, and the third collides. That is where
    // a non-transactional prepare leaves half of one configuration and half of
    // another -- a run with faults nobody configured.
    ReplayFaultSpec other = stall;
    other.target.data_message_ordinal = 0;
    other.effect = ReplayFaultEffect::read_failure;
    other.stall_ns = 0;
    ReplayFaultSpec last = stall;
    last.target.data_message_ordinal = 3;
    const ReplayFaultSpec colliding[] = {other, last, last};
    ReplaySourceConfig conflicting = config;
    conflicting.faults = colliding;
    CHECK(source.prepare(conflicting) == ReplayConfigStatus::duplicate_fault_target);

    // Nothing moved: same identity, same fault vector, same position in the
    // run. In particular the run was *not* restarted, because a prepare that
    // refused never got as far as resetting anything.
    CHECK(source.session_id() == identity);
    CHECK(source.fault_count() == 1);
    CHECK(source.fault_report(0).effect == ReplayFaultEffect::stall);
    CHECK(!source.fault_report(0).fired);
    CHECK(reader.next(source) == StreamStatus::ok);
    CHECK(reader.frame.frame().view().header.sequence == 42U);
    CHECK(source.stats().injected_delay_ns == 500);
    CHECK(source.fault_report(0).fired);
    return 0;
}

/// One gate, and no window in which two operations are both inside it.
int check_operation_gate(const Scratch& scratch)
{
    auto builder = ledger_builder(true);
    NativeReplaySource source;
    FakeClock clock;
    CHECK(write_and_open(builder, scratch.path("gate.nrimg"), source) == ReplayImageStatus::ok);
    ReplaySourceConfig config;
    config.clock = &clock;
    config.pacing = ReplayPacing::step;
    CHECK(source.prepare(config) == ReplayConfigStatus::ok);

    // Park a read inside the gate: step pacing with no permit and a frozen
    // clock blocks until somebody grants one.
    clock.freeze(true);
    Reader reader;
    std::atomic<bool> inside{false};
    std::thread worker{[&]
                       {
                           inside.store(true, std::memory_order_release);
                           static_cast<void>(reader.next(source));
                       }};
    while (!inside.load(std::memory_order_acquire) || clock.waits() == 0)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }

    // Both control-plane entry points are refused while the read holds the
    // gate, and neither of them touched anything on the way out.
    CHECK(source.reset() == StreamStatus::invalid_state);
    CHECK(source.prepare(config) == ReplayConfigStatus::operation_in_flight);
    // Status is still readable from another thread while the read is inside.
    CHECK(source.stats().items_emitted == 0);
    CHECK(source.fault_count() == 0);

    source.advance(1);
    clock.freeze(false);
    worker.join();
    CHECK(reader.status == StreamStatus::ok);
    CHECK(source.reset() == StreamStatus::ok);
    return 0;
}

/// Two threads competing for the run, plus a third reading its status.
///
/// This case asserts only that every call returns a legal answer, because that
/// is all a single run of it can prove. Its real job is to be the vehicle a
/// thread sanitizer build drives: a read racing a reset, and a status snapshot
/// racing both, are exactly the shapes that were undefined behaviour before the
/// gate and the atomic counters.
int check_concurrent_stress(const Scratch& scratch)
{
    auto builder = ledger_builder(true);
    NativeReplaySource source;
    CHECK(write_and_open(builder, scratch.path("stress.nrimg"), source) == ReplayImageStatus::ok);
    // One fault, so the reading thread really does write a fault report while
    // the watcher below is asking for one.
    ReplayFaultSpec skip;
    skip.target.kind = ReplayItemKind::frame;
    skip.target.data_message_ordinal = 1;
    skip.effect = ReplayFaultEffect::sequence_gap;
    const ReplayFaultSpec faults[] = {skip};
    ReplaySourceConfig config;
    config.faults = faults;
    CHECK(source.prepare(config) == ReplayConfigStatus::ok);

    constexpr int kIterations = 2000;
    std::atomic<bool> stop{false};
    std::atomic<int> illegal{0};

    std::thread readers{[&]
                        {
                            Reader local;
                            for (int i = 0; i < kIterations; ++i)
                            {
                                const auto status = local.next(source);
                                const bool legal = status == StreamStatus::ok ||
                                                   status == StreamStatus::discontinuity ||
                                                   status == StreamStatus::end_of_stream ||
                                                   status == StreamStatus::invalid_state;
                                if (!legal)
                                {
                                    illegal.fetch_add(1, std::memory_order_acq_rel);
                                }
                            }
                            stop.store(true, std::memory_order_release);
                        }};
    std::thread resetter{
        [&]
        {
            while (!stop.load(std::memory_order_acquire))
            {
                const auto status = source.reset();
                if (status != StreamStatus::ok && status != StreamStatus::invalid_state)
                {
                    illegal.fetch_add(1, std::memory_order_acq_rel);
                }
            }
        }};
    std::thread watcher{[&]
                        {
                            while (!stop.load(std::memory_order_acquire))
                            {
                                // Each counter on its own is a value the run really stored, so it
                                // can never exceed what the image holds. A *relation between* two
                                // of them is not asserted, and deliberately so: a snapshot is not
                                // a transaction, and a reset that lands between two of these loads
                                // is allowed to make the pair disagree.
                                const auto snapshot = source.stats();
                                if (snapshot.items_emitted > 3 || snapshot.frames_emitted > 2 ||
                                    snapshot.discontinuities_emitted > 1)
                                {
                                    illegal.fetch_add(1, std::memory_order_acq_rel);
                                }
                                static_cast<void>(source.fault_report(0));
                                static_cast<void>(source.terminal());
                                static_cast<void>(source.session_id());
                            }
                        }};
    readers.join();
    resetter.join();
    watcher.join();
    CHECK(illegal.load(std::memory_order_acquire) == 0);
    return 0;
}

int run_contract()
{
    const Scratch scratch{"neurale-replay-source-"};
    if (const int status = check_image_semantics(scratch); status != 0)
    {
        return status;
    }
    if (const int status = check_pacing(scratch); status != 0)
    {
        return status;
    }
    if (const int status = check_step(scratch); status != 0)
    {
        return status;
    }
    if (const int status = check_cancel_and_reset(scratch); status != 0)
    {
        return status;
    }
    if (const int status = check_faults(scratch); status != 0)
    {
        return status;
    }
    if (const int status = check_refusals(scratch); status != 0)
    {
        return status;
    }
    if (const int status = check_incomplete(scratch); status != 0)
    {
        return status;
    }
    if (const int status = check_schema_is_whole(scratch); status != 0)
    {
        return status;
    }
    if (const int status = check_session_identity(scratch); status != 0)
    {
        return status;
    }
    if (const int status = check_prepare_is_transactional(scratch); status != 0)
    {
        return status;
    }
    if (const int status = check_operation_gate(scratch); status != 0)
    {
        return status;
    }
    if (const int status = check_concurrent_stress(scratch); status != 0)
    {
        return status;
    }
    return 0;
}

// --- the allocation gate -----------------------------------------------------

int run_allocation()
{
    const Scratch scratch{"neurale-replay-alloc-"};
    ReplayImageBuilder builder;
    for (std::uint64_t i = 0; i < 64; ++i)
    {
        builder.add_frame(make_frame(i * 2, i, i, i * 1'000, {1, 2, 3, 4, 5, 6, 7, 8}));
        BuiltDiscontinuity record;
        record.data_message_ordinal = i * 2 + 1;
        record.replay_sequence = i;
        record.previous_replay_sequence = i;
        record.timeline_ns = i * 1'000 + 500;
        BuiltGap gap;
        gap.missing_samples = 4;
        record.gaps.push_back(gap);
        builder.add_discontinuity(std::move(record));
    }

    NativeReplaySource source;
    FakeClock clock;
    if (write_and_open(builder, scratch.path("allocation.nrimg"), source) != ReplayImageStatus::ok)
    {
        std::cerr << "could not build the allocation image\n";
        return 2;
    }
    ReplaySourceConfig config;
    config.clock = &clock;
    if (source.prepare(config) != ReplayConfigStatus::ok)
    {
        std::cerr << "could not prepare the allocation run\n";
        return 2;
    }

    Reader reader;
    // Warm up outside the measurement: the first reads touch pages and the
    // pools, and this gate is about the steady state, not the first touch.
    for (int i = 0; i < 8; ++i)
    {
        static_cast<void>(reader.next(source));
    }

    neurale::benchmark::reset_allocation_count();
    neurale::benchmark::set_allocation_tracking(true);
    std::uint64_t observed = 0;
    for (;;)
    {
        const auto status = reader.next(source);
        if (status == StreamStatus::end_of_stream)
        {
            break;
        }
        ++observed;
    }
    neurale::benchmark::set_allocation_tracking(false);
    const auto allocations = neurale::benchmark::allocation_count();

    std::cout << "{\"case\":\"replay_source_steady_state\",\"items\":" << observed
              << ",\"allocations\":" << allocations << ",\"backend\":\""
              << neurale::benchmark::allocation_tracking_backend() << "\"}\n";
    if (allocations != 0)
    {
        std::cerr << "the steady-state replay read allocated " << allocations << " times\n";
        return 2;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::string mode;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument == "--contract-only" || argument == "--allocation-only")
        {
            if (!mode.empty())
            {
                std::cerr << "give exactly one of --contract-only and --allocation-only\n";
                return 2;
            }
            mode = argument;
            continue;
        }
        std::cerr << "unknown replay source test option " << argument << '\n';
        return 2;
    }
    if (mode.empty())
    {
        std::cerr << "give exactly one of --contract-only and --allocation-only\n";
        return 2;
    }
    if (mode == "--allocation-only")
    {
        return run_allocation();
    }
    const int status = run_contract();
    if (status == 0)
    {
        std::cout << "replay source ok\n";
        return 0;
    }
    return 1;
}
