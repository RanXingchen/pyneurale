/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// Behaviour of the native bounded recorder core, and the gate that proves its
/// critical callback allocates nothing.
///
/// Two CTest entries share this executable:
///
///   --contract-only    ordering, lifecycle, drain, accounting, close
///   --allocation-only  the callback's steady-state allocation gate
///
/// The gate is here rather than in a benchmark for the same reason the spool
/// writer's is: "the critical callback does not allocate" is a contract, and a
/// contract belongs in a test that fails the build.

#include "recorder_test_support.h"

#include "allocation_tracker.h"
#include "check_returns.h"
#include "record_payloads.h"
#include "recorder.h"
#include "spool_scanner.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace
{

using namespace neurale::recording;
using namespace neurale::recording::test;
namespace streaming = neurale::streaming;

/// One committed record, read back out of the spool.
struct ReadRecord
{
    std::uint16_t kind{};
    std::uint64_t logical_ordinal{};
    std::vector<std::byte> payload{};
};

/// Read the whole committed prefix. Used by every ordering assertion, because
/// what a reader promotes is the only definition of "the recorder recorded it"
/// that matters.
[[nodiscard]] std::vector<ReadRecord> read_committed(SpoolFile& file, SpoolScanReport& report)
{
    std::array<std::byte, 4096> scratch{};
    SpoolScanner scanner;
    report = scanner.scan(file, scratch);

    std::vector<ReadRecord> out;
    SpoolRecordCursor cursor(file, report);
    SpoolRecordView view{};
    while (cursor.next(view))
    {
        ReadRecord record{
            .kind = view.kind, .logical_ordinal = view.logical_ordinal, .payload = {}};
        record.payload.resize(view.payload_bytes);
        if (!cursor.read_payload(view, record.payload))
        {
            break;
        }
        out.push_back(std::move(record));
    }
    return out;
}

/// Drive a recorder from `created` to `recording` with the fixture's plan.
struct RunningRecorder
{
    PlanFixture fixture{};
    RecorderMemorySpoolFile spool{};
    CountingUnixClock clock{};
    NativeRecorderCore recorder{};

    [[nodiscard]] RecorderStatusCode start()
    {
        const auto plan = fixture.plan();
        auto status = recorder.prepare(plan, spool, clock);
        if (status != RecorderStatusCode::ok)
        {
            return status;
        }
        status = recorder.pass_readiness_gate(true);
        if (status != RecorderStatusCode::ok)
        {
            return status;
        }
        return recorder.start();
    }
};

/// Wait until the worker has committed at least *expected_items*, under a
/// generous bound so a loaded machine cannot make this flake.
[[nodiscard]] bool wait_for_drain(const NativeRecorderCore& recorder, std::uint64_t expected_items)
{
    for (int attempt = 0; attempt < 100000; ++attempt)
    {
        if (recorder.status().spool_committed >= expected_items)
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return false;
}

/// Wait until the data queue has room. This is the *test's* pacing, not the
/// recorder's: a real producer cannot block, which is exactly why the queue
/// depth has to be planned for the source rate. Here it lets a test push more
/// items than the queue holds without turning the run into a saturation test.
[[nodiscard]] bool wait_for_queue_space(const NativeRecorderCore& recorder)
{
    for (int attempt = 0; attempt < 100000; ++attempt)
    {
        const auto status = recorder.status();
        if (status.data_queue_pending + 1 < status.data_queue_capacity)
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return false;
}

// ---------------------------------------------------------------------------
// plan validation
// ---------------------------------------------------------------------------

int test_plan_is_validated_before_allocation()
{
    PlanFixture fixture;
    CHECK(fixture.plan().validate() == RecordingPlanStatus::ok);

    auto empty_identity = fixture.plan();
    empty_identity.session_id = {};
    CHECK(empty_identity.validate() == RecordingPlanStatus::invalid_identity);

    auto zero_queue = fixture.plan();
    zero_queue.frame_queue_capacity = 0;
    CHECK(zero_queue.validate() == RecordingPlanStatus::invalid_bound);

    // A signal table that is not ascending cannot be searched, and searching
    // it is what the critical callback does per block.
    std::vector<PlannedSignalRecording> unsorted{
        PlannedSignalRecording{.signal_id = 5, .max_block_bytes = 64, .max_block_samples = 8},
        PlannedSignalRecording{.signal_id = 2, .max_block_bytes = 64, .max_block_samples = 8},
    };
    auto bad_table = fixture.plan();
    bad_table.recorded_signals = unsorted;
    CHECK(bad_table.validate() == RecordingPlanStatus::invalid_signal_table);

    // One frame must fit one transaction, or the recorder could never make
    // progress and would discover it only at runtime.
    auto tiny_transaction = fixture.plan();
    tiny_transaction.max_transaction_bytes = 256;
    CHECK(tiny_transaction.validate() == RecordingPlanStatus::inconsistent_bounds);

    RecorderMemorySpoolFile spool;
    CountingUnixClock clock;
    NativeRecorderCore recorder;
    CHECK(recorder.prepare(zero_queue, spool, clock) == RecorderStatusCode::invalid_plan);
    // Nothing was created on disk and nothing is retained.
    CHECK(spool.snapshot().empty());
    CHECK(recorder.state() == RecorderLifecycleState::created);
    return 0;
}

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

int test_lifecycle_follows_contract_state_machine()
{
    RunningRecorder run;
    const auto plan = run.fixture.plan();

    // stop() and abort() are a state error where there is nothing to stop.
    CHECK(run.recorder.stop("early") == RecorderStatusCode::wrong_state);
    CHECK(run.recorder.abort("early") == RecorderStatusCode::wrong_state);
    CHECK(run.recorder.start() == RecorderStatusCode::wrong_state);

    CHECK(run.recorder.prepare(plan, run.spool, run.clock) == RecorderStatusCode::ok);
    CHECK(run.recorder.state() == RecorderLifecycleState::prepared);
    // prepare() is not idempotent; a second call is a state error.
    CHECK(run.recorder.prepare(plan, run.spool, run.clock) == RecorderStatusCode::wrong_state);
    // start() is legal only in `ready`.
    CHECK(run.recorder.start() == RecorderStatusCode::wrong_state);

    CHECK(run.recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(run.recorder.state() == RecorderLifecycleState::ready);
    CHECK(run.recorder.stop("early") == RecorderStatusCode::wrong_state);

    CHECK(run.recorder.start() == RecorderStatusCode::ok);
    CHECK(run.recorder.state() == RecorderLifecycleState::recording);
    CHECK(run.recorder.start() == RecorderStatusCode::wrong_state);

    CHECK(run.recorder.stop("done") == RecorderStatusCode::ok);
    CHECK(run.recorder.state() == RecorderLifecycleState::stopped);

    // Finalization is a separate step, and the recorder says so rather than
    // reporting a session that does not exist.
    CHECK(run.recorder.finalize() == RecorderStatusCode::not_implemented);
    CHECK(run.recorder.abandon_finalization() == RecorderStatusCode::not_implemented);

    CHECK(run.recorder.close() == RecorderStatusCode::ok);
    CHECK(run.recorder.state() == RecorderLifecycleState::closed);
    return 0;
}

int test_failed_readiness_gate_releases_everything()
{
    PlanFixture fixture;
    RecorderMemorySpoolFile spool;
    CountingUnixClock clock;
    NativeRecorderCore recorder;

    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(false) == RecorderStatusCode::not_ready);
    CHECK(recorder.state() == RecorderLifecycleState::failed);

    const auto status = recorder.status();
    // Every prepared resource is gone: no worker, no queues.
    CHECK(!status.worker_running);
    CHECK(status.data_queue_capacity == 0);
    CHECK(status.control_queue_capacity == 0);
    // The superblock was written but no record was committed, so there is no
    // session artifact to retain -- which is the honest answer here.
    CHECK(!status.session_created);
    CHECK(!status.spool_holds_committed_record);
    CHECK(status.primary_fault.present);
    CHECK(status.primary_fault.reason == RecorderFaultReason::readiness_gate_failed);

    // close() is idempotent from a terminal state.
    CHECK(recorder.close() == RecorderStatusCode::ok);
    CHECK(recorder.close() == RecorderStatusCode::ok);
    CHECK(recorder.state() == RecorderLifecycleState::failed);
    return 0;
}

int test_uncommittable_superblock_fails_readiness_gate()
{
    PlanFixture fixture;
    RecorderMemorySpoolFile spool;
    // The superblock write is the first append, and it happens at the readiness
    // gate, not at prepare(). Failing it makes the gate unable to commit.
    spool.fail_append_call.store(1, std::memory_order_relaxed);
    spool.append_fault = SpoolIoStatus::io_error;
    CountingUnixClock clock;
    NativeRecorderCore recorder;

    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.state() == RecorderLifecycleState::prepared);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::not_ready);
    CHECK(recorder.state() == RecorderLifecycleState::failed);

    const auto status = recorder.status();
    CHECK(!status.worker_running);
    CHECK(status.data_queue_capacity == 0);
    CHECK(status.control_queue_capacity == 0);
    CHECK(status.primary_fault.present);
    CHECK(status.primary_fault.reason == RecorderFaultReason::spool_writer_failed);
    CHECK(recorder.close() == RecorderStatusCode::ok);
    return 0;
}

int test_close_is_idempotent_in_every_state()
{
    {
        NativeRecorderCore recorder;
        CHECK(recorder.close() == RecorderStatusCode::ok);
        CHECK(recorder.state() == RecorderLifecycleState::closed);
        CHECK(recorder.close() == RecorderStatusCode::ok);
        CHECK(!recorder.status().session_created);
    }
    {
        RunningRecorder run;
        CHECK(run.recorder.prepare(run.fixture.plan(), run.spool, run.clock) ==
              RecorderStatusCode::ok);
        CHECK(run.recorder.close() == RecorderStatusCode::ok);
        CHECK(run.recorder.state() == RecorderLifecycleState::closed);
        CHECK(!run.recorder.status().worker_running);
        // The writer was cancelled with nothing committed, so the spool holds
        // no record and there is no session artifact to retain. `prepare()`
        // commits nothing: the superblock is committed at the readiness gate,
        // which this run never passed, so the file is empty and the status
        // says plainly that there is no session -- which is what a retention
        // policy needs to know.
        const auto status = run.recorder.status();
        CHECK(!status.spool_holds_committed_record);
        CHECK(!status.session_created);
        CHECK(!status.finalization_required);
        CHECK(!status.recovery_required);
        CHECK(status.recoverable == RecoverabilityAnswer::not_applicable);
        CHECK(run.spool.snapshot().empty());
    }
    {
        // From `recording`, close() performs the graceful stop first.
        RunningRecorder run;
        CHECK(run.start() == RecorderStatusCode::ok);
        auto frame = make_frame(1, std::array<std::uint32_t, 1>{kSignalA});
        CHECK(run.recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);
        CHECK(run.recorder.close() == RecorderStatusCode::ok);
        CHECK(run.recorder.state() == RecorderLifecycleState::closed);
        CHECK(run.recorder.close() == RecorderStatusCode::ok);
        CHECK(run.recorder.close() == RecorderStatusCode::ok);

        SpoolScanReport report;
        const auto records = read_committed(run.spool, report);
        CHECK(report.session_end_present());
        CHECK(report.data_items() == 1);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// ordering
// ---------------------------------------------------------------------------

int test_frames_discontinuities_and_control_keep_their_order()
{
    RunningRecorder run;
    CHECK(run.start() == RecorderStatusCode::ok);

    // Interleave the two planes. Within a plane the order is the order they
    // were accepted in; between planes there is no promise and none is tested.
    const std::array<std::uint32_t, 2> both{kSignalA, kSignalB};
    for (std::uint64_t sequence = 0; sequence < 6; ++sequence)
    {
        if (sequence == 3)
        {
            auto discontinuity = make_discontinuity(2, 3, 2);
            CHECK(run.recorder.observe_discontinuity(discontinuity.view()) ==
                  streaming::StreamStatus::ok);
        }
        auto frame = make_frame(sequence, both);
        CHECK(run.recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);

        const std::array<std::byte, 3> body{std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
        CHECK(run.recorder.submit_control(ControlSubmission{
            .kind = ProducerIdentityKind::events,
            .identity = sequence,
            .clock_domain = 1,
            .time_ns = 500 + sequence,
            .body = body,
        }));
    }

    CHECK(run.recorder.stop("done") == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto records = read_committed(run.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(report.session_end_present());
    CHECK(report.data_items() == 7);
    CHECK(report.control_items() == 6);

    // The data-message ordinal is monotonic across frames *and*
    // discontinuities: that is precisely what a bare frame sequence could not
    // express, because the discontinuity before frame 3 and frame 3 itself
    // would both be "3".
    std::uint64_t expected_ordinal = 0;
    std::uint64_t expected_frame_ordinal = 0;
    std::uint64_t expected_block_ordinal = 0;
    std::uint64_t expected_gap_ordinal = 0;
    std::uint64_t expected_control_ordinal = 0;
    std::uint64_t owning_ordinal = 0;
    int frame_rows = 0;
    int discontinuity_rows = 0;
    int block_rows = 0;
    int gap_rows = 0;
    int control_rows = 0;

    for (const auto& record : records)
    {
        switch (static_cast<RecordKind>(record.kind))
        {
        case RecordKind::frame:
        {
            FramePayloadFields fields{};
            CHECK(decode_frame_payload(record.payload, fields));
            CHECK(fields.data_message_ordinal == expected_ordinal);
            CHECK(record.logical_ordinal == expected_ordinal);
            CHECK(fields.frame_ordinal == expected_frame_ordinal);
            CHECK(fields.native_session_id == kTestNativeSessionId);
            CHECK(fields.native_schema_id == kTestSchemaId);
            CHECK(fields.signal_block_count == 2);
            CHECK(fields.recorded_signal_block_count == 2);
            CHECK(fields.total_payload_byte_count == 32);
            owning_ordinal = expected_ordinal;
            ++expected_ordinal;
            ++expected_frame_ordinal;
            ++frame_rows;
            break;
        }
        case RecordKind::signal_block:
        {
            SignalBlockPayloadFields fields{};
            CHECK(decode_signal_block_payload(record.payload, fields));
            // A block carries its owning frame's ordinal, not its own: that is
            // what lets a reader attribute it without parsing the payload.
            CHECK(record.logical_ordinal == owning_ordinal);
            CHECK(fields.data_message_ordinal == owning_ordinal);
            CHECK(fields.signal_block_ordinal == expected_block_ordinal);
            CHECK(fields.block_idx_in_frame == expected_block_ordinal % 2);
            CHECK(fields.payload_byte_count == 16);
            CHECK(record.payload.size() == kSignalBlockHeaderPayloadBytes + 16);
            ++expected_block_ordinal;
            ++block_rows;
            break;
        }
        case RecordKind::discontinuity:
        {
            DiscontinuityPayloadFields fields{};
            CHECK(decode_discontinuity_payload(record.payload, fields));
            CHECK(fields.data_message_ordinal == expected_ordinal);
            CHECK(record.logical_ordinal == expected_ordinal);
            CHECK(fields.signal_gap_count == 2);
            CHECK(fields.previous_frame_sequence == 2);
            CHECK(fields.actual_frame_sequence == 3);
            owning_ordinal = expected_ordinal;
            ++expected_ordinal;
            ++discontinuity_rows;
            break;
        }
        case RecordKind::signal_gap:
        {
            SignalGapPayloadFields fields{};
            CHECK(decode_signal_gap_payload(record.payload, fields));
            CHECK(record.logical_ordinal == owning_ordinal);
            CHECK(fields.data_message_ordinal == owning_ordinal);
            CHECK(fields.signal_gap_ordinal == expected_gap_ordinal);
            ++expected_gap_ordinal;
            ++gap_rows;
            break;
        }
        case RecordKind::control:
        {
            ControlPayloadFields fields{};
            CHECK(decode_control_payload(record.payload, fields));
            CHECK(fields.submission_ordinal == expected_control_ordinal);
            CHECK(record.logical_ordinal == expected_control_ordinal);
            CHECK(fields.identity == expected_control_ordinal);
            CHECK(fields.control_kind == static_cast<std::uint32_t>(ProducerIdentityKind::events));
            CHECK(record.payload.size() == kControlHeaderPayloadBytes + 3);
            ++expected_control_ordinal;
            ++control_rows;
            break;
        }
        default:
            break;
        }
    }

    CHECK(frame_rows == 6);
    CHECK(discontinuity_rows == 1);
    CHECK(block_rows == 12);
    CHECK(gap_rows == 2);
    CHECK(control_rows == 6);
    return 0;
}

int test_frame_payload_survives_copy_byte_for_byte()
{
    RunningRecorder run;
    CHECK(run.start() == RecorderStatusCode::ok);

    const std::array<std::uint32_t, 2> both{kSignalA, kSignalB};
    auto frame = make_frame(11, both, 24);
    CHECK(run.recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);
    CHECK(run.recorder.stop("done") == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto records = read_committed(run.spool, report);
    std::size_t checked = 0;
    for (const auto& record : records)
    {
        if (static_cast<RecordKind>(record.kind) != RecordKind::signal_block)
        {
            continue;
        }
        SignalBlockPayloadFields fields{};
        CHECK(decode_signal_block_payload(record.payload, fields));
        const auto* recorded = record.payload.data() + kSignalBlockHeaderPayloadBytes;
        const auto* original = frame.payload.data() + fields.payload_offset;
        CHECK(std::memcmp(recorded, original,
                          static_cast<std::size_t>(fields.payload_byte_count)) == 0);
        ++checked;
    }
    CHECK(checked == 2);
    return 0;
}

int test_unplanned_signal_is_skipped_not_rejected()
{
    RunningRecorder run;
    CHECK(run.start() == RecorderStatusCode::ok);

    // Partial coverage is a decision the plan compiler already made. The
    // recorder skips the block and still accepts the frame.
    const std::array<std::uint32_t, 3> mixed{kSignalA, kUnplannedSignal, kSignalB};
    auto frame = make_frame(4, mixed);
    CHECK(run.recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);
    CHECK(run.recorder.stop("done") == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto records = read_committed(run.spool, report);
    int block_rows = 0;
    for (const auto& record : records)
    {
        if (static_cast<RecordKind>(record.kind) == RecordKind::frame)
        {
            FramePayloadFields fields{};
            CHECK(decode_frame_payload(record.payload, fields));
            // Both counts are recorded, so a reader can see that the session
            // deliberately holds fewer blocks than the frame carried.
            CHECK(fields.signal_block_count == 3);
            CHECK(fields.recorded_signal_block_count == 2);
        }
        if (static_cast<RecordKind>(record.kind) == RecordKind::signal_block)
        {
            SignalBlockPayloadFields fields{};
            CHECK(decode_signal_block_payload(record.payload, fields));
            CHECK(fields.native_signal_id != kUnplannedSignal);
            // The index is the block's position in the original frame, so the
            // skipped one leaves a hole rather than renumbering its neighbour.
            CHECK(fields.block_idx_in_frame == (block_rows == 0 ? 0U : 2U));
            ++block_rows;
        }
    }
    CHECK(block_rows == 2);
    CHECK(run.recorder.status().runtime_accepted == 1);
    CHECK(run.recorder.status().recorder_accepted == 1);
    return 0;
}

// ---------------------------------------------------------------------------
// the control plane's promises
// ---------------------------------------------------------------------------

int test_control_acceptance_is_exactly_control_accepted()
{
    RunningRecorder run;
    CHECK(run.start() == RecorderStatusCode::ok);

    const std::array<std::byte, 2> body{std::byte{1}, std::byte{2}};
    CHECK(run.recorder.submit_control(ControlSubmission{.kind = ProducerIdentityKind::trials,
                                                        .identity = 42,
                                                        .clock_domain = 0,
                                                        .time_ns = 7,
                                                        .body = body}));

    // True meant control accepted and nothing further: at this instant the
    // record may well not be in a committed transaction yet, and the status
    // reports the two stages separately rather than as one number.
    auto status = run.recorder.status();
    CHECK(status.control_accepted == 1);
    CHECK(status.control_offered == 1);
    CHECK(status.control_rejected == 0);

    // A data-plane registry value is not a control kind, and its position has
    // no expressible form on the control plane. It is refused and named by the
    // primary fault rather than by a counter that would have to store a kind
    // no finalizer could map -- and never silently discarded.
    CHECK(!run.recorder.submit_control(ControlSubmission{.kind = ProducerIdentityKind::frame,
                                                         .identity = 1,
                                                         .clock_domain = 0,
                                                         .time_ns = 8,
                                                         .body = {}}));
    status = run.recorder.status();
    CHECK(status.control_rejected == 0);
    CHECK(status.control_offered == 1);
    CHECK(!status.control_first_rejection.present());
    CHECK(status.primary_fault.present);
    CHECK(status.primary_fault.reason == RecorderFaultReason::plan_violation);

    // The refusal faulted the recorder, so the session ends faulted -- but the
    // recorder's own lifecycle still completes, and the accepted record is
    // still delivered. `faulted` belongs to the session, `failed` to the
    // recorder, and this is the case that separates them.
    CHECK(run.recorder.close() == RecorderStatusCode::ok);
    CHECK(run.recorder.state() == RecorderLifecycleState::closed);
    status = run.recorder.status();
    CHECK(status.effective_session_outcome == EffectiveSessionOutcome::faulted);
    CHECK(status.control_spool_committed == 1);

    // An oversized body *is* a control record the path refused, so it does
    // carry a position -- and its kind is one the registry has. A fresh
    // recorder, because the one above has already latched its fault.
    RunningRecorder oversized;
    CHECK(oversized.start() == RecorderStatusCode::ok);
    const std::vector<std::byte> huge(1024, std::byte{0});
    CHECK(!oversized.recorder.submit_control(ControlSubmission{.kind = ProducerIdentityKind::labels,
                                                               .identity = 5,
                                                               .clock_domain = 0,
                                                               .time_ns = 9,
                                                               .body = huge}));
    status = oversized.recorder.status();
    CHECK(status.control_rejected == 1);
    CHECK(status.control_offered == 1);
    CHECK(status.control_accepted == 0);
    CHECK(status.control_first_rejection.present());
    CHECK(status.control_first_rejection.tag == PositionTag::producer_identity);
    CHECK(status.control_first_rejection.identity_kind == ProducerIdentityKind::labels);
    CHECK(status.control_first_rejection.identity_value == 5);
    CHECK(oversized.recorder.close() == RecorderStatusCode::ok);
    return 0;
}

int test_records_after_close_are_caller_misuse()
{
    RunningRecorder run;
    CHECK(run.start() == RecorderStatusCode::ok);

    auto frame = make_frame(1, std::array<std::uint32_t, 1>{kSignalA});
    CHECK(run.recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);
    CHECK(run.recorder.stop("done") == RecorderStatusCode::ok);

    auto late = make_frame(2, std::array<std::uint32_t, 1>{kSignalA});
    CHECK(run.recorder.observe_frame(late.view()) == streaming::StreamStatus::stopped);
    CHECK(!run.recorder.submit_control(ControlSubmission{.kind = ProducerIdentityKind::events,
                                                         .identity = 9,
                                                         .clock_domain = 0,
                                                         .time_ns = 9,
                                                         .body = {}}));

    const auto status = run.recorder.status();
    CHECK(status.rejected_after_close_data == 1);
    CHECK(status.rejected_after_close_control == 1);
    // Not a recording loss: it belongs to no identity and did not move the
    // acceptance counters at all.
    CHECK(status.runtime_accepted == 1);
    CHECK(status.control_offered == 0);
    CHECK(status.failed_between_runtime_and_recorder == 0);
    CHECK(!status.data_first_loss.present());
    CHECK(status.effective_session_outcome == EffectiveSessionOutcome::normal);
    CHECK(status.capture_outcome == CaptureOutcome::normal);
    CHECK(status.capture_outcome.has_value());
    return 0;
}

// ---------------------------------------------------------------------------
// drain and the sealed accounting
// ---------------------------------------------------------------------------

int test_graceful_stop_delivers_accepted_items()
{
    PlanFixture fixture;
    fixture.checkpoint_interval = 4;
    RecorderMemorySpoolFile spool;
    CountingUnixClock clock;
    NativeRecorderCore recorder;

    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);

    constexpr std::uint64_t kFrames = 200;
    const std::array<std::uint32_t, 2> both{kSignalA, kSignalB};
    for (std::uint64_t sequence = 0; sequence < kFrames; ++sequence)
    {
        CHECK(wait_for_queue_space(recorder));
        auto frame = make_frame(sequence, both);
        CHECK(recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);
    }

    CHECK(recorder.stop("done") == RecorderStatusCode::ok);
    CHECK(recorder.state() == RecorderLifecycleState::stopped);

    const auto status = recorder.status();
    CHECK(status.runtime_accepted == kFrames);
    CHECK(status.recorder_accepted == kFrames);
    CHECK(status.spool_committed == kFrames);
    CHECK(status.failed_between_runtime_and_recorder == 0);
    CHECK(status.lost_between_recorder_and_spool == 0);
    // Every producer-accepted item reached stage 3, and none of the later
    // stages is claimed: no finalizer has run.
    CHECK(status.nrf_committed == 0);
    CHECK(status.finalization_status == FinalizationStatus::not_started);
    CHECK(status.finalization_required);
    CHECK(!status.sealed);
    // Absent, not false: completeness is a property of a sealed session, and
    // `false` would claim this one was judged incomplete rather than not judged.
    CHECK(!status.complete.has_value());
    CHECK(!status.completeness_verdict.has_value());
    CHECK(!status.accounting_verified);
    CHECK(!status.termination_kind.has_value());
    CHECK(status.finalization_attempts.empty());
    CHECK(status.spool_ended_cleanly);
    CHECK(!status.recovery_required);
    CHECK(status.recoverable == RecoverabilityAnswer::not_applicable);
    CHECK(status.capture_outcome == CaptureOutcome::normal);
    CHECK(status.capture_outcome.has_value());
    CHECK(status.effective_session_outcome == EffectiveSessionOutcome::normal);
    CHECK(status.requested_terminal_intent == RequestedTerminalIntent::normal);
    CHECK(status.data_queue_high_water_mark <= status.data_queue_capacity);

    SpoolScanReport report;
    const auto records = read_committed(spool, report);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(report.finalizable());
    CHECK(report.data_items() == kFrames);
    CHECK(report.session_end_present());
    CHECK(report.capture_outcome() == CaptureOutcome::normal);
    CHECK(report.terminal_reason() == "done");
    // A checkpoint every four transactions, and at least one of them landed.
    bool saw_checkpoint = false;
    bool saw_accounting = false;
    for (const auto& record : records)
    {
        saw_checkpoint |= static_cast<RecordKind>(record.kind) == RecordKind::checkpoint;
        saw_accounting |= static_cast<RecordKind>(record.kind) == RecordKind::accounting_snapshot;
    }
    CHECK(saw_checkpoint);
    CHECK(saw_accounting);
    return 0;
}

int test_requested_checkpoint_is_written_without_interval()
{
    // `checkpoint_interval` stays 0, so the only checkpoint that can appear is
    // the requested one. With an interval configured the assertion would not
    // distinguish the request from the schedule.
    RunningRecorder run;
    CHECK(run.fixture.checkpoint_interval == 0);
    CHECK(run.start() == RecorderStatusCode::ok);

    const std::array<std::uint32_t, 2> both{kSignalA, kSignalB};
    auto frame = make_frame(0, both);
    CHECK(run.recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);
    CHECK(run.recorder.request_checkpoint());

    CHECK(run.recorder.stop("done") == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto records = read_committed(run.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    bool saw_checkpoint = false;
    for (const auto& record : records)
    {
        saw_checkpoint |= static_cast<RecordKind>(record.kind) == RecordKind::checkpoint;
    }
    CHECK(saw_checkpoint);

    // And a request is refused once the session is no longer accepting, rather
    // than being taken and quietly never honoured.
    CHECK(!run.recorder.request_checkpoint());
    return 0;
}

int test_abort_still_drains_accepted_items()
{
    RunningRecorder run;
    CHECK(run.start() == RecorderStatusCode::ok);

    const std::array<std::uint32_t, 1> one{kSignalA};
    for (std::uint64_t sequence = 0; sequence < 5; ++sequence)
    {
        auto frame = make_frame(sequence, one);
        CHECK(run.recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);
    }
    CHECK(run.recorder.abort("operator stopped the run") == RecorderStatusCode::ok);

    const auto status = run.recorder.status();
    // An abort is a decision to stop recording new data, not a licence to
    // destroy data already handed over.
    CHECK(status.recorder_accepted == 5);
    CHECK(status.spool_committed == 5);
    CHECK(status.lost_between_recorder_and_spool == 0);
    CHECK(status.requested_terminal_intent == RequestedTerminalIntent::aborted);
    CHECK(status.capture_outcome == CaptureOutcome::aborted);
    CHECK(status.effective_session_outcome == EffectiveSessionOutcome::aborted);
    // No fault record is invented for an abort.
    CHECK(!status.primary_fault.present);

    SpoolScanReport report;
    const auto records = read_committed(run.spool, report);
    CHECK(report.session_end_present());
    CHECK(report.capture_outcome() == CaptureOutcome::aborted);
    CHECK(report.requested_terminal_intent() == RequestedTerminalIntent::aborted);
    CHECK(report.terminal_reason() == "operator stopped the run");
    for (const auto& record : records)
    {
        CHECK(static_cast<RecordKind>(record.kind) != RecordKind::fault);
    }
    return 0;
}

int test_first_shutdown_latches_intent()
{
    RunningRecorder run;
    CHECK(run.start() == RecorderStatusCode::ok);

    auto frame = make_frame(1, std::array<std::uint32_t, 1>{kSignalA});
    CHECK(run.recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);

    CHECK(run.recorder.abort("first") == RecorderStatusCode::ok);
    // A later stop() waits for the shutdown already run and returns its
    // status; it must not rewrite what the first call decided.
    CHECK(run.recorder.stop("second") == RecorderStatusCode::ok);
    CHECK(run.recorder.abort("third") == RecorderStatusCode::ok);

    const auto status = run.recorder.status();
    CHECK(status.requested_terminal_intent == RequestedTerminalIntent::aborted);
    CHECK(status.capture_outcome == CaptureOutcome::aborted);

    SpoolScanReport report;
    static_cast<void>(read_committed(run.spool, report));
    CHECK(report.terminal_reason() == "first");
    return 0;
}

// ---------------------------------------------------------------------------
// the allocation gate
// ---------------------------------------------------------------------------

/// Run a steady-state session and report what it allocated.
///
/// The tracker is process-wide, so this measures the callback **and** the
/// worker: everything either of them does between `prepare()` and the drain.
/// That is the stronger claim and the one the contract asks for -- "no
/// allocation on the data path after prepare" is not a statement about one
/// thread. The spool sink is preallocated for the same reason: a test double
/// that grew a vector per append would be measuring the fixture.
int run_allocation_gate()
{
    PlanFixture fixture;
    fixture.frame_queue_capacity = 64;
    fixture.control_queue_capacity = 64;
    fixture.checkpoint_interval = 8;
    RecorderMemorySpoolFile spool;
    spool.reserve_bytes(8u << 20u);
    CountingUnixClock clock;
    NativeRecorderCore recorder;

    if (recorder.prepare(fixture.plan(), spool, clock) != RecorderStatusCode::ok ||
        recorder.pass_readiness_gate(true) != RecorderStatusCode::ok ||
        recorder.start() != RecorderStatusCode::ok)
    {
        std::fprintf(stderr, "allocation gate could not start the recorder\n");
        return 2;
    }

    // Build every input up front: the fixtures allocate, and what is being
    // measured is the recorder, not the test's own scaffolding.
    const std::array<std::uint32_t, 2> both{kSignalA, kSignalB};
    std::vector<FrameFixture> frames;
    frames.reserve(256);
    for (std::uint64_t sequence = 0; sequence < 256; ++sequence)
    {
        frames.push_back(make_frame(sequence, both));
    }
    auto discontinuity = make_discontinuity(9, 10, 4);
    const std::array<std::byte, 16> control_body{};

    // Warm up outside the measured window, so a first-touch page fault or a
    // lazily materialised worker buffer is not counted as steady state.
    constexpr std::size_t kWarmup = 16;
    for (std::size_t i = 0; i < kWarmup; ++i)
    {
        if (!wait_for_queue_space(recorder))
        {
            std::fprintf(stderr, "allocation gate: warm-up stalled\n");
            return 2;
        }
        static_cast<void>(recorder.observe_frame(frames[i].view()));
    }
    if (!wait_for_drain(recorder, kWarmup))
    {
        std::fprintf(stderr, "allocation gate: warm-up did not drain\n");
        return 2;
    }

    neurale::benchmark::reset_allocation_count();
    neurale::benchmark::set_allocation_tracking(true);
    for (std::size_t i = kWarmup; i < frames.size(); ++i)
    {
        if (!wait_for_queue_space(recorder))
        {
            neurale::benchmark::set_allocation_tracking(false);
            std::fprintf(stderr, "allocation gate: steady state stalled\n");
            return 2;
        }
        static_cast<void>(recorder.observe_frame(frames[i].view()));
        if (i % 32 == 0)
        {
            static_cast<void>(recorder.observe_discontinuity(discontinuity.view()));
        }
        static_cast<void>(
            recorder.submit_control(ControlSubmission{.kind = ProducerIdentityKind::events,
                                                      .identity = i,
                                                      .clock_domain = 1,
                                                      .time_ns = 1000 + i,
                                                      .body = control_body}));
    }
    if (!wait_for_drain(recorder, frames.size()))
    {
        neurale::benchmark::set_allocation_tracking(false);
        const auto stuck = recorder.status();
        std::fprintf(stderr,
                     "allocation gate: steady state did not drain "
                     "(accepted=%llu committed=%llu pending=%llu fault=%d state=%d)\n",
                     static_cast<unsigned long long>(stuck.recorder_accepted),
                     static_cast<unsigned long long>(stuck.spool_committed),
                     static_cast<unsigned long long>(stuck.data_queue_pending),
                     static_cast<int>(stuck.primary_fault.reason), static_cast<int>(stuck.state));
        return 2;
    }
    neurale::benchmark::set_allocation_tracking(false);
    const auto allocations = neurale::benchmark::allocation_count();

    const auto status = recorder.status();
    std::printf("recorder_steady_state backend=%.*s allocations=%llu frames=%llu "
                "discontinuities=%llu control=%llu committed=%llu\n",
                static_cast<int>(neurale::benchmark::allocation_tracking_backend().size()),
                neurale::benchmark::allocation_tracking_backend().data(),
                static_cast<unsigned long long>(allocations),
                static_cast<unsigned long long>(status.frames_accepted),
                static_cast<unsigned long long>(status.discontinuities_accepted),
                static_cast<unsigned long long>(status.control_accepted),
                static_cast<unsigned long long>(status.spool_committed));

    static_cast<void>(recorder.close());
    if (allocations != 0)
    {
        std::fprintf(stderr, "the recorder allocated %llu times in steady state\n",
                     static_cast<unsigned long long>(allocations));
        return 2;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    const std::string_view mode = argc > 1 ? argv[1] : "";
    if (mode != "--contract-only" && mode != "--allocation-only")
    {
        std::fprintf(stderr, "usage: %s --contract-only | --allocation-only\n", argv[0]);
        return 2;
    }
    if (mode == "--allocation-only")
    {
        return run_allocation_gate();
    }

    for (const auto test : {
             test_plan_is_validated_before_allocation,
             test_lifecycle_follows_contract_state_machine,
             test_failed_readiness_gate_releases_everything,
             test_uncommittable_superblock_fails_readiness_gate,
             test_close_is_idempotent_in_every_state,
             test_frames_discontinuities_and_control_keep_their_order,
             test_frame_payload_survives_copy_byte_for_byte,
             test_unplanned_signal_is_skipped_not_rejected,
             test_control_acceptance_is_exactly_control_accepted,
             test_records_after_close_are_caller_misuse,
             test_graceful_stop_delivers_accepted_items,
             test_requested_checkpoint_is_written_without_interval,
             test_abort_still_drains_accepted_items,
             test_first_shutdown_latches_intent,
         })
    {
        if (const auto line = test(); line != 0)
        {
            return line;
        }
    }
    return 0;
}
