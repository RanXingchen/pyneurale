/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// Every way the native recorder core is supposed to fail.
///
/// The point of each case is the same: a critical recorder is
/// **lossless-until-fault**, so the interesting question is never "did it drop
/// the item quietly" -- there is no policy under which it does -- but "did it
/// latch the right primary fault, count the loss at the handoff where it
/// happened, and still deliver everything it had already taken over".

#include "recorder_test_support.h"
#include "spool_test_support.h"

#include "check_returns.h"
#include "record_payloads.h"
#include "recorder.h"
#include "spool_scanner.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
// clang-format off
#include <windows.h>
#include <tlhelp32.h>
// clang-format on
#else
#include <dirent.h>
#endif

namespace
{

using namespace neurale::recording;
using namespace neurale::recording::test;
namespace streaming = neurale::streaming;

/// A worker that never runs, so the data queue fills and stays full. Parking
/// the worker rather than slowing the file is what makes saturation
/// deterministic: nothing here depends on the two threads racing.
class BlockedSpoolFile final : public SpoolFile
{
  public:
    std::atomic<bool> blocked{false};
    std::atomic<bool> entered{false};
    bool cancellable{true};

    SpoolIoResult append(std::span<const std::byte> bytes) noexcept override
    {
        entered.store(true, std::memory_order_release);
        while (blocked.load(std::memory_order_acquire))
        {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        const std::lock_guard<std::mutex> guard(mutex_);
        data_.insert(data_.end(), bytes.begin(), bytes.end());
        return SpoolIoResult{
            .status = SpoolIoStatus::ok, .transferred = bytes.size(), .platform_error = 0};
    }

    SpoolIoResult read_at(std::uint64_t offset, std::span<std::byte> out) noexcept override
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        if (offset >= data_.size())
        {
            return SpoolIoResult{
                .status = SpoolIoStatus::incomplete, .transferred = 0, .platform_error = 0};
        }
        const auto take =
            std::min<std::size_t>(out.size(), data_.size() - static_cast<std::size_t>(offset));
        std::copy_n(data_.begin() + static_cast<std::ptrdiff_t>(offset), take, out.begin());
        return SpoolIoResult{.status =
                                 take == out.size() ? SpoolIoStatus::ok : SpoolIoStatus::incomplete,
                             .transferred = take,
                             .platform_error = 0};
    }

    SpoolIoResult sync() noexcept override
    {
        return SpoolIoResult{};
    }

    SpoolIoResult truncate(std::uint64_t) noexcept override
    {
        return SpoolIoResult{};
    }

    [[nodiscard]] std::uint64_t size() const noexcept override
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        return data_.size();
    }

    // A cooperative backend: request_cancel unblocks the append the worker is
    // stuck in, so a bounded shutdown can join the worker and reclaim its
    // resources instead of leaving it (and its queues) behind. It reports the
    // capability the readiness gate requires of a critical recorder's backend.
    void request_cancel() noexcept override
    {
        if (cancellable)
            blocked.store(false, std::memory_order_release);
    }

    [[nodiscard]] bool supports_bounded_cancel() const noexcept override
    {
        return cancellable;
    }

  private:
    mutable std::mutex mutex_{};
    std::vector<std::byte> data_{};
};

/// An in-memory backend that reports `supports_bounded_cancel() == false`: it
/// stands in for a plain blocking `pwrite`/`fsync` backend whose syscalls
/// cannot be interrupted from another thread. Only the worker accesses its
/// bytes while recording; the caller reads them after stopping.
class NonCancellableSpoolFile final : public SpoolFile
{
  public:
    SpoolIoResult append(std::span<const std::byte> bytes) noexcept override
    {
        data_.insert(data_.end(), bytes.begin(), bytes.end());
        return SpoolIoResult{
            .status = SpoolIoStatus::ok, .transferred = bytes.size(), .platform_error = 0};
    }
    SpoolIoResult read_at(std::uint64_t, std::span<std::byte>) noexcept override
    {
        return SpoolIoResult{
            .status = SpoolIoStatus::incomplete, .transferred = 0, .platform_error = 0};
    }
    SpoolIoResult sync() noexcept override
    {
        return SpoolIoResult{};
    }
    SpoolIoResult truncate(std::uint64_t bytes) noexcept override
    {
        if (bytes < data_.size())
        {
            data_.resize(static_cast<std::size_t>(bytes));
        }
        return SpoolIoResult{};
    }
    [[nodiscard]] std::uint64_t size() const noexcept override
    {
        return data_.size();
    }
    [[nodiscard]] bool supports_bounded_cancel() const noexcept override
    {
        return false;
    }

  private:
    std::vector<std::byte> data_{};
};

[[nodiscard]] std::size_t open_descriptor_count()
{
#ifdef _WIN32
    DWORD count = 0;
    return ::GetProcessHandleCount(::GetCurrentProcess(), &count) != 0
               ? static_cast<std::size_t>(count)
               : 0;
#else
    std::size_t count = 0;
    DIR* directory = ::opendir("/proc/self/fd");
    if (directory == nullptr)
    {
        return 0;
    }
    while (::readdir(directory) != nullptr)
    {
        ++count;
    }
    ::closedir(directory);
    return count;
#endif
}

/// Live threads in this process. A shutdown that claims to have released its
/// worker but left a thread running somewhere else has not released anything,
/// and a count is the one check that does not depend on where it was left.
[[nodiscard]] std::size_t live_thread_count()
{
#ifdef _WIN32
    const auto process_id = ::GetCurrentProcessId();
    const auto snapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE)
    {
        return 0;
    }
    std::size_t count = 0;
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    if (::Thread32First(snapshot, &entry) != 0)
    {
        do
        {
            count += entry.th32OwnerProcessID == process_id ? 1U : 0U;
        } while (::Thread32Next(snapshot, &entry) != 0);
    }
    ::CloseHandle(snapshot);
    return count;
#else
    std::size_t count = 0;
    DIR* directory = ::opendir("/proc/self/task");
    if (directory == nullptr)
    {
        return 0;
    }
    while (::readdir(directory) != nullptr)
    {
        ++count;
    }
    ::closedir(directory);
    return count;
#endif
}

[[nodiscard]] std::vector<std::uint16_t> committed_kinds(SpoolFile& file, SpoolScanReport& report)
{
    std::array<std::byte, 4096> scratch{};
    SpoolScanner scanner;
    report = scanner.scan(file, scratch);

    std::vector<std::uint16_t> kinds;
    SpoolRecordCursor cursor(file, report);
    SpoolRecordView view{};
    while (cursor.next(view))
    {
        kinds.push_back(view.kind);
    }
    return kinds;
}

// ---------------------------------------------------------------------------
// queue saturation
// ---------------------------------------------------------------------------

int test_full_data_queue_faults_without_dropping()
{
    PlanFixture fixture;
    fixture.frame_queue_capacity = 4;
    BlockedSpoolFile spool;
    CountingUnixClock clock;
    NativeRecorderCore recorder;

    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);
    spool.blocked.store(true, std::memory_order_release);

    // Every input is built before the measured loop: the fixtures allocate,
    // and what is being timed is the callback.
    const std::array<std::uint32_t, 1> one{kSignalA};
    std::vector<FrameFixture> frames;
    frames.reserve(64);
    for (std::uint64_t sequence = 0; sequence < 64; ++sequence)
    {
        frames.push_back(make_frame(sequence, one));
    }

    std::uint64_t accepted = 0;
    streaming::StreamStatus last = streaming::StreamStatus::ok;
    const auto callbacks_started = std::chrono::steady_clock::now();
    for (auto& frame : frames)
    {
        last = recorder.observe_frame(frame.view());
        if (last != streaming::StreamStatus::ok)
        {
            break;
        }
        ++accepted;
    }
    const auto callbacks_elapsed = std::chrono::steady_clock::now() - callbacks_started;

    // The worker is parked inside a store call that will not return until this
    // test lets it, and the callbacks all completed anyway. That is the
    // "no blocking wait" constraint stated as an observation rather than as a
    // comment: a callback that waited on the store, on the worker, or on a
    // lock the worker holds could not have finished here at all.
    CHECK(callbacks_elapsed < std::chrono::seconds(2));

    // Saturation is reported, not absorbed. There is no drop policy to select.
    CHECK(last == streaming::StreamStatus::queue_overflow);
    CHECK(accepted >= fixture.frame_queue_capacity);

    auto status = recorder.status();
    CHECK(status.primary_fault.present);
    CHECK(status.primary_fault.origin == FaultOrigin::recorder);
    CHECK(status.primary_fault.reason == RecorderFaultReason::data_queue_saturated);
    // The refused item was runtime accepted and never recorder accepted, so it
    // is counted at that handoff and nowhere else.
    CHECK(status.runtime_accepted == accepted + 1);
    CHECK(status.recorder_accepted == accepted);
    CHECK(status.failed_between_runtime_and_recorder == 1);
    CHECK(status.rejected_before_runtime_acceptance == 0);
    CHECK(status.data_first_loss.present());
    CHECK(status.data_first_loss.tag == PositionTag::ordinal);
    CHECK(status.data_first_loss.ordinal == accepted);
    CHECK(status.effective_session_outcome == EffectiveSessionOutcome::faulted);

    // The recorder has stopped accepting, so a later frame is late rather than
    // a second loss.
    auto late = make_frame(99, one);
    CHECK(recorder.observe_frame(late.view()) == streaming::StreamStatus::stopped);
    CHECK(recorder.status().rejected_after_close_data == 1);
    CHECK(recorder.status().failed_between_runtime_and_recorder == 1);

    // A fault does not skip the drain: what was already accepted is still the
    // recorder's responsibility and still gets written.
    spool.blocked.store(false, std::memory_order_release);
    CHECK(recorder.stop("saturated") == RecorderStatusCode::ok);
    status = recorder.status();
    CHECK(status.spool_committed == accepted);
    CHECK(status.lost_between_recorder_and_spool == 0);
    CHECK(status.capture_outcome == CaptureOutcome::faulted);
    CHECK(status.capture_outcome.has_value());
    // The recorder's own lifecycle completed: `faulted` is the session's word.
    CHECK(recorder.state() == RecorderLifecycleState::stopped);

    SpoolScanReport report;
    const auto kinds = committed_kinds(spool, report);
    CHECK(report.session_end_present());
    CHECK(report.capture_outcome() == CaptureOutcome::faulted);
    CHECK(report.primary_fault_committed());
    bool saw_fault_row = false;
    for (const auto kind : kinds)
    {
        saw_fault_row |= kind == static_cast<std::uint16_t>(RecordKind::fault);
    }
    CHECK(saw_fault_row);
    CHECK(recorder.close() == RecorderStatusCode::ok);
    return 0;
}

int test_full_control_queue_uses_reserved_fault_path()
{
    PlanFixture fixture;
    fixture.control_queue_capacity = 2;
    BlockedSpoolFile spool;
    CountingUnixClock clock;
    NativeRecorderCore recorder;

    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);
    spool.blocked.store(true, std::memory_order_release);

    std::uint64_t accepted = 0;
    bool refused = false;
    for (std::uint64_t i = 0; i < 16; ++i)
    {
        const bool taken = recorder.submit_control(ControlSubmission{
            .kind = ProducerIdentityKind::events,
            .identity = i,
            .clock_domain = 1,
            .time_ns = i,
            .body = {},
        });
        if (!taken)
        {
            refused = true;
            break;
        }
        ++accepted;
    }
    CHECK(refused);

    const auto status = recorder.status();
    // On the control plane stages 1 and 2 coincide, so a refusal means the
    // record was never accepted -- and an item never accepted has no ordinal,
    // which is why its position is the (kind, identity) pair.
    CHECK(status.control_offered == accepted + 1);
    CHECK(status.control_accepted == accepted);
    CHECK(status.control_rejected == 1);
    CHECK(status.control_first_rejection.present());
    CHECK(status.control_first_rejection.tag == PositionTag::producer_identity);
    CHECK(status.control_first_rejection.identity_kind == ProducerIdentityKind::events);
    CHECK(status.control_first_rejection.identity_value == accepted);
    CHECK(status.primary_fault.present);
    CHECK(status.primary_fault.reason == RecorderFaultReason::control_queue_saturated);

    // The fault row does not need free space in the control queue -- which is
    // the whole point, because that queue is exactly the thing that is full.
    spool.blocked.store(false, std::memory_order_release);
    CHECK(recorder.stop("control saturated") == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto kinds = committed_kinds(spool, report);
    bool saw_fault_row = false;
    for (const auto kind : kinds)
    {
        saw_fault_row |= kind == static_cast<std::uint16_t>(RecordKind::fault);
    }
    CHECK(saw_fault_row);
    CHECK(report.primary_fault_committed());
    CHECK(recorder.status().control_spool_committed == accepted);
    CHECK(recorder.close() == RecorderStatusCode::ok);
    return 0;
}

// ---------------------------------------------------------------------------
// plan violations
// ---------------------------------------------------------------------------

int test_plan_violation_is_recorder_fault()
{
    struct Case
    {
        std::string_view name;
        void (*mutate)(FrameFixture&);
    };
    const Case cases[] = {
        {"wrong session", [](FrameFixture& frame) { frame.header.session_id = 999; }},
        {"block count disagrees", [](FrameFixture& frame) { frame.header.signal_block_count = 5; }},
        {"block extent outside the payload",
         [](FrameFixture& frame) { frame.blocks[0].payload_byte_count = 4096; }},
        {"block larger than its plan allows",
         [](FrameFixture& frame)
         {
             frame.payload.resize(900);
             frame.blocks[0].payload_offset = 0;
             frame.blocks[0].payload_byte_count = 900;
         }},
        {"more samples than its plan allows",
         [](FrameFixture& frame) { frame.blocks[0].n_samples = 4096; }},
        {"schema id the plan did not compile against",
         [](FrameFixture& frame) { frame.header.schema_id = 999; }},
    };

    for (const auto& scenario : cases)
    {
        PlanFixture fixture;
        RecorderMemorySpoolFile spool;
        CountingUnixClock clock;
        NativeRecorderCore recorder;
        CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
        CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
        CHECK(recorder.start() == RecorderStatusCode::ok);

        auto frame = make_frame(1, std::array<std::uint32_t, 1>{kSignalA});
        scenario.mutate(frame);
        if (recorder.observe_frame(frame.view()) != streaming::StreamStatus::invalid_frame)
        {
            std::fprintf(stderr, "case not refused: %.*s\n", static_cast<int>(scenario.name.size()),
                         scenario.name.data());
            return __LINE__;
        }
        const auto status = recorder.status();
        CHECK(status.primary_fault.reason == RecorderFaultReason::plan_violation);
        CHECK(status.runtime_accepted == 1);
        CHECK(status.recorder_accepted == 0);
        CHECK(status.failed_between_runtime_and_recorder == 1);
        CHECK(status.data_first_loss.present());
        CHECK(status.data_first_loss.ordinal == 0);
        CHECK(recorder.close() == RecorderStatusCode::ok);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// writer failure
// ---------------------------------------------------------------------------

int test_writer_failure_is_capture_fault_keeping_prefix()
{
    PlanFixture fixture;
    fixture.frame_queue_capacity = 32;
    RecorderMemorySpoolFile spool;
    CountingUnixClock clock;
    NativeRecorderCore recorder;

    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);

    const std::array<std::uint32_t, 1> one{kSignalA};
    auto first = make_frame(0, one);
    CHECK(recorder.observe_frame(first.view()) == streaming::StreamStatus::ok);
    for (int attempt = 0; attempt < 100000 && recorder.status().spool_committed == 0; ++attempt)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    CHECK(recorder.status().spool_committed == 1);

    // The very next append fails. Everything staged for it is lost; everything
    // committed before it stays exactly where it was.
    spool.fail_append_call.store(spool.append_calls() + 1, std::memory_order_release);
    auto second = make_frame(1, one);
    CHECK(recorder.observe_frame(second.view()) == streaming::StreamStatus::ok);

    for (int attempt = 0; attempt < 100000; ++attempt)
    {
        if (recorder.status().primary_fault.present)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    auto status = recorder.status();
    CHECK(status.primary_fault.present);
    CHECK(status.primary_fault.reason == RecorderFaultReason::spool_writer_failed);
    CHECK(status.effective_session_outcome == EffectiveSessionOutcome::faulted);
    CHECK(status.data_first_loss.present());
    CHECK(status.data_first_loss.ordinal == 1);

    // A spool writer failure is a capture fault, so the drain cannot seal and
    // the recorder's own lifecycle could not complete: `failed`, with the
    // committed prefix retained for the offline finalizer.
    CHECK(recorder.stop("writer failed") == RecorderStatusCode::writer_failed);
    CHECK(recorder.state() == RecorderLifecycleState::failed);

    status = recorder.status();
    CHECK(status.spool_committed == 1);
    CHECK(status.lost_between_recorder_and_spool == 1);
    CHECK(!status.spool_ended_cleanly);
    // No session-end record froze the capture outcome, so it is not known --
    // and nothing may guess one on its behalf.
    CHECK(!status.capture_outcome.has_value());
    CHECK(status.recovery_required);
    CHECK(status.recoverable == RecoverabilityAnswer::recoverable);
    CHECK(status.finalization_required);

    SpoolScanReport report;
    static_cast<void>(committed_kinds(spool, report));
    // The committed prefix survives and must not be discarded.
    CHECK(report.readable());
    CHECK(report.committed_transactions() >= 1);
    CHECK(!report.session_end_present());
    CHECK(report.data_items() == 1);

    CHECK(recorder.close() == RecorderStatusCode::ok);
    CHECK(recorder.state() == RecorderLifecycleState::failed);
    return 0;
}

// ---------------------------------------------------------------------------
// runtime faults and edge rejections
// ---------------------------------------------------------------------------

int test_primary_fault_is_not_displaced()
{
    PlanFixture fixture;
    RecorderMemorySpoolFile spool;
    CountingUnixClock clock;
    NativeRecorderCore recorder;

    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);

    auto frame = make_frame(0, std::array<std::uint32_t, 1>{kSignalA});
    CHECK(recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);

    streaming::FaultRecord fault{};
    fault.code = streaming::FaultCode::source_read;
    fault.status = streaming::StreamStatus::source_failure;
    fault.stage = streaming::FaultStage::source;
    fault.session_id = kTestNativeSessionId;
    fault.frame_sequence = 7;
    fault.detected_at_ns = 4242;
    fault.component_id = 3;
    recorder.publish_runtime_fault(fault);

    // A fault raised while handling a fault is secondary and is dropped.
    recorder.raise_recorder_fault(RecorderFaultReason::plan_violation, 9999);

    const auto status = recorder.status();
    CHECK(status.primary_fault.present);
    CHECK(status.primary_fault.origin == FaultOrigin::runtime);
    CHECK(status.primary_fault.reason == RecorderFaultReason::none);
    CHECK(status.primary_fault.detected_at_ns == 4242);
    CHECK(status.primary_fault.frame_sequence == 7);
    CHECK(status.effective_session_outcome == EffectiveSessionOutcome::faulted);
    CHECK(status.requested_terminal_intent == RequestedTerminalIntent::fault);

    // Step 4 of the runtime fault path: the items already accepted by the
    // recorder edge are drained, because they are often the only evidence
    // about the fault there is.
    CHECK(recorder.stop("runtime fault") == RecorderStatusCode::ok);
    CHECK(recorder.status().spool_committed == 1);

    SpoolScanReport report;
    static_cast<void>(committed_kinds(spool, report));
    CHECK(report.session_end_present());
    CHECK(report.capture_outcome() == CaptureOutcome::faulted);
    CHECK(report.requested_terminal_intent() == RequestedTerminalIntent::fault);

    // The fault row is on disk, and the finalizer can rebuild it exactly.
    SpoolRecordCursor cursor(spool, report);
    SpoolRecordView view{};
    bool checked = false;
    while (cursor.next(view))
    {
        if (view.kind != static_cast<std::uint16_t>(RecordKind::fault))
        {
            continue;
        }
        // A fault has no owning item, so its logical ordinal is zero.
        CHECK(view.logical_ordinal == 0);
        std::vector<std::byte> payload(view.payload_bytes);
        CHECK(cursor.read_payload(view, payload));
        FaultPayloadFields fields{};
        CHECK(decode_fault_payload(payload, fields));
        CHECK(fields.origin == FaultOrigin::runtime);
        CHECK(fields.frame_sequence == 7);
        CHECK(fields.detected_at_ns == 4242);
        CHECK(fields.fault_code == static_cast<std::uint8_t>(streaming::FaultCode::source_read));
        CHECK(fields.component_id == 3);
        checked = true;
    }
    CHECK(checked);
    CHECK(recorder.close() == RecorderStatusCode::ok);
    return 0;
}

int test_fault_without_frame_context_keeps_session_id()
{
    // A source that fails before it ever filled in a frame header gives the
    // runtime no session context, and that absence reaches the recorder as
    // `session_id == 0`. Every frame, discontinuity, and fault in a recording
    // must carry one native session identity, so the unknown value must not
    // displace the identity the plan already froze -- otherwise finalization
    // rejects the session it just recorded.
    PlanFixture fixture;
    RecorderMemorySpoolFile spool;
    CountingUnixClock clock;
    NativeRecorderCore recorder;

    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);

    streaming::FaultRecord fault{};
    fault.code = streaming::FaultCode::source_read;
    fault.status = streaming::StreamStatus::source_failure;
    fault.stage = streaming::FaultStage::source;
    fault.session_id = 0;
    fault.frame_sequence = 0;
    fault.detected_at_ns = 1234;
    recorder.publish_runtime_fault(fault);

    CHECK(recorder.stop("runtime fault without context") == RecorderStatusCode::ok);
    SpoolScanReport report;
    static_cast<void>(committed_kinds(spool, report));
    SpoolRecordCursor cursor(spool, report);
    SpoolRecordView view{};
    bool checked = false;
    while (cursor.next(view))
    {
        if (view.kind != static_cast<std::uint16_t>(RecordKind::fault))
        {
            continue;
        }
        std::vector<std::byte> payload(view.payload_bytes);
        CHECK(cursor.read_payload(view, payload));
        FaultPayloadFields fields{};
        CHECK(decode_fault_payload(payload, fields));
        CHECK(fields.origin == FaultOrigin::runtime);
        CHECK(fields.native_session_id == kTestNativeSessionId);
        checked = true;
    }
    CHECK(checked);
    CHECK(recorder.close() == RecorderStatusCode::ok);
    return 0;
}

int test_foreign_session_fault_is_recorded_verbatim()
{
    // The zero guard above is about missing context, not about repairing a
    // wrong identity. A non-zero session ID that disagrees with the plan is
    // real cross-session corruption and has to reach the finalizer intact.
    PlanFixture fixture;
    RecorderMemorySpoolFile spool;
    CountingUnixClock clock;
    NativeRecorderCore recorder;

    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);

    streaming::FaultRecord fault{};
    fault.code = streaming::FaultCode::source_read;
    fault.status = streaming::StreamStatus::source_failure;
    fault.stage = streaming::FaultStage::source;
    fault.session_id = kTestNativeSessionId + 1;
    fault.detected_at_ns = 5678;
    recorder.publish_runtime_fault(fault);

    CHECK(recorder.stop("runtime fault from another session") == RecorderStatusCode::ok);
    SpoolScanReport report;
    static_cast<void>(committed_kinds(spool, report));
    SpoolRecordCursor cursor(spool, report);
    SpoolRecordView view{};
    bool checked = false;
    while (cursor.next(view))
    {
        if (view.kind != static_cast<std::uint16_t>(RecordKind::fault))
        {
            continue;
        }
        std::vector<std::byte> payload(view.payload_bytes);
        CHECK(cursor.read_payload(view, payload));
        FaultPayloadFields fields{};
        CHECK(decode_fault_payload(payload, fields));
        CHECK(fields.native_session_id == kTestNativeSessionId + 1);
        checked = true;
    }
    CHECK(checked);
    CHECK(recorder.close() == RecorderStatusCode::ok);
    return 0;
}

int test_edge_rejection_is_counted_before_acceptance()
{
    PlanFixture fixture;
    RecorderMemorySpoolFile spool;
    CountingUnixClock clock;
    NativeRecorderCore recorder;

    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);

    auto frame = make_frame(0, std::array<std::uint32_t, 1>{kSignalA});
    CHECK(recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);
    recorder.note_edge_rejection(ProducerIdentityKind::frame, 41);

    const auto status = recorder.status();
    // The item never became runtime accepted for that edge, so it never had an
    // ordinal and its position is the (message kind, frame sequence) pair.
    CHECK(status.rejected_before_runtime_acceptance == 1);
    CHECK(status.runtime_accepted == 1);
    CHECK(status.failed_between_runtime_and_recorder == 0);
    CHECK(status.data_first_rejection.present());
    CHECK(status.data_first_rejection.tag == PositionTag::producer_identity);
    CHECK(status.data_first_rejection.identity_kind == ProducerIdentityKind::frame);
    CHECK(status.data_first_rejection.identity_value == 41);
    CHECK(status.data_first_rejection.ordinal == 0);
    CHECK(!status.data_first_loss.present());
    CHECK(status.primary_fault.reason == RecorderFaultReason::edge_rejection);

    CHECK(recorder.stop("edge rejected") == RecorderStatusCode::ok);
    SpoolScanReport report;
    static_cast<void>(committed_kinds(spool, report));
    CHECK(report.session_end_present());
    CHECK(report.finalizable());
    CHECK(recorder.close() == RecorderStatusCode::ok);
    return 0;
}

int test_fault_ends_recording_without_stop()
{
    // Contract section 4.2: a recorder/runtime fault ends recording. The fault
    // path itself initiates the asynchronous shutdown (recording -> draining),
    // so a faulted recorder does not keep claiming to record and wait for
    // someone to remember to call stop() -- the runtime sees a faulted
    // recorder, not one that pretends it is still recording.
    PlanFixture fixture;
    RecorderMemorySpoolFile spool;
    CountingUnixClock clock;
    NativeRecorderCore recorder;
    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);

    auto frame = make_frame(0, std::array<std::uint32_t, 1>{kSignalA});
    CHECK(recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);

    streaming::FaultRecord fault{};
    fault.code = streaming::FaultCode::source_read;
    fault.status = streaming::StreamStatus::source_failure;
    fault.stage = streaming::FaultStage::source;
    fault.session_id = kTestNativeSessionId;
    fault.frame_sequence = 7;
    fault.detected_at_ns = 4242;
    recorder.publish_runtime_fault(fault);

    // The fault left recording; it did not stay in `recording`. The worker
    // drains autonomously, so the state is `draining` while it drains and
    // `stopped` once it has sealed -- either means a fault ended recording.
    const auto state_after_fault = recorder.state();
    CHECK(state_after_fault == RecorderLifecycleState::draining ||
          state_after_fault == RecorderLifecycleState::stopped);
    const auto status = recorder.status();
    CHECK(status.primary_fault.present);
    CHECK(status.effective_session_outcome == EffectiveSessionOutcome::faulted);
    CHECK(status.requested_terminal_intent == RequestedTerminalIntent::fault);

    // A fault stops accepting: a later frame is late, not a second loss.
    auto late = make_frame(99, std::array<std::uint32_t, 1>{kSignalA});
    CHECK(recorder.observe_frame(late.view()) == streaming::StreamStatus::stopped);
    CHECK(recorder.status().rejected_after_close_data == 1);

    // The autonomous drain completes on its own to `stopped` (contract section
    // 3.1: draining + drain completes -> stopped), not stuck in `draining` until
    // someone calls stop() -- that is the bug this round fixes. Wait for that
    // terminal state directly: the worker sets it only after it has sealed, so
    // it is a reliable "the drain finished" signal (worker_running alone is not
    // -- it is false before the worker thread first runs).
    for (int attempt = 0; attempt < 100000; ++attempt)
    {
        if (recorder.state() == RecorderLifecycleState::stopped ||
            recorder.state() == RecorderLifecycleState::failed)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    CHECK(recorder.state() == RecorderLifecycleState::stopped);
    CHECK(recorder.status().spool_ended_cleanly);
    CHECK(recorder.status().spool_committed == 1);

    // stop() is now idempotent: the autonomous drain already reached `stopped`.
    CHECK(recorder.stop("runtime fault") == RecorderStatusCode::ok);
    CHECK(recorder.state() == RecorderLifecycleState::stopped);
    CHECK(recorder.close() == RecorderStatusCode::ok);
    return 0;
}

// ---------------------------------------------------------------------------
// the bounded shutdown
// ---------------------------------------------------------------------------

int test_over_bound_drain_is_reportable()
{
    PlanFixture fixture;
    fixture.frame_queue_capacity = 8;
    // The bound of contract section 4.7: exceeding it is itself the outcome,
    // not a longer wait.
    fixture.drain_timeout_nanos = 50000000ULL;
    BlockedSpoolFile spool;
    CountingUnixClock clock;
    NativeRecorderCore recorder;

    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);
    spool.blocked.store(true, std::memory_order_release);

    const std::array<std::uint32_t, 1> one{kSignalA};
    auto frame = make_frame(0, one);
    CHECK(recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);

    const auto started = std::chrono::steady_clock::now();
    const auto result = recorder.stop("stalled store");
    const auto elapsed = std::chrono::steady_clock::now() - started;

    CHECK(result == RecorderStatusCode::drain_timed_out);
    CHECK(recorder.state() == RecorderLifecycleState::failed);
    // The runtime must not wait indefinitely for a stalled recorder: two
    // bounded waits, not one unbounded one.
    CHECK(elapsed < std::chrono::seconds(30));

    const auto status = recorder.status();
    CHECK(status.primary_fault.present);
    CHECK(status.primary_fault.reason == RecorderFaultReason::drain_timeout);
    CHECK(status.effective_session_outcome == EffectiveSessionOutcome::faulted);
    // The cooperative backend unblocked on cancel, so the worker sealed what it
    // had: the capture outcome is frozen (faulted, from the drain timeout) and
    // the spool ended cleanly. The session is still faulted -- the drain
    // exceeded its bound -- but no data was lost.
    CHECK(status.spool_ended_cleanly);
    CHECK(status.capture_outcome.has_value());
    CHECK(status.capture_outcome == CaptureOutcome::faulted);

    // Repeating the call returns the same answer and changes nothing.
    CHECK(recorder.stop("again") == RecorderStatusCode::drain_timed_out);
    CHECK(recorder.abort("again") == RecorderStatusCode::drain_timed_out);
    CHECK(recorder.state() == RecorderLifecycleState::failed);
    // The cooperative backend unblocked and the worker joined inside stop()'s
    // bound, so no thread or queue survives stop(): a bounded shutdown that
    // leaves nothing behind, not one that defers an unbounded join to close().
    CHECK(!recorder.status().worker_running);
    CHECK(recorder.status().data_queue_capacity == 0);

    // close() is idempotent from this terminal state (resources already
    // reclaimed), and stays so.
    CHECK(recorder.close() == RecorderStatusCode::ok);
    CHECK(recorder.state() == RecorderLifecycleState::failed);
    CHECK(!recorder.status().worker_running);
    CHECK(recorder.status().data_queue_capacity == 0);
    CHECK(recorder.close() == RecorderStatusCode::ok);
    return 0;
}

// ---------------------------------------------------------------------------
// resource cleanup
// ---------------------------------------------------------------------------

int test_no_thread_or_descriptor_survives_shutdown()
{
    const auto before = open_descriptor_count();

    for (int cycle = 0; cycle < 24; ++cycle)
    {
        PlanFixture fixture;
        RecorderMemorySpoolFile spool;
        CountingUnixClock clock;
        NativeRecorderCore recorder;

        CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
        CHECK(recorder.status().worker_running ||
              recorder.state() == RecorderLifecycleState::prepared);
        CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
        CHECK(recorder.start() == RecorderStatusCode::ok);

        auto frame =
            make_frame(static_cast<std::uint64_t>(cycle), std::array<std::uint32_t, 1>{kSignalA});
        CHECK(recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);

        // Alternate the exit path, so every route into a terminal state is
        // covered by the same cleanup assertion.
        if (cycle % 3 == 0)
        {
            CHECK(recorder.stop("cycle") == RecorderStatusCode::ok);
        }
        else if (cycle % 3 == 1)
        {
            CHECK(recorder.abort("cycle") == RecorderStatusCode::ok);
        }
        CHECK(recorder.close() == RecorderStatusCode::ok);

        const auto status = recorder.status();
        CHECK(!status.worker_running);
        CHECK(status.data_queue_capacity == 0);
        CHECK(status.control_queue_capacity == 0);
    }

    // A recorder destroyed without a close still releases its worker: the
    // destructor calls the idempotent close, because a thread outliving the
    // queues it reads is the one leak that cannot be diagnosed afterwards.
    {
        PlanFixture fixture;
        RecorderMemorySpoolFile spool;
        CountingUnixClock clock;
        NativeRecorderCore recorder;
        CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
        CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
        CHECK(recorder.start() == RecorderStatusCode::ok);
        auto frame = make_frame(0, std::array<std::uint32_t, 1>{kSignalA});
        CHECK(recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);
    }

    const auto after = open_descriptor_count();
    CHECK(after == before);
    return 0;
}

// ---------------------------------------------------------------------------
// multi-producer control plane
// ---------------------------------------------------------------------------

int test_concurrent_control_submissions_keep_queue_intact()
{
    // The control plane is produced by the experiment or application, which is
    // not a single thread the way the critical observer edge is. Several
    // submit_control() callers must not take the same slot: that would be
    // control data corruption, not a performance problem. The queue is large
    // enough that nothing should saturate, so every record must be accepted
    // and committed exactly once.
    PlanFixture fixture;
    fixture.control_queue_capacity = 16384;
    RecorderMemorySpoolFile spool;
    spool.reserve_bytes(8ULL * 1024ULL * 1024ULL);
    CountingUnixClock clock;
    NativeRecorderCore recorder;

    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);

    constexpr int kThreads = 8;
    constexpr int kPerThread = 500;
    std::atomic<std::uint64_t> accepted{0};
    std::atomic<std::uint64_t> reported_true{0};

    std::vector<std::thread> producers;
    producers.reserve(kThreads);
    for (int thread_idx = 0; thread_idx < kThreads; ++thread_idx)
    {
        producers.emplace_back(
            [&, thread_idx]
            {
                for (int i = 0; i < kPerThread; ++i)
                {
                    const auto identity = static_cast<std::uint64_t>(thread_idx) * kPerThread + i;
                    const bool taken = recorder.submit_control(ControlSubmission{
                        .kind = ProducerIdentityKind::events,
                        .identity = identity,
                        .clock_domain = 1,
                        .time_ns = identity,
                        .body = {},
                    });
                    if (taken)
                    {
                        accepted.fetch_add(1, std::memory_order_relaxed);
                        reported_true.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });
    }
    for (auto& thread : producers)
    {
        thread.join();
    }

    // No saturation, so no rejection and no fault: exactly every record was
    // accepted. A corrupted (SPSC) queue would either lose slots or report more
    // accepts than committed, and would often fault the writer.
    CHECK(reported_true.load() == static_cast<std::uint64_t>(kThreads * kPerThread));
    CHECK(accepted.load() == static_cast<std::uint64_t>(kThreads * kPerThread));
    CHECK(!recorder.status().primary_fault.present);

    CHECK(recorder.stop("done") == RecorderStatusCode::ok);
    const auto status = recorder.status();
    CHECK(status.control_offered == static_cast<std::uint64_t>(kThreads * kPerThread));
    CHECK(status.control_accepted == static_cast<std::uint64_t>(kThreads * kPerThread));
    CHECK(status.control_spool_committed == static_cast<std::uint64_t>(kThreads * kPerThread));
    CHECK(status.control_rejected == 0);
    CHECK(status.lost_between_control_acceptance_and_spool == 0);
    CHECK(!status.primary_fault.present);
    CHECK(recorder.close() == RecorderStatusCode::ok);
    return 0;
}

int test_failed_prepare_leaves_reusable_object()
{
    // Contract section 3.1: a prepare() that fails leaves the recorder `created`
    // with nothing retained, the object may be prepared again, and -- because
    // prepare() commits no durable byte (the superblock is committed at the
    // readiness gate) -- no session artifact is left on the failed spool.
    PlanFixture fixture;
    CountingUnixClock clock;
    NativeRecorderCore recorder;

    // A non-empty spool: the writer refuses a file that is not empty, so
    // prepare() fails after the recorder's own queues were reserved. No worker
    // is started and no I/O is done at prepare, so a plain (non-thread-safe)
    // spool double is enough here.
    MemorySpoolFile bad_spool;
    bad_spool.data.push_back(std::byte{0xAB});
    CHECK(recorder.prepare(fixture.plan(), bad_spool, clock) == RecorderStatusCode::prepare_failed);
    CHECK(recorder.state() == RecorderLifecycleState::created);
    // The failed spool is untouched: nothing was written at prepare.
    CHECK(bad_spool.data.size() == 1);

    // The object is exactly as it was: a second prepare against a fresh, empty
    // spool succeeds, the gate commits the superblock, and the session records.
    RecorderMemorySpoolFile good_spool;
    good_spool.reserve_bytes(1ULL << 20U);
    CHECK(recorder.prepare(fixture.plan(), good_spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.state() == RecorderLifecycleState::prepared);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);

    auto frame = make_frame(0, std::array<std::uint32_t, 1>{kSignalA});
    CHECK(recorder.observe_frame(frame.view()) == streaming::StreamStatus::ok);
    CHECK(recorder.stop("done") == RecorderStatusCode::ok);
    CHECK(recorder.status().spool_committed == 1);
    CHECK(recorder.close() == RecorderStatusCode::ok);
    return 0;
}

// ---------------------------------------------------------------------------
// shutdown quiescence (BLOCKER 1)
// ---------------------------------------------------------------------------

int test_shutdown_waits_for_in_flight_control_submission()
{
    // BLOCKER 1: a producer holds a span into control-queue storage between
    // try_acquire() and publish(). A shutdown that freed that storage first
    // would be a use-after-free, so the worker must treat a claimed-but-
    // unpublished MPSC cell (pending > 0, try_peek empty) as "producer not done"
    // -- not "queue empty" -- and shutdown must wait for in-flight producers
    // before releasing storage. This test parks a producer mid-copy (a large
    // body makes the memcpy long enough to observe the claimed-but-unpublished
    // state) and shuts down while it is there; without the quiescence barrier
    // the worker would seal the in-flight cell as if the queue were empty and
    // free the storage under the producer's memcpy.
    PlanFixture fixture;
    auto plan = fixture.plan();
    plan.control_queue_capacity = 4;
    plan.max_control_payload_bytes = 1ULL << 20U; // 1 MiB body -> a long copy
    plan.max_transaction_bytes = 2ULL << 20U;     // the control fits one txn
    RecorderMemorySpoolFile spool;
    spool.reserve_bytes(16ULL << 20U);
    CountingUnixClock clock;
    NativeRecorderCore recorder;
    CHECK(recorder.prepare(plan, spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);

    std::vector<std::byte> body(plan.max_control_payload_bytes, std::byte{0x5A});
    std::atomic<bool> producer_returned{false};
    std::atomic<bool> producer_ready{false};

    std::thread producer(
        [&]
        {
            // The body is at the plan limit (the check is strict-greater), so this
            // is accepted and published. What matters is that the copy is still
            // running when stop() is called below.
            producer_ready.store(true, std::memory_order_release);
            const bool taken = recorder.submit_control(ControlSubmission{
                .kind = ProducerIdentityKind::events,
                .identity = 0,
                .clock_domain = 1,
                .time_ns = 0,
                .body = body,
            });
            (void)taken;
            producer_returned.store(true, std::memory_order_release);
        });

    // Wait until the producer has claimed a slot but not published: the MPSC
    // head has advanced (control_queue_pending == 1) while the copy is still
    // running (control_accepted == 0). This is the span-holding window. The
    // handshake and the sleepless poll are what make it observable in an
    // optimised build too, where the copy takes a few hundred microseconds and
    // a polling sleep would step straight over it.
    while (!producer_ready.load(std::memory_order_acquire))
    {
        std::this_thread::yield();
    }
    bool caught_mid_publish = false;
    while (!producer_returned.load(std::memory_order_acquire))
    {
        const auto status = recorder.status();
        if (status.control_queue_pending == 1 && status.control_accepted == 0)
        {
            caught_mid_publish = true;
            break;
        }
    }
    if (!caught_mid_publish)
    {
        // Join before failing: a test that returns here would leave a running
        // thread holding a reference to a recorder about to be destroyed.
        producer.join();
        CHECK(caught_mid_publish);
    }

    // Shut down while the producer is still mid-copy. The worker waits for the
    // producer to publish (active_control_submissions > 0) before it seals, and
    // release waits for the same counter before freeing the queue, so the
    // producer's span outlives the storage. (Under ASan this is the test that
    // catches the heap-use-after-free the detach-free path used to produce.)
    CHECK(recorder.stop("in-flight control") == RecorderStatusCode::ok);
    producer.join();
    CHECK(producer_returned.load());
    CHECK(recorder.state() == RecorderLifecycleState::stopped);
    // The in-flight control was published and drained, not lost.
    CHECK(recorder.status().control_spool_committed == 1);
    CHECK(recorder.status().lost_between_control_acceptance_and_spool == 0);
    CHECK(!recorder.status().worker_running);
    CHECK(recorder.close() == RecorderStatusCode::ok);
    // close() released the queues; stop() alone does not.
    CHECK(recorder.status().control_queue_capacity == 0);
    CHECK(recorder.status().data_queue_capacity == 0);
    return 0;
}

// ---------------------------------------------------------------------------
// ordinary disk cancellation capability and deferred reclamation
// ---------------------------------------------------------------------------

int test_non_cancellable_backend_passes_readiness_gate()
{
    // Readiness no longer conflates bounded acquisition with bounded disk-I/O
    // cancellation. Ordinary files can record without claiming the latter.
    PlanFixture fixture;
    NonCancellableSpoolFile spool;
    CountingUnixClock clock;
    NativeRecorderCore recorder;
    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.state() == RecorderLifecycleState::prepared);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);
    CHECK(recorder.stop("file recording") == RecorderStatusCode::ok);

    const auto status = recorder.status();
    CHECK(!status.primary_fault.present);
    // Its accepted capability and committed session remain observable.
    CHECK(status.session_created);
    CHECK(!status.spool_backend_cancellable);
    CHECK(spool.size() > 0);
    CHECK(recorder.close() == RecorderStatusCode::ok);
    return 0;
}

int test_non_cancellable_timeout_retains_storage_until_io_finishes()
{
    PlanFixture fixture;
    auto plan = fixture.plan();
    plan.drain_timeout_nanos = 10'000'000;
    BlockedSpoolFile spool;
    spool.cancellable = false;
    CountingUnixClock clock;
    NativeRecorderCore recorder;
    CHECK(recorder.prepare(plan, spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);
    spool.entered.store(false);
    spool.blocked.store(true);
    auto frame = make_frame(0, std::array<std::uint32_t, 1>{kSignalA});
    const auto offered = recorder.observe_frame(frame.view());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!spool.entered.load() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    const bool entered = spool.entered.load();
    const auto before = std::chrono::steady_clock::now();
    const auto stopped = recorder.stop("stalled disk");
    const auto closed = recorder.close();
    const auto elapsed = std::chrono::steady_clock::now() - before;
    const auto held = recorder.status();
    // Always unblock before any assertion can unwind the borrowed storage.
    spool.blocked.store(false);
    CHECK(offered == streaming::StreamStatus::ok);
    CHECK(entered);
    CHECK(stopped == RecorderStatusCode::drain_timed_out);
    CHECK(closed == RecorderStatusCode::drain_timed_out);
    CHECK(elapsed < std::chrono::seconds(1));
    CHECK(held.worker_running);
    CHECK(!held.queue_storage_released);
    CHECK(held.queue_storage_release_deferred);
    CHECK(held.primary_fault.reason == RecorderFaultReason::drain_timeout);
    CHECK(recorder.close() == RecorderStatusCode::ok);
    CHECK(recorder.status().queue_storage_released);
    CHECK(!recorder.status().worker_running);
    return 0;
}

// ---------------------------------------------------------------------------
// discarding an empty spool (HIGH 2)
// ---------------------------------------------------------------------------

int test_ready_close_discards_empty_spool()
{
    // HIGH 2: a readiness gate that passes commits a valid-empty superblock --
    // a `finalizable` artifact -- and sets `session_created`. A close from
    // `ready`, where no record was ever committed, discards that artifact
    // (truncates the spool to nothing) and reports no session, so a scanner
    // never sees a `finalizable` spool the recorder says does not exist
    // (contract section 3.1: prepared/ready -> close discards the spool if it
    // holds no committed record).
    PlanFixture fixture;
    RecorderMemorySpoolFile spool;
    CountingUnixClock clock;
    NativeRecorderCore recorder;
    CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.state() == RecorderLifecycleState::ready);

    // The gate committed the superblock: a session artifact exists even though
    // no record was committed, and `session_created` says so independently of
    // `spool_holds_committed_record`.
    const auto ready_status = recorder.status();
    CHECK(ready_status.session_created);
    CHECK(!ready_status.spool_holds_committed_record);
    CHECK(spool.size() > 0);

    // close() from `ready` discards the empty spool.
    CHECK(recorder.close() == RecorderStatusCode::ok);
    CHECK(recorder.state() == RecorderLifecycleState::closed);
    const auto closed_status = recorder.status();
    CHECK(!closed_status.session_created);
    CHECK(!closed_status.spool_holds_committed_record);
    CHECK(!closed_status.finalization_required);
    CHECK(!closed_status.recovery_required);
    CHECK(spool.snapshot().empty());
    CHECK(!closed_status.worker_running);
    CHECK(closed_status.data_queue_capacity == 0);
    CHECK(closed_status.control_queue_capacity == 0);
    return 0;
}

// ---------------------------------------------------------------------------
// bounded shutdown against a stalled store
// ---------------------------------------------------------------------------

int test_stalled_store_is_reclaimed_within_bound()
{
    // What `supports_bounded_cancel()` has to actually deliver. Not "the caller
    // stopped waiting" -- that leaves the operation, its thread, and its
    // descriptor running, which is the leak rather than the fix, and it recurs
    // on every fault. The operation itself must end, so that the worker exits,
    // the join completes, and nothing is left holding the store.
    //
    // The measurement is deliberately blunt: thread count and descriptor count
    // before and after, with the test doing nothing between them to help. If
    // the shutdown needed the store to be released by hand afterwards, these
    // two numbers would not come back.
    const auto threads_before = live_thread_count();
    const auto descriptors_before = open_descriptor_count();

    PlanFixture fixture;
    fixture.drain_timeout_nanos = 100000000ULL; // 100 ms
    BlockedSpoolFile spool;
    CountingUnixClock clock;

    {
        NativeRecorderCore recorder;
        CHECK(recorder.prepare(fixture.plan(), spool, clock) == RecorderStatusCode::ok);
        CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
        CHECK(recorder.start() == RecorderStatusCode::ok);
        CHECK(live_thread_count() == threads_before + 1);

        // Stall the store after the superblock, then feed it: the worker enters
        // `append` and stays there.
        spool.blocked.store(true, std::memory_order_release);
        const std::array<std::uint32_t, 1> signals{kSignalA};
        for (std::uint64_t sequence = 0; sequence < 4; ++sequence)
        {
            const auto frame = make_frame(sequence, signals);
            static_cast<void>(recorder.observe_frame(frame.view()));
        }

        const auto start = std::chrono::steady_clock::now();
        const auto stopped = recorder.stop("stalled store");
        const auto elapsed = std::chrono::steady_clock::now() - start;
        CHECK(stopped == RecorderStatusCode::drain_timed_out ||
              stopped == RecorderStatusCode::writer_failed);
        CHECK(elapsed < std::chrono::seconds(10));
        CHECK(recorder.state() == RecorderLifecycleState::failed);

        // The recorder ended the operation itself. Nothing in this test touched
        // the store after stalling it, so a store that is no longer blocked is
        // proof that the shutdown's cancel reached it rather than merely
        // abandoning it.
        CHECK(!spool.blocked.load(std::memory_order_acquire));

        const auto status = recorder.status();
        CHECK(status.primary_fault.present);
        CHECK(!status.worker_running);
        CHECK(status.queue_storage_released);
        CHECK(status.data_queue_capacity == 0);
        CHECK(status.control_queue_capacity == 0);
        // The worker thread is gone before the recorder is, not after.
        CHECK(live_thread_count() == threads_before);
        CHECK(recorder.close() == RecorderStatusCode::ok);
    }

    CHECK(live_thread_count() == threads_before);
    CHECK(open_descriptor_count() == descriptors_before);
    return 0;
}

int test_reclamation_is_deferred_under_producer()
{
    // The residual half of BLOCKER 1. The quiescence barrier waits for
    // in-flight producers, but a wait is a bound and a bound can expire -- and
    // when it does, freeing anyway is a use-after-free. This drives the
    // shutdown *past* its bound with a producer holding a queue span, and
    // asserts the recorder chose the third option: retire the storage, hand the
    // free to whoever leaves last, and say so in the status.
    PlanFixture fixture;
    // The drain gives up immediately. A test-only barrier below holds the
    // producer after it claims queue storage, so this race does not depend on
    // memcpy speed or OS scheduling.
    fixture.drain_timeout_nanos = 1000ULL; // 1 us
    auto plan = fixture.plan();
    plan.control_queue_capacity = 4;
    plan.max_control_payload_bytes = 64;
    plan.max_transaction_bytes = 4096;
    RecorderMemorySpoolFile spool;
    spool.reserve_bytes(64ULL << 20U);
    CountingUnixClock clock;
    NativeRecorderCore recorder;
    std::atomic<bool> producer_holds_claim{false};
    std::atomic<bool> release_producer{false};
    RecorderCoreTestHooks hooks{.control_claimed = &producer_holds_claim,
                                .release_control_claim = &release_producer};
    recorder.set_test_hooks(&hooks);
    CHECK(recorder.prepare(plan, spool, clock) == RecorderStatusCode::ok);
    CHECK(recorder.pass_readiness_gate(true) == RecorderStatusCode::ok);
    CHECK(recorder.start() == RecorderStatusCode::ok);

    std::array<std::byte, 64> body{};
    body.fill(std::byte{0x37});
    std::atomic<bool> producer_returned{false};
    std::thread producer(
        [&]
        {
            static_cast<void>(
                recorder.submit_control(ControlSubmission{.kind = ProducerIdentityKind::events,
                                                          .identity = 0,
                                                          .clock_domain = 1,
                                                          .time_ns = 0,
                                                          .body = body}));
            producer_returned.store(true, std::memory_order_release);
        });

    const auto hook_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!producer_holds_claim.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < hook_deadline)
    {
        std::this_thread::yield();
    }
    if (!producer_holds_claim.load(std::memory_order_acquire))
    {
        release_producer.store(true, std::memory_order_release);
        release_producer.notify_all();
        producer.join();
        CHECK(producer_holds_claim.load(std::memory_order_acquire));
    }

    const auto start = std::chrono::steady_clock::now();
    const auto stopped = recorder.stop("shutdown past its bound");
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const auto during = recorder.status();

    release_producer.store(true, std::memory_order_release);
    release_producer.notify_all();
    producer.join();
    CHECK(producer_returned.load(std::memory_order_acquire));

    CHECK(stopped == RecorderStatusCode::drain_timed_out);
    // Bounded, even though a producer is still inside the recorder.
    CHECK(elapsed < std::chrono::seconds(5));
    CHECK(during.queue_storage_release_deferred);
    // The storage the producer is writing into was *not* freed. Under ASan the
    // producer's memcpy is what would have reported it if it had been.
    CHECK(!during.queue_storage_released);

    // The last one out did the free, without the shutdown ever waiting for it.
    const auto after = recorder.status();
    CHECK(after.queue_storage_released);
    CHECK(after.queue_storage_release_deferred);
    CHECK(after.data_queue_capacity == 0);
    CHECK(after.control_queue_capacity == 0);

    CHECK(recorder.close() == RecorderStatusCode::ok);
    // Idempotent through the deferred path too: nothing is freed twice.
    CHECK(recorder.close() == RecorderStatusCode::ok);
    CHECK(recorder.status().queue_storage_released);
    return 0;
}

} // namespace

int main()
{
    for (const auto test : {
             test_full_data_queue_faults_without_dropping,
             test_full_control_queue_uses_reserved_fault_path,
             test_plan_violation_is_recorder_fault,
             test_writer_failure_is_capture_fault_keeping_prefix,
             test_primary_fault_is_not_displaced,
             test_fault_without_frame_context_keeps_session_id,
             test_foreign_session_fault_is_recorded_verbatim,
             test_edge_rejection_is_counted_before_acceptance,
             test_fault_ends_recording_without_stop,
             test_over_bound_drain_is_reportable,
             test_no_thread_or_descriptor_survives_shutdown,
             test_concurrent_control_submissions_keep_queue_intact,
             test_failed_prepare_leaves_reusable_object,
             test_shutdown_waits_for_in_flight_control_submission,
             test_non_cancellable_backend_passes_readiness_gate,
             test_non_cancellable_timeout_retains_storage_until_io_finishes,
             test_ready_close_discards_empty_spool,
             test_stalled_store_is_reclaimed_within_bound,
             test_reclamation_is_deferred_under_producer,
         })
    {
        if (const auto line = test(); line != 0)
        {
            return line;
        }
    }
    return 0;
}
