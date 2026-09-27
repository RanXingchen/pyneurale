/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// The spool writer, against the normative bytes and against its own bounds.
///
/// The strongest check here is byte-for-byte reproduction of the normative
/// vectors. Specification section 12 asks a conforming writer to "reproduce
/// the vectors themselves byte for byte if it can write", and the vector
/// generator has no clock and no random source, so with the same fixed inputs
/// the writer either produces the normative image or it does not. A verdict
/// comparison would not catch a field written in the wrong place that happens
/// to scan the same.
///
/// Two entry points, like the strict-realtime contract test:
///   --contract-only    behaviour
///   --allocation-only  the steady-state allocation gate (returns 2 on failure)

#include "allocation_tracker.h"
#include "check_returns.h"
#include "sha256.h"
#include "spool_scanner.h"
#include "spool_test_support.h"
#include "spool_writer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using namespace neurale::recording;
using namespace neurale::recording::test;

/// Owns the plan document the identity's span points at, so the span cannot
/// outlive it.
struct VectorInputs
{
    std::vector<std::byte> plan{as_bytes(kVectorPlanDocument)};

    [[nodiscard]] SpoolSessionIdentity identity() const
    {
        SpoolSessionIdentity value;
        value.session_id = kVectorSessionId;
        value.session_uuid = kVectorSessionUuid;
        value.created_unix_nanos = kVectorCreatedUnixNanos;
        value.plan_document = plan;
        value.plan_fingerprint = sha256(plan);
        return value;
    }
};

[[nodiscard]] SpoolWriterLimits generous_limits()
{
    return SpoolWriterLimits{.max_records_per_transaction = 16, .max_transaction_bytes = 4096};
}

/// Commit the six-record data transaction the vectors carry.
[[nodiscard]] SpoolWriterStatus write_data_transaction(SpoolWriter& writer)
{
    const auto frame = vector_frame_payload();
    const auto block_a = vector_block_payload_a();
    const auto block_b = vector_block_payload_b();
    const auto discontinuity = vector_discontinuity_payload();
    const auto gap = vector_gap_payload();
    const auto control = vector_control_payload();

    if (const auto status = writer.begin_transaction(kVectorNanos + 1);
        status != SpoolWriterStatus::ok)
    {
        return status;
    }
    struct Staged
    {
        RecordKind kind;
        const std::vector<std::byte>* payload;
        std::uint64_t ordinal;
        std::uint64_t nanos;
    };
    const Staged staged[] = {
        {RecordKind::frame, &frame, 1, kVectorNanos},
        {RecordKind::signal_block, &block_a, 1, 0},
        {RecordKind::signal_block, &block_b, 1, 0},
        {RecordKind::discontinuity, &discontinuity, 2, kVectorNanos},
        {RecordKind::signal_gap, &gap, 2, 0},
        {RecordKind::control, &control, 1, 0},
    };
    for (const auto& record : staged)
    {
        if (const auto status =
                writer.append_record(record.kind, *record.payload, record.ordinal, record.nanos);
            status != SpoolWriterStatus::ok)
        {
            return status;
        }
    }
    return writer.commit_transaction();
}

[[nodiscard]] SpoolAccounting vector_accounting()
{
    SpoolAccounting accounting;
    accounting.runtime_accepted = 2;
    accounting.recorder_accepted = 2;
    accounting.spool_committed = 2;
    accounting.control_offered = 1;
    accounting.control_accepted = 1;
    accounting.control_spool_committed = 1;
    return accounting;
}

/// Build the `_complete` image of the vector generator under *policy*.
[[nodiscard]] SpoolWriterStatus write_complete(SpoolWriter& writer, MemorySpoolFile& file,
                                               DurabilityPolicy policy)
{
    const VectorInputs inputs;
    if (const auto status = writer.prepare(file, inputs.identity(), policy, generous_limits());
        status != SpoolWriterStatus::ok)
    {
        return status;
    }
    if (const auto status = write_data_transaction(writer); status != SpoolWriterStatus::ok)
    {
        return status;
    }
    if (const auto status = writer.checkpoint(kVectorNanos + 2, kVectorNanos + 100);
        status != SpoolWriterStatus::ok)
    {
        return status;
    }
    SpoolSessionEnd session_end;
    session_end.capture_outcome = CaptureOutcome::normal;
    session_end.requested_terminal_intent = RequestedTerminalIntent::normal;
    session_end.end_unix_nanos = kVectorNanos + 200;
    return writer.seal(vector_accounting(), session_end, kVectorNanos + 3);
}

[[nodiscard]] std::vector<std::byte> load_vector(const char* name, bool& ok)
{
    return read_file(std::string{NEURALE_NATIVE_SPOOL_VECTOR_DIR} + "/" + name, ok);
}

int test_prepared_spool_matches_empty_vector()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const VectorInputs inputs;
    CHECK(writer.prepare(file, inputs.identity(), DurabilityPolicy::transaction_sync,
                         generous_limits()) == SpoolWriterStatus::ok);

    bool ok = false;
    const auto expected = load_vector("valid-empty.spool", ok);
    CHECK(ok);
    CHECK(file.data == expected);
    CHECK(writer.committed_extent() == expected.size());
    // Synced before the spool is reported ready, under every policy.
    CHECK(file.sync_calls == 1);
    CHECK(writer.durable_extent() == expected.size());
    return 0;
}

int test_prepare_allocates_staging_before_superblock()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const VectorInputs inputs;
    const auto max_size = std::vector<std::byte>{}.max_size();
    if (max_size == std::numeric_limits<std::size_t>::max())
    {
        return 0;
    }
    const SpoolWriterLimits impossible{.max_records_per_transaction = 2,
                                       .max_transaction_bytes = max_size + 1};
    bool allocation_refused = false;
    try
    {
        static_cast<void>(writer.prepare(file, inputs.identity(),
                                         DurabilityPolicy::transaction_sync, impossible));
    }
    catch (const std::length_error&)
    {
        allocation_refused = true;
    }
    catch (const std::bad_alloc&)
    {
        allocation_refused = true;
    }
    CHECK(allocation_refused);
    CHECK(file.data.empty());
    CHECK(file.append_calls == 0);
    CHECK(file.sync_calls == 0);
    CHECK(writer.state() == SpoolWriterState::constructed);
    return 0;
}

int test_one_data_transaction_matches_minimal_vector()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const VectorInputs inputs;
    CHECK(writer.prepare(file, inputs.identity(), DurabilityPolicy::transaction_sync,
                         generous_limits()) == SpoolWriterStatus::ok);
    CHECK(write_data_transaction(writer) == SpoolWriterStatus::ok);

    bool ok = false;
    const auto expected = load_vector("valid-minimal.spool", ok);
    CHECK(ok);
    CHECK(file.data == expected);
    CHECK(writer.data_items() == 2);
    CHECK(writer.control_items() == 1);
    CHECK(writer.committed_transactions() == 1);
    return 0;
}

int test_complete_session_matches_complete_vector()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    CHECK(write_complete(writer, file, DurabilityPolicy::transaction_sync) ==
          SpoolWriterStatus::ok);

    bool ok = false;
    const auto expected = load_vector("valid-complete.spool", ok);
    CHECK(ok);
    CHECK(file.data == expected);
    CHECK(writer.state() == SpoolWriterState::sealed);
    CHECK(writer.durable_extent() == expected.size());
    return 0;
}

int test_checkpoint_sync_session_matches_vector()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    CHECK(write_complete(writer, file, DurabilityPolicy::checkpoint_sync) == SpoolWriterStatus::ok);

    bool ok = false;
    const auto expected = load_vector("valid-checkpoint-sync.spool", ok);
    CHECK(ok);
    CHECK(file.data == expected);

    // The same bytes, a different promise: the durable extent is what the last
    // committed checkpoint claimed, not the committed extent.
    std::vector<std::byte> scratch(4096, std::byte{0});
    SpoolScanner scanner;
    const auto report = scanner.scan(file, scratch);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(report.durable_extent_bytes() < report.committed_prefix_end());
    CHECK(writer.durable_extent() == report.durable_extent_bytes());
    return 0;
}

int test_buffered_writer_claims_only_superblock_region()
{
    // The buffered vector exists to show the same bytes under a third policy;
    // a conforming *writer* under `buffered` will not produce them, because it
    // may not claim a durable extent it never synced for. What it must do is
    // claim exactly the superblock region -- the one thing synced before the
    // spool was reported ready -- and nothing else.
    MemorySpoolFile file;
    SpoolWriter writer;
    CHECK(write_complete(writer, file, DurabilityPolicy::buffered) == SpoolWriterStatus::ok);
    CHECK(writer.durable_extent() == writer.first_transaction_offset());

    std::vector<std::byte> scratch(4096, std::byte{0});
    SpoolScanner scanner;
    const auto report = scanner.scan(file, scratch);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(report.finalizable());
    CHECK(report.durable_extent_bytes() == report.first_transaction_offset());
    CHECK(writer.durable_extent() == report.durable_extent_bytes());

    // Exactly one sync, at prepare: nothing after it is covered.
    CHECK(file.sync_calls == 1);

    bool ok = false;
    const auto vector_image = load_vector("valid-buffered.spool", ok);
    CHECK(ok);
    CHECK(file.data.size() == vector_image.size());
    CHECK(file.data != vector_image);
    return 0;
}

int test_policy_decides_writer_sync_frequency()
{
    for (const auto policy : {DurabilityPolicy::buffered, DurabilityPolicy::checkpoint_sync,
                              DurabilityPolicy::transaction_sync})
    {
        MemorySpoolFile file;
        SpoolWriter writer;
        CHECK(write_complete(writer, file, policy) == SpoolWriterStatus::ok);
        switch (policy)
        {
        case DurabilityPolicy::buffered:
            CHECK(file.sync_calls == 1);
            break;
        case DurabilityPolicy::checkpoint_sync:
            // prepare, the checkpoint, and the seal.
            CHECK(file.sync_calls == 3);
            break;
        case DurabilityPolicy::transaction_sync:
            // prepare, then every commit, and the checkpoint's own sync.
            CHECK(file.sync_calls == 5);
            break;
        }
    }
    return 0;
}

int test_writer_rejects_plan_outside_fingerprint()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const VectorInputs inputs;
    auto identity = inputs.identity();
    identity.plan_fingerprint[0] ^= 0xFF;
    CHECK(writer.prepare(file, identity, DurabilityPolicy::transaction_sync, generous_limits()) ==
          SpoolWriterStatus::plan_fingerprint_mismatch);
    // Nothing was written: a spool whose stored plan is not the planned one
    // would be matched to the wrong finalization target.
    CHECK(file.data.empty());
    CHECK(writer.state() == SpoolWriterState::constructed);
    return 0;
}

int test_writer_rejects_oversized_identifiers()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const VectorInputs inputs;
    auto identity = inputs.identity();
    const std::string too_long(kSessionIdBytes + 1, 'x');
    identity.session_id = too_long;
    CHECK(writer.prepare(file, identity, DurabilityPolicy::transaction_sync, generous_limits()) ==
          SpoolWriterStatus::invalid_argument);
    CHECK(file.data.empty());

    MemorySpoolFile sealed_file;
    SpoolWriter sealing;
    const VectorInputs good;
    CHECK(sealing.prepare(sealed_file, good.identity(), DurabilityPolicy::transaction_sync,
                          generous_limits()) == SpoolWriterStatus::ok);
    const std::string long_reason(kTerminalReasonBytes + 1, 'y');
    SpoolSessionEnd session_end;
    session_end.terminal_reason = long_reason;
    SpoolAccounting empty;
    CHECK(sealing.seal(empty, session_end, 1) == SpoolWriterStatus::invalid_argument);
    return 0;
}

int test_writer_rejects_record_reader_would_reject()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const VectorInputs inputs;
    CHECK(writer.prepare(file, inputs.identity(), DurabilityPolicy::transaction_sync,
                         generous_limits()) == SpoolWriterStatus::ok);
    CHECK(writer.begin_transaction(kVectorNanos) == SpoolWriterStatus::ok);

    const auto frame = vector_frame_payload();
    const auto block = vector_block_payload_a();

    // A block with no owning frame in its transaction.
    CHECK(writer.append_record(RecordKind::signal_block, block, 1, 0) ==
          SpoolWriterStatus::invalid_argument);
    CHECK(writer.append_record(RecordKind::frame, frame, 7, kVectorNanos) == SpoolWriterStatus::ok);
    // A block whose ordinal is not its owning frame's.
    CHECK(writer.append_record(RecordKind::signal_block, block, 8, 0) ==
          SpoolWriterStatus::invalid_argument);
    CHECK(writer.append_record(RecordKind::signal_block, block, 7, 0) == SpoolWriterStatus::ok);
    // A gap with no owning discontinuity.
    CHECK(writer.append_record(RecordKind::signal_gap, block, 7, 0) ==
          SpoolWriterStatus::invalid_argument);
    // A record with no owning item may not carry an ordinal.
    CHECK(writer.append_record(RecordKind::fault, block, 3, 0) ==
          SpoolWriterStatus::invalid_argument);
    // The container-owned kinds are not the caller's to write.
    for (const auto kind :
         {RecordKind::accounting_snapshot, RecordKind::session_end, RecordKind::checkpoint})
    {
        CHECK(writer.append_record(kind, block, 0, 0) == SpoolWriterStatus::invalid_argument);
    }
    CHECK(writer.append_record(static_cast<RecordKind>(4095), block, 0, 0) ==
          SpoolWriterStatus::invalid_argument);

    CHECK(writer.commit_transaction() == SpoolWriterStatus::ok);

    std::vector<std::byte> scratch(4096, std::byte{0});
    SpoolScanner scanner;
    const auto report = scanner.scan(file, scratch);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(report.finalizable());
    CHECK(report.findings().empty());
    return 0;
}

int test_writer_rejects_summary_reader_would_reject()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const VectorInputs inputs;
    CHECK(writer.prepare(file, inputs.identity(), DurabilityPolicy::transaction_sync,
                         generous_limits()) == SpoolWriterStatus::ok);
    CHECK(write_data_transaction(writer) == SpoolWriterStatus::ok);

    SpoolSessionEnd session_end;
    session_end.end_unix_nanos = kVectorNanos;

    // A broken layer-1 identity.
    auto broken = vector_accounting();
    broken.runtime_accepted = 3;
    CHECK(writer.seal(broken, session_end, kVectorNanos) ==
          SpoolWriterStatus::accounting_inconsistent);

    // Layer 2: counters that disagree with what this writer committed.
    auto inflated = vector_accounting();
    inflated.runtime_accepted = 99;
    inflated.recorder_accepted = 99;
    inflated.spool_committed = 99;
    CHECK(writer.seal(inflated, session_end, kVectorNanos) ==
          SpoolWriterStatus::accounting_inconsistent);

    // A loss with no first-loss position latched.
    auto unlatched = vector_accounting();
    unlatched.runtime_accepted = 3;
    unlatched.failed_between_runtime_and_recorder = 1;
    CHECK(writer.seal(unlatched, session_end, kVectorNanos) ==
          SpoolWriterStatus::accounting_inconsistent);

    // A loss recorded as a producer identity reads like a pre-acceptance event.
    auto wrong_form = unlatched;
    wrong_form.data_first_loss.tag = PositionTag::producer_identity;
    wrong_form.data_first_loss.identity_kind = ProducerIdentityKind::frame;
    wrong_form.data_first_loss.identity_value = 5;
    CHECK(writer.seal(wrong_form, session_end, kVectorNanos) ==
          SpoolWriterStatus::accounting_inconsistent);

    // A control rejection naming a data-plane kind.
    auto wrong_plane = vector_accounting();
    wrong_plane.control_offered = 2;
    wrong_plane.control_rejected = 1;
    wrong_plane.control_first_rejection.tag = PositionTag::producer_identity;
    wrong_plane.control_first_rejection.identity_kind = ProducerIdentityKind::frame;
    wrong_plane.control_first_rejection.identity_value = 9;
    CHECK(writer.seal(wrong_plane, session_end, kVectorNanos) ==
          SpoolWriterStatus::accounting_inconsistent);

    // Nothing above reached the file.
    CHECK(writer.committed_transactions() == 1);

    // The corrected form is accepted, and scans clean.
    auto latched = unlatched;
    latched.data_first_loss.tag = PositionTag::ordinal;
    latched.data_first_loss.ordinal = 3;
    CHECK(writer.seal(latched, session_end, kVectorNanos) == SpoolWriterStatus::ok);

    std::vector<std::byte> scratch(4096, std::byte{0});
    SpoolScanner scanner;
    const auto report = scanner.scan(file, scratch);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(report.finalizable());
    CHECK(report.session_end_present());
    return 0;
}

int test_accounting_identities_do_not_wrap()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const VectorInputs inputs;
    CHECK(writer.prepare(file, inputs.identity(), DurabilityPolicy::transaction_sync,
                         generous_limits()) == SpoolWriterStatus::ok);

    SpoolSessionEnd session_end;
    auto wrapped = SpoolAccounting{};
    wrapped.recorder_accepted = std::numeric_limits<std::uint64_t>::max();
    wrapped.failed_between_runtime_and_recorder = 1;
    wrapped.lost_between_recorder_and_spool = std::numeric_limits<std::uint64_t>::max();
    CHECK(writer.seal(wrapped, session_end, kVectorNanos) ==
          SpoolWriterStatus::accounting_inconsistent);

    // A large but mathematically valid identity remains valid: the fix rejects
    // overflow, not large counters.
    auto valid = SpoolAccounting{};
    valid.runtime_accepted = std::numeric_limits<std::uint64_t>::max();
    valid.recorder_accepted = std::numeric_limits<std::uint64_t>::max();
    valid.lost_between_recorder_and_spool = std::numeric_limits<std::uint64_t>::max();
    valid.data_first_loss.tag = PositionTag::ordinal;
    valid.data_first_loss.ordinal = 1;
    session_end.capture_outcome = CaptureOutcome::aborted;
    CHECK(writer.seal(valid, session_end, kVectorNanos) == SpoolWriterStatus::ok);

    std::vector<std::byte> scratch(4096, std::byte{0});
    SpoolScanner scanner;
    const auto report = scanner.scan(file, scratch);
    CHECK(report.finalizable());
    return 0;
}

int test_bounds_are_refusals_and_not_growth()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const VectorInputs inputs;
    // The smallest budget that can still seal a session.
    const SpoolWriterLimits tight{.max_records_per_transaction = 2, .max_transaction_bytes = 448};
    CHECK(writer.prepare(file, inputs.identity(), DurabilityPolicy::transaction_sync, tight) ==
          SpoolWriterStatus::ok);

    const auto frame = vector_frame_payload();
    CHECK(writer.begin_transaction(kVectorNanos) == SpoolWriterStatus::ok);
    CHECK(writer.append_record(RecordKind::frame, frame, 1, 0) == SpoolWriterStatus::ok);
    CHECK(writer.append_record(RecordKind::frame, frame, 2, 0) == SpoolWriterStatus::ok);
    CHECK(writer.append_record(RecordKind::frame, frame, 3, 0) ==
          SpoolWriterStatus::limit_exceeded);
    CHECK(writer.staged_records() == 2);

    const std::vector<std::byte> huge(1024, std::byte{7});
    CHECK(writer.append_record(RecordKind::frame, huge, 4, 0) == SpoolWriterStatus::limit_exceeded);
    CHECK(writer.commit_transaction() == SpoolWriterStatus::ok);

    // A budget too small to seal is refused up front rather than discovered at
    // the end of a session.
    MemorySpoolFile small_file;
    SpoolWriter small;
    const SpoolWriterLimits too_small{.max_records_per_transaction = 2,
                                      .max_transaction_bytes = 447};
    CHECK(small.prepare(small_file, inputs.identity(), DurabilityPolicy::transaction_sync,
                        too_small) == SpoolWriterStatus::limit_exceeded);
    CHECK(small_file.data.empty());
    return 0;
}

int test_rejected_owner_cannot_authorize_child_record()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const VectorInputs inputs;
    const SpoolWriterLimits tight{.max_records_per_transaction = 2, .max_transaction_bytes = 448};
    CHECK(writer.prepare(file, inputs.identity(), DurabilityPolicy::transaction_sync, tight) ==
          SpoolWriterStatus::ok);

    const std::vector<std::byte> oversized(1024, std::byte{7});
    const auto child = vector_block_payload_a();

    CHECK(writer.begin_transaction(kVectorNanos) == SpoolWriterStatus::ok);
    CHECK(writer.append_record(RecordKind::frame, oversized, 11, 0) ==
          SpoolWriterStatus::limit_exceeded);
    CHECK(writer.append_record(RecordKind::signal_block, child, 11, 0) ==
          SpoolWriterStatus::invalid_argument);
    CHECK(writer.commit_transaction() == SpoolWriterStatus::invalid_argument);
    writer.discard_transaction();

    CHECK(writer.begin_transaction(kVectorNanos) == SpoolWriterStatus::ok);
    CHECK(writer.append_record(RecordKind::discontinuity, oversized, 12, 0) ==
          SpoolWriterStatus::limit_exceeded);
    CHECK(writer.append_record(RecordKind::signal_gap, child, 12, 0) ==
          SpoolWriterStatus::invalid_argument);
    CHECK(writer.commit_transaction() == SpoolWriterStatus::invalid_argument);
    writer.discard_transaction();

    CHECK(writer.committed_transactions() == 0);
    return 0;
}

int test_discarded_transaction_never_reaches_file()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const VectorInputs inputs;
    CHECK(writer.prepare(file, inputs.identity(), DurabilityPolicy::transaction_sync,
                         generous_limits()) == SpoolWriterStatus::ok);
    const auto after_prepare = file.data.size();
    const auto appends = file.append_calls;

    const auto frame = vector_frame_payload();
    CHECK(writer.begin_transaction(kVectorNanos) == SpoolWriterStatus::ok);
    CHECK(writer.append_record(RecordKind::frame, frame, 1, 0) == SpoolWriterStatus::ok);
    writer.discard_transaction();

    CHECK(file.data.size() == after_prepare);
    CHECK(file.append_calls == appends);
    CHECK(writer.state() == SpoolWriterState::prepared);

    // And the writer keeps going: a discarded transaction is not a fault.
    CHECK(write_data_transaction(writer) == SpoolWriterStatus::ok);
    CHECK(writer.committed_transactions() == 1);
    return 0;
}

int test_no_record_may_follow_seal()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    CHECK(write_complete(writer, file, DurabilityPolicy::transaction_sync) ==
          SpoolWriterStatus::ok);
    CHECK(writer.begin_transaction(kVectorNanos) == SpoolWriterStatus::wrong_state);
    CHECK(writer.checkpoint(kVectorNanos, kVectorNanos) == SpoolWriterStatus::wrong_state);
    SpoolSessionEnd session_end;
    CHECK(writer.seal(vector_accounting(), session_end, kVectorNanos) ==
          SpoolWriterStatus::wrong_state);
    return 0;
}

int test_late_cancel_never_rewrites_terminal_state()
{
    // Contract section 4.5: every terminal state is terminal. A late cancel,
    // a repeated shutdown, or a shutdown race must not turn a successful seal
    // into a cancelled outcome, nor mask a recorded fault. The writer reports
    // its own state to its caller, so a sealed writer that already committed its
    // accounting and session-end stays sealed no matter what arrives later.

    // seal -> cancel -> sealed, and the committed picture is unchanged.
    MemorySpoolFile file;
    SpoolWriter writer;
    CHECK(write_complete(writer, file, DurabilityPolicy::transaction_sync) ==
          SpoolWriterStatus::ok);
    CHECK(writer.state() == SpoolWriterState::sealed);
    const auto committed = writer.committed_extent();
    const auto durable = writer.durable_extent();
    const auto data_items = writer.data_items();
    const auto control_items = writer.control_items();
    const auto transactions = writer.committed_transactions();
    const auto fault = writer.fault();

    writer.cancel();
    CHECK(writer.state() == SpoolWriterState::sealed);
    CHECK(writer.committed_extent() == committed);
    CHECK(writer.durable_extent() == durable);
    CHECK(writer.data_items() == data_items);
    CHECK(writer.control_items() == control_items);
    CHECK(writer.committed_transactions() == transactions);
    CHECK(writer.fault() == fault);
    CHECK(writer.fault() == SpoolWriterStatus::ok);

    // A repeated cancel is the same no-op: the writer is already terminal.
    writer.cancel();
    writer.cancel();
    CHECK(writer.state() == SpoolWriterState::sealed);
    CHECK(writer.committed_extent() == committed);

    // failed -> cancel -> failed, with the fault preserved.
    MemorySpoolFile failing;
    SpoolWriter failed_writer;
    const VectorInputs inputs;
    CHECK(failed_writer.prepare(failing, inputs.identity(), DurabilityPolicy::transaction_sync,
                                generous_limits()) == SpoolWriterStatus::ok);
    failing.fail_append_call = failing.append_calls + 1;
    failing.append_fault = SpoolIoStatus::out_of_space;
    const auto frame = vector_frame_payload();
    CHECK(failed_writer.begin_transaction(kVectorNanos) == SpoolWriterStatus::ok);
    // Records stage in memory; the platform failure surfaces at commit, when
    // the staged bytes reach the file.
    CHECK(failed_writer.append_record(RecordKind::frame, frame, 1, kVectorNanos) ==
          SpoolWriterStatus::ok);
    CHECK(failed_writer.commit_transaction() == SpoolWriterStatus::out_of_space);
    CHECK(failed_writer.state() == SpoolWriterState::failed);
    CHECK(failed_writer.fault() == SpoolWriterStatus::out_of_space);
    failed_writer.cancel();
    CHECK(failed_writer.state() == SpoolWriterState::failed);
    CHECK(failed_writer.fault() == SpoolWriterStatus::out_of_space);

    // cancelled -> cancel -> cancelled: a cancelled writer stays cancelled.
    MemorySpoolFile cancelled_file;
    SpoolWriter cancelled_writer;
    CHECK(cancelled_writer.prepare(cancelled_file, inputs.identity(),
                                   DurabilityPolicy::transaction_sync,
                                   generous_limits()) == SpoolWriterStatus::ok);
    cancelled_writer.cancel();
    CHECK(cancelled_writer.state() == SpoolWriterState::cancelled);
    cancelled_writer.cancel();
    CHECK(cancelled_writer.state() == SpoolWriterState::cancelled);
    return 0;
}

// --- the allocation gate ---------------------------------------------------

/// Run one whole session under the allocation tracker, after prepare.
[[nodiscard]] std::uint64_t measure_steady_state_allocations(bool& ok)
{
    MemorySpoolFile file;
    // The test double's own buffer would allocate as it grows, which says
    // nothing about the writer. Reserve it up front so what the tracker sees
    // is the writer and nothing else.
    file.data.reserve(1U << 20U);

    SpoolWriter writer;
    const VectorInputs inputs;
    if (writer.prepare(file, inputs.identity(), DurabilityPolicy::transaction_sync,
                       generous_limits()) != SpoolWriterStatus::ok)
    {
        ok = false;
        return 0;
    }
    const auto frame = vector_frame_payload();
    const auto block = vector_block_payload_a();
    const auto control = vector_control_payload();

    neurale::benchmark::reset_allocation_count();
    neurale::benchmark::set_allocation_tracking(true);
    for (std::uint64_t i = 1; i <= 256; ++i)
    {
        if (writer.begin_transaction(kVectorNanos + i) != SpoolWriterStatus::ok ||
            writer.append_record(RecordKind::frame, frame, i, kVectorNanos) !=
                SpoolWriterStatus::ok ||
            writer.append_record(RecordKind::signal_block, block, i, 0) != SpoolWriterStatus::ok ||
            writer.append_record(RecordKind::control, control, i, 0) != SpoolWriterStatus::ok ||
            writer.commit_transaction() != SpoolWriterStatus::ok)
        {
            neurale::benchmark::set_allocation_tracking(false);
            ok = false;
            return 0;
        }
        if (i % 64 == 0 &&
            writer.checkpoint(kVectorNanos + i, kVectorNanos + i) != SpoolWriterStatus::ok)
        {
            neurale::benchmark::set_allocation_tracking(false);
            ok = false;
            return 0;
        }
    }
    SpoolAccounting accounting;
    accounting.runtime_accepted = writer.data_items();
    accounting.recorder_accepted = writer.data_items();
    accounting.spool_committed = writer.data_items();
    accounting.control_offered = writer.control_items();
    accounting.control_accepted = writer.control_items();
    accounting.control_spool_committed = writer.control_items();
    SpoolSessionEnd session_end;
    session_end.end_unix_nanos = kVectorNanos;
    const auto sealed = writer.seal(accounting, session_end, kVectorNanos);
    neurale::benchmark::set_allocation_tracking(false);

    ok = sealed == SpoolWriterStatus::ok;
    return neurale::benchmark::allocation_count();
}

int run_allocation_gate()
{
    bool ok = false;
    const auto allocations = measure_steady_state_allocations(ok);
    if (!ok)
    {
        std::cerr << "the writer did not complete the session under the allocation gate\n";
        return 2;
    }
    std::cerr << "backend=" << neurale::benchmark::allocation_tracking_backend()
              << " allocations=" << allocations << '\n';
    if (allocations != 0)
    {
        std::cerr << "the writer allocated after prepare\n";
        return 2;
    }
    return 0;
}

int run_contract()
{
    for (const auto test : {
             test_prepared_spool_matches_empty_vector,
             test_prepare_allocates_staging_before_superblock,
             test_one_data_transaction_matches_minimal_vector,
             test_complete_session_matches_complete_vector,
             test_checkpoint_sync_session_matches_vector,
             test_buffered_writer_claims_only_superblock_region,
             test_policy_decides_writer_sync_frequency,
             test_writer_rejects_plan_outside_fingerprint,
             test_writer_rejects_oversized_identifiers,
             test_writer_rejects_record_reader_would_reject,
             test_writer_rejects_summary_reader_would_reject,
             test_accounting_identities_do_not_wrap,
             test_bounds_are_refusals_and_not_growth,
             test_rejected_owner_cannot_authorize_child_record,
             test_discarded_transaction_never_reaches_file,
             test_no_record_may_follow_seal,
             test_late_cancel_never_rewrites_terminal_state,
         })
    {
        if (const auto line = test(); line != 0)
        {
            return line;
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::string_view mode = argc > 1 ? argv[1] : "";
    if (argc != 2 || (mode != "--contract-only" && mode != "--allocation-only"))
    {
        std::cerr << "usage: neurale_recording_spool_writer_test "
                     "(--contract-only|--allocation-only)\n";
        return 2;
    }
    if (mode == "--allocation-only")
    {
        return run_allocation_gate();
    }
    return run_contract();
}
