/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// Explicit tail repair: what it removes, what it preserves, and when it
/// refuses.
///
/// Contract section 4.5 makes truncating an invalid tail an explicitly
/// requested, auditable operation, and section 7 makes quarantine -- not
/// deletion -- the way a damaged artifact is handled. Both are checked here as
/// behaviour rather than as documentation.

#include "check_returns.h"
#include "sha256.h"
#include "spool_repair.h"
#include "spool_scanner.h"
#include "spool_test_support.h"
#include "spool_writer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace
{

using namespace neurale::recording;
using namespace neurale::recording::test;

struct Fixture
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

[[nodiscard]] SpoolWriterLimits limits()
{
    return SpoolWriterLimits{.max_records_per_transaction = 16, .max_transaction_bytes = 4096};
}

/// A spool with *transactions* committed transactions, plus *tail* trailing
/// bytes taken from a fourth transaction that never completed.
struct Damaged
{
    MemorySpoolFile file;
    std::uint64_t committed_extent{};
    std::uint64_t last_transaction_id{};
};

[[nodiscard]] bool build_damaged(Damaged& out, std::size_t tail_bytes)
{
    SpoolWriter writer;
    const Fixture fixture;
    if (writer.prepare(out.file, fixture.identity(), DurabilityPolicy::transaction_sync,
                       limits()) != SpoolWriterStatus::ok)
    {
        return false;
    }
    const auto frame = vector_frame_payload();
    for (std::uint64_t ordinal = 1; ordinal <= 2; ++ordinal)
    {
        if (writer.begin_transaction(kVectorNanos + ordinal) != SpoolWriterStatus::ok ||
            writer.append_record(RecordKind::frame, frame, ordinal, kVectorNanos) !=
                SpoolWriterStatus::ok ||
            writer.commit_transaction() != SpoolWriterStatus::ok)
        {
            return false;
        }
    }
    out.committed_extent = writer.committed_extent();
    out.last_transaction_id = writer.committed_transactions();

    // A third transaction that stopped partway, which is what a process crash
    // leaves behind.
    MemorySpoolFile scratch_file;
    SpoolWriter scratch_writer;
    if (scratch_writer.prepare(scratch_file, fixture.identity(), DurabilityPolicy::transaction_sync,
                               limits()) != SpoolWriterStatus::ok)
    {
        return false;
    }
    if (scratch_writer.begin_transaction(kVectorNanos + 3) != SpoolWriterStatus::ok ||
        scratch_writer.append_record(RecordKind::frame, frame, 3, kVectorNanos) !=
            SpoolWriterStatus::ok ||
        scratch_writer.commit_transaction() != SpoolWriterStatus::ok)
    {
        return false;
    }
    const auto source = scratch_file.data;
    const auto begin = source.size() - tail_bytes;
    out.file.data.insert(out.file.data.end(), source.begin() + static_cast<std::ptrdiff_t>(begin),
                         source.end());
    return true;
}

[[nodiscard]] SpoolScanReport rescan(MemorySpoolFile& file)
{
    std::vector<std::byte> scratch(4096, std::byte{0});
    SpoolScanner scanner;
    return scanner.scan(file, scratch);
}

int test_repair_quarantines_before_truncating()
{
    Damaged damaged;
    CHECK(build_damaged(damaged, 40));
    const auto before_bytes = damaged.file.data;
    const auto before = rescan(damaged.file);
    CHECK(before.status() != ScanStatus::ok);
    CHECK(before.committed_prefix_end() == damaged.committed_extent);

    MemorySpoolFile quarantine;
    std::vector<std::byte> scratch(4096, std::byte{0});
    const auto report = repair_spool_tail(damaged.file, &quarantine, scratch);

    CHECK(report.repaired());
    CHECK(report.committed_prefix_end() == damaged.committed_extent);
    CHECK(report.removed_offset() == damaged.committed_extent);
    CHECK(report.removed_bytes() == before_bytes.size() - damaged.committed_extent);
    CHECK(report.quarantined_bytes() == report.removed_bytes());
    CHECK(report.last_committed_transaction_id() == damaged.last_transaction_id);
    CHECK(report.first_removed_transaction_id() == damaged.last_transaction_id + 1);

    // Quarantine, never delete: the removed bytes survive the repair exactly.
    CHECK(quarantine.data.size() == report.removed_bytes());
    CHECK(std::equal(quarantine.data.begin(), quarantine.data.end(),
                     before_bytes.begin() + static_cast<std::ptrdiff_t>(damaged.committed_extent)));
    CHECK(quarantine.sync_calls >= 1);

    // Truncated at the committed prefix end, and at no other offset.
    CHECK(damaged.file.data.size() == damaged.committed_extent);
    CHECK(std::equal(damaged.file.data.begin(), damaged.file.data.end(), before_bytes.begin()));

    CHECK(report.has_after());
    CHECK(report.after().status() == ScanStatus::ok);
    CHECK(report.after().committed_prefix_end() == damaged.committed_extent);
    return 0;
}

int test_repair_without_quarantine_sink_is_rejected()
{
    Damaged damaged;
    CHECK(build_damaged(damaged, 40));
    const auto before_bytes = damaged.file.data;

    std::vector<std::byte> scratch(4096, std::byte{0});
    const auto report = repair_spool_tail(damaged.file, nullptr, scratch);
    CHECK(report.outcome() == SpoolRepairOutcome::no_quarantine);
    CHECK(!report.repaired());
    CHECK(damaged.file.data == before_bytes);
    CHECK(damaged.file.read_calls == 0);
    CHECK(!report.has_after());
    return 0;
}

int test_repair_rejects_aliasing_or_nonempty_quarantine()
{
    Damaged damaged;
    CHECK(build_damaged(damaged, 40));
    const auto before_bytes = damaged.file.data;
    std::vector<std::byte> scratch(4096, std::byte{0});

    const auto alias = repair_spool_tail(damaged.file, &damaged.file, scratch);
    CHECK(alias.outcome() == SpoolRepairOutcome::invalid_quarantine);
    CHECK(damaged.file.data == before_bytes);
    CHECK(damaged.file.read_calls == 0);
    CHECK(damaged.file.truncate_calls == 0);

    MemorySpoolFile nonempty;
    nonempty.data.push_back(std::byte{0x7F});
    const auto existing = nonempty.data;
    const auto occupied = repair_spool_tail(damaged.file, &nonempty, scratch);
    CHECK(occupied.outcome() == SpoolRepairOutcome::invalid_quarantine);
    CHECK(damaged.file.data == before_bytes);
    CHECK(nonempty.data == existing);
    CHECK(damaged.file.truncate_calls == 0);
    return 0;
}

int test_failed_quarantine_leaves_spool_alone()
{
    Damaged damaged;
    CHECK(build_damaged(damaged, 40));
    const auto before_bytes = damaged.file.data;

    MemorySpoolFile quarantine;
    quarantine.fail_append_call = 1;
    quarantine.append_fault = SpoolIoStatus::out_of_space;
    std::vector<std::byte> scratch(4096, std::byte{0});
    const auto report = repair_spool_tail(damaged.file, &quarantine, scratch);
    CHECK(report.outcome() == SpoolRepairOutcome::quarantine_failed);
    CHECK(damaged.file.data == before_bytes);
    CHECK(damaged.file.truncate_calls == 0);

    // A quarantine that was written but not synced is not evidence yet.
    MemorySpoolFile unsynced;
    unsynced.fail_sync_call = 1;
    unsynced.sync_fault = SpoolIoStatus::io_error;
    const auto second = repair_spool_tail(damaged.file, &unsynced, scratch);
    CHECK(second.outcome() == SpoolRepairOutcome::quarantine_failed);
    CHECK(damaged.file.data == before_bytes);
    CHECK(damaged.file.truncate_calls == 0);
    return 0;
}

int test_clean_spool_has_nothing_to_repair()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const Fixture fixture;
    CHECK(writer.prepare(file, fixture.identity(), DurabilityPolicy::transaction_sync, limits()) ==
          SpoolWriterStatus::ok);
    const auto frame = vector_frame_payload();
    CHECK(writer.begin_transaction(kVectorNanos) == SpoolWriterStatus::ok);
    CHECK(writer.append_record(RecordKind::frame, frame, 1, kVectorNanos) == SpoolWriterStatus::ok);
    CHECK(writer.commit_transaction() == SpoolWriterStatus::ok);
    const auto before_bytes = file.data;

    MemorySpoolFile quarantine;
    std::vector<std::byte> scratch(4096, std::byte{0});
    const auto report = repair_spool_tail(file, &quarantine, scratch);
    CHECK(report.outcome() == SpoolRepairOutcome::nothing_to_remove);
    CHECK(file.data == before_bytes);
    CHECK(quarantine.data.empty());
    CHECK(file.truncate_calls == 0);
    return 0;
}

int test_unreadable_spool_is_out_of_scope()
{
    Damaged damaged;
    CHECK(build_damaged(damaged, 40));
    // Break the superblock: without it nothing establishes where a committed
    // prefix ends, so there is no offset a truncation is allowed to use.
    damaged.file.data[0] ^= std::byte{0xFF};
    const auto before_bytes = damaged.file.data;

    MemorySpoolFile quarantine;
    std::vector<std::byte> scratch(4096, std::byte{0});
    const auto report = repair_spool_tail(damaged.file, &quarantine, scratch);
    CHECK(report.outcome() == SpoolRepairOutcome::not_readable);
    CHECK(damaged.file.data == before_bytes);
    CHECK(quarantine.data.empty());
    CHECK(damaged.file.truncate_calls == 0);
    return 0;
}

int test_corrupt_tail_is_repaired_like_torn_tail()
{
    Damaged damaged;
    // A whole transaction's worth of bytes whose begin magic is wrong: present
    // and wrong is corruption, not truncation, and it calls for a different
    // human response but the same removal.
    CHECK(build_damaged(damaged, 128));
    damaged.file.data[static_cast<std::size_t>(damaged.committed_extent)] ^= std::byte{0xFF};

    const auto before = rescan(damaged.file);
    CHECK(before.status() == ScanStatus::corrupt_tail);
    CHECK(before.has_code(SpoolCode::corrupt_transaction_header));

    MemorySpoolFile quarantine;
    std::vector<std::byte> scratch(4096, std::byte{0});
    const auto report = repair_spool_tail(damaged.file, &quarantine, scratch);
    CHECK(report.repaired());
    CHECK(report.before().status() == ScanStatus::corrupt_tail);
    CHECK(report.before().has_code(SpoolCode::corrupt_transaction_header));
    CHECK(report.after().status() == ScanStatus::ok);
    CHECK(quarantine.data.size() == 128);
    return 0;
}

int test_repair_never_forges_session_end()
{
    Damaged damaged;
    CHECK(build_damaged(damaged, 40));
    MemorySpoolFile quarantine;
    std::vector<std::byte> scratch(4096, std::byte{0});
    const auto report = repair_spool_tail(damaged.file, &quarantine, scratch);
    CHECK(report.repaired());

    // The spool's committed prefix is the evidence and it is left exactly as
    // found. A prefix with no session end still has none afterwards, which is
    // what makes the finalizer treat the session as incomplete.
    CHECK(!report.after().session_end_present());
    CHECK(!report.after().has_capture_outcome());
    CHECK(report.after().committed_transactions() == report.before().committed_transactions());
    CHECK(report.after().data_items() == report.before().data_items());
    return 0;
}

int test_scan_never_touches_spool()
{
    Damaged damaged;
    CHECK(build_damaged(damaged, 40));
    const auto before_bytes = damaged.file.data;
    const auto appends_before = damaged.file.append_calls;
    const auto truncates_before = damaged.file.truncate_calls;

    for (int i = 0; i < 3; ++i)
    {
        const auto report = rescan(damaged.file);
        CHECK(report.status() == ScanStatus::torn_tail);
        CHECK(damaged.file.data == before_bytes);
        // Diagnosis is read-only: the scan issues no write of any kind.
        CHECK(damaged.file.truncate_calls == truncates_before);
        CHECK(damaged.file.append_calls == appends_before);
    }
    return 0;
}

int test_failed_truncate_keeps_tail_in_place()
{
    // The truncate syscall failed: the tail was quarantined and synced, but the
    // spool still has it. Nothing was applied, so the report must not claim the
    // file was modified, and there is no new view to scan.
    Damaged damaged;
    CHECK(build_damaged(damaged, 40));
    const auto before_bytes = damaged.file.data;

    MemorySpoolFile quarantine;
    std::vector<std::byte> scratch(4096, std::byte{0});
    damaged.file.fail_truncate_call = damaged.file.truncate_calls + 1;
    damaged.file.truncate_fault = SpoolIoStatus::io_error;
    const auto report = repair_spool_tail(damaged.file, &quarantine, scratch);

    CHECK(report.outcome() == SpoolRepairOutcome::truncate_failed);
    CHECK(!report.truncate_applied());
    CHECK(!report.truncate_sync_succeeded());
    // The tail is still in the file, byte for byte.
    CHECK(damaged.file.data == before_bytes);
    CHECK(damaged.file.data.size() == before_bytes.size());
    // The quarantine was written and synced before the truncate was attempted,
    // so the evidence survives and the same quarantine may be retried.
    CHECK(quarantine.data.size() == before_bytes.size() - damaged.committed_extent);
    CHECK(quarantine.sync_calls >= 1);
    // Nothing was applied, so there is no new state to report.
    CHECK(!report.has_after());
    return 0;
}

int test_late_syncing_truncate_reports_current_state()
{
    // The truncate syscall succeeded but the durability sync that followed it
    // failed. The tail is gone from the current file view, so the report must
    // not say "the spool still has it"; it re-scans and records what is there
    // now, and distinguishes the outcome from a truncate that never happened.
    Damaged damaged;
    CHECK(build_damaged(damaged, 40));
    const auto before_bytes = damaged.file.data;

    MemorySpoolFile quarantine;
    std::vector<std::byte> scratch(4096, std::byte{0});
    // The next sync on the spool is the one the repair issues after the
    // truncate; failing it is the post-truncate sync failure.
    damaged.file.fail_sync_call = damaged.file.sync_calls + 1;
    damaged.file.sync_fault = SpoolIoStatus::io_error;
    const auto report = repair_spool_tail(damaged.file, &quarantine, scratch);

    CHECK(report.outcome() == SpoolRepairOutcome::truncate_sync_failed);
    CHECK(report.truncate_applied());
    CHECK(!report.truncate_sync_succeeded());
    // The quarantine evidence was synced before the truncate was attempted.
    CHECK(quarantine.sync_calls >= 1);
    CHECK(quarantine.data.size() == before_bytes.size() - damaged.committed_extent);
    // The tail is gone from the current file view: the truncate happened.
    CHECK(damaged.file.data.size() == damaged.committed_extent);
    CHECK(std::equal(damaged.file.data.begin(), damaged.file.data.end(), before_bytes.begin()));
    // The report re-scanned the current view rather than claiming "still has
    // it": the after scan shows the committed prefix, clean and unchanged.
    CHECK(report.has_after());
    CHECK(report.after().status() == ScanStatus::ok);
    CHECK(report.after().committed_prefix_end() == damaged.committed_extent);
    return 0;
}

} // namespace

int main()
{
    for (const auto test : {
             test_repair_quarantines_before_truncating,
             test_repair_without_quarantine_sink_is_rejected,
             test_repair_rejects_aliasing_or_nonempty_quarantine,
             test_failed_quarantine_leaves_spool_alone,
             test_clean_spool_has_nothing_to_repair,
             test_unreadable_spool_is_out_of_scope,
             test_corrupt_tail_is_repaired_like_torn_tail,
             test_repair_never_forges_session_end,
             test_scan_never_touches_spool,
             test_failed_truncate_keeps_tail_in_place,
             test_late_syncing_truncate_reports_current_state,
         })
    {
        if (const auto line = test(); line != 0)
        {
            return line;
        }
    }
    return 0;
}
