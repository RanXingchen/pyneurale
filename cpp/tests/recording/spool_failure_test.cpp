/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// The writer's failure paths, and what a reader finds afterwards.
///
/// Every fault here is injected through the `SpoolFile` interface, so each
/// case is one deterministic run rather than an attempt to arrange a full disk
/// or a stalled device on a real filesystem. What is being checked in all of
/// them is the same invariant: whatever the failure, the committed prefix is
/// exactly what it was before the failed operation, and nothing partially
/// written is ever promoted.

#include "check_returns.h"
#include "sha256.h"
#include "spool_file.h"
#include "spool_scanner.h"
#include "spool_test_support.h"
#include "spool_writer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
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

[[nodiscard]] SpoolWriterStatus commit_frames(SpoolWriter& writer, std::uint64_t ordinal)
{
    const auto frame = vector_frame_payload();
    if (const auto status = writer.begin_transaction(kVectorNanos + ordinal);
        status != SpoolWriterStatus::ok)
    {
        return status;
    }
    if (const auto status = writer.append_record(RecordKind::frame, frame, ordinal, kVectorNanos);
        status != SpoolWriterStatus::ok)
    {
        return status;
    }
    return writer.commit_transaction();
}

[[nodiscard]] SpoolScanReport rescan(MemorySpoolFile& file)
{
    std::vector<std::byte> scratch(4096, std::byte{0});
    SpoolScanner scanner;
    return scanner.scan(file, scratch);
}

int test_short_write_is_retried_without_changing_bytes()
{
    // A platform that accepts seven bytes at a time is still a platform the
    // writer has to produce the normative image on.
    MemorySpoolFile file;
    file.max_append_bytes = 7;
    SpoolWriter writer;
    const Fixture fixture;
    CHECK(writer.prepare(file, fixture.identity(), DurabilityPolicy::transaction_sync, limits()) ==
          SpoolWriterStatus::ok);
    CHECK(commit_frames(writer, 1) == SpoolWriterStatus::ok);

    MemorySpoolFile reference;
    SpoolWriter reference_writer;
    CHECK(reference_writer.prepare(reference, fixture.identity(),
                                   DurabilityPolicy::transaction_sync,
                                   limits()) == SpoolWriterStatus::ok);
    CHECK(commit_frames(reference_writer, 1) == SpoolWriterStatus::ok);

    CHECK(file.data == reference.data);
    CHECK(file.append_calls > reference.append_calls);
    CHECK(rescan(file).status() == ScanStatus::ok);
    return 0;
}

int test_stalled_write_fails_instead_of_spinning()
{
    MemorySpoolFile file;
    file.max_append_bytes = 0;
    SpoolWriter writer;
    const Fixture fixture;
    // Retrying a platform that reports success while transferring nothing is
    // how a writer hangs a recorder, so it is reported instead.
    CHECK(writer.prepare(file, fixture.identity(), DurabilityPolicy::transaction_sync, limits()) ==
          SpoolWriterStatus::io_error);
    CHECK(writer.state() == SpoolWriterState::failed);
    return 0;
}

int test_full_store_keeps_committed_prefix()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const Fixture fixture;
    CHECK(writer.prepare(file, fixture.identity(), DurabilityPolicy::transaction_sync, limits()) ==
          SpoolWriterStatus::ok);
    CHECK(commit_frames(writer, 1) == SpoolWriterStatus::ok);

    const auto committed = writer.committed_extent();
    CHECK(file.data.size() == committed);
    // Room for part of the next transaction and no more.
    file.capacity_bytes = static_cast<std::size_t>(committed) + 40;

    CHECK(commit_frames(writer, 2) == SpoolWriterStatus::out_of_space);
    CHECK(writer.state() == SpoolWriterState::failed);
    CHECK(writer.fault() == SpoolWriterStatus::out_of_space);
    CHECK(writer.committed_extent() == committed);

    const auto report = rescan(file);
    CHECK(report.committed_prefix_end() == committed);
    CHECK(report.committed_transactions() == 1);
    // The partial transaction is a tail, and the ordinary crash path: the
    // prefix before it is still promotable.
    CHECK(report.status() == ScanStatus::torn_tail);
    CHECK(report.finalizable());

    // A failed writer is finished. Appending after a failed append would put a
    // valid transaction behind a tail no reader will pass.
    CHECK(writer.begin_transaction(kVectorNanos) == SpoolWriterStatus::wrong_state);
    return 0;
}

int test_stalled_store_is_reported_not_awaited()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const Fixture fixture;
    CHECK(writer.prepare(file, fixture.identity(), DurabilityPolicy::transaction_sync, limits()) ==
          SpoolWriterStatus::ok);

    file.fail_append_call = file.append_calls + 1;
    file.append_fault = SpoolIoStatus::stalled;
    CHECK(commit_frames(writer, 1) == SpoolWriterStatus::stalled);
    CHECK(writer.state() == SpoolWriterState::failed);
    CHECK(writer.fault() == SpoolWriterStatus::stalled);
    // Nothing of the stalled transaction reached the file.
    CHECK(file.data.size() == writer.committed_extent());
    CHECK(rescan(file).status() == ScanStatus::ok);

    // A stalled sync is the same story with a different call.
    MemorySpoolFile other;
    SpoolWriter stalling_sync;
    CHECK(stalling_sync.prepare(other, fixture.identity(), DurabilityPolicy::transaction_sync,
                                limits()) == SpoolWriterStatus::ok);
    other.fail_sync_call = other.sync_calls + 1;
    other.sync_fault = SpoolIoStatus::stalled;
    CHECK(commit_frames(stalling_sync, 1) == SpoolWriterStatus::stalled);
    CHECK(stalling_sync.fault() == SpoolWriterStatus::stalled);
    return 0;
}

int test_failed_sync_costs_durability_not_commit()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const Fixture fixture;
    CHECK(writer.prepare(file, fixture.identity(), DurabilityPolicy::transaction_sync, limits()) ==
          SpoolWriterStatus::ok);
    const auto durable_after_prepare = writer.durable_extent();

    file.fail_sync_call = file.sync_calls + 1;
    file.sync_fault = SpoolIoStatus::io_error;
    CHECK(commit_frames(writer, 1) == SpoolWriterStatus::sync_failed);
    CHECK(writer.state() == SpoolWriterState::failed);

    // The transaction is on the platform and a reader will promote it: what
    // the failure cost is the durability claim, not the commit.
    const auto report = rescan(file);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(report.committed_transactions() == 1);
    CHECK(writer.committed_extent() == report.committed_prefix_end());
    CHECK(writer.durable_extent() == durable_after_prepare);

    // And the two disagree, in the safe direction: a reader deriving the
    // durable extent from the policy alone cannot see a sync that failed, so
    // the writer's own number is the smaller one. Nothing in the container can
    // fix this; it is why the writer reports its own extent at all.
    CHECK(writer.durable_extent() < report.durable_extent_bytes());
    return 0;
}

int test_cancelled_writer_leaves_readable_spool()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const Fixture fixture;
    CHECK(writer.prepare(file, fixture.identity(), DurabilityPolicy::transaction_sync, limits()) ==
          SpoolWriterStatus::ok);
    CHECK(commit_frames(writer, 1) == SpoolWriterStatus::ok);

    const auto frame = vector_frame_payload();
    CHECK(writer.begin_transaction(kVectorNanos) == SpoolWriterStatus::ok);
    CHECK(writer.append_record(RecordKind::frame, frame, 2, kVectorNanos) == SpoolWriterStatus::ok);
    const auto committed = writer.committed_extent();

    writer.cancel();
    CHECK(writer.state() == SpoolWriterState::cancelled);
    // The staged transaction never reached the file, so cancellation leaves no
    // tail at all.
    CHECK(file.data.size() == committed);
    const auto report = rescan(file);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(report.committed_prefix_end() == committed);
    CHECK(report.committed_transactions() == 1);
    CHECK(!report.session_end_present());

    CHECK(writer.begin_transaction(kVectorNanos) == SpoolWriterStatus::wrong_state);
    SpoolAccounting accounting;
    accounting.runtime_accepted = 1;
    accounting.recorder_accepted = 1;
    accounting.spool_committed = 1;
    SpoolSessionEnd session_end;
    CHECK(writer.seal(accounting, session_end, kVectorNanos) == SpoolWriterStatus::wrong_state);

    // A cancelled operation reported by the platform reaches the caller as
    // itself rather than as a generic error.
    MemorySpoolFile other;
    SpoolWriter interrupted;
    CHECK(interrupted.prepare(other, fixture.identity(), DurabilityPolicy::transaction_sync,
                              limits()) == SpoolWriterStatus::ok);
    other.fail_append_call = other.append_calls + 1;
    other.append_fault = SpoolIoStatus::cancelled;
    CHECK(commit_frames(interrupted, 1) == SpoolWriterStatus::cancelled);
    return 0;
}

int test_truncation_stops_at_transaction_boundary()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const Fixture fixture;
    CHECK(writer.prepare(file, fixture.identity(), DurabilityPolicy::transaction_sync, limits()) ==
          SpoolWriterStatus::ok);
    std::vector<std::uint64_t> boundaries{writer.committed_extent()};
    for (std::uint64_t ordinal = 1; ordinal <= 3; ++ordinal)
    {
        CHECK(commit_frames(writer, ordinal) == SpoolWriterStatus::ok);
        boundaries.push_back(writer.committed_extent());
    }
    const auto whole = file.data;

    // Truncated anywhere at all, a spool the writer produced promotes exactly
    // the transactions that completed before the cut.
    for (std::size_t cut = fixture.plan.size(); cut <= whole.size(); ++cut)
    {
        MemorySpoolFile cut_file;
        cut_file.data.assign(whole.begin(), whole.begin() + static_cast<std::ptrdiff_t>(cut));
        const auto report = rescan(cut_file);
        if (!report.readable())
        {
            // Below the superblock region nothing is readable, which is the
            // only case with no committed prefix at all.
            CHECK(cut < boundaries.front());
            continue;
        }
        bool at_boundary = false;
        for (const auto boundary : boundaries)
        {
            if (report.committed_prefix_end() == boundary)
            {
                at_boundary = true;
            }
        }
        CHECK(at_boundary);
        CHECK(report.committed_prefix_end() <= cut);
        if (report.committed_prefix_end() == cut)
        {
            // The cut fell exactly on a transaction boundary: there is no tail.
            CHECK(report.status() == ScanStatus::ok);
        }
        else
        {
            CHECK(report.status() == ScanStatus::torn_tail);
            // The ordinary crash path does not block promotion of the prefix.
            CHECK(report.finalizable());
        }
    }
    return 0;
}

int test_reader_rejects_plan_not_named_by_superblock()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const Fixture fixture;
    CHECK(writer.prepare(file, fixture.identity(), DurabilityPolicy::transaction_sync, limits()) ==
          SpoolWriterStatus::ok);
    CHECK(commit_frames(writer, 1) == SpoolWriterStatus::ok);
    CHECK(rescan(file).status() == ScanStatus::ok);

    // One byte of the stored plan document, changed.
    file.data[kSuperblockBytes] ^= std::byte{0x01};
    const auto report = rescan(file);
    CHECK(report.status() == ScanStatus::rejected);
    CHECK(report.has_code(SpoolCode::plan_mismatch));
    CHECK(report.committed_prefix_end() == 0);
    return 0;
}

int test_read_failure_is_reported_apart_from_verdict()
{
    MemorySpoolFile file;
    SpoolWriter writer;
    const Fixture fixture;
    CHECK(writer.prepare(file, fixture.identity(), DurabilityPolicy::transaction_sync, limits()) ==
          SpoolWriterStatus::ok);
    CHECK(commit_frames(writer, 1) == SpoolWriterStatus::ok);

    const auto image = file.data;
    const auto complete = rescan(file);
    CHECK(!complete.read_failed());
    const auto read_calls = file.read_calls;
    CHECK(read_calls > 1);

    // Every physical read, including the second semantic pass, is a gate. A
    // scanner that could not complete any one of them cannot authorize a
    // finalizer or a committed-record cursor.
    for (std::size_t call = 1; call <= read_calls; ++call)
    {
        MemorySpoolFile failing;
        failing.data = image;
        failing.fail_read_call = call;
        failing.read_fault = SpoolIoStatus::io_error;
        const auto report = rescan(failing);
        CHECK(report.read_failed());
        CHECK(!report.finalizable());
        CHECK(report.status() != ScanStatus::corrupt_tail);
        CHECK(report.status() != ScanStatus::torn_tail);

        SpoolRecordCursor cursor(failing, report);
        SpoolRecordView record;
        CHECK(cursor.failed());
        CHECK(!cursor.next(record));
    }
    return 0;
}

int test_platform_file_holds_and_returns_one_descriptor()
{
    const auto directory = std::filesystem::temp_directory_path() / "neurale-spool-fd-test";
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    CHECK(std::filesystem::create_directories(directory, error));

    const auto count_descriptors = []() -> std::size_t
    {
#ifdef __linux__
        std::error_code iteration_error;
        std::size_t count = 0;
        for (const auto& entry :
             std::filesystem::directory_iterator("/proc/self/fd", iteration_error))
        {
            static_cast<void>(entry);
            ++count;
        }
        return iteration_error ? 0 : count;
#else
        return 0;
#endif
    };

    const auto before = count_descriptors();
    for (int i = 0; i < 32; ++i)
    {
        const auto path = (directory / ("spool-" + std::to_string(i) + ".nspool")).string();
        PlatformSpoolFile file;
        CHECK(file.create(path).complete());
        CHECK(file.is_open());

        SpoolWriter writer;
        const Fixture fixture;
        CHECK(writer.prepare(file, fixture.identity(), DurabilityPolicy::transaction_sync,
                             limits()) == SpoolWriterStatus::ok);
        CHECK(commit_frames(writer, 1) == SpoolWriterStatus::ok);

        std::vector<std::byte> scratch(4096, std::byte{0});
        SpoolScanner scanner;
        const auto report = scanner.scan(file, scratch);
        CHECK(report.status() == ScanStatus::ok);
        CHECK(report.session_id() == kVectorSessionId);

        // Creating over an existing spool is refused: appending to a file
        // whose tail was never scanned is how a committed prefix is orphaned.
        PlatformSpoolFile again;
        CHECK(!again.create(path).complete());
        CHECK(!again.is_open());

        const auto size = file.size();
        CHECK(file.close().complete());
        CHECK(!file.is_open());

        // An explicit repair re-acquires a read-write handle: only repair is
        // allowed to truncate, so the reopen that follows uses read_write.
        PlatformSpoolFile reopened;
        CHECK(reopened.open_existing(path, SpoolOpenMode::read_write).complete());
        CHECK(reopened.is_writable());
        CHECK(reopened.size() == size);
        std::array<std::byte, sizeof(kSpoolMagic)> magic{};
        CHECK(reopened.read_at(0, magic).complete());
        CHECK(std::memcmp(magic.data(), kSpoolMagic, sizeof(kSpoolMagic)) == 0);
        CHECK(reopened.sync().complete());
        CHECK(reopened.truncate(size - 1).complete());
        CHECK(reopened.size() == size - 1);
        CHECK(reopened.close().complete());
    }
    const auto after = count_descriptors();
    CHECK(after == before);

    std::filesystem::remove_all(directory, error);
    return 0;
}

int test_read_only_open_diagnoses_repair_reacquires()
{
    // Contract section 4.5: ordinary diagnosis is read-only. A read of a spool
    // never writes, never truncates, and never repairs, so the production file
    // has a read-only open the scanner uses, and only an explicit repair opens
    // read-write. This is what lets a scanner read a spool on a read-only mount,
    // behind 0444 evidence permissions, or from a quarantine the caller only
    // holds read -- the motivating cases in the review.
    const auto directory = std::filesystem::temp_directory_path() / "neurale-spool-readonly-test";
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    CHECK(std::filesystem::create_directories(directory, error));

    const auto path = (directory / "spool.nspool").string();
    {
        PlatformSpoolFile file;
        CHECK(file.create(path).complete());
        CHECK(file.is_writable());
        SpoolWriter writer;
        const Fixture fixture;
        CHECK(writer.prepare(file, fixture.identity(), DurabilityPolicy::transaction_sync,
                             limits()) == SpoolWriterStatus::ok);
        CHECK(commit_frames(writer, 1) == SpoolWriterStatus::ok);
        CHECK(file.close().complete());
    }

    const auto size_on_disk = std::filesystem::file_size(path, error);
    CHECK(!error);

    // A read-only handle reads and sizes but refuses every mutating and
    // durability operation: capability, not courtesy. None of these refusals
    // is a silent no-op -- each returns a failure so a mistaken caller learns.
    {
        PlatformSpoolFile file;
        CHECK(file.open_existing(path, SpoolOpenMode::read_only).complete());
        CHECK(file.is_open());
        CHECK(!file.is_writable());
        CHECK(file.size() == size_on_disk);

        std::array<std::byte, sizeof(kSpoolMagic)> magic{};
        CHECK(file.read_at(0, magic).complete());
        CHECK(std::memcmp(magic.data(), kSpoolMagic, sizeof(kSpoolMagic)) == 0);

        const std::vector<std::byte> bytes(8, std::byte{0x01});
        CHECK(!file.append(bytes).complete());
        CHECK(!file.sync().complete());
        CHECK(!file.truncate(0).complete());
        CHECK(file.size() == size_on_disk);

        // The scanner reads through the read-only handle exactly as through a
        // read-write one: diagnosis needs read_at and size, and nothing more.
        std::vector<std::byte> scratch(4096, std::byte{0});
        SpoolScanner scanner;
        const auto report = scanner.scan(file, scratch);
        CHECK(report.status() == ScanStatus::ok);
        CHECK(report.session_id() == kVectorSessionId);
        CHECK(report.committed_transactions() == 1);
        CHECK(file.close().complete());
        CHECK(!file.is_writable());
    }

    // A read_write reopen still does everything repair needs (this is the path
    // the existing test exercises in full); the capability flag tracks it.
    {
        PlatformSpoolFile file;
        CHECK(file.open_existing(path, SpoolOpenMode::read_write).complete());
        CHECK(file.is_writable());
        CHECK(file.sync().complete());
        CHECK(file.truncate(size_on_disk).complete());
        CHECK(file.close().complete());
    }

    // Freeze the evidence: revoke write permission. Diagnosis opens read-only
    // and still scans; repair, which needs write access, is refused by the
    // platform -- exactly why diagnosis must not require a writable handle.
    std::filesystem::permissions(path,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::group_read |
                                     std::filesystem::perms::others_read,
                                 std::filesystem::perm_options::replace, error);
    CHECK(!error);
    {
        PlatformSpoolFile file;
        CHECK(file.open_existing(path, SpoolOpenMode::read_only).complete());
        CHECK(!file.is_writable());
        std::vector<std::byte> scratch(4096, std::byte{0});
        SpoolScanner scanner;
        const auto report = scanner.scan(file, scratch);
        CHECK(report.status() == ScanStatus::ok);
        CHECK(report.committed_transactions() == 1);
        CHECK(file.close().complete());
    }
    {
        // Repair re-acquires a writable handle; on a 0444 spool the platform
        // refuses it unless the process can bypass permission checks (root or
        // a privileged Windows token). Either outcome is correct -- the point
        // is that read-only diagnosis never depended on it.
        PlatformSpoolFile file;
        const auto result = file.open_existing(path, SpoolOpenMode::read_write);
        if (!result.complete())
        {
            CHECK(!file.is_open());
        }
        else
        {
            CHECK(file.is_writable());
            CHECK(file.close().complete());
        }
    }

    // Restore permissions so cleanup is unconditional.
    std::filesystem::permissions(path, std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::replace, error);
    std::filesystem::remove_all(directory, error);
    return 0;
}

} // namespace

int main()
{
    for (const auto test : {
             test_short_write_is_retried_without_changing_bytes,
             test_stalled_write_fails_instead_of_spinning,
             test_full_store_keeps_committed_prefix,
             test_stalled_store_is_reported_not_awaited,
             test_failed_sync_costs_durability_not_commit,
             test_cancelled_writer_leaves_readable_spool,
             test_truncation_stops_at_transaction_boundary,
             test_reader_rejects_plan_not_named_by_superblock,
             test_read_failure_is_reported_apart_from_verdict,
             test_platform_file_holds_and_returns_one_descriptor,
             test_read_only_open_diagnoses_repair_reacquires,
         })
    {
        if (const auto line = test(); line != 0)
        {
            return line;
        }
    }
    return 0;
}
