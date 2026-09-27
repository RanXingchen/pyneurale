/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "spool_repair.h"

#include <algorithm>

namespace neurale::recording
{

SpoolRepairReport repair_spool_tail(SpoolFile& spool, SpoolFile* quarantine,
                                    std::span<std::byte> scratch)
{
    SpoolRepairReport report;

    if (quarantine == nullptr)
    {
        report.outcome_ = SpoolRepairOutcome::no_quarantine;
        return report;
    }
    if (quarantine == &spool || quarantine->size() != 0)
    {
        report.outcome_ = SpoolRepairOutcome::invalid_quarantine;
        return report;
    }

    SpoolScanner scanner;
    report.before_ = scanner.scan(spool, scratch);
    report.committed_prefix_end_ = report.before_.committed_prefix_end();
    report.removed_offset_ = report.before_.committed_prefix_end();
    report.last_committed_transaction_id_ = report.before_.last_transaction_id();
    report.first_removed_transaction_id_ = report.before_.last_transaction_id() + 1;

    if (report.before_.read_failed())
    {
        report.outcome_ = SpoolRepairOutcome::scan_failed;
        return report;
    }
    if (!report.before_.readable())
    {
        // Nothing establishes where a committed prefix ends, so there is no
        // offset this operation is allowed to truncate at.
        report.outcome_ = SpoolRepairOutcome::not_readable;
        return report;
    }

    const auto file_size = spool.size();
    if (file_size <= report.before_.committed_prefix_end())
    {
        report.outcome_ = SpoolRepairOutcome::nothing_to_remove;
        return report;
    }
    report.removed_bytes_ = file_size - report.before_.committed_prefix_end();

    std::uint64_t copied = 0;
    while (copied < report.removed_bytes_)
    {
        const auto take = static_cast<std::size_t>(std::min<std::uint64_t>(
            report.removed_bytes_ - copied, static_cast<std::uint64_t>(scratch.size())));
        auto chunk = scratch.first(take);
        const auto read = spool.read_at(report.removed_offset_ + copied, chunk);
        if (read.status != SpoolIoStatus::ok || read.transferred != take)
        {
            report.outcome_ = SpoolRepairOutcome::quarantine_failed;
            report.quarantined_bytes_ = copied;
            return report;
        }
        std::size_t written = 0;
        while (written < take)
        {
            const auto result = quarantine->append(chunk.subspan(written));
            written += result.transferred;
            if (result.status == SpoolIoStatus::ok)
            {
                continue;
            }
            if (result.status == SpoolIoStatus::incomplete && result.transferred != 0)
            {
                continue;
            }
            report.outcome_ = SpoolRepairOutcome::quarantine_failed;
            report.quarantined_bytes_ = copied + written;
            return report;
        }
        copied += take;
    }
    if (quarantine->sync().status != SpoolIoStatus::ok)
    {
        // The evidence has to be on the platform before the spool loses it.
        report.outcome_ = SpoolRepairOutcome::quarantine_failed;
        report.quarantined_bytes_ = copied;
        return report;
    }
    report.quarantined_bytes_ = copied;

    if (spool.truncate(report.removed_offset_).status != SpoolIoStatus::ok)
    {
        // The truncate syscall failed: the tail is quarantined and still in the
        // file. Nothing was applied, so the same quarantine can be retried.
        report.outcome_ = SpoolRepairOutcome::truncate_failed;
        report.truncate_applied_ = false;
        report.truncate_sync_succeeded_ = false;
        return report;
    }
    report.truncate_applied_ = true;

    if (spool.sync().status != SpoolIoStatus::ok)
    {
        // The truncate syscall succeeded -- the tail is gone from the current
        // file view -- but the durability sync that followed it failed. That is
        // not the same condition as a truncate that never happened, and
        // reporting `truncate_failed` ("the spool still has it") would mislead a
        // caller about whether the file was modified, whether the quarantine
        // can be retried, and what the current disk view shows. Re-scan the
        // current view so the report records what is actually there now.
        report.outcome_ = SpoolRepairOutcome::truncate_sync_failed;
        report.truncate_sync_succeeded_ = false;
        SpoolScanner after_scanner;
        report.after_ = after_scanner.scan(spool, scratch);
        report.has_after_ = true;
        return report;
    }
    report.truncate_sync_succeeded_ = true;

    SpoolScanner after_scanner;
    report.after_ = after_scanner.scan(spool, scratch);
    report.has_after_ = true;
    if (report.after_.status() != ScanStatus::ok ||
        report.after_.committed_prefix_end() != report.before_.committed_prefix_end())
    {
        // A repair that did not leave the committed prefix exactly as it found
        // it has done something other than what it claimed.
        report.outcome_ = SpoolRepairOutcome::verification_failed;
        return report;
    }

    report.outcome_ = SpoolRepairOutcome::repaired;
    return report;
}

} // namespace neurale::recording
