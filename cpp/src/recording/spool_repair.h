/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// Explicit removal of an invalid uncommitted tail.
///
/// This is deliberately a separate entry point from the scan. Contract section
/// 4.5 requires ordinary diagnosis to be read-only and truncating an invalid
/// tail to be an explicitly requested, auditable operation that emits a
/// report, so nothing in `spool_scanner.h` can reach this code and no code
/// path arrives here without a caller having asked for it by name.
///
/// The operation, in order (specification section 8):
///
///  1. verify the committed prefix in full;
///  2. copy the bytes it is about to remove into a quarantine sink and sync
///     them -- quarantine, never delete, so the evidence of what went wrong
///     survives the repair;
///  3. truncate at the committed prefix end, and at no other offset;
///  4. re-scan, and report.
///
/// A missing quarantine sink is a refusal, not a warning: removing the only
/// copy of the evidence is the failure mode the retention rules exist to
/// prevent.

#include "spool_file.h"
#include "spool_scanner.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace neurale::recording
{

enum class SpoolRepairOutcome : std::uint8_t
{
    /// The tail was quarantined and removed.
    repaired = 0,
    /// The committed prefix is the whole file; there is nothing to remove.
    nothing_to_remove = 1,
    /// The superblock did not validate. There is no committed prefix to
    /// truncate at, so the whole file is quarantine material, not this
    /// operation's business.
    not_readable = 2,
    /// No quarantine sink was supplied.
    no_quarantine = 3,
    /// The quarantine copy failed. Nothing was truncated.
    quarantine_failed = 4,
    /// The truncate syscall failed. The tail is quarantined and still in the
    /// file: nothing was applied, and the same quarantine may be retried.
    truncate_failed = 5,
    /// The truncate syscall succeeded -- the tail is gone from the current
    /// file view -- but the durability `sync` that followed it failed. This is
    /// distinct from `truncate_failed`: the file has already been modified, so
    /// reporting "the spool still has it" would be a lie. The report's `after`
    /// scan records what the current view actually shows.
    truncate_sync_failed = 6,
    /// The spool could not be read.
    scan_failed = 7,
    /// After truncation the spool did not scan clean. Reported so a repair
    /// that did not achieve what it claimed cannot pass silently.
    verification_failed = 8,
    /// The quarantine aliases the spool or already contains bytes. Either
    /// would make the evidence copy ambiguous or destructive.
    invalid_quarantine = 9,
};

/// What an explicit repair did, and to which bytes.
class SpoolRepairReport
{
  public:
    [[nodiscard]] SpoolRepairOutcome outcome() const noexcept
    {
        return outcome_;
    }

    [[nodiscard]] bool repaired() const noexcept
    {
        return outcome_ == SpoolRepairOutcome::repaired;
    }

    /// The scan that justified the repair, taken before anything was touched.
    [[nodiscard]] const SpoolScanReport& before() const noexcept
    {
        return before_;
    }

    /// The scan taken after the truncation. Meaningful only when the
    /// truncation happened.
    [[nodiscard]] const SpoolScanReport& after() const noexcept
    {
        return after_;
    }

    [[nodiscard]] bool has_after() const noexcept
    {
        return has_after_;
    }

    /// Whether the truncate syscall was applied to the file. False when the
    /// outcome is `truncate_failed` (the syscall failed) and true when it is
    /// `truncate_sync_failed` or later, because the bytes were already removed
    /// from the current view before the failure that followed.
    [[nodiscard]] bool truncate_applied() const noexcept
    {
        return truncate_applied_;
    }

    /// Whether the durability `sync` after a successful truncate was
    /// acknowledged. False when the outcome is `truncate_sync_failed`: the
    /// truncate happened but its durability was not confirmed, so a power loss
    /// may or may not undo it.
    [[nodiscard]] bool truncate_sync_succeeded() const noexcept
    {
        return truncate_sync_succeeded_;
    }

    [[nodiscard]] std::uint64_t committed_prefix_end() const noexcept
    {
        return committed_prefix_end_;
    }

    /// First byte of the removed range. Equal to `committed_prefix_end()`:
    /// truncation happens at the committed prefix end and at no other offset.
    [[nodiscard]] std::uint64_t removed_offset() const noexcept
    {
        return removed_offset_;
    }

    [[nodiscard]] std::uint64_t removed_bytes() const noexcept
    {
        return removed_bytes_;
    }

    [[nodiscard]] std::uint64_t quarantined_bytes() const noexcept
    {
        return quarantined_bytes_;
    }

    [[nodiscard]] std::uint64_t last_committed_transaction_id() const noexcept
    {
        return last_committed_transaction_id_;
    }

    /// The id the first removed transaction would have carried. It is the
    /// successor of the last committed one by construction, because a
    /// transaction that is not the successor is exactly what ends a prefix.
    [[nodiscard]] std::uint64_t first_removed_transaction_id() const noexcept
    {
        return first_removed_transaction_id_;
    }

  private:
    friend SpoolRepairReport repair_spool_tail(SpoolFile&, SpoolFile*, std::span<std::byte>);

    SpoolRepairOutcome outcome_{SpoolRepairOutcome::nothing_to_remove};
    SpoolScanReport before_{};
    SpoolScanReport after_{};
    bool has_after_{};
    bool truncate_applied_{};
    bool truncate_sync_succeeded_{};
    std::uint64_t committed_prefix_end_{};
    std::uint64_t removed_offset_{};
    std::uint64_t removed_bytes_{};
    std::uint64_t quarantined_bytes_{};
    std::uint64_t last_committed_transaction_id_{};
    std::uint64_t first_removed_transaction_id_{};
};

/// Remove the invalid tail of *spool*, preserving it in *quarantine* first.
///
/// *quarantine* must be an empty, writable file the caller owns; passing
/// nullptr is refused. *scratch* is borrowed for the scan and for the copy.
[[nodiscard]] SpoolRepairReport repair_spool_tail(SpoolFile& spool, SpoolFile* quarantine,
                                                  std::span<std::byte> scratch);

} // namespace neurale::recording
