/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The native mirror of a prepared `RecordingPlan`.
///
/// The Python compiler already decided what a session records and produced a
/// canonical document and its fingerprint. This struct is what the native
/// recorder is held to: every bound it allocates against, and the identity it
/// stamps into the spool superblock. It is deliberately **not** a parser --
/// the canonical document travels through as opaque bytes, because a second
/// implementation that read the plan would be a second place the plan could be
/// interpreted differently, and because parsing it would put JSON on a path
/// the contract forbids it on.
///
/// Everything here is fixed at `prepare()` and never changes afterwards. A
/// recorder whose bounds could move after preparation would not be a bounded
/// recorder.

#include "spool_layout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace neurale::recording
{

/// What the plan says about one signal the session records.
///
/// `max_block_bytes` and `max_block_samples` come from `PlannedSignal`; they
/// are the bound the critical callback validates an incoming block against,
/// before it copies anything. A block larger than its plan allows is a plan
/// violation, which is a recorder fault (contract section 4.3) rather than a
/// resize.
struct PlannedSignalRecording
{
    std::uint32_t signal_id{};
    std::uint64_t max_block_bytes{};
    std::uint64_t max_block_samples{};
};

/// Why a plan was refused.
enum class RecordingPlanStatus : std::uint8_t
{
    ok = 0,
    /// A required identifier is empty, oversized, or otherwise unusable.
    invalid_identity = 1,
    /// A bound is zero, or large enough that its derived slot size overflows.
    invalid_bound = 2,
    /// The recorded-signal table is empty, unsorted, or has a duplicate id.
    invalid_signal_table = 3,
    /// The bounds do not admit a single largest frame: one item would never
    /// fit in one spool transaction, so the recorder could never make progress.
    inconsistent_bounds = 4,
};

/// The immutable plan a native recorder prepares against.
struct NativeRecordingPlan
{
    // --- session identity, stored in the spool superblock -------------------

    /// The NRF session id. Must fit the superblock's fixed 128-byte field.
    std::string_view session_id{};
    std::array<std::uint8_t, kSessionUuidBytes> session_uuid{};
    std::uint64_t created_unix_nanos{};
    /// The canonical plan document. Stored whole, never parsed.
    std::span<const std::byte> plan_document{};
    /// SHA-256 of `plan_document`. The spool writer refuses a mismatch.
    std::array<std::uint8_t, kPlanFingerprintBytes> plan_fingerprint{};

    /// The native streaming session the recorder expects on the data plane. A
    /// frame from another session is a plan violation, not data to record.
    std::uint64_t native_session_id{};
    /// The schema id the plan compiled against.
    std::uint32_t native_schema_id{};

    // --- what is recorded --------------------------------------------------

    /// Signals this session records, ascending by `signal_id`. A block whose
    /// signal is absent is not recorded; that is `coverage: "partial"` in the
    /// plan document, and it is a decision the compiler already made.
    std::span<const PlannedSignalRecording> recorded_signals{};

    // --- bounds, all fixed at prepare --------------------------------------

    /// Data-plane queue depth, in items (`ResourceBounds.frame_queue_capacity`).
    std::size_t frame_queue_capacity{};
    /// Control-plane queue depth, in records.
    std::size_t control_queue_capacity{};
    /// Largest block count one frame may carry.
    std::uint32_t max_blocks_per_frame{};
    /// Largest total payload one frame may carry, in bytes.
    std::uint64_t max_frame_payload_bytes{};
    /// Largest gap count one discontinuity may carry.
    std::uint32_t max_signal_gaps_per_discontinuity{};
    /// Largest opaque control body one control record may carry, in bytes.
    std::size_t max_control_payload_bytes{};

    /// Records the spool writer accepts in one transaction.
    std::size_t max_records_per_transaction{};
    /// Bytes the spool writer accepts in one transaction, framing included.
    std::size_t max_transaction_bytes{};
    /// Commit a spool checkpoint after this many committed transactions. Zero
    /// disables checkpoints, which is legal: a checkpoint is a durability
    /// claim, not a correctness requirement.
    std::uint64_t checkpoint_interval_transactions{};

    /// How long the worker sleeps when both queues are empty. The worker
    /// polls rather than waiting on a condition variable so the critical
    /// callback performs no wake syscall at all -- see `recorder.h`.
    std::uint64_t worker_idle_poll_nanos{200000};
    /// The bounded shutdown timeout of contract section 4.7. Exceeding it is a
    /// reportable outcome, not a longer wait.
    std::uint64_t drain_timeout_nanos{5000000000ULL};

    /// The durability policy the spool is written under (contract section 6).
    DurabilityPolicy durability_policy{DurabilityPolicy::checkpoint_sync};

    /// Check every rule above. Cheap, total, and called by `prepare()` before
    /// a single byte is allocated.
    [[nodiscard]] RecordingPlanStatus validate() const noexcept;

    /// The recorded-signal entry for *signal_id*, or nullptr. Binary search
    /// over an ascending table: bounded, allocation-free, and callable from
    /// the critical callback.
    [[nodiscard]] const PlannedSignalRecording* find_signal(std::uint32_t signal_id) const noexcept;
};

/// Spelling used in status text and test names.
[[nodiscard]] std::string_view to_string(RecordingPlanStatus status) noexcept;

} // namespace neurale::recording
