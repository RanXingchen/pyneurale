/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The bounded, append-only writer for the private native spool, version 1.
///
/// What this class is for, and what it deliberately is not:
///
/// - It writes the container of `specifications/native-spool/v1/README.md` and
///   nothing else. It does not know what a frame is; a record payload is an
///   opaque byte range whose interior belongs to the recorder core.
/// - It has no observer, no runtime, no NRF, and no Python. There is no schema
///   lookup, no JSON, no logging, and no allocation on the path between
///   `prepare()` and `seal()`: every buffer the writer uses is sized from the
///   configured limits and allocated once, in `prepare()`.
/// - It takes wall-clock nanoseconds from its caller rather than reading a
///   clock. The recorder already timestamps what it hands over, a clock read
///   on the write path is a syscall nobody asked for, and a writer without a
///   clock is a writer whose output a test can compare byte for byte.
/// - It never rewrites a committed byte. The interface has no seek and no
///   positional write, and an interrupted transaction is discarded from a
///   staging buffer, not truncated out of the file.
///
/// Durability follows the declared policy exactly, and the writer never
/// reports a durable extent the policy does not support (contract section 6).

#include "spool_file.h"
#include "spool_layout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace neurale::recording
{

/// Why a writer call did not do what was asked.
enum class SpoolWriterStatus : std::uint8_t
{
    ok = 0,
    /// The call is not legal in the writer's current state.
    wrong_state = 1,
    /// An argument is outside what the format or this writer defines: an
    /// oversized identifier, an undefined enum, an ordinal that does not match
    /// its owning item, a record kind the writer owns.
    invalid_argument = 2,
    /// The request does not fit the bounds configured at `prepare()`. Nothing
    /// grows to accommodate it -- that is what "bounded" means here.
    limit_exceeded = 3,
    /// The stored plan document is not the document the caller's fingerprint
    /// names.
    plan_fingerprint_mismatch = 4,
    /// An accounting snapshot the reader defined by the specification would
    /// reject: a broken identity, a position whose tag or presence disagrees
    /// with its counters, or counters that disagree with what this writer
    /// committed.
    accounting_inconsistent = 5,
    /// The store is full.
    out_of_space = 6,
    /// The store did not complete an operation within the caller's bound.
    stalled = 7,
    /// An explicit sync failed. Everything committed before it is still
    /// committed; nothing after the last successful sync is durable.
    sync_failed = 8,
    /// Any other platform write failure.
    io_error = 9,
    /// The writer was cancelled and will not write again.
    cancelled = 10,
};

/// Writer lifecycle. Every terminal state is terminal: a writer that failed or
/// was cancelled never writes another byte, because appending after a failed
/// append would put valid transactions behind a tail no reader will pass.
enum class SpoolWriterState : std::uint8_t
{
    constructed = 0,
    /// Resources allocated and the superblock built in memory but not yet
    /// published: a writer whose `prepare()` separated allocation from the
    /// durable commit so a recorder that never passes its readiness gate
    /// leaves no artifact. `commit_superblock()` moves this to `prepared`.
    allocated = 1,
    prepared = 2,
    transaction_open = 3,
    sealed = 4,
    cancelled = 5,
    failed = 6,
};

/// The bounds the writer is held to. Both are hard: a record or a transaction
/// that does not fit is refused, never accommodated.
struct SpoolWriterLimits
{
    /// Maximum records in one transaction.
    std::size_t max_records_per_transaction{};
    /// Maximum bytes of one whole transaction, header and trailer included.
    /// This is the staging-buffer capacity, allocated once in `prepare()`.
    std::size_t max_transaction_bytes{};
};

/// What identifies the session a spool captures. All of it is stored in the
/// superblock, because contract section 5 requires it to survive a process
/// crash *in the spool*.
struct SpoolSessionIdentity
{
    /// The NRF session id. Must fit the fixed 128-byte field; truncating an
    /// identifier is never allowed.
    std::string_view session_id{};
    std::array<std::uint8_t, kSessionUuidBytes> session_uuid{};
    std::uint64_t created_unix_nanos{};
    /// The canonical recording-plan document, stored whole and never parsed.
    std::span<const std::byte> plan_document{};
    /// The plan fingerprint the caller expects. `prepare()` refuses when it is
    /// not the SHA-256 of `plan_document`: a spool whose stored plan is not the
    /// planned one would be matched to the wrong finalization target.
    std::array<std::uint8_t, kPlanFingerprintBytes> plan_fingerprint{};
};

/// One first-failed-or-lost position (contract section 5.1).
struct SpoolAccountingPosition
{
    PositionTag tag{PositionTag::absent};
    ProducerIdentityKind identity_kind{ProducerIdentityKind::none};
    std::uint64_t ordinal{};
    std::uint64_t identity_value{};
};

/// The capture-side session accounting. The finalization-side counters are
/// absent by construction: the spool is written before finalization runs, so a
/// spool that carried them would carry numbers nothing had measured. There is
/// no `accounting_verified` field, and there must not be one.
struct SpoolAccounting
{
    std::uint64_t runtime_accepted{};
    std::uint64_t recorder_accepted{};
    std::uint64_t spool_committed{};
    std::uint64_t rejected_before_runtime_acceptance{};
    std::uint64_t failed_between_runtime_and_recorder{};
    std::uint64_t lost_between_recorder_and_spool{};
    std::uint64_t control_offered{};
    std::uint64_t control_accepted{};
    std::uint64_t control_spool_committed{};
    std::uint64_t control_rejected{};
    std::uint64_t lost_between_control_acceptance_and_spool{};
    std::uint64_t rejected_after_close_data{};
    std::uint64_t rejected_after_close_control{};

    bool control_offered_present{true};
    bool producer_acceptance_known{true};

    SpoolAccountingPosition data_first_loss{};
    SpoolAccountingPosition data_first_rejection{};
    SpoolAccountingPosition control_first_loss{};
    SpoolAccountingPosition control_first_rejection{};
};

/// The terminal provenance the session-end record freezes (contract 3.2).
/// Four separate facts because the contract forbids recovering any one of them
/// from the others.
struct SpoolSessionEnd
{
    RequestedTerminalIntent requested_terminal_intent{RequestedTerminalIntent::normal};
    CaptureOutcome capture_outcome{CaptureOutcome::normal};
    bool primary_fault_committed{};
    /// Bounded reason the finalizer copies into the NRF termination record.
    /// Must fit 64 bytes; a longer one is refused rather than truncated.
    std::string_view terminal_reason{};
    std::uint64_t end_unix_nanos{};
};

class SpoolWriter
{
  public:
    SpoolWriter() noexcept = default;

    SpoolWriter(const SpoolWriter&) = delete;
    SpoolWriter& operator=(const SpoolWriter&) = delete;
    SpoolWriter(SpoolWriter&&) = delete;
    SpoolWriter& operator=(SpoolWriter&&) = delete;

    /// Allocate every resource the writer will ever use and build the superblock
    /// in memory, without writing a durable byte. On any failure the writer is
    /// left `constructed` (nothing retained), so a recorder whose `prepare()`
    /// failed on allocation or validation may be prepared again. The superblock
    /// is not published until `commit_superblock()`, so a run that never passes
    /// its readiness gate leaves no artifact.
    ///
    /// *file* must be empty and must outlive the writer.
    [[nodiscard]] SpoolWriterStatus allocate(SpoolFile& file, const SpoolSessionIdentity& identity,
                                             DurabilityPolicy policy,
                                             const SpoolWriterLimits& limits);

    /// Write and sync the superblock `allocate()` built. Moves the writer from
    /// `allocated` to `prepared`. The superblock region is synced under every
    /// policy: a spool whose own identity did not survive a crash could not be
    /// matched to a finalization target at all.
    [[nodiscard]] SpoolWriterStatus commit_superblock() noexcept;

    /// Allocate and publish the superblock in one call. Equivalent to
    /// `allocate()` followed by `commit_superblock()`, kept for the writer's
    /// own tests and any caller that has no readiness gate of its own.
    [[nodiscard]] SpoolWriterStatus prepare(SpoolFile& file, const SpoolSessionIdentity& identity,
                                            DurabilityPolicy policy,
                                            const SpoolWriterLimits& limits);

    /// Open a transaction. Nothing reaches the file until `commit_transaction()`.
    [[nodiscard]] SpoolWriterStatus begin_transaction(std::uint64_t begin_unix_nanos) noexcept;

    /// Stage one record. `kind` must be a data-plane, control-plane, or fault
    /// record: the container-owned kinds are written by `checkpoint()` and
    /// `seal()`, which is what keeps their ordering and quiet-transaction rules
    /// true by construction rather than by convention.
    [[nodiscard]] SpoolWriterStatus append_record(RecordKind kind,
                                                  std::span<const std::byte> payload,
                                                  std::uint64_t logical_ordinal,
                                                  std::uint64_t record_unix_nanos) noexcept;

    /// Write the staged transaction and its commit trailer in one append, then
    /// sync if the policy says to. On success the records are committed, which
    /// is stage 3 of the acceptance ladder and, under `transaction_sync` only,
    /// also durable.
    [[nodiscard]] SpoolWriterStatus commit_transaction() noexcept;

    /// Drop the staged transaction. Nothing was written, so nothing is undone.
    void discard_transaction() noexcept;

    /// Sync, then commit a checkpoint record naming the extent that sync
    /// covered. Under `buffered` the writer does not sync and the checkpoint
    /// claims only the superblock region, because that is the whole of what the
    /// policy delivers.
    [[nodiscard]] SpoolWriterStatus checkpoint(std::uint64_t begin_unix_nanos,
                                               std::uint64_t synced_unix_nanos) noexcept;

    /// Commit the accounting snapshot and the session-end record, in that
    /// order, in one transaction that carries no item. The snapshot is checked
    /// against this writer's own committed counts first: a writer must not
    /// write a summary the reader would refuse.
    [[nodiscard]] SpoolWriterStatus seal(const SpoolAccounting& accounting,
                                         const SpoolSessionEnd& session_end,
                                         std::uint64_t begin_unix_nanos) noexcept;

    /// Abandon the session. Any staged transaction is dropped; the committed
    /// prefix is untouched and remains exactly what a reader would promote.
    /// A sealed, failed, or already-cancelled writer is left in that terminal
    /// state: a late cancel never rewrites a successful outcome or masks a
    /// recorded fault, so a recorder that reads writer state after the call
    /// sees what actually happened.
    void cancel() noexcept;

    [[nodiscard]] SpoolWriterState state() const noexcept
    {
        return state_;
    }

    /// The status that put the writer in `failed`, or `ok`.
    [[nodiscard]] SpoolWriterStatus fault() const noexcept
    {
        return fault_;
    }

    /// One past the last committed byte.
    [[nodiscard]] std::uint64_t committed_extent() const noexcept
    {
        return committed_extent_;
    }

    /// How far synchronization has been acknowledged under the policy in
    /// force. Never a count of records, and never more than the policy
    /// promises: under `buffered` this is the superblock region and no record.
    [[nodiscard]] std::uint64_t durable_extent() const noexcept;

    [[nodiscard]] std::uint64_t committed_transactions() const noexcept
    {
        return committed_transactions_;
    }

    [[nodiscard]] std::uint64_t data_items() const noexcept
    {
        return data_items_;
    }

    [[nodiscard]] std::uint64_t control_items() const noexcept
    {
        return control_items_;
    }

    [[nodiscard]] DurabilityPolicy durability_policy() const noexcept
    {
        return policy_;
    }

    [[nodiscard]] std::uint64_t first_transaction_offset() const noexcept
    {
        return first_transaction_offset_;
    }

    /// Records staged in the open transaction.
    [[nodiscard]] std::size_t staged_records() const noexcept
    {
        return staged_records_;
    }

  private:
    [[nodiscard]] SpoolWriterStatus stage_record(std::uint16_t kind,
                                                 std::span<const std::byte> payload,
                                                 std::uint64_t logical_ordinal,
                                                 std::uint64_t record_unix_nanos) noexcept;
    [[nodiscard]] SpoolWriterStatus commit_staged() noexcept;
    [[nodiscard]] SpoolWriterStatus write_all(std::span<const std::byte> data) noexcept;
    [[nodiscard]] SpoolWriterStatus sync_now() noexcept;
    void fail(SpoolWriterStatus status) noexcept;
    void open_staging(std::uint64_t begin_unix_nanos) noexcept;

    SpoolFile* file_{};
    SpoolWriterState state_{SpoolWriterState::constructed};
    SpoolWriterStatus fault_{SpoolWriterStatus::ok};
    DurabilityPolicy policy_{DurabilityPolicy::transaction_sync};
    SpoolWriterLimits limits_{};
    /// The identity and the built-but-unpublished superblock, held between
    /// `allocate()` and `commit_superblock()`.
    SpoolSessionIdentity identity_{};
    std::vector<std::byte> superblock_region_{};

    std::vector<std::byte> staging_{};
    std::size_t staged_bytes_{};
    std::size_t staged_records_{};
    std::uint32_t staged_data_items_{};
    std::uint32_t staged_control_items_{};
    bool staged_frame_seen_{};
    bool staged_discontinuity_seen_{};
    std::uint64_t staged_frame_ordinal_{};
    std::uint64_t staged_discontinuity_ordinal_{};

    std::uint64_t first_transaction_offset_{};
    std::uint64_t committed_extent_{};
    std::uint64_t synced_extent_{};
    std::uint64_t checkpoint_extent_{};
    std::uint64_t committed_transactions_{};
    std::uint64_t next_transaction_id_{1};
    std::uint64_t previous_transaction_offset_{};
    std::uint64_t data_items_{};
    std::uint64_t control_items_{};
};

/// Encode an accounting payload into *out* (208 bytes). Exposed so tests and,
/// later, the finalizer can build the same bytes the writer does.
void encode_accounting_payload(const SpoolAccounting& accounting,
                               std::span<std::byte, kAccountingPayloadBytes> out) noexcept;

/// Check a snapshot against the rules a conforming reader applies: the layer-1
/// identities, the position tag forms, the identity-kind registry, and
/// presence consistency. `committed_data_items` and `committed_control_items`
/// supply layer 2. Returns `ok` when a reader would accept it.
[[nodiscard]] SpoolWriterStatus validate_accounting(const SpoolAccounting& accounting,
                                                    std::uint64_t committed_data_items,
                                                    std::uint64_t committed_control_items) noexcept;

} // namespace neurale::recording
