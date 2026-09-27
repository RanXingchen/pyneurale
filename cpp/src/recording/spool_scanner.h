/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The read-only scanner for the private native spool, version 1.
///
/// A read of a spool never writes, never truncates, and never repairs
/// (contract section 4.5). This scanner therefore takes a `SpoolFile` and only
/// ever calls `read_at` and `size` on it. Removing an invalid tail is a
/// separate, explicitly requested operation and lives in `spool_repair.h`.
///
/// The result is a `SpoolScanReport`: a value with no mutating member, built
/// once by the scanner and readable afterwards. What it reports is exactly
/// what the committed prefix establishes -- the committed extent, the item
/// counts, the terminal provenance if a session-end record was committed, the
/// durable extent the declared policy supports, and the diagnostics -- and
/// nothing more.

#include "spool_file.h"
#include "spool_layout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace neurale::recording
{

/// One diagnostic and the byte offset that produced it.
struct SpoolFinding
{
    SpoolCode code{SpoolCode::none};
    std::uint64_t offset{};
};

/// A first-loss or first-rejection position, as a scan read it.
///
/// Distinct from `SpoolAccountingPosition` in `spool_writer.h`, and the
/// difference is the point rather than an oversight: that one is what a writer
/// is given, so its `identity_kind` is a `ProducerIdentityKind` and only a
/// registered kind can be expressed. This one is what a reader found, and a
/// spool may store an unregistered number -- the scan reports that as a
/// finding, which it can only do by having read the number. Narrowing it to the
/// enum here would make the value unrepresentable and the diagnosis impossible.
///
/// The two forms of the union are not interchangeable. A loss after acceptance
/// names the accepted item's `ordinal`; a rejection before acceptance never
/// received an ordinal and names the producer identity instead. Which form is
/// legal in which slot is checked during the scan.
struct ScannedAccountingPosition
{
    PositionTag tag{PositionTag::absent};
    std::uint32_t identity_kind{};
    std::uint64_t ordinal{};
    std::uint64_t identity_value{};

    [[nodiscard]] bool present() const noexcept
    {
        return tag != PositionTag::absent;
    }
};

/// The capture-side accounting snapshot the spool committed (section 4.4).
///
/// The finalization-side counters are deliberately absent: the spool is written
/// before finalization runs, so a spool carrying them would carry numbers
/// nothing had measured. Retained here because the scan already decodes every
/// field to check it, and a caller that had to decode the payload a second time
/// would be a second reader of the same bytes.
struct SpoolAccountingSnapshot
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
    /// Whether `control_offered` was measured. A producer that cannot count
    /// what it offered writes false, and the counter is then not a zero.
    bool control_offered_present{};
    bool producer_acceptance_known{};
    /// Always zero in a spool: spool accounting is written by the recorder.
    /// Retained rather than dropped so a caller reports what it read.
    std::uint8_t accounting_origin{};
    ScannedAccountingPosition data_first_loss{};
    ScannedAccountingPosition data_first_rejection{};
    ScannedAccountingPosition control_first_loss{};
    ScannedAccountingPosition control_first_rejection{};
};

/// What a read of a spool establishes, and nothing more.
class SpoolScanReport
{
  public:
    /// Findings are bounded like everything else here. A spool that produces
    /// more than this is already unfinalizable many times over.
    static constexpr std::size_t kMaxFindings = 64;

    [[nodiscard]] ScanStatus status() const noexcept
    {
        return status_;
    }

    /// One past the last committed byte. Everything at or after it is an
    /// invalid tail: it MUST NOT be promoted, and an ordinary read MUST NOT
    /// remove it.
    [[nodiscard]] std::uint64_t committed_prefix_end() const noexcept
    {
        return committed_prefix_end_;
    }

    [[nodiscard]] std::uint64_t committed_transactions() const noexcept
    {
        return committed_transactions_;
    }

    [[nodiscard]] std::uint64_t last_transaction_id() const noexcept
    {
        return last_transaction_id_;
    }

    [[nodiscard]] std::uint64_t data_items() const noexcept
    {
        return data_items_;
    }

    [[nodiscard]] std::uint64_t control_items() const noexcept
    {
        return control_items_;
    }

    [[nodiscard]] bool session_end_present() const noexcept
    {
        return session_end_present_;
    }

    /// The capture outcome the session-end record froze. Absent when no
    /// session-end record was committed -- which is what `unknown` means, and
    /// why the format has no value for it.
    [[nodiscard]] bool has_capture_outcome() const noexcept
    {
        return has_capture_outcome_;
    }

    [[nodiscard]] CaptureOutcome capture_outcome() const noexcept
    {
        return capture_outcome_;
    }

    [[nodiscard]] bool has_requested_terminal_intent() const noexcept
    {
        return has_requested_terminal_intent_;
    }

    [[nodiscard]] RequestedTerminalIntent requested_terminal_intent() const noexcept
    {
        return requested_terminal_intent_;
    }

    [[nodiscard]] bool has_primary_fault_committed() const noexcept
    {
        return has_primary_fault_committed_;
    }

    [[nodiscard]] bool primary_fault_committed() const noexcept
    {
        return primary_fault_committed_;
    }

    /// Whether the session-end record was decoded at all. A record of the
    /// wrong size is reported as present and left undecoded rather than read
    /// out of a payload this version does not define.
    [[nodiscard]] bool has_terminal_reason() const noexcept
    {
        return has_terminal_reason_;
    }

    [[nodiscard]] std::string_view terminal_reason() const noexcept
    {
        return {terminal_reason_.data(), terminal_reason_length_};
    }

    [[nodiscard]] bool has_durability_policy() const noexcept
    {
        return has_durability_policy_;
    }

    [[nodiscard]] DurabilityPolicy durability_policy() const noexcept
    {
        return durability_policy_;
    }

    /// How far synchronization was acknowledged, derived from the declared
    /// policy. An extent, never a count of records.
    [[nodiscard]] std::uint64_t durable_extent_bytes() const noexcept
    {
        return durable_extent_bytes_;
    }

    /// The end time the session-end record froze. Meaningful only when
    /// `has_terminal_reason()` -- a record this version could not decode has no
    /// time either, and zero is a legal instant.
    [[nodiscard]] std::uint64_t session_end_unix_nanos() const noexcept
    {
        return session_end_unix_nanos_;
    }

    [[nodiscard]] std::string_view session_id() const noexcept
    {
        return {session_id_.data(), session_id_length_};
    }

    /// The spool's minor version. Read and deliberately not acted on here: a
    /// minor may only add record kinds, so an unknown one is refused where such
    /// a kind is met rather than at the superblock.
    [[nodiscard]] std::uint16_t version_minor() const noexcept
    {
        return version_minor_;
    }

    [[nodiscard]] const std::array<std::uint8_t, kSessionUuidBytes>& session_uuid() const noexcept
    {
        return session_uuid_;
    }

    [[nodiscard]] std::uint64_t created_unix_nanos() const noexcept
    {
        return created_unix_nanos_;
    }

    /// Where the plan document begins, and how long it is. The document itself
    /// is not held: it is unbounded, and this report is a fixed-size value that
    /// allocates nothing. A caller that wants it reads that range from the same
    /// file, whose bytes the scan has already checked against the stored CRC
    /// and fingerprint.
    [[nodiscard]] std::uint64_t plan_document_offset() const noexcept
    {
        return plan_document_offset_;
    }

    [[nodiscard]] std::uint32_t plan_document_bytes() const noexcept
    {
        return plan_document_bytes_;
    }

    /// Whether an accounting snapshot was committed and decoded.
    [[nodiscard]] bool has_accounting() const noexcept
    {
        return has_accounting_;
    }

    [[nodiscard]] const SpoolAccountingSnapshot& accounting() const noexcept
    {
        return accounting_;
    }

    [[nodiscard]] const std::array<std::uint8_t, kPlanFingerprintBytes>&
    plan_fingerprint() const noexcept
    {
        return plan_fingerprint_;
    }

    [[nodiscard]] std::uint64_t first_transaction_offset() const noexcept
    {
        return first_transaction_offset_;
    }

    [[nodiscard]] std::span<const SpoolFinding> findings() const noexcept
    {
        return {findings_.data(), finding_count_};
    }

    /// True when the scanner had more to say than `kMaxFindings` allows.
    [[nodiscard]] bool findings_truncated() const noexcept
    {
        return findings_truncated_;
    }

    [[nodiscard]] bool has_code(SpoolCode code) const noexcept;

    /// Whether the superblock validated and a committed prefix exists.
    [[nodiscard]] bool readable() const noexcept
    {
        return status_ != ScanStatus::rejected;
    }

    /// Whether a finalizer may promote this committed prefix. A torn or
    /// corrupt tail does not block it -- that is the ordinary crash path. A
    /// conformance finding does, as does an incomplete read or a truncated
    /// finding set: promoting a prefix a reader did not fully establish is
    /// exactly the silent loss the contract does not tolerate.
    [[nodiscard]] bool finalizable() const noexcept;

    /// True when the scan could not read bytes the file claims to have. A
    /// read failure is not a verdict about the spool, so it is reported apart
    /// from the diagnostics.
    [[nodiscard]] bool read_failed() const noexcept
    {
        return read_failed_;
    }

  private:
    friend class SpoolScanner;

    ScanStatus status_{ScanStatus::ok};
    std::uint64_t committed_prefix_end_{};
    std::uint64_t committed_transactions_{};
    std::uint64_t last_transaction_id_{};
    std::uint64_t data_items_{};
    std::uint64_t control_items_{};
    bool session_end_present_{};
    bool has_capture_outcome_{};
    CaptureOutcome capture_outcome_{CaptureOutcome::normal};
    bool has_requested_terminal_intent_{};
    RequestedTerminalIntent requested_terminal_intent_{RequestedTerminalIntent::normal};
    bool has_primary_fault_committed_{};
    bool primary_fault_committed_{};
    bool has_terminal_reason_{};
    std::array<char, kTerminalReasonBytes> terminal_reason_{};
    std::size_t terminal_reason_length_{};
    std::uint64_t session_end_unix_nanos_{};
    bool has_durability_policy_{};
    DurabilityPolicy durability_policy_{DurabilityPolicy::transaction_sync};
    std::uint64_t durable_extent_bytes_{};
    std::array<char, kSessionIdBytes> session_id_{};
    std::size_t session_id_length_{};
    std::uint16_t version_minor_{};
    std::array<std::uint8_t, kSessionUuidBytes> session_uuid_{};
    std::uint64_t created_unix_nanos_{};
    std::uint64_t plan_document_offset_{};
    std::uint32_t plan_document_bytes_{};
    std::array<std::uint8_t, kPlanFingerprintBytes> plan_fingerprint_{};
    std::uint64_t first_transaction_offset_{};
    bool has_accounting_{};
    SpoolAccountingSnapshot accounting_{};
    std::array<SpoolFinding, kMaxFindings> findings_{};
    std::size_t finding_count_{};
    bool findings_truncated_{};
    bool read_failed_{};
};

/// One committed record, as the cursor reports it.
struct SpoolRecordView
{
    std::uint64_t transaction_id{};
    std::uint64_t transaction_offset{};
    std::uint64_t record_offset{};
    std::uint64_t payload_offset{};
    std::uint16_t kind{};
    std::uint32_t payload_bytes{};
    std::uint64_t logical_ordinal{};
    std::uint64_t record_unix_nanos{};
};

/// Walks the committed records of a scanned spool, in the order they were
/// committed. It allocates nothing and verifies nothing: the report it is
/// built from already established that this prefix is framed and checksummed,
/// and it must have been produced from the same bytes.
class SpoolRecordCursor
{
  public:
    SpoolRecordCursor(SpoolFile& file, const SpoolScanReport& report) noexcept;

    /// Advance to the next committed record. Returns false at the end of the
    /// committed prefix, and also on a read failure -- check `failed()`.
    [[nodiscard]] bool next(SpoolRecordView& out) noexcept;

    /// Read the payload of *view* into *out*, which must be exactly
    /// `view.payload_bytes` long.
    [[nodiscard]] bool read_payload(const SpoolRecordView& view, std::span<std::byte> out) noexcept;

    [[nodiscard]] bool failed() const noexcept
    {
        return failed_;
    }

  private:
    SpoolFile* file_{};
    std::uint64_t committed_prefix_end_{};
    std::uint64_t transaction_offset_{};
    std::uint64_t transaction_id_{};
    std::uint64_t cursor_{};
    std::uint32_t records_remaining_{};
    bool failed_{};
};

/// The read-only scan of specification section 6.
class SpoolScanner
{
  public:
    /// The scratch buffer must be at least this long. It bounds how much of a
    /// record payload the scanner holds at once; nothing here is sized by the
    /// file it is reading.
    static constexpr std::size_t kMinimumScratchBytes = 512;

    /// Scan *file*. *scratch* is borrowed for the duration of the call.
    [[nodiscard]] SpoolScanReport scan(SpoolFile& file, std::span<std::byte> scratch);

  private:
    struct FramedTransaction
    {
        std::uint64_t end_offset{};
        std::uint64_t records_offset{};
        std::uint32_t n_records{};
        std::uint32_t data_items{};
        std::uint32_t control_items{};
    };

    struct RecordHeaderFields
    {
        std::uint16_t kind{};
        std::uint16_t flags{};
        std::uint32_t payload_bytes{};
        std::uint64_t logical_ordinal{};
        std::uint64_t record_unix_nanos{};
        std::uint32_t payload_crc32c{};
    };

    void add_finding(SpoolCode code, std::uint64_t offset) noexcept;
    [[nodiscard]] bool read_exact(std::uint64_t offset, std::span<std::byte> out) noexcept;
    [[nodiscard]] bool scan_superblock() noexcept;
    [[nodiscard]] bool frame_transaction(std::uint64_t offset, std::uint64_t expected_id,
                                         std::uint64_t previous_offset,
                                         FramedTransaction& out) noexcept;
    [[nodiscard]] bool frame_record(std::uint64_t offset, std::uint32_t& body_crc,
                                    RecordHeaderFields& fields,
                                    std::uint64_t& next_offset) noexcept;
    [[nodiscard]] bool apply_semantics(std::uint64_t transaction_offset,
                                       const FramedTransaction& framed) noexcept;
    void check_accounting_payload(std::span<const std::byte> payload,
                                  std::uint64_t offset) noexcept;
    void check_accounting_against_prefix(std::span<const std::byte> payload,
                                         std::uint64_t offset) noexcept;
    void decode_session_end(std::span<const std::byte> payload, std::uint64_t offset) noexcept;
    [[nodiscard]] std::uint64_t derive_durable_extent() const noexcept;

    SpoolFile* file_{};
    std::span<std::byte> scratch_{};
    std::uint64_t file_size_{};
    SpoolScanReport report_{};
    std::uint64_t session_end_count_{};
    std::uint64_t accounting_count_{};
    std::uint64_t checkpoint_extent_{};

    /// Accounting payloads found in the transaction being scanned, held only
    /// until layer 2 can be checked against the totals that transaction
    /// produced. A conforming spool has one; a second is already reported as a
    /// violation, so two slots is all the evidence a reader needs.
    std::array<std::array<std::byte, kAccountingPayloadBytes>, 2> pending_accounting_{};
    std::array<std::uint64_t, 2> pending_accounting_offset_{};
    std::size_t pending_accounting_count_{};
};

/// Convenience wrapper for callers that have no scratch buffer of their own.
/// It allocates one, which is why the scanner itself does not.
[[nodiscard]] SpoolScanReport scan_spool(SpoolFile& file);

} // namespace neurale::recording
