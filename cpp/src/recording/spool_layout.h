/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// Field offsets and constants of the private native spool, version 1.
///
/// The normative source is `specifications/native-spool/v1/README.md`. Every
/// value here is a byte offset or a byte count copied from a table in that
/// document, never `sizeof` anything: the container layout is defined by the
/// specification, and a writer that emitted the memory image of a C++ struct
/// would be conforming only by accident.

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace neurale::recording
{

/// Alignment every structure in the container starts at.
inline constexpr std::size_t kSpoolAlignment = 8;

inline constexpr std::size_t kSuperblockBytes = 256;
inline constexpr std::size_t kTransactionHeaderBytes = 48;
inline constexpr std::size_t kTransactionTrailerBytes = 48;
inline constexpr std::size_t kRecordHeaderBytes = 32;

inline constexpr std::size_t kAccountingPayloadBytes = 208;
inline constexpr std::size_t kTerminalReasonBytes = 64;
inline constexpr std::size_t kSessionEndPayloadBytes = 8 + kTerminalReasonBytes + 8;
inline constexpr std::size_t kCheckpointPayloadBytes = 24;
inline constexpr std::size_t kPositionBytes = 24;

inline constexpr std::size_t kSessionIdBytes = 128;
inline constexpr std::size_t kSessionUuidBytes = 16;
inline constexpr std::size_t kPlanFingerprintBytes = 32;

inline constexpr std::uint16_t kVersionMajor = 1;
/// 1.1 adds `task_variables` to the producer-identity registry and nothing
/// else. Section 9 allows exactly that under a minor: no table in sections 2,
/// 3, or 4 changed a field, an offset, or a size, so a 1.0 reader reads a 1.1
/// container normally and only meets the new kind if a task-variable record was
/// the first one refused.
inline constexpr std::uint16_t kVersionMinor = 1;

inline constexpr std::uint8_t kChecksumCrc32c = 1;
inline constexpr std::uint8_t kPlanEncodingRecordingPlanJcs = 1;

inline constexpr char kSpoolMagic[8] = {'N', 'R', 'L', 'S', 'P', 'O', 'O', 'L'};
inline constexpr char kTransactionBeginMagic[8] = {'N', 'N', 'S', 'T', 'X', 'B', 'E', 'G'};
inline constexpr char kTransactionEndMagic[8] = {'N', 'N', 'S', 'T', 'X', 'E', 'N', 'D'};

/// Superblock field offsets (specification section 2.1).
namespace superblock
{
inline constexpr std::size_t kMagic = 0;
inline constexpr std::size_t kVersionMajorOffset = 8;
inline constexpr std::size_t kVersionMinorOffset = 10;
inline constexpr std::size_t kSuperblockBytesOffset = 12;
inline constexpr std::size_t kDurabilityPolicy = 16;
inline constexpr std::size_t kChecksumAlgorithm = 17;
inline constexpr std::size_t kPlanEncoding = 18;
inline constexpr std::size_t kAlignment = 20;
inline constexpr std::size_t kPlanBytes = 24;
inline constexpr std::size_t kPlanCrc32c = 28;
inline constexpr std::size_t kPlanFingerprint = 32;
inline constexpr std::size_t kSessionUuid = 64;
inline constexpr std::size_t kCreatedUnixNanos = 80;
inline constexpr std::size_t kFirstTransactionOffset = 88;
inline constexpr std::size_t kSessionId = 96;
inline constexpr std::size_t kCrc32c = 252;
} // namespace superblock

/// Transaction header field offsets (specification section 2.2).
namespace transaction_header
{
inline constexpr std::size_t kMagic = 0;
inline constexpr std::size_t kTransactionId = 8;
inline constexpr std::size_t kPreviousTransactionOffset = 16;
inline constexpr std::size_t kRecordCount = 24;
inline constexpr std::size_t kHeaderBytes = 28;
inline constexpr std::size_t kBeginUnixNanos = 32;
inline constexpr std::size_t kCrc32c = 44;
} // namespace transaction_header

/// Commit trailer field offsets (specification section 2.2).
namespace transaction_trailer
{
inline constexpr std::size_t kMagic = 0;
inline constexpr std::size_t kTransactionId = 8;
inline constexpr std::size_t kBodyBytes = 16;
inline constexpr std::size_t kRecordCount = 24;
inline constexpr std::size_t kDataItems = 28;
inline constexpr std::size_t kControlItems = 32;
inline constexpr std::size_t kBodyCrc32c = 36;
inline constexpr std::size_t kFlags = 40;
inline constexpr std::size_t kCrc32c = 44;
} // namespace transaction_trailer

/// Record header field offsets (specification section 2.3).
namespace record_header
{
inline constexpr std::size_t kRecordKind = 0;
inline constexpr std::size_t kRecordFlags = 2;
inline constexpr std::size_t kPayloadBytes = 4;
inline constexpr std::size_t kLogicalOrdinal = 8;
inline constexpr std::size_t kRecordUnixNanos = 16;
inline constexpr std::size_t kPayloadCrc32c = 24;
inline constexpr std::size_t kCrc32c = 28;
} // namespace record_header

/// Accounting payload field offsets (specification section 4.4).
namespace accounting
{
inline constexpr std::size_t kRuntimeAccepted = 0;
inline constexpr std::size_t kRecorderAccepted = 8;
inline constexpr std::size_t kSpoolCommitted = 16;
inline constexpr std::size_t kRejectedBeforeRuntimeAcceptance = 24;
inline constexpr std::size_t kFailedBetweenRuntimeAndRecorder = 32;
inline constexpr std::size_t kLostBetweenRecorderAndSpool = 40;
inline constexpr std::size_t kControlOffered = 48;
inline constexpr std::size_t kControlAccepted = 56;
inline constexpr std::size_t kControlSpoolCommitted = 64;
inline constexpr std::size_t kControlRejected = 72;
inline constexpr std::size_t kLostBetweenControlAcceptanceAndSpool = 80;
inline constexpr std::size_t kRejectedAfterCloseData = 88;
inline constexpr std::size_t kRejectedAfterCloseControl = 96;
inline constexpr std::size_t kControlOfferedPresent = 104;
inline constexpr std::size_t kProducerAcceptanceKnown = 105;
inline constexpr std::size_t kAccountingOrigin = 106;
inline constexpr std::size_t kDataFirstLoss = 112;
inline constexpr std::size_t kDataFirstRejection = 136;
inline constexpr std::size_t kControlFirstLoss = 160;
inline constexpr std::size_t kControlFirstRejection = 184;
} // namespace accounting

/// Position field offsets within a 24-byte position (specification section 4.4).
namespace position
{
inline constexpr std::size_t kTag = 0;
inline constexpr std::size_t kIdentityKind = 4;
inline constexpr std::size_t kOrdinal = 8;
inline constexpr std::size_t kIdentityValue = 16;
} // namespace position

/// Session-end payload field offsets (specification section 4.5).
namespace session_end
{
inline constexpr std::size_t kRequestedTerminalIntent = 0;
inline constexpr std::size_t kCaptureOutcome = 1;
inline constexpr std::size_t kPrimaryFaultCommitted = 2;
inline constexpr std::size_t kTerminalReason = 8;
inline constexpr std::size_t kEndUnixNanos = 72;
} // namespace session_end

/// Checkpoint payload field offsets (specification section 4.6).
namespace checkpoint
{
inline constexpr std::size_t kDurableExtentBytes = 0;
inline constexpr std::size_t kSyncedUnixNanos = 8;
inline constexpr std::size_t kDurabilityPolicy = 16;
} // namespace checkpoint

/// The three spool durability policies, named as the lifecycle contract names
/// them (contract section 6, specification section 5).
enum class DurabilityPolicy : std::uint8_t
{
    buffered = 0,
    checkpoint_sync = 1,
    transaction_sync = 2,
};

/// Record kinds (specification section 4.1).
enum class RecordKind : std::uint16_t
{
    frame = 1,
    signal_block = 2,
    discontinuity = 3,
    signal_gap = 4,
    control = 5,
    fault = 6,
    accounting_snapshot = 7,
    session_end = 8,
    checkpoint = 9,
};

/// Capture outcome frozen by the session-end record (contract section 3.2).
/// There is no `unknown` value on purpose: `unknown` is what a *missing*
/// session-end record means, and a writer may not state it.
enum class CaptureOutcome : std::uint8_t
{
    normal = 0,
    aborted = 1,
    faulted = 2,
};

/// What the first exit from `recording` latched (contract section 3.2).
enum class RequestedTerminalIntent : std::uint8_t
{
    normal = 0,
    aborted = 1,
    fault = 2,
};

/// Tag of a first-failed-or-lost position (contract section 5.1).
enum class PositionTag : std::uint8_t
{
    absent = 0,
    ordinal = 1,
    producer_identity = 2,
};

/// The producer-identity registry of specification section 4.4. The number is
/// the only bridge between a spool position and the NRF kind string, so the
/// values are fixed here and nowhere else.
enum class ProducerIdentityKind : std::uint32_t
{
    none = 0,
    frame = 1,
    discontinuity = 2,
    events = 3,
    trials = 4,
    experiment_states = 5,
    commands = 6,
    targets = 7,
    labels = 8,
    assistance = 9,
    faults = 10,
    /// Added in container version 1.1. The Python `SessionRecorder` already
    /// records task variables, so a native path without this number could not
    /// express one of its own control kinds -- and a kind the registry cannot
    /// name is a kind whose first refusal has no expressible position.
    task_variables = 11,
};

/// Diagnostic codes (specification section 11). The enumerator value is the
/// number in the code text: wording may change, a code may not.
enum class SpoolCode : std::uint16_t
{
    none = 0,
    bad_magic = 1,
    unsupported_major = 2,
    superblock_checksum = 3,
    truncated_superblock = 4,
    plan_mismatch = 5,
    superblock_field = 6,
    torn_tail = 10,
    transaction_header_checksum = 11,
    transaction_trailer_checksum = 12,
    transaction_body_checksum = 13,
    transaction_id_sequence = 14,
    back_link = 15,
    n_records = 16,
    record_header_checksum = 17,
    record_payload_checksum = 18,
    nonzero_padding = 19,
    empty_transaction = 20,
    trailer_item_counts = 21,
    unknown_record_kind = 30,
    records_after_session_end = 31,
    session_end_without_accounting = 32,
    accounting_transaction_not_quiet = 33,
    accounting_identity = 34,
    accounting_exceeds_prefix = 35,
    checkpoint_extent = 36,
    accounting_origin = 37,
    container_payload_size = 38,
    ordinal = 39,
    flags_nonzero = 40,
    session_end_field = 41,
    checkpoint_policy = 42,
    accounting_flag = 43,
    position_tag = 44,
    position_consistency = 45,
    multiple_accounting = 46,
    multiple_session_end = 47,
    accounting_without_session_end = 48,
    corrupt_transaction_header = 49,
    position_tag_form = 50,
    identity_kind = 51,
};

/// Status of a completed scan (specification section 6.1).
enum class ScanStatus : std::uint8_t
{
    ok = 0,
    torn_tail = 1,
    corrupt_tail = 2,
    rejected = 3,
};

/// Whether *code* ends the committed prefix where it is found. A torn or
/// corrupt tail ends the prefix without blocking finalization: recovery starts
/// from the last valid committed transaction, which is the ordinary crash path.
[[nodiscard]] constexpr bool ends_committed_prefix(SpoolCode code) noexcept
{
    switch (code)
    {
    case SpoolCode::torn_tail:
    case SpoolCode::transaction_header_checksum:
    case SpoolCode::transaction_trailer_checksum:
    case SpoolCode::transaction_body_checksum:
    case SpoolCode::transaction_id_sequence:
    case SpoolCode::back_link:
    case SpoolCode::n_records:
    case SpoolCode::record_header_checksum:
    case SpoolCode::record_payload_checksum:
    case SpoolCode::nonzero_padding:
    case SpoolCode::empty_transaction:
    case SpoolCode::trailer_item_counts:
    case SpoolCode::corrupt_transaction_header:
        return true;
    default:
        return false;
    }
}

/// Whether *kind* is a data-plane item for the counts of specification 4.3.
[[nodiscard]] constexpr bool is_data_plane_item(std::uint16_t kind) noexcept
{
    return kind == static_cast<std::uint16_t>(RecordKind::frame) ||
           kind == static_cast<std::uint16_t>(RecordKind::discontinuity);
}

/// Whether *kind* is a control-plane item for the counts of specification 4.3.
[[nodiscard]] constexpr bool is_control_plane_item(std::uint16_t kind) noexcept
{
    return kind == static_cast<std::uint16_t>(RecordKind::control);
}

/// Whether *kind* is one this container version defines.
[[nodiscard]] constexpr bool is_known_record_kind(std::uint16_t kind) noexcept
{
    return kind >= static_cast<std::uint16_t>(RecordKind::frame) &&
           kind <= static_cast<std::uint16_t>(RecordKind::checkpoint);
}

/// Whether *kind* carries no owning item and therefore MUST have a
/// `logical_ordinal` of 0 (specification section 2.3).
[[nodiscard]] constexpr bool has_no_owning_item(std::uint16_t kind) noexcept
{
    return kind == static_cast<std::uint16_t>(RecordKind::fault) ||
           kind == static_cast<std::uint16_t>(RecordKind::accounting_snapshot) ||
           kind == static_cast<std::uint16_t>(RecordKind::session_end) ||
           kind == static_cast<std::uint16_t>(RecordKind::checkpoint);
}

/// Whether the container owns the payload of *kind*, and therefore fixes its
/// size (specification sections 4.4 to 4.6).
[[nodiscard]] constexpr bool is_container_owned_kind(std::uint16_t kind) noexcept
{
    return kind == static_cast<std::uint16_t>(RecordKind::accounting_snapshot) ||
           kind == static_cast<std::uint16_t>(RecordKind::session_end) ||
           kind == static_cast<std::uint16_t>(RecordKind::checkpoint);
}

/// The fixed payload size of a container-owned *kind*, or 0 for a kind whose
/// payload the container carries opaquely.
[[nodiscard]] constexpr std::size_t container_owned_payload_bytes(std::uint16_t kind) noexcept
{
    switch (static_cast<RecordKind>(kind))
    {
    case RecordKind::accounting_snapshot:
        return kAccountingPayloadBytes;
    case RecordKind::session_end:
        return kSessionEndPayloadBytes;
    case RecordKind::checkpoint:
        return kCheckpointPayloadBytes;
    default:
        return 0;
    }
}

/// Round *length* up to the container alignment.
[[nodiscard]] constexpr std::uint64_t spool_padded_length(std::uint64_t length) noexcept
{
    const auto remainder = length % kSpoolAlignment;
    return remainder == 0 ? length : length + (kSpoolAlignment - remainder);
}

/// The stable text of *code*, as it appears in the specification and in the
/// vector index (`SPOOL-001` and so on).
[[nodiscard]] std::string_view spool_code_text(SpoolCode code) noexcept;

/// The name of *status*, as the vector index spells it.
[[nodiscard]] std::string_view scan_status_text(ScanStatus status) noexcept;

/// The name of *policy*, as the vector index and the lifecycle contract spell it.
[[nodiscard]] std::string_view durability_policy_text(DurabilityPolicy policy) noexcept;

/// The name of *outcome*, as the vector index spells it.
[[nodiscard]] std::string_view capture_outcome_text(CaptureOutcome outcome) noexcept;

/// The name of *intent*, as the vector index spells it.
[[nodiscard]] std::string_view
requested_terminal_intent_text(RequestedTerminalIntent intent) noexcept;

} // namespace neurale::recording
