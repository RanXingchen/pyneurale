/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "spool_layout.h"

namespace neurale::recording
{

std::string_view spool_code_text(SpoolCode code) noexcept
{
    switch (code)
    {
    case SpoolCode::none:
        return "";
    case SpoolCode::bad_magic:
        return "SPOOL-001";
    case SpoolCode::unsupported_major:
        return "SPOOL-002";
    case SpoolCode::superblock_checksum:
        return "SPOOL-003";
    case SpoolCode::truncated_superblock:
        return "SPOOL-004";
    case SpoolCode::plan_mismatch:
        return "SPOOL-005";
    case SpoolCode::superblock_field:
        return "SPOOL-006";
    case SpoolCode::torn_tail:
        return "SPOOL-010";
    case SpoolCode::transaction_header_checksum:
        return "SPOOL-011";
    case SpoolCode::transaction_trailer_checksum:
        return "SPOOL-012";
    case SpoolCode::transaction_body_checksum:
        return "SPOOL-013";
    case SpoolCode::transaction_id_sequence:
        return "SPOOL-014";
    case SpoolCode::back_link:
        return "SPOOL-015";
    case SpoolCode::n_records:
        return "SPOOL-016";
    case SpoolCode::record_header_checksum:
        return "SPOOL-017";
    case SpoolCode::record_payload_checksum:
        return "SPOOL-018";
    case SpoolCode::nonzero_padding:
        return "SPOOL-019";
    case SpoolCode::empty_transaction:
        return "SPOOL-020";
    case SpoolCode::trailer_item_counts:
        return "SPOOL-021";
    case SpoolCode::unknown_record_kind:
        return "SPOOL-030";
    case SpoolCode::records_after_session_end:
        return "SPOOL-031";
    case SpoolCode::session_end_without_accounting:
        return "SPOOL-032";
    case SpoolCode::accounting_transaction_not_quiet:
        return "SPOOL-033";
    case SpoolCode::accounting_identity:
        return "SPOOL-034";
    case SpoolCode::accounting_exceeds_prefix:
        return "SPOOL-035";
    case SpoolCode::checkpoint_extent:
        return "SPOOL-036";
    case SpoolCode::accounting_origin:
        return "SPOOL-037";
    case SpoolCode::container_payload_size:
        return "SPOOL-038";
    case SpoolCode::ordinal:
        return "SPOOL-039";
    case SpoolCode::flags_nonzero:
        return "SPOOL-040";
    case SpoolCode::session_end_field:
        return "SPOOL-041";
    case SpoolCode::checkpoint_policy:
        return "SPOOL-042";
    case SpoolCode::accounting_flag:
        return "SPOOL-043";
    case SpoolCode::position_tag:
        return "SPOOL-044";
    case SpoolCode::position_consistency:
        return "SPOOL-045";
    case SpoolCode::multiple_accounting:
        return "SPOOL-046";
    case SpoolCode::multiple_session_end:
        return "SPOOL-047";
    case SpoolCode::accounting_without_session_end:
        return "SPOOL-048";
    case SpoolCode::corrupt_transaction_header:
        return "SPOOL-049";
    case SpoolCode::position_tag_form:
        return "SPOOL-050";
    case SpoolCode::identity_kind:
        return "SPOOL-051";
    }
    return "";
}

std::string_view scan_status_text(ScanStatus status) noexcept
{
    switch (status)
    {
    case ScanStatus::ok:
        return "ok";
    case ScanStatus::torn_tail:
        return "torn_tail";
    case ScanStatus::corrupt_tail:
        return "corrupt_tail";
    case ScanStatus::rejected:
        return "rejected";
    }
    return "";
}

std::string_view durability_policy_text(DurabilityPolicy policy) noexcept
{
    switch (policy)
    {
    case DurabilityPolicy::buffered:
        return "buffered";
    case DurabilityPolicy::checkpoint_sync:
        return "checkpoint_sync";
    case DurabilityPolicy::transaction_sync:
        return "transaction_sync";
    }
    return "";
}

std::string_view capture_outcome_text(CaptureOutcome outcome) noexcept
{
    switch (outcome)
    {
    case CaptureOutcome::normal:
        return "normal";
    case CaptureOutcome::aborted:
        return "aborted";
    case CaptureOutcome::faulted:
        return "faulted";
    }
    return "";
}

std::string_view requested_terminal_intent_text(RequestedTerminalIntent intent) noexcept
{
    switch (intent)
    {
    case RequestedTerminalIntent::normal:
        return "normal";
    case RequestedTerminalIntent::aborted:
        return "aborted";
    case RequestedTerminalIntent::fault:
        return "fault";
    }
    return "";
}

} // namespace neurale::recording
