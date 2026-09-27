/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "spool_scanner.h"

#include "byte_order.h"
#include "crc32c.h"
#include "sha256.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace neurale::recording
{
namespace
{

/// The four first-failed-or-lost positions, with the tag form the contract
/// fixes for each, the counters whose first event it names, and whether it
/// belongs to the data plane (which decides the identity-kind partition).
struct PositionRule
{
    std::size_t base;
    PositionTag required_tag;
    std::size_t counter_a;
    std::size_t counter_b; // equal to counter_a when the position names one counter
    bool data_plane;
    bool checks_identity_kind;
    /// Where a retained copy of this position goes. Carried in the rule so the
    /// table stays the one place that pairs a payload offset with the position
    /// it holds -- a separate list would be a second ordering to keep in step.
    ScannedAccountingPosition SpoolAccountingSnapshot::* slot;
};

constexpr std::array<PositionRule, 4> kPositionRules = {
    PositionRule{accounting::kDataFirstLoss, PositionTag::ordinal,
                 accounting::kFailedBetweenRuntimeAndRecorder,
                 accounting::kLostBetweenRecorderAndSpool, true, false,
                 &SpoolAccountingSnapshot::data_first_loss},
    PositionRule{accounting::kDataFirstRejection, PositionTag::producer_identity,
                 accounting::kRejectedBeforeRuntimeAcceptance,
                 accounting::kRejectedBeforeRuntimeAcceptance, true, true,
                 &SpoolAccountingSnapshot::data_first_rejection},
    PositionRule{accounting::kControlFirstLoss, PositionTag::ordinal,
                 accounting::kLostBetweenControlAcceptanceAndSpool,
                 accounting::kLostBetweenControlAcceptanceAndSpool, false, false,
                 &SpoolAccountingSnapshot::control_first_loss},
    PositionRule{accounting::kControlFirstRejection, PositionTag::producer_identity,
                 accounting::kControlRejected, accounting::kControlRejected, false, true,
                 &SpoolAccountingSnapshot::control_first_rejection},
};

[[nodiscard]] bool identity_kind_is_registered(std::uint32_t kind, bool data_plane) noexcept
{
    if (data_plane)
    {
        return kind == 1 || kind == 2;
    }
    return kind >= 3 && kind <= 11;
}

[[nodiscard]] std::size_t trim_nuls(const char* data, std::size_t size) noexcept
{
    while (size > 0 && data[size - 1] == '\0')
    {
        --size;
    }
    return size;
}

} // namespace

bool SpoolScanReport::has_code(SpoolCode code) const noexcept
{
    for (std::size_t i = 0; i < finding_count_; ++i)
    {
        if (findings_[i].code == code)
        {
            return true;
        }
    }
    return false;
}

bool SpoolScanReport::finalizable() const noexcept
{
    if (read_failed_ || findings_truncated_ || status_ == ScanStatus::rejected)
    {
        return false;
    }
    for (std::size_t i = 0; i < finding_count_; ++i)
    {
        if (!ends_committed_prefix(findings_[i].code))
        {
            return false;
        }
    }
    return true;
}

void SpoolScanner::add_finding(SpoolCode code, std::uint64_t offset) noexcept
{
    if (report_.finding_count_ == SpoolScanReport::kMaxFindings)
    {
        report_.findings_truncated_ = true;
        return;
    }
    report_.findings_[report_.finding_count_] = SpoolFinding{.code = code, .offset = offset};
    ++report_.finding_count_;
}

bool SpoolScanner::read_exact(std::uint64_t offset, std::span<std::byte> out) noexcept
{
    const auto result = file_->read_at(offset, out);
    if (result.status != SpoolIoStatus::ok || result.transferred != out.size())
    {
        // A read failure is not a verdict about the spool's contents: it says
        // the reader could not look, not that what it would have seen is wrong.
        report_.read_failed_ = true;
        return false;
    }
    return true;
}

bool SpoolScanner::scan_superblock() noexcept
{
    if (file_size_ < kSuperblockBytes)
    {
        add_finding(SpoolCode::truncated_superblock, 0);
        return false;
    }
    auto header = scratch_.first(kSuperblockBytes);
    if (!read_exact(0, header))
    {
        return false;
    }
    if (std::memcmp(header.data(), kSpoolMagic, sizeof(kSpoolMagic)) != 0)
    {
        add_finding(SpoolCode::bad_magic, 0);
        return false;
    }
    const auto stored_crc = load_u32le(header, superblock::kCrc32c);
    if (crc32c(header.first(superblock::kCrc32c)) != stored_crc)
    {
        add_finding(SpoolCode::superblock_checksum, 0);
        return false;
    }

    // The minor version is read and deliberately not acted on: a minor may
    // only add record kinds, so an unknown one is refused where such a kind is
    // met, not here.
    const auto major = load_u16le(header, superblock::kVersionMajorOffset);
    if (major != kVersionMajor)
    {
        add_finding(SpoolCode::unsupported_major, superblock::kVersionMajorOffset);
        return false;
    }
    const auto superblock_bytes = load_u32le(header, superblock::kSuperblockBytesOffset);
    const auto policy = load_u8(header, superblock::kDurabilityPolicy);
    const auto checksum_algorithm = load_u8(header, superblock::kChecksumAlgorithm);
    const auto plan_encoding = load_u8(header, superblock::kPlanEncoding);
    const auto alignment = load_u32le(header, superblock::kAlignment);
    if (superblock_bytes != kSuperblockBytes || alignment != kSpoolAlignment ||
        checksum_algorithm != kChecksumCrc32c || plan_encoding != kPlanEncodingRecordingPlanJcs ||
        policy > static_cast<std::uint8_t>(DurabilityPolicy::transaction_sync))
    {
        add_finding(SpoolCode::superblock_field, superblock::kDurabilityPolicy);
        return false;
    }

    const auto plan_bytes = load_u32le(header, superblock::kPlanBytes);
    const auto plan_crc = load_u32le(header, superblock::kPlanCrc32c);
    std::array<std::uint8_t, kPlanFingerprintBytes> stored_fingerprint{};
    for (std::size_t i = 0; i < kPlanFingerprintBytes; ++i)
    {
        stored_fingerprint[i] = load_u8(header, superblock::kPlanFingerprint + i);
    }
    std::array<char, kSessionIdBytes> session_id{};
    for (std::size_t i = 0; i < kSessionIdBytes; ++i)
    {
        session_id[i] = static_cast<char>(load_u8(header, superblock::kSessionId + i));
    }
    const auto first_transaction_offset = load_u64le(header, superblock::kFirstTransactionOffset);
    // Read out of the header while it is still the header: `scratch_` is the
    // one buffer this scanner has, and the plan-digest loop below refills it
    // with plan bytes. Anything taken from `header` after that point is plan
    // text wearing a superblock field's name.
    const auto version_minor = load_u16le(header, superblock::kVersionMinorOffset);
    std::array<std::uint8_t, kSessionUuidBytes> session_uuid{};
    for (std::size_t i = 0; i < kSessionUuidBytes; ++i)
    {
        session_uuid[i] = load_u8(header, superblock::kSessionUuid + i);
    }
    const auto created_unix_nanos = load_u64le(header, superblock::kCreatedUnixNanos);

    if (first_transaction_offset != kSuperblockBytes + spool_padded_length(plan_bytes))
    {
        add_finding(SpoolCode::superblock_field, superblock::kFirstTransactionOffset);
        return false;
    }
    if (file_size_ < first_transaction_offset)
    {
        add_finding(SpoolCode::truncated_superblock, kSuperblockBytes);
        return false;
    }

    // No checksum reaches the padding behind the plan document -- the plan CRC
    // stops at plan_bytes and the superblock CRC at byte 252 -- so the zero
    // rule is the only thing covering that gap.
    const auto padding_offset = static_cast<std::uint64_t>(kSuperblockBytes) + plan_bytes;
    const auto padding_bytes = static_cast<std::size_t>(first_transaction_offset - padding_offset);
    if (padding_bytes != 0)
    {
        auto padding = scratch_.first(padding_bytes);
        if (!read_exact(padding_offset, padding))
        {
            return false;
        }
        for (const auto value : padding)
        {
            if (value != std::byte{0})
            {
                add_finding(SpoolCode::nonzero_padding, padding_offset);
                return false;
            }
        }
    }

    std::uint32_t plan_crc_actual = 0;
    Sha256 hasher;
    std::uint64_t remaining = plan_bytes;
    std::uint64_t cursor = kSuperblockBytes;
    while (remaining > 0)
    {
        const auto take = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, static_cast<std::uint64_t>(scratch_.size())));
        auto chunk = scratch_.first(take);
        if (!read_exact(cursor, chunk))
        {
            return false;
        }
        plan_crc_actual = crc32c(chunk, plan_crc_actual);
        hasher.update(chunk);
        cursor += take;
        remaining -= take;
    }
    const auto digest = hasher.finish();
    if (plan_crc_actual != plan_crc || digest != stored_fingerprint)
    {
        add_finding(SpoolCode::plan_mismatch, kSuperblockBytes);
        return false;
    }

    report_.has_durability_policy_ = true;
    report_.durability_policy_ = static_cast<DurabilityPolicy>(policy);
    report_.first_transaction_offset_ = first_transaction_offset;
    report_.plan_fingerprint_ = digest;
    report_.session_id_ = session_id;
    report_.session_id_length_ = trim_nuls(session_id.data(), session_id.size());
    report_.version_minor_ = version_minor;
    report_.session_uuid_ = session_uuid;
    report_.created_unix_nanos_ = created_unix_nanos;
    // The document itself is not copied here: the range is what a caller reads,
    // and the loop above has already checked those bytes against both the
    // stored CRC and the stored fingerprint.
    report_.plan_document_offset_ = kSuperblockBytes;
    report_.plan_document_bytes_ = plan_bytes;
    return true;
}

bool SpoolScanner::frame_record(std::uint64_t offset, std::uint32_t& body_crc,
                                RecordHeaderFields& fields, std::uint64_t& next_offset) noexcept
{
    if (file_size_ - offset < kRecordHeaderBytes)
    {
        add_finding(SpoolCode::torn_tail, offset);
        return false;
    }
    auto header = scratch_.first(kRecordHeaderBytes);
    if (!read_exact(offset, header))
    {
        return false;
    }
    const auto stored_crc = load_u32le(header, record_header::kCrc32c);
    if (crc32c(header.first(record_header::kCrc32c)) != stored_crc)
    {
        add_finding(SpoolCode::record_header_checksum, offset);
        return false;
    }
    fields.kind = load_u16le(header, record_header::kRecordKind);
    fields.flags = load_u16le(header, record_header::kRecordFlags);
    fields.payload_bytes = load_u32le(header, record_header::kPayloadBytes);
    fields.logical_ordinal = load_u64le(header, record_header::kLogicalOrdinal);
    fields.record_unix_nanos = load_u64le(header, record_header::kRecordUnixNanos);
    fields.payload_crc32c = load_u32le(header, record_header::kPayloadCrc32c);
    if (fields.flags != 0)
    {
        // Reserved for a future minor version: a reader reports a nonzero
        // value rather than ignoring it.
        add_finding(SpoolCode::flags_nonzero, offset);
    }
    body_crc = crc32c(header, body_crc);

    const auto payload_start = offset + kRecordHeaderBytes;
    const auto padded = spool_padded_length(fields.payload_bytes);
    if (file_size_ - payload_start < padded)
    {
        add_finding(SpoolCode::torn_tail, offset);
        return false;
    }

    std::uint32_t payload_crc = 0;
    std::uint64_t remaining = fields.payload_bytes;
    std::uint64_t cursor = payload_start;
    while (remaining > 0)
    {
        const auto take = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, static_cast<std::uint64_t>(scratch_.size())));
        auto chunk = scratch_.first(take);
        if (!read_exact(cursor, chunk))
        {
            return false;
        }
        payload_crc = crc32c(chunk, payload_crc);
        body_crc = crc32c(chunk, body_crc);
        cursor += take;
        remaining -= take;
    }
    if (payload_crc != fields.payload_crc32c)
    {
        add_finding(SpoolCode::record_payload_checksum, offset);
        return false;
    }

    const auto padding_bytes = static_cast<std::size_t>(padded - fields.payload_bytes);
    if (padding_bytes != 0)
    {
        auto padding = scratch_.first(padding_bytes);
        if (!read_exact(cursor, padding))
        {
            return false;
        }
        for (const auto value : padding)
        {
            if (value != std::byte{0})
            {
                add_finding(SpoolCode::nonzero_padding, offset);
                return false;
            }
        }
        body_crc = crc32c(padding, body_crc);
    }

    next_offset = payload_start + padded;
    return true;
}

bool SpoolScanner::frame_transaction(std::uint64_t offset, std::uint64_t expected_id,
                                     std::uint64_t previous_offset, FramedTransaction& out) noexcept
{
    if (file_size_ - offset < kTransactionHeaderBytes)
    {
        add_finding(SpoolCode::torn_tail, offset);
        return false;
    }
    auto header = scratch_.first(kTransactionHeaderBytes);
    if (!read_exact(offset, header))
    {
        return false;
    }
    if (std::memcmp(header.data(), kTransactionBeginMagic, sizeof(kTransactionBeginMagic)) != 0)
    {
        // A complete header that is present and wrong is corruption, not
        // truncation: the bytes are there, they just do not begin a
        // transaction.
        add_finding(SpoolCode::corrupt_transaction_header, offset);
        return false;
    }
    const auto stored_crc = load_u32le(header, transaction_header::kCrc32c);
    if (crc32c(header.first(transaction_header::kCrc32c)) != stored_crc)
    {
        add_finding(SpoolCode::transaction_header_checksum, offset);
        return false;
    }
    const auto transaction_id = load_u64le(header, transaction_header::kTransactionId);
    const auto back_link = load_u64le(header, transaction_header::kPreviousTransactionOffset);
    const auto declared_records = load_u32le(header, transaction_header::kRecordCount);
    const auto header_bytes = load_u32le(header, transaction_header::kHeaderBytes);
    if (transaction_id != expected_id)
    {
        add_finding(SpoolCode::transaction_id_sequence, offset);
        return false;
    }
    if (back_link != previous_offset)
    {
        add_finding(SpoolCode::back_link, offset);
        return false;
    }
    if (header_bytes != kTransactionHeaderBytes)
    {
        add_finding(SpoolCode::corrupt_transaction_header, offset);
        return false;
    }
    if (declared_records == 0)
    {
        add_finding(SpoolCode::empty_transaction, offset);
        return false;
    }

    auto body_crc = crc32c(header);
    auto cursor = offset + kTransactionHeaderBytes;
    std::uint32_t counted_data = 0;
    std::uint32_t counted_control = 0;
    for (std::uint32_t i = 0; i < declared_records; ++i)
    {
        RecordHeaderFields fields{};
        std::uint64_t next = 0;
        if (!frame_record(cursor, body_crc, fields, next))
        {
            return false;
        }
        if (is_data_plane_item(fields.kind))
        {
            ++counted_data;
        }
        if (is_control_plane_item(fields.kind))
        {
            ++counted_control;
        }
        cursor = next;
    }

    if (file_size_ - cursor < kTransactionTrailerBytes)
    {
        add_finding(SpoolCode::torn_tail, cursor);
        return false;
    }
    auto trailer = scratch_.first(kTransactionTrailerBytes);
    if (!read_exact(cursor, trailer))
    {
        return false;
    }
    if (std::memcmp(trailer.data(), kTransactionEndMagic, sizeof(kTransactionEndMagic)) != 0)
    {
        add_finding(SpoolCode::n_records, cursor);
        return false;
    }
    const auto stored_trailer_crc = load_u32le(trailer, transaction_trailer::kCrc32c);
    if (crc32c(trailer.first(transaction_trailer::kCrc32c)) != stored_trailer_crc)
    {
        add_finding(SpoolCode::transaction_trailer_checksum, cursor);
        return false;
    }
    const auto trailer_id = load_u64le(trailer, transaction_trailer::kTransactionId);
    const auto body_bytes = load_u64le(trailer, transaction_trailer::kBodyBytes);
    const auto trailer_records = load_u32le(trailer, transaction_trailer::kRecordCount);
    const auto data_items = load_u32le(trailer, transaction_trailer::kDataItems);
    const auto control_items = load_u32le(trailer, transaction_trailer::kControlItems);
    const auto stored_body_crc = load_u32le(trailer, transaction_trailer::kBodyCrc32c);
    const auto flags = load_u32le(trailer, transaction_trailer::kFlags);
    if (flags != 0)
    {
        add_finding(SpoolCode::flags_nonzero, cursor);
    }
    if (trailer_id != transaction_id || body_bytes != cursor - offset)
    {
        add_finding(SpoolCode::transaction_trailer_checksum, cursor);
        return false;
    }
    if (trailer_records != declared_records)
    {
        add_finding(SpoolCode::n_records, cursor);
        return false;
    }
    if (body_crc != stored_body_crc)
    {
        add_finding(SpoolCode::transaction_body_checksum, offset);
        return false;
    }
    if (data_items != counted_data || control_items != counted_control)
    {
        add_finding(SpoolCode::trailer_item_counts, cursor);
        return false;
    }

    out.end_offset = cursor + kTransactionTrailerBytes;
    out.records_offset = offset + kTransactionHeaderBytes;
    out.n_records = declared_records;
    out.data_items = data_items;
    out.control_items = control_items;
    return true;
}

void SpoolScanner::decode_session_end(std::span<const std::byte> payload,
                                      std::uint64_t offset) noexcept
{
    const auto requested = load_u8(payload, session_end::kRequestedTerminalIntent);
    const auto outcome = load_u8(payload, session_end::kCaptureOutcome);
    const auto primary_fault = load_u8(payload, session_end::kPrimaryFaultCommitted);

    if (requested <= 2)
    {
        report_.has_requested_terminal_intent_ = true;
        report_.requested_terminal_intent_ = static_cast<RequestedTerminalIntent>(requested);
    }
    else
    {
        add_finding(SpoolCode::session_end_field, offset);
    }
    if (outcome <= 2)
    {
        report_.has_capture_outcome_ = true;
        report_.capture_outcome_ = static_cast<CaptureOutcome>(outcome);
    }
    else
    {
        add_finding(SpoolCode::session_end_field, offset);
    }
    if (primary_fault <= 1)
    {
        report_.has_primary_fault_committed_ = true;
        report_.primary_fault_committed_ = primary_fault == 1;
    }
    else
    {
        add_finding(SpoolCode::session_end_field, offset);
    }

    for (std::size_t i = 0; i < kTerminalReasonBytes; ++i)
    {
        report_.terminal_reason_[i] =
            static_cast<char>(load_u8(payload, session_end::kTerminalReason + i));
    }
    report_.terminal_reason_length_ =
        trim_nuls(report_.terminal_reason_.data(), report_.terminal_reason_.size());
    report_.has_terminal_reason_ = true;
    report_.session_end_unix_nanos_ = load_u64le(payload, session_end::kEndUnixNanos);
}

void SpoolScanner::check_accounting_payload(std::span<const std::byte> payload,
                                            std::uint64_t offset) noexcept
{
    const auto control_offered_present = load_u8(payload, accounting::kControlOfferedPresent);
    const auto producer_acceptance_known = load_u8(payload, accounting::kProducerAcceptanceKnown);
    const auto origin = load_u8(payload, accounting::kAccountingOrigin);
    if (control_offered_present > 1)
    {
        add_finding(SpoolCode::accounting_flag, offset);
    }
    if (producer_acceptance_known > 1)
    {
        add_finding(SpoolCode::accounting_flag, offset);
    }
    if (origin != 0)
    {
        // Spool accounting is always written by the recorder; recovery-rebuilt
        // accounting belongs to the NRF session.
        add_finding(SpoolCode::accounting_origin, offset);
    }

    for (const auto& rule : kPositionRules)
    {
        const auto tag = load_u8(payload, rule.base + position::kTag);
        const auto identity_kind = load_u32le(payload, rule.base + position::kIdentityKind);
        const auto ordinal = load_u64le(payload, rule.base + position::kOrdinal);
        const auto identity_value = load_u64le(payload, rule.base + position::kIdentityValue);
        if (tag > static_cast<std::uint8_t>(PositionTag::producer_identity))
        {
            add_finding(SpoolCode::position_tag, offset);
            continue;
        }
        if (tag == static_cast<std::uint8_t>(PositionTag::ordinal) &&
            (identity_kind != 0 || identity_value != 0))
        {
            add_finding(SpoolCode::position_tag, offset);
        }
        else if (tag == static_cast<std::uint8_t>(PositionTag::producer_identity) && ordinal != 0)
        {
            add_finding(SpoolCode::position_tag, offset);
        }
        else if (tag == static_cast<std::uint8_t>(PositionTag::absent) &&
                 (identity_kind != 0 || ordinal != 0 || identity_value != 0))
        {
            add_finding(SpoolCode::position_tag, offset);
        }

        const auto present = tag != static_cast<std::uint8_t>(PositionTag::absent);
        // A loss after acceptance names the accepted item's ordinal; a
        // rejection before acceptance names the producer identity. Swapping
        // them would describe a different event than the one that happened.
        if (present && tag != static_cast<std::uint8_t>(rule.required_tag))
        {
            add_finding(SpoolCode::position_tag_form, offset);
        }
        if (rule.checks_identity_kind &&
            tag == static_cast<std::uint8_t>(PositionTag::producer_identity) &&
            !identity_kind_is_registered(identity_kind, rule.data_plane))
        {
            add_finding(SpoolCode::identity_kind, offset);
        }

        const bool related =
            load_u64le(payload, rule.counter_a) != 0 ||
            (rule.counter_b != rule.counter_a && load_u64le(payload, rule.counter_b) != 0);
        // The first loss or rejection is latched in bounded storage, so a
        // position is present exactly when at least one counter it names is
        // nonzero. This boolean form cannot wrap at UINT64_MAX.
        if (present && !related)
        {
            add_finding(SpoolCode::position_consistency, offset);
        }
        if (!present && related)
        {
            add_finding(SpoolCode::position_consistency, offset);
        }
    }

    const auto runtime_accepted = load_u64le(payload, accounting::kRuntimeAccepted);
    const auto recorder_accepted = load_u64le(payload, accounting::kRecorderAccepted);
    const auto spool_committed = load_u64le(payload, accounting::kSpoolCommitted);
    const auto failed_between = load_u64le(payload, accounting::kFailedBetweenRuntimeAndRecorder);
    const auto lost_between = load_u64le(payload, accounting::kLostBetweenRecorderAndSpool);
    const auto control_offered = load_u64le(payload, accounting::kControlOffered);
    const auto control_accepted = load_u64le(payload, accounting::kControlAccepted);
    const auto control_committed = load_u64le(payload, accounting::kControlSpoolCommitted);
    const auto control_rejected = load_u64le(payload, accounting::kControlRejected);
    const auto control_lost =
        load_u64le(payload, accounting::kLostBetweenControlAcceptanceAndSpool);

    if (recorder_accepted > runtime_accepted ||
        failed_between != runtime_accepted - recorder_accepted)
    {
        add_finding(SpoolCode::accounting_identity, offset);
    }
    if (spool_committed > recorder_accepted || lost_between != recorder_accepted - spool_committed)
    {
        add_finding(SpoolCode::accounting_identity, offset);
    }
    if (control_committed > control_accepted ||
        control_lost != control_accepted - control_committed)
    {
        add_finding(SpoolCode::accounting_identity, offset);
    }
    if (control_offered_present == 1 && (control_accepted > control_offered ||
                                         control_rejected != control_offered - control_accepted))
    {
        add_finding(SpoolCode::accounting_identity, offset);
    }

    // Retain the first snapshot only. A second is already reported as a
    // violation, and keeping the first means the numbers a caller reads are the
    // ones the checks above were applied to.
    if (report_.has_accounting_)
    {
        return;
    }
    report_.has_accounting_ = true;
    auto& kept = report_.accounting_;
    kept.runtime_accepted = runtime_accepted;
    kept.recorder_accepted = recorder_accepted;
    kept.spool_committed = spool_committed;
    kept.rejected_before_runtime_acceptance =
        load_u64le(payload, accounting::kRejectedBeforeRuntimeAcceptance);
    kept.failed_between_runtime_and_recorder = failed_between;
    kept.lost_between_recorder_and_spool = lost_between;
    kept.control_offered = control_offered;
    kept.control_accepted = control_accepted;
    kept.control_spool_committed = control_committed;
    kept.control_rejected = control_rejected;
    kept.lost_between_control_acceptance_and_spool = control_lost;
    kept.rejected_after_close_data = load_u64le(payload, accounting::kRejectedAfterCloseData);
    kept.rejected_after_close_control = load_u64le(payload, accounting::kRejectedAfterCloseControl);
    kept.control_offered_present = control_offered_present == 1;
    kept.producer_acceptance_known = producer_acceptance_known == 1;
    kept.accounting_origin = origin;
    for (const auto& rule : kPositionRules)
    {
        auto& slot = kept.*(rule.slot);
        slot.tag = static_cast<PositionTag>(load_u8(payload, rule.base + position::kTag));
        slot.identity_kind = load_u32le(payload, rule.base + position::kIdentityKind);
        slot.ordinal = load_u64le(payload, rule.base + position::kOrdinal);
        slot.identity_value = load_u64le(payload, rule.base + position::kIdentityValue);
    }
}

void SpoolScanner::check_accounting_against_prefix(std::span<const std::byte> payload,
                                                   std::uint64_t offset) noexcept
{
    if (load_u64le(payload, accounting::kSpoolCommitted) != report_.data_items_)
    {
        add_finding(SpoolCode::accounting_exceeds_prefix, offset);
    }
    if (load_u64le(payload, accounting::kControlSpoolCommitted) != report_.control_items_)
    {
        add_finding(SpoolCode::accounting_exceeds_prefix, offset);
    }
}

bool SpoolScanner::apply_semantics(std::uint64_t transaction_offset,
                                   const FramedTransaction& framed) noexcept
{
    pending_accounting_count_ = 0;

    bool frame_seen = false;
    bool discontinuity_seen = false;
    std::uint64_t frame_ordinal = 0;
    std::uint64_t discontinuity_ordinal = 0;
    bool transaction_has_accounting = false;
    bool transaction_has_session_end = false;
    bool accounting_seen_here = false;
    const bool transaction_has_item = framed.data_items != 0 || framed.control_items != 0;

    auto cursor = framed.records_offset;
    for (std::uint32_t i = 0; i < framed.n_records; ++i)
    {
        // The framing pass already verified every checksum in this
        // transaction, so this pass only reads what the rules need.
        auto header = scratch_.first(kRecordHeaderBytes);
        if (!read_exact(cursor, header))
        {
            return false;
        }
        const auto kind = load_u16le(header, record_header::kRecordKind);
        const auto payload_bytes = load_u32le(header, record_header::kPayloadBytes);
        const auto logical_ordinal = load_u64le(header, record_header::kLogicalOrdinal);
        const auto payload_offset = cursor + kRecordHeaderBytes;
        cursor = payload_offset + spool_padded_length(payload_bytes);

        if (!is_known_record_kind(kind))
        {
            // Promoting a prefix that skipped a committed record would lose a
            // record the writer counted.
            add_finding(SpoolCode::unknown_record_kind, transaction_offset);
            continue;
        }
        if (is_container_owned_kind(kind) && payload_bytes != container_owned_payload_bytes(kind))
        {
            add_finding(SpoolCode::container_payload_size, transaction_offset);
        }

        switch (static_cast<RecordKind>(kind))
        {
        case RecordKind::frame:
            frame_seen = true;
            frame_ordinal = logical_ordinal;
            break;
        case RecordKind::discontinuity:
            discontinuity_seen = true;
            discontinuity_ordinal = logical_ordinal;
            break;
        case RecordKind::signal_block:
            if (!frame_seen || logical_ordinal != frame_ordinal)
            {
                add_finding(SpoolCode::ordinal, transaction_offset);
            }
            break;
        case RecordKind::signal_gap:
            if (!discontinuity_seen || logical_ordinal != discontinuity_ordinal)
            {
                add_finding(SpoolCode::ordinal, transaction_offset);
            }
            break;
        default:
            if (has_no_owning_item(kind) && logical_ordinal != 0)
            {
                add_finding(SpoolCode::ordinal, transaction_offset);
            }
            break;
        }

        if (kind == static_cast<std::uint16_t>(RecordKind::session_end))
        {
            if (!accounting_seen_here)
            {
                add_finding(SpoolCode::session_end_without_accounting, transaction_offset);
            }
            ++session_end_count_;
            if (session_end_count_ == 1)
            {
                report_.session_end_present_ = true;
                if (payload_bytes == kSessionEndPayloadBytes)
                {
                    auto payload = scratch_.first(kSessionEndPayloadBytes);
                    if (!read_exact(payload_offset, payload))
                    {
                        return false;
                    }
                    decode_session_end(payload, transaction_offset);
                }
            }
            else
            {
                add_finding(SpoolCode::multiple_session_end, transaction_offset);
            }
            transaction_has_session_end = true;
        }
        else if (kind == static_cast<std::uint16_t>(RecordKind::accounting_snapshot))
        {
            if (transaction_has_item)
            {
                // The counters must name a settled prefix, not a moment inside
                // one.
                add_finding(SpoolCode::accounting_transaction_not_quiet, transaction_offset);
            }
            ++accounting_count_;
            if (accounting_count_ > 1)
            {
                add_finding(SpoolCode::multiple_accounting, transaction_offset);
            }
            transaction_has_accounting = true;
            accounting_seen_here = true;
            if (payload_bytes == kAccountingPayloadBytes)
            {
                auto payload = scratch_.first(kAccountingPayloadBytes);
                if (!read_exact(payload_offset, payload))
                {
                    return false;
                }
                if (pending_accounting_count_ < pending_accounting_.size())
                {
                    std::copy(payload.begin(), payload.end(),
                              pending_accounting_[pending_accounting_count_].begin());
                    pending_accounting_offset_[pending_accounting_count_] = transaction_offset;
                    ++pending_accounting_count_;
                }
                check_accounting_payload(payload, transaction_offset);
            }
        }
        else if (kind == static_cast<std::uint16_t>(RecordKind::checkpoint) &&
                 payload_bytes == kCheckpointPayloadBytes)
        {
            auto payload = scratch_.first(kCheckpointPayloadBytes);
            if (!read_exact(payload_offset, payload))
            {
                return false;
            }
            const auto claimed_policy = load_u8(payload, checkpoint::kDurabilityPolicy);
            if (claimed_policy != static_cast<std::uint8_t>(report_.durability_policy_))
            {
                // The policy is stored twice so each record is
                // self-describing; a disagreement is a writer bug, not a
                // choice a reader may pick a side of.
                add_finding(SpoolCode::checkpoint_policy, transaction_offset);
            }
            const auto claimed = load_u64le(payload, checkpoint::kDurableExtentBytes);
            if (claimed > transaction_offset)
            {
                add_finding(SpoolCode::checkpoint_extent, transaction_offset);
            }
            else
            {
                checkpoint_extent_ = std::max(checkpoint_extent_, claimed);
            }
        }
    }

    if (transaction_has_accounting && !transaction_has_session_end)
    {
        add_finding(SpoolCode::accounting_without_session_end, transaction_offset);
    }
    return true;
}

std::uint64_t SpoolScanner::derive_durable_extent() const noexcept
{
    // The floor is the superblock region, which is synced before the spool is
    // reported ready under every policy. Under `buffered` that floor is also
    // the ceiling: no record is covered, which is what "guarantees nothing
    // against power loss" is, expressed as an extent.
    switch (report_.durability_policy_)
    {
    case DurabilityPolicy::transaction_sync:
        return report_.committed_prefix_end_;
    case DurabilityPolicy::checkpoint_sync:
        return std::max(report_.first_transaction_offset_,
                        std::min(checkpoint_extent_, report_.committed_prefix_end_));
    case DurabilityPolicy::buffered:
        return report_.first_transaction_offset_;
    }
    return report_.first_transaction_offset_;
}

SpoolScanReport SpoolScanner::scan(SpoolFile& file, std::span<std::byte> scratch)
{
    report_ = SpoolScanReport{};
    session_end_count_ = 0;
    accounting_count_ = 0;
    checkpoint_extent_ = 0;
    pending_accounting_count_ = 0;
    file_ = &file;
    scratch_ = scratch;
    file_size_ = file.size();

    if (scratch.size() < kMinimumScratchBytes)
    {
        report_.read_failed_ = true;
        report_.status_ = ScanStatus::rejected;
        return report_;
    }
    if (!scan_superblock())
    {
        report_.status_ = ScanStatus::rejected;
        return report_;
    }

    report_.committed_prefix_end_ = report_.first_transaction_offset_;
    auto offset = report_.first_transaction_offset_;
    std::uint64_t expected_id = 1;
    std::uint64_t previous_offset = 0;

    while (offset < file_size_)
    {
        FramedTransaction framed{};
        if (!frame_transaction(offset, expected_id, previous_offset, framed))
        {
            break;
        }
        if (session_end_count_ > 0)
        {
            add_finding(SpoolCode::records_after_session_end, offset);
        }
        if (!apply_semantics(offset, framed))
        {
            break;
        }

        ++report_.committed_transactions_;
        report_.last_transaction_id_ = expected_id;
        report_.data_items_ += framed.data_items;
        report_.control_items_ += framed.control_items;
        report_.committed_prefix_end_ = framed.end_offset;
        previous_offset = offset;
        offset = framed.end_offset;
        ++expected_id;

        // Layer 2 is checked at the prefix the snapshot seals, with the running
        // totals at that point: a transaction committed later must not mask a
        // summary that claimed more than the sealing prefix held.
        for (std::size_t i = 0; i < pending_accounting_count_; ++i)
        {
            check_accounting_against_prefix(pending_accounting_[i], pending_accounting_offset_[i]);
        }
        pending_accounting_count_ = 0;
    }

    if (!report_.read_failed_ && report_.committed_prefix_end_ < file_size_)
    {
        report_.status_ = report_.has_code(SpoolCode::torn_tail) ? ScanStatus::torn_tail
                                                                 : ScanStatus::corrupt_tail;
    }
    report_.durable_extent_bytes_ = derive_durable_extent();
    return report_;
}

SpoolScanReport scan_spool(SpoolFile& file)
{
    std::vector<std::byte> scratch(64U * 1024U, std::byte{0});
    SpoolScanner scanner;
    return scanner.scan(file, scratch);
}

SpoolRecordCursor::SpoolRecordCursor(SpoolFile& file, const SpoolScanReport& report) noexcept
    : file_(&file), committed_prefix_end_(report.committed_prefix_end()),
      cursor_(report.first_transaction_offset())
{
    if (!report.readable() || report.read_failed() || report.findings_truncated())
    {
        committed_prefix_end_ = 0;
        cursor_ = 0;
        failed_ = true;
    }
}

bool SpoolRecordCursor::next(SpoolRecordView& out) noexcept
{
    if (failed_)
    {
        return false;
    }
    while (records_remaining_ == 0)
    {
        if (cursor_ >= committed_prefix_end_)
        {
            return false;
        }
        std::array<std::byte, kTransactionHeaderBytes> header{};
        const auto result = file_->read_at(cursor_, header);
        if (result.status != SpoolIoStatus::ok || result.transferred != header.size())
        {
            failed_ = true;
            return false;
        }
        transaction_offset_ = cursor_;
        transaction_id_ = load_u64le(header, transaction_header::kTransactionId);
        records_remaining_ = load_u32le(header, transaction_header::kRecordCount);
        cursor_ += kTransactionHeaderBytes;
        if (records_remaining_ == 0)
        {
            // A committed transaction always carries at least one record, so
            // this cannot happen for a report built from these bytes.
            failed_ = true;
            return false;
        }
    }

    std::array<std::byte, kRecordHeaderBytes> header{};
    const auto result = file_->read_at(cursor_, header);
    if (result.status != SpoolIoStatus::ok || result.transferred != header.size())
    {
        failed_ = true;
        return false;
    }
    out.transaction_id = transaction_id_;
    out.transaction_offset = transaction_offset_;
    out.record_offset = cursor_;
    out.payload_offset = cursor_ + kRecordHeaderBytes;
    out.kind = load_u16le(header, record_header::kRecordKind);
    out.payload_bytes = load_u32le(header, record_header::kPayloadBytes);
    out.logical_ordinal = load_u64le(header, record_header::kLogicalOrdinal);
    out.record_unix_nanos = load_u64le(header, record_header::kRecordUnixNanos);

    cursor_ = out.payload_offset + spool_padded_length(out.payload_bytes);
    --records_remaining_;
    if (records_remaining_ == 0)
    {
        cursor_ += kTransactionTrailerBytes;
    }
    return true;
}

bool SpoolRecordCursor::read_payload(const SpoolRecordView& view, std::span<std::byte> out) noexcept
{
    if (out.size() != view.payload_bytes)
    {
        return false;
    }
    if (out.empty())
    {
        return true;
    }
    const auto result = file_->read_at(view.payload_offset, out);
    if (result.status != SpoolIoStatus::ok || result.transferred != out.size())
    {
        failed_ = true;
        return false;
    }
    return true;
}

} // namespace neurale::recording
