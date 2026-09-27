/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "spool_writer.h"

#include "byte_order.h"
#include "crc32c.h"
#include "sha256.h"

#include <algorithm>
#include <cstring>

namespace neurale::recording
{
namespace
{

/// The smallest transaction budget a writer can be given and still be able to
/// end its session honestly: the sealing transaction carries the accounting
/// snapshot and the session-end record, and a writer that cannot write those
/// is a writer whose spool can only ever be finalized as incomplete.
constexpr std::size_t kSealTransactionBytes = kTransactionHeaderBytes + kTransactionTrailerBytes +
                                              kRecordHeaderBytes + kAccountingPayloadBytes +
                                              kRecordHeaderBytes + kSessionEndPayloadBytes;

[[nodiscard]] SpoolWriterStatus from_io(SpoolIoStatus status) noexcept
{
    switch (status)
    {
    case SpoolIoStatus::ok:
        return SpoolWriterStatus::ok;
    case SpoolIoStatus::out_of_space:
        return SpoolWriterStatus::out_of_space;
    case SpoolIoStatus::stalled:
        return SpoolWriterStatus::stalled;
    case SpoolIoStatus::cancelled:
        return SpoolWriterStatus::cancelled;
    case SpoolIoStatus::incomplete:
    case SpoolIoStatus::io_error:
        return SpoolWriterStatus::io_error;
    }
    return SpoolWriterStatus::io_error;
}

void encode_position(const SpoolAccountingPosition& pos, std::span<std::byte> out,
                     std::size_t base) noexcept
{
    store_u8(out, base + position::kTag, static_cast<std::uint8_t>(pos.tag));
    store_u8(out, base + 1, 0);
    store_u8(out, base + 2, 0);
    store_u8(out, base + 3, 0);
    store_u32le(out, base + position::kIdentityKind, static_cast<std::uint32_t>(pos.identity_kind));
    store_u64le(out, base + position::kOrdinal, pos.ordinal);
    store_u64le(out, base + position::kIdentityValue, pos.identity_value);
}

/// A position's members that its tag does not select must be zero, its tag
/// must be the form the contract fixes for the event it names, and it must be
/// present exactly when the counters it names sum to nonzero.
[[nodiscard]] bool position_is_consistent(const SpoolAccountingPosition& pos,
                                          PositionTag required_tag, bool related_present,
                                          bool data_plane) noexcept
{
    switch (pos.tag)
    {
    case PositionTag::absent:
        if (pos.identity_kind != ProducerIdentityKind::none || pos.ordinal != 0 ||
            pos.identity_value != 0)
        {
            return false;
        }
        return !related_present;
    case PositionTag::ordinal:
        if (pos.identity_kind != ProducerIdentityKind::none || pos.identity_value != 0)
        {
            return false;
        }
        break;
    case PositionTag::producer_identity:
        if (pos.ordinal != 0)
        {
            return false;
        }
        break;
    default:
        return false;
    }

    if (pos.tag != required_tag)
    {
        return false;
    }
    if (!related_present)
    {
        return false;
    }
    if (pos.tag == PositionTag::producer_identity)
    {
        const auto kind = static_cast<std::uint32_t>(pos.identity_kind);
        // The registry is partitioned by plane: a data rejection names a
        // stream message kind, a control rejection a control record-set kind.
        // A kind from the other plane is one no finalizer can map to a single
        // NRF kind string.
        if (data_plane)
        {
            return kind == 1 || kind == 2;
        }
        return kind >= 3 && kind <= 11;
    }
    return true;
}

} // namespace

void encode_accounting_payload(const SpoolAccounting& value,
                               std::span<std::byte, kAccountingPayloadBytes> out) noexcept
{
    std::fill(out.begin(), out.end(), std::byte{0});
    store_u64le(out, accounting::kRuntimeAccepted, value.runtime_accepted);
    store_u64le(out, accounting::kRecorderAccepted, value.recorder_accepted);
    store_u64le(out, accounting::kSpoolCommitted, value.spool_committed);
    store_u64le(out, accounting::kRejectedBeforeRuntimeAcceptance,
                value.rejected_before_runtime_acceptance);
    store_u64le(out, accounting::kFailedBetweenRuntimeAndRecorder,
                value.failed_between_runtime_and_recorder);
    store_u64le(out, accounting::kLostBetweenRecorderAndSpool,
                value.lost_between_recorder_and_spool);
    store_u64le(out, accounting::kControlOffered, value.control_offered);
    store_u64le(out, accounting::kControlAccepted, value.control_accepted);
    store_u64le(out, accounting::kControlSpoolCommitted, value.control_spool_committed);
    store_u64le(out, accounting::kControlRejected, value.control_rejected);
    store_u64le(out, accounting::kLostBetweenControlAcceptanceAndSpool,
                value.lost_between_control_acceptance_and_spool);
    store_u64le(out, accounting::kRejectedAfterCloseData, value.rejected_after_close_data);
    store_u64le(out, accounting::kRejectedAfterCloseControl, value.rejected_after_close_control);
    store_u8(out, accounting::kControlOfferedPresent, value.control_offered_present ? 1 : 0);
    store_u8(out, accounting::kProducerAcceptanceKnown, value.producer_acceptance_known ? 1 : 0);
    // The only legal origin in a spool: recovery-rebuilt accounting belongs to
    // the NRF session, not here.
    store_u8(out, accounting::kAccountingOrigin, 0);
    encode_position(value.data_first_loss, out, accounting::kDataFirstLoss);
    encode_position(value.data_first_rejection, out, accounting::kDataFirstRejection);
    encode_position(value.control_first_loss, out, accounting::kControlFirstLoss);
    encode_position(value.control_first_rejection, out, accounting::kControlFirstRejection);
}

SpoolWriterStatus validate_accounting(const SpoolAccounting& accounting,
                                      std::uint64_t committed_data_items,
                                      std::uint64_t committed_control_items) noexcept
{
    // Layer 1 -- the internal identities. A refused item was never inside the
    // stage it was refused entry to, so no rejection counter appears here
    // except in the optional control_offered identity.
    if (accounting.recorder_accepted > accounting.runtime_accepted ||
        accounting.failed_between_runtime_and_recorder !=
            accounting.runtime_accepted - accounting.recorder_accepted)
    {
        return SpoolWriterStatus::accounting_inconsistent;
    }
    if (accounting.spool_committed > accounting.recorder_accepted ||
        accounting.lost_between_recorder_and_spool !=
            accounting.recorder_accepted - accounting.spool_committed)
    {
        return SpoolWriterStatus::accounting_inconsistent;
    }
    if (accounting.control_spool_committed > accounting.control_accepted ||
        accounting.lost_between_control_acceptance_and_spool !=
            accounting.control_accepted - accounting.control_spool_committed)
    {
        return SpoolWriterStatus::accounting_inconsistent;
    }
    if (accounting.control_offered_present &&
        (accounting.control_accepted > accounting.control_offered ||
         accounting.control_rejected != accounting.control_offered - accounting.control_accepted))
    {
        return SpoolWriterStatus::accounting_inconsistent;
    }

    // Layer 2 -- the committed counters name what the container actually holds.
    if (accounting.spool_committed != committed_data_items ||
        accounting.control_spool_committed != committed_control_items)
    {
        return SpoolWriterStatus::accounting_inconsistent;
    }

    const bool data_loss_present = accounting.failed_between_runtime_and_recorder != 0 ||
                                   accounting.lost_between_recorder_and_spool != 0;
    if (!position_is_consistent(accounting.data_first_loss, PositionTag::ordinal, data_loss_present,
                                true) ||
        !position_is_consistent(accounting.data_first_rejection, PositionTag::producer_identity,
                                accounting.rejected_before_runtime_acceptance != 0, true) ||
        !position_is_consistent(accounting.control_first_loss, PositionTag::ordinal,
                                accounting.lost_between_control_acceptance_and_spool != 0, false) ||
        !position_is_consistent(accounting.control_first_rejection, PositionTag::producer_identity,
                                accounting.control_rejected != 0, false))
    {
        return SpoolWriterStatus::accounting_inconsistent;
    }
    return SpoolWriterStatus::ok;
}

void SpoolWriter::fail(SpoolWriterStatus status) noexcept
{
    state_ = SpoolWriterState::failed;
    fault_ = status;
    staged_bytes_ = 0;
    staged_records_ = 0;
}

SpoolWriterStatus SpoolWriter::write_all(std::span<const std::byte> data) noexcept
{
    std::size_t written = 0;
    while (written < data.size())
    {
        const auto result = file_->append(data.subspan(written));
        written += result.transferred;
        if (result.status == SpoolIoStatus::ok)
        {
            continue;
        }
        if (result.status == SpoolIoStatus::incomplete)
        {
            if (result.transferred == 0)
            {
                // No progress and no error: retrying forever is how a writer
                // hangs a recorder. Report it as an I/O failure instead.
                return SpoolWriterStatus::io_error;
            }
            continue;
        }
        return from_io(result.status);
    }
    return SpoolWriterStatus::ok;
}

SpoolWriterStatus SpoolWriter::sync_now() noexcept
{
    const auto result = file_->sync();
    if (result.status != SpoolIoStatus::ok)
    {
        // A failed sync is not a failed commit: everything committed before it
        // is still committed. What it costs is the durability claim, so the
        // synced extent stays where the last successful sync left it.
        return result.status == SpoolIoStatus::stalled ? SpoolWriterStatus::stalled
                                                       : SpoolWriterStatus::sync_failed;
    }
    synced_extent_ = committed_extent_;
    return SpoolWriterStatus::ok;
}

SpoolWriterStatus SpoolWriter::allocate(SpoolFile& file, const SpoolSessionIdentity& identity,
                                        DurabilityPolicy policy, const SpoolWriterLimits& limits)
{
    if (state_ != SpoolWriterState::constructed)
    {
        return SpoolWriterStatus::wrong_state;
    }
    if (policy != DurabilityPolicy::buffered && policy != DurabilityPolicy::checkpoint_sync &&
        policy != DurabilityPolicy::transaction_sync)
    {
        return SpoolWriterStatus::invalid_argument;
    }
    if (identity.session_id.size() > kSessionIdBytes)
    {
        return SpoolWriterStatus::invalid_argument;
    }
    if (identity.plan_document.size() > 0xFFFFFFFFULL)
    {
        return SpoolWriterStatus::invalid_argument;
    }
    if (limits.max_records_per_transaction < 2 ||
        limits.max_transaction_bytes < kSealTransactionBytes)
    {
        return SpoolWriterStatus::limit_exceeded;
    }
    if (file.size() != 0)
    {
        return SpoolWriterStatus::wrong_state;
    }

    const auto digest = sha256(identity.plan_document);
    if (digest != identity.plan_fingerprint)
    {
        return SpoolWriterStatus::plan_fingerprint_mismatch;
    }

    const auto plan_bytes = identity.plan_document.size();
    const auto padded_plan = static_cast<std::size_t>(spool_padded_length(plan_bytes));
    const auto region_bytes = kSuperblockBytes + padded_plan;

    // Complete every allocation before any writer state changes, so a failure
    // (a returned status or a thrown bad_alloc) leaves the writer `constructed`
    // and the file empty -- the object may be prepared again, and no artifact
    // exists. Nothing here is durable: the superblock is built in memory only.
    std::vector<std::byte> region(region_bytes, std::byte{0});
    std::vector<std::byte> staging(limits.max_transaction_bytes, std::byte{0});
    std::span<std::byte> out{region};
    std::memcpy(out.data() + superblock::kMagic, kSpoolMagic, sizeof(kSpoolMagic));
    store_u16le(out, superblock::kVersionMajorOffset, kVersionMajor);
    store_u16le(out, superblock::kVersionMinorOffset, kVersionMinor);
    store_u32le(out, superblock::kSuperblockBytesOffset,
                static_cast<std::uint32_t>(kSuperblockBytes));
    store_u8(out, superblock::kDurabilityPolicy, static_cast<std::uint8_t>(policy));
    store_u8(out, superblock::kChecksumAlgorithm, kChecksumCrc32c);
    store_u8(out, superblock::kPlanEncoding, kPlanEncodingRecordingPlanJcs);
    store_u32le(out, superblock::kAlignment, static_cast<std::uint32_t>(kSpoolAlignment));
    store_u32le(out, superblock::kPlanBytes, static_cast<std::uint32_t>(plan_bytes));
    store_u32le(out, superblock::kPlanCrc32c, crc32c(identity.plan_document));
    for (std::size_t i = 0; i < kPlanFingerprintBytes; ++i)
    {
        store_u8(out, superblock::kPlanFingerprint + i, digest[i]);
    }
    for (std::size_t i = 0; i < kSessionUuidBytes; ++i)
    {
        store_u8(out, superblock::kSessionUuid + i, identity.session_uuid[i]);
    }
    store_u64le(out, superblock::kCreatedUnixNanos, identity.created_unix_nanos);
    store_u64le(out, superblock::kFirstTransactionOffset, region_bytes);
    if (!identity.session_id.empty())
    {
        std::memcpy(out.data() + superblock::kSessionId, identity.session_id.data(),
                    identity.session_id.size());
    }
    store_u32le(out, superblock::kCrc32c, crc32c(out.first(superblock::kCrc32c)));
    if (plan_bytes != 0)
    {
        std::memcpy(out.data() + kSuperblockBytes, identity.plan_document.data(), plan_bytes);
    }

    file_ = &file;
    policy_ = policy;
    limits_ = limits;
    identity_ = identity;
    staging_.swap(staging);
    superblock_region_.swap(region);
    state_ = SpoolWriterState::allocated;
    return SpoolWriterStatus::ok;
}

SpoolWriterStatus SpoolWriter::commit_superblock() noexcept
{
    if (state_ != SpoolWriterState::allocated)
    {
        return SpoolWriterStatus::wrong_state;
    }

    if (const auto status = write_all(superblock_region_); status != SpoolWriterStatus::ok)
    {
        fail(status);
        return status;
    }
    const auto region_bytes = superblock_region_.size();
    committed_extent_ = region_bytes;
    // Under every policy: a spool whose identity did not survive the crash
    // could not be matched to a finalization target at all.
    if (const auto result = file_->sync(); result.status != SpoolIoStatus::ok)
    {
        const auto status = result.status == SpoolIoStatus::stalled
                                ? SpoolWriterStatus::stalled
                                : SpoolWriterStatus::sync_failed;
        fail(status);
        return status;
    }
    synced_extent_ = region_bytes;
    first_transaction_offset_ = region_bytes;
    checkpoint_extent_ = region_bytes;

    // The superblock is published; the in-memory copy is no longer needed.
    superblock_region_.clear();
    superblock_region_.shrink_to_fit();

    state_ = SpoolWriterState::prepared;
    return SpoolWriterStatus::ok;
}

SpoolWriterStatus SpoolWriter::prepare(SpoolFile& file, const SpoolSessionIdentity& identity,
                                       DurabilityPolicy policy, const SpoolWriterLimits& limits)
{
    if (const auto status = allocate(file, identity, policy, limits);
        status != SpoolWriterStatus::ok)
    {
        return status;
    }
    return commit_superblock();
}

void SpoolWriter::open_staging(std::uint64_t begin_unix_nanos) noexcept
{
    std::span<std::byte> out{staging_};
    std::fill_n(out.begin(), kTransactionHeaderBytes, std::byte{0});
    std::memcpy(out.data() + transaction_header::kMagic, kTransactionBeginMagic,
                sizeof(kTransactionBeginMagic));
    store_u64le(out, transaction_header::kTransactionId, next_transaction_id_);
    store_u64le(out, transaction_header::kPreviousTransactionOffset, previous_transaction_offset_);
    store_u32le(out, transaction_header::kHeaderBytes,
                static_cast<std::uint32_t>(kTransactionHeaderBytes));
    store_u64le(out, transaction_header::kBeginUnixNanos, begin_unix_nanos);

    staged_bytes_ = kTransactionHeaderBytes;
    staged_records_ = 0;
    staged_data_items_ = 0;
    staged_control_items_ = 0;
    staged_frame_seen_ = false;
    staged_discontinuity_seen_ = false;
    staged_frame_ordinal_ = 0;
    staged_discontinuity_ordinal_ = 0;
}

SpoolWriterStatus SpoolWriter::begin_transaction(std::uint64_t begin_unix_nanos) noexcept
{
    if (state_ != SpoolWriterState::prepared)
    {
        return SpoolWriterStatus::wrong_state;
    }
    open_staging(begin_unix_nanos);
    state_ = SpoolWriterState::transaction_open;
    return SpoolWriterStatus::ok;
}

SpoolWriterStatus SpoolWriter::stage_record(std::uint16_t kind, std::span<const std::byte> payload,
                                            std::uint64_t logical_ordinal,
                                            std::uint64_t record_unix_nanos) noexcept
{
    if (payload.size() > 0xFFFFFFFFULL)
    {
        return SpoolWriterStatus::invalid_argument;
    }
    if (staged_records_ >= limits_.max_records_per_transaction)
    {
        return SpoolWriterStatus::limit_exceeded;
    }
    const auto padded = static_cast<std::size_t>(spool_padded_length(payload.size()));
    const auto needed = kRecordHeaderBytes + padded;
    // The trailer has to fit too: a staged transaction that cannot be closed
    // is one the writer would have to either grow for or throw away.
    if (staged_bytes_ + needed + kTransactionTrailerBytes > staging_.size())
    {
        return SpoolWriterStatus::limit_exceeded;
    }

    std::span<std::byte> out{staging_};
    const auto base = staged_bytes_;
    std::fill_n(out.begin() + static_cast<std::ptrdiff_t>(base), kRecordHeaderBytes, std::byte{0});
    store_u16le(out, base + record_header::kRecordKind, kind);
    store_u16le(out, base + record_header::kRecordFlags, 0);
    store_u32le(out, base + record_header::kPayloadBytes,
                static_cast<std::uint32_t>(payload.size()));
    store_u64le(out, base + record_header::kLogicalOrdinal, logical_ordinal);
    store_u64le(out, base + record_header::kRecordUnixNanos, record_unix_nanos);
    store_u32le(out, base + record_header::kPayloadCrc32c, crc32c(payload));
    store_u32le(out, base + record_header::kCrc32c,
                crc32c(out.subspan(base, record_header::kCrc32c)));
    if (!payload.empty())
    {
        std::memcpy(out.data() + base + kRecordHeaderBytes, payload.data(), payload.size());
    }
    // Padding carries no information, so a nonzero byte there is unexplained
    // and a reader treats it as corruption.
    std::fill_n(out.begin() +
                    static_cast<std::ptrdiff_t>(base + kRecordHeaderBytes + payload.size()),
                padded - payload.size(), std::byte{0});

    staged_bytes_ += needed;
    ++staged_records_;
    if (is_data_plane_item(kind))
    {
        ++staged_data_items_;
    }
    if (is_control_plane_item(kind))
    {
        ++staged_control_items_;
    }
    return SpoolWriterStatus::ok;
}

SpoolWriterStatus SpoolWriter::append_record(RecordKind kind, std::span<const std::byte> payload,
                                             std::uint64_t logical_ordinal,
                                             std::uint64_t record_unix_nanos) noexcept
{
    if (state_ != SpoolWriterState::transaction_open)
    {
        return SpoolWriterStatus::wrong_state;
    }
    const auto raw = static_cast<std::uint16_t>(kind);
    if (!is_known_record_kind(raw))
    {
        return SpoolWriterStatus::invalid_argument;
    }
    // The container-owned kinds are written by checkpoint() and seal(), which
    // is what keeps their ordering, quiet-transaction, and multiplicity rules
    // true by construction.
    if (is_container_owned_kind(raw))
    {
        return SpoolWriterStatus::invalid_argument;
    }

    switch (kind)
    {
    case RecordKind::frame:
        break;
    case RecordKind::discontinuity:
        break;
    case RecordKind::signal_block:
        // A block is part of the frame that owns it, and the owner is the
        // nearest preceding frame in the same transaction.
        if (!staged_frame_seen_ || logical_ordinal != staged_frame_ordinal_)
        {
            return SpoolWriterStatus::invalid_argument;
        }
        break;
    case RecordKind::signal_gap:
        if (!staged_discontinuity_seen_ || logical_ordinal != staged_discontinuity_ordinal_)
        {
            return SpoolWriterStatus::invalid_argument;
        }
        break;
    case RecordKind::fault:
        if (logical_ordinal != 0)
        {
            return SpoolWriterStatus::invalid_argument;
        }
        break;
    default:
        break;
    }

    const auto status = stage_record(raw, payload, logical_ordinal, record_unix_nanos);
    if (status != SpoolWriterStatus::ok)
    {
        return status;
    }
    if (kind == RecordKind::frame)
    {
        staged_frame_seen_ = true;
        staged_frame_ordinal_ = logical_ordinal;
    }
    else if (kind == RecordKind::discontinuity)
    {
        staged_discontinuity_seen_ = true;
        staged_discontinuity_ordinal_ = logical_ordinal;
    }
    return SpoolWriterStatus::ok;
}

SpoolWriterStatus SpoolWriter::commit_staged() noexcept
{
    if (staged_records_ == 0)
    {
        // A transaction must carry at least one record: an empty one commits
        // nothing and a reader ends its prefix there.
        return SpoolWriterStatus::invalid_argument;
    }

    std::span<std::byte> out{staging_};
    // The record count is only known now, so the header is completed and
    // checksummed here rather than when the transaction was opened. These are
    // staging bytes: nothing committed is being rewritten.
    store_u32le(out, transaction_header::kRecordCount, static_cast<std::uint32_t>(staged_records_));
    store_u32le(out, transaction_header::kCrc32c, crc32c(out.first(transaction_header::kCrc32c)));

    const auto body_bytes = staged_bytes_;
    const auto trailer = body_bytes;
    std::fill_n(out.begin() + static_cast<std::ptrdiff_t>(trailer), kTransactionTrailerBytes,
                std::byte{0});
    std::memcpy(out.data() + trailer + transaction_trailer::kMagic, kTransactionEndMagic,
                sizeof(kTransactionEndMagic));
    store_u64le(out, trailer + transaction_trailer::kTransactionId, next_transaction_id_);
    store_u64le(out, trailer + transaction_trailer::kBodyBytes, body_bytes);
    store_u32le(out, trailer + transaction_trailer::kRecordCount,
                static_cast<std::uint32_t>(staged_records_));
    store_u32le(out, trailer + transaction_trailer::kDataItems, staged_data_items_);
    store_u32le(out, trailer + transaction_trailer::kControlItems, staged_control_items_);
    store_u32le(out, trailer + transaction_trailer::kBodyCrc32c, crc32c(out.first(body_bytes)));
    store_u32le(out, trailer + transaction_trailer::kFlags, 0);
    store_u32le(out, trailer + transaction_trailer::kCrc32c,
                crc32c(out.subspan(trailer, transaction_trailer::kCrc32c)));

    const auto total = body_bytes + kTransactionTrailerBytes;
    const auto transaction_offset = committed_extent_;
    if (const auto status = write_all(out.first(total)); status != SpoolWriterStatus::ok)
    {
        fail(status);
        return status;
    }

    committed_extent_ = transaction_offset + total;
    previous_transaction_offset_ = transaction_offset;
    ++next_transaction_id_;
    ++committed_transactions_;
    data_items_ += staged_data_items_;
    control_items_ += staged_control_items_;
    staged_bytes_ = 0;
    staged_records_ = 0;
    state_ = SpoolWriterState::prepared;

    if (policy_ == DurabilityPolicy::transaction_sync)
    {
        if (const auto status = sync_now(); status != SpoolWriterStatus::ok)
        {
            fail(status);
            return status;
        }
    }
    return SpoolWriterStatus::ok;
}

SpoolWriterStatus SpoolWriter::commit_transaction() noexcept
{
    if (state_ != SpoolWriterState::transaction_open)
    {
        return SpoolWriterStatus::wrong_state;
    }
    return commit_staged();
}

void SpoolWriter::discard_transaction() noexcept
{
    if (state_ != SpoolWriterState::transaction_open)
    {
        return;
    }
    staged_bytes_ = 0;
    staged_records_ = 0;
    state_ = SpoolWriterState::prepared;
}

SpoolWriterStatus SpoolWriter::checkpoint(std::uint64_t begin_unix_nanos,
                                          std::uint64_t synced_unix_nanos) noexcept
{
    if (state_ != SpoolWriterState::prepared)
    {
        return SpoolWriterStatus::wrong_state;
    }

    std::uint64_t claimed = first_transaction_offset_;
    if (policy_ != DurabilityPolicy::buffered)
    {
        if (const auto status = sync_now(); status != SpoolWriterStatus::ok)
        {
            fail(status);
            return status;
        }
        claimed = synced_extent_;
    }
    // Under `buffered` the writer does not sync here, so the only extent it may
    // claim is the superblock region -- the one thing that was synced before
    // the spool was reported ready. Claiming the committed extent would promise
    // what the policy does not deliver.

    std::array<std::byte, kCheckpointPayloadBytes> payload{};
    store_u64le(payload, checkpoint::kDurableExtentBytes, claimed);
    store_u64le(payload, checkpoint::kSyncedUnixNanos, synced_unix_nanos);
    store_u8(payload, checkpoint::kDurabilityPolicy, static_cast<std::uint8_t>(policy_));

    open_staging(begin_unix_nanos);
    state_ = SpoolWriterState::transaction_open;
    // The record header carries no timestamp of its own: the payload's
    // `synced_unix_nanos` is when the sync was acknowledged, which is the only
    // time this record has to report.
    if (const auto status =
            stage_record(static_cast<std::uint16_t>(RecordKind::checkpoint), payload, 0, 0);
        status != SpoolWriterStatus::ok)
    {
        discard_transaction();
        return status;
    }
    if (const auto status = commit_staged(); status != SpoolWriterStatus::ok)
    {
        return status;
    }
    checkpoint_extent_ = std::max(checkpoint_extent_, claimed);
    return SpoolWriterStatus::ok;
}

SpoolWriterStatus SpoolWriter::seal(const SpoolAccounting& accounting,
                                    const SpoolSessionEnd& session_end,
                                    std::uint64_t begin_unix_nanos) noexcept
{
    if (state_ != SpoolWriterState::prepared)
    {
        return SpoolWriterStatus::wrong_state;
    }
    if (session_end.terminal_reason.size() > kTerminalReasonBytes)
    {
        return SpoolWriterStatus::invalid_argument;
    }
    const auto intent = static_cast<std::uint8_t>(session_end.requested_terminal_intent);
    const auto outcome = static_cast<std::uint8_t>(session_end.capture_outcome);
    if (intent > 2 || outcome > 2)
    {
        return SpoolWriterStatus::invalid_argument;
    }
    if (const auto status = validate_accounting(accounting, data_items_, control_items_);
        status != SpoolWriterStatus::ok)
    {
        return status;
    }

    std::array<std::byte, kAccountingPayloadBytes> accounting_payload{};
    encode_accounting_payload(accounting, accounting_payload);

    std::array<std::byte, kSessionEndPayloadBytes> end_payload{};
    store_u8(end_payload, session_end::kRequestedTerminalIntent, intent);
    store_u8(end_payload, session_end::kCaptureOutcome, outcome);
    store_u8(end_payload, session_end::kPrimaryFaultCommitted,
             session_end.primary_fault_committed ? 1 : 0);
    if (!session_end.terminal_reason.empty())
    {
        std::memcpy(end_payload.data() + session_end::kTerminalReason,
                    session_end.terminal_reason.data(), session_end.terminal_reason.size());
    }
    store_u64le(end_payload, session_end::kEndUnixNanos, session_end.end_unix_nanos);

    open_staging(begin_unix_nanos);
    state_ = SpoolWriterState::transaction_open;
    // The snapshot precedes the session end in the same transaction: the
    // summary seals exactly one session, and the pair is what freezes the
    // capture outcome.
    if (const auto status = stage_record(
            static_cast<std::uint16_t>(RecordKind::accounting_snapshot), accounting_payload, 0, 0);
        status != SpoolWriterStatus::ok)
    {
        discard_transaction();
        return status;
    }
    // The payload carries `end_unix_nanos`; the record header has no separate
    // time to report.
    if (const auto status =
            stage_record(static_cast<std::uint16_t>(RecordKind::session_end), end_payload, 0, 0);
        status != SpoolWriterStatus::ok)
    {
        discard_transaction();
        return status;
    }
    if (const auto status = commit_staged(); status != SpoolWriterStatus::ok)
    {
        return status;
    }
    if (policy_ == DurabilityPolicy::checkpoint_sync)
    {
        // The seal is a sync boundary -- the session-end record is what freezes
        // the capture outcome and it may not be the least durable record in the
        // file. The reported durable extent is deliberately *not* raised here:
        // a reader credits only extents a committed checkpoint record claims,
        // and a writer that reported more than a reader can derive would be
        // making a durability claim nothing in the spool backs.
        if (const auto status = sync_now(); status != SpoolWriterStatus::ok)
        {
            fail(status);
            return status;
        }
    }
    state_ = SpoolWriterState::sealed;
    return SpoolWriterStatus::ok;
}

void SpoolWriter::cancel() noexcept
{
    switch (state_)
    {
    case SpoolWriterState::sealed:
    case SpoolWriterState::cancelled:
    case SpoolWriterState::failed:
        // Terminal states are terminal: a late cancel must not rewrite a
        // successful seal, mask a recorded fault, or repeat itself. Contract
        // section 4.5 makes every terminal state terminal, and a recorder
        // that reports writer state to its caller depends on that.
        return;

    case SpoolWriterState::transaction_open:
        staged_bytes_ = 0;
        staged_records_ = 0;
        break;

    case SpoolWriterState::constructed:
    case SpoolWriterState::allocated:
    case SpoolWriterState::prepared:
        break;
    }

    state_ = SpoolWriterState::cancelled;
}

std::uint64_t SpoolWriter::durable_extent() const noexcept
{
    switch (policy_)
    {
    case DurabilityPolicy::transaction_sync:
        return synced_extent_;
    case DurabilityPolicy::checkpoint_sync:
        return std::max(first_transaction_offset_, std::min(checkpoint_extent_, committed_extent_));
    case DurabilityPolicy::buffered:
        return first_transaction_offset_;
    }
    return first_transaction_offset_;
}

} // namespace neurale::recording
