/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "recording_plan.h"

#include "record_payloads.h"

#include <limits>

namespace neurale::recording
{
namespace
{

/// Largest slot a single item can need, so `validate()` can prove one item
/// fits one transaction before anything is allocated.
[[nodiscard]] bool frame_fits_transaction(const NativeRecordingPlan& plan) noexcept
{
    // One frame becomes one frame record plus one block record per block, and
    // the whole thing has to fit in one transaction: the recorder never splits
    // an item across transactions, because a torn tail would then leave a
    // frame committed without its blocks and the item would be neither
    // committed nor lost.
    const auto blocks = static_cast<std::uint64_t>(plan.max_blocks_per_frame);
    const std::uint64_t frame_record = kRecordHeaderBytes + kFramePayloadBytes;
    const std::uint64_t block_records =
        blocks * (kRecordHeaderBytes + kSignalBlockHeaderPayloadBytes) +
        spool_padded_length(plan.max_frame_payload_bytes) + blocks * 8;
    const std::uint64_t framing = kTransactionHeaderBytes + kTransactionTrailerBytes;
    const std::uint64_t needed = framing + frame_record + block_records;
    return needed <= plan.max_transaction_bytes &&
           static_cast<std::uint64_t>(plan.max_blocks_per_frame) + 1U <=
               plan.max_records_per_transaction;
}

[[nodiscard]] bool discontinuity_fits_transaction(const NativeRecordingPlan& plan) noexcept
{
    const auto gaps = static_cast<std::uint64_t>(plan.max_signal_gaps_per_discontinuity);
    const std::uint64_t needed = kTransactionHeaderBytes + kTransactionTrailerBytes +
                                 kRecordHeaderBytes + kDiscontinuityPayloadBytes +
                                 gaps * (kRecordHeaderBytes + kSignalGapPayloadBytes);
    return needed <= plan.max_transaction_bytes && gaps + 1U <= plan.max_records_per_transaction;
}

[[nodiscard]] bool control_fits_transaction(const NativeRecordingPlan& plan) noexcept
{
    const std::uint64_t needed =
        kTransactionHeaderBytes + kTransactionTrailerBytes + kRecordHeaderBytes +
        spool_padded_length(kControlHeaderPayloadBytes + plan.max_control_payload_bytes);
    return needed <= plan.max_transaction_bytes && plan.max_records_per_transaction >= 1;
}

} // namespace

RecordingPlanStatus NativeRecordingPlan::validate() const noexcept
{
    if (session_id.empty() || session_id.size() > kSessionIdBytes)
    {
        return RecordingPlanStatus::invalid_identity;
    }
    if (plan_document.empty())
    {
        return RecordingPlanStatus::invalid_identity;
    }

    if (frame_queue_capacity == 0 || control_queue_capacity == 0 || max_blocks_per_frame == 0 ||
        max_frame_payload_bytes == 0 || max_records_per_transaction == 0 ||
        max_transaction_bytes == 0 || worker_idle_poll_nanos == 0 || drain_timeout_nanos == 0)
    {
        return RecordingPlanStatus::invalid_bound;
    }

    // The slot sizing below multiplies these; refuse anything whose product
    // would not be representable rather than discovering it as a wrap.
    constexpr std::uint64_t kSaneLimit = 1ULL << 40U;
    if (max_frame_payload_bytes > kSaneLimit || max_transaction_bytes > kSaneLimit ||
        static_cast<std::uint64_t>(frame_queue_capacity) > kSaneLimit ||
        static_cast<std::uint64_t>(control_queue_capacity) > kSaneLimit ||
        max_blocks_per_frame > (1U << 20U) || max_signal_gaps_per_discontinuity > (1U << 20U) ||
        max_control_payload_bytes > kSaneLimit)
    {
        return RecordingPlanStatus::invalid_bound;
    }

    if (recorded_signals.empty())
    {
        return RecordingPlanStatus::invalid_signal_table;
    }
    for (std::size_t i = 0; i < recorded_signals.size(); ++i)
    {
        const auto& signal = recorded_signals[i];
        if (signal.max_block_bytes == 0 || signal.max_block_bytes > max_frame_payload_bytes)
        {
            return RecordingPlanStatus::invalid_signal_table;
        }
        if (i > 0 && recorded_signals[i - 1].signal_id >= signal.signal_id)
        {
            return RecordingPlanStatus::invalid_signal_table;
        }
    }

    if (!frame_fits_transaction(*this) || !discontinuity_fits_transaction(*this) ||
        !control_fits_transaction(*this))
    {
        return RecordingPlanStatus::inconsistent_bounds;
    }
    return RecordingPlanStatus::ok;
}

const PlannedSignalRecording*
NativeRecordingPlan::find_signal(std::uint32_t signal_id) const noexcept
{
    std::size_t low = 0;
    std::size_t high = recorded_signals.size();
    while (low < high)
    {
        const auto middle = low + (high - low) / 2;
        const auto candidate = recorded_signals[middle].signal_id;
        if (candidate == signal_id)
        {
            return &recorded_signals[middle];
        }
        if (candidate < signal_id)
        {
            low = middle + 1;
        }
        else
        {
            high = middle;
        }
    }
    return nullptr;
}

std::string_view to_string(RecordingPlanStatus status) noexcept
{
    switch (status)
    {
    case RecordingPlanStatus::ok:
        return "ok";
    case RecordingPlanStatus::invalid_identity:
        return "invalid_identity";
    case RecordingPlanStatus::invalid_bound:
        return "invalid_bound";
    case RecordingPlanStatus::invalid_signal_table:
        return "invalid_signal_table";
    case RecordingPlanStatus::inconsistent_bounds:
        return "inconsistent_bounds";
    }
    return "unknown";
}

} // namespace neurale::recording
