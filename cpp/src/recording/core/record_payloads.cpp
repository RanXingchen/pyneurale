/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "record_payloads.h"

#include "byte_order.h"

#include <algorithm>

namespace neurale::recording
{
namespace
{

void zero(std::span<std::byte> out) noexcept
{
    std::fill(out.begin(), out.end(), std::byte{0});
}

} // namespace

std::string_view to_string(FaultOrigin origin) noexcept
{
    switch (origin)
    {
    case FaultOrigin::runtime:
        return "runtime";
    case FaultOrigin::recorder:
        return "recorder";
    }
    return "unknown";
}

std::string_view to_string(RecorderFaultReason reason) noexcept
{
    switch (reason)
    {
    case RecorderFaultReason::none:
        return "none";
    case RecorderFaultReason::data_queue_saturated:
        return "data_queue_saturated";
    case RecorderFaultReason::control_queue_saturated:
        return "control_queue_saturated";
    case RecorderFaultReason::plan_violation:
        return "plan_violation";
    case RecorderFaultReason::spool_writer_failed:
        return "spool_writer_failed";
    case RecorderFaultReason::drain_timeout:
        return "drain_timeout";
    case RecorderFaultReason::edge_rejection:
        return "edge_rejection";
    case RecorderFaultReason::readiness_gate_failed:
        return "readiness_gate_failed";
    }
    return "unknown";
}

void encode_frame_payload(const FramePayloadFields& fields,
                          std::span<std::byte, kFramePayloadBytes> out) noexcept
{
    zero(out);
    store_u64le(out, frame_payload::kDataMessageOrdinal, fields.data_message_ordinal);
    store_u64le(out, frame_payload::kNativeSessionId, fields.native_session_id);
    store_u64le(out, frame_payload::kFrameSequence, fields.frame_sequence);
    store_u64le(out, frame_payload::kFrameOrdinal, fields.frame_ordinal);
    store_u64le(out, frame_payload::kHostReceivedNanos, fields.host_received_ns);
    store_u64le(out, frame_payload::kSourceTick, fields.source_tick);
    store_u64le(out, frame_payload::kValidUntilNanos, fields.valid_until_ns);
    store_u32le(out, frame_payload::kNativeSchemaId, fields.native_schema_id);
    store_u32le(out, frame_payload::kSourceClockDomain, fields.source_clock_domain);
    store_u32le(out, frame_payload::kFrameFlags, fields.frame_flags);
    store_u32le(out, frame_payload::kSignalBlockCount, fields.signal_block_count);
    store_u32le(out, frame_payload::kRecordedSignalBlockCount, fields.recorded_signal_block_count);
    store_u64le(out, frame_payload::kTotalPayloadByteCount, fields.total_payload_byte_count);
    store_u64le(out, frame_payload::kRuntimeAcceptedHostNanos,
                fields.runtime_accepted_host_time_ns);
}

void encode_signal_block_payload(const SignalBlockPayloadFields& fields,
                                 std::span<std::byte, kSignalBlockHeaderPayloadBytes> out) noexcept
{
    zero(out);
    using namespace signal_block_payload;
    store_u64le(out, kSignalBlockOrdinal, fields.signal_block_ordinal);
    store_u64le(out, kDataMessageOrdinal, fields.data_message_ordinal);
    store_u64le(out, kFrameOrdinal, fields.frame_ordinal);
    store_u32le(out, kBlockIndexInFrame, fields.block_idx_in_frame);
    store_u32le(out, kNativeSignalId, fields.native_signal_id);
    store_u64le(out, kSampleIndexStart, fields.sample_idx_start);
    store_u64le(out, kLastSampleIndex, fields.last_sample_idx);
    store_u32le(out, kSampleCount, fields.n_samples);
    store_u32le(out, kClockSyncClockDomain, fields.clock_sync_clock_domain);
    store_u64le(out, kDeviceTickStart, fields.device_tick_start);
    store_u64le(out, kObservationTimeStartNanos, fields.observation_time_start_ns);
    store_u64le(out, kPayloadOffset, fields.payload_offset);
    store_u64le(out, kPayloadByteCount, fields.payload_byte_count);
    store_u64le(out, kClockSyncDeviceTickReference, fields.clock_sync_device_tick_reference);
    store_u64le(out, kClockSyncHostTimeReferenceNanos, fields.clock_sync_host_time_reference_ns);
    store_u64le(out, kClockSyncRateNumerator, fields.clock_sync_rate_numerator);
    store_u64le(out, kClockSyncRateDenominator, fields.clock_sync_rate_denominator);
    store_u64le(out, kClockSyncUncertaintyNanos, fields.clock_sync_uncertainty_ns);
    store_u32le(out, kClockSyncGeneration, fields.clock_sync_generation);
    store_u32le(out, kClockSyncFlags, fields.clock_sync_flags);
}

void encode_discontinuity_payload(const DiscontinuityPayloadFields& fields,
                                  std::span<std::byte, kDiscontinuityPayloadBytes> out) noexcept
{
    zero(out);
    using namespace discontinuity_payload;
    store_u64le(out, kDataMessageOrdinal, fields.data_message_ordinal);
    store_u64le(out, kNativeSessionId, fields.native_session_id);
    store_u64le(out, kPreviousFrameSequence, fields.previous_frame_sequence);
    store_u64le(out, kActualFrameSequence, fields.actual_frame_sequence);
    store_u64le(out, kRuntimeAcceptedHostNanos, fields.runtime_accepted_host_time_ns);
    store_u32le(out, kSignalGapCount, fields.signal_gap_count);
    store_u32le(out, kReason, fields.reason);
}

void encode_signal_gap_payload(const SignalGapPayloadFields& fields,
                               std::span<std::byte, kSignalGapPayloadBytes> out) noexcept
{
    zero(out);
    using namespace signal_gap_payload;
    store_u64le(out, kSignalGapOrdinal, fields.signal_gap_ordinal);
    store_u64le(out, kDataMessageOrdinal, fields.data_message_ordinal);
    store_u64le(out, kExpectedSampleIndex, fields.expected_sample_idx);
    store_u64le(out, kActualSampleIndex, fields.actual_sample_idx);
    store_u64le(out, kMissingSamples, fields.missing_samples);
    store_u64le(out, kExpectedDeviceTick, fields.expected_device_tick);
    store_u64le(out, kActualDeviceTick, fields.actual_device_tick);
    store_u32le(out, kGapIndexInMessage, fields.gap_idx_in_message);
    store_u32le(out, kNativeSignalId, fields.native_signal_id);
    store_u32le(out, kReason, fields.reason);
    store_u32le(out, kGapFlags, fields.gap_flags);
}

void encode_control_payload(const ControlPayloadFields& fields,
                            std::span<std::byte, kControlHeaderPayloadBytes> out) noexcept
{
    zero(out);
    using namespace control_payload;
    store_u64le(out, kSubmissionOrdinal, fields.submission_ordinal);
    store_u64le(out, kIdentity, fields.identity);
    store_u64le(out, kTimeNanos, fields.time_ns);
    store_u32le(out, kControlKind, fields.control_kind);
    store_u32le(out, kClockDomain, fields.clock_domain);
}

void encode_fault_payload(const FaultPayloadFields& fields,
                          std::span<std::byte, kFaultPayloadBytes> out) noexcept
{
    zero(out);
    using namespace fault_payload;
    store_u64le(out, kNativeSessionId, fields.native_session_id);
    store_u64le(out, kRuntimeGeneration, fields.runtime_generation);
    store_u64le(out, kFrameSequence, fields.frame_sequence);
    store_u64le(out, kSampleIndex, fields.sample_idx);
    store_u64le(out, kDeviceTick, fields.device_tick);
    store_u64le(out, kDetectedAtNanos, fields.detected_at_ns);
    store_u64le(out, kDataOrdinalAtFault, fields.data_ordinal_at_fault);
    store_u64le(out, kControlOrdinalAtFault, fields.control_ordinal_at_fault);
    store_u32le(out, kComponentId, fields.component_id);
    store_u32le(out, kDetail, fields.detail);
    store_u32le(out, kSchemaId, fields.schema_id);
    store_u32le(out, kClockDomain, fields.clock_domain);
    store_u32le(out, kSignalId, fields.signal_id);
    out[kFaultCode] = static_cast<std::byte>(fields.fault_code);
    out[kStreamStatus] = static_cast<std::byte>(fields.stream_status);
    out[kFaultStage] = static_cast<std::byte>(fields.fault_stage);
    out[kFaultOrigin] = static_cast<std::byte>(fields.origin);
    store_u32le(out, kRecorderReason, static_cast<std::uint32_t>(fields.recorder_reason));
}

bool decode_frame_payload(std::span<const std::byte> in, FramePayloadFields& out) noexcept
{
    if (in.size() < kFramePayloadBytes)
    {
        return false;
    }
    using namespace frame_payload;
    out.data_message_ordinal = load_u64le(in, kDataMessageOrdinal);
    out.native_session_id = load_u64le(in, kNativeSessionId);
    out.frame_sequence = load_u64le(in, kFrameSequence);
    out.frame_ordinal = load_u64le(in, kFrameOrdinal);
    out.host_received_ns = load_u64le(in, kHostReceivedNanos);
    out.source_tick = load_u64le(in, kSourceTick);
    out.valid_until_ns = load_u64le(in, kValidUntilNanos);
    out.native_schema_id = load_u32le(in, kNativeSchemaId);
    out.source_clock_domain = load_u32le(in, kSourceClockDomain);
    out.frame_flags = load_u32le(in, kFrameFlags);
    out.signal_block_count = load_u32le(in, kSignalBlockCount);
    out.recorded_signal_block_count = load_u32le(in, kRecordedSignalBlockCount);
    out.total_payload_byte_count = load_u64le(in, kTotalPayloadByteCount);
    out.runtime_accepted_host_time_ns = load_u64le(in, kRuntimeAcceptedHostNanos);
    return true;
}

bool decode_signal_block_payload(std::span<const std::byte> in,
                                 SignalBlockPayloadFields& out) noexcept
{
    if (in.size() < kSignalBlockHeaderPayloadBytes)
    {
        return false;
    }
    using namespace signal_block_payload;
    out.signal_block_ordinal = load_u64le(in, kSignalBlockOrdinal);
    out.data_message_ordinal = load_u64le(in, kDataMessageOrdinal);
    out.frame_ordinal = load_u64le(in, kFrameOrdinal);
    out.block_idx_in_frame = load_u32le(in, kBlockIndexInFrame);
    out.native_signal_id = load_u32le(in, kNativeSignalId);
    out.sample_idx_start = load_u64le(in, kSampleIndexStart);
    out.last_sample_idx = load_u64le(in, kLastSampleIndex);
    out.n_samples = load_u32le(in, kSampleCount);
    out.clock_sync_clock_domain = load_u32le(in, kClockSyncClockDomain);
    out.device_tick_start = load_u64le(in, kDeviceTickStart);
    out.observation_time_start_ns = load_u64le(in, kObservationTimeStartNanos);
    out.payload_offset = load_u64le(in, kPayloadOffset);
    out.payload_byte_count = load_u64le(in, kPayloadByteCount);
    out.clock_sync_device_tick_reference = load_u64le(in, kClockSyncDeviceTickReference);
    out.clock_sync_host_time_reference_ns = load_u64le(in, kClockSyncHostTimeReferenceNanos);
    out.clock_sync_rate_numerator = load_u64le(in, kClockSyncRateNumerator);
    out.clock_sync_rate_denominator = load_u64le(in, kClockSyncRateDenominator);
    out.clock_sync_uncertainty_ns = load_u64le(in, kClockSyncUncertaintyNanos);
    out.clock_sync_generation = load_u32le(in, kClockSyncGeneration);
    out.clock_sync_flags = load_u32le(in, kClockSyncFlags);
    return true;
}

bool decode_discontinuity_payload(std::span<const std::byte> in,
                                  DiscontinuityPayloadFields& out) noexcept
{
    if (in.size() < kDiscontinuityPayloadBytes)
    {
        return false;
    }
    using namespace discontinuity_payload;
    out.data_message_ordinal = load_u64le(in, kDataMessageOrdinal);
    out.native_session_id = load_u64le(in, kNativeSessionId);
    out.previous_frame_sequence = load_u64le(in, kPreviousFrameSequence);
    out.actual_frame_sequence = load_u64le(in, kActualFrameSequence);
    out.runtime_accepted_host_time_ns = load_u64le(in, kRuntimeAcceptedHostNanos);
    out.signal_gap_count = load_u32le(in, kSignalGapCount);
    out.reason = load_u32le(in, kReason);
    return true;
}

bool decode_signal_gap_payload(std::span<const std::byte> in, SignalGapPayloadFields& out) noexcept
{
    if (in.size() < kSignalGapPayloadBytes)
    {
        return false;
    }
    using namespace signal_gap_payload;
    out.signal_gap_ordinal = load_u64le(in, kSignalGapOrdinal);
    out.data_message_ordinal = load_u64le(in, kDataMessageOrdinal);
    out.expected_sample_idx = load_u64le(in, kExpectedSampleIndex);
    out.actual_sample_idx = load_u64le(in, kActualSampleIndex);
    out.missing_samples = load_u64le(in, kMissingSamples);
    out.expected_device_tick = load_u64le(in, kExpectedDeviceTick);
    out.actual_device_tick = load_u64le(in, kActualDeviceTick);
    out.gap_idx_in_message = load_u32le(in, kGapIndexInMessage);
    out.native_signal_id = load_u32le(in, kNativeSignalId);
    out.reason = load_u32le(in, kReason);
    out.gap_flags = load_u32le(in, kGapFlags);
    return true;
}

bool decode_control_payload(std::span<const std::byte> in, ControlPayloadFields& out) noexcept
{
    if (in.size() < kControlHeaderPayloadBytes)
    {
        return false;
    }
    using namespace control_payload;
    out.submission_ordinal = load_u64le(in, kSubmissionOrdinal);
    out.identity = load_u64le(in, kIdentity);
    out.time_ns = load_u64le(in, kTimeNanos);
    out.control_kind = load_u32le(in, kControlKind);
    out.clock_domain = load_u32le(in, kClockDomain);
    return true;
}

bool decode_fault_payload(std::span<const std::byte> in, FaultPayloadFields& out) noexcept
{
    if (in.size() < kFaultPayloadBytes)
    {
        return false;
    }
    using namespace fault_payload;
    out.native_session_id = load_u64le(in, kNativeSessionId);
    out.runtime_generation = load_u64le(in, kRuntimeGeneration);
    out.frame_sequence = load_u64le(in, kFrameSequence);
    out.sample_idx = load_u64le(in, kSampleIndex);
    out.device_tick = load_u64le(in, kDeviceTick);
    out.detected_at_ns = load_u64le(in, kDetectedAtNanos);
    out.data_ordinal_at_fault = load_u64le(in, kDataOrdinalAtFault);
    out.control_ordinal_at_fault = load_u64le(in, kControlOrdinalAtFault);
    out.component_id = load_u32le(in, kComponentId);
    out.detail = load_u32le(in, kDetail);
    out.schema_id = load_u32le(in, kSchemaId);
    out.clock_domain = load_u32le(in, kClockDomain);
    out.signal_id = load_u32le(in, kSignalId);
    out.fault_code = static_cast<std::uint8_t>(in[kFaultCode]);
    out.stream_status = static_cast<std::uint8_t>(in[kStreamStatus]);
    out.fault_stage = static_cast<std::uint8_t>(in[kFaultStage]);
    out.origin = static_cast<FaultOrigin>(in[kFaultOrigin]);
    out.recorder_reason = static_cast<RecorderFaultReason>(load_u32le(in, kRecorderReason));
    return true;
}

} // namespace neurale::recording
