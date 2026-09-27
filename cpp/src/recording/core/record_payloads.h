/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The interiors of the six recorder-owned spool record payloads.
///
/// The container specification (`specifications/native-spool/v1/README.md`,
/// section 13) deliberately leaves these undefined and carries them as opaque,
/// length-framed, checksummed byte strings. They are defined here because the
/// recorder core owns them, and they match the NRF ledger columns field for
/// field, so that the finalizer can build a `native-frames-v1`,
/// `native-signal-blocks-v1`, `native-discontinuities-v1`, or
/// `native-signal-gaps-v1` row without inventing a value or re-deriving one
/// from something that was never recorded.
///
/// Four columns are deliberately **not** here, because they are not facts the
/// capture path has:
///
/// - `stream_id` -- the plan maps a `native_signal_id` to it, so recording it
///   per block would store the same string once per block and give it two
///   sources of truth.
/// - `row_offset` -- a per-stream running position in the finalized target,
///   which does not exist until the finalizer lays the target out.
/// - `first_signal_block_ordinal` / `first_signal_gap_ordinal` -- anchors into
///   the child ledger, likewise a property of the target.
///
/// Everything is fixed-width little-endian, laid out at named offsets, and
/// written a byte at a time. Nothing here does `sizeof` on a struct, so the
/// encoding does not depend on host byte order, alignment, or padding.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace neurale::recording
{

// --- payload sizes ---------------------------------------------------------

inline constexpr std::size_t kFramePayloadBytes = 96;
/// The fixed part of a `signal_block` record. The block's sample bytes follow
/// it inside the same record, because a block and its samples are one thing
/// and splitting them would let a torn tail commit one without the other.
inline constexpr std::size_t kSignalBlockHeaderPayloadBytes = 136;
inline constexpr std::size_t kDiscontinuityPayloadBytes = 48;
inline constexpr std::size_t kSignalGapPayloadBytes = 72;
/// The fixed part of a `control` record; the caller's opaque body follows.
inline constexpr std::size_t kControlHeaderPayloadBytes = 32;
inline constexpr std::size_t kFaultPayloadBytes = 96;

// --- field offsets ---------------------------------------------------------

namespace frame_payload
{
inline constexpr std::size_t kDataMessageOrdinal = 0;
inline constexpr std::size_t kNativeSessionId = 8;
inline constexpr std::size_t kFrameSequence = 16;
inline constexpr std::size_t kFrameOrdinal = 24;
inline constexpr std::size_t kHostReceivedNanos = 32;
inline constexpr std::size_t kSourceTick = 40;
inline constexpr std::size_t kValidUntilNanos = 48;
inline constexpr std::size_t kNativeSchemaId = 56;
inline constexpr std::size_t kSourceClockDomain = 60;
inline constexpr std::size_t kFrameFlags = 64;
inline constexpr std::size_t kSignalBlockCount = 68;
inline constexpr std::size_t kRecordedSignalBlockCount = 72;
inline constexpr std::size_t kReserved = 76;
inline constexpr std::size_t kTotalPayloadByteCount = 80;
inline constexpr std::size_t kRuntimeAcceptedHostNanos = 88;
} // namespace frame_payload

namespace signal_block_payload
{
inline constexpr std::size_t kSignalBlockOrdinal = 0;
inline constexpr std::size_t kDataMessageOrdinal = 8;
inline constexpr std::size_t kFrameOrdinal = 16;
inline constexpr std::size_t kBlockIndexInFrame = 24;
inline constexpr std::size_t kNativeSignalId = 28;
inline constexpr std::size_t kSampleIndexStart = 32;
inline constexpr std::size_t kLastSampleIndex = 40;
inline constexpr std::size_t kSampleCount = 48;
inline constexpr std::size_t kClockSyncClockDomain = 52;
inline constexpr std::size_t kDeviceTickStart = 56;
inline constexpr std::size_t kObservationTimeStartNanos = 64;
inline constexpr std::size_t kPayloadOffset = 72;
inline constexpr std::size_t kPayloadByteCount = 80;
inline constexpr std::size_t kClockSyncDeviceTickReference = 88;
inline constexpr std::size_t kClockSyncHostTimeReferenceNanos = 96;
inline constexpr std::size_t kClockSyncRateNumerator = 104;
inline constexpr std::size_t kClockSyncRateDenominator = 112;
inline constexpr std::size_t kClockSyncUncertaintyNanos = 120;
inline constexpr std::size_t kClockSyncGeneration = 128;
inline constexpr std::size_t kClockSyncFlags = 132;
} // namespace signal_block_payload

namespace discontinuity_payload
{
inline constexpr std::size_t kDataMessageOrdinal = 0;
inline constexpr std::size_t kNativeSessionId = 8;
inline constexpr std::size_t kPreviousFrameSequence = 16;
inline constexpr std::size_t kActualFrameSequence = 24;
inline constexpr std::size_t kRuntimeAcceptedHostNanos = 32;
inline constexpr std::size_t kSignalGapCount = 40;
inline constexpr std::size_t kReason = 44;
} // namespace discontinuity_payload

namespace signal_gap_payload
{
inline constexpr std::size_t kSignalGapOrdinal = 0;
inline constexpr std::size_t kDataMessageOrdinal = 8;
inline constexpr std::size_t kExpectedSampleIndex = 16;
inline constexpr std::size_t kActualSampleIndex = 24;
inline constexpr std::size_t kMissingSamples = 32;
inline constexpr std::size_t kExpectedDeviceTick = 40;
inline constexpr std::size_t kActualDeviceTick = 48;
inline constexpr std::size_t kGapIndexInMessage = 56;
inline constexpr std::size_t kNativeSignalId = 60;
inline constexpr std::size_t kReason = 64;
inline constexpr std::size_t kGapFlags = 68;
} // namespace signal_gap_payload

namespace control_payload
{
inline constexpr std::size_t kSubmissionOrdinal = 0;
inline constexpr std::size_t kIdentity = 8;
inline constexpr std::size_t kTimeNanos = 16;
inline constexpr std::size_t kControlKind = 24;
inline constexpr std::size_t kClockDomain = 28;
/// The caller's opaque body starts here.
inline constexpr std::size_t kBody = 32;
} // namespace control_payload

namespace fault_payload
{
inline constexpr std::size_t kNativeSessionId = 0;
inline constexpr std::size_t kRuntimeGeneration = 8;
inline constexpr std::size_t kFrameSequence = 16;
inline constexpr std::size_t kSampleIndex = 24;
inline constexpr std::size_t kDeviceTick = 32;
inline constexpr std::size_t kDetectedAtNanos = 40;
inline constexpr std::size_t kDataOrdinalAtFault = 48;
inline constexpr std::size_t kControlOrdinalAtFault = 56;
inline constexpr std::size_t kComponentId = 64;
inline constexpr std::size_t kDetail = 68;
inline constexpr std::size_t kSchemaId = 72;
inline constexpr std::size_t kClockDomain = 76;
inline constexpr std::size_t kSignalId = 80;
inline constexpr std::size_t kFaultCode = 84;
inline constexpr std::size_t kStreamStatus = 85;
inline constexpr std::size_t kFaultStage = 86;
inline constexpr std::size_t kFaultOrigin = 87;
inline constexpr std::size_t kRecorderReason = 88;
inline constexpr std::size_t kReserved = 92;
} // namespace fault_payload

/// Who raised a fault. The distinction matters because contract section 4.3
/// routes a recorder fault into the runtime fault path of section 4.2, and a
/// reader has to be able to tell which end the session actually failed at.
enum class FaultOrigin : std::uint8_t
{
    runtime = 0,
    recorder = 1,
};

/// Why the recorder itself faulted. `none` is what a harvested runtime fault
/// carries: the recorder did not fail, it recorded that something else did.
enum class RecorderFaultReason : std::uint8_t
{
    none = 0,
    /// A bounded queue was full and the recorder is lossless-until-fault.
    data_queue_saturated = 1,
    control_queue_saturated = 2,
    /// The item did not satisfy the plan: unknown signal, oversized block,
    /// block extent outside the frame payload, wrong session.
    plan_violation = 3,
    /// The spool writer failed, so accepted items have no path to stage 3.
    spool_writer_failed = 4,
    /// The drain did not finish inside the configured bound (section 4.7).
    drain_timeout = 5,
    /// An observer edge refused an item before runtime acceptance. For a
    /// critical recorder this is a fault, not a statistic.
    edge_rejection = 6,
    /// The readiness gate did not pass.
    readiness_gate_failed = 7,
};

[[nodiscard]] std::string_view to_string(FaultOrigin origin) noexcept;
[[nodiscard]] std::string_view to_string(RecorderFaultReason reason) noexcept;

// --- the values each encoder needs -----------------------------------------

struct FramePayloadFields
{
    std::uint64_t data_message_ordinal{};
    std::uint64_t native_session_id{};
    std::uint64_t frame_sequence{};
    std::uint64_t frame_ordinal{};
    std::uint64_t host_received_ns{};
    std::uint64_t source_tick{};
    std::uint64_t valid_until_ns{};
    std::uint64_t total_payload_byte_count{};
    std::uint64_t runtime_accepted_host_time_ns{};
    std::uint32_t native_schema_id{};
    std::uint32_t source_clock_domain{};
    std::uint32_t frame_flags{};
    std::uint32_t signal_block_count{};
    std::uint32_t recorded_signal_block_count{};
};

struct SignalBlockPayloadFields
{
    std::uint64_t signal_block_ordinal{};
    std::uint64_t data_message_ordinal{};
    std::uint64_t frame_ordinal{};
    std::uint64_t sample_idx_start{};
    std::uint64_t last_sample_idx{};
    std::uint64_t device_tick_start{};
    std::uint64_t observation_time_start_ns{};
    std::uint64_t payload_offset{};
    std::uint64_t payload_byte_count{};
    std::uint64_t clock_sync_device_tick_reference{};
    std::uint64_t clock_sync_host_time_reference_ns{};
    std::uint64_t clock_sync_rate_numerator{};
    std::uint64_t clock_sync_rate_denominator{};
    std::uint64_t clock_sync_uncertainty_ns{};
    std::uint32_t block_idx_in_frame{};
    std::uint32_t native_signal_id{};
    std::uint32_t n_samples{};
    std::uint32_t clock_sync_clock_domain{};
    std::uint32_t clock_sync_generation{};
    std::uint32_t clock_sync_flags{};
};

struct DiscontinuityPayloadFields
{
    std::uint64_t data_message_ordinal{};
    std::uint64_t native_session_id{};
    std::uint64_t previous_frame_sequence{};
    std::uint64_t actual_frame_sequence{};
    std::uint64_t runtime_accepted_host_time_ns{};
    std::uint32_t signal_gap_count{};
    std::uint32_t reason{};
};

struct SignalGapPayloadFields
{
    std::uint64_t signal_gap_ordinal{};
    std::uint64_t data_message_ordinal{};
    std::uint64_t expected_sample_idx{};
    std::uint64_t actual_sample_idx{};
    std::uint64_t missing_samples{};
    std::uint64_t expected_device_tick{};
    std::uint64_t actual_device_tick{};
    std::uint32_t gap_idx_in_message{};
    std::uint32_t native_signal_id{};
    std::uint32_t reason{};
    std::uint32_t gap_flags{};
};

struct ControlPayloadFields
{
    std::uint64_t submission_ordinal{};
    std::uint64_t identity{};
    std::uint64_t time_ns{};
    std::uint32_t control_kind{};
    std::uint32_t clock_domain{};
};

struct FaultPayloadFields
{
    std::uint64_t native_session_id{};
    std::uint64_t runtime_generation{};
    std::uint64_t frame_sequence{};
    std::uint64_t sample_idx{};
    std::uint64_t device_tick{};
    std::uint64_t detected_at_ns{};
    std::uint64_t data_ordinal_at_fault{};
    std::uint64_t control_ordinal_at_fault{};
    std::uint32_t component_id{};
    std::uint32_t detail{};
    std::uint32_t schema_id{};
    std::uint32_t clock_domain{};
    std::uint32_t signal_id{};
    std::uint8_t fault_code{};
    std::uint8_t stream_status{};
    std::uint8_t fault_stage{};
    FaultOrigin origin{FaultOrigin::runtime};
    RecorderFaultReason recorder_reason{RecorderFaultReason::none};
};

// --- encoders --------------------------------------------------------------
//
// Each writes exactly its fixed size and nothing else. They are `noexcept`,
// take no allocation, and are safe to call from the critical callback -- which
// is where the frame, block, discontinuity, and gap encoders actually run.

void encode_frame_payload(const FramePayloadFields& fields,
                          std::span<std::byte, kFramePayloadBytes> out) noexcept;
void encode_signal_block_payload(const SignalBlockPayloadFields& fields,
                                 std::span<std::byte, kSignalBlockHeaderPayloadBytes> out) noexcept;
void encode_discontinuity_payload(const DiscontinuityPayloadFields& fields,
                                  std::span<std::byte, kDiscontinuityPayloadBytes> out) noexcept;
void encode_signal_gap_payload(const SignalGapPayloadFields& fields,
                               std::span<std::byte, kSignalGapPayloadBytes> out) noexcept;
void encode_control_payload(const ControlPayloadFields& fields,
                            std::span<std::byte, kControlHeaderPayloadBytes> out) noexcept;
void encode_fault_payload(const FaultPayloadFields& fields,
                          std::span<std::byte, kFaultPayloadBytes> out) noexcept;

// --- decoders --------------------------------------------------------------
//
// Only the tests and, later, the finalizer read these back. They return false
// on a short span rather than trusting a length they did not check.

[[nodiscard]] bool decode_frame_payload(std::span<const std::byte> in,
                                        FramePayloadFields& out) noexcept;
[[nodiscard]] bool decode_signal_block_payload(std::span<const std::byte> in,
                                               SignalBlockPayloadFields& out) noexcept;
[[nodiscard]] bool decode_discontinuity_payload(std::span<const std::byte> in,
                                                DiscontinuityPayloadFields& out) noexcept;
[[nodiscard]] bool decode_signal_gap_payload(std::span<const std::byte> in,
                                             SignalGapPayloadFields& out) noexcept;
[[nodiscard]] bool decode_control_payload(std::span<const std::byte> in,
                                          ControlPayloadFields& out) noexcept;
[[nodiscard]] bool decode_fault_payload(std::span<const std::byte> in,
                                        FaultPayloadFields& out) noexcept;

} // namespace neurale::recording
