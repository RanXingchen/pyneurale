/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

#include <neurale/streaming/clock.h>

namespace neurale::streaming
{

/// Non-throwing result returned by native realtime callbacks.
enum class StreamStatus : std::uint8_t
{
    ok,
    end_of_stream,
    would_block,
    buffer_exhausted,
    queue_overflow,
    discontinuity,
    invalid_frame,
    source_failure,
    processor_failure,
    consumer_failure,
    actuator_failure,
    observer_overrun,
    stopped,
    invalid_state,
    output_limit,
    deadline_exceeded,
    safety_failure,
    realtime_configuration_failed,
};

enum class FaultCode : std::uint8_t
{
    none,
    source_read,
    source_reset,
    frame_pool_exhausted,
    continuity_validation,
    processor_discontinuity,
    processor_process,
    processor_flush,
    processor_reset,
    output_contract,
    output_validation,
    output_pool_exhausted,
    consumer_discontinuity,
    consumer_consume,
    consumer_flush,
    consumer_reset,
    queue_overrun,
    runtime_start,
    source_stall,
    ingress_dwell_timeout,
    processor_deadline,
    actuator_input_stale,
    output_stale,
    shutdown_timeout,
    safety_controller_failure,
    actuator_queue_overrun,
    actuator_deadline,
    actuator_write,
    actuator_flush,
    actuator_reset,
    observer_dispatch_overrun,
    critical_observer_overrun,
    critical_observer_failure,
    realtime_configuration,
    processor_prepare,
};

/// Native runtime stage that produced a fault.
enum class FaultStage : std::uint8_t
{
    source,
    continuity,
    processor,
    output,
    consumer,
    actuator,
    observer,
    runtime,
};

/// Fixed-size fault record writable without allocation or exception handling.
struct FaultRecord
{
    FaultCode code{FaultCode::none};
    StreamStatus status{StreamStatus::ok};
    FaultStage stage{FaultStage::runtime};
    std::uint32_t component_id{};
    std::uint32_t detail{};
    SessionId session_id{};
    std::uint64_t runtime_generation{};
    std::uint64_t frame_sequence{};
    std::uint32_t schema_id{};
    ClockDomainId clock_domain{};
    std::uint32_t signal_id{};
    SampleIndex sample_idx{};
    DeviceTick device_tick{};
    HostTimeNs detected_at_ns{};
};

} // namespace neurale::streaming
