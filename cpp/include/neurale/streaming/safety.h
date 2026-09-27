/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

#include <neurale/streaming/fault.h>

namespace neurale::streaming
{

enum class SafetyReason : std::uint8_t
{
    startup,
    explicit_stop,
    end_of_stream,
    source_stall,
    ingress_dwell,
    processor_deadline,
    output_stale,
    shutdown_timeout,
    actuator_failure,
    critical_observer_failure,
    runtime_fault,
};

/// Native safety boundary. Implementations must be bounded and allocation-free.
class SafetyController
{
  public:
    virtual ~SafetyController() = default;

    virtual StreamStatus inhibit(SafetyReason reason) noexcept = 0;
    virtual StreamStatus release() noexcept = 0;
};

[[nodiscard]] SafetyController& default_safety_controller() noexcept;

} // namespace neurale::streaming
