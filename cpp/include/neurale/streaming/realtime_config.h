/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>

#include <neurale/streaming/clock.h>
#include <neurale/streaming/realtime_thread.h>

namespace neurale::streaming
{

/// Maximum number of pool leases retained concurrently by each topology stage.
struct PoolCapacityBudget
{
    std::size_t source_owned{};
    std::size_t ingress_capacity{};
    std::size_t processor_owned{};
    std::size_t critical_edge_capacity{};
    std::size_t actuator_owned{};
    std::size_t observer_edge_capacity{};
    std::size_t reserve{};

    [[nodiscard]] std::size_t required_buffer_count() const;
};

/// Capacity and deadline values validated before native runtime startup.
struct RealtimeConfig
{
    bool automatic_resources{};
    RealtimePlatformConfig platform{};
    PoolCapacityBudget pool_capacity{};
    std::size_t buffer_size{};
    std::size_t max_signal_blocks{};
    std::size_t discontinuity_capacity{};
    std::size_t gaps_per_discontinuity{};
    std::size_t max_process_outputs{1};
    std::size_t max_flush_outputs{};
    std::size_t fault_history_capacity{8};
    RealtimeDuration source_stall_timeout{1'000'000'000};
    RealtimeDuration max_ingress_dwell{100'000'000};
    RealtimeDuration processor_execution_deadline{100'000'000};
    RealtimeDuration max_source_to_actuator_age{100'000'000};
    RealtimeDuration max_output_age{1'000'000'000};
    RealtimeDuration actuator_deadline{100'000'000};
    RealtimeDuration shutdown_deadline{1'000'000'000};
    RealtimeDuration watchdog_period{10'000'000};

    [[nodiscard]] std::size_t required_buffer_count() const
    {
        return pool_capacity.required_buffer_count();
    }

    void validate() const;
};

} // namespace neurale::streaming
