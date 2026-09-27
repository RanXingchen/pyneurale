/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/realtime_config.h>

#include <limits>
#include <stdexcept>

namespace neurale::streaming
{
namespace
{

[[nodiscard]] std::size_t checked_add(std::size_t left, std::size_t right)
{
    if (right > std::numeric_limits<std::size_t>::max() - left)
    {
        throw std::overflow_error("buffer pool capacity overflows size_t");
    }
    return left + right;
}

} // namespace

std::size_t PoolCapacityBudget::required_buffer_count() const
{
    auto total = source_owned;
    total = checked_add(total, ingress_capacity);
    total = checked_add(total, processor_owned);
    total = checked_add(total, critical_edge_capacity);
    total = checked_add(total, actuator_owned);
    total = checked_add(total, observer_edge_capacity);
    return checked_add(total, reserve);
}

void RealtimeConfig::validate() const
{
    platform.validate();
    if (pool_capacity.source_owned == 0 || pool_capacity.ingress_capacity == 0 ||
        pool_capacity.processor_owned == 0 || pool_capacity.critical_edge_capacity == 0 ||
        pool_capacity.actuator_owned == 0 || required_buffer_count() == 0 || buffer_size == 0 ||
        max_signal_blocks == 0 || discontinuity_capacity == 0 || gaps_per_discontinuity == 0 ||
        max_process_outputs == 0 || fault_history_capacity == 0)
    {
        throw std::invalid_argument(
            "source, ingress, processor, critical edge, actuator, and storage "
            "capacities must be positive");
    }
    if (source_stall_timeout.count() <= 0 || max_ingress_dwell.count() <= 0 ||
        processor_execution_deadline.count() <= 0 || max_source_to_actuator_age.count() <= 0 ||
        max_output_age.count() <= 0 || actuator_deadline.count() <= 0 ||
        shutdown_deadline.count() <= 0 || watchdog_period.count() <= 0)
    {
        throw std::invalid_argument("realtime deadlines must be positive");
    }
}

} // namespace neurale::streaming
