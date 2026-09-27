/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include <neurale/streaming/clock.h>

namespace neurale::streaming
{

/// Fixed-size snapshot copied from native runtime counters off the hot path.
struct RuntimeStats
{
    std::uint64_t frames_acquired{};
    std::uint64_t frames_read{};
    std::uint64_t frames_processed{};
    std::uint64_t frames_published{};
    std::uint64_t frames_consumed{};
    std::uint64_t zero_output_frames{};
    std::uint64_t processor_outputs{};
    std::uint64_t flush_outputs{};
    std::uint64_t discontinuities{};
    std::uint64_t end_of_streams{};
    std::uint64_t ingress_high_water_mark{};
    std::uint64_t queue_overruns{};
    std::uint64_t pool_exhaustions{};
    std::uint64_t buffer_exhaustions{};
    std::uint64_t aborted_frames{};
    std::uint64_t deadline_faults{};
    std::uint64_t safety_inhibitions{};
    std::uint64_t faults{};
    std::uint64_t secondary_faults{};
    std::uint64_t fault_history_dropped{};
    std::uint64_t actuator_commands_enqueued{};
    std::uint64_t actuator_commands_applied{};
    std::uint64_t actuator_queue_high_water_mark{};
    std::uint64_t actuator_failures{};
    std::uint64_t actuator_deadline_misses{};
    std::uint64_t observer_dispatch_drops{};
    HostTimeNs max_ingress_dwell_ns{};
    HostTimeNs max_processor_execution_ns{};
    HostTimeNs max_source_to_actuator_ns{};
};

struct RuntimeHeartbeatSnapshot
{
    HostTimeNs last_source_heartbeat{};
    HostTimeNs current_processor_start{};
    HostTimeNs last_processor_completion{};
    HostTimeNs last_valid_output{};
    HostTimeNs current_actuator_start{};
    HostTimeNs last_inhibit{};
    std::uint64_t runtime_generation{};
    bool processor_active{};
    bool actuator_active{};
    bool output_valid{};
    bool safety_inhibited{true};
};

class RuntimeCounters
{
  public:
    void reset() noexcept
    {
        frames_acquired_.store(0, std::memory_order_relaxed);
        frames_processed_.store(0, std::memory_order_relaxed);
        frames_published_.store(0, std::memory_order_relaxed);
        frames_consumed_.store(0, std::memory_order_relaxed);
        zero_output_frames_.store(0, std::memory_order_relaxed);
        processor_outputs_.store(0, std::memory_order_relaxed);
        flush_outputs_.store(0, std::memory_order_relaxed);
        discontinuities_.store(0, std::memory_order_relaxed);
        end_of_streams_.store(0, std::memory_order_relaxed);
        ingress_high_water_mark_.store(0, std::memory_order_relaxed);
        queue_overruns_.store(0, std::memory_order_relaxed);
        pool_exhaustions_.store(0, std::memory_order_relaxed);
        aborted_frames_.store(0, std::memory_order_relaxed);
        deadline_faults_.store(0, std::memory_order_relaxed);
        safety_inhibitions_.store(0, std::memory_order_relaxed);
        faults_.store(0, std::memory_order_relaxed);
        secondary_faults_.store(0, std::memory_order_relaxed);
        fault_history_dropped_.store(0, std::memory_order_relaxed);
        actuator_commands_enqueued_.store(0, std::memory_order_relaxed);
        actuator_commands_applied_.store(0, std::memory_order_relaxed);
        actuator_queue_high_water_mark_.store(0, std::memory_order_relaxed);
        actuator_failures_.store(0, std::memory_order_relaxed);
        actuator_deadline_misses_.store(0, std::memory_order_relaxed);
        observer_dispatch_drops_.store(0, std::memory_order_relaxed);
        max_ingress_dwell_ns_.store(0, std::memory_order_relaxed);
        max_processor_execution_ns_.store(0, std::memory_order_relaxed);
        max_source_to_actuator_ns_.store(0, std::memory_order_relaxed);
    }

    void acquired() noexcept
    {
        increment(frames_acquired_);
    }
    void processed() noexcept
    {
        increment(frames_processed_);
    }
    void published(bool flushing) noexcept
    {
        increment(frames_published_);
        increment(flushing ? flush_outputs_ : processor_outputs_);
    }
    void zero_output() noexcept
    {
        increment(zero_output_frames_);
    }
    void discontinuity() noexcept
    {
        increment(discontinuities_);
    }
    void end_of_stream() noexcept
    {
        increment(end_of_streams_);
    }
    void queue_overrun() noexcept
    {
        increment(queue_overruns_);
    }
    void pool_exhaustion() noexcept
    {
        increment(pool_exhaustions_);
    }
    void aborted_frame() noexcept
    {
        increment(aborted_frames_);
    }
    void deadline_fault() noexcept
    {
        increment(deadline_faults_);
    }
    void safety_inhibition() noexcept
    {
        increment(safety_inhibitions_);
    }
    void fault() noexcept
    {
        increment(faults_);
    }
    void secondary_fault() noexcept
    {
        increment(secondary_faults_);
    }
    void fault_history_dropped() noexcept
    {
        increment(fault_history_dropped_);
    }
    void actuator_enqueued() noexcept
    {
        increment(actuator_commands_enqueued_);
    }
    void actuator_applied() noexcept
    {
        increment(actuator_commands_applied_);
        increment(frames_consumed_);
    }
    void actuator_failure() noexcept
    {
        increment(actuator_failures_);
    }
    void actuator_deadline_miss() noexcept
    {
        increment(actuator_deadline_misses_);
    }
    void observer_dispatch_drop() noexcept
    {
        increment(observer_dispatch_drops_);
    }

    void observe_ingress_size(std::size_t size) noexcept
    {
        auto current = ingress_high_water_mark_.load(std::memory_order_relaxed);
        while (current < size && !ingress_high_water_mark_.compare_exchange_weak(
                                     current, static_cast<std::uint64_t>(size),
                                     std::memory_order_relaxed, std::memory_order_relaxed))
        {
        }
    }

    void observe_actuator_size(std::size_t size) noexcept
    {
        observe_max(actuator_queue_high_water_mark_, size);
    }
    void observe_ingress_dwell(HostTimeNs duration_ns) noexcept
    {
        observe_max(max_ingress_dwell_ns_, duration_ns);
    }
    void observe_processor_execution(HostTimeNs duration_ns) noexcept
    {
        observe_max(max_processor_execution_ns_, duration_ns);
    }
    void observe_source_to_actuator(HostTimeNs duration_ns) noexcept
    {
        observe_max(max_source_to_actuator_ns_, duration_ns);
    }

    [[nodiscard]] RuntimeStats snapshot() const noexcept
    {
        const auto acquired = load(frames_acquired_);
        const auto exhausted = load(pool_exhaustions_);
        return RuntimeStats{
            .frames_acquired = acquired,
            .frames_read = acquired,
            .frames_processed = load(frames_processed_),
            .frames_published = load(frames_published_),
            .frames_consumed = load(frames_consumed_),
            .zero_output_frames = load(zero_output_frames_),
            .processor_outputs = load(processor_outputs_),
            .flush_outputs = load(flush_outputs_),
            .discontinuities = load(discontinuities_),
            .end_of_streams = load(end_of_streams_),
            .ingress_high_water_mark = load(ingress_high_water_mark_),
            .queue_overruns = load(queue_overruns_),
            .pool_exhaustions = exhausted,
            .buffer_exhaustions = exhausted,
            .aborted_frames = load(aborted_frames_),
            .deadline_faults = load(deadline_faults_),
            .safety_inhibitions = load(safety_inhibitions_),
            .faults = load(faults_),
            .secondary_faults = load(secondary_faults_),
            .fault_history_dropped = load(fault_history_dropped_),
            .actuator_commands_enqueued = load(actuator_commands_enqueued_),
            .actuator_commands_applied = load(actuator_commands_applied_),
            .actuator_queue_high_water_mark = load(actuator_queue_high_water_mark_),
            .actuator_failures = load(actuator_failures_),
            .actuator_deadline_misses = load(actuator_deadline_misses_),
            .observer_dispatch_drops = load(observer_dispatch_drops_),
            .max_ingress_dwell_ns = load(max_ingress_dwell_ns_),
            .max_processor_execution_ns = load(max_processor_execution_ns_),
            .max_source_to_actuator_ns = load(max_source_to_actuator_ns_),
        };
    }

  private:
    static void increment(std::atomic<std::uint64_t>& value) noexcept
    {
        value.fetch_add(1, std::memory_order_relaxed);
    }
    static void observe_max(std::atomic<std::uint64_t>& value, std::uint64_t observed) noexcept
    {
        auto current = value.load(std::memory_order_relaxed);
        while (current < observed &&
               !value.compare_exchange_weak(current, observed, std::memory_order_relaxed,
                                            std::memory_order_relaxed))
        {
        }
    }
    [[nodiscard]] static std::uint64_t load(const std::atomic<std::uint64_t>& value) noexcept
    {
        return value.load(std::memory_order_relaxed);
    }

    std::atomic<std::uint64_t> frames_acquired_{};
    std::atomic<std::uint64_t> frames_processed_{};
    std::atomic<std::uint64_t> frames_published_{};
    std::atomic<std::uint64_t> frames_consumed_{};
    std::atomic<std::uint64_t> zero_output_frames_{};
    std::atomic<std::uint64_t> processor_outputs_{};
    std::atomic<std::uint64_t> flush_outputs_{};
    std::atomic<std::uint64_t> discontinuities_{};
    std::atomic<std::uint64_t> end_of_streams_{};
    std::atomic<std::uint64_t> ingress_high_water_mark_{};
    std::atomic<std::uint64_t> queue_overruns_{};
    std::atomic<std::uint64_t> pool_exhaustions_{};
    std::atomic<std::uint64_t> aborted_frames_{};
    std::atomic<std::uint64_t> deadline_faults_{};
    std::atomic<std::uint64_t> safety_inhibitions_{};
    std::atomic<std::uint64_t> faults_{};
    std::atomic<std::uint64_t> secondary_faults_{};
    std::atomic<std::uint64_t> fault_history_dropped_{};
    std::atomic<std::uint64_t> actuator_commands_enqueued_{};
    std::atomic<std::uint64_t> actuator_commands_applied_{};
    std::atomic<std::uint64_t> actuator_queue_high_water_mark_{};
    std::atomic<std::uint64_t> actuator_failures_{};
    std::atomic<std::uint64_t> actuator_deadline_misses_{};
    std::atomic<std::uint64_t> observer_dispatch_drops_{};
    std::atomic<std::uint64_t> max_ingress_dwell_ns_{};
    std::atomic<std::uint64_t> max_processor_execution_ns_{};
    std::atomic<std::uint64_t> max_source_to_actuator_ns_{};
};

using StreamStatsSnapshot = RuntimeStats;

} // namespace neurale::streaming
