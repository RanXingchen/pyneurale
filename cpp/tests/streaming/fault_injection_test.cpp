/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "check_returns.h"
#include "fault_injection.h"

#include <neurale/streaming/runtime.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string_view>
#include <thread>
#include <utility>

using namespace neurale::streaming;
using namespace neurale::streaming::test;

namespace
{

[[nodiscard]] bool prepare_arm_start(NativeStreamRunner& runtime)
{
    return runtime.prepare() == StreamStatus::ok && runtime.arm() == StreamStatus::ok &&
           runtime.start() == StreamStatus::ok;
}

[[nodiscard]] bool verify_failed_runtime(const NativeStreamRunner& runtime,
                                         const FaultSafetyController& safety, FaultCode expected,
                                         FaultRecord original)
{
    const auto current = runtime.primary_fault();
    if (!current.has_value() || current->code != expected || current->code != original.code ||
        current->status != original.status || !safety.inhibited() ||
        !runtime.heartbeat().safety_inhibited || runtime.outstanding_frames() != 0 ||
        runtime.outstanding_discontinuities() != 0)
    {
        return false;
    }
    std::array<FaultRecord, 32> history{};
    return runtime.copy_fault_history(history) <= runtime.fault_history_capacity();
}

int test_source_validation_faults()
{
    for (const auto scenario : {
             std::pair{FaultPoint::source_malformed_header, FaultCode::continuity_validation},
             std::pair{FaultPoint::source_restart, FaultCode::continuity_validation},
         })
    {
        const FaultInjectionConfig injection{scenario.first, 1, 8};
        InjectedSource source{injection};
        InjectedProcessor processor{injection};
        InjectedActuator actuator{injection};
        source.pace_with(actuator.applied_counter());
        FaultClock clock;
        FaultSafetyController safety;
        NativeStreamRunner runtime{
            make_fault_schema(), make_fault_config(), source, processor, actuator, clock, safety};

        CHECK(prepare_arm_start(runtime));
        CHECK(runtime.join() == StreamStatus::invalid_frame);
        const auto primary = runtime.primary_fault();
        CHECK(primary.has_value());
        CHECK(verify_failed_runtime(runtime, safety, scenario.second, *primary));
        CHECK(actuator.attempts() == 1);
    }
    return 0;
}

int test_sample_gap_is_ordered_and_nonfatal()
{
    const FaultInjectionConfig injection{FaultPoint::sample_idx_gap, 3, 12};
    InjectedSource source{injection};
    InjectedProcessor processor{injection};
    InjectedActuator actuator{injection};
    source.pace_with(actuator.applied_counter());
    NativeStreamRunner runtime{make_fault_schema(), make_fault_config(), source, processor,
                               actuator};

    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.run() == StreamStatus::ok);
    CHECK(!runtime.primary_fault().has_value());
    CHECK(processor.discontinuities() == 1);
    CHECK(actuator.attempts() == injection.n_frames);
    CHECK(runtime.stats().discontinuities == 1);
    CHECK(runtime.outstanding_frames() == 0);
    CHECK(runtime.outstanding_discontinuities() == 0);
    return 0;
}

int test_processor_faults_recover_every_lease()
{
    for (const auto point : {FaultPoint::processor_fatal, FaultPoint::pool_lease_leak})
    {
        const FaultInjectionConfig injection{point, 1, 8};
        InjectedSource source{injection};
        InjectedProcessor processor{injection};
        InjectedActuator actuator{injection};
        source.pace_with(actuator.applied_counter());
        FaultClock clock;
        FaultSafetyController safety;
        NativeStreamRunner runtime{
            make_fault_schema(), make_fault_config(), source, processor, actuator, clock, safety};

        CHECK(prepare_arm_start(runtime));
        const auto expected_status = point == FaultPoint::pool_lease_leak
                                         ? StreamStatus::invalid_state
                                         : StreamStatus::processor_failure;
        const auto expected_fault = point == FaultPoint::pool_lease_leak
                                        ? FaultCode::output_contract
                                        : FaultCode::processor_process;
        CHECK(runtime.join() == expected_status);
        const auto primary = runtime.primary_fault();
        CHECK(primary.has_value());
        CHECK(verify_failed_runtime(runtime, safety, expected_fault, *primary));
    }
    return 0;
}

int test_ingress_overrun_is_critical()
{
    const FaultInjectionConfig injection{FaultPoint::ingress_overrun, 0, 32};
    InjectedSource source{injection};
    InjectedProcessor processor{injection};
    InjectedActuator actuator{injection};
    std::atomic<std::uint64_t> source_limit{};
    source.pace_with(source_limit);
    FaultClock clock;
    FaultSafetyController safety;
    auto config = make_fault_config(1);
    NativeStreamRunner runtime{
        make_fault_schema(), config, source, processor, actuator, clock, safety};

    CHECK(prepare_arm_start(runtime));
    CHECK(wait_until([&] { return processor.blocked(); }));
    source_limit.store(injection.n_frames, std::memory_order_release);
    CHECK(wait_until(
        [&]
        {
            const auto fault = runtime.primary_fault();
            return fault.has_value() && fault->code == FaultCode::queue_overrun;
        }));
    const auto primary = *runtime.primary_fault();
    processor.release();
    CHECK(runtime.join() == StreamStatus::queue_overflow);
    CHECK(verify_failed_runtime(runtime, safety, FaultCode::queue_overrun, primary));
    CHECK(runtime.stats().queue_overruns == 1);
    return 0;
}

int test_source_and_stage_deadlines()
{
    {
        const FaultInjectionConfig injection{FaultPoint::source_stall, 0, 1};
        InjectedSource source{injection};
        InjectedProcessor processor{injection};
        InjectedActuator actuator{injection};
        FaultClock clock;
        FaultSafetyController safety;
        auto config = make_fault_config();
        config.source_stall_timeout = RealtimeDuration{10};
        NativeStreamRunner runtime{
            make_fault_schema(), config, source, processor, actuator, clock, safety};
        CHECK(prepare_arm_start(runtime));
        CHECK(wait_until([&] { return source.blocked() && clock.waiters() != 0; }));
        clock.advance(10);
        CHECK(wait_until([&] { return runtime.primary_fault().has_value(); }));
        const auto primary = *runtime.primary_fault();
        CHECK(runtime.join() == StreamStatus::deadline_exceeded);
        CHECK(verify_failed_runtime(runtime, safety, FaultCode::source_stall, primary));
    }
    {
        const FaultInjectionConfig injection{FaultPoint::processor_deadline, 0, 1};
        InjectedSource source{injection};
        InjectedProcessor processor{injection};
        InjectedActuator actuator{injection};
        FaultClock clock;
        FaultSafetyController safety;
        auto config = make_fault_config();
        config.processor_execution_deadline = RealtimeDuration{10};
        NativeStreamRunner runtime{
            make_fault_schema(), config, source, processor, actuator, clock, safety};
        CHECK(prepare_arm_start(runtime));
        CHECK(wait_until([&] { return processor.blocked() && clock.waiters() != 0; }));
        clock.advance(10);
        CHECK(wait_until([&] { return runtime.primary_fault().has_value(); }));
        const auto primary = *runtime.primary_fault();
        processor.release();
        CHECK(runtime.join() == StreamStatus::deadline_exceeded);
        CHECK(verify_failed_runtime(runtime, safety, FaultCode::processor_deadline, primary));
    }
    {
        const FaultInjectionConfig injection{FaultPoint::actuator_timeout, 0, 1};
        InjectedSource source{injection};
        InjectedProcessor processor{injection};
        InjectedActuator actuator{injection};
        FaultClock clock;
        FaultSafetyController safety;
        auto config = make_fault_config();
        config.actuator_deadline = RealtimeDuration{10};
        NativeStreamRunner runtime{
            make_fault_schema(), config, source, processor, actuator, clock, safety};
        CHECK(prepare_arm_start(runtime));
        CHECK(wait_until([&] { return actuator.blocked() && clock.waiters() != 0; }));
        clock.advance(10);
        CHECK(wait_until([&] { return runtime.primary_fault().has_value(); }));
        const auto primary = *runtime.primary_fault();
        CHECK(runtime.join() == StreamStatus::deadline_exceeded);
        CHECK(verify_failed_runtime(runtime, safety, FaultCode::actuator_deadline, primary));
        CHECK(actuator.attempts() == 1);
    }
    return 0;
}

int test_actuator_failure_stops_new_commands()
{
    const FaultInjectionConfig injection{FaultPoint::actuator_failure, 2, 32};
    InjectedSource source{injection};
    InjectedProcessor processor{injection};
    InjectedActuator actuator{injection};
    FaultClock clock;
    FaultSafetyController safety;
    auto config = make_fault_config(64);
    config.pool_capacity.critical_edge_capacity = 64;
    NativeStreamRunner runtime{
        make_fault_schema(), config, source, processor, actuator, clock, safety};

    CHECK(prepare_arm_start(runtime));
    CHECK(wait_until([&] { return runtime.primary_fault().has_value(); }));
    const auto primary = *runtime.primary_fault();
    const auto attempts_after_fault = actuator.attempts();
    CHECK(runtime.join() == StreamStatus::actuator_failure);
    CHECK(verify_failed_runtime(runtime, safety, FaultCode::actuator_write, primary));
    CHECK(attempts_after_fault == 3);
    CHECK(actuator.attempts() == attempts_after_fault);
    CHECK(actuator.applied_counter().load(std::memory_order_acquire) == 2);
    return 0;
}

int test_blocked_observer_is_noncritical()
{
    const FaultInjectionConfig injection{FaultPoint::observer_blocked, 0, 64};
    InjectedSource source{injection};
    InjectedProcessor processor{injection};
    InjectedActuator actuator{injection};
    InjectedObserver observer{injection};
    source.pace_with(actuator.applied_counter());
    NativeStreamRunner runtime{make_fault_schema(), make_fault_config(), source, processor,
                               actuator};
    const ObserverEdgeConfig edge{1, 2, 4, ObserverDropPolicy::drop_newest, false};

    CHECK(runtime.add_observer(observer, edge) == StreamStatus::ok);
    CHECK(prepare_arm_start(runtime));
    CHECK(wait_until([&] { return observer.blocked(); }));
    CHECK(wait_until(
        [&]
        {
            return actuator.applied_counter().load(std::memory_order_acquire) == injection.n_frames;
        }));
    CHECK(!runtime.primary_fault().has_value());
    const auto attempts = actuator.attempts();
    observer.release();
    CHECK(runtime.join() == StreamStatus::ok);
    CHECK(actuator.attempts() == attempts);
    CHECK(runtime.observer_stats(1)->dropped != 0);
    CHECK(runtime.outstanding_frames() == 0);
    return 0;
}

int test_delayed_watchdog_detects_after_release()
{
    const FaultInjectionConfig injection{FaultPoint::watchdog_delay, 0, 1};
    InjectedSource source{injection};
    InjectedProcessor processor{injection};
    InjectedActuator actuator{injection};
    FaultClock clock;
    clock.delay_waits();
    FaultSafetyController safety;
    auto config = make_fault_config();
    config.source_stall_timeout = RealtimeDuration{10};
    config.watchdog_period = RealtimeDuration{1};
    NativeStreamRunner runtime{
        make_fault_schema(), config, source, processor, actuator, clock, safety};

    CHECK(prepare_arm_start(runtime));
    CHECK(wait_until([&] { return source.blocked() && clock.waiters() != 0; }));
    clock.advance(10);
    CHECK(!runtime.primary_fault().has_value());
    clock.release_waits();
    CHECK(wait_until([&] { return runtime.primary_fault().has_value(); }));
    const auto primary = *runtime.primary_fault();
    CHECK(runtime.join() == StreamStatus::deadline_exceeded);
    CHECK(verify_failed_runtime(runtime, safety, FaultCode::source_stall, primary));
    return 0;
}

int test_shutdown_timeout_and_bounded_secondary_history()
{
    {
        const FaultInjectionConfig injection{FaultPoint::shutdown_timeout, 0, 1};
        InjectedSource source{injection};
        InjectedProcessor processor{injection};
        InjectedActuator actuator{injection};
        FaultClock clock;
        FaultSafetyController safety;
        auto config = make_fault_config();
        config.shutdown_deadline = RealtimeDuration{10};
        NativeStreamRunner runtime{
            make_fault_schema(), config, source, processor, actuator, clock, safety};
        CHECK(prepare_arm_start(runtime));
        CHECK(wait_until([&] { return processor.blocked(); }));
        std::atomic<StreamStatus> stop_status{StreamStatus::invalid_state};
        std::thread stopper{[&] { stop_status.store(runtime.stop(), std::memory_order_release); }};
        CHECK(wait_until(
            [&]
            {
                return runtime.state() == RuntimeState::stopping && safety.inhibited() &&
                       clock.waiters() != 0;
            }));
        clock.advance(10);
        CHECK(wait_until(
            [&]
            {
                return stop_status.load(std::memory_order_acquire) ==
                       StreamStatus::deadline_exceeded;
            }));
        stopper.join();
        const auto primary = *runtime.primary_fault();
        processor.release();
        CHECK(wait_until(
            [&]
            {
                return runtime.join() == StreamStatus::deadline_exceeded &&
                       verify_failed_runtime(runtime, safety, FaultCode::shutdown_timeout, primary);
            }));
    }
    {
        const FaultInjectionConfig injection{FaultPoint::processor_deadline, 0, 1};
        InjectedSource source{injection};
        InjectedProcessor processor{injection};
        InjectedActuator actuator{injection};
        FaultClock clock;
        FaultSafetyController safety;
        auto config = make_fault_config();
        config.fault_history_capacity = 1;
        config.processor_execution_deadline = RealtimeDuration{10};
        NativeStreamRunner runtime{
            make_fault_schema(), config, source, processor, actuator, clock, safety};
        CHECK(prepare_arm_start(runtime));
        CHECK(wait_until([&] { return processor.blocked() && clock.waiters() != 0; }));
        safety.fail_inhibit();
        clock.advance(10);
        CHECK(wait_until([&] { return runtime.primary_fault().has_value(); }));
        const auto primary = *runtime.primary_fault();
        processor.fail_after_release();
        processor.release();
        CHECK(runtime.join() == StreamStatus::deadline_exceeded);
        std::array<FaultRecord, 4> history{};
        CHECK(runtime.copy_fault_history(history) == 1);
        CHECK(runtime.stats().secondary_faults >= 2);
        CHECK(runtime.stats().fault_history_dropped >= 1);
        CHECK(verify_failed_runtime(runtime, safety, FaultCode::processor_deadline, primary));
    }
    return 0;
}

int test_running_runtime_is_safely_destructible()
{
    const FaultInjectionConfig injection{FaultPoint::source_stall, 0, 1};
    InjectedSource source{injection};
    InjectedProcessor processor{injection};
    InjectedActuator actuator{injection};
    FaultClock clock;
    FaultSafetyController safety;
    {
        NativeStreamRunner runtime{
            make_fault_schema(), make_fault_config(), source, processor, actuator, clock, safety};
        CHECK(prepare_arm_start(runtime));
        CHECK(wait_until([&] { return source.blocked(); }));
    }
    CHECK(safety.inhibited());
    return 0;
}

} // namespace

int main()
{
    using Test = int (*)();
    const std::array<std::pair<std::string_view, Test>, 10> tests{
        std::pair<std::string_view, Test>{"source_validation", &test_source_validation_faults},
        std::pair<std::string_view, Test>{"sample_gap", &test_sample_gap_is_ordered_and_nonfatal},
        std::pair<std::string_view, Test>{"processor_faults",
                                          &test_processor_faults_recover_every_lease},
        std::pair<std::string_view, Test>{"ingress_overrun", &test_ingress_overrun_is_critical},
        std::pair<std::string_view, Test>{"stage_deadlines", &test_source_and_stage_deadlines},
        std::pair<std::string_view, Test>{"actuator_failure",
                                          &test_actuator_failure_stops_new_commands},
        std::pair<std::string_view, Test>{"blocked_observer",
                                          &test_blocked_observer_is_noncritical},
        std::pair<std::string_view, Test>{"watchdog_delay",
                                          &test_delayed_watchdog_detects_after_release},
        std::pair<std::string_view, Test>{"shutdown_history",
                                          &test_shutdown_timeout_and_bounded_secondary_history},
        std::pair<std::string_view, Test>{"safe_destructor",
                                          &test_running_runtime_is_safely_destructible},
    };
    for (const auto& [name, test] : tests)
    {
        if (const auto result = test(); result != 0)
        {
            std::cerr << name << " failed at line " << result << '\n';
            return result;
        }
    }
    return 0;
}
