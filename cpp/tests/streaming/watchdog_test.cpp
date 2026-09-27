/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>

#include "check_returns.h"

namespace
{

using namespace neurale::streaming;

template <typename Predicate> [[nodiscard]] bool wait_until(Predicate predicate) noexcept
{
    for (std::size_t attempt = 0; attempt < 2'000'000; ++attempt)
    {
        if (predicate())
        {
            return true;
        }
        std::this_thread::yield();
    }
    return false;
}

class FakeClock final : public NativeClock
{
  public:
    [[nodiscard]] HostTimeNs now_ns() noexcept override
    {
        return now_.load(std::memory_order_acquire);
    }

    void wait_until(HostTimeNs deadline_ns) noexcept override
    {
        // Fake time advances only when the test requests it. Poll at a short
        // bounded real interval so a wake immediately before this call cannot
        // become an indefinite atomic wait on an unchanged fake deadline.
        if (now_ns() < deadline_ns)
        {
            std::this_thread::sleep_for(std::chrono::microseconds{50});
        }
    }

    void wake() noexcept override
    {
        revision_.fetch_add(1, std::memory_order_release);
        revision_.notify_all();
    }

    void advance(HostTimeNs delta) noexcept
    {
        now_.fetch_add(delta, std::memory_order_release);
        wake();
    }

  private:
    std::atomic<HostTimeNs> now_{1};
    std::atomic<std::uint64_t> revision_{};
};

class RecordingSafetyController final : public SafetyController
{
  public:
    StreamStatus inhibit(SafetyReason reason) noexcept override
    {
        last_reason_.store(reason, std::memory_order_relaxed);
        inhibit_calls_.fetch_add(1, std::memory_order_relaxed);
        return fail_next_inhibit_.exchange(false, std::memory_order_relaxed)
                   ? StreamStatus::safety_failure
                   : StreamStatus::ok;
    }

    StreamStatus release() noexcept override
    {
        release_calls_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    void fail_next_inhibit() noexcept
    {
        fail_next_inhibit_.store(true, std::memory_order_relaxed);
    }

    [[nodiscard]] std::size_t inhibit_calls() const noexcept
    {
        return inhibit_calls_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] SafetyReason last_reason() const noexcept
    {
        return last_reason_.load(std::memory_order_relaxed);
    }

  private:
    std::atomic<std::size_t> inhibit_calls_{};
    std::atomic<std::size_t> release_calls_{};
    std::atomic<SafetyReason> last_reason_{SafetyReason::startup};
    std::atomic<bool> fail_next_inhibit_{};
};

[[nodiscard]] StreamSchema make_schema()
{
    const std::array signals{
        SignalSchema{1, SignalDType::float32, 2, 4, 4, {1'000, 1}, 2},
    };
    return StreamSchema{7, signals};
}

[[nodiscard]] RealtimeConfig make_config()
{
    RealtimeConfig config;
    config.pool_capacity.source_owned = 1;
    config.pool_capacity.ingress_capacity = 4;
    config.pool_capacity.processor_owned = 2;
    config.pool_capacity.critical_edge_capacity = 2;
    config.pool_capacity.actuator_owned = 1;
    config.buffer_size = 64;
    config.max_signal_blocks = 1;
    config.discontinuity_capacity = 2;
    config.gaps_per_discontinuity = 1;
    config.max_process_outputs = 1;
    config.max_flush_outputs = 1;
    config.fault_history_capacity = 8;
    config.source_stall_timeout = RealtimeDuration{1'000};
    config.max_ingress_dwell = RealtimeDuration{1'000};
    config.processor_execution_deadline = RealtimeDuration{1'000};
    config.max_source_to_actuator_age = RealtimeDuration{1'000};
    config.max_output_age = RealtimeDuration{1'000};
    config.actuator_deadline = RealtimeDuration{1'000};
    config.shutdown_deadline = RealtimeDuration{1'000};
    config.watchdog_period = RealtimeDuration{1};
    return config;
}

[[nodiscard]] StreamStatus fill_frame(MutableFrame& frame, std::uint64_t sequence) noexcept
{
    frame.header() = FrameHeader{
        .session_id = 1,
        .sequence = sequence,
        .host_received_ns = sequence + 1,
        .schema_id = 7,
    };
    frame.block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = sequence * 4,
        .device_tick_start = sequence * 4,
        .payload_offset = 0,
        .payload_byte_count = 32,
        .signal_id = 1,
        .n_samples = 4,
    };
    return frame.set_used_sizes(1, 32);
}

class BlockingSource final : public NativeFrameSource
{
  public:
    StreamStatus read(MutableFrame&) noexcept override
    {
        entered_.store(true, std::memory_order_release);
        entered_.notify_all();
        cancelled_.wait(false, std::memory_order_acquire);
        return StreamStatus::stopped;
    }

    void cancel() noexcept override
    {
        cancelled_.store(true, std::memory_order_release);
        cancelled_.notify_all();
    }

    StreamStatus reset() noexcept override
    {
        entered_.store(false, std::memory_order_relaxed);
        cancelled_.store(false, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    [[nodiscard]] bool entered() const noexcept
    {
        return entered_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<bool> entered_{};
    std::atomic<bool> cancelled_{};
};

class ResponsiveEmptySource final : public NativeFrameSource
{
  public:
    StreamStatus read(MutableFrame&) noexcept override
    {
        return cancelled_.load(std::memory_order_acquire) ? StreamStatus::stopped
                                                          : StreamStatus::would_block;
    }
    void cancel() noexcept override
    {
        cancelled_.store(true, std::memory_order_release);
    }
    StreamStatus reset() noexcept override
    {
        cancelled_.store(false, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

  private:
    std::atomic<bool> cancelled_{};
};

class FiniteSource final : public NativeFrameSource
{
  public:
    explicit FiniteSource(std::size_t count) noexcept : count_(count) {}

    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (cancelled_.load(std::memory_order_acquire))
        {
            return StreamStatus::stopped;
        }
        if (idx_ == count_)
        {
            return StreamStatus::end_of_stream;
        }
        const auto status = fill_frame(frame, idx_);
        if (status == StreamStatus::ok)
        {
            ++idx_;
        }
        return status;
    }
    void cancel() noexcept override
    {
        cancelled_.store(true, std::memory_order_release);
    }
    StreamStatus reset() noexcept override
    {
        idx_ = 0;
        cancelled_.store(false, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

  private:
    std::size_t count_{};
    std::size_t idx_{};
    std::atomic<bool> cancelled_{};
};

class IdentityProcessor final : public NativeFrameProcessor
{
  public:
    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return {
            .accepted_input_schema =
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
            .output_schema =
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
            .max_process_outputs_per_input = 1,
            .max_flush_outputs = 0,
            .can_forward_input = true,
            .required_resources = ProcessorResourceBounds{.frame_pool_leases = 1},
        };
    }
    StreamStatus process(FrameBorrow&, FrameEmitter& output) noexcept override
    {
        return output.publish_input();
    }
    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus flush(FrameEmitter&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }
};

class AdvancingProcessor final : public NativeFrameProcessor
{
  public:
    AdvancingProcessor(FakeClock& clock, HostTimeNs advance_ns) noexcept
        : clock_(clock), advance_ns_(advance_ns)
    {
    }

    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return {
            .accepted_input_schema =
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
            .output_schema =
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
            .max_process_outputs_per_input = 1,
            .max_flush_outputs = 0,
            .can_forward_input = true,
            .required_resources = ProcessorResourceBounds{.frame_pool_leases = 1},
        };
    }

    StreamStatus process(FrameBorrow&, FrameEmitter& output) noexcept override
    {
        clock_.advance(advance_ns_);
        return output.publish_input();
    }
    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus flush(FrameEmitter&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }

  private:
    FakeClock& clock_;
    HostTimeNs advance_ns_{};
};

class ZeroOutputProcessor final : public NativeFrameProcessor
{
  public:
    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return {
            .accepted_input_schema =
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
            .output_schema =
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
            .max_process_outputs_per_input = 0,
            .max_flush_outputs = 0,
            .can_forward_input = false,
            .required_resources = ProcessorResourceBounds{.frame_pool_leases = 1},
        };
    }
    StreamStatus process(FrameBorrow&, FrameEmitter&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus flush(FrameEmitter&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }
};

class BlockingProcessor final : public NativeFrameProcessor
{
  public:
    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return {
            .accepted_input_schema =
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
            .output_schema =
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
            .max_process_outputs_per_input = 1,
            .max_flush_outputs = 0,
            .can_forward_input = true,
            .required_resources = ProcessorResourceBounds{.frame_pool_leases = 1},
        };
    }
    StreamStatus process(FrameBorrow&, FrameEmitter& output) noexcept override
    {
        entered_.store(true, std::memory_order_release);
        entered_.notify_all();
        released_.wait(false, std::memory_order_acquire);
        if (fail_after_release_)
        {
            return StreamStatus::processor_failure;
        }
        return output.publish_input();
    }
    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus flush(FrameEmitter&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        entered_.store(false, std::memory_order_relaxed);
        released_.store(false, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    void release() noexcept
    {
        released_.store(true, std::memory_order_release);
        released_.notify_all();
    }
    void fail_after_release() noexcept
    {
        fail_after_release_ = true;
    }
    [[nodiscard]] bool entered() const noexcept
    {
        return entered_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<bool> entered_{};
    std::atomic<bool> released_{};
    bool fail_after_release_{};
};

class CountingConsumer final : public NativeFrameConsumer
{
  public:
    StreamStatus consume(FrameView) noexcept override
    {
        count_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }
    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus flush() noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }

  private:
    std::atomic<std::size_t> count_{};
};

class BlockingActuator final : public NativeActuator
{
  public:
    StreamStatus flush() noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }
    void cancel() noexcept override {}
    bool entered() const noexcept
    {
        return entered_.load(std::memory_order_acquire);
    }
    void release() noexcept
    {
        released_.store(true, std::memory_order_release);
        released_.notify_all();
    }

  private:
    StreamStatus write(const ActuatorCommand&) noexcept override
    {
        entered_.store(true, std::memory_order_release);
        while (!released_.load(std::memory_order_acquire))
        {
            released_.wait(false, std::memory_order_acquire);
        }
        return StreamStatus::ok;
    }
    std::atomic<bool> entered_{};
    std::atomic<bool> released_{};
};

class CountingActuator final : public NativeActuator
{
  public:
    StreamStatus flush() noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }
    void cancel() noexcept override {}
    [[nodiscard]] std::size_t count() const noexcept
    {
        return count_.load(std::memory_order_relaxed);
    }

  private:
    StreamStatus write(const ActuatorCommand&) noexcept override
    {
        count_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    std::atomic<std::size_t> count_{};
};

[[nodiscard]] bool prepare_arm_start(NativeStreamRunner& runtime)
{
    return runtime.prepare() == StreamStatus::ok && runtime.heartbeat().safety_inhibited &&
           runtime.arm() == StreamStatus::ok && !runtime.heartbeat().safety_inhibited &&
           runtime.start() == StreamStatus::ok;
}

int test_source_stall_and_fault_to_inhibit_latency()
{
    FakeClock clock;
    RecordingSafetyController safety;
    BlockingSource source;
    IdentityProcessor processor;
    CountingConsumer consumer;
    auto config = make_config();
    config.source_stall_timeout = RealtimeDuration{10};
    NativeStreamRunner runtime{make_schema(), config, source, processor, consumer, clock, safety};

    CHECK(prepare_arm_start(runtime));
    CHECK(wait_until([&] { return source.entered(); }));
    clock.advance(10);
    CHECK(wait_until(
        [&]
        {
            const auto fault = runtime.primary_fault();
            return fault.has_value() && fault->code == FaultCode::source_stall;
        }));
    CHECK(runtime.join() == StreamStatus::deadline_exceeded);
    const auto fault = runtime.primary_fault();
    const auto heartbeat = runtime.heartbeat();
    CHECK(fault.has_value());
    CHECK(heartbeat.safety_inhibited);
    CHECK(heartbeat.last_inhibit >= fault->detected_at_ns);
    CHECK(heartbeat.last_inhibit - fault->detected_at_ns <= 1);
    CHECK(safety.inhibit_calls() == 2);
    CHECK(safety.last_reason() == SafetyReason::source_stall);
    CHECK(runtime.arm() == StreamStatus::invalid_state);
    return 0;
}

int test_armed_unstarted_runtime_reinhibits_on_stop()
{
    FakeClock clock;
    RecordingSafetyController safety;
    ResponsiveEmptySource source;
    IdentityProcessor processor;
    CountingConsumer consumer;
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor,
                               consumer,      clock,         safety};

    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(!runtime.heartbeat().safety_inhibited);
    CHECK(runtime.stop() == StreamStatus::ok);
    CHECK(runtime.heartbeat().safety_inhibited);
    CHECK(safety.inhibit_calls() == 2);
    return 0;
}

int test_processor_deadline_is_detected_while_blocked()
{
    FakeClock clock;
    RecordingSafetyController safety;
    FiniteSource source{1};
    BlockingProcessor processor;
    CountingConsumer consumer;
    auto config = make_config();
    config.processor_execution_deadline = RealtimeDuration{10};
    NativeStreamRunner runtime{make_schema(), config, source, processor, consumer, clock, safety};

    CHECK(prepare_arm_start(runtime));
    CHECK(wait_until([&] { return processor.entered(); }));
    clock.advance(10);
    const auto detected = wait_until(
        [&]
        {
            const auto fault = runtime.primary_fault();
            return fault.has_value() && fault->code == FaultCode::processor_deadline &&
                   runtime.heartbeat().safety_inhibited;
        });
    const auto heartbeat = runtime.heartbeat();
    processor.release();
    const auto status = runtime.join();
    CHECK(detected);
    CHECK(heartbeat.processor_active);
    CHECK(heartbeat.safety_inhibited);
    CHECK(status == StreamStatus::deadline_exceeded);
    return 0;
}

int test_queue_dwell_is_detected_independently()
{
    FakeClock clock;
    RecordingSafetyController safety;
    FiniteSource source{2};
    BlockingProcessor processor;
    CountingConsumer consumer;
    auto config = make_config();
    config.max_ingress_dwell = RealtimeDuration{10};
    NativeStreamRunner runtime{make_schema(), config, source, processor, consumer, clock, safety};

    CHECK(prepare_arm_start(runtime));
    CHECK(wait_until([&] { return processor.entered(); }));
    CHECK(wait_until([&] { return runtime.stats().frames_acquired == 2; }));
    clock.advance(10);
    const auto detected = wait_until(
        [&]
        {
            const auto fault = runtime.primary_fault();
            return fault.has_value() && fault->code == FaultCode::ingress_dwell_timeout;
        });
    processor.release();
    const auto status = runtime.join();
    CHECK(detected);
    CHECK(status == StreamStatus::deadline_exceeded);
    CHECK(runtime.stats().aborted_frames >= 1);
    return 0;
}

int test_actuator_deadline_is_detected_while_write_is_blocked()
{
    FakeClock clock;
    RecordingSafetyController safety;
    FiniteSource source{1};
    IdentityProcessor processor;
    BlockingActuator actuator;
    auto config = make_config();
    config.actuator_deadline = RealtimeDuration{10};
    NativeStreamRunner runtime{make_schema(), config, source, processor, actuator, clock, safety};

    CHECK(prepare_arm_start(runtime));
    CHECK(wait_until([&] { return actuator.entered(); }));
    clock.advance(10);
    const auto detected = wait_until(
        [&]
        {
            const auto fault = runtime.primary_fault();
            return fault.has_value() && fault->code == FaultCode::actuator_deadline &&
                   runtime.heartbeat().safety_inhibited;
        });
    const auto heartbeat = runtime.heartbeat();
    const auto stats = runtime.stats();
    actuator.release();
    const auto status = runtime.join();
    CHECK(detected);
    CHECK(heartbeat.safety_inhibited);
    CHECK(stats.actuator_deadline_misses == 1);
    CHECK(status == StreamStatus::deadline_exceeded);
    return 0;
}

int test_source_to_actuator_age_is_bounded()
{
    FakeClock clock;
    RecordingSafetyController safety;
    FiniteSource source{1};
    AdvancingProcessor processor{clock, 11};
    CountingActuator actuator;
    auto config = make_config();
    config.max_source_to_actuator_age = RealtimeDuration{10};
    NativeStreamRunner runtime{make_schema(), config, source, processor, actuator, clock, safety};

    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.run() == StreamStatus::deadline_exceeded);
    const auto fault = runtime.primary_fault();
    CHECK(fault->code == FaultCode::actuator_input_stale);
    CHECK(fault->session_id == 1);
    CHECK(fault->schema_id == 7);
    CHECK(fault->runtime_generation != 0);
    CHECK(runtime.heartbeat().safety_inhibited);
    const auto stats = runtime.stats();
    CHECK(stats.actuator_deadline_misses == 1);
    CHECK(stats.max_processor_execution_ns == 11);
    CHECK(stats.max_source_to_actuator_ns == 11);
    CHECK(actuator.count() == 0);
    return 0;
}

int test_stale_output_without_source_stall()
{
    FakeClock clock;
    RecordingSafetyController safety;
    ResponsiveEmptySource source;
    ZeroOutputProcessor processor;
    CountingConsumer consumer;
    auto config = make_config();
    config.max_output_age = RealtimeDuration{10};
    NativeStreamRunner runtime{make_schema(), config, source, processor, consumer, clock, safety};

    CHECK(prepare_arm_start(runtime));
    clock.advance(10);
    CHECK(wait_until(
        [&]
        {
            const auto fault = runtime.primary_fault();
            return fault.has_value() && fault->code == FaultCode::output_stale;
        }));
    CHECK(runtime.join() == StreamStatus::deadline_exceeded);
    CHECK(!runtime.heartbeat().output_valid);
    return 0;
}

int test_shutdown_timeout_does_not_wait_for_processor()
{
    FakeClock clock;
    RecordingSafetyController safety;
    FiniteSource source{1};
    BlockingProcessor processor;
    CountingConsumer consumer;
    auto config = make_config();
    config.shutdown_deadline = RealtimeDuration{10};
    NativeStreamRunner runtime{make_schema(), config, source, processor, consumer, clock, safety};

    CHECK(prepare_arm_start(runtime));
    CHECK(wait_until([&] { return processor.entered(); }));
    std::atomic<StreamStatus> stop_status{StreamStatus::invalid_state};
    std::thread stopper([&] { stop_status.store(runtime.stop(), std::memory_order_release); });
    CHECK(wait_until([&] { return runtime.heartbeat().safety_inhibited; }));
    clock.advance(10);
    CHECK(wait_until(
        [&]
        {
            return stop_status.load(std::memory_order_acquire) == StreamStatus::deadline_exceeded;
        }));
    stopper.join();
    const auto fault = runtime.primary_fault();
    CHECK(fault.has_value());
    CHECK(fault->code == FaultCode::shutdown_timeout);
    CHECK(runtime.heartbeat().processor_active);
    processor.release();
    CHECK(wait_until(
        [&]
        {
            return runtime.join() == StreamStatus::deadline_exceeded &&
                   runtime.state() == RuntimeState::failed && runtime.outstanding_frames() == 0;
        }));
    return 0;
}

int test_overlapping_faults_inhibit_once()
{
    FakeClock clock;
    RecordingSafetyController safety;
    FiniteSource source{1};
    BlockingProcessor processor;
    CountingConsumer consumer;
    processor.fail_after_release();
    auto config = make_config();
    config.processor_execution_deadline = RealtimeDuration{10};
    NativeStreamRunner runtime{make_schema(), config, source, processor, consumer, clock, safety};

    CHECK(prepare_arm_start(runtime));
    CHECK(wait_until([&] { return processor.entered(); }));
    clock.advance(10);
    CHECK(wait_until([&] { return runtime.primary_fault().has_value(); }));
    processor.release();
    CHECK(runtime.join() == StreamStatus::deadline_exceeded);
    CHECK(runtime.stats().secondary_faults >= 1);
    CHECK(safety.inhibit_calls() == 2);
    return 0;
}

int test_safety_failure_is_secondary()
{
    FakeClock clock;
    RecordingSafetyController safety;
    BlockingSource source;
    IdentityProcessor processor;
    CountingConsumer consumer;
    auto config = make_config();
    config.source_stall_timeout = RealtimeDuration{10};
    NativeStreamRunner runtime{make_schema(), config, source, processor, consumer, clock, safety};

    CHECK(prepare_arm_start(runtime));
    CHECK(wait_until([&] { return source.entered(); }));
    safety.fail_next_inhibit();
    clock.advance(10);
    CHECK(wait_until([&] { return runtime.primary_fault().has_value(); }));
    CHECK(runtime.join() == StreamStatus::deadline_exceeded);
    const auto primary = runtime.primary_fault();
    CHECK(primary.has_value());
    CHECK(primary->code == FaultCode::source_stall);
    std::array<FaultRecord, 8> history{};
    const auto count = runtime.copy_fault_history(history);
    CHECK(count >= 1);
    CHECK(history[0].code == FaultCode::safety_controller_failure);
    CHECK(runtime.stats().secondary_faults >= 1);
    return 0;
}

} // namespace

int main()
{
    const std::array tests{
        test_source_stall_and_fault_to_inhibit_latency,
        test_armed_unstarted_runtime_reinhibits_on_stop,
        test_processor_deadline_is_detected_while_blocked,
        test_actuator_deadline_is_detected_while_write_is_blocked,
        test_source_to_actuator_age_is_bounded,
        test_queue_dwell_is_detected_independently,
        test_stale_output_without_source_stall,
        test_shutdown_timeout_does_not_wait_for_processor,
        test_overlapping_faults_inhibit_once,
        test_safety_failure_is_secondary,
    };
    for (const auto test : tests)
    {
        if (const auto result = test(); result != 0)
        {
            return result;
        }
    }
    return 0;
}
