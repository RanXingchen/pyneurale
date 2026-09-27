/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "check_returns.h"
#include "recorder_test_support.h"

#include <neurale/streaming/runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <system_error>
#include <thread>

using namespace neurale::recording;
using namespace neurale::recording::test;
using namespace neurale::streaming;

namespace
{

StreamSchema make_schema()
{
    const SignalSchema signal{kSignalA, SignalDType::float64, 1, 2, 2, {1'000, 1}, 3};
    return StreamSchema{SchemaId{kTestSchemaId}, std::span{&signal, 1}};
}

RealtimeConfig make_config()
{
    RealtimeConfig config;
    config.pool_capacity.source_owned = 1;
    config.pool_capacity.ingress_capacity = 8;
    config.pool_capacity.processor_owned = 2;
    config.pool_capacity.critical_edge_capacity = 8;
    config.pool_capacity.actuator_owned = 1;
    config.pool_capacity.observer_edge_capacity = 8;
    config.buffer_size = 64;
    config.max_signal_blocks = 1;
    config.discontinuity_capacity = 4;
    config.gaps_per_discontinuity = 1;
    config.max_process_outputs = 1;
    config.source_stall_timeout = RealtimeDuration{10'000'000'000};
    config.max_ingress_dwell = RealtimeDuration{10'000'000'000};
    config.processor_execution_deadline = RealtimeDuration{10'000'000'000};
    config.max_source_to_actuator_age = RealtimeDuration{10'000'000'000};
    config.max_output_age = RealtimeDuration{10'000'000'000};
    config.actuator_deadline = RealtimeDuration{10'000'000'000};
    config.shutdown_deadline = RealtimeDuration{5'000'000'000};
    return config;
}

class Source final : public NativeFrameSource
{
  public:
    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (next_ == 4)
            return StreamStatus::end_of_stream;
        frame.header() = FrameHeader{
            .session_id = SessionId{kTestNativeSessionId},
            .sequence = next_,
            .host_received_ns = 100 + next_,
            .schema_id = SchemaId{kTestSchemaId},
            .source_clock_domain = ClockDomainId{3},
        };
        frame.block_storage()[0] = SignalBlockHeader{
            .sample_idx_start = next_ * 2,
            .device_tick_start = next_ * 2,
            .payload_offset = 0,
            .payload_byte_count = 16,
            .signal_id = SignalId{kSignalA},
            .n_samples = 2,
            .last_sample_idx = next_ * 2 + 1,
        };
        if (frame.set_used_sizes(1, 16) != StreamStatus::ok)
            return StreamStatus::invalid_frame;
        ++next_;
        return StreamStatus::ok;
    }
    void cancel() noexcept override {}
    StreamStatus reset() noexcept override
    {
        next_ = 0;
        return StreamStatus::ok;
    }

  private:
    std::uint64_t next_{};
};

class GatedSource final : public NativeFrameSource
{
  public:
    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (next_ == 1 && !release_second_.load(std::memory_order_acquire))
            return StreamStatus::would_block;
        if (next_ == 2)
            return StreamStatus::end_of_stream;
        frame.header() = FrameHeader{
            .session_id = SessionId{kTestNativeSessionId},
            .sequence = next_,
            .host_received_ns = 100 + next_,
            .schema_id = SchemaId{kTestSchemaId},
            .source_clock_domain = ClockDomainId{3},
        };
        frame.block_storage()[0] = SignalBlockHeader{
            .sample_idx_start = next_ * 2,
            .device_tick_start = next_ * 2,
            .payload_offset = 0,
            .payload_byte_count = 16,
            .signal_id = SignalId{kSignalA},
            .n_samples = 2,
            .last_sample_idx = next_ * 2 + 1,
        };
        if (frame.set_used_sizes(1, 16) != StreamStatus::ok)
            return StreamStatus::invalid_frame;
        ++next_;
        return StreamStatus::ok;
    }
    void release_second() noexcept
    {
        release_second_.store(true, std::memory_order_release);
    }
    void cancel() noexcept override
    {
        release_second();
    }
    StreamStatus reset() noexcept override
    {
        next_ = 0;
        release_second_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

  private:
    std::uint64_t next_{};
    std::atomic<bool> release_second_{};
};

class Processor final : public NativeFrameProcessor
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

class Actuator final : public NativeActuator
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

  private:
    StreamStatus write(const ActuatorCommand&) noexcept override
    {
        return StreamStatus::ok;
    }
};

int test_runtime_drives_the_native_recorder_lifecycle()
{
    PlanFixture fixture;
    RecorderMemorySpoolFile spool;
    spool.reserve_bytes(2 * 1024 * 1024);
    CountingUnixClock recorder_clock;
    NativeRecorderCore recorder;
    CHECK(recorder.prepare(fixture.plan(), spool, recorder_clock) == RecorderStatusCode::ok);
    RecorderMemorySpoolFile replacement_spool;
    replacement_spool.reserve_bytes(2 * 1024 * 1024);
    CountingUnixClock replacement_clock;
    NativeRecorderCore replacement;
    CHECK(replacement.prepare(fixture.plan(), replacement_spool, replacement_clock) ==
          RecorderStatusCode::ok);

    Source source;
    Processor processor;
    Actuator actuator;
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    const ObserverEdgeConfig edge{
        .id = 1,
        .capacity = 8,
        .drop_history_capacity = 8,
        .drop_policy = ObserverDropPolicy::fault,
        .critical_recorder = true,
    };
    CHECK(runtime.add_critical_observer(recorder, edge) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(recorder.state() == RecorderLifecycleState::ready);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(runtime.join() == StreamStatus::ok);

    const auto status = recorder.status();
    CHECK(recorder.state() == RecorderLifecycleState::stopped);
    CHECK(status.runtime_accepted == 4);
    CHECK(status.recorder_accepted == 4);
    CHECK(status.spool_committed == 4);
    CHECK(status.frames_accepted == 4);
    CHECK(status.spool_ended_cleanly);
    CHECK(status.requested_terminal_intent == RequestedTerminalIntent::normal);
    CHECK(!runtime.primary_fault().has_value());
    CHECK(runtime.state() == RuntimeState::stopped);
    CHECK(runtime.outstanding_frames() == 0);
    CHECK(runtime.outstanding_discontinuities() == 0);
    CHECK(runtime.detach_observer(1) == StreamStatus::ok);
    CHECK(recorder.state() == RecorderLifecycleState::stopped);
    CHECK(!recorder.status().primary_fault.present);
    CHECK(runtime.reset() == StreamStatus::ok);
    CHECK(runtime.state() == RuntimeState::created);

    CHECK(runtime.add_critical_observer(replacement, edge) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(runtime.join() == StreamStatus::ok);
    CHECK(replacement.status().spool_committed == 4);
    CHECK(runtime.detach_observer(1) == StreamStatus::ok);
    return 0;
}

int test_async_recorder_fault_enters_runtime_fault_path()
{
    PlanFixture fixture;
    fixture.frame_queue_capacity = 32;
    RecorderMemorySpoolFile spool;
    spool.reserve_bytes(2 * 1024 * 1024);
    CountingUnixClock recorder_clock;
    NativeRecorderCore recorder;
    CHECK(recorder.prepare(fixture.plan(), spool, recorder_clock) == RecorderStatusCode::ok);

    GatedSource source;
    Processor processor;
    Actuator actuator;
    auto config = make_config();
    config.watchdog_period = RealtimeDuration{1'000'000};
    NativeStreamRunner runtime{make_schema(), config, source, processor, actuator};
    const ObserverEdgeConfig edge{
        .id = 2,
        .capacity = 8,
        .drop_history_capacity = 8,
        .drop_policy = ObserverDropPolicy::fault,
        .critical_recorder = true,
    };
    CHECK(runtime.add_critical_observer(recorder, edge) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    for (std::size_t attempt = 0; attempt < 100'000 && recorder.status().spool_committed == 0;
         ++attempt)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    CHECK(recorder.status().spool_committed == 1);
    spool.fail_append_call.store(spool.append_calls() + 1, std::memory_order_release);
    source.release_second();

    CHECK(runtime.join() != StreamStatus::ok);
    const auto runtime_fault = runtime.primary_fault();
    CHECK(runtime_fault.has_value());
    CHECK(runtime_fault->stage == FaultStage::observer);
    CHECK(runtime_fault->code == FaultCode::critical_observer_failure);
    const auto recorder_status = recorder.status();
    CHECK(recorder_status.primary_fault.present);
    CHECK(recorder_status.primary_fault.origin == FaultOrigin::recorder);
    CHECK(recorder_status.primary_fault.reason == RecorderFaultReason::spool_writer_failed);
    CHECK(recorder_status.effective_session_outcome == EffectiveSessionOutcome::faulted);
    CHECK(runtime.heartbeat().safety_inhibited);
    CHECK(runtime.state() == RuntimeState::failed);
    CHECK(runtime.outstanding_frames() == 0);
    return 0;
}

#if defined(NEURALE_STREAMING_TEST_HOOKS)
int test_thread_creation_failure_closes_unstarted_recorder()
{
    PlanFixture fixture;
    RecorderMemorySpoolFile spool;
    spool.reserve_bytes(2 * 1024 * 1024);
    CountingUnixClock recorder_clock;
    NativeRecorderCore recorder;
    CHECK(recorder.prepare(fixture.plan(), spool, recorder_clock) == RecorderStatusCode::ok);

    Source source;
    Processor processor;
    Actuator actuator;
    std::atomic<std::size_t> starts_before_failure{0};
    RuntimeTestHooks hooks{.thread_starts_before_failure = &starts_before_failure};
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    runtime.set_test_hooks(&hooks);
    const ObserverEdgeConfig edge{
        .id = 3,
        .capacity = 8,
        .drop_history_capacity = 8,
        .drop_policy = ObserverDropPolicy::fault,
        .critical_recorder = true,
    };
    CHECK(runtime.add_critical_observer(recorder, edge) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(recorder.state() == RecorderLifecycleState::ready);

    bool failed_as_injected = false;
    try
    {
        static_cast<void>(runtime.start());
    }
    catch (const std::system_error&)
    {
        failed_as_injected = true;
    }
    CHECK(failed_as_injected);

    const auto status = recorder.status();
    CHECK(runtime.state() == RuntimeState::failed);
    CHECK(runtime.primary_fault().has_value());
    CHECK(recorder.state() == RecorderLifecycleState::closed);
    CHECK(!status.worker_running);
    CHECK(status.primary_fault.present);
    CHECK(status.primary_fault.origin == FaultOrigin::runtime);
    CHECK(status.effective_session_outcome == EffectiveSessionOutcome::faulted);
    CHECK(!status.session_created);
    CHECK(status.spool_committed_transactions == 0);
    CHECK(status.data_queue_pending == 0);
    CHECK(status.control_queue_pending == 0);
    CHECK(status.queue_storage_released);
    CHECK(spool.size() == 0);
    CHECK(runtime.outstanding_frames() == 0);
    CHECK(runtime.outstanding_discontinuities() == 0);
    CHECK(runtime.detach_observer(3) == StreamStatus::ok);
    CHECK(runtime.reset() == StreamStatus::ok);
    return 0;
}
#endif

} // namespace

int main()
{
    for (const auto test : {test_runtime_drives_the_native_recorder_lifecycle,
                            test_async_recorder_fault_enters_runtime_fault_path})
    {
        if (const auto line = test(); line != 0)
            return line;
    }
#if defined(NEURALE_STREAMING_TEST_HOOKS)
    if (const auto line = test_thread_creation_failure_closes_unstarted_recorder(); line != 0)
        return line;
#endif
    return 0;
}
