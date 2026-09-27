/* SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT */

#include <neurale/streaming/observer_queue.h>
#include <neurale/streaming/runtime.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <span>
#include <thread>

#include "check_returns.h"

using namespace neurale::streaming;

namespace
{

StreamSchema make_schema()
{
    const SignalSchema signal{1, SignalDType::float64, 1, 1, 1, {1'000, 1}, 1};
    return StreamSchema{SchemaId{1}, std::span{&signal, 1}};
}

RealtimeConfig make_config(std::size_t count = 16)
{
    RealtimeConfig config;
    config.pool_capacity.source_owned = 1;
    config.pool_capacity.ingress_capacity = count;
    config.pool_capacity.processor_owned = 2;
    config.pool_capacity.critical_edge_capacity = count;
    config.pool_capacity.actuator_owned = 1;
    config.pool_capacity.observer_edge_capacity = count;
    config.buffer_size = 64;
    config.max_signal_blocks = 1;
    config.discontinuity_capacity = 4;
    config.gaps_per_discontinuity = 1;
    config.max_process_outputs = 1;
    config.max_flush_outputs = 1;
    config.source_stall_timeout = RealtimeDuration{10'000'000'000};
    config.max_ingress_dwell = RealtimeDuration{10'000'000'000};
    config.processor_execution_deadline = RealtimeDuration{10'000'000'000};
    config.max_source_to_actuator_age = RealtimeDuration{10'000'000'000};
    config.max_output_age = RealtimeDuration{10'000'000'000};
    config.actuator_deadline = RealtimeDuration{10'000'000'000};
    config.shutdown_deadline = RealtimeDuration{10'000'000'000};
    return config;
}

template <typename Predicate> bool wait_until(Predicate predicate)
{
    for (std::size_t attempt = 0; attempt < 4'000'000; ++attempt)
    {
        if (predicate())
            return true;
        std::this_thread::yield();
    }
    return false;
}

class SyntheticSource final : public NativeFrameSource
{
  public:
    explicit SyntheticSource(std::uint64_t count) : count_(count) {}
    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (next_ >= count_)
            return StreamStatus::end_of_stream;
        if (pause_ && next_ != 0 && !released_.load(std::memory_order_acquire))
        {
            return StreamStatus::would_block;
        }
        const auto sequence = next_ + (gap_ && next_ != 0 ? 1 : 0);
        frame.header() = FrameHeader{
            .session_id = SessionId{1},
            .sequence = sequence,
            .host_received_ns = next_ + 1,
            .schema_id = SchemaId{1},
            .source_clock_domain = ClockDomainId{1},
        };
        if (expired_)
        {
            frame.header().flags = FrameFlags::valid_until;
            frame.header().valid_until_ns = 1;
        }
        frame.block_storage()[0] = SignalBlockHeader{
            .sample_idx_start = sequence,
            .device_tick_start = sequence,
            .payload_offset = 0,
            .payload_byte_count = sizeof(double),
            .signal_id = SignalId{1},
            .n_samples = 1,
        };
        if (frame.set_used_sizes(1, sizeof(double)) != StreamStatus::ok)
        {
            return StreamStatus::invalid_frame;
        }
        ++next_;
        return StreamStatus::ok;
    }
    void cancel() noexcept override
    {
        release();
    }
    StreamStatus reset() noexcept override
    {
        next_ = 0;
        return StreamStatus::ok;
    }
    void pause_after_first() noexcept
    {
        pause_ = true;
    }
    void expire_commands() noexcept
    {
        expired_ = true;
    }
    void gap_after_first() noexcept
    {
        gap_ = true;
    }
    void release() noexcept
    {
        released_.store(true, std::memory_order_release);
    }

  private:
    std::uint64_t count_{};
    std::uint64_t next_{};
    bool pause_{};
    bool expired_{};
    bool gap_{};
    std::atomic<bool> released_{};
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
        discontinuities_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }
    StreamStatus flush(FrameEmitter&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        discontinuities_.store(0, std::memory_order_relaxed);
        return StreamStatus::ok;
    }
    std::uint64_t discontinuities() const noexcept
    {
        return discontinuities_.load(std::memory_order_relaxed);
    }

  private:
    std::atomic<std::uint64_t> discontinuities_{};
};

class RecordingActuator final : public NativeActuator
{
  public:
    explicit RecordingActuator(std::uint64_t fail_at = UINT64_MAX) : fail_at_(fail_at) {}
    StreamStatus flush() noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }
    void cancel() noexcept override {}
    std::uint64_t count() const noexcept
    {
        return count_.load();
    }

  private:
    StreamStatus write(const ActuatorCommand&) noexcept override
    {
        const auto idx = count_.fetch_add(1, std::memory_order_relaxed);
        return idx == fail_at_ ? StreamStatus::actuator_failure : StreamStatus::ok;
    }
    std::uint64_t fail_at_{};
    std::atomic<std::uint64_t> count_{};
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
    void cancel() noexcept override
    {
        released_.store(true, std::memory_order_release);
        released_.notify_all();
    }
    bool entered() const noexcept
    {
        return entered_.load();
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

class BlockingObserver final : public NativeObserver
{
  public:
    StreamStatus observe(FrameView) noexcept override
    {
        entered_.store(true, std::memory_order_release);
        while (!released_.load(std::memory_order_acquire))
        {
            released_.wait(false, std::memory_order_acquire);
        }
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
    void cancel() noexcept override
    {
        release();
    }
    void release() noexcept
    {
        released_.store(true, std::memory_order_release);
        released_.notify_all();
    }
    bool entered() const noexcept
    {
        return entered_.load();
    }

  private:
    std::atomic<bool> entered_{};
    std::atomic<bool> released_{};
};

class CountingObserver final : public NativeObserver
{
  public:
    explicit CountingObserver(bool fail = false) : fail_(fail) {}
    StreamStatus observe(FrameView frame) noexcept override
    {
        if (fail_)
            return StreamStatus::consumer_failure;
        frames_.fetch_add(1, std::memory_order_relaxed);
        events_[event_count_++] = frame.header.sequence * 2;
        return StreamStatus::ok;
    }
    StreamStatus handle_discontinuity(const Discontinuity& discontinuity) noexcept override
    {
        discontinuities_.fetch_add(1, std::memory_order_relaxed);
        events_[event_count_++] = discontinuity.actual_frame_sequence * 2 - 1;
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
    void cancel() noexcept override {}
    std::uint64_t discontinuities() const noexcept
    {
        return discontinuities_.load();
    }
    std::uint64_t event(std::size_t idx) const noexcept
    {
        return events_[idx];
    }

  private:
    bool fail_{};
    std::atomic<std::uint64_t> frames_{};
    std::atomic<std::uint64_t> discontinuities_{};
    std::array<std::uint64_t, 8> events_{};
    std::size_t event_count_{};
};

struct MoveGate
{
    std::atomic<bool> entered{};
    std::atomic<bool> released{};
};

struct BlockingMove
{
    int value{};
    MoveGate* gate{};

    BlockingMove() noexcept = default;
    BlockingMove(int item, MoveGate* move_gate) noexcept : value(item), gate(move_gate) {}
    BlockingMove(const BlockingMove&) = delete;
    BlockingMove& operator=(const BlockingMove&) = delete;
    BlockingMove(BlockingMove&& other) noexcept : value(other.value), gate(other.gate)
    {
        other.gate = nullptr;
    }
    BlockingMove& operator=(BlockingMove&& other) noexcept
    {
        if (other.gate != nullptr)
        {
            other.gate->entered.store(true, std::memory_order_release);
            other.gate->entered.notify_all();
            other.gate->released.wait(false, std::memory_order_acquire);
        }
        value = other.value;
        gate = other.gate;
        other.gate = nullptr;
        return *this;
    }
};

class RecordingSafety final : public SafetyController
{
  public:
    StreamStatus inhibit(SafetyReason) noexcept override
    {
        inhibits.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }
    StreamStatus release() noexcept override
    {
        return StreamStatus::ok;
    }
    std::atomic<std::uint64_t> inhibits{};
};

ObserverEdgeConfig observer_config(ObserverId id,
                                   ObserverDropPolicy policy = ObserverDropPolicy::drop_newest,
                                   bool critical = false)
{
    return {.id = id,
            .capacity = 1,
            .drop_history_capacity = 32,
            .drop_policy = policy,
            .critical_recorder = critical};
}

int test_single_observer_slot_is_not_reused_while_claimed()
{
    ObserverQueue<BlockingMove> queue{1};
    MoveGate gate;
    BlockingMove first{1, &gate};
    CHECK(queue.try_push(std::move(first)) == StreamStatus::ok);

    BlockingMove output;
    StreamStatus pop_status{};
    std::thread consumer([&] { pop_status = queue.try_pop(output); });
    const auto entered = wait_until([&] { return gate.entered.load(std::memory_order_acquire); });
    BlockingMove second{2, nullptr};
    const auto push_status =
        entered ? queue.try_push(std::move(second)) : StreamStatus::invalid_state;
    BlockingMove competing_output;
    const auto competing_pop_status =
        entered ? queue.try_pop(competing_output) : StreamStatus::invalid_state;
    gate.released.store(true, std::memory_order_release);
    gate.released.notify_all();
    consumer.join();
    CHECK(entered);
    CHECK(push_status == StreamStatus::queue_overflow);
    CHECK(competing_pop_status == StreamStatus::would_block);
    CHECK(pop_status == StreamStatus::ok);
    CHECK(output.value == 1);
    CHECK(queue.try_push(std::move(second)) == StreamStatus::ok);
    CHECK(queue.try_pop(output) == StreamStatus::ok);
    CHECK(output.value == 2);
    return 0;
}

int test_blocked_observer_does_not_delay_actuator()
{
    SyntheticSource source{12};
    IdentityProcessor processor;
    RecordingActuator actuator;
    BlockingObserver observer;
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    CHECK(runtime.add_observer(observer, observer_config(1)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(wait_until([&] { return observer.entered(); }));
    CHECK(wait_until([&] { return actuator.count() == 12; }));
    observer.release();
    CHECK(runtime.join() == StreamStatus::ok);
    return 0;
}

int test_critical_recorder_overrun_inhibits_runtime()
{
    class SaturatingCriticalObserver final : public NativeCriticalObserver
    {
      public:
        StreamStatus ready_for_runtime() noexcept override
        {
            return StreamStatus::ok;
        }
        StreamStatus start_observing() noexcept override
        {
            return StreamStatus::ok;
        }
        StreamStatus health() const noexcept override
        {
            return failed_.load(std::memory_order_acquire) ? StreamStatus::observer_overrun
                                                           : StreamStatus::ok;
        }
        StreamStatus observe_accepted(FrameView, RuntimeAcceptance) noexcept override
        {
            if (count_.fetch_add(1, std::memory_order_relaxed) == 0)
                return StreamStatus::ok;
            failed_.store(true, std::memory_order_release);
            return StreamStatus::observer_overrun;
        }
        StreamStatus handle_accepted_discontinuity(const Discontinuity&,
                                                   RuntimeAcceptance) noexcept override
        {
            return StreamStatus::ok;
        }
        void note_rejected_before_acceptance(RejectedMessage) noexcept override
        {
            failed_.store(true, std::memory_order_release);
        }
        void publish_primary_fault(const FaultRecord&) noexcept override {}
        StreamStatus drain(RuntimeTerminalNotice) noexcept override
        {
            return StreamStatus::ok;
        }
        void cancel() noexcept override {}
        StreamStatus reset() noexcept override
        {
            return StreamStatus::ok;
        }

      private:
        std::atomic<std::uint64_t> count_{};
        std::atomic<bool> failed_{};
    };

    SyntheticSource source{12};
    IdentityProcessor processor;
    RecordingActuator actuator;
    SaturatingCriticalObserver recorder;
    RecordingSafety safety;
    NativeStreamRunner runtime{make_schema(), make_config(),          source, processor,
                               actuator,      default_native_clock(), safety};
    CHECK(runtime.add_critical_observer(
              recorder, observer_config(7, ObserverDropPolicy::fault, true)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(wait_until([&] { return runtime.primary_fault().has_value(); }));
    CHECK(runtime.join() == StreamStatus::observer_overrun);
    CHECK(runtime.primary_fault()->code == FaultCode::critical_observer_failure);
    CHECK(runtime.heartbeat().safety_inhibited);
    return 0;
}

int test_actuator_rejects_expired_command()
{
    RecordingActuator actuator;
    const ActuatorCommand command{
        .id = 1, .sequence = 2, .generated_at_ns = 10, .valid_until_ns = 20};
    CHECK(actuator.submit(command, 20) == StreamStatus::deadline_exceeded);
    CHECK(actuator.count() == 0);
    return 0;
}

int test_runtime_rejects_expired_actuator_output()
{
    SyntheticSource source{1};
    source.expire_commands();
    IdentityProcessor processor;
    RecordingActuator actuator;
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.run() == StreamStatus::deadline_exceeded);
    CHECK(runtime.primary_fault()->code == FaultCode::actuator_deadline);
    CHECK(runtime.stats().actuator_deadline_misses == 1);
    CHECK(actuator.count() == 0);
    return 0;
}

int test_actuator_failure_is_critical()
{
    SyntheticSource source{2};
    IdentityProcessor processor;
    RecordingActuator actuator{0};
    RecordingSafety safety;
    NativeStreamRunner runtime{make_schema(), make_config(),          source, processor,
                               actuator,      default_native_clock(), safety};
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.run() == StreamStatus::actuator_failure);
    CHECK(runtime.primary_fault()->code == FaultCode::actuator_write);
    CHECK(runtime.heartbeat().safety_inhibited);
    return 0;
}

int test_actuator_queue_overrun_is_critical()
{
    SyntheticSource source{8};
    IdentityProcessor processor;
    BlockingActuator actuator;
    RecordingSafety safety;
    auto config = make_config();
    config.pool_capacity.critical_edge_capacity = 1;
    NativeStreamRunner runtime{make_schema(),          config, source, processor, actuator,
                               default_native_clock(), safety};
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(wait_until([&] { return runtime.primary_fault().has_value(); }));
    CHECK(runtime.join() == StreamStatus::queue_overflow);
    CHECK(runtime.primary_fault()->code == FaultCode::actuator_queue_overrun);
    CHECK(runtime.heartbeat().safety_inhibited);
    return 0;
}

int test_observer_detach_is_noncritical()
{
    SyntheticSource source{4};
    source.pause_after_first();
    IdentityProcessor processor;
    RecordingActuator actuator;
    BlockingObserver observer;
    observer.release();
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    CHECK(runtime.add_observer(observer, observer_config(3)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(wait_until([&] { return actuator.count() == 1; }));
    CHECK(runtime.detach_observer(3) == StreamStatus::ok);
    source.release();
    CHECK(runtime.join() == StreamStatus::ok);
    CHECK(runtime.observer_stats(3)->detached);
    return 0;
}

int test_noncritical_observer_failure_is_isolated()
{
    SyntheticSource source{3};
    IdentityProcessor processor;
    RecordingActuator actuator;
    CountingObserver observer{true};
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    CHECK(runtime.add_observer(observer, observer_config(4)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.run() == StreamStatus::ok);
    const auto stats = runtime.observer_stats(4);
    CHECK(!runtime.primary_fault().has_value());
    CHECK(stats.has_value());
    CHECK(stats->failures == 1);
    CHECK(stats->detached);
    CHECK(actuator.count() == 3);
    return 0;
}

int test_discontinuity_reaches_observer_edge()
{
    SyntheticSource source{2};
    source.gap_after_first();
    IdentityProcessor processor;
    RecordingActuator actuator;
    CountingObserver observer;
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    auto config = observer_config(5);
    config.capacity = 4;
    CHECK(runtime.add_observer(observer, config) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.run() == StreamStatus::ok);
    CHECK(processor.discontinuities() == 1);
    CHECK(actuator.count() == 2);
    CHECK(runtime.stats().discontinuities == 1);
    CHECK(observer.discontinuities() == 1);
    CHECK(runtime.observer_stats(5)->discontinuities == 1);
    CHECK(observer.event(0) == 0);
    CHECK(observer.event(1) == 3);
    CHECK(observer.event(2) == 4);
    CHECK(runtime.outstanding_frames() == 0);
    CHECK(runtime.outstanding_discontinuities() == 0);
    return 0;
}

int test_per_edge_drop_ranges_are_independent()
{
    SyntheticSource source{10};
    IdentityProcessor processor;
    RecordingActuator actuator;
    BlockingObserver slow;
    BlockingObserver latest;
    BlockingObserver oldest;
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    CHECK(runtime.add_observer(slow, observer_config(10)) == StreamStatus::ok);
    CHECK(runtime.add_observer(latest, observer_config(11, ObserverDropPolicy::latest_value)) ==
          StreamStatus::ok);
    CHECK(runtime.add_observer(oldest, observer_config(12, ObserverDropPolicy::drop_oldest)) ==
          StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(wait_until([&] { return slow.entered() && latest.entered() && oldest.entered(); }));
    CHECK(wait_until([&] { return actuator.count() == 10; }));
    CHECK(wait_until(
        [&]
        {
            return runtime.observer_stats(10)->dropped != 0 &&
                   runtime.observer_stats(11)->dropped != 0;
        }));
    CHECK(wait_until([&] { return runtime.observer_stats(12)->dropped != 0; }));
    slow.release();
    latest.release();
    oldest.release();
    CHECK(runtime.join() == StreamStatus::ok);
    std::array<ObserverDropRange, 32> a{};
    std::array<ObserverDropRange, 32> b{};
    CHECK(runtime.copy_observer_drop_ranges(10, a) != 0);
    CHECK(runtime.copy_observer_drop_ranges(11, b) != 0);
    CHECK(runtime.copy_observer_drop_ranges(12, b) != 0);
    CHECK(runtime.observer_stats(10)->id == 10);
    CHECK(runtime.observer_stats(11)->id == 11);
    return 0;
}

} // namespace

int main()
{
    for (const auto test :
         {test_single_observer_slot_is_not_reused_while_claimed,
          test_blocked_observer_does_not_delay_actuator,
          test_critical_recorder_overrun_inhibits_runtime, test_actuator_rejects_expired_command,
          test_runtime_rejects_expired_actuator_output, test_actuator_failure_is_critical,
          test_actuator_queue_overrun_is_critical, test_observer_detach_is_noncritical,
          test_noncritical_observer_failure_is_isolated, test_discontinuity_reaches_observer_edge,
          test_per_edge_drop_ranges_are_independent})
    {
        if (const auto line = test(); line != 0)
            return line;
    }
    return 0;
}
