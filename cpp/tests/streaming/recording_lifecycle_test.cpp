/* SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT */

/* Streaming recording lifecycle acceptance suite.
 * Noncritical observers retain their best-effort abort behavior. Critical
 * observers instead prove readiness, lossless registration, accepted-message
 * drain, ordinal assignment, bounded cancellation, primary-fault delivery,
 * stable terminal state, and cleanup. */

#include <neurale/streaming/runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <system_error>
#include <thread>

#include "check_returns.h"

using namespace neurale::streaming;

namespace
{

constexpr std::size_t kEdgeCapacity = 8;
constexpr std::uint64_t kFrameCount = 8;

StreamSchema make_schema()
{
    const SignalSchema signal{1, SignalDType::float64, 1, 1, 1, {1'000, 1}, 1};
    return StreamSchema{SchemaId{1}, std::span{&signal, 1}};
}

RealtimeConfig make_config()
{
    RealtimeConfig config;
    config.pool_capacity.source_owned = 1;
    config.pool_capacity.ingress_capacity = 16;
    config.pool_capacity.processor_owned = 2;
    config.pool_capacity.critical_edge_capacity = 16;
    config.pool_capacity.actuator_owned = 1;
    config.pool_capacity.observer_edge_capacity = kEdgeCapacity;
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

ObserverEdgeConfig observer_config(ObserverId id, bool critical)
{
    return {.id = id,
            .capacity = kEdgeCapacity,
            .drop_history_capacity = 32,
            .drop_policy = critical ? ObserverDropPolicy::fault : ObserverDropPolicy::drop_newest,
            .critical_recorder = critical};
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

template <typename Predicate> bool wait_until_for(Predicate predicate, std::chrono::seconds limit)
{
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (predicate())
            return true;
        std::this_thread::yield();
    }
    return predicate();
}

/// Produces a fixed number of frames, then stalls until it is cancelled.
/// Stalling rather than ending keeps the session alive so a test can decide how
/// it terminates.
class StallingSource final : public NativeFrameSource
{
  public:
    explicit StallingSource(std::uint64_t count, bool end_after_count, bool gap_after_first = false)
        : count_(count), end_after_count_(end_after_count), gap_after_first_(gap_after_first)
    {
    }

    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (next_ >= count_)
        {
            if (end_after_count_ || cancelled_.load(std::memory_order_acquire))
                return StreamStatus::end_of_stream;
            return StreamStatus::would_block;
        }
        const auto sequence = gap_after_first_ && next_ != 0 ? next_ + 1 : next_;
        frame.header() = FrameHeader{
            .session_id = SessionId{1},
            .sequence = sequence,
            .host_received_ns = next_ + 1,
            .schema_id = SchemaId{1},
            .source_clock_domain = ClockDomainId{1},
        };
        frame.block_storage()[0] = SignalBlockHeader{
            .sample_idx_start = sequence,
            .device_tick_start = sequence,
            .payload_offset = 0,
            .payload_byte_count = sizeof(double),
            .signal_id = SignalId{1},
            .n_samples = 1,
        };
        if (frame.set_used_sizes(1, sizeof(double)) != StreamStatus::ok)
            return StreamStatus::invalid_frame;
        ++next_;
        return StreamStatus::ok;
    }

    void cancel() noexcept override
    {
        cancelled_.store(true, std::memory_order_release);
    }

    StreamStatus reset() noexcept override
    {
        next_ = 0;
        cancelled_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

  private:
    std::uint64_t count_{};
    bool end_after_count_{};
    bool gap_after_first_{};
    std::uint64_t next_{};
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

class NoOutputProcessor final : public NativeFrameProcessor
{
  public:
    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        const SignalSchema feature{2, SignalDType::float64, 1, 1, 1, {1'000, 1}, 1};
        return {
            .accepted_input_schema =
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
            .output_schema = StreamSchema{SchemaId{2}, std::span{&feature, 1}},
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
    std::uint64_t count() const noexcept
    {
        return count_.load(std::memory_order_acquire);
    }

  private:
    StreamStatus write(const ActuatorCommand&) noexcept override
    {
        count_.fetch_add(1, std::memory_order_release);
        return StreamStatus::ok;
    }
    std::atomic<std::uint64_t> count_{};
};

class FailingSource final : public NativeFrameSource
{
  public:
    StreamStatus read(MutableFrame&) noexcept override
    {
        return StreamStatus::source_failure;
    }
    void cancel() noexcept override {}
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }
};

class FailingProcessor final : public NativeFrameProcessor
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
    StreamStatus process(FrameBorrow&, FrameEmitter&) noexcept override
    {
        return StreamStatus::processor_failure;
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

class FailingActuator final : public NativeActuator
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
        return StreamStatus::actuator_failure;
    }
};

class FailingDiscontinuityConsumer final : public NativeFrameConsumer
{
  public:
    StreamStatus consume(FrameView) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        return StreamStatus::consumer_failure;
    }
    StreamStatus flush() noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }
};

/// Counts what it was actually handed, and optionally blocks inside the first
/// observe() so a test can let its edge queue fill.
class GatedObserver final : public NativeObserver
{
  public:
    explicit GatedObserver(bool block_first) : block_first_(block_first) {}

    StreamStatus observe(FrameView) noexcept override
    {
        if (block_first_ && !gate_entered_.exchange(true, std::memory_order_acq_rel))
        {
            blocked_.store(true, std::memory_order_release);
            released_.wait(false, std::memory_order_acquire);
            blocked_.store(false, std::memory_order_release);
        }
        delivered_.fetch_add(1, std::memory_order_release);
        return StreamStatus::ok;
    }
    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus flush() noexcept override
    {
        flushes_.fetch_add(1, std::memory_order_release);
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
    bool blocked() const noexcept
    {
        return blocked_.load(std::memory_order_acquire);
    }
    std::uint64_t delivered() const noexcept
    {
        return delivered_.load(std::memory_order_acquire);
    }
    std::uint64_t flushes() const noexcept
    {
        return flushes_.load(std::memory_order_acquire);
    }

  private:
    bool block_first_{};
    std::atomic<bool> gate_entered_{};
    std::atomic<bool> blocked_{};
    std::atomic<bool> released_{};
    std::atomic<std::uint64_t> delivered_{};
    std::atomic<std::uint64_t> flushes_{};
};

class CriticalObserver final : public NativeCriticalObserver
{
  public:
    explicit CriticalObserver(bool block_first = false) : block_first_(block_first) {}

    StreamStatus ready_for_runtime() noexcept override
    {
        readiness_calls_.fetch_add(1, std::memory_order_relaxed);
        return readiness_status_;
    }
    StreamStatus start_observing() noexcept override
    {
        start_calls_.fetch_add(1, std::memory_order_relaxed);
        return start_status_;
    }
    StreamStatus health() const noexcept override
    {
        return health_.load(std::memory_order_acquire);
    }
    StreamStatus observe_accepted(FrameView frame, RuntimeAcceptance acceptance) noexcept override
    {
        if (block_first_ && !gate_entered_.exchange(true, std::memory_order_acq_rel))
        {
            blocked_.store(true, std::memory_order_release);
            released_.wait(false, std::memory_order_acquire);
            blocked_.store(false, std::memory_order_release);
        }
        const auto idx = acceptance_count_.fetch_add(1, std::memory_order_relaxed);
        if (idx < acceptances_.size())
        {
            acceptances_[idx] = acceptance;
            kinds_[idx] = AcceptedMessageKind::frame;
            sequences_[idx] = frame.header.sequence;
        }
        if (fail_observe_.load(std::memory_order_acquire))
        {
            health_.store(StreamStatus::consumer_failure, std::memory_order_release);
            return StreamStatus::consumer_failure;
        }
        return StreamStatus::ok;
    }
    StreamStatus handle_accepted_discontinuity(const Discontinuity& discontinuity,
                                               RuntimeAcceptance acceptance) noexcept override
    {
        const auto idx = acceptance_count_.fetch_add(1, std::memory_order_relaxed);
        if (idx < acceptances_.size())
        {
            acceptances_[idx] = acceptance;
            kinds_[idx] = AcceptedMessageKind::discontinuity;
            sequences_[idx] = discontinuity.actual_frame_sequence;
        }
        return StreamStatus::ok;
    }
    void note_rejected_before_acceptance(RejectedMessage) noexcept override
    {
        rejections_.fetch_add(1, std::memory_order_relaxed);
        health_.store(StreamStatus::observer_overrun, std::memory_order_release);
    }
    void publish_primary_fault(const FaultRecord& fault) noexcept override
    {
        published_fault_ = fault;
        fault_publications_.fetch_add(1, std::memory_order_release);
    }
    StreamStatus drain(RuntimeTerminalNotice terminal) noexcept override
    {
        terminal_ = terminal;
        drain_calls_.fetch_add(1, std::memory_order_release);
        if (block_drain_.load(std::memory_order_acquire))
        {
            drain_blocked_.store(true, std::memory_order_release);
            drain_released_.wait(false, std::memory_order_acquire);
            drain_blocked_.store(false, std::memory_order_release);
        }
        return drain_status_.load(std::memory_order_acquire);
    }
    void cancel() noexcept override
    {
        cancel_calls_.fetch_add(1, std::memory_order_release);
        release();
        drain_released_.store(true, std::memory_order_release);
        drain_released_.notify_all();
    }
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }

    void release() noexcept
    {
        released_.store(true, std::memory_order_release);
        released_.notify_all();
    }
    void fail_on_observe() noexcept
    {
        fail_observe_.store(true, std::memory_order_release);
    }
    void fail_readiness(StreamStatus status) noexcept
    {
        readiness_status_ = status;
    }
    void block_drain() noexcept
    {
        block_drain_.store(true, std::memory_order_release);
    }
    bool blocked() const noexcept
    {
        return blocked_.load(std::memory_order_acquire);
    }
    bool drain_blocked() const noexcept
    {
        return drain_blocked_.load(std::memory_order_acquire);
    }
    std::uint64_t acceptance_count() const noexcept
    {
        return acceptance_count_.load(std::memory_order_acquire);
    }
    std::uint64_t fault_publications() const noexcept
    {
        return fault_publications_.load(std::memory_order_acquire);
    }
    std::uint64_t start_calls() const noexcept
    {
        return start_calls_.load(std::memory_order_acquire);
    }
    std::uint64_t drain_calls() const noexcept
    {
        return drain_calls_.load(std::memory_order_acquire);
    }
    std::uint64_t cancel_calls() const noexcept
    {
        return cancel_calls_.load(std::memory_order_acquire);
    }
    RuntimeTerminalNotice terminal() const noexcept
    {
        return terminal_;
    }
    FaultRecord published_fault() const noexcept
    {
        return published_fault_;
    }

    std::array<RuntimeAcceptance, 64> acceptances_{};
    std::array<AcceptedMessageKind, 64> kinds_{};
    std::array<std::uint64_t, 64> sequences_{};

  private:
    bool block_first_{};
    StreamStatus readiness_status_{StreamStatus::ok};
    StreamStatus start_status_{StreamStatus::ok};
    std::atomic<StreamStatus> health_{StreamStatus::ok};
    std::atomic<StreamStatus> drain_status_{StreamStatus::ok};
    std::atomic<bool> fail_observe_{};
    std::atomic<bool> gate_entered_{};
    std::atomic<bool> blocked_{};
    std::atomic<bool> released_{};
    std::atomic<bool> block_drain_{};
    std::atomic<bool> drain_blocked_{};
    std::atomic<bool> drain_released_{};
    std::atomic<std::uint64_t> readiness_calls_{};
    std::atomic<std::uint64_t> start_calls_{};
    std::atomic<std::uint64_t> acceptance_count_{};
    std::atomic<std::uint64_t> rejections_{};
    std::atomic<std::uint64_t> fault_publications_{};
    std::atomic<std::uint64_t> drain_calls_{};
    std::atomic<std::uint64_t> cancel_calls_{};
    FaultRecord published_fault_{};
    RuntimeTerminalNotice terminal_{};
};

class CountingSafety final : public SafetyController
{
  public:
    StreamStatus inhibit(SafetyReason) noexcept override
    {
        inhibits.fetch_add(1, std::memory_order_release);
        return StreamStatus::ok;
    }
    StreamStatus release() noexcept override
    {
        return release_status.load(std::memory_order_acquire);
    }
    std::atomic<std::uint64_t> inhibits{};
    std::atomic<StreamStatus> release_status{StreamStatus::ok};
};

/* --- characterized behavior ------------------------------------------------ */

/// Run until every produced frame has been enqueued on the observer edge and
/// has cleared the critical path, so nothing is in flight when the test decides
/// how the session ends. The observer is held inside its first observe()
/// throughout, so what has been enqueued is a genuine undelivered backlog.
int quiesce(NativeStreamRunner& runtime, const CountingActuator& actuator,
            const GatedObserver& observer, ObserverId id)
{
    CHECK(wait_until([&] { return observer.blocked(); }));
    CHECK(wait_until(
        [&]
        {
            const auto stats = runtime.observer_stats(id);
            return stats.has_value() && stats->enqueued == kFrameCount;
        }));
    CHECK(wait_until([&] { return actuator.count() == kFrameCount; }));
    return 0;
}

template <typename Source, typename Processor, typename Terminal>
int check_fault_reaches_critical_observer(Source& source, Processor& processor, Terminal& terminal,
                                          FaultStage expected_stage, ObserverId observer_id)
{
    CriticalObserver recorder;
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, terminal};
    CHECK(runtime.add_critical_observer(recorder, observer_config(observer_id, true)) ==
          StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(runtime.join() != StreamStatus::ok);
    const auto fault = runtime.primary_fault();
    CHECK(fault.has_value());
    CHECK(fault->stage == expected_stage);
    CHECK(recorder.fault_publications() == 1);
    CHECK(recorder.drain_calls() == 1);
    CHECK(recorder.terminal().reason == RuntimeTerminalReason::fault);
    CHECK(runtime.state() == RuntimeState::failed);
    CHECK(runtime.outstanding_frames() == 0);
    CHECK(runtime.outstanding_discontinuities() == 0);
    return 0;
}

/// A session that ends because the source ran out delivers everything the edge
/// accepted -- including a backlog that was still queued when the edge was
/// closed -- and then flushes it exactly once.
///
/// The backlog is the point of the test. The observer is held inside its first
/// observe() until the source has ended, all kFrameCount frames have been
/// enqueued on the edge, the critical path has drained, and the runtime has
/// reached its terminal state; only then is it released. Without the gate this
/// test would pass on a runtime that dropped its queue at end of stream,
/// because nothing would ever be queued.
int test_graceful_end_of_stream_drains_and_flushes()
{
    StallingSource source{kFrameCount, true};
    IdentityProcessor processor;
    CountingActuator actuator;
    GatedObserver observer{true};
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    CHECK(runtime.add_observer(observer, observer_config(1, false)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    if (const auto line = quiesce(runtime, actuator, observer, 1); line != 0)
        return line;
    // Terminal state is published only after every worker has joined. While
    // this observer deliberately holds a backlog, the runtime remains stopping.
    CHECK(wait_until([&] { return runtime.state() == RuntimeState::stopping; }));

    // The backlog the drain has to survive: everything accepted, nothing yet
    // delivered, because the observer is still blocked in its first call.
    const auto before = runtime.observer_stats(1);
    CHECK(before.has_value());
    CHECK(before->enqueued == kFrameCount);
    CHECK(before->delivered == 0);
    CHECK(before->dropped == 0);
    CHECK(observer.flushes() == 0);

    observer.release();
    CHECK(runtime.join() == StreamStatus::ok);

    const auto stats = runtime.observer_stats(1);
    CHECK(stats.has_value());
    CHECK(stats->enqueued == kFrameCount);
    CHECK(stats->delivered == kFrameCount);
    CHECK(stats->dropped == 0);
    CHECK(observer.delivered() == kFrameCount);
    CHECK(observer.flushes() == 1);
    CHECK(runtime.state() == RuntimeState::stopped);
    CHECK(!runtime.primary_fault().has_value());
    return 0;
}

/// CHARACTERIZATION, NOT A REQUIREMENT. abort() abandons everything the edge
/// already accepted: the worker leaves its loop as soon as the edge is marked
/// inactive, and the remainder of the queue is counted as dropped rather than
/// delivered. flush() is not called at all.
int test_abort_drops_accepted_noncritical_edge_items()
{
    StallingSource source{kFrameCount, false};
    IdentityProcessor processor;
    CountingActuator actuator;
    GatedObserver observer{true};
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    CHECK(runtime.add_observer(observer, observer_config(1, false)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    if (const auto line = quiesce(runtime, actuator, observer, 1); line != 0)
        return line;

    CHECK(runtime.abort() == StreamStatus::ok);

    const auto stats = runtime.observer_stats(1);
    CHECK(stats.has_value());
    CHECK(stats->enqueued == kFrameCount);
    // Only the message the observer was holding when the abort arrived.
    CHECK(stats->delivered == 1);
    CHECK(stats->dropped == kFrameCount - 1);
    CHECK(stats->delivered + stats->dropped == stats->enqueued);
    CHECK(observer.flushes() == 0);
    return 0;
}

/// Explicit abort preserves the noncritical behavior above, but drains every
/// item already accepted on a critical edge before delivering the terminal.
int test_abort_drains_accepted_critical_edge_items()
{
    StallingSource source{kFrameCount, false};
    IdentityProcessor processor;
    CountingActuator actuator;
    CriticalObserver recorder{true};
    CountingSafety safety;
    NativeStreamRunner runtime{make_schema(), make_config(),          source, processor,
                               actuator,      default_native_clock(), safety};
    CHECK(runtime.add_critical_observer(recorder, observer_config(7, true)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(wait_until([&] { return recorder.blocked(); }));
    CHECK(wait_until(
        [&]
        {
            const auto stats = runtime.observer_stats(7);
            return stats.has_value() && stats->enqueued == kFrameCount;
        }));
    std::atomic<StreamStatus> abort_status{StreamStatus::would_block};
    std::thread aborter([&] { abort_status.store(runtime.abort(), std::memory_order_release); });
    CHECK(wait_until([&] { return runtime.state() == RuntimeState::stopping; }));
    recorder.release();
    aborter.join();
    CHECK(abort_status.load(std::memory_order_acquire) == StreamStatus::ok);

    const auto stats = runtime.observer_stats(7);
    CHECK(stats.has_value());
    CHECK(stats->enqueued == kFrameCount);
    CHECK(stats->delivered == kFrameCount);
    CHECK(stats->dropped == 0);
    CHECK(recorder.acceptance_count() == kFrameCount);
    CHECK(recorder.drain_calls() == 1);
    CHECK(recorder.terminal().reason == RuntimeTerminalReason::abort);
    CHECK(recorder.terminal().accepted_message_count == kFrameCount);
    CHECK(!runtime.primary_fault().has_value());
    CHECK(runtime.state() == RuntimeState::stopped);
    CHECK(safety.inhibits.load(std::memory_order_acquire) != 0);
    return 0;
}

/// An input critical observer records continuity-accepted acquisition frames
/// even when the processor emits no output and declares a different schema.
int test_input_critical_observer_precedes_processing()
{
    constexpr std::uint64_t count = 3;
    StallingSource source{count, true};
    NoOutputProcessor processor;
    CountingActuator actuator;
    CriticalObserver recorder;
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    CHECK(runtime.add_input_critical_observer(recorder, observer_config(8, true)) ==
          StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(runtime.join() == StreamStatus::ok);

    const auto stats = runtime.observer_stats(8);
    CHECK(stats.has_value());
    CHECK(stats->enqueued == count);
    CHECK(stats->delivered == count);
    CHECK(stats->dropped == 0);
    CHECK(recorder.acceptance_count() == count);
    for (std::uint64_t i = 0; i < count; ++i)
    {
        CHECK(recorder.acceptances_[i].data_message_ordinal == i);
        CHECK(recorder.sequences_[i] == i);
    }
    CHECK(recorder.terminal().accepted_message_count == count);
    CHECK(actuator.count() == 0);
    return 0;
}

/// A failure reported by a critical edge escalates to a runtime fault and a
/// safety inhibit. The *terminal state* is deliberately not asserted: see the
/// comment at the end of this test.
int test_critical_edge_failure_faults_runtime()
{
    StallingSource source{kFrameCount, true};
    IdentityProcessor processor;
    CountingActuator actuator;
    CriticalObserver recorder;
    recorder.fail_on_observe();
    CountingSafety safety;
    NativeStreamRunner runtime{make_schema(), make_config(),          source, processor,
                               actuator,      default_native_clock(), safety};
    CHECK(runtime.add_critical_observer(recorder, observer_config(3, true)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    const auto join_status = runtime.join();

    const auto fault = runtime.primary_fault();
    CHECK(fault.has_value());
    CHECK(fault->stage == FaultStage::observer);
    CHECK(fault->code == FaultCode::critical_observer_failure ||
          fault->code == FaultCode::critical_observer_overrun);
    CHECK(runtime.heartbeat().safety_inhibited);
    // join() is computed after every worker has been joined, so it always
    // reports the fault.
    CHECK(join_status == fault->status);
    CHECK(join_status != StreamStatus::ok);
    CHECK(runtime.state() == RuntimeState::failed);
    CHECK(recorder.fault_publications() == 1);
    CHECK(recorder.drain_calls() == 1);
    CHECK(recorder.terminal().reason == RuntimeTerminalReason::fault);
    return 0;
}

/// The same failure on a noncritical edge deactivates only that edge.
int test_noncritical_edge_failure_does_not_fault_runtime()
{
    class FailingObserver final : public NativeObserver
    {
      public:
        StreamStatus observe(FrameView) noexcept override
        {
            return StreamStatus::consumer_failure;
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
        void cancel() noexcept override {}
    };

    StallingSource source{kFrameCount, true};
    IdentityProcessor processor;
    CountingActuator actuator;
    FailingObserver observer;
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    CHECK(runtime.add_observer(observer, observer_config(4, false)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(runtime.join() == StreamStatus::ok);

    CHECK(!runtime.primary_fault().has_value());
    CHECK(runtime.state() == RuntimeState::stopped);
    const auto stats = runtime.observer_stats(4);
    CHECK(stats.has_value());
    CHECK(stats->failures != 0);
    CHECK(stats->detached);
    // The critical path completed regardless.
    CHECK(actuator.count() == kFrameCount);
    return 0;
}

/// arm() is a safety transition, not a lifecycle state. There is no armed state
/// for a recorder readiness gate to hang off yet; this suite adds the gate here.
int test_arm_introduces_no_lifecycle_state()
{
    StallingSource source{kFrameCount, true};
    IdentityProcessor processor;
    CountingActuator actuator;
    GatedObserver observer{false};
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    CHECK(runtime.add_observer(observer, observer_config(1, false)) == StreamStatus::ok);
    CHECK(runtime.state() == RuntimeState::created);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.state() == RuntimeState::prepared);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.state() == RuntimeState::prepared);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(runtime.join() == StreamStatus::ok);
    CHECK(runtime.state() == RuntimeState::stopped);
    return 0;
}

int test_critical_edge_rejects_lossy_policy()
{
    StallingSource source{kFrameCount, true};
    IdentityProcessor processor;
    CountingActuator actuator;
    CriticalObserver recorder;
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    auto config = observer_config(9, true);
    config.drop_policy = ObserverDropPolicy::drop_oldest;
    bool rejected = false;
    try
    {
        static_cast<void>(runtime.add_critical_observer(recorder, config));
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    CHECK(rejected);
    return 0;
}

int test_arm_rejects_unready_critical_observer()
{
    StallingSource source{kFrameCount, true};
    IdentityProcessor processor;
    CountingActuator actuator;
    CriticalObserver recorder;
    recorder.fail_readiness(StreamStatus::invalid_state);
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    CHECK(runtime.add_critical_observer(recorder, observer_config(13, true)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::invalid_state);
    CHECK(runtime.state() == RuntimeState::failed);
    CHECK(runtime.heartbeat().safety_inhibited);
    return 0;
}

int test_arm_cancels_observer_on_safety_release_failure()
{
    StallingSource source{kFrameCount, true};
    IdentityProcessor processor;
    CountingActuator actuator;
    CriticalObserver recorder;
    CountingSafety safety;
    safety.release_status.store(StreamStatus::consumer_failure, std::memory_order_release);
    NativeStreamRunner runtime{make_schema(), make_config(),          source, processor,
                               actuator,      default_native_clock(), safety};
    CHECK(runtime.add_critical_observer(recorder, observer_config(14, true)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::consumer_failure);
    CHECK(runtime.state() == RuntimeState::failed);
    CHECK(recorder.cancel_calls() == 1);
    CHECK(recorder.fault_publications() == 1);
    return 0;
}

int test_stop_before_start_cancels_critical_observer()
{
    StallingSource source{kFrameCount, true};
    IdentityProcessor processor;
    CountingActuator actuator;
    CriticalObserver recorder;
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    CHECK(runtime.add_critical_observer(recorder, observer_config(16, true)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.stop() == StreamStatus::ok);
    CHECK(runtime.state() == RuntimeState::stopped);
    CHECK(recorder.cancel_calls() == 1);
    CHECK(recorder.acceptance_count() == 0);
    return 0;
}

int test_runtime_assigns_one_ordinal_sequence()
{
    StallingSource source{2, true, true};
    IdentityProcessor processor;
    CountingActuator actuator;
    CriticalObserver recorder;
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    CHECK(runtime.add_critical_observer(recorder, observer_config(15, true)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(runtime.join() == StreamStatus::ok);
    CHECK(recorder.acceptance_count() == 3);
    CHECK(recorder.kinds_[0] == AcceptedMessageKind::frame);
    CHECK(recorder.kinds_[1] == AcceptedMessageKind::discontinuity);
    CHECK(recorder.kinds_[2] == AcceptedMessageKind::frame);
    CHECK(recorder.sequences_[0] == 0);
    CHECK(recorder.sequences_[1] == 2);
    CHECK(recorder.sequences_[2] == 2);
    for (std::size_t i = 0; i < 3; ++i)
    {
        CHECK(recorder.acceptances_[i].data_message_ordinal == i);
        CHECK(recorder.acceptances_[i].accepted_at_ns != 0);
    }
    return 0;
}

int test_stalled_critical_drain_is_cancelled_at_bound()
{
    StallingSource source{1, true};
    IdentityProcessor processor;
    CountingActuator actuator;
    CriticalObserver recorder;
    recorder.block_drain();
    auto config = make_config();
    config.shutdown_deadline = RealtimeDuration{50'000'000};
    config.watchdog_period = RealtimeDuration{1'000'000};
    NativeStreamRunner runtime{make_schema(), config, source, processor, actuator};
    CHECK(runtime.add_critical_observer(recorder, observer_config(17, true)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    std::atomic<StreamStatus> first_join{StreamStatus::would_block};
    std::thread joiner([&] { first_join.store(runtime.join(), std::memory_order_release); });
    CHECK(wait_until([&] { return recorder.drain_blocked(); }));
    CHECK(wait_until([&] { return recorder.cancel_calls() != 0; }));
    joiner.join();
    CHECK(first_join.load(std::memory_order_acquire) == StreamStatus::deadline_exceeded);
    const auto first_detach = runtime.detach_observer(17);
    if (first_detach == StreamStatus::invalid_state)
    {
        CHECK(wait_until(
            [&]
            {
                return runtime.join() == StreamStatus::deadline_exceeded &&
                       runtime.state() == RuntimeState::failed;
            }));
        CHECK(runtime.detach_observer(17) == StreamStatus::ok);
    }
    else
    {
        // The first join may finish joining the workers and return the primary
        // shutdown-timeout fault, whose status is also deadline_exceeded.
        CHECK(first_detach == StreamStatus::ok);
        CHECK(runtime.state() == RuntimeState::failed);
    }
    CHECK(recorder.drain_calls() == 1);
    CHECK(recorder.fault_publications() == 1);
    CHECK(runtime.outstanding_frames() == 0);
    CHECK(runtime.outstanding_discontinuities() == 0);
    return 0;
}

int test_runtime_fault_stages_reach_observer_once()
{
    {
        FailingSource source;
        IdentityProcessor processor;
        CountingActuator actuator;
        if (const auto line = check_fault_reaches_critical_observer(source, processor, actuator,
                                                                    FaultStage::source, 21);
            line != 0)
            return line;
    }
    {
        StallingSource source{1, true};
        FailingProcessor processor;
        CountingActuator actuator;
        if (const auto line = check_fault_reaches_critical_observer(source, processor, actuator,
                                                                    FaultStage::processor, 22);
            line != 0)
            return line;
    }
    {
        StallingSource source{1, true};
        IdentityProcessor processor;
        FailingActuator actuator;
        if (const auto line = check_fault_reaches_critical_observer(source, processor, actuator,
                                                                    FaultStage::actuator, 23);
            line != 0)
            return line;
    }
    {
        StallingSource source{2, true, true};
        IdentityProcessor processor;
        FailingDiscontinuityConsumer consumer;
        if (const auto line = check_fault_reaches_critical_observer(source, processor, consumer,
                                                                    FaultStage::consumer, 24);
            line != 0)
            return line;
    }
    return 0;
}

int test_terminal_critical_recorder_can_be_replaced()
{
    StallingSource source{kFrameCount, true};
    IdentityProcessor processor;
    CountingActuator actuator;
    CriticalObserver first;
    CriticalObserver replacement;
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};

    CHECK(runtime.add_critical_observer(first, observer_config(31, true)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(runtime.join() == StreamStatus::ok);
    CHECK(first.drain_calls() == 1);

    CHECK(runtime.detach_observer(31) == StreamStatus::ok);
    CHECK(!runtime.observer_stats(31).has_value());
    CHECK(runtime.reset() == StreamStatus::ok);
    CHECK(runtime.state() == RuntimeState::created);

    CHECK(runtime.add_critical_observer(replacement, observer_config(31, true)) ==
          StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(runtime.join() == StreamStatus::ok);
    CHECK(replacement.drain_calls() == 1);
    CHECK(replacement.acceptance_count() == kFrameCount);
    return 0;
}

int test_failed_readiness_recorder_can_be_replaced()
{
    StallingSource source{kFrameCount, true};
    IdentityProcessor processor;
    CountingActuator actuator;
    CriticalObserver rejected;
    rejected.fail_readiness(StreamStatus::invalid_state);
    CriticalObserver replacement;
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};

    CHECK(runtime.add_critical_observer(rejected, observer_config(32, true)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::invalid_state);
    CHECK(runtime.detach_observer(32) == StreamStatus::ok);
    CHECK(runtime.reset() == StreamStatus::ok);

    CHECK(runtime.add_critical_observer(replacement, observer_config(32, true)) ==
          StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(runtime.join() == StreamStatus::ok);
    CHECK(replacement.drain_calls() == 1);
    return 0;
}

#if defined(NEURALE_STREAMING_TEST_HOOKS)
int test_thread_start_failure_joins_created_workers()
{
    // With one critical observer and no noncritical observers the disabled
    // profile starts watchdog, observer-dispatch, actuator, processing, then
    // acquisition workers in that order.
    for (std::size_t successful_starts = 0; successful_starts < 5; ++successful_starts)
    {
        StallingSource source{kFrameCount, true};
        IdentityProcessor processor;
        CountingActuator actuator;
        CriticalObserver recorder;
        std::atomic<std::size_t> starts_before_failure{successful_starts};
        RuntimeTestHooks hooks{.thread_starts_before_failure = &starts_before_failure};
        NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
        runtime.set_test_hooks(&hooks);

        CHECK(runtime.add_critical_observer(recorder, observer_config(33, true)) ==
              StreamStatus::ok);
        CHECK(runtime.prepare() == StreamStatus::ok);
        CHECK(runtime.arm() == StreamStatus::ok);
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
        CHECK(runtime.state() == RuntimeState::failed);
        CHECK(runtime.primary_fault().has_value());
        CHECK(recorder.fault_publications() == 1);
        CHECK(recorder.start_calls() == 0);
        CHECK(recorder.drain_calls() == 0);
        CHECK(recorder.cancel_calls() == 1);
        CHECK(runtime.outstanding_frames() == 0);
        CHECK(runtime.outstanding_discontinuities() == 0);
        CHECK(runtime.detach_observer(33) == StreamStatus::ok);
        CHECK(runtime.reset() == StreamStatus::ok);
    }
    return 0;
}

int test_terminal_delivery_waits_for_primary_fault()
{
    FailingSource source;
    IdentityProcessor processor;
    CountingActuator actuator;
    CriticalObserver recorder;
    std::atomic<bool> fault_claimed{false};
    std::atomic<bool> reader_waiting{false};
    std::atomic<bool> release_fault{false};
    RuntimeTestHooks hooks{.primary_fault_claimed = &fault_claimed,
                           .primary_fault_reader_waiting = &reader_waiting,
                           .release_primary_fault = &release_fault};
    NativeStreamRunner runtime{make_schema(), make_config(), source, processor, actuator};
    runtime.set_test_hooks(&hooks);

    CHECK(runtime.add_critical_observer(recorder, observer_config(34, true)) == StreamStatus::ok);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.start() == StreamStatus::ok);
    if (!wait_until_for([&] { return fault_claimed.load(std::memory_order_acquire); },
                        std::chrono::seconds{5}))
    {
        release_fault.store(true, std::memory_order_release);
        release_fault.notify_all();
        CHECK(false);
    }

    std::optional<FaultRecord> fault_seen_while_publishing;
    std::atomic<bool> reader_returned{false};
    std::thread reader(
        [&]
        {
            fault_seen_while_publishing = runtime.primary_fault();
            reader_returned.store(true, std::memory_order_release);
        });
    if (!wait_until_for([&] { return reader_waiting.load(std::memory_order_acquire); },
                        std::chrono::seconds{5}))
    {
        release_fault.store(true, std::memory_order_release);
        release_fault.notify_all();
        reader.join();
        CHECK(false);
    }
    const auto returned_before_primary_was_ready = reader_returned.load(std::memory_order_acquire);
    release_fault.store(true, std::memory_order_release);
    release_fault.notify_all();
    reader.join();

    CHECK(!returned_before_primary_was_ready);
    CHECK(fault_seen_while_publishing.has_value());
    CHECK(fault_seen_while_publishing->stage == FaultStage::source);
    CHECK(runtime.join() != StreamStatus::ok);
    const auto primary = runtime.primary_fault();
    CHECK(primary.has_value());
    CHECK(primary->stage == FaultStage::source);
    CHECK(recorder.fault_publications() == 1);
    CHECK(recorder.published_fault().stage == FaultStage::source);
    CHECK(recorder.drain_calls() == 1);
    return 0;
}
#endif

} // namespace

int main()
{
    using Test = int (*)();
    const Test tests[]{
        test_graceful_end_of_stream_drains_and_flushes,
        test_abort_drops_accepted_noncritical_edge_items,
        test_abort_drains_accepted_critical_edge_items,
        test_input_critical_observer_precedes_processing,
        test_critical_edge_failure_faults_runtime,
        test_noncritical_edge_failure_does_not_fault_runtime,
        test_arm_introduces_no_lifecycle_state,
        test_critical_edge_rejects_lossy_policy,
        test_arm_rejects_unready_critical_observer,
        test_arm_cancels_observer_on_safety_release_failure,
        test_stop_before_start_cancels_critical_observer,
        test_runtime_assigns_one_ordinal_sequence,
        test_stalled_critical_drain_is_cancelled_at_bound,
        test_runtime_fault_stages_reach_observer_once,
        test_terminal_critical_recorder_can_be_replaced,
        test_failed_readiness_recorder_can_be_replaced,
#if defined(NEURALE_STREAMING_TEST_HOOKS)
        test_thread_start_failure_joins_created_workers,
        test_terminal_delivery_waits_for_primary_fault,
#endif
    };
    for (const auto test : tests)
    {
        if (const auto line = test(); line != 0)
            return line;
    }
    return 0;
}
