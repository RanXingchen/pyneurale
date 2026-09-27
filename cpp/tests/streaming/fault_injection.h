/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <neurale/streaming/actuator.h>
#include <neurale/streaming/clock.h>
#include <neurale/streaming/observer.h>
#include <neurale/streaming/processor.h>
#include <neurale/streaming/realtime_config.h>
#include <neurale/streaming/safety.h>
#include <neurale/streaming/schema.h>
#include <neurale/streaming/source.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <thread>

namespace neurale::streaming::test
{

enum class FaultPoint : std::uint8_t
{
    none,
    source_stall,
    source_malformed_header,
    sample_idx_gap,
    source_restart,
    pool_lease_leak,
    ingress_overrun,
    processor_deadline,
    processor_fatal,
    actuator_timeout,
    actuator_failure,
    observer_blocked,
    watchdog_delay,
    shutdown_timeout,
};

struct FaultInjectionConfig
{
    FaultPoint point{FaultPoint::none};
    std::size_t trigger_after{};
    std::size_t n_frames{16};
};

template <typename Predicate>
[[nodiscard]] bool wait_until(Predicate predicate, std::size_t attempts = 4'000'000) noexcept
{
    for (std::size_t attempt = 0; attempt < attempts; ++attempt)
    {
        if (predicate())
        {
            return true;
        }
        std::this_thread::yield();
    }
    return false;
}

class FaultClock final : public NativeClock
{
  public:
    [[nodiscard]] HostTimeNs now_ns() noexcept override
    {
        return now_.load(std::memory_order_acquire);
    }

    void wait_until(HostTimeNs deadline_ns) noexcept override
    {
        std::unique_lock lock(mutex_);
        waiters_.fetch_add(1, std::memory_order_release);
        // Delay phase: block while delay_waits() is armed. A bounded real-time
        // poll (mirroring the real clock's deadline fallback) guarantees a missed
        // release_waits() notification cannot strand the runtime.
        while (delay_waits_.load(std::memory_order_acquire))
        {
            condition_.wait_for(lock, std::chrono::milliseconds(1),
                                [this] { return !delay_waits_.load(std::memory_order_acquire); });
        }
        // Deadline phase: block until the virtual clock reaches the deadline or a
        // wake arrives. The bounded real-time poll lets a caller re-evaluate its
        // own stop flag even when a wake was consumed before entry (e.g. the
        // shutdown wake issued by join_threads), which a purely virtual wait could
        // otherwise miss forever.
        const auto revision = revision_.load(std::memory_order_acquire);
        condition_.wait_for(lock, std::chrono::milliseconds(1),
                            [this, deadline_ns, revision]
                            {
                                return now_.load(std::memory_order_acquire) >= deadline_ns ||
                                       revision_.load(std::memory_order_acquire) != revision;
                            });
        waiters_.fetch_sub(1, std::memory_order_release);
    }

    void wake() noexcept override
    {
        {
            std::lock_guard lock(mutex_);
            revision_.fetch_add(1, std::memory_order_release);
        }
        condition_.notify_all();
    }

    void advance(HostTimeNs delta) noexcept
    {
        {
            std::lock_guard lock(mutex_);
            now_.fetch_add(delta, std::memory_order_release);
            revision_.fetch_add(1, std::memory_order_release);
        }
        condition_.notify_all();
    }

    void delay_waits() noexcept
    {
        std::lock_guard lock(mutex_);
        delay_waits_.store(true, std::memory_order_release);
    }

    void release_waits() noexcept
    {
        {
            std::lock_guard lock(mutex_);
            delay_waits_.store(false, std::memory_order_release);
            revision_.fetch_add(1, std::memory_order_release);
        }
        condition_.notify_all();
    }

    [[nodiscard]] std::size_t waiters() const noexcept
    {
        return waiters_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<HostTimeNs> now_{1};
    std::atomic<std::uint64_t> revision_{};
    std::atomic<std::size_t> waiters_{};
    std::atomic<bool> delay_waits_{};
    mutable std::mutex mutex_;
    std::condition_variable condition_;
};

class FaultSafetyController final : public SafetyController
{
  public:
    StreamStatus inhibit(SafetyReason reason) noexcept override
    {
        last_reason_.store(reason, std::memory_order_relaxed);
        inhibit_calls_.fetch_add(1, std::memory_order_relaxed);
        inhibited_.store(true, std::memory_order_release);
        return fail_inhibit_.load(std::memory_order_relaxed) ? StreamStatus::safety_failure
                                                             : StreamStatus::ok;
    }

    StreamStatus release() noexcept override
    {
        inhibited_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

    void fail_inhibit() noexcept
    {
        fail_inhibit_.store(true, std::memory_order_relaxed);
    }
    [[nodiscard]] bool inhibited() const noexcept
    {
        return inhibited_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::size_t inhibit_calls() const noexcept
    {
        return inhibit_calls_.load(std::memory_order_relaxed);
    }

  private:
    std::atomic<std::size_t> inhibit_calls_{};
    std::atomic<SafetyReason> last_reason_{SafetyReason::startup};
    std::atomic<bool> inhibited_{true};
    std::atomic<bool> fail_inhibit_{};
};

[[nodiscard]] inline StreamSchema make_fault_schema()
{
    const std::array signals{
        SignalSchema{SignalId{1}, SignalDType::float32, 2, 4, 4, {1'000, 1}, ClockDomainId{1}},
    };
    return StreamSchema{SchemaId{7}, signals};
}

[[nodiscard]] inline RealtimeConfig make_fault_config(std::size_t ingress_capacity = 4)
{
    RealtimeConfig config;
    config.pool_capacity.source_owned = 1;
    config.pool_capacity.ingress_capacity = ingress_capacity;
    config.pool_capacity.processor_owned = 2;
    config.pool_capacity.critical_edge_capacity = ingress_capacity + 2;
    config.pool_capacity.actuator_owned = 1;
    config.pool_capacity.observer_edge_capacity = 8;
    config.pool_capacity.reserve = 2;
    config.buffer_size = 64;
    config.max_signal_blocks = 1;
    config.discontinuity_capacity = 4;
    config.gaps_per_discontinuity = 1;
    config.max_process_outputs = 1;
    config.max_flush_outputs = 1;
    config.fault_history_capacity = 4;
    constexpr auto generous_deadline = RealtimeDuration{10'000'000'000};
    config.source_stall_timeout = generous_deadline;
    config.max_ingress_dwell = generous_deadline;
    config.processor_execution_deadline = generous_deadline;
    config.max_output_age = generous_deadline;
    config.actuator_deadline = generous_deadline;
    config.shutdown_deadline = generous_deadline;
    config.watchdog_period = RealtimeDuration{1'000'000};
    return config;
}

[[nodiscard]] inline StreamStatus fill_fault_frame(MutableFrame& frame, std::uint64_t sequence,
                                                   SampleIndex sample_idx,
                                                   SessionId session_id = 1) noexcept
{
    frame.header() = FrameHeader{
        .session_id = session_id,
        .sequence = sequence,
        .host_received_ns = sequence + 1,
        .schema_id = SchemaId{7},
        .source_clock_domain = ClockDomainId{1},
    };
    frame.block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = sample_idx,
        .device_tick_start = sample_idx,
        .payload_offset = 0,
        .payload_byte_count = 32,
        .signal_id = SignalId{1},
        .n_samples = 4,
    };
    return frame.set_used_sizes(1, 32);
}

class InjectedSource final : public NativeFrameSource
{
  public:
    explicit InjectedSource(FaultInjectionConfig config) noexcept : config_(config) {}

    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (cancelled_.load(std::memory_order_acquire))
        {
            return StreamStatus::stopped;
        }
        const auto idx = emitted_.load(std::memory_order_relaxed);
        if ((config_.point == FaultPoint::source_stall ||
             config_.point == FaultPoint::watchdog_delay) &&
            idx == config_.trigger_after)
        {
            blocked_.store(true, std::memory_order_release);
            cancelled_.wait(false, std::memory_order_acquire);
            return StreamStatus::stopped;
        }
        if (idx == config_.n_frames)
        {
            return StreamStatus::end_of_stream;
        }
        if (paced_count_ != nullptr && paced_count_->load(std::memory_order_acquire) < idx)
        {
            return StreamStatus::would_block;
        }

        const auto shifted =
            config_.point == FaultPoint::sample_idx_gap && idx >= config_.trigger_after;
        const auto restarted =
            config_.point == FaultPoint::source_restart && idx >= config_.trigger_after;
        auto status = fill_fault_frame(frame, idx, idx * 4 + (shifted ? 4 : 0), restarted ? 2 : 1);
        if (status == StreamStatus::ok && config_.point == FaultPoint::source_malformed_header &&
            idx == config_.trigger_after)
        {
            frame.header().schema_id = SchemaId{999};
        }
        if (status == StreamStatus::ok)
        {
            emitted_.fetch_add(1, std::memory_order_release);
        }
        return status;
    }

    void cancel() noexcept override
    {
        cancelled_.store(true, std::memory_order_release);
        cancelled_.notify_all();
    }

    StreamStatus reset() noexcept override
    {
        emitted_.store(0, std::memory_order_relaxed);
        blocked_.store(false, std::memory_order_relaxed);
        cancelled_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

    void pace_with(const std::atomic<std::uint64_t>& count) noexcept
    {
        paced_count_ = &count;
    }
    [[nodiscard]] bool blocked() const noexcept
    {
        return blocked_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::size_t emitted() const noexcept
    {
        return emitted_.load(std::memory_order_acquire);
    }

  private:
    FaultInjectionConfig config_{};
    const std::atomic<std::uint64_t>* paced_count_{};
    std::atomic<std::size_t> emitted_{};
    std::atomic<bool> blocked_{};
    std::atomic<bool> cancelled_{};
};

class InjectedProcessor final : public NativeFrameProcessor
{
  public:
    explicit InjectedProcessor(FaultInjectionConfig config) noexcept : config_(config) {}

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
            .required_resources = ProcessorResourceBounds{.frame_pool_leases = 2},
        };
    }

    StreamStatus process(FrameBorrow&, FrameEmitter& output) noexcept override
    {
        const auto idx = processed_.fetch_add(1, std::memory_order_relaxed);
        if ((config_.point == FaultPoint::processor_deadline ||
             config_.point == FaultPoint::ingress_overrun ||
             config_.point == FaultPoint::shutdown_timeout) &&
            idx == config_.trigger_after)
        {
            blocked_.store(true, std::memory_order_release);
            released_.wait(false, std::memory_order_acquire);
        }
        if (config_.point == FaultPoint::processor_fatal && idx == config_.trigger_after)
        {
            return StreamStatus::processor_failure;
        }
        if (config_.point == FaultPoint::pool_lease_leak && idx == config_.trigger_after)
        {
            FrameBorrow unused{};
            const auto status = output.try_acquire(unused);
            return status == StreamStatus::ok ? StreamStatus::processor_failure : status;
        }
        if (fatal_after_release_.load(std::memory_order_relaxed) && idx == config_.trigger_after)
        {
            return StreamStatus::processor_failure;
        }
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
        processed_.store(0, std::memory_order_relaxed);
        discontinuities_.store(0, std::memory_order_relaxed);
        blocked_.store(false, std::memory_order_relaxed);
        released_.store(false, std::memory_order_relaxed);
        fatal_after_release_.store(false, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    void release() noexcept
    {
        released_.store(true, std::memory_order_release);
        released_.notify_all();
    }
    void fail_after_release() noexcept
    {
        fatal_after_release_.store(true, std::memory_order_relaxed);
    }
    [[nodiscard]] bool blocked() const noexcept
    {
        return blocked_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::size_t discontinuities() const noexcept
    {
        return discontinuities_.load(std::memory_order_acquire);
    }

  private:
    FaultInjectionConfig config_{};
    std::atomic<std::size_t> processed_{};
    std::atomic<std::size_t> discontinuities_{};
    std::atomic<bool> blocked_{};
    std::atomic<bool> released_{};
    std::atomic<bool> fatal_after_release_{};
};

class InjectedActuator final : public NativeActuator
{
  public:
    explicit InjectedActuator(FaultInjectionConfig config) noexcept : config_(config) {}

    StreamStatus flush() noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        attempts_.store(0, std::memory_order_relaxed);
        applied_.store(0, std::memory_order_relaxed);
        blocked_.store(false, std::memory_order_relaxed);
        released_.store(false, std::memory_order_relaxed);
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
    [[nodiscard]] const std::atomic<std::uint64_t>& applied_counter() const noexcept
    {
        return applied_;
    }
    [[nodiscard]] std::uint64_t attempts() const noexcept
    {
        return attempts_.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool blocked() const noexcept
    {
        return blocked_.load(std::memory_order_acquire);
    }

  private:
    StreamStatus write(const ActuatorCommand&) noexcept override
    {
        const auto idx = attempts_.fetch_add(1, std::memory_order_relaxed);
        if (config_.point == FaultPoint::actuator_timeout && idx == config_.trigger_after)
        {
            blocked_.store(true, std::memory_order_release);
            released_.wait(false, std::memory_order_acquire);
        }
        if (config_.point == FaultPoint::actuator_failure && idx == config_.trigger_after)
        {
            return StreamStatus::actuator_failure;
        }
        applied_.fetch_add(1, std::memory_order_release);
        return StreamStatus::ok;
    }

    FaultInjectionConfig config_{};
    std::atomic<std::uint64_t> attempts_{};
    std::atomic<std::uint64_t> applied_{};
    std::atomic<bool> blocked_{};
    std::atomic<bool> released_{};
};

class InjectedObserver final : public NativeObserver
{
  public:
    explicit InjectedObserver(FaultInjectionConfig config) noexcept : config_(config) {}

    StreamStatus observe(FrameView) noexcept override
    {
        observed_.fetch_add(1, std::memory_order_relaxed);
        if (config_.point == FaultPoint::observer_blocked)
        {
            blocked_.store(true, std::memory_order_release);
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
        observed_.store(0, std::memory_order_relaxed);
        blocked_.store(false, std::memory_order_relaxed);
        released_.store(false, std::memory_order_relaxed);
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
    [[nodiscard]] bool blocked() const noexcept
    {
        return blocked_.load(std::memory_order_acquire);
    }

  private:
    FaultInjectionConfig config_{};
    std::atomic<std::size_t> observed_{};
    std::atomic<bool> blocked_{};
    std::atomic<bool> released_{};
};

} // namespace neurale::streaming::test
