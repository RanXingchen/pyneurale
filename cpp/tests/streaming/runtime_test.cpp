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
#include <cstdlib>
#include <optional>
#include <thread>
#include <type_traits>

#include "allocation_counter.h"
#include "check_returns.h"

namespace
{

using namespace neurale::streaming;

static_assert(std::is_abstract_v<FrameEmitter>);
static_assert(!std::is_copy_constructible_v<FrameEmitter>);
static_assert(!std::is_move_constructible_v<FrameEmitter>);

[[nodiscard]] StreamSchema make_schema()
{
    const std::array signals{
        SignalSchema{1, SignalDType::float32, 2, 4, 4, {1'000, 1}, 2},
    };
    return StreamSchema{7, signals};
}

[[nodiscard]] RealtimeConfig make_config(std::size_t ingress_capacity = 8)
{
    RealtimeConfig config;
    config.pool_capacity.source_owned = 1;
    config.pool_capacity.ingress_capacity = ingress_capacity;
    config.pool_capacity.processor_owned = 2;
    config.pool_capacity.critical_edge_capacity = ingress_capacity + 2;
    config.pool_capacity.actuator_owned = 1;
    config.buffer_size = 64;
    config.max_signal_blocks = 1;
    config.discontinuity_capacity = 2;
    config.gaps_per_discontinuity = 1;
    config.max_process_outputs = 2;
    config.max_flush_outputs = 1;
    config.fault_history_capacity = 4;
    return config;
}

[[nodiscard]] StreamStatus fill_frame(MutableFrame& frame, std::uint64_t sequence,
                                      SampleIndex sample_idx) noexcept
{
    frame.header() = FrameHeader{
        .session_id = 1,
        .sequence = sequence,
        .host_received_ns = static_cast<HostTimeNs>(sequence),
        .schema_id = 7,
    };
    frame.block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = sample_idx,
        .device_tick_start = sample_idx,
        .payload_offset = 0,
        .payload_byte_count = 32,
        .signal_id = 1,
        .n_samples = 4,
    };
    return frame.set_used_sizes(1, 32);
}

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

class SyntheticNativeSource final : public NativeFrameSource
{
  public:
    explicit SyntheticNativeSource(std::size_t n_frames) noexcept : n_frames_(n_frames) {}

    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (cancelled_.load(std::memory_order_acquire))
        {
            return StreamStatus::stopped;
        }
        const auto idx = emitted_.load(std::memory_order_relaxed);
        if (pause_at_.has_value() && idx == *pause_at_)
        {
            paused_.store(true, std::memory_order_release);
            paused_.notify_all();
            cancelled_.wait(false, std::memory_order_acquire);
            return StreamStatus::stopped;
        }
        if (idx != 0 && after_first_gate_ != nullptr &&
            !after_first_gate_->load(std::memory_order_acquire))
        {
            return StreamStatus::would_block;
        }
        if (paced_consumed_ != nullptr && paced_consumed_->load(std::memory_order_acquire) < idx)
        {
            return StreamStatus::would_block;
        }
        if (fail_at_.has_value() && idx == *fail_at_)
        {
            return StreamStatus::source_failure;
        }
        if (idx == n_frames_)
        {
            return StreamStatus::end_of_stream;
        }
        const auto gap = gap_after_first_ && idx != 0 ? std::size_t{1} : 0;
        const auto status = fill_frame(frame, idx + gap, (idx + gap) * 4);
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
        if (reset_status_ != StreamStatus::ok)
        {
            return reset_status_;
        }
        emitted_.store(0, std::memory_order_relaxed);
        paused_.store(false, std::memory_order_relaxed);
        cancelled_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

    void pace_with(std::atomic<std::size_t>& consumed) noexcept
    {
        paced_consumed_ = &consumed;
    }
    void gate_after_first(std::atomic<bool>& gate) noexcept
    {
        after_first_gate_ = &gate;
    }
    void pause_at(std::size_t idx) noexcept
    {
        pause_at_ = idx;
    }
    void fail_at(std::size_t idx) noexcept
    {
        fail_at_ = idx;
    }
    void gap_after_first() noexcept
    {
        gap_after_first_ = true;
    }
    void fail_reset(StreamStatus status) noexcept
    {
        reset_status_ = status;
    }

    [[nodiscard]] std::size_t emitted() const noexcept
    {
        return emitted_.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool paused() const noexcept
    {
        return paused_.load(std::memory_order_acquire);
    }

  private:
    std::size_t n_frames_{};
    std::optional<std::size_t> pause_at_;
    std::optional<std::size_t> fail_at_;
    std::atomic<std::size_t>* paced_consumed_{};
    std::atomic<bool>* after_first_gate_{};
    std::atomic<std::size_t> emitted_{};
    std::atomic<bool> paused_{};
    std::atomic<bool> cancelled_{};
    StreamStatus reset_status_{StreamStatus::ok};
    bool gap_after_first_{};
};

class BlockingNativeSource final : public NativeFrameSource
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
        cancel_calls_.fetch_add(1, std::memory_order_relaxed);
        cancelled_.store(true, std::memory_order_release);
        cancelled_.notify_all();
    }

    StreamStatus reset() noexcept override
    {
        entered_.store(false, std::memory_order_relaxed);
        cancelled_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

    [[nodiscard]] bool entered() const noexcept
    {
        return entered_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::size_t cancel_calls() const noexcept
    {
        return cancel_calls_.load(std::memory_order_relaxed);
    }

  private:
    std::atomic<bool> entered_{};
    std::atomic<bool> cancelled_{};
    std::atomic<std::size_t> cancel_calls_{};
};

class GatedNativeSource final : public NativeFrameSource
{
  public:
    explicit GatedNativeSource(std::size_t n_frames) noexcept : n_frames_(n_frames) {}

    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (!open_.load(std::memory_order_acquire))
        {
            entered_.store(true, std::memory_order_release);
            entered_.notify_all();
            open_.wait(false, std::memory_order_acquire);
        }
        if (cancelled_.load(std::memory_order_acquire))
        {
            return StreamStatus::stopped;
        }
        if (idx_ == n_frames_)
        {
            return StreamStatus::end_of_stream;
        }
        const auto status = fill_frame(frame, idx_, idx_ * 4);
        if (status == StreamStatus::ok)
        {
            ++idx_;
        }
        return status;
    }

    void cancel() noexcept override
    {
        cancelled_.store(true, std::memory_order_release);
        release();
    }

    StreamStatus reset() noexcept override
    {
        idx_ = 0;
        entered_.store(false, std::memory_order_relaxed);
        cancelled_.store(false, std::memory_order_relaxed);
        open_.store(false, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    void release() noexcept
    {
        open_.store(true, std::memory_order_release);
        open_.notify_all();
    }
    [[nodiscard]] bool entered() const noexcept
    {
        return entered_.load(std::memory_order_acquire);
    }

  private:
    std::size_t n_frames_{};
    std::size_t idx_{};
    std::atomic<bool> entered_{};
    std::atomic<bool> open_{};
    std::atomic<bool> cancelled_{};
};

class IdentityNativeProcessor final : public NativeFrameProcessor
{
  public:
    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        ++prepare_count_;
        const auto accepted_id =
            mismatched_input_ ? context.input_schema.id() + 1 : context.input_schema.id();
        const auto output_id =
            changed_output_schema_ ? context.input_schema.id() + 1 : context.input_schema.id();
        return {
            .accepted_input_schema = StreamSchema{accepted_id, context.input_schema.signals()},
            .output_schema = StreamSchema{output_id, context.input_schema.signals()},
            .max_process_outputs_per_input = declared_process_outputs_,
            .max_flush_outputs = declared_flush_outputs_,
            .can_forward_input = can_forward_input_,
            .required_resources = ProcessorResourceBounds{.workspace_bytes = workspace_bytes_,
                                                          .frame_pool_leases = required_leases_},
        };
    }

    StreamStatus process(FrameBorrow& frame, FrameEmitter& output) noexcept override
    {
        if (prepare_count_ == 0)
        {
            process_before_prepare_ = true;
            return StreamStatus::invalid_state;
        }
        const auto idx = processed_.fetch_add(1, std::memory_order_relaxed);
        if (block_first_ && idx == 0)
        {
            entered_.store(true, std::memory_order_release);
            entered_.notify_all();
            gate_.wait(false, std::memory_order_acquire);
        }
        if (fail_)
        {
            return StreamStatus::processor_failure;
        }
        if (zero_output_)
        {
            return StreamStatus::ok;
        }
        if (!acquired_output_)
        {
            if (retain_borrows_)
            {
                retained_input_ = std::move(frame);
            }
            const auto status = output.publish_input();
            invalidated_during_publish_ = !retain_borrows_ || retained_input_.get() == nullptr;
            return status;
        }

        const auto input_sequence = frame.header().sequence;
        if (multiple_outputs_)
        {
            const auto status = output.publish_input();
            if (status != StreamStatus::ok)
            {
                return status;
            }
        }
        FrameBorrow acquired{};
        auto status = output.try_acquire(acquired);
        if (status != StreamStatus::ok)
        {
            return status;
        }
        status = fill_frame(*acquired, input_sequence + 100, (input_sequence + 100) * 4);
        if (status != StreamStatus::ok)
        {
            return status;
        }
        if (retain_borrows_)
        {
            retained_input_ = std::move(frame);
            retained_output_ = std::move(acquired);
        }
        status = output.publish_acquired();
        invalidated_during_publish_ = !retain_borrows_ || retained_output_.get() == nullptr;
        return status;
    }

    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        discontinuities_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    StreamStatus flush(FrameEmitter& output) noexcept override
    {
        flushes_.fetch_add(1, std::memory_order_relaxed);
        if (!flush_output_)
        {
            return StreamStatus::ok;
        }
        FrameBorrow frame{};
        auto status = output.try_acquire(frame);
        if (status != StreamStatus::ok)
        {
            return status;
        }
        status = fill_frame(*frame, 1'000, 4'000);
        return status == StreamStatus::ok ? output.publish_acquired() : status;
    }

    StreamStatus reset() noexcept override
    {
        processed_.store(0, std::memory_order_relaxed);
        discontinuities_.store(0, std::memory_order_relaxed);
        flushes_.store(0, std::memory_order_relaxed);
        entered_.store(false, std::memory_order_relaxed);
        gate_.store(false, std::memory_order_relaxed);
        if (change_contract_on_reset_)
        {
            changed_output_schema_ = true;
        }
        return reset_status_;
    }

    void block_first() noexcept
    {
        block_first_ = true;
    }
    void release() noexcept
    {
        gate_.store(true, std::memory_order_release);
        gate_.notify_all();
    }
    [[nodiscard]] bool entered() const noexcept
    {
        return entered_.load(std::memory_order_acquire);
    }
    void fail_process() noexcept
    {
        fail_ = true;
    }
    void zero_output() noexcept
    {
        zero_output_ = true;
    }
    void acquired_output() noexcept
    {
        acquired_output_ = true;
    }
    void multiple_outputs() noexcept
    {
        acquired_output_ = true;
        multiple_outputs_ = true;
    }
    void emit_on_flush() noexcept
    {
        flush_output_ = true;
    }
    void retain_borrows() noexcept
    {
        retain_borrows_ = true;
    }
    [[nodiscard]] bool retained_borrows_are_stale() const noexcept
    {
        return retained_input_.get() == nullptr && retained_output_.get() == nullptr;
    }
    [[nodiscard]] bool invalidated_during_publish() const noexcept
    {
        return invalidated_during_publish_;
    }
    void declare_process_outputs(std::size_t count) noexcept
    {
        declared_process_outputs_ = count;
    }
    void declare_flush_outputs(std::size_t count) noexcept
    {
        declared_flush_outputs_ = count;
    }
    void disable_input_forwarding() noexcept
    {
        can_forward_input_ = false;
    }
    void mismatch_input_schema() noexcept
    {
        mismatched_input_ = true;
    }
    void change_output_schema() noexcept
    {
        changed_output_schema_ = true;
    }
    void change_contract_on_reset() noexcept
    {
        change_contract_on_reset_ = true;
    }
    void require_leases(std::size_t count) noexcept
    {
        required_leases_ = count;
    }
    [[nodiscard]] std::size_t prepare_count() const noexcept
    {
        return prepare_count_;
    }
    [[nodiscard]] bool process_before_prepare() const noexcept
    {
        return process_before_prepare_;
    }
    void fail_reset(StreamStatus status) noexcept
    {
        reset_status_ = status;
    }
    [[nodiscard]] std::size_t processed() const noexcept
    {
        return processed_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::size_t discontinuities() const noexcept
    {
        return discontinuities_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::size_t flushes() const noexcept
    {
        return flushes_.load(std::memory_order_relaxed);
    }

  private:
    std::size_t prepare_count_{};
    std::size_t declared_process_outputs_{2};
    std::size_t declared_flush_outputs_{1};
    std::size_t workspace_bytes_{};
    std::size_t required_leases_{2};
    std::atomic<std::size_t> processed_{};
    std::atomic<std::size_t> discontinuities_{};
    std::atomic<std::size_t> flushes_{};
    std::atomic<bool> entered_{};
    std::atomic<bool> gate_{};
    StreamStatus reset_status_{StreamStatus::ok};
    bool block_first_{};
    bool fail_{};
    bool zero_output_{};
    bool acquired_output_{};
    bool multiple_outputs_{};
    bool flush_output_{};
    bool can_forward_input_{true};
    bool mismatched_input_{};
    bool changed_output_schema_{};
    bool change_contract_on_reset_{};
    bool process_before_prepare_{};
    bool retain_borrows_{};
    bool invalidated_during_publish_{};
    FrameBorrow retained_input_{};
    FrameBorrow retained_output_{};
};

class CountingNativeConsumer final : public NativeFrameConsumer
{
  public:
    StreamStatus consume(FrameView frame) noexcept override
    {
        if (block_first_ && consumed_.load(std::memory_order_relaxed) == 0)
        {
            entered_.store(true, std::memory_order_release);
            entered_.notify_all();
            gate_.wait(false, std::memory_order_acquire);
        }
        if (fail_)
        {
            return StreamStatus::consumer_failure;
        }
        const auto idx = consumed_.fetch_add(1, std::memory_order_release);
        if (idx < sequences_.size())
        {
            sequences_[idx] = frame.header.sequence;
        }
        return StreamStatus::ok;
    }

    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        discontinuities_.fetch_add(1, std::memory_order_relaxed);
        return discontinuity_status_;
    }

    StreamStatus flush() noexcept override
    {
        flushes_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    StreamStatus reset() noexcept override
    {
        consumed_.store(0, std::memory_order_relaxed);
        discontinuities_.store(0, std::memory_order_relaxed);
        flushes_.store(0, std::memory_order_relaxed);
        entered_.store(false, std::memory_order_relaxed);
        gate_.store(false, std::memory_order_relaxed);
        return reset_status_;
    }

    void block_first() noexcept
    {
        block_first_ = true;
    }
    void release() noexcept
    {
        gate_.store(true, std::memory_order_release);
        gate_.notify_all();
    }
    void fail_consume() noexcept
    {
        fail_ = true;
    }
    void fail_discontinuity(StreamStatus status) noexcept
    {
        discontinuity_status_ = status;
    }
    void fail_reset(StreamStatus status) noexcept
    {
        reset_status_ = status;
    }

    [[nodiscard]] std::atomic<std::size_t>& consumed_counter() noexcept
    {
        return consumed_;
    }
    [[nodiscard]] std::size_t consumed() const noexcept
    {
        return consumed_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::size_t flushes() const noexcept
    {
        return flushes_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::size_t discontinuities() const noexcept
    {
        return discontinuities_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool entered() const noexcept
    {
        return entered_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t sequence(std::size_t idx) const noexcept
    {
        return sequences_[idx];
    }

  private:
    std::array<std::uint64_t, 256> sequences_{};
    std::atomic<std::size_t> consumed_{};
    std::atomic<std::size_t> discontinuities_{};
    std::atomic<std::size_t> flushes_{};
    std::atomic<bool> entered_{};
    std::atomic<bool> gate_{};
    StreamStatus reset_status_{StreamStatus::ok};
    StreamStatus discontinuity_status_{StreamStatus::ok};
    bool block_first_{};
    bool fail_{};
};

[[nodiscard]] bool prepare_and_arm(NativeStreamRunner& runtime)
{
    return runtime.prepare() == StreamStatus::ok && runtime.arm() == StreamStatus::ok;
}

int test_lifecycle_eos_order_and_consumer_faster()
{
    SyntheticNativeSource source{32};
    IdentityNativeProcessor processor;
    CountingNativeConsumer consumer;
    source.pace_with(consumer.consumed_counter());
    NativeStreamRunner runtime{make_schema(), make_config(4), source, processor, consumer};

    CHECK(runtime.state() == RuntimeState::created);
    CHECK(runtime.start() == StreamStatus::invalid_state);
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(processor.prepare_count() == 1);
    CHECK(runtime.prepare() == StreamStatus::invalid_state);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.run() == StreamStatus::ok);
    CHECK(runtime.state() == RuntimeState::stopped);
    CHECK(!processor.process_before_prepare());
    CHECK(consumer.consumed() == 32);
    for (std::size_t i = 0; i < 32; ++i)
    {
        CHECK(consumer.sequence(i) == i);
    }
    const auto stats = runtime.stats();
    CHECK(stats.frames_acquired == 32);
    CHECK(stats.frames_read == 32);
    CHECK(stats.frames_processed == 32);
    CHECK(stats.end_of_streams == 1);
    CHECK(stats.ingress_high_water_mark <= 1);
    CHECK(stats.queue_overruns == 0);
    CHECK(runtime.primary_fault() == std::nullopt);
    CHECK(runtime.outstanding_frames() == 0);
    CHECK(runtime.outstanding_discontinuities() == 0);
    CHECK(runtime.stop() == StreamStatus::ok);
    return 0;
}

int test_prepared_processor_contract_validation()
{
    {
        SyntheticNativeSource source{0};
        IdentityNativeProcessor processor;
        CountingNativeConsumer consumer;
        processor.mismatch_input_schema();
        NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

        CHECK(runtime.prepare() == StreamStatus::invalid_frame);
        CHECK(runtime.state() == RuntimeState::failed);
        CHECK(processor.prepare_count() == 1);
        CHECK(runtime.primary_fault().has_value());
        CHECK(runtime.primary_fault()->code == FaultCode::processor_prepare);
        CHECK(runtime.primary_fault()->stage == FaultStage::processor);
    }
    {
        SyntheticNativeSource source{0};
        IdentityNativeProcessor processor;
        CountingNativeConsumer consumer;
        processor.require_leases(3);
        NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

        CHECK(runtime.prepare() == StreamStatus::buffer_exhausted);
        CHECK(runtime.state() == RuntimeState::failed);
        CHECK(runtime.primary_fault().has_value());
        CHECK(runtime.primary_fault()->code == FaultCode::processor_prepare);
    }
    return 0;
}

int test_runtime_processor_contract_violations_use_fault_path()
{
    {
        SyntheticNativeSource source{1};
        IdentityNativeProcessor processor;
        CountingNativeConsumer consumer;
        processor.change_output_schema();
        NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

        CHECK(prepare_and_arm(runtime));
        CHECK(runtime.run() == StreamStatus::invalid_frame);
        CHECK(runtime.primary_fault().has_value());
        CHECK(runtime.primary_fault()->code == FaultCode::output_validation);
        CHECK(runtime.primary_fault()->stage == FaultStage::output);
        CHECK(runtime.outstanding_frames() == 0);
    }
    {
        SyntheticNativeSource source{1};
        IdentityNativeProcessor processor;
        CountingNativeConsumer consumer;
        processor.multiple_outputs();
        processor.declare_process_outputs(1);
        NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

        CHECK(prepare_and_arm(runtime));
        CHECK(runtime.run() == StreamStatus::output_limit);
        CHECK(runtime.primary_fault().has_value());
        CHECK(runtime.primary_fault()->code == FaultCode::output_contract);
        CHECK(runtime.outstanding_frames() == 0);
    }
    {
        SyntheticNativeSource source{1};
        IdentityNativeProcessor processor;
        CountingNativeConsumer consumer;
        processor.disable_input_forwarding();
        NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

        CHECK(prepare_and_arm(runtime));
        CHECK(runtime.run() == StreamStatus::invalid_state);
        CHECK(runtime.primary_fault().has_value());
        CHECK(runtime.primary_fault()->code == FaultCode::output_contract);
        CHECK(runtime.outstanding_frames() == 0);
    }
    {
        SyntheticNativeSource source{0};
        IdentityNativeProcessor processor;
        CountingNativeConsumer consumer;
        processor.emit_on_flush();
        processor.declare_flush_outputs(0);
        NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

        CHECK(prepare_and_arm(runtime));
        CHECK(runtime.run() == StreamStatus::output_limit);
        CHECK(runtime.primary_fault().has_value());
        CHECK(runtime.primary_fault()->code == FaultCode::output_contract);
        CHECK(runtime.outstanding_frames() == 0);
    }
    return 0;
}

int test_reset_preserves_prepared_contract()
{
    SyntheticNativeSource source{0};
    IdentityNativeProcessor processor;
    CountingNativeConsumer consumer;
    processor.change_contract_on_reset();
    NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

    CHECK(prepare_and_arm(runtime));
    CHECK(runtime.run() == StreamStatus::ok);
    CHECK(runtime.reset() == StreamStatus::ok);
    CHECK(runtime.state() == RuntimeState::created);
    CHECK(runtime.prepare() == StreamStatus::invalid_state);
    CHECK(runtime.state() == RuntimeState::failed);
    CHECK(processor.prepare_count() == 2);
    CHECK(runtime.primary_fault().has_value());
    CHECK(runtime.primary_fault()->code == FaultCode::processor_prepare);
    return 0;
}

int test_frame_emitter_publication_and_terminal_delivery()
{
    {
        SyntheticNativeSource source{1};
        IdentityNativeProcessor processor;
        CountingNativeConsumer consumer;
        NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

        CHECK(prepare_and_arm(runtime));
        CHECK(runtime.run() == StreamStatus::ok);
        const auto stats = runtime.stats();
        CHECK(consumer.consumed() == 1);
        CHECK(consumer.sequence(0) == 0);
        CHECK(consumer.flushes() == 1);
        CHECK(stats.frames_processed == 1);
        CHECK(stats.processor_outputs == 1);
        CHECK(stats.actuator_commands_enqueued == 1);
        CHECK(stats.actuator_commands_applied == 1);
        CHECK(runtime.outstanding_frames() == 0);
    }
    {
        SyntheticNativeSource source{1};
        IdentityNativeProcessor processor;
        CountingNativeConsumer consumer;
        processor.acquired_output();
        NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

        CHECK(prepare_and_arm(runtime));
        CHECK(runtime.run() == StreamStatus::ok);
        const auto stats = runtime.stats();
        CHECK(consumer.consumed() == 1);
        CHECK(consumer.sequence(0) == 100);
        CHECK(stats.processor_outputs == 1);
        CHECK(stats.actuator_commands_applied == 1);
        CHECK(runtime.outstanding_frames() == 0);
    }
    {
        SyntheticNativeSource source{1};
        IdentityNativeProcessor processor;
        CountingNativeConsumer consumer;
        processor.multiple_outputs();
        NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

        CHECK(prepare_and_arm(runtime));
        CHECK(runtime.run() == StreamStatus::ok);
        const auto stats = runtime.stats();
        CHECK(consumer.consumed() == 2);
        CHECK(consumer.sequence(0) == 0);
        CHECK(consumer.sequence(1) == 100);
        CHECK(consumer.flushes() == 1);
        CHECK(stats.frames_processed == 1);
        CHECK(stats.processor_outputs == 2);
        CHECK(stats.actuator_commands_enqueued == 2);
        CHECK(stats.actuator_commands_applied == 2);
        CHECK(runtime.outstanding_frames() == 0);
    }
    return 0;
}

int test_producer_faster_records_overrun_and_reclaims()
{
    constexpr std::size_t capacity = 2;
    SyntheticNativeSource source{100};
    IdentityNativeProcessor processor;
    CountingNativeConsumer consumer;
    // Block the processor (not the consumer) so it cannot drain the ingress while
    // the source bursts frames. This deterministically overruns the ingress with
    // FaultCode::queue_overrun as the sole fault: the critical edge never fills
    // (the processor publishes nothing while blocked) so FaultCode::
    // actuator_queue_overrun cannot race it for the primary slot, and the
    // actuator never blocks on the consumer so a failed precondition cannot strand
    // it in submit() during destruction.
    processor.block_first();
    std::atomic<bool> first_consuming{};
    source.gate_after_first(first_consuming);
    NativeStreamRunner runtime{make_schema(), make_config(capacity), source, processor, consumer};

    CHECK(prepare_and_arm(runtime));
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(wait_until([&] { return processor.entered(); }));
    first_consuming.store(true, std::memory_order_release);
    CHECK(wait_until([&] { return runtime.stats().queue_overruns == 1; }));
    processor.release();
    CHECK(runtime.join() == StreamStatus::queue_overflow);
    CHECK(runtime.state() == RuntimeState::failed);
    const auto fault = runtime.primary_fault();
    CHECK(fault.has_value());
    CHECK(fault->code == FaultCode::queue_overrun);
    CHECK(fault->stage == FaultStage::runtime);
    const auto stats = runtime.stats();
    CHECK(stats.ingress_high_water_mark == capacity);
    CHECK(stats.queue_overruns == 1);
    CHECK(stats.frames_acquired == stats.frames_processed + stats.aborted_frames);
    CHECK(runtime.outstanding_frames() == 0);
    return 0;
}

int test_terminal_emitter_invalidates_retained_borrows()
{
    for (const auto acquired_output : {false, true})
    {
        SyntheticNativeSource source{1};
        IdentityNativeProcessor processor;
        CountingNativeConsumer consumer;
        processor.retain_borrows();
        if (acquired_output)
        {
            processor.acquired_output();
        }
        NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

        CHECK(prepare_and_arm(runtime));
        CHECK(runtime.run() == StreamStatus::ok);
        CHECK(processor.invalidated_during_publish());
        CHECK(processor.retained_borrows_are_stale());
        CHECK(runtime.outstanding_frames() == 0);
    }
    return 0;
}

int test_stop_unblocks_source_read()
{
    BlockingNativeSource source;
    IdentityNativeProcessor processor;
    CountingNativeConsumer consumer;
    NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

    CHECK(prepare_and_arm(runtime));
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(wait_until([&] { return source.entered(); }));
    CHECK(runtime.stop() == StreamStatus::ok);
    CHECK(runtime.state() == RuntimeState::stopped);
    CHECK(source.cancel_calls() == 1);
    CHECK(runtime.stats().end_of_streams == 0);
    CHECK(runtime.outstanding_frames() == 0);
    return 0;
}

int test_abort_unblocks_source_without_flush()
{
    BlockingNativeSource source;
    IdentityNativeProcessor processor;
    CountingNativeConsumer consumer;
    NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

    CHECK(prepare_and_arm(runtime));
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(wait_until([&] { return source.entered(); }));
    CHECK(runtime.abort() == StreamStatus::ok);
    CHECK(runtime.state() == RuntimeState::stopped);
    CHECK(consumer.flushes() == 0);
    CHECK(runtime.outstanding_frames() == 0);
    return 0;
}

int test_run_and_concurrent_stop_share_join_safely()
{
    BlockingNativeSource source;
    IdentityNativeProcessor processor;
    CountingNativeConsumer consumer;
    NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};
    CHECK(prepare_and_arm(runtime));

    std::atomic<StreamStatus> run_status{StreamStatus::invalid_state};
    std::thread control([&] { run_status.store(runtime.run(), std::memory_order_release); });
    CHECK(wait_until([&] { return source.entered(); }));
    CHECK(runtime.stop() == StreamStatus::ok);
    control.join();
    CHECK(run_status.load(std::memory_order_acquire) == StreamStatus::ok);
    CHECK(runtime.state() == RuntimeState::stopped);
    CHECK(runtime.outstanding_frames() == 0);
    return 0;
}

int test_stop_with_full_ingress_drains_without_drop()
{
    constexpr std::size_t capacity = 4;
    SyntheticNativeSource source{100};
    IdentityNativeProcessor processor;
    CountingNativeConsumer consumer;
    std::atomic<bool> allow_remaining{};
    processor.block_first();
    source.gate_after_first(allow_remaining);
    source.pause_at(capacity + 1);
    NativeStreamRunner runtime{make_schema(), make_config(capacity), source, processor, consumer};

    CHECK(prepare_and_arm(runtime));
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(wait_until([&] { return processor.entered(); }));
    allow_remaining.store(true, std::memory_order_release);
    CHECK(wait_until([&] { return source.paused(); }));
    CHECK(runtime.stats().ingress_high_water_mark == capacity);
    processor.release();
    CHECK(runtime.stop() == StreamStatus::ok);
    CHECK(runtime.state() == RuntimeState::stopped);
    CHECK(runtime.stats().queue_overruns == 0);
    CHECK(runtime.stats().aborted_frames == 0);
    CHECK(consumer.consumed() <= capacity + 1);
    CHECK(runtime.outstanding_frames() == 0);
    return 0;
}

int test_processing_fault_drains_all_leases()
{
    SyntheticNativeSource source{100};
    IdentityNativeProcessor processor;
    CountingNativeConsumer consumer;
    processor.fail_process();
    source.pace_with(consumer.consumed_counter());
    NativeStreamRunner runtime{make_schema(), make_config(4), source, processor, consumer};

    CHECK(prepare_and_arm(runtime));
    CHECK(runtime.run() == StreamStatus::processor_failure);
    CHECK(runtime.state() == RuntimeState::failed);
    const auto fault = runtime.primary_fault();
    CHECK(fault.has_value());
    CHECK(fault->code == FaultCode::processor_process);
    CHECK(fault->stage == FaultStage::processor);
    CHECK(runtime.stats().frames_acquired == 1);
    CHECK(consumer.consumed() == 0);
    CHECK(processor.flushes() == 0);
    CHECK(runtime.outstanding_frames() == 0);
    CHECK(runtime.outstanding_discontinuities() == 0);
    return 0;
}

int test_source_and_consumer_fault_roles()
{
    {
        SyntheticNativeSource source{2};
        IdentityNativeProcessor processor;
        CountingNativeConsumer consumer;
        source.fail_at(0);
        NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};
        CHECK(prepare_and_arm(runtime));
        CHECK(runtime.run() == StreamStatus::source_failure);
        const auto fault = runtime.primary_fault();
        CHECK(fault.has_value());
        CHECK(fault->code == FaultCode::source_read);
        CHECK(fault->stage == FaultStage::source);
        CHECK(runtime.outstanding_frames() == 0);
    }
    {
        SyntheticNativeSource source{2};
        IdentityNativeProcessor processor;
        CountingNativeConsumer consumer;
        consumer.fail_consume();
        source.pace_with(consumer.consumed_counter());
        NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};
        CHECK(prepare_and_arm(runtime));
        CHECK(runtime.run() == StreamStatus::consumer_failure);
        const auto fault = runtime.primary_fault();
        CHECK(fault.has_value());
        CHECK(fault->code == FaultCode::actuator_write);
        CHECK(fault->stage == FaultStage::actuator);
        CHECK(runtime.outstanding_frames() == 0);
    }
    return 0;
}

int test_terminal_consumer_delivers_gaps_and_faults()
{
    {
        SyntheticNativeSource source{2};
        source.gap_after_first();
        IdentityNativeProcessor processor;
        CountingNativeConsumer consumer;
        source.pace_with(consumer.consumed_counter());
        NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};
        CHECK(prepare_and_arm(runtime));
        CHECK(runtime.run() == StreamStatus::ok);
        CHECK(processor.discontinuities() == 1);
        CHECK(consumer.discontinuities() == 1);
        CHECK(runtime.stats().discontinuities == 1);
        CHECK(runtime.outstanding_frames() == 0);
        CHECK(runtime.outstanding_discontinuities() == 0);
    }
    {
        SyntheticNativeSource source{2};
        source.gap_after_first();
        IdentityNativeProcessor processor;
        CountingNativeConsumer consumer;
        consumer.fail_discontinuity(StreamStatus::consumer_failure);
        source.pace_with(consumer.consumed_counter());
        NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};
        CHECK(prepare_and_arm(runtime));
        CHECK(runtime.run() == StreamStatus::consumer_failure);
        const auto fault = runtime.primary_fault();
        CHECK(fault.has_value());
        CHECK(fault->code == FaultCode::consumer_discontinuity);
        CHECK(fault->stage == FaultStage::consumer);
        CHECK(consumer.discontinuities() == 1);
        CHECK(runtime.outstanding_frames() == 0);
        CHECK(runtime.outstanding_discontinuities() == 0);
    }
    return 0;
}

int test_steady_state_does_not_allocate()
{
    constexpr std::size_t n_frames = 32;
    GatedNativeSource source{n_frames};
    IdentityNativeProcessor processor;
    CountingNativeConsumer consumer;
    processor.multiple_outputs();
    auto config = make_config(n_frames);
    config.pool_capacity.critical_edge_capacity = n_frames * 2;
    NativeStreamRunner runtime{make_schema(), config, source, processor, consumer};

    CHECK(prepare_and_arm(runtime));
    CHECK(runtime.start() == StreamStatus::ok);
    CHECK(wait_until([&] { return source.entered(); }));
    const auto before = allocations.load(std::memory_order_relaxed);
    source.release();
    CHECK(runtime.join() == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == before);
    CHECK(consumer.consumed() == n_frames * 2);
    CHECK(runtime.stats().processor_outputs == n_frames * 2);
    CHECK(runtime.outstanding_frames() == 0);
    return 0;
}

int test_zero_output_and_bounded_flush()
{
    SyntheticNativeSource source{1};
    IdentityNativeProcessor processor;
    CountingNativeConsumer consumer;
    processor.zero_output();
    processor.emit_on_flush();
    NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

    CHECK(prepare_and_arm(runtime));
    CHECK(runtime.run() == StreamStatus::ok);
    const auto stats = runtime.stats();
    CHECK(stats.zero_output_frames == 1);
    CHECK(stats.processor_outputs == 0);
    CHECK(stats.flush_outputs == 1);
    CHECK(consumer.consumed() == 1);
    CHECK(consumer.sequence(0) == 1'000);
    CHECK(stats.actuator_commands_enqueued == 1);
    CHECK(stats.actuator_commands_applied == 1);
    CHECK(processor.flushes() == 1);
    CHECK(consumer.flushes() == 1);
    CHECK(runtime.outstanding_frames() == 0);
    return 0;
}

int test_repeated_prepare_run_reset()
{
    SyntheticNativeSource source{2};
    IdentityNativeProcessor processor;
    CountingNativeConsumer consumer;
    NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

    for (std::size_t iteration = 0; iteration < 100; ++iteration)
    {
        CHECK(prepare_and_arm(runtime));
        CHECK(runtime.run() == StreamStatus::ok);
        CHECK(processor.processed() == 2);
        CHECK(consumer.consumed() == 2);
        CHECK(runtime.outstanding_frames() == 0);
        CHECK(runtime.reset() == StreamStatus::ok);
        CHECK(runtime.state() == RuntimeState::created);
        CHECK(processor.processed() == 0);
        CHECK(processor.discontinuities() == 0);
        CHECK(processor.flushes() == 0);
        CHECK(consumer.consumed() == 0);
        CHECK(consumer.discontinuities() == 0);
        CHECK(consumer.flushes() == 0);
    }
    return 0;
}

int test_diagnostics_are_safe_during_prepare_and_reset()
{
    SyntheticNativeSource source{1};
    IdentityNativeProcessor processor;
    CountingNativeConsumer consumer;
    NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};
    std::atomic<bool> done{false};
    std::thread reader{[&]
                       {
                           std::array<FaultRecord, 4> faults{};
                           std::array<ObserverDropRange, 4> drops{};
                           while (!done.load(std::memory_order_acquire))
                           {
                               static_cast<void>(runtime.primary_fault());
                               static_cast<void>(runtime.copy_fault_history(faults));
                               static_cast<void>(runtime.outstanding_frames());
                               static_cast<void>(runtime.outstanding_discontinuities());
                               static_cast<void>(runtime.observer_stats(1));
                               static_cast<void>(runtime.copy_observer_drop_ranges(1, drops));
                               static_cast<void>(runtime.realtime_configuration_status());
                               std::this_thread::sleep_for(std::chrono::microseconds(50));
                           }
                       }};

    bool success = true;
    for (std::size_t iteration = 0; iteration < 25; ++iteration)
    {
        if (!prepare_and_arm(runtime) || runtime.run() != StreamStatus::ok ||
            runtime.reset() != StreamStatus::ok)
        {
            success = false;
            break;
        }
    }
    done.store(true, std::memory_order_release);
    reader.join();
    CHECK(success);
    return 0;
}

int test_reset_fault_is_stable()
{
    SyntheticNativeSource source{0};
    IdentityNativeProcessor processor;
    CountingNativeConsumer consumer;
    NativeStreamRunner runtime{make_schema(), make_config(2), source, processor, consumer};

    CHECK(prepare_and_arm(runtime));
    CHECK(runtime.run() == StreamStatus::ok);
    source.fail_reset(StreamStatus::source_failure);
    processor.fail_reset(StreamStatus::processor_failure);
    consumer.fail_reset(StreamStatus::consumer_failure);
    CHECK(runtime.reset() == StreamStatus::source_failure);
    const auto primary = runtime.primary_fault();
    CHECK(primary.has_value());
    CHECK(primary->code == FaultCode::source_reset);
    std::array<FaultRecord, 4> history{};
    CHECK(runtime.copy_fault_history(history) == 2);
    CHECK(history[0].code == FaultCode::processor_reset);
    CHECK(history[1].code == FaultCode::actuator_reset);
    return 0;
}

int test_automatic_runtime_resources_follow_prepared_contracts()
{
    auto config = RealtimeConfig{};
    config.automatic_resources = true;
    config.max_output_age = RealtimeDuration{0};
    config.watchdog_period = RealtimeDuration{0};
    SyntheticNativeSource source{3};
    IdentityNativeProcessor processor;
    CountingNativeConsumer consumer;
    NativeStreamRunner runtime{make_schema(), config, source, processor, consumer};

    CHECK(runtime.prepare() == StreamStatus::ok);
    const auto& resolved = runtime.resolved_config();
    CHECK(resolved.buffer_size == 32);
    CHECK(resolved.max_signal_blocks == 1);
    CHECK(resolved.max_process_outputs == 2);
    CHECK(resolved.max_flush_outputs == 1);
    CHECK(resolved.discontinuity_capacity == 1);
    CHECK(resolved.gaps_per_discontinuity == 1);
    CHECK(resolved.pool_capacity.source_owned == 1);
    CHECK(resolved.pool_capacity.ingress_capacity == 27);
    CHECK(resolved.pool_capacity.processor_owned == 2);
    CHECK(resolved.pool_capacity.critical_edge_capacity == 27);
    CHECK(resolved.pool_capacity.actuator_owned == 1);
    CHECK(resolved.pool_capacity.observer_edge_capacity == 0);
    CHECK(resolved.pool_capacity.reserve == 2);
    CHECK(resolved.max_output_age == std::chrono::milliseconds{108});
    CHECK(resolved.watchdog_period == std::chrono::milliseconds{10});
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.run() == StreamStatus::ok);
    CHECK(consumer.consumed() == 3);
    return 0;
}

} // namespace

int main()
{
    const std::array tests{
        test_lifecycle_eos_order_and_consumer_faster,
        test_prepared_processor_contract_validation,
        test_runtime_processor_contract_violations_use_fault_path,
        test_reset_preserves_prepared_contract,
        test_frame_emitter_publication_and_terminal_delivery,
        test_terminal_emitter_invalidates_retained_borrows,
        test_producer_faster_records_overrun_and_reclaims,
        test_stop_unblocks_source_read,
        test_abort_unblocks_source_without_flush,
        test_run_and_concurrent_stop_share_join_safely,
        test_stop_with_full_ingress_drains_without_drop,
        test_processing_fault_drains_all_leases,
        test_source_and_consumer_fault_roles,
        test_terminal_consumer_delivers_gaps_and_faults,
        test_steady_state_does_not_allocate,
        test_zero_output_and_bounded_flush,
        test_repeated_prepare_run_reset,
        test_diagnostics_are_safe_during_prepare_and_reset,
        test_reset_fault_is_stable,
        test_automatic_runtime_resources_follow_prepared_contracts,
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
