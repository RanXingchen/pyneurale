/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/devices/simulation.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "allocation_counter.h"
#include "check_returns.h"

namespace
{

namespace ds = neurale::devices::simulation;
namespace ss = neurale::signal::simulation;
using namespace neurale::streaming;

class BlockingClock final : public NativeClock
{
  public:
    [[nodiscard]] HostTimeNs now_ns() noexcept override
    {
        return now_.load(std::memory_order_acquire);
    }

    void wait_until(HostTimeNs deadline) noexcept override
    {
        std::unique_lock lock{mutex_};
        const auto revision = revision_;
        condition_.wait(
            lock, [&]
            { return revision_ != revision || now_.load(std::memory_order_acquire) >= deadline; });
    }

    void wake() noexcept override
    {
        {
            std::lock_guard lock{mutex_};
            ++revision_;
        }
        condition_.notify_all();
    }

  private:
    std::atomic<HostTimeNs> now_{1'000'000};
    std::mutex mutex_;
    std::condition_variable condition_;
    std::uint64_t revision_{};
};

class AdvancingClock final : public NativeClock
{
  public:
    [[nodiscard]] HostTimeNs now_ns() noexcept override
    {
        return now_;
    }
    void wait_until(HostTimeNs deadline) noexcept override
    {
        waited_ns_ += deadline - now_;
        now_ = deadline;
        last_deadline_ = deadline;
        ++wait_count_;
    }
    void wake() noexcept override {}

    void set_now(HostTimeNs value) noexcept
    {
        now_ = value;
    }
    [[nodiscard]] std::size_t wait_count() const noexcept
    {
        return wait_count_;
    }
    [[nodiscard]] HostTimeNs last_deadline() const noexcept
    {
        return last_deadline_;
    }
    [[nodiscard]] HostTimeNs waited_ns() const noexcept
    {
        return waited_ns_;
    }

  private:
    HostTimeNs now_{1'000};
    HostTimeNs last_deadline_{};
    HostTimeNs waited_ns_{};
    std::size_t wait_count_{};
};

class Passthrough final : public NativeFrameProcessor
{
  public:
    [[nodiscard]] PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return {
            .accepted_input_schema = context.input_schema.clone(),
            .output_schema = context.input_schema.clone(),
            .max_process_outputs_per_input = 1,
            .max_flush_outputs = 0,
            .can_forward_input = true,
            .required_resources = {.workspace_bytes = 0, .frame_pool_leases = 1},
        };
    }

    StreamStatus process(FrameBorrow&, FrameEmitter& emitter) noexcept override
    {
        return emitter.publish_input();
    }
    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        ++discontinuities;
        return StreamStatus::ok;
    }
    StreamStatus flush(FrameEmitter&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        discontinuities = 0;
        return StreamStatus::ok;
    }

    std::size_t discontinuities{};
};

class TimestampConsumer final : public NativeFrameConsumer
{
  public:
    StreamStatus consume(FrameView frame) noexcept override
    {
        ++count;
        received = frame.header.host_received_ns;
        source_received = has_flag(frame.header.flags, FrameFlags::source_received);
        return StreamStatus::ok;
    }
    StreamStatus handle_discontinuity(const Discontinuity& discontinuity) noexcept override
    {
        ++discontinuities;
        last_reason = discontinuity.reason;
        return StreamStatus::ok;
    }
    StreamStatus flush() noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        count = 0;
        received = 0;
        source_received = false;
        discontinuities = 0;
        return StreamStatus::ok;
    }

    std::size_t count{};
    HostTimeNs received{};
    bool source_received{};
    std::size_t discontinuities{};
    GapReason last_reason{GapReason::source_gap};
};

[[nodiscard]] RealtimeConfig runtime_config(std::size_t buffer_size)
{
    RealtimeConfig result;
    result.pool_capacity.source_owned = 1;
    result.pool_capacity.ingress_capacity = 4;
    result.pool_capacity.processor_owned = 2;
    result.pool_capacity.critical_edge_capacity = 5;
    result.pool_capacity.actuator_owned = 1;
    result.buffer_size = buffer_size;
    result.max_signal_blocks = 1;
    result.discontinuity_capacity = 2;
    result.gaps_per_discontinuity = 1;
    result.max_process_outputs = 1;
    result.fault_history_capacity = 2;
    return result;
}

[[nodiscard]] ds::SimulatedNeuralSourceConfig config(std::uint32_t nominal, std::uint32_t maximum,
                                                     std::uint64_t total)
{
    return {
        .session_id = 4,
        .schema_id = 7,
        .signal_id = 9,
        .clock_domain = 3,
        .nominal_samples_per_frame = nominal,
        .max_samples_per_frame = maximum,
        .fs = {1'000, 1},
        .physical_unit = PhysicalUnit::volts,
        .channel_set_id = 11,
        .calibration_id = 12,
        .reference_id = 13,
        .initial_sample_idx = 100,
        .total_sample_count = total,
        .device_ticks = true,
        .initial_device_tick = 5'000,
        .clock_sync_uncertainty_ns = 25,
        .paced = false,
    };
}

[[nodiscard]] int read_payload(ds::SimulatedNeuralSource& source, FrameLease& lease,
                               std::vector<double>& values)
{
    const auto status = source.read(lease.frame());
    if (status != StreamStatus::ok)
        return -static_cast<int>(status);
    const auto view = lease.view();
    const auto* begin = reinterpret_cast<const double*>(view.payload.data());
    values.insert(values.end(), begin, begin + view.payload.size() / sizeof(double));
    return 0;
}

int run()
{
    neurale::streaming::NeuralIntentState intent_state;
    std::atomic<bool> finished{false};
    std::thread publisher(
        [&]
        {
            for (std::uint64_t sequence = 1; sequence <= 100000; ++sequence)
            {
                intent_state.publish({sequence, static_cast<double>(sequence),
                                      -static_cast<double>(sequence), sequence, sequence, sequence,
                                      sequence, true});
            }
            finished.store(true, std::memory_order_release);
        });
    bool coherent = true;
    do
    {
        const auto snapshot = intent_state.read_intent();
        if (snapshot.valid)
            coherent = coherent && snapshot.intent_x == static_cast<double>(snapshot.sequence) &&
                       snapshot.intent_y == -snapshot.intent_x &&
                       snapshot.context_ordinal == snapshot.sequence &&
                       snapshot.source_time_ns == snapshot.sequence &&
                       snapshot.source_frame_sequence == snapshot.sequence &&
                       snapshot.source_sample_index == snapshot.sequence;
    } while (!finished.load(std::memory_order_acquire));
    publisher.join();
    CHECK(coherent);

    const std::array constants = {2.0, -1.0};
    const auto generator = ss::SignalGenerator::constant(2, 1'000.0, constants);
    auto source_config = config(4, 6, 10);
    source_config.channel_names = {"left", "right"};
    source_config.channel_impedances_ohm = {5'000.0, 20'000.0};
    ds::SimulatedNeuralSource source{generator, std::move(source_config)};
    const auto& schema = source.schema();
    CHECK(schema.id() == 7);
    CHECK(schema.signals().size() == 1);
    const auto& signal = schema.signals().front();
    CHECK(signal.id == 9);
    CHECK(signal.n_channels == 2);
    CHECK(signal.nominal_block_samples == 4);
    CHECK(signal.max_block_samples == 6);
    CHECK(signal.fs.numerator == 1'000);
    CHECK(signal.fs.denominator == 1);
    CHECK(signal.dtype == SignalDType::float64);
    CHECK(signal.layout == SignalLayout::sample_major);
    CHECK(signal.kind == SignalKind::sampled);
    CHECK(signal.device_tick_tracking == DeviceTickTracking::sample_counter);
    CHECK(signal.physical_unit == PhysicalUnit::volts);
    CHECK(signal.channel_set_id == 11);
    CHECK(signal.channel_names == std::vector<std::string>({"left", "right"}));
    CHECK(signal.channel_impedances_ohm == std::vector<double>({5'000.0, 20'000.0}));
    CHECK(signal.calibration_id == 12);
    CHECK(signal.reference_id == 13);

    FramePool pool{2, signal.max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    FrameValidator validator{schema};
    std::vector<double> collected;
    for (std::uint64_t frame_idx = 0; frame_idx < 3; ++frame_idx)
    {
        CHECK(source.read(lease.frame()) == StreamStatus::ok);
        const auto view = lease.view();
        const auto n_expected = frame_idx < 2 ? 4U : 2U;
        CHECK(view.header.sequence == frame_idx);
        CHECK(view.header.host_received_ns == 0);
        CHECK(!has_flag(view.header.flags, FrameFlags::source_received));
        CHECK(has_flag(view.header.flags, FrameFlags::source_tick));
        CHECK(view.header.source_tick == 5'000 + frame_idx * 4);
        CHECK(view.blocks.front().sample_idx_start == 100 + frame_idx * 4);
        CHECK(view.blocks.front().n_samples == n_expected);
        CHECK(view.blocks.front().device_tick_start == 5'000 + frame_idx * 4);
        CHECK(view.blocks.front().clock_sync.device_tick_reference == 5'000 + frame_idx * 4);
        CHECK(view.blocks.front().clock_sync.device_tick_rate.numerator == 1'000);
        CHECK(view.blocks.front().clock_sync.clock_domain == 3);
        CHECK(view.blocks.front().clock_sync.uncertainty_ns == 25);
        CHECK(has_flag(view.blocks.front().clock_sync.flags, ClockSyncFlags::synchronized));
        CHECK(validator.validate(view) == FrameValidationError::none);
        const auto* begin = reinterpret_cast<const double*>(view.payload.data());
        collected.insert(collected.end(), begin, begin + n_expected * 2);
    }
    CHECK(source.read(lease.frame()) == StreamStatus::end_of_stream);
    CHECK(source.frames_emitted() == 3);
    CHECK(source.samples_emitted() == 10);
    CHECK(collected.size() == 20);
    for (std::size_t i = 0; i < collected.size(); i += 2)
    {
        CHECK(collected[i] == 2.0);
        CHECK(collected[i + 1] == -1.0);
    }

    CHECK(source.reset() == StreamStatus::ok);
    CHECK(source.read(lease.frame()) == StreamStatus::ok);
    CHECK(lease.view().header.sequence == 0);
    CHECK(lease.view().blocks.front().sample_idx_start == 100);

    // A zero control-plane ID requests a source-owned run identity. Separate
    // sources receive distinct nonzero IDs, while reset keeps the same run ID.
    auto first_auto_config = config(4, 4, 4);
    auto second_auto_config = config(4, 4, 4);
    first_auto_config.session_id = 0;
    second_auto_config.session_id = 0;
    ds::SimulatedNeuralSource first_auto_source{generator, first_auto_config};
    ds::SimulatedNeuralSource second_auto_source{generator, second_auto_config};
    FramePool auto_pool{2, first_auto_source.schema().signals().front().max_block_bytes, 1};
    FrameLease first_auto_lease;
    FrameLease second_auto_lease;
    CHECK(auto_pool.try_acquire(first_auto_lease) == StreamStatus::ok);
    CHECK(auto_pool.try_acquire(second_auto_lease) == StreamStatus::ok);
    CHECK(first_auto_source.read(first_auto_lease.frame()) == StreamStatus::ok);
    CHECK(second_auto_source.read(second_auto_lease.frame()) == StreamStatus::ok);
    const auto first_auto_session_id = first_auto_lease.view().header.session_id;
    CHECK(first_auto_session_id != 0);
    CHECK(second_auto_lease.view().header.session_id != 0);
    CHECK(first_auto_session_id != second_auto_lease.view().header.session_id);
    CHECK(first_auto_source.reset() == StreamStatus::ok);
    CHECK(first_auto_source.read(first_auto_lease.frame()) == StreamStatus::ok);
    CHECK(first_auto_lease.view().header.session_id == first_auto_session_id);

    // Different valid frame sizes preserve the absolute-position signal.
    const std::array freqs = {17.0, 23.0};
    const std::array amps = {1.0, 0.5};
    const std::array phases = {0.0, 0.25};
    const auto tones = ss::SignalGenerator::tones(2, 1'000.0, 1, freqs, amps, phases);
    ds::SimulatedNeuralSource chunks3{tones, config(3, 5, 15)};
    ds::SimulatedNeuralSource chunks5{tones, config(5, 5, 15)};
    FramePool pool3{1, chunks3.schema().signals().front().max_block_bytes, 1};
    FramePool pool5{1, chunks5.schema().signals().front().max_block_bytes, 1};
    FrameLease lease3;
    FrameLease lease5;
    CHECK(pool3.try_acquire(lease3) == StreamStatus::ok);
    CHECK(pool5.try_acquire(lease5) == StreamStatus::ok);
    std::vector<double> values3;
    std::vector<double> values5;
    for (int i = 0; i < 5; ++i)
        CHECK(read_payload(chunks3, lease3, values3) == 0);
    for (int i = 0; i < 3; ++i)
        CHECK(read_payload(chunks5, lease5, values5) == 0);
    CHECK(values3 == values5);
    std::array<double, 30> expected_tones{};
    CHECK(tones.generate(100, 15, expected_tones) == ss::GenerationStatus::ok);
    CHECK(std::equal(values3.begin(), values3.end(), expected_tones.begin()));

    // Construction and reset delays are control-plane time, not acquisition time.
    AdvancingClock epoch_clock;
    auto epoch_config = config(4, 4, 8);
    epoch_config.initial_sample_idx = 0;
    epoch_config.paced = true;
    ds::SimulatedNeuralSource epoch_source{generator, epoch_config, epoch_clock};
    FramePool epoch_pool{1, epoch_source.schema().signals().front().max_block_bytes, 1};
    FrameLease epoch_lease;
    CHECK(epoch_pool.try_acquire(epoch_lease) == StreamStatus::ok);
    epoch_clock.set_now(1'000'000'000);
    CHECK(epoch_source.read(epoch_lease.frame()) == StreamStatus::ok);
    CHECK(epoch_clock.wait_count() == 0);
    CHECK(epoch_lease.view().blocks.front().clock_sync.host_time_reference_ns == 1'000'000'000);
    CHECK(epoch_source.read(epoch_lease.frame()) == StreamStatus::ok);
    CHECK(epoch_clock.wait_count() > 0);
    CHECK(epoch_clock.waited_ns() == 4'000'000);
    CHECK(epoch_clock.last_deadline() == 1'004'000'000);

    CHECK(epoch_source.reset() == StreamStatus::ok);
    epoch_clock.set_now(2'000'000'000);
    CHECK(epoch_source.read(epoch_lease.frame()) == StreamStatus::ok);
    const auto waits_before_second_segment = epoch_clock.wait_count();
    CHECK(epoch_clock.waited_ns() == 4'000'000);
    CHECK(epoch_lease.view().blocks.front().clock_sync.host_time_reference_ns == 2'000'000'000);
    CHECK(epoch_source.read(epoch_lease.frame()) == StreamStatus::ok);
    CHECK(epoch_clock.wait_count() > waits_before_second_segment);
    CHECK(epoch_clock.waited_ns() == 8'000'000);
    CHECK(epoch_clock.last_deadline() == 2'004'000'000);

    // Invalid caller capacity does not consume sequence or sample position.
    FramePool too_small{1, 8, 1};
    FrameLease small_lease;
    CHECK(too_small.try_acquire(small_lease) == StreamStatus::ok);
    AdvancingClock invalid_frame_clock;
    auto invalid_frame_config = config(4, 4, 4);
    invalid_frame_config.paced = true;
    ds::SimulatedNeuralSource invalid_frame_source{generator, invalid_frame_config,
                                                   invalid_frame_clock};
    CHECK(invalid_frame_source.read(small_lease.frame()) == StreamStatus::invalid_frame);
    CHECK(invalid_frame_source.frames_emitted() == 0);
    CHECK(invalid_frame_source.samples_emitted() == 0);
    invalid_frame_clock.set_now(3'000'000'000);
    CHECK(invalid_frame_source.read(lease.frame()) == StreamStatus::ok);
    CHECK(invalid_frame_clock.wait_count() == 0);
    CHECK(lease.view().blocks.front().clock_sync.host_time_reference_ns == 3'000'000'000);

    auto unticked_config = config(4, 4, 4);
    unticked_config.device_ticks = false;
    unticked_config.clock_sync_uncertainty_ns = 0;
    ds::SimulatedNeuralSource unticked_source{generator, unticked_config};
    FramePool unticked_pool{1, unticked_source.schema().signals().front().max_block_bytes, 1};
    FrameLease unticked_lease;
    CHECK(unticked_pool.try_acquire(unticked_lease) == StreamStatus::ok);
    CHECK(unticked_source.read(unticked_lease.frame()) == StreamStatus::ok);
    CHECK(unticked_source.schema().signals().front().device_tick_tracking ==
          DeviceTickTracking::unavailable);
    CHECK(!has_flag(unticked_lease.view().header.flags, FrameFlags::source_tick));
    CHECK(unticked_lease.view().blocks.front().clock_sync.generation == 0);

    auto invalid_unticked_clock_config = unticked_config;
    invalid_unticked_clock_config.clock_sync_uncertainty_ns = 1;
    bool rejected_unticked_uncertainty = false;
    try
    {
        ds::SimulatedNeuralSource invalid_unticked_clock_source{generator,
                                                                invalid_unticked_clock_config};
        static_cast<void>(invalid_unticked_clock_source.schema());
    }
    catch (const std::invalid_argument&)
    {
        rejected_unticked_uncertainty = true;
    }
    CHECK(rejected_unticked_uncertainty);

    // A prepared direct read allocates no C++ heap memory.
    auto allocation_config = config(4, 4, 8);
    allocation_config.paced = true;
    AdvancingClock allocation_clock;
    ds::SimulatedNeuralSource allocation_source{generator, allocation_config, allocation_clock};
    FramePool allocation_pool{1, allocation_source.schema().signals().front().max_block_bytes, 1};
    FrameLease allocation_lease;
    CHECK(allocation_pool.try_acquire(allocation_lease) == StreamStatus::ok);
    CHECK(allocation_source.read(allocation_lease.frame()) == StreamStatus::ok);
    const auto before = allocations.load(std::memory_order_acquire);
    CHECK(allocation_source.read(allocation_lease.frame()) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_acquire) == before);

    // Cancellation wakes a paced read rather than waiting for its deadline.
    BlockingClock clock;
    auto paced_config = config(1, 1, 2);
    paced_config.fs = {1, 1};
    paced_config.initial_sample_idx = 0;
    paced_config.paced = true;
    const auto slow_generator = ss::SignalGenerator::zeros(1, 1.0);
    ds::SimulatedNeuralSource paced_source{slow_generator, paced_config, clock};
    FramePool paced_pool{2, paced_source.schema().signals().front().max_block_bytes, 1};
    FrameLease first;
    FrameLease blocked;
    CHECK(paced_pool.try_acquire(first) == StreamStatus::ok);
    CHECK(paced_pool.try_acquire(blocked) == StreamStatus::ok);
    CHECK(paced_source.read(first.frame()) == StreamStatus::ok);
    std::atomic<StreamStatus> blocked_status{StreamStatus::ok};
    std::thread reader([&] { blocked_status.store(paced_source.read(blocked.frame())); });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const auto cancel_started = std::chrono::steady_clock::now();
    paced_source.cancel();
    reader.join();
    CHECK(std::chrono::steady_clock::now() - cancel_started < std::chrono::milliseconds(500));
    CHECK(blocked_status.load() == StreamStatus::stopped);
    CHECK(paced_source.reset() == StreamStatus::ok);
    CHECK(paced_source.read(first.frame()) == StreamStatus::ok);

    // Host-receive provenance belongs to the generic runtime ingress edge.
    ds::SimulatedNeuralSource runtime_source{generator, config(4, 4, 4)};
    Passthrough processor;
    TimestampConsumer consumer;
    NativeStreamRunner runner{
        runtime_source.schema().clone(),
        runtime_config(runtime_source.schema().signals().front().max_block_bytes), runtime_source,
        processor, consumer};
    CHECK(runner.prepare() == StreamStatus::ok);
    CHECK(runner.arm() == StreamStatus::ok);
    CHECK(runner.run() == StreamStatus::ok);
    CHECK(consumer.count == 1);
    CHECK(consumer.received != 0);
    CHECK(consumer.source_received);
    CHECK(runner.outstanding_frames() == 0);
    CHECK(runner.outstanding_discontinuities() == 0);

    // Acquisition events are fixed to the next data-frame ordinal. Known loss
    // advances the absolute sample/device timeline but not data-frame sequence.
    auto loss_config = config(4, 4, 16);
    loss_config.events = {
        {.frame_ordinal = 1, .kind = ds::AcquisitionEventKind::sample_loss, .n_samples = 2},
        {.frame_ordinal = 2, .kind = ds::AcquisitionEventKind::sample_loss, .n_samples = 1},
    };
    ds::SimulatedNeuralSource loss_source{tones, loss_config};
    CHECK(loss_source.produces_discontinuities());
    FramePool loss_pool{1, loss_source.schema().signals().front().max_block_bytes, 1};
    DiscontinuityPool loss_discontinuities{2, 1};
    FrameLease loss_frame;
    CHECK(loss_pool.try_acquire(loss_frame) == StreamStatus::ok);
    DiscontinuityLease loss_gap;
    CHECK(loss_discontinuities.try_acquire(loss_gap) == StreamStatus::ok);
    CHECK(loss_source.read_message(loss_frame.frame(), loss_gap) == StreamStatus::ok);
    CHECK(loss_frame.view().blocks.front().sample_idx_start == 100);
    CHECK(loss_source.read_message(loss_frame.frame(), loss_gap) == StreamStatus::discontinuity);
    auto gap = loss_gap.view();
    CHECK(gap.reason == GapReason::sample_gap);
    CHECK(gap.previous_frame_sequence == 0);
    CHECK(gap.actual_frame_sequence == 1);
    CHECK(gap.signal_gaps.size() == 1);
    CHECK(gap.signal_gaps[0].expected_sample_idx == 104);
    CHECK(gap.signal_gaps[0].actual_sample_idx == 106);
    CHECK(gap.signal_gaps[0].missing_samples == 2);
    CHECK(gap.signal_gaps[0].expected_device_tick == 5'004);
    CHECK(gap.signal_gaps[0].actual_device_tick == 5'006);
    CHECK(loss_gap.reset() == StreamStatus::ok);
    CHECK(loss_discontinuities.try_acquire(loss_gap) == StreamStatus::ok);
    CHECK(loss_source.read_message(loss_frame.frame(), loss_gap) == StreamStatus::ok);
    CHECK(loss_frame.view().header.sequence == 1);
    CHECK(loss_frame.view().blocks.front().sample_idx_start == 106);
    CHECK(loss_frame.view().blocks.front().device_tick_start == 5'006);
    std::array<double, 8> expected_after_loss{};
    CHECK(tones.generate(106, 4, expected_after_loss) == ss::GenerationStatus::ok);
    const auto* after_loss = reinterpret_cast<const double*>(loss_frame.view().payload.data());
    CHECK(std::equal(expected_after_loss.begin(), expected_after_loss.end(), after_loss));
    CHECK(loss_source.read_message(loss_frame.frame(), loss_gap) == StreamStatus::discontinuity);
    CHECK(loss_gap.view().signal_gaps[0].expected_sample_idx == 110);
    CHECK(loss_gap.view().signal_gaps[0].actual_sample_idx == 111);
    CHECK(loss_gap.reset() == StreamStatus::ok);

    // The generic runtime sees exactly the explicit gaps; continuity inference
    // does not duplicate them, and both processor and terminal receive them.
    auto runtime_loss_source_config = loss_config;
    runtime_loss_source_config.paced = true;
    ds::SimulatedNeuralSource runtime_loss_source{tones, runtime_loss_source_config};
    Passthrough loss_processor;
    TimestampConsumer loss_consumer;
    auto loss_runtime_config =
        runtime_config(runtime_loss_source.schema().signals().front().max_block_bytes);
    loss_runtime_config.discontinuity_capacity = 8;
    NativeStreamRunner loss_runner{runtime_loss_source.schema().clone(), loss_runtime_config,
                                   runtime_loss_source, loss_processor, loss_consumer};
    CHECK(loss_runner.prepare() == StreamStatus::ok);
    CHECK(loss_runner.arm() == StreamStatus::ok);
    CHECK(loss_runner.run() == StreamStatus::ok);
    CHECK(loss_processor.discontinuities == 2);
    CHECK(loss_consumer.discontinuities == 2);
    CHECK(loss_consumer.count == 4);
    CHECK(loss_runner.outstanding_frames() == 0);
    CHECK(loss_runner.outstanding_discontinuities() == 0);

    // would_block is one-shot and does not consume frame/sample position.
    auto transient_config = config(4, 4, 8);
    transient_config.events = {
        {.frame_ordinal = 0, .kind = ds::AcquisitionEventKind::would_block},
        {.frame_ordinal = 1, .kind = ds::AcquisitionEventKind::would_block},
    };
    ds::SimulatedNeuralSource transient_source{generator, transient_config};
    CHECK(transient_source.read(loss_frame.frame()) == StreamStatus::would_block);
    CHECK(transient_source.frames_emitted() == 0);
    CHECK(transient_source.read(loss_frame.frame()) == StreamStatus::ok);
    CHECK(transient_source.read(loss_frame.frame()) == StreamStatus::would_block);
    CHECK(transient_source.read(loss_frame.frame()) == StreamStatus::ok);

    AdvancingClock transient_epoch_clock;
    auto transient_epoch_config = config(4, 4, 4);
    transient_epoch_config.paced = true;
    transient_epoch_config.events = {
        {.frame_ordinal = 0, .kind = ds::AcquisitionEventKind::would_block},
    };
    ds::SimulatedNeuralSource transient_epoch_source{generator, transient_epoch_config,
                                                     transient_epoch_clock};
    CHECK(transient_epoch_source.read(loss_frame.frame()) == StreamStatus::would_block);
    transient_epoch_clock.set_now(4'000'000'000);
    CHECK(transient_epoch_source.read(loss_frame.frame()) == StreamStatus::ok);
    CHECK(loss_frame.view().blocks.front().clock_sync.host_time_reference_ns == 4'000'000'000);

    // An unpaced acquisition still honors an explicitly scheduled bounded
    // stall. The fake clock makes the exact delay deterministic and cheap.
    AdvancingClock event_clock;
    auto stall_config = config(4, 4, 4);
    stall_config.events = {
        {.frame_ordinal = 0, .kind = ds::AcquisitionEventKind::stall, .duration_ns = 750},
    };
    ds::SimulatedNeuralSource stall_source{generator, stall_config, event_clock};
    CHECK(stall_source.read(loss_frame.frame()) == StreamStatus::ok);
    CHECK(event_clock.waited_ns() == 750);
    CHECK(event_clock.last_deadline() == 1'750);
    CHECK(loss_frame.view().blocks.front().clock_sync.host_time_reference_ns == 1'000);

    BlockingClock stall_block_clock;
    auto cancellable_stall_config = config(4, 4, 4);
    cancellable_stall_config.events = {
        {.frame_ordinal = 0, .kind = ds::AcquisitionEventKind::stall, .duration_ns = 1'000'000'000},
    };
    ds::SimulatedNeuralSource cancellable_stall_source{generator, cancellable_stall_config,
                                                       stall_block_clock};
    std::atomic<StreamStatus> stalled_status{StreamStatus::ok};
    std::thread stalled_reader(
        [&] { stalled_status.store(cancellable_stall_source.read(loss_frame.frame())); });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // The reader is parked inside read() on a clock that never advances, so it
    // holds the operation gate. A reset here would rewrite the sample offset,
    // device tick, epoch, and schedule cursor that read_impl is using; it is
    // refused rather than raced. cancel() is deliberately still available --
    // it is what ends the read the reset is waiting on.
    CHECK(cancellable_stall_source.reset() == StreamStatus::invalid_state);
    const auto stall_cancel_started = std::chrono::steady_clock::now();
    cancellable_stall_source.cancel();
    stalled_reader.join();
    CHECK(std::chrono::steady_clock::now() - stall_cancel_started < std::chrono::milliseconds(500));
    CHECK(stalled_status.load() == StreamStatus::stopped);
    cancellable_stall_source.cancel();
    CHECK(cancellable_stall_source.reset() == StreamStatus::ok);
    CHECK(!cancellable_stall_source.closed());

    stalled_status.store(StreamStatus::ok);
    std::thread closing_reader(
        [&] { stalled_status.store(cancellable_stall_source.read(loss_frame.frame())); });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const auto stall_close_started = std::chrono::steady_clock::now();
    cancellable_stall_source.close();
    cancellable_stall_source.close();
    closing_reader.join();
    CHECK(std::chrono::steady_clock::now() - stall_close_started < std::chrono::milliseconds(500));
    CHECK(stalled_status.load() == StreamStatus::stopped);
    CHECK(cancellable_stall_source.closed());
    CHECK(cancellable_stall_source.reset() == StreamStatus::invalid_state);
    CHECK(cancellable_stall_source.read(loss_frame.frame()) == StreamStatus::stopped);

    // Offset changes only the tick-to-host mapping. Positive drift means a
    // faster device/sample clock and therefore a shorter absolute deadline.
    AdvancingClock drift_clock;
    drift_clock.set_now(1'000'000'000);
    auto drift_config = config(4, 4, 8);
    drift_config.paced = true;
    drift_config.clock_offset_ns = 500;
    drift_config.clock_drift_ppm = 1'000;
    ds::SimulatedNeuralSource drift_source{generator, drift_config, drift_clock};
    CHECK(drift_source.read(loss_frame.frame()) == StreamStatus::ok);
    CHECK(loss_frame.view().blocks.front().clock_sync.host_time_reference_ns == 1'000'000'500);
    CHECK(loss_frame.view().blocks.front().clock_sync.device_tick_rate.numerator == 1'001);
    CHECK(loss_frame.view().blocks.front().clock_sync.device_tick_rate.denominator == 1);
    CHECK(drift_source.read(loss_frame.frame()) == StreamStatus::ok);
    CHECK(drift_clock.last_deadline() == 1'003'996'004);

    auto manual_host_clock = std::make_shared<ds::ManualHostClock>(10'000);
    auto manual_config = config(1, 1, 2);
    manual_config.fs = {1'000, 1};
    manual_config.initial_sample_idx = 0;
    manual_config.paced = true;
    const auto manual_generator = ss::SignalGenerator::zeros(2, 1'000.0);
    ds::SimulatedNeuralSource manual_source{manual_generator, manual_config, manual_host_clock};
    CHECK(manual_source.read(loss_frame.frame()) == StreamStatus::ok);
    std::atomic<StreamStatus> manual_status{StreamStatus::would_block};
    std::thread manual_reader([&] { manual_status.store(manual_source.read(loss_frame.frame())); });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(manual_status.load() == StreamStatus::would_block);
    manual_host_clock->advance(1'000'000);
    manual_reader.join();
    CHECK(manual_status.load() == StreamStatus::ok);
    CHECK(manual_host_clock->now_ns() == 1'010'000);

    // Tick jumps and restarts are ordered explicit barriers. A restart starts
    // a fresh sync generation and never reuses the pre-restart tick mapping.
    auto tick_config = config(4, 4, 8);
    tick_config.events = {
        {.frame_ordinal = 1,
         .kind = ds::AcquisitionEventKind::device_tick_jump,
         .device_tick_delta = 5},
        {.frame_ordinal = 1, .kind = ds::AcquisitionEventKind::device_restart, .restart_tick = 200},
    };
    ds::SimulatedNeuralSource tick_source{generator, tick_config};
    DiscontinuityPool tick_discontinuities{1, 1};
    DiscontinuityLease tick_gap;
    CHECK(tick_discontinuities.try_acquire(tick_gap) == StreamStatus::ok);
    CHECK(tick_source.read_message(loss_frame.frame(), tick_gap) == StreamStatus::ok);
    CHECK(loss_frame.view().blocks.front().clock_sync.generation == 1);
    CHECK(tick_source.read_message(loss_frame.frame(), tick_gap) == StreamStatus::discontinuity);
    CHECK(tick_gap.view().reason == GapReason::device_tick_gap);
    CHECK(tick_gap.view().signal_gaps[0].expected_device_tick == 5'004);
    CHECK(tick_gap.view().signal_gaps[0].actual_device_tick == 5'009);
    CHECK(tick_gap.reset() == StreamStatus::ok);
    CHECK(tick_discontinuities.try_acquire(tick_gap) == StreamStatus::ok);
    CHECK(tick_source.read_message(loss_frame.frame(), tick_gap) == StreamStatus::discontinuity);
    CHECK(tick_gap.view().reason == GapReason::device_restart);
    CHECK(tick_gap.view().signal_gaps[0].expected_device_tick == 5'009);
    CHECK(tick_gap.view().signal_gaps[0].actual_device_tick == 200);
    CHECK(tick_gap.reset() == StreamStatus::ok);
    CHECK(tick_source.read_message(loss_frame.frame(), tick_gap) == StreamStatus::ok);
    CHECK(loss_frame.view().blocks.front().device_tick_start == 200);
    CHECK(loss_frame.view().blocks.front().clock_sync.generation == 3);

    // Terminal device faults reuse the stable public source-failure status.
    for (const auto kind :
         {ds::AcquisitionEventKind::disconnect, ds::AcquisitionEventKind::source_fault})
    {
        auto terminal_config = config(4, 4, 8);
        terminal_config.events = {{.frame_ordinal = 1, .kind = kind}};
        ds::SimulatedNeuralSource terminal_source{generator, terminal_config};
        CHECK(terminal_source.read(loss_frame.frame()) == StreamStatus::ok);
        CHECK(terminal_source.read(loss_frame.frame()) == StreamStatus::source_failure);
        CHECK(terminal_source.frames_emitted() == 1);
    }

    auto runtime_fault_config = config(4, 4, 8);
    runtime_fault_config.paced = true;
    runtime_fault_config.events = {
        {.frame_ordinal = 1, .kind = ds::AcquisitionEventKind::source_fault},
    };
    ds::SimulatedNeuralSource runtime_fault_source{generator, runtime_fault_config};
    Passthrough runtime_fault_processor;
    TimestampConsumer runtime_fault_consumer;
    NativeStreamRunner fault_runner{
        runtime_fault_source.schema().clone(),
        runtime_config(runtime_fault_source.schema().signals().front().max_block_bytes),
        runtime_fault_source, runtime_fault_processor, runtime_fault_consumer};
    CHECK(fault_runner.prepare() == StreamStatus::ok);
    CHECK(fault_runner.arm() == StreamStatus::ok);
    CHECK(fault_runner.run() == StreamStatus::source_failure);
    CHECK(fault_runner.primary_fault().has_value());
    CHECK(fault_runner.primary_fault()->code == FaultCode::source_read);
    CHECK(fault_runner.primary_fault()->status == StreamStatus::source_failure);
    CHECK(fault_runner.primary_fault()->session_id == 4);
    CHECK(fault_runner.primary_fault()->frame_sequence == 0);
    CHECK(fault_runner.outstanding_frames() == 0);
    CHECK(fault_runner.outstanding_discontinuities() == 0);

    // The full schedule is restored by reset and repeats byte-for-byte.
    CHECK(transient_source.reset() == StreamStatus::ok);
    CHECK(transient_source.read(loss_frame.frame()) == StreamStatus::would_block);
    CHECK(transient_source.read(loss_frame.frame()) == StreamStatus::ok);
    CHECK(loss_frame.view().header.sequence == 0);
    CHECK(loss_frame.view().blocks.front().sample_idx_start == 100);

    // Scheduled state transitions, including a gap publication, allocate
    // nothing after construction and pool preparation.
    auto allocation_event_config = config(4, 4, 8);
    allocation_event_config.events = {
        {.frame_ordinal = 0, .kind = ds::AcquisitionEventKind::sample_loss, .n_samples = 1},
    };
    ds::SimulatedNeuralSource allocation_event_source{generator, allocation_event_config};
    DiscontinuityPool allocation_discontinuities{1, 1};
    DiscontinuityLease allocation_gap;
    CHECK(allocation_discontinuities.try_acquire(allocation_gap) == StreamStatus::ok);
    const auto before_event = allocations.load(std::memory_order_acquire);
    CHECK(allocation_event_source.read_message(loss_frame.frame(), allocation_gap) ==
          StreamStatus::discontinuity);
    CHECK(allocations.load(std::memory_order_acquire) == before_event);
    CHECK(allocation_gap.reset() == StreamStatus::ok);

    // An int16 device ships raw counts: values round to nearest, saturate at the
    // 16-bit bounds, and the schema reports half the float64 payload size.
    const std::array counts = {1.4, -2.6, 40'000.0, -40'000.0};
    const auto count_generator = ss::SignalGenerator::constant(4, 1'000.0, counts);
    auto int16_config = config(4, 6, 8);
    int16_config.sample_dtype = SignalDType::int16;
    int16_config.physical_unit = PhysicalUnit::dimensionless;
    ds::SimulatedNeuralSource int16_source{count_generator, int16_config};
    const auto& int16_signal = int16_source.schema().signals().front();
    CHECK(int16_signal.dtype == SignalDType::int16);
    CHECK(int16_signal.n_channels == 4);
    CHECK(int16_signal.max_block_bytes == 4 * 6 * sizeof(std::int16_t));
    CHECK(int16_signal.physical_unit == PhysicalUnit::dimensionless);
    FramePool int16_pool{1, int16_signal.max_block_bytes, 1};
    FrameLease int16_lease;
    CHECK(int16_pool.try_acquire(int16_lease) == StreamStatus::ok);
    const auto before_int16 = allocations.load(std::memory_order_acquire);
    CHECK(int16_source.read(int16_lease.frame()) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_acquire) == before_int16);
    const auto int16_view = int16_lease.view();
    CHECK(int16_view.blocks.front().payload_byte_count == 4 * 4 * sizeof(std::int16_t));
    const auto* int16_payload = reinterpret_cast<const std::int16_t*>(int16_view.payload.data());
    for (std::size_t sample = 0; sample < 4; ++sample)
    {
        CHECK(int16_payload[sample * 4 + 0] == 1);
        CHECK(int16_payload[sample * 4 + 1] == -3);
        CHECK(int16_payload[sample * 4 + 2] == std::numeric_limits<std::int16_t>::max());
        CHECK(int16_payload[sample * 4 + 3] == std::numeric_limits<std::int16_t>::min());
    }
    CHECK(int16_lease.reset() == StreamStatus::ok);
    CHECK(int16_pool.outstanding() == 0);

    // A dtype the simulator does not produce is rejected, not approximated.
    auto float32_config = config(4, 6, 8);
    float32_config.sample_dtype = SignalDType::float32;
    bool float32_rejected = false;
    try
    {
        ds::SimulatedNeuralSource rejected{count_generator, float32_config};
        static_cast<void>(rejected.schema().id());
    }
    catch (const std::invalid_argument&)
    {
        float32_rejected = true;
    }
    CHECK(float32_rejected);

    // A real monotonic clock follows an absolute 100 Hz acquisition timeline.
    const auto realtime_generator = ss::SignalGenerator::zeros(1, 100.0);
    auto realtime_config = config(1, 1, 3);
    realtime_config.fs = {100, 1};
    realtime_config.initial_sample_idx = 0;
    realtime_config.paced = true;
    ds::SimulatedNeuralSource realtime_source{realtime_generator, realtime_config};
    FramePool realtime_pool{1, realtime_source.schema().signals().front().max_block_bytes, 1};
    FrameLease realtime_frame;
    CHECK(realtime_pool.try_acquire(realtime_frame) == StreamStatus::ok);
    const auto realtime_started = std::chrono::steady_clock::now();
    CHECK(realtime_source.read(realtime_frame.frame()) == StreamStatus::ok);
    CHECK(realtime_source.read(realtime_frame.frame()) == StreamStatus::ok);
    CHECK(realtime_source.read(realtime_frame.frame()) == StreamStatus::ok);
    const auto realtime_elapsed = std::chrono::steady_clock::now() - realtime_started;
    CHECK(realtime_elapsed >= std::chrono::milliseconds(15));
    CHECK(realtime_elapsed < std::chrono::seconds(1));

    CHECK(loss_frame.reset() == StreamStatus::ok);
    CHECK(realtime_frame.reset() == StreamStatus::ok);
    CHECK(loss_pool.outstanding() == 0);
    CHECK(loss_discontinuities.outstanding() == 0);
    CHECK(tick_discontinuities.outstanding() == 0);
    CHECK(allocation_discontinuities.outstanding() == 0);
    CHECK(realtime_pool.outstanding() == 0);

    // The feedback simulator is opt-in, reads native intent without a Python
    // callback, and records the exact control applied to each emitted frame.
    auto neural_config = ss::NeuralSignalConfig{};
    neural_config.n_channels = 2;
    neural_config.fs = 1'000.0;
    neural_config.seed = 31;
    neural_config.drift_rotation_degrees = 0.0;
    neural_config.drift_per_unit_rotation_std_degrees = 30.0;
    neural_config.drift_per_unit_rotation_limit_degrees = 60.0;
    auto intent_config = config(4, 4, 8);
    intent_config.initial_sample_idx = 0;
    ds::IntentDrivenNeuralSource intent_source{neural_config, intent_config,
                                               ds::NeuralDriftSchedule{0, 2, 0.0, 1.0}, 4};
    FramePool intent_pool{1, intent_source.schema().signals().front().max_block_bytes, 1};
    FrameLease intent_lease;
    CHECK(intent_pool.try_acquire(intent_lease) == StreamStatus::ok);
    CHECK(intent_source.publish_control(0.25, -0.5, 0.3, 7) == StreamStatus::ok);
    CHECK(intent_source.read(intent_lease.frame()) == StreamStatus::ok);
    ds::AppliedNeuralControl applied{};
    CHECK(intent_source.try_pop_applied_control(applied) == StreamStatus::ok);
    CHECK(applied.intent_sequence == 1);
    CHECK(applied.intent_x == 0.25);
    CHECK(applied.intent_y == -0.5);
    CHECK(applied.context_ordinal == 7);
    CHECK(applied.intent_valid);
    CHECK(applied.drift_progress == 0.3);
    CHECK(applied.first_sample_index == 0);
    CHECK(applied.last_sample_index == 3);
    CHECK(applied.generated_frame_sequence == 0);
    CHECK(intent_source.dropped_applied_control_count() == 0);
    CHECK(intent_source.reset() == StreamStatus::ok);

    auto bound_state = std::make_shared<NeuralIntentState>();
    CHECK(intent_source.bind_intent_source(bound_state) == StreamStatus::ok);
    bound_state->publish({9, -0.2, 0.4, 1, 50, 6, 30, true});
    CHECK(intent_source.read(intent_lease.frame()) == StreamStatus::ok);
    CHECK(intent_source.try_pop_applied_control(applied) == StreamStatus::ok);
    CHECK(applied.intent_sequence == 9);
    CHECK(applied.intent_x == -0.2);
    CHECK(applied.intent_y == 0.4);
    CHECK(applied.context_ordinal == 1);
    CHECK(applied.intent_valid);
    CHECK(applied.drift_progress == 0.5);
    CHECK(intent_source.publish_control(0.0, 0.0, 0.0, 0) == StreamStatus::invalid_state);
    CHECK(intent_lease.reset() == StreamStatus::ok);
    CHECK(intent_pool.outstanding() == 0);

    // Binding and first generation compete for one atomic mode transition.
    // If binding wins, the first frame must observe the fully published source;
    // otherwise generation freezes manual mode and binding is refused.
    for (std::size_t iteration = 0; iteration < 64; ++iteration)
    {
        auto race_neural_config = ss::NeuralSignalConfig{};
        race_neural_config.n_channels = 1;
        race_neural_config.fs = 1'000.0;
        race_neural_config.seed = iteration + 1;
        auto race_source_config = config(1, 1, 1);
        race_source_config.initial_sample_idx = 0;
        ds::IntentDrivenNeuralSource race_source{race_neural_config, race_source_config,
                                                 std::nullopt, 2};
        auto race_state = std::make_shared<NeuralIntentState>();
        race_state->publish({55, 0.2, -0.3, 4, 10, 2, 0, true});
        FramePool race_pool{1, race_source.schema().signals().front().max_block_bytes, 1};
        FrameLease race_lease;
        CHECK(race_pool.try_acquire(race_lease) == StreamStatus::ok);

        std::atomic<std::uint32_t> ready{};
        std::atomic<bool> go{};
        auto bind_status = StreamStatus::source_failure;
        auto read_status = StreamStatus::source_failure;
        std::thread binder(
            [&]
            {
                ready.fetch_add(1, std::memory_order_release);
                while (!go.load(std::memory_order_acquire))
                {
                }
                bind_status = race_source.bind_intent_source(race_state);
            });
        std::thread reader(
            [&]
            {
                ready.fetch_add(1, std::memory_order_release);
                while (!go.load(std::memory_order_acquire))
                {
                }
                read_status = race_source.read(race_lease.frame());
            });
        while (ready.load(std::memory_order_acquire) != 2)
        {
        }
        go.store(true, std::memory_order_release);
        binder.join();
        reader.join();

        CHECK(read_status == StreamStatus::ok);
        CHECK(bind_status == StreamStatus::ok || bind_status == StreamStatus::invalid_state);
        ds::AppliedNeuralControl race_applied{};
        CHECK(race_source.try_pop_applied_control(race_applied) == StreamStatus::ok);
        CHECK((bind_status == StreamStatus::ok && race_applied.intent_sequence == 55) ||
              (bind_status == StreamStatus::invalid_state && race_applied.intent_sequence == 0));
        CHECK(race_lease.reset() == StreamStatus::ok);
        CHECK(race_pool.outstanding() == 0);
    }

    CHECK(lease.reset() == StreamStatus::ok);
    CHECK(lease3.reset() == StreamStatus::ok);
    CHECK(lease5.reset() == StreamStatus::ok);
    CHECK(epoch_lease.reset() == StreamStatus::ok);
    CHECK(small_lease.reset() == StreamStatus::ok);
    CHECK(unticked_lease.reset() == StreamStatus::ok);
    CHECK(allocation_lease.reset() == StreamStatus::ok);
    CHECK(first.reset() == StreamStatus::ok);
    CHECK(blocked.reset() == StreamStatus::ok);
    CHECK(pool.outstanding() == 0);
    CHECK(pool3.outstanding() == 0);
    CHECK(pool5.outstanding() == 0);
    CHECK(epoch_pool.outstanding() == 0);
    CHECK(too_small.outstanding() == 0);
    CHECK(unticked_pool.outstanding() == 0);
    CHECK(allocation_pool.outstanding() == 0);
    CHECK(paced_pool.outstanding() == 0);

    return 0;
}

} // namespace

int main()
{
    try
    {
        return run();
    }
    catch (const std::exception& exception)
    {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
