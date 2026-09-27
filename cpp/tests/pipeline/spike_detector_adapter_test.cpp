/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "allocation_counter.h"
#include "check_returns.h"
#include "spike_detector_adapter.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

#include <neurale/sorting/online_detection.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame_validation.h>

namespace
{

using namespace neurale::streaming;

StreamSchema input_schema()
{
    const std::array signals{SignalSchema{11,
                                          SignalDType::float64,
                                          1,
                                          8,
                                          16,
                                          {1'000, 1},
                                          41,
                                          SignalLayout::sample_major,
                                          DeviceTickTracking::sample_counter,
                                          PhysicalUnit::volts,
                                          17,
                                          19,
                                          23}};
    return StreamSchema{29, signals};
}

neurale::pipeline::SpikeDetectorAdapterConfig adapter_config(
    std::size_t capacity = 4,
    neurale::sorting::SpikeBlockOverflowPolicy policy =
        neurale::sorting::SpikeBlockOverflowPolicy::fault,
    neurale::sorting::BoundaryBehavior boundary = neurale::sorting::BoundaryBehavior::Drop)
{
    return {
        .output_schema_id = 37,
        .output_signal_id = 43,
        .block_capacity = capacity,
        .refractory_samples = 0,
        .alignment_search_radius = 0,
        .pre_samples = 1,
        .post_samples = 1,
        .polarity = neurale::sorting::DetectionPolarity::Negative,
        .boundary_behavior = boundary,
        .overflow_policy = policy,
        .channel_centers = {0.0},
        .channel_thresholds = {1.0},
    };
}

StreamStatus fill(MutableFrame& frame, const StreamSchema& schema, std::span<const double> values,
                  std::uint64_t sequence, SampleIndex first,
                  std::uint64_t host_reference_ns = 10'000'000, std::uint32_t generation = 1)
{
    frame.header() = FrameHeader{.session_id = 5,
                                 .sequence = sequence,
                                 .host_received_ns = 20'000'000,
                                 .source_tick = 5'000 + first,
                                 .valid_until_ns = 30'000'000,
                                 .schema_id = schema.id(),
                                 .source_clock_domain = 41,
                                 .flags = FrameFlags::source_tick | FrameFlags::valid_until};
    frame.block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = first,
        .device_tick_start = 5'000 + first,
        .payload_offset = 0,
        .payload_byte_count = values.size_bytes(),
        .signal_id = 11,
        .n_samples = static_cast<std::uint32_t>(values.size()),
        .clock_sync = ClockSyncSnapshot{.device_tick_reference = 5'000,
                                        .host_time_reference_ns = host_reference_ns,
                                        .device_tick_rate = {1'000, 1},
                                        .uncertainty_ns = 1,
                                        .clock_domain = 41,
                                        .generation = generation,
                                        .flags = ClockSyncFlags::synchronized}};
    std::memcpy(frame.payload_storage().data(), values.data(), values.size_bytes());
    return frame.set_used_sizes(1, values.size_bytes());
}

class Sink final : public FrameEmitter
{
  public:
    explicit Sink(const StreamSchema& schema)
        : pool_(1, schema.signals().front().max_block_bytes, 1), validator_(schema),
          payload_(schema.signals().front().max_block_bytes)
    {
    }

    void begin(MutableFrame& input) noexcept
    {
        input_ = &input;
        count_ = 0;
        static_cast<void>(lease_.reset());
    }
    StreamStatus try_acquire_frame(MutableFrame*& frame) noexcept override
    {
        const auto status = pool_.try_acquire(lease_);
        frame = status == StreamStatus::ok ? &lease_.frame() : nullptr;
        return status;
    }
    StreamStatus publish_acquired_frame() noexcept override
    {
        const auto view = lease_.view();
        if (validator_.validate(view) != FrameValidationError::none)
            return StreamStatus::invalid_frame;
        header_ = view.header;
        block_ = view.blocks.front();
        std::memcpy(payload_.data(), view.payload.data(), view.payload.size());
        ++count_;
        return lease_.reset();
    }
    StreamStatus publish_input() noexcept override
    {
        return input_ ? StreamStatus::ok : StreamStatus::invalid_state;
    }
    [[nodiscard]] std::size_t count() const noexcept
    {
        return count_;
    }
    [[nodiscard]] const SignalBlockHeader& block() const noexcept
    {
        return block_;
    }
    [[nodiscard]] const neurale::sorting::SpikeBlockHeader& spike_header() const noexcept
    {
        return *reinterpret_cast<const neurale::sorting::SpikeBlockHeader*>(payload_.data());
    }
    [[nodiscard]] std::span<const std::byte> spike_payload() const noexcept
    {
        return {payload_.data(), block_.payload_byte_count};
    }

  private:
    StreamStatus publish_owned(FrameLease lease) noexcept override
    {
        return lease.reset();
    }
    FramePool pool_;
    FrameValidator validator_;
    std::vector<std::byte> payload_;
    MutableFrame* input_{};
    FrameLease lease_{};
    FrameHeader header_{};
    SignalBlockHeader block_{};
    std::size_t count_{};
};

int run()
{
    auto schema = input_schema();
    auto cfg = adapter_config();
    neurale::pipeline::SpikeDetectorAdapter adapter{cfg};
    auto contract = adapter.prepare({schema, 1, 1, 1});
    CHECK(contract.max_process_outputs_per_input == 1);
    CHECK(contract.max_flush_outputs == 1);
    CHECK(contract.required_resources.frame_pool_leases == 1);
    CHECK(contract.output_schema.signals().front().kind == SignalKind::spike);
    CHECK(contract.output_schema.signals().front().max_block_samples == 4);
    // The contract must declare the detector's full fixed workspace, not the
    // adapter's old buffer+waveform estimate. Rebuild the equivalent detector
    // config from the same inputs and confirm the declared bytes match exactly.
    {
        const auto& input = schema.signals().front();
        neurale::sorting::OnlineThresholdDetectorConfig equiv{
            .max_input_samples = input.max_block_samples,
            .block_capacity = cfg.block_capacity,
            .refractory_samples = cfg.refractory_samples,
            .alignment_search_radius = cfg.alignment_search_radius,
            .pre_samples = cfg.pre_samples,
            .post_samples = cfg.post_samples,
            .polarity = cfg.polarity,
            .boundary_behavior = cfg.boundary_behavior,
            .overflow_policy = cfg.overflow_policy,
            .channel_centers = cfg.channel_centers,
            .channel_thresholds = cfg.channel_thresholds,
            .electrode_groups = cfg.electrode_groups,
        };
        neurale::sorting::OnlineThresholdDetector detector{equiv};
        CHECK(contract.required_resources.workspace_bytes == detector.workspace_bytes());
    }

    FramePool input_pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease input{};
    CHECK(input_pool.try_acquire(input) == StreamStatus::ok);
    Sink sink{contract.output_schema};
    const std::array burst{0.0, -2.0, 0.0, -3.0, 0.0, -4.0, 0.0};
    CHECK(fill(input.frame(), schema, burst, 0, 0) == StreamStatus::ok);
    sink.begin(input.frame());
    CHECK(adapter.process(input.frame(), sink) == StreamStatus::ok);
    CHECK(sink.count() == 1);
    CHECK(sink.block().n_samples == 3);
    CHECK(sink.spike_header().n_valid == 3);
    CHECK(sink.block().sample_idx_start == 1);
    CHECK(sink.block().last_sample_idx == 5);

    CHECK(adapter.reset() == StreamStatus::ok);
    const std::array boundary{0.0, -5.0};
    CHECK(fill(input.frame(), schema, boundary, 1, 10) == StreamStatus::ok);
    sink.begin(input.frame());
    CHECK(adapter.process(input.frame(), sink) == StreamStatus::ok);
    CHECK(sink.spike_header().n_valid == 0);
    CHECK(sink.block().last_sample_idx == 0);
    const std::array tail{0.0, 0.0};
    CHECK(fill(input.frame(), schema, tail, 2, 12) == StreamStatus::ok);
    sink.begin(input.frame());
    CHECK(adapter.process(input.frame(), sink) == StreamStatus::ok);
    CHECK(sink.spike_header().n_valid == 1);

    CHECK(adapter.handle_discontinuity({}) == StreamStatus::ok);
    CHECK(fill(input.frame(), schema, tail, 3, 20) == StreamStatus::ok);
    sink.begin(input.frame());
    CHECK(adapter.process(input.frame(), sink) == StreamStatus::ok);
    CHECK(sink.spike_header().n_valid == 0);
    CHECK(sink.spike_header().segment_id == 1);

    CHECK(adapter.reset() == StreamStatus::ok);
    CHECK(fill(input.frame(), schema, burst, 4, 0) == StreamStatus::ok);
    sink.begin(input.frame());
    CHECK(adapter.process(input.frame(), sink) == StreamStatus::ok);
    CHECK(adapter.reset() == StreamStatus::ok);
    const auto before = allocations.load(std::memory_order_relaxed);
    for (std::size_t repetition = 0; repetition < 100; ++repetition)
    {
        CHECK(fill(input.frame(), schema, burst, repetition + 5, 0) == StreamStatus::ok);
        sink.begin(input.frame());
        CHECK(adapter.process(input.frame(), sink) == StreamStatus::ok);
        CHECK(adapter.reset() == StreamStatus::ok);
    }
    CHECK(allocations.load(std::memory_order_relaxed) == before);

    neurale::pipeline::SpikeDetectorAdapter overflow{
        adapter_config(2, neurale::sorting::SpikeBlockOverflowPolicy::fault)};
    auto overflow_contract = overflow.prepare({schema, 1, 1, 1});
    Sink overflow_sink{overflow_contract.output_schema};
    CHECK(fill(input.frame(), schema, burst, 200, 0) == StreamStatus::ok);
    overflow_sink.begin(input.frame());
    CHECK(overflow.process(input.frame(), overflow_sink) == StreamStatus::output_limit);
    CHECK(overflow_sink.count() == 0);

    neurale::pipeline::SpikeDetectorAdapter drop{
        adapter_config(2, neurale::sorting::SpikeBlockOverflowPolicy::drop_newest)};
    auto drop_contract = drop.prepare({schema, 1, 1, 1});
    Sink drop_sink{drop_contract.output_schema};
    CHECK(fill(input.frame(), schema, burst, 201, 0) == StreamStatus::ok);
    drop_sink.begin(input.frame());
    CHECK(drop.process(input.frame(), drop_sink) == StreamStatus::ok);
    CHECK(drop_sink.count() == 1);
    CHECK(drop_sink.spike_header().n_valid == 2);
    CHECK(drop_sink.spike_header().overflow_count == 1);
    CHECK(drop_sink.spike_header().flags == neurale::sorting::SpikeBlockFlags::overflowed);

    // Multi-channel, multi-group, large max_input_samples: the same equality
    // must hold, and the declared workspace must be far larger than the old
    // buffer+waveform underestimate so the realtime profile budgets the real
    // fixed reservation.
    {
        constexpr std::size_t channels = 32;
        constexpr std::size_t max_input = 1024;
        const std::array multi_signals{SignalSchema{12,
                                                    SignalDType::float64,
                                                    static_cast<std::uint32_t>(channels),
                                                    8,
                                                    static_cast<std::uint32_t>(max_input),
                                                    {1'000, 1},
                                                    42,
                                                    SignalLayout::sample_major,
                                                    DeviceTickTracking::sample_counter,
                                                    PhysicalUnit::volts,
                                                    18,
                                                    20,
                                                    24}};
        StreamSchema multi_schema{30, multi_signals};
        std::vector<double> centers(channels, 0.0);
        std::vector<double> thresholds(channels, 1.0);
        std::vector<std::vector<std::size_t>> groups(2);
        groups[0].reserve(channels / 2);
        groups[1].reserve(channels / 2);
        for (std::size_t channel = 0; channel < channels; ++channel)
        {
            groups[channel / (channels / 2)].push_back(channel);
        }
        neurale::pipeline::SpikeDetectorAdapterConfig multi_cfg{
            .output_schema_id = 38,
            .output_signal_id = 44,
            .block_capacity = 64,
            .refractory_samples = 4,
            .alignment_search_radius = 4,
            .pre_samples = 16,
            .post_samples = 31,
            .polarity = neurale::sorting::DetectionPolarity::Negative,
            .boundary_behavior = neurale::sorting::BoundaryBehavior::Drop,
            .overflow_policy = neurale::sorting::SpikeBlockOverflowPolicy::fault,
            .channel_centers = centers,
            .channel_thresholds = thresholds,
            .electrode_groups = groups,
        };
        neurale::pipeline::SpikeDetectorAdapter multi_adapter{multi_cfg};
        auto multi_contract = multi_adapter.prepare({multi_schema, 1, 1, 1});
        neurale::sorting::OnlineThresholdDetectorConfig equiv{
            .max_input_samples = max_input,
            .block_capacity = 64,
            .refractory_samples = 4,
            .alignment_search_radius = 4,
            .pre_samples = 16,
            .post_samples = 31,
            .polarity = neurale::sorting::DetectionPolarity::Negative,
            .boundary_behavior = neurale::sorting::BoundaryBehavior::Drop,
            .overflow_policy = neurale::sorting::SpikeBlockOverflowPolicy::fault,
            .channel_centers = centers,
            .channel_thresholds = thresholds,
            .electrode_groups = groups,
        };
        neurale::sorting::OnlineThresholdDetector detector{equiv};
        CHECK(multi_contract.required_resources.workspace_bytes == detector.workspace_bytes());
        // The old adapter estimate counted only the input buffer and waveform
        // scratch; the full workspace must strictly exceed that.
        const auto retention = 2 * multi_cfg.alignment_search_radius + multi_cfg.pre_samples +
                               multi_cfg.post_samples + 1;
        const auto old_estimate =
            ((retention + max_input) * channels +
             (multi_cfg.pre_samples + 1 + multi_cfg.post_samples) * channels) *
            sizeof(double);
        CHECK(multi_contract.required_resources.workspace_bytes > old_estimate);
    }

    // A spike pending in frame 1 completes in frame 2 under a new clock-sync
    // mapping (updated generation and host reference offset). The payload event
    // time must use frame 2's mapping, matching the output block's clock_sync,
    // not the frame-1 anchor the detector used to save.
    {
        neurale::pipeline::SpikeDetectorAdapter adapter{adapter_config()};
        auto contract = adapter.prepare({schema, 1, 1, 1});
        Sink sink{contract.output_schema};
        FramePool input_pool{1, schema.signals().front().max_block_bytes, 1};
        FrameLease input{};
        CHECK(input_pool.try_acquire(input) == StreamStatus::ok);

        // Frame 1: crossing at sample 11 stays pending (post=1 needs sample 12).
        const std::array f1{0.0, -5.0};
        CHECK(fill(input.frame(), schema, f1, 10, 10, 10'000'000, 1) == StreamStatus::ok);
        sink.begin(input.frame());
        CHECK(adapter.process(input.frame(), sink) == StreamStatus::ok);
        CHECK(sink.count() == 1);
        CHECK(sink.spike_header().n_valid == 0);

        // Frame 2: new clock-sync generation and host reference; sample 12
        // completes the pending spike whose peak is the retained sample 11.
        const std::array f2{0.0, 0.0};
        CHECK(fill(input.frame(), schema, f2, 11, 12, 100'000'000, 2) == StreamStatus::ok);
        sink.begin(input.frame());
        CHECK(adapter.process(input.frame(), sink) == StreamStatus::ok);
        CHECK(sink.spike_header().n_valid == 1);
        const auto snapshot = neurale::sorting::decode_spike_block(sink.spike_payload());
        CHECK(snapshot.sample_indices.size() == 1);
        CHECK(snapshot.sample_indices[0] == 11);

        // Map a device tick to host seconds, mirroring block_time_seconds in the
        // adapter, so the payload time can be checked against the output block's
        // own clock_sync rather than a hardcoded constant.
        const auto time_at = [](const ClockSyncSnapshot& sync, DeviceTick tick) -> double
        {
            const auto scale = 1e9 * static_cast<double>(sync.device_tick_rate.denominator) /
                               static_cast<double>(sync.device_tick_rate.numerator);
            const auto nanos =
                (tick >= sync.device_tick_reference)
                    ? static_cast<double>(sync.host_time_reference_ns) +
                          static_cast<double>(tick - sync.device_tick_reference) * scale
                    : static_cast<double>(sync.host_time_reference_ns) -
                          static_cast<double>(sync.device_tick_reference - tick) * scale;
            return nanos * 1e-9;
        };
        // The event sits at the retained peak (sample 11); under frame 2 its
        // device tick is the frame-2 input tick offset back to sample 11.
        const auto event_tick = sink.block().device_tick_start + (11 - 12);
        const auto expected = time_at(sink.block().clock_sync, event_tick);
        CHECK(std::abs(snapshot.times[0] - expected) < 1e-9);
        // The output block must carry frame 2's clock-sync mapping, and the
        // payload time is computed from that same mapping.
        CHECK(sink.block().clock_sync.generation == 2);
        CHECK(sink.block().clock_sync.host_time_reference_ns == 100'000'000);
        // The payload time must not match the stale frame-1 mapping (the old
        // first-frame anchor), which would place the event at ~0.021 s.
        ClockSyncSnapshot frame1_sync{};
        frame1_sync.device_tick_reference = 5'000;
        frame1_sync.host_time_reference_ns = 10'000'000;
        frame1_sync.device_tick_rate = {1'000, 1};
        frame1_sync.uncertainty_ns = 1;
        frame1_sync.clock_domain = 41;
        frame1_sync.generation = 1;
        frame1_sync.flags = ClockSyncFlags::synchronized;
        const auto stale = time_at(frame1_sync, 5'000 + 11);
        CHECK(std::abs(snapshot.times[0] - stale) > 1e-3);
    }

    // Cross-frame aligned peaks are globally ordered, not merely ordered by
    // the process call in which their waveform tail became available. The
    // completed-but-watermark-retained event is also drained by flush.
    {
        const std::array signals{SignalSchema{11,
                                              SignalDType::float64,
                                              2,
                                              16,
                                              64,
                                              {1'000, 1},
                                              41,
                                              SignalLayout::sample_major,
                                              DeviceTickTracking::sample_counter,
                                              PhysicalUnit::volts,
                                              17,
                                              19,
                                              23}};
        const StreamSchema two_channel_schema{29, signals};
        neurale::pipeline::SpikeDetectorAdapterConfig reorder_config{
            .output_schema_id = 37,
            .output_signal_id = 43,
            .block_capacity = 8,
            .refractory_samples = 5,
            .alignment_search_radius = 4,
            .pre_samples = 1,
            .post_samples = 3,
            .polarity = neurale::sorting::DetectionPolarity::Positive,
            .boundary_behavior = neurale::sorting::BoundaryBehavior::Drop,
            .overflow_policy = neurale::sorting::SpikeBlockOverflowPolicy::fault,
            .channel_centers = {0.0, 0.0},
            .channel_thresholds = {1.0, 1.0},
        };
        neurale::pipeline::SpikeDetectorAdapter reorder_adapter{reorder_config};
        const auto reorder_contract = reorder_adapter.prepare({two_channel_schema, 1, 1, 1});
        Sink reorder_sink{reorder_contract.output_schema};
        FramePool reorder_pool{1, 64 * 2 * sizeof(double), 1};
        FrameLease reorder_input{};
        CHECK(reorder_pool.try_acquire(reorder_input) == StreamStatus::ok);

        std::vector<double> first(50 * 2, 0.0);
        first[41 * 2] = 2.0;
        first[45 * 2] = 5.0;
        first[43 * 2 + 1] = 6.0;
        CHECK(fill(reorder_input.frame(), two_channel_schema, first, 0, 0) == StreamStatus::ok);
        reorder_input.frame().block_storage()[0].n_samples = 50;
        reorder_sink.begin(reorder_input.frame());
        CHECK(reorder_adapter.process(reorder_input.frame(), reorder_sink) == StreamStatus::ok);
        CHECK(reorder_sink.spike_header().n_valid == 0);

        const std::vector<double> second(6 * 2, 0.0);
        CHECK(fill(reorder_input.frame(), two_channel_schema, second, 1, 50) == StreamStatus::ok);
        reorder_input.frame().block_storage()[0].n_samples = 6;
        reorder_sink.begin(reorder_input.frame());
        CHECK(reorder_adapter.process(reorder_input.frame(), reorder_sink) == StreamStatus::ok);
        auto snapshot = neurale::sorting::decode_spike_block(reorder_sink.spike_payload());
        CHECK(snapshot.sample_indices == std::vector<std::int64_t>({43, 45}));
        CHECK(reorder_sink.block().sample_idx_start == 43);
        CHECK(reorder_sink.block().last_sample_idx == 45);

        CHECK(reorder_adapter.reset() == StreamStatus::ok);
        CHECK(fill(reorder_input.frame(), two_channel_schema, first, 10, 0) == StreamStatus::ok);
        reorder_input.frame().block_storage()[0].n_samples = 50;
        reorder_sink.begin(reorder_input.frame());
        CHECK(reorder_adapter.process(reorder_input.frame(), reorder_sink) == StreamStatus::ok);
        reorder_sink.begin(reorder_input.frame());
        CHECK(reorder_adapter.flush(reorder_sink) == StreamStatus::ok);
        snapshot = neurale::sorting::decode_spike_block(reorder_sink.spike_payload());
        CHECK(snapshot.sample_indices == std::vector<std::int64_t>({45}));
        CHECK(reorder_sink.block().sample_idx_start == 45);
        CHECK(reorder_sink.block().last_sample_idx == 45);
    }

    // End-of-stream flush vs discontinuity: a pending spike whose post-tail
    // never arrived cannot complete. Under ``Raise`` flush reports a tail
    // boundary error (processor_failure); under ``Drop`` it discards and
    // succeeds. A discontinuity always drops pending (per the streaming ADR)
    // and never reports a boundary error, even with ``Raise``.
    {
        const std::array boundary{0.0, -5.0}; // crossing at sample 11 -> pending (post=1 needs 12)

        neurale::pipeline::SpikeDetectorAdapter raise_adapter{
            adapter_config(4, neurale::sorting::SpikeBlockOverflowPolicy::fault,
                           neurale::sorting::BoundaryBehavior::Raise)};
        auto raise_contract = raise_adapter.prepare({schema, 1, 1, 1});
        Sink raise_sink{raise_contract.output_schema};
        FramePool raise_pool{1, schema.signals().front().max_block_bytes, 1};
        FrameLease raise_input{};
        CHECK(raise_pool.try_acquire(raise_input) == StreamStatus::ok);
        CHECK(fill(raise_input.frame(), schema, boundary, 0, 10) == StreamStatus::ok);
        raise_sink.begin(raise_input.frame());
        CHECK(raise_adapter.process(raise_input.frame(), raise_sink) == StreamStatus::ok);
        CHECK(raise_sink.spike_header().n_valid == 0); // crossing stays pending
        CHECK(raise_adapter.flush(raise_sink) == StreamStatus::processor_failure);

        neurale::pipeline::SpikeDetectorAdapter drop_adapter{
            adapter_config(4, neurale::sorting::SpikeBlockOverflowPolicy::fault,
                           neurale::sorting::BoundaryBehavior::Drop)};
        auto drop_contract = drop_adapter.prepare({schema, 1, 1, 1});
        Sink drop_sink{drop_contract.output_schema};
        FramePool drop_pool{1, schema.signals().front().max_block_bytes, 1};
        FrameLease drop_input{};
        CHECK(drop_pool.try_acquire(drop_input) == StreamStatus::ok);
        CHECK(fill(drop_input.frame(), schema, boundary, 0, 10) == StreamStatus::ok);
        drop_sink.begin(drop_input.frame());
        CHECK(drop_adapter.process(drop_input.frame(), drop_sink) == StreamStatus::ok);
        CHECK(drop_sink.spike_header().n_valid == 0);
        CHECK(drop_adapter.flush(drop_sink) == StreamStatus::ok);

        // Discontinuity drops pending without a boundary check, even under Raise.
        neurale::pipeline::SpikeDetectorAdapter disc_adapter{
            adapter_config(4, neurale::sorting::SpikeBlockOverflowPolicy::fault,
                           neurale::sorting::BoundaryBehavior::Raise)};
        auto disc_contract = disc_adapter.prepare({schema, 1, 1, 1});
        Sink disc_sink{disc_contract.output_schema};
        FramePool disc_pool{1, schema.signals().front().max_block_bytes, 1};
        FrameLease disc_input{};
        CHECK(disc_pool.try_acquire(disc_input) == StreamStatus::ok);
        CHECK(fill(disc_input.frame(), schema, boundary, 0, 10) == StreamStatus::ok);
        disc_sink.begin(disc_input.frame());
        CHECK(disc_adapter.process(disc_input.frame(), disc_sink) == StreamStatus::ok);
        CHECK(disc_sink.spike_header().n_valid == 0);
        CHECK(disc_adapter.handle_discontinuity({}) == StreamStatus::ok);
    }

    return 0;
}

} // namespace

int main()
{
    return run();
}
