/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "allocation_counter.h"
#include "check_returns.h"
#include "resampler_adapter.h"
#include "sos_filter_adapter.h"
#include "sos_realtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <span>

#include <neurale/signal/resample.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/linear_processor_chain.h>

namespace
{

using namespace neurale::streaming;

constexpr std::size_t channels = 2;
constexpr std::size_t input_max_samples = 8;
constexpr std::size_t max_values = 512;
constexpr std::size_t max_frames = 64;
constexpr std::array<double, 5> filter{0.05, 0.2, 0.5, 0.2, 0.05};
constexpr std::array<double, 6> sos{0.5, 0.25, 0.125, 1.0, -0.2, 0.05};

[[nodiscard]] StreamSchema make_schema()
{
    const std::array signals{
        SignalSchema{
            11,
            SignalDType::float64,
            channels,
            4,
            input_max_samples,
            {30'000, 1},
            41,
            SignalLayout::sample_major,
            DeviceTickTracking::sample_counter,
            PhysicalUnit::volts,
            17,
            19,
            23,
        },
    };
    return StreamSchema{29, signals};
}

[[nodiscard]] ProcessorPrepareContext context(const StreamSchema& schema) noexcept
{
    return {
        .input_schema = schema,
        .max_process_outputs = 16,
        .max_flush_outputs = 16,
        .available_frame_pool_leases = 2,
    };
}

[[nodiscard]] StreamStatus fill_frame(MutableFrame& frame, std::span<const double> values,
                                      std::uint64_t sequence, SampleIndex sample_start) noexcept
{
    if (values.empty() || values.size() % channels != 0)
    {
        return StreamStatus::invalid_frame;
    }
    const auto n_samples = values.size() / channels;
    frame.header() = FrameHeader{
        .session_id = 5,
        .sequence = sequence,
        .host_received_ns = 10'000 + sample_start,
        .source_tick = 20'000 + sample_start,
        .valid_until_ns = 30'000 + sample_start,
        .schema_id = 29,
        .source_clock_domain = 41,
        .flags = FrameFlags::source_tick | FrameFlags::valid_until | FrameFlags::source_received,
    };
    frame.block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = sample_start,
        .device_tick_start = 20'000 + sample_start,
        .payload_offset = 0,
        .payload_byte_count = values.size_bytes(),
        .signal_id = 11,
        .n_samples = static_cast<std::uint32_t>(n_samples),
        .clock_sync =
            ClockSyncSnapshot{
                .device_tick_reference = 20'000,
                .host_time_reference_ns = 40'000,
                .device_tick_rate = {30'000, 1},
                .uncertainty_ns = 3,
                .clock_domain = 41,
                .generation = 2,
                .flags = ClockSyncFlags::synchronized,
            },
    };
    std::memcpy(frame.payload_storage().data(), values.data(), values.size_bytes());
    return frame.set_used_sizes(1, values.size_bytes());
}

[[nodiscard]] bool nearly_equal(std::span<const double> left,
                                std::span<const double> right) noexcept
{
    if (left.size() != right.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        const auto scale = std::max({1.0, std::abs(left[i]), std::abs(right[i])});
        if (std::abs(left[i] - right[i]) > 1e-12 * scale)
        {
            return false;
        }
    }
    return true;
}

class CollectingEmitter final : public FrameEmitter
{
  public:
    explicit CollectingEmitter(const StreamSchema& schema)
        : pool_(1, schema.signals().front().max_block_bytes, 1), validator_(schema)
    {
    }

    void clear() noexcept
    {
        frame_count_ = 0;
        value_count_ = 0;
        failure_ = StreamStatus::ok;
        static_cast<void>(acquired_.reset());
    }

    StreamStatus try_acquire_frame(MutableFrame*& frame) noexcept override
    {
        frame = nullptr;
        if (acquired_)
        {
            return failure_ = StreamStatus::invalid_state;
        }
        const auto status = pool_.try_acquire(acquired_);
        if (status == StreamStatus::ok)
        {
            frame = &acquired_.frame();
        }
        return failure_ = status;
    }

    StreamStatus publish_acquired_frame() noexcept override
    {
        return record(std::move(acquired_));
    }

    StreamStatus publish_input() noexcept override
    {
        return failure_ = StreamStatus::invalid_state;
    }

    [[nodiscard]] std::size_t frame_count() const noexcept
    {
        return frame_count_;
    }
    [[nodiscard]] std::size_t value_count() const noexcept
    {
        return value_count_;
    }
    [[nodiscard]] std::span<const double> values() const noexcept
    {
        return {values_.data(), value_count_};
    }
    [[nodiscard]] const FrameHeader& header(std::size_t idx) const noexcept
    {
        return headers_[idx];
    }
    [[nodiscard]] const SignalBlockHeader& block(std::size_t idx) const noexcept
    {
        return blocks_[idx];
    }

  private:
    StreamStatus publish_owned(FrameLease lease) noexcept override
    {
        return record(std::move(lease));
    }

    StreamStatus record(FrameLease lease) noexcept
    {
        if (!lease || frame_count_ == max_frames ||
            validator_.validate(lease.view()) != FrameValidationError::none)
        {
            return failure_ = StreamStatus::invalid_frame;
        }
        const auto view = lease.view();
        const auto n_scalars = view.payload.size() / sizeof(double);
        if (n_scalars > max_values - value_count_)
        {
            return failure_ = StreamStatus::invalid_frame;
        }
        headers_[frame_count_] = view.header;
        blocks_[frame_count_] = view.blocks.front();
        std::memcpy(values_.data() + value_count_, view.payload.data(), view.payload.size());
        value_count_ += n_scalars;
        ++frame_count_;
        return failure_ = lease.reset();
    }

    FramePool pool_;
    FrameValidator validator_;
    FrameLease acquired_{};
    std::array<FrameHeader, max_frames> headers_{};
    std::array<SignalBlockHeader, max_frames> blocks_{};
    std::array<double, max_values> values_{};
    std::size_t frame_count_{};
    std::size_t value_count_{};
    StreamStatus failure_{StreamStatus::ok};
};

[[nodiscard]] std::size_t reference_output(std::size_t up, std::size_t down,
                                           std::span<const double> input, std::span<double> output)
{
    neurale::signal::Resampler reference{up, down, filter};
    const auto n_samples = input.size() / channels;
    reference.prepare(channels, n_samples);
    std::array<double, max_values> scratch{};
    auto written = reference.process(input, n_samples, channels, scratch);
    std::copy_n(scratch.begin(), written * channels, output.begin());
    const auto tail = reference.flush(scratch);
    std::copy_n(scratch.begin(), tail * channels,
                output.begin() + static_cast<std::ptrdiff_t>(written * channels));
    return (written + tail) * channels;
}

template <std::size_t ChunkCount>
[[nodiscard]] int run_adapter(std::size_t up, std::size_t down, std::span<const double> input,
                              const std::array<std::size_t, ChunkCount>& chunks,
                              std::span<double> output, std::span<SignalBlockHeader> output_blocks,
                              std::size_t& output_value_count, std::size_t& output_frame_count)
{
    auto input_schema = make_schema();
    neurale::pipeline::ResamplerAdapter adapter{31, up, down, filter};
    const auto contract = adapter.prepare(context(input_schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool input_pool{1, input_schema.signals().front().max_block_bytes, 1};
    FrameLease input_lease;
    CHECK(input_pool.try_acquire(input_lease) == StreamStatus::ok);
    std::size_t scalar_offset = 0;
    SampleIndex sample_offset = 0;
    for (const auto chunk : chunks)
    {
        const auto before = sink.frame_count();
        const auto n_scalars = chunk * channels;
        CHECK(fill_frame(input_lease.frame(), input.subspan(scalar_offset, n_scalars),
                         sample_offset + 1, sample_offset) == StreamStatus::ok);
        CHECK(adapter.process(input_lease.frame(), sink) == StreamStatus::ok);
        CHECK(sink.frame_count() - before <= contract.max_process_outputs_per_input);
        scalar_offset += n_scalars;
        sample_offset += chunk;
    }
    CHECK(scalar_offset == input.size());
    const auto before_flush = sink.frame_count();
    CHECK(adapter.flush(sink) == StreamStatus::ok);
    CHECK(sink.frame_count() - before_flush <= contract.max_flush_outputs);
    CHECK(sink.value_count() <= output.size());
    CHECK(sink.frame_count() <= output_blocks.size());
    std::copy(sink.values().begin(), sink.values().end(), output.begin());
    for (std::size_t i = 0; i < sink.frame_count(); ++i)
    {
        output_blocks[i] = sink.block(i);
    }
    output_value_count = sink.value_count();
    output_frame_count = sink.frame_count();
    return 0;
}

[[nodiscard]] int test_rational_rates_and_metadata()
{
    for (const auto [up, down] : std::array<std::array<std::size_t, 2>, 2>{{{3, 2}, {2, 3}}})
    {
        auto schema = make_schema();
        neurale::pipeline::ResamplerAdapter adapter{31, up, down, filter};
        const auto contract = adapter.prepare(context(schema));
        const auto& input = schema.signals().front();
        const auto& output = contract.output_schema.signals().front();
        const auto rate_common = std::gcd(30'000 * up, down);
        CHECK(contract.output_schema.id() == 31);
        CHECK(output.fs.numerator == 30'000 * up / rate_common);
        CHECK(output.fs.denominator == down / rate_common);
        CHECK(output.id == input.id);
        CHECK(output.n_channels == input.n_channels);
        CHECK(output.clock_domain == input.clock_domain);
        CHECK(output.layout == input.layout);
        CHECK(output.physical_unit == input.physical_unit);
        CHECK(output.channel_set_id == input.channel_set_id);
        CHECK(output.calibration_id == input.calibration_id);
        CHECK(output.reference_id == input.reference_id);
        CHECK(output.device_tick_tracking == DeviceTickTracking::unavailable);
        CHECK(!contract.can_forward_input);
    }
    return 0;
}

[[nodiscard]] int test_chunk_invariance_timing_bounds_and_flush()
{
    std::array<double, 32> input{};
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = std::sin(static_cast<double>(i) * 0.17) + static_cast<double>(i % channels);
    }
    std::array<double, max_values> expected{};
    const auto n_expected = reference_output(3, 2, input, expected);

    std::array<double, max_values> first{};
    std::array<double, max_values> second{};
    std::array<SignalBlockHeader, max_frames> first_blocks{};
    std::array<SignalBlockHeader, max_frames> second_blocks{};
    std::size_t first_count{};
    std::size_t second_count{};
    std::size_t first_frames{};
    std::size_t second_frames{};
    CHECK(run_adapter(3, 2, input, std::array<std::size_t, 5>{1, 2, 5, 3, 5}, first, first_blocks,
                      first_count, first_frames) == 0);
    CHECK(run_adapter(3, 2, input, std::array<std::size_t, 2>{8, 8}, second, second_blocks,
                      second_count, second_frames) == 0);
    CHECK(first_count == n_expected);
    CHECK(second_count == n_expected);
    CHECK(nearly_equal(std::span<const double>{first.data(), first_count},
                       std::span<const double>{expected.data(), n_expected}));
    CHECK(nearly_equal(std::span<const double>{first.data(), first_count},
                       std::span<const double>{second.data(), second_count}));
    CHECK(second_frames >= 2);
    for (std::size_t i = 1; i < first_frames; ++i)
    {
        CHECK(first_blocks[i].sample_idx_start ==
              first_blocks[i - 1].sample_idx_start + first_blocks[i - 1].n_samples);
        CHECK(first_blocks[i].device_tick_start >= first_blocks[i - 1].device_tick_start);
        CHECK(first_blocks[i].clock_sync.clock_domain == 41);
    }
    return 0;
}

[[nodiscard]] int test_discontinuity_restarts_state_and_timing()
{
    auto schema = make_schema();
    neurale::pipeline::ResamplerAdapter adapter{31, 2, 3, filter};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool input_pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(input_pool.try_acquire(lease) == StreamStatus::ok);
    const std::array warmup{1.0, -1.0, 2.0, -2.0, 3.0, -3.0};
    const std::array probe{0.5, 1.5, -0.5, 2.5, 3.5, -1.5, 4.5, -2.5};
    std::array<double, max_values> expected{};
    const auto n_expected = reference_output(2, 3, probe, expected);
    CHECK(fill_frame(lease.frame(), warmup, 1, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(adapter.reset() == StreamStatus::ok);
    sink.clear();
    CHECK(fill_frame(lease.frame(), probe, 2, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(adapter.flush(sink) == StreamStatus::ok);
    CHECK(sink.value_count() == n_expected);
    CHECK(nearly_equal(sink.values(), std::span<const double>{expected.data(), n_expected}));

    sink.clear();
    CHECK(fill_frame(lease.frame(), warmup, 3, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    const Discontinuity discontinuity{
        .session_id = 5,
        .previous_frame_sequence = 3,
        .actual_frame_sequence = 9,
        .reason = GapReason::source_gap,
    };
    CHECK(adapter.handle_discontinuity(discontinuity) == StreamStatus::ok);
    sink.clear();
    CHECK(fill_frame(lease.frame(), probe, 9, 90) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(adapter.flush(sink) == StreamStatus::ok);
    CHECK(sink.value_count() == n_expected);
    CHECK(nearly_equal(sink.values(), std::span<const double>{expected.data(), n_expected}));
    CHECK(sink.block(0).sample_idx_start == 60);
    return 0;
}

[[nodiscard]] int test_capacity_rejection_and_no_allocation()
{
    auto schema = make_schema();
    neurale::pipeline::ResamplerAdapter adapter{31, 3, 2, filter};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool input_pool{1, (input_max_samples + 1) * channels * sizeof(double), 1};
    FrameLease lease;
    CHECK(input_pool.try_acquire(lease) == StreamStatus::ok);
    std::array<double, (input_max_samples + 1) * channels> too_large{};
    CHECK(fill_frame(lease.frame(), too_large, 1, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::invalid_frame);

    CHECK(adapter.reset() == StreamStatus::ok);
    sink.clear();
    const std::array input{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    CHECK(fill_frame(lease.frame(), input, 2, 0) == StreamStatus::ok);
    const auto before = allocations.load(std::memory_order_relaxed);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(adapter.flush(sink) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == before);
    return 0;
}

[[nodiscard]] int test_sos_resampler_chain()
{
    auto schema = make_schema();
    neurale::pipeline::SosFilterAdapter sos_adapter{sos};
    neurale::pipeline::ResamplerAdapter resampler_adapter{31, 3, 2, filter};
    std::array<NativeFrameProcessor*, 2> stages{&sos_adapter, &resampler_adapter};
    LinearProcessorChain chain{stages};
    const auto contract = chain.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};

    std::array<double, 16> input{};
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = static_cast<double>(i) - 3.0;
    }
    auto filtered = input;
    neurale::signal::detail::SosRealtimeProcessor sos_reference{sos, channels,
                                                                std::span<const double>{}};
    sos_reference.process(filtered, filtered.size() / channels);
    std::array<double, max_values> expected{};
    const auto n_expected = reference_output(3, 2, filtered, expected);

    FramePool input_pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(input_pool.try_acquire(lease) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), input, 1, 0) == StreamStatus::ok);
    CHECK(chain.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(chain.flush(sink) == StreamStatus::ok);
    CHECK(sink.value_count() == n_expected);
    CHECK(nearly_equal(sink.values(), std::span<const double>{expected.data(), n_expected}));
    return 0;
}

} // namespace

int main()
{
    if (const auto status = test_rational_rates_and_metadata(); status != 0)
    {
        return status;
    }
    if (const auto status = test_chunk_invariance_timing_bounds_and_flush(); status != 0)
    {
        return status;
    }
    if (const auto status = test_discontinuity_restarts_state_and_timing(); status != 0)
    {
        return status;
    }
    if (const auto status = test_capacity_rejection_and_no_allocation(); status != 0)
    {
        return status;
    }
    return test_sos_resampler_chain();
}
