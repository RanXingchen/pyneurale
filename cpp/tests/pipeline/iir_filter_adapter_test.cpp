/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "allocation_counter.h"
#include "check_returns.h"
#include "iir_filter_adapter.h"
#include "inplace_adapter_test_support.h"
#include "lmp_feature_adapter.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <neurale/features/online.h>
#include <neurale/signal/iir.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/linear_processor_chain.h>

namespace
{

using namespace neurale::streaming;

constexpr std::size_t maximum_channels = 128;
constexpr std::size_t maximum_samples = 16;
constexpr std::size_t maximum_scalars = maximum_channels * maximum_samples;
constexpr std::size_t maximum_values = 4096;
constexpr std::array<double, 2> first_order_b{0.2, 0.1};
constexpr std::array<double, 2> first_order_a{1.0, -0.7};
constexpr std::array<double, 4> higher_order_b{
    0.00482434335771623,
    0.0144730300731487,
    0.0144730300731487,
    0.00482434335771623,
};
constexpr std::array<double, 4> higher_order_a{
    1.0,
    -2.36951300718204,
    1.92935566909122,
    -0.532075368312092,
};
constexpr std::array<double, 6> lmp_sos{0.5, 0.25, 0.125, 1.0, -0.2, 0.05};

namespace test_support = neurale::pipeline::test_support;

constexpr test_support::InplaceGeometry base_geometry{
    .n_channels = 2,
    .nominal_block_samples = 4,
    .max_block_samples = static_cast<std::uint32_t>(maximum_samples),
    .fs = {1'000, 1},
};

[[nodiscard]] StreamSchema make_schema(std::uint32_t channels = 2,
                                       SignalDType dtype = SignalDType::float64,
                                       SignalLayout layout = SignalLayout::sample_major,
                                       std::uint32_t max_samples = maximum_samples)
{
    auto geometry = base_geometry;
    geometry.n_channels = channels;
    geometry.dtype = dtype;
    geometry.layout = layout;
    geometry.max_block_samples = max_samples;
    return test_support::make_schema(geometry);
}

// Eight flush outputs offered on purpose: the adapter must still declare zero
// rather than inherit the offer.
[[nodiscard]] ProcessorPrepareContext context(const StreamSchema& schema, std::size_t outputs = 8,
                                              std::size_t leases = 4) noexcept
{
    return test_support::make_context(schema, outputs, leases, 8);
}

[[nodiscard]] StreamStatus fill_frame(MutableFrame& frame, const StreamSchema& schema,
                                      std::span<const double> values, std::uint64_t sequence = 7,
                                      SampleIndex sample_start = 100) noexcept
{
    return test_support::fill_frame(frame, schema, base_geometry, values, sequence, sample_start);
}

using test_support::nearly_equal;

[[nodiscard]] std::vector<double> offline_reference(std::span<const double> input,
                                                    std::size_t channels,
                                                    std::span<const double> b = first_order_b,
                                                    std::span<const double> a = first_order_a)
{
    const auto samples = input.size() / channels;
    std::vector<double> output(input.size());
    std::vector<double> state((b.size() - 1) * channels, 0.0);
    std::vector<double> final_state(state.size());
    neurale::signal::iir_filter(input, samples, channels, b, a, state, output, final_state);
    return output;
}

// A pool of one: the IIR adapter never acquires, but the LMP adapter this test
// puts downstream does.
class TerminalSink final : public test_support::TerminalSink
{
  public:
    explicit TerminalSink(const StreamSchema& schema)
        : test_support::TerminalSink(schema, 1, maximum_values)
    {
    }
};

using test_support::throws_invalid_argument;

[[nodiscard]] int test_single_and_multichannel_parity_and_metadata()
{
    for (const auto n_channels : std::array<std::uint32_t, 2>{1, 3})
    {
        auto schema = make_schema(n_channels);
        neurale::pipeline::IirFilterAdapter adapter{first_order_b, first_order_a};
        const auto contract = adapter.prepare(context(schema));
        CHECK(contract.accepted_input_schema.equivalent(schema));
        CHECK(contract.output_schema.equivalent(schema));
        CHECK(contract.max_process_outputs_per_input == 1);
        CHECK(contract.max_flush_outputs == 0);
        CHECK(contract.can_forward_input);

        std::vector<double> input(7 * n_channels);
        for (std::size_t i = 0; i < input.size(); ++i)
        {
            input[i] =
                std::sin(static_cast<double>(i) * 0.19) + static_cast<double>(i % n_channels);
        }
        const auto expected = offline_reference(input, n_channels);
        FramePool pool{1, schema.signals().front().max_block_bytes, 1};
        FrameLease lease;
        CHECK(pool.try_acquire(lease) == StreamStatus::ok);
        CHECK(fill_frame(lease.frame(), schema, input) == StreamStatus::ok);
        const auto original_header = lease.frame().header();
        const auto original_block = lease.frame().blocks().front();
        TerminalSink sink{schema};
        sink.begin(lease.frame());
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
        CHECK(sink.publication_count() == 1);
        CHECK(nearly_equal(sink.values(), expected));
        CHECK(sink.header().session_id == original_header.session_id);
        CHECK(sink.header().sequence == original_header.sequence);
        CHECK(sink.header().host_received_ns == original_header.host_received_ns);
        CHECK(sink.header().source_tick == original_header.source_tick);
        CHECK(sink.header().valid_until_ns == original_header.valid_until_ns);
        CHECK(sink.header().schema_id == original_header.schema_id);
        CHECK(sink.header().source_clock_domain == original_header.source_clock_domain);
        CHECK(sink.header().signal_block_count == original_header.signal_block_count);
        CHECK(sink.header().flags == original_header.flags);
        CHECK(sink.block().sample_idx_start == original_block.sample_idx_start);
        CHECK(sink.block().device_tick_start == original_block.device_tick_start);
        CHECK(sink.block().observation_time_start_ns == original_block.observation_time_start_ns);
        CHECK(sink.block().payload_offset == original_block.payload_offset);
        CHECK(sink.block().payload_byte_count == original_block.payload_byte_count);
        CHECK(sink.block().signal_id == original_block.signal_id);
        CHECK(sink.block().n_samples == original_block.n_samples);
        CHECK(sink.block().clock_sync.device_tick_reference ==
              original_block.clock_sync.device_tick_reference);
        CHECK(sink.block().clock_sync.host_time_reference_ns ==
              original_block.clock_sync.host_time_reference_ns);
        CHECK(sink.block().clock_sync.device_tick_rate.numerator ==
              original_block.clock_sync.device_tick_rate.numerator);
        CHECK(sink.block().clock_sync.device_tick_rate.denominator ==
              original_block.clock_sync.device_tick_rate.denominator);
        CHECK(sink.block().clock_sync.uncertainty_ns == original_block.clock_sync.uncertainty_ns);
        CHECK(sink.block().clock_sync.clock_domain == original_block.clock_sync.clock_domain);
        CHECK(sink.block().clock_sync.generation == original_block.clock_sync.generation);
        CHECK(sink.block().clock_sync.flags == original_block.clock_sync.flags);
    }
    return 0;
}

template <std::size_t ChunkCount>
[[nodiscard]] int run_chunks(std::span<const double> input,
                             const std::array<std::size_t, ChunkCount>& chunks,
                             std::span<double> output)
{
    constexpr std::size_t channels = 2;
    auto schema = make_schema(channels);
    neurale::pipeline::IirFilterAdapter adapter{first_order_b, first_order_a};
    static_cast<void>(adapter.prepare(context(schema)));
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    TerminalSink sink{schema};
    std::size_t sample_offset = 0;
    for (const auto chunk : chunks)
    {
        CHECK(chunk <= maximum_samples);
        CHECK(fill_frame(lease.frame(), schema,
                         input.subspan(sample_offset * channels, chunk * channels),
                         sample_offset + 1, sample_offset) == StreamStatus::ok);
        sink.begin(lease.frame());
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
        std::copy(sink.values().begin(), sink.values().end(),
                  output.begin() + static_cast<std::ptrdiff_t>(sample_offset * channels));
        sample_offset += chunk;
    }
    CHECK(sample_offset * channels == input.size());
    return 0;
}

[[nodiscard]] int test_variable_chunks_and_chunk_invariance()
{
    std::array<double, 34> input{};
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = std::cos(static_cast<double>(i) * 0.11) + static_cast<double>(i % 2);
    }
    const auto expected = offline_reference(input, 2);
    std::array<double, input.size()> first{};
    std::array<double, input.size()> second{};
    CHECK(run_chunks(input, std::array<std::size_t, 5>{1, 4, 2, 8, 2}, first) == 0);
    CHECK(run_chunks(input, std::array<std::size_t, 3>{8, 1, 8}, second) == 0);
    CHECK(nearly_equal(first, expected));
    CHECK(nearly_equal(second, expected));
    CHECK(nearly_equal(first, second));
    return 0;
}

[[nodiscard]] int test_higher_order_stable_filter_parity()
{
    constexpr std::size_t channels = 3;
    auto schema = make_schema(channels);
    neurale::pipeline::IirFilterAdapter adapter{higher_order_b, higher_order_a};
    static_cast<void>(adapter.prepare(context(schema)));
    std::array<double, maximum_samples * channels> input{};
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] =
            std::sin(static_cast<double>(i) * 0.13) + static_cast<double>(i % channels) * 0.25;
    }
    const auto expected = offline_reference(input, channels, higher_order_b, higher_order_a);
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, input) == StreamStatus::ok);
    TerminalSink sink{schema};
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(nearly_equal(sink.values(), expected));
    return 0;
}

[[nodiscard]] int test_reset_discontinuity_empty_and_flush()
{
    constexpr std::size_t channels = 2;
    auto schema = make_schema(channels);
    neurale::pipeline::IirFilterAdapter adapter{first_order_b, first_order_a};
    static_cast<void>(adapter.prepare(context(schema)));
    const std::array warmup{2.0, -1.0, 1.0, 0.5};
    const std::array probe{0.25, 0.75, -0.5, 1.5, 2.0, -2.0};
    const auto expected = offline_reference(probe, channels);
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    TerminalSink sink{schema};

    CHECK(fill_frame(lease.frame(), schema, warmup) == StreamStatus::ok);
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    const auto before_flush = sink.publication_count();
    CHECK(adapter.flush(sink) == StreamStatus::ok);
    CHECK(sink.publication_count() == before_flush);
    CHECK(adapter.reset() == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, probe) == StreamStatus::ok);
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(nearly_equal(sink.values(), expected));

    CHECK(fill_frame(lease.frame(), schema, warmup) == StreamStatus::ok);
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    const Discontinuity discontinuity{
        .session_id = 5,
        .previous_frame_sequence = 10,
        .actual_frame_sequence = 12,
        .reason = GapReason::source_gap,
    };
    CHECK(adapter.handle_discontinuity(discontinuity) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, probe) == StreamStatus::ok);
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(nearly_equal(sink.values(), expected));

    auto empty_block = lease.frame().blocks().front();
    empty_block.payload_offset = 0;
    empty_block.payload_byte_count = 0;
    empty_block.n_samples = 0;
    lease.frame().header() = FrameHeader{.schema_id = schema.id()};
    lease.frame().block_storage()[0] = empty_block;
    CHECK(lease.frame().set_used_sizes(1, 0) == StreamStatus::ok);
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::invalid_frame);
    CHECK(sink.publication_count() == 0);
    return 0;
}

[[nodiscard]] int test_invalid_coefficients_schema_layout_and_capacity()
{
    auto schema = make_schema();
    TerminalSink unprepared_sink{schema};
    neurale::pipeline::IirFilterAdapter unprepared{first_order_b, first_order_a};
    CHECK(unprepared.reset() == StreamStatus::invalid_state);
    CHECK(unprepared.flush(unprepared_sink) == StreamStatus::invalid_state);

    CHECK(throws_invalid_argument(
        []
        {
            neurale::pipeline::IirFilterAdapter adapter{std::span<const double>{},
                                                        std::span<const double>{}};
        }));
    CHECK(throws_invalid_argument(
        []
        {
            const std::array invalid_b{1.0, std::numeric_limits<double>::quiet_NaN()};
            neurale::pipeline::IirFilterAdapter adapter{invalid_b, first_order_a};
        }));
    CHECK(throws_invalid_argument(
        []
        {
            const std::array short_a{1.0};
            neurale::pipeline::IirFilterAdapter adapter{first_order_b, short_a};
        }));
    CHECK(throws_invalid_argument(
        []
        {
            const std::array unnormalized_a{2.0, -0.7};
            neurale::pipeline::IirFilterAdapter adapter{first_order_b, unnormalized_a};
        }));
    auto float_schema = make_schema(2, SignalDType::float32);
    CHECK(throws_invalid_argument(
        [&]
        {
            neurale::pipeline::IirFilterAdapter adapter{first_order_b, first_order_a};
            static_cast<void>(adapter.prepare(context(float_schema)));
        }));
    auto channel_major = make_schema(2, SignalDType::float64, SignalLayout::channel_major);
    CHECK(throws_invalid_argument(
        [&]
        {
            neurale::pipeline::IirFilterAdapter adapter{first_order_b, first_order_a};
            static_cast<void>(adapter.prepare(context(channel_major)));
        }));
    CHECK(throws_invalid_argument(
        [&]
        {
            neurale::pipeline::IirFilterAdapter adapter{first_order_b, first_order_a};
            static_cast<void>(adapter.prepare(context(schema, 0, 1)));
        }));
    CHECK(throws_invalid_argument(
        [&]
        {
            neurale::pipeline::IirFilterAdapter adapter{first_order_b, first_order_a};
            static_cast<void>(adapter.prepare(context(schema, 1, 0)));
        }));

    neurale::pipeline::IirFilterAdapter adapter{first_order_b, first_order_a};
    static_cast<void>(adapter.prepare(context(schema)));
    auto changed_channels = make_schema(3);
    CHECK(throws_invalid_argument(
        [&] { static_cast<void>(adapter.prepare(context(changed_channels))); }));
    FramePool pool{1, (maximum_samples + 1) * 2 * sizeof(double), 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    std::array<double, (maximum_samples + 1) * 2> too_large{};
    CHECK(fill_frame(lease.frame(), schema, too_large) == StreamStatus::ok);
    TerminalSink sink{schema};
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::invalid_frame);
    return 0;
}

[[nodiscard]] int test_coefficients_are_owned_and_fixed()
{
    auto b = first_order_b;
    auto a = first_order_a;
    neurale::pipeline::IirFilterAdapter adapter{b, a};
    b = {1.0, 0.0};
    a = {1.0, 0.0};

    auto schema = make_schema(2);
    static_cast<void>(adapter.prepare(context(schema)));
    const std::array input{1.0, -1.0, 0.5, 2.0, -0.25, 0.75};
    const auto expected = offline_reference(input, 2, first_order_b, first_order_a);
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, input) == StreamStatus::ok);
    TerminalSink sink{schema};
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(nearly_equal(sink.values(), expected));
    return 0;
}

[[nodiscard]] int test_allocation_tracking_and_terminal_chain()
{
    constexpr std::size_t channels = maximum_channels;
    auto schema = make_schema(channels);
    neurale::pipeline::IirFilterAdapter adapter{higher_order_b, higher_order_a};
    std::array<NativeFrameProcessor*, 1> stages{&adapter};
    LinearProcessorChain chain{stages};
    const auto contract = chain.prepare(context(schema));
    CHECK(contract.output_schema.equivalent(schema));
    std::array<double, maximum_scalars> input{};
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = static_cast<double>(i) * 0.01;
    }
    const auto expected = offline_reference(input, channels, higher_order_b, higher_order_a);
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    TerminalSink sink{schema};
    const auto before = allocations.load(std::memory_order_relaxed);
    for (std::size_t iteration = 0; iteration < 64; ++iteration)
    {
        CHECK(adapter.reset() == StreamStatus::ok);
        CHECK(fill_frame(lease.frame(), schema, input, iteration + 1,
                         iteration * maximum_samples) == StreamStatus::ok);
        sink.begin(lease.frame());
        CHECK(chain.process(lease.frame(), sink) == StreamStatus::ok);
    }
    CHECK(allocations.load(std::memory_order_relaxed) == before);
    CHECK(sink.publication_count() == 1);
    CHECK(nearly_equal(sink.values(), expected));
    CHECK(chain.flush(sink) == StreamStatus::ok);
    return 0;
}

[[nodiscard]] neurale::pipeline::LmpFeatureAdapterConfig lmp_config()
{
    return {
        .output_schema_id = 37,
        .output_signal_id = 43,
        .feature_set_id = 47,
        .feature_unit =
            UnitDescriptor{
                .id = 53,
                .symbol = "V",
                .description = "volts",
            },
        .feature_names = {"lmp:left", "lmp:right"},
        .source_stream = "electrode-voltage",
        .algorithm_version = "1",
        .window_samples = 4,
        .shift_samples = 2,
        .sos = {lmp_sos.begin(), lmp_sos.end()},
        .sos_sections = 1,
    };
}

[[nodiscard]] int test_iir_lmp_chain()
{
    constexpr std::size_t channels = 2;
    auto schema = make_schema(channels, SignalDType::float64, SignalLayout::sample_major, 8);
    neurale::pipeline::IirFilterAdapter iir_adapter{first_order_b, first_order_a};
    neurale::pipeline::LmpFeatureAdapter lmp_adapter{lmp_config()};
    std::array<NativeFrameProcessor*, 2> stages{&iir_adapter, &lmp_adapter};
    LinearProcessorChain chain{stages};
    const auto contract = chain.prepare(context(schema));
    CHECK(contract.output_schema.signals().front().kind == SignalKind::feature);
    TerminalSink sink{contract.output_schema};

    std::array<double, 24> input{};
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = std::sin(static_cast<double>(i) * 0.17) + static_cast<double>(i % channels);
    }
    const auto filtered = offline_reference(input, channels);
    neurale::features::LmpProcessor reference{4, 2, channels, lmp_sos, 1};
    const auto expected_rows = reference.output_count(input.size() / channels);
    std::array<double, maximum_values> expected{};
    std::array<double, maximum_values> actual{};
    reference.process(filtered, input.size() / channels,
                      {expected.data(), expected_rows * channels});

    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    std::size_t sample_offset = 0;
    std::size_t output_offset = 0;
    for (const auto chunk : std::array<std::size_t, 2>{5, 7})
    {
        CHECK(fill_frame(lease.frame(), schema,
                         std::span<const double>{input}.subspan(sample_offset * channels,
                                                                chunk * channels),
                         sample_offset + 1, sample_offset) == StreamStatus::ok);
        sink.begin(lease.frame());
        CHECK(chain.process(lease.frame(), sink) == StreamStatus::ok);
        std::copy(sink.values().begin(), sink.values().end(),
                  actual.begin() + static_cast<std::ptrdiff_t>(output_offset));
        output_offset += sink.values().size();
        sample_offset += chunk;
    }
    CHECK(output_offset == expected_rows * channels);
    CHECK(
        nearly_equal({actual.data(), output_offset}, {expected.data(), expected_rows * channels}));
    return 0;
}

} // namespace

int main()
{
    if (const auto status = test_single_and_multichannel_parity_and_metadata(); status != 0)
    {
        return status;
    }
    if (const auto status = test_variable_chunks_and_chunk_invariance(); status != 0)
    {
        return status;
    }
    if (const auto status = test_higher_order_stable_filter_parity(); status != 0)
    {
        return status;
    }
    if (const auto status = test_reset_discontinuity_empty_and_flush(); status != 0)
    {
        return status;
    }
    if (const auto status = test_invalid_coefficients_schema_layout_and_capacity(); status != 0)
    {
        return status;
    }
    if (const auto status = test_coefficients_are_owned_and_fixed(); status != 0)
    {
        return status;
    }
    if (const auto status = test_allocation_tracking_and_terminal_chain(); status != 0)
    {
        return status;
    }
    return test_iir_lmp_chain();
}
