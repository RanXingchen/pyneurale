/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "allocation_counter.h"
#include "feature_adapter_test_support.h"
#include "lmp_feature_adapter.h"
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
#include <span>
#include <string>
#include <vector>

#include <neurale/features/online.h>
#include <neurale/signal/resample.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/linear_processor_chain.h>

namespace
{

using namespace neurale::streaming;
using namespace neurale::pipeline::test;

constexpr std::array<double, 6> lmp_sos{0.5, 0.25, 0.125, 1.0, -0.2, 0.05};
constexpr std::array<double, 6> chain_sos{0.75, 0.1, 0.05, 1.0, -0.1, 0.02};
constexpr std::array<double, 5> resampler_filter{0.05, 0.2, 0.5, 0.2, 0.05};

[[nodiscard]] StreamSchema make_schema(std::uint32_t nominal_samples = 6,
                                       std::uint32_t maximum_samples = max_input_samples,
                                       RationalRate rate = {1'000, 1})
{
    const std::array signals{
        SignalSchema{
            11,
            SignalDType::float64,
            channels,
            nominal_samples,
            maximum_samples,
            rate,
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

[[nodiscard]] neurale::pipeline::LmpFeatureAdapterConfig lmp_config(std::size_t window_samples = 4,
                                                                    std::size_t shift_samples = 2)
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
        .window_samples = window_samples,
        .shift_samples = shift_samples,
        .sos = {lmp_sos.begin(), lmp_sos.end()},
        .sos_sections = 1,
    };
}

[[nodiscard]] std::size_t offline_lmp_reference(std::span<const double> input,
                                                std::size_t window_samples,
                                                std::size_t shift_samples, std::span<double> output)
{
    std::vector<double> filtered{input.begin(), input.end()};
    neurale::signal::detail::SosRealtimeProcessor filter{lmp_sos, channels,
                                                         std::span<const double>{}};
    filter.process(filtered, filtered.size() / channels);
    const auto input_samples = filtered.size() / channels;
    std::size_t output_row = 0;
    for (std::size_t start = 0; start + window_samples <= input_samples; start += shift_samples)
    {
        for (std::size_t channel = 0; channel < channels; ++channel)
        {
            double sum = 0.0;
            for (std::size_t sample = 0; sample < window_samples; ++sample)
            {
                sum += filtered[(start + sample) * channels + channel];
            }
            output[output_row * channels + channel] = sum / static_cast<double>(window_samples);
        }
        ++output_row;
    }
    return output_row * channels;
}

template <std::size_t ChunkCount>
[[nodiscard]] int run_lmp(std::span<const double> input,
                          const std::array<std::size_t, ChunkCount>& chunks,
                          std::span<double> output, std::span<SignalBlockHeader> blocks,
                          std::size_t& n_outputs, std::size_t& n_frames)
{
    auto schema = make_schema();
    neurale::pipeline::LmpFeatureAdapter adapter{lmp_config()};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool input_pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(input_pool.try_acquire(lease) == StreamStatus::ok);
    std::size_t scalar_offset = 0;
    SampleIndex sample_offset = 0;
    for (const auto chunk : chunks)
    {
        const auto prior_frames = sink.frame_count();
        CHECK(fill_frame(lease.frame(), schema, input.subspan(scalar_offset, chunk * channels),
                         sample_offset + 1, sample_offset) == StreamStatus::ok);
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
        CHECK(sink.frame_count() - prior_frames <= contract.max_process_outputs_per_input);
        scalar_offset += chunk * channels;
        sample_offset += chunk;
    }
    CHECK(scalar_offset == input.size());
    CHECK(adapter.flush(sink) == StreamStatus::ok);
    CHECK(sink.values().size() <= output.size());
    CHECK(sink.frame_count() <= blocks.size());
    std::copy(sink.values().begin(), sink.values().end(), output.begin());
    for (std::size_t i = 0; i < sink.frame_count(); ++i)
    {
        blocks[i] = sink.block(i);
    }
    n_outputs = sink.values().size();
    n_frames = sink.frame_count();
    return 0;
}

[[nodiscard]] int test_descriptor_and_timing()
{
    auto schema = make_schema();
    neurale::pipeline::LmpFeatureAdapter adapter{lmp_config()};
    const auto contract = adapter.prepare(context(schema));
    const auto& output = contract.output_schema.signals().front();
    CHECK(output.kind == SignalKind::feature);
    CHECK(output.id == 43);
    CHECK(output.feature_set_id == 47);
    CHECK(output.observation_timing == ObservationTiming::regular);
    CHECK(output.fs.numerator == 500);
    CHECK(output.fs.denominator == 1);
    CHECK(output.n_channels == channels);
    CHECK(output.channel_set_id == schema.signals().front().channel_set_id);
    CHECK(output.clock_domain == schema.signals().front().clock_domain);
    CHECK(output.physical_unit == PhysicalUnit::unspecified);
    const auto* descriptor = contract.output_schema.feature_sets().find(47);
    CHECK(descriptor != nullptr);
    CHECK(descriptor->feature_names == std::vector<std::string>({"lmp:left", "lmp:right"}));
    CHECK(descriptor->unit_ids == std::vector<UnitId>({53, 53}));
    CHECK(descriptor->source_stream_id == 11);
    CHECK(descriptor->source_stream == "electrode-voltage");
    CHECK(descriptor->algorithm_name == "lmp");
    CHECK(descriptor->algorithm_version == "1");
    CHECK(descriptor->window_length_ns == 4'000'000);
    CHECK(descriptor->shift_ns == 2'000'000);
    CHECK(descriptor->timestamp_reference == FeatureTimestampReference::window_center);
    CHECK(contract.output_schema.units().find(53) != nullptr);
    CHECK(contract.max_process_outputs_per_input == 1);
    CHECK(contract.max_flush_outputs == 0);
    CHECK(!contract.can_forward_input);

    const std::array<double, 12> input{1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                                       7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, input, 1, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.frame_count() == 1);
    CHECK(sink.block(0).observation_time_start_ns == 12'000'000);
    CHECK(sink.block(0).n_samples == 2);
    CHECK(sink.block(0).sample_idx_start == 0);
    return 0;
}

[[nodiscard]] int test_offline_parity_and_chunk_invariance()
{
    std::array<double, 40> input{};
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = std::sin(static_cast<double>(i) * 0.13) + static_cast<double>(i % channels);
    }
    std::array<double, max_values> expected{};
    const auto n_expected = offline_lmp_reference(input, 4, 2, expected);
    std::array<double, max_values> first{};
    std::array<double, max_values> second{};
    std::array<SignalBlockHeader, max_frames> first_blocks{};
    std::array<SignalBlockHeader, max_frames> second_blocks{};
    std::size_t first_count{};
    std::size_t second_count{};
    std::size_t first_frames{};
    std::size_t second_frames{};
    CHECK(run_lmp(input, std::array<std::size_t, 5>{1, 3, 8, 2, 6}, first, first_blocks,
                  first_count, first_frames) == 0);
    CHECK(run_lmp(input, std::array<std::size_t, 3>{8, 8, 4}, second, second_blocks, second_count,
                  second_frames) == 0);
    CHECK(first_count == n_expected);
    CHECK(second_count == n_expected);
    CHECK(nearly_equal({first.data(), first_count}, {expected.data(), n_expected}));
    CHECK(nearly_equal({first.data(), first_count}, {second.data(), second_count}));
    std::uint64_t observation_idx = 0;
    for (std::size_t i = 0; i < first_frames; ++i)
    {
        CHECK(first_blocks[i].sample_idx_start == observation_idx);
        CHECK(first_blocks[i].observation_time_start_ns ==
              12'000'000 + observation_idx * 2'000'000);
        observation_idx += first_blocks[i].n_samples;
    }
    return 0;
}

[[nodiscard]] int test_reset_and_discontinuity_barriers()
{
    auto schema = make_schema();
    neurale::pipeline::LmpFeatureAdapter adapter{lmp_config()};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    const std::array<double, 6> partial{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::array<double, 2> one_sample{7.0, 8.0};
    const std::array<double, 6> remaining{9.0, 10.0, 11.0, 12.0, 13.0, 14.0};

    CHECK(fill_frame(lease.frame(), schema, partial, 1, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.frame_count() == 0);
    CHECK(adapter.reset() == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, one_sample, 2, 100) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.frame_count() == 0);
    CHECK(fill_frame(lease.frame(), schema, remaining, 3, 101) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.frame_count() == 1);
    CHECK(sink.block(0).observation_time_start_ns == 112'000'000);

    CHECK(adapter.reset() == StreamStatus::ok);
    sink.clear();
    CHECK(fill_frame(lease.frame(), schema, partial, 4, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    const Discontinuity discontinuity{
        .session_id = 5,
        .previous_frame_sequence = 4,
        .actual_frame_sequence = 9,
        .reason = GapReason::source_gap,
    };
    CHECK(adapter.handle_discontinuity(discontinuity) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, one_sample, 9, 200) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.frame_count() == 0);
    CHECK(fill_frame(lease.frame(), schema, remaining, 10, 201) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.frame_count() == 1);
    CHECK(sink.block(0).observation_time_start_ns == 212'000'000);
    return 0;
}

[[nodiscard]] int test_validation_and_no_allocation()
{
    auto schema = make_schema();
    neurale::pipeline::LmpFeatureAdapter adapter{lmp_config()};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    const std::array<double, 16> input{1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                                       9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    CHECK(fill_frame(lease.frame(), schema, input, 1, 0) == StreamStatus::ok);
    const auto before = allocations.load(std::memory_order_relaxed);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(adapter.flush(sink) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == before);

    CHECK(adapter.reset() == StreamStatus::ok);
    sink.clear();
    CHECK(fill_frame(lease.frame(), schema, input, 2, 0) == StreamStatus::ok);
    lease.frame().block_storage()[0].clock_sync.flags = ClockSyncFlags::none;
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::invalid_frame);
    return 0;
}

[[nodiscard]] int test_sos_resampler_lmp_chain()
{
    constexpr std::size_t source_samples = 60;
    auto schema = make_schema(20, 20, {1'000, 1});
    neurale::pipeline::SosFilterAdapter sos_adapter{chain_sos};
    neurale::pipeline::ResamplerAdapter resampler_adapter{31, 2, 1, resampler_filter};
    auto config = lmp_config(8, 4);
    neurale::pipeline::LmpFeatureAdapter lmp_adapter{std::move(config)};
    std::array<NativeFrameProcessor*, 3> stages{&sos_adapter, &resampler_adapter, &lmp_adapter};
    LinearProcessorChain chain{stages};
    const auto contract = chain.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};

    std::array<double, source_samples * channels> input{};
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = std::cos(static_cast<double>(i) * 0.07) + static_cast<double>(i % channels);
    }
    auto filtered = input;
    neurale::signal::detail::SosRealtimeProcessor sos_reference{chain_sos, channels,
                                                                std::span<const double>{}};
    sos_reference.process(filtered, source_samples);
    neurale::signal::Resampler resampler_reference{2, 1, resampler_filter};
    resampler_reference.prepare(channels, source_samples);
    std::array<double, max_values> resampled{};
    auto n_resampled = resampler_reference.process(filtered, source_samples, channels, resampled);
    n_resampled +=
        resampler_reference.flush(std::span<double>{resampled}.subspan(n_resampled * channels));
    neurale::features::LmpProcessor lmp_reference{8, 4, channels, lmp_sos, 1};
    const auto expected_rows = lmp_reference.output_count(n_resampled);
    std::array<double, max_values> expected{};
    lmp_reference.process(std::span<const double>{resampled.data(), n_resampled * channels},
                          n_resampled,
                          std::span<double>{expected.data(), expected_rows * channels});

    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    std::size_t offset = 0;
    std::uint64_t sequence = 1;
    while (offset < source_samples)
    {
        const auto chunk = std::min<std::size_t>(20, source_samples - offset);
        CHECK(
            fill_frame(lease.frame(), schema,
                       std::span<const double>{input}.subspan(offset * channels, chunk * channels),
                       sequence++, offset) == StreamStatus::ok);
        CHECK(chain.process(lease.frame(), sink) == StreamStatus::ok);
        offset += chunk;
    }
    CHECK(chain.flush(sink) == StreamStatus::ok);
    CHECK(sink.values().size() == expected_rows * channels);
    CHECK(nearly_equal(sink.values(),
                       std::span<const double>{expected.data(), expected_rows * channels}));
    CHECK(contract.output_schema.signals().front().kind == SignalKind::feature);
    return 0;
}

} // namespace

int main()
{
    if (const auto status = test_descriptor_and_timing(); status != 0)
    {
        return status;
    }
    if (const auto status = test_offline_parity_and_chunk_invariance(); status != 0)
    {
        return status;
    }
    if (const auto status = test_reset_and_discontinuity_barriers(); status != 0)
    {
        return status;
    }
    if (const auto status = test_validation_and_no_allocation(); status != 0)
    {
        return status;
    }
    return test_sos_resampler_lmp_chain();
}
