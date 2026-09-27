/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "feature_adapter_test_support.h"
#include "feature_stack_adapter.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <neurale/features/online.h>
#include <neurale/signal/iir.h>
#include <neurale/signal/windows.h>
#include <neurale/streaming/buffer_pool.h>

void* operator new(std::size_t size, const std::nothrow_t&) noexcept
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}

namespace
{

using namespace neurale::streaming;
using namespace neurale::pipeline::test;

[[nodiscard]] StreamSchema make_schema()
{
    std::array signals{
        SignalSchema{11, SignalDType::float64, channels, 6, max_input_samples,
                     RationalRate{1'000, 1}, 41, SignalLayout::sample_major,
                     DeviceTickTracking::sample_counter, PhysicalUnit::volts, 17, 19, 23},
    };
    signals.front().channel_names = {"left", "right"};
    return StreamSchema{29, signals};
}

[[nodiscard]] neurale::pipeline::FeatureStackAdapterConfig stack_config()
{
    neurale::pipeline::LmpBranchConfig lmp{
        .cutoff_hz = 4.0,
        .filter_order = 2,
    };
    neurale::pipeline::MultitaperBandpowerBranchConfig pmtm{
        .bands = {{.name = "wide", .low_hz = 0.0, .high_hz = 100.0}},
        .time_bandwidth = 2.5,
        .weighting = neurale::signal::MultitaperWeighting::adaptive,
    };
    neurale::pipeline::HilbertEnvelopeBranchConfig hilbert{
        .bands = {{.name = "alpha", .low_hz = 8.0, .high_hz = 12.0}},
        .filter_order = 2,
    };
    return {
        .algorithm_version = "test-stack",
        .window_ns = 8'000'000,
        .update_interval_ns = 4'000'000,
        .branches = {std::move(lmp), std::move(pmtm), std::move(hilbert)},
    };
}

[[nodiscard]] int test_descriptor_and_order()
{
    auto schema = make_schema();
    neurale::pipeline::FeatureStackAdapter adapter{stack_config()};
    const auto contract = adapter.prepare(context(schema));
    const auto& output = contract.output_schema.signals().front();
    CHECK(output.kind == SignalKind::feature);
    CHECK(output.id == 12);
    CHECK(output.feature_set_id == 1);
    CHECK(output.n_channels == 6);
    CHECK(output.fs.numerator == 250);
    CHECK(output.fs.denominator == 1);
    const auto* descriptor = contract.output_schema.feature_sets().find(1);
    CHECK(descriptor != nullptr);
    CHECK(descriptor->feature_names ==
          std::vector<std::string>({"lmp:left", "lmp:right", "pmtm:wide:left", "pmtm:wide:right",
                                    "hilbert:alpha:left", "hilbert:alpha:right"}));
    CHECK(descriptor->unit_ids == std::vector<UnitId>({1, 1, 2, 2, 1, 1}));
    CHECK(descriptor->algorithm_name == "feature-stack");
    CHECK(descriptor->algorithm_version == "test-stack");
    CHECK(descriptor->window_length_ns == 8'000'000);
    CHECK(descriptor->shift_ns == 4'000'000);
    CHECK(contract.max_process_outputs_per_input == 1);
    CHECK(contract.max_flush_outputs == 0);
    CHECK(!contract.can_forward_input);
    return 0;
}

[[nodiscard]] bool prepare_rejects(neurale::pipeline::FeatureStackAdapterConfig config)
{
    try
    {
        auto schema = make_schema();
        neurale::pipeline::FeatureStackAdapter adapter{std::move(config)};
        static_cast<void>(adapter.prepare(context(schema)));
    }
    catch (const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

[[nodiscard]] int test_channel_name_fallback()
{
    const std::array signals{
        SignalSchema{11, SignalDType::float64, channels, 6, max_input_samples,
                     RationalRate{1'000, 1}, 41, SignalLayout::sample_major,
                     DeviceTickTracking::sample_counter, PhysicalUnit::volts, 17, 19, 23},
    };
    auto schema = StreamSchema{29, signals};
    neurale::pipeline::FeatureStackAdapter adapter{stack_config()};
    const auto contract = adapter.prepare(context(schema));
    const auto* descriptor = contract.output_schema.feature_sets().find(1);
    CHECK(descriptor != nullptr);
    CHECK(descriptor->feature_names.front() == "lmp:channel_0");
    return 0;
}

[[nodiscard]] int test_lmp_unit_is_derived_from_input()
{
    auto config = stack_config();
    config.branches = {neurale::pipeline::LmpBranchConfig{
        .cutoff_hz = 4.0,
        .filter_order = 2,
    }};
    auto schema = make_schema();
    neurale::pipeline::FeatureStackAdapter adapter{std::move(config)};
    const auto contract = adapter.prepare(context(schema));
    const auto* descriptor = contract.output_schema.feature_sets().find(1);
    CHECK(descriptor != nullptr);
    CHECK(descriptor->unit_ids == std::vector<UnitId>({1, 1}));
    CHECK(contract.output_schema.units().find(1)->symbol == "V");
    return 0;
}

[[nodiscard]] int test_filter_design_validation()
{
    auto lmp_at_nyquist = stack_config();
    std::get<neurale::pipeline::LmpBranchConfig>(lmp_at_nyquist.branches[0]).cutoff_hz = 500.0;
    CHECK(prepare_rejects(std::move(lmp_at_nyquist)));

    auto hilbert_above_nyquist = stack_config();
    std::get<neurale::pipeline::HilbertEnvelopeBranchConfig>(hilbert_above_nyquist.branches[2])
        .bands = {{.name = "invalid", .low_hz = 400.0, .high_hz = 600.0}};
    CHECK(prepare_rejects(std::move(hilbert_above_nyquist)));

    auto elliptic = stack_config();
    auto& lmp = std::get<neurale::pipeline::LmpBranchConfig>(elliptic.branches[0]);
    lmp.filter_kind = "elliptic";
    lmp.passband_ripple_db = 0.5;
    lmp.stopband_attenuation_db = 30.0;
    auto schema = make_schema();
    neurale::pipeline::FeatureStackAdapter adapter{std::move(elliptic)};
    static_cast<void>(adapter.prepare(context(schema)));
    return 0;
}

[[nodiscard]] int test_numerical_parity_chunking_and_allocation()
{
    auto schema = make_schema();
    neurale::pipeline::FeatureStackAdapter adapter{stack_config()};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);

    std::array<double, 24> input{};
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = std::sin(static_cast<double>(i) * 0.17) + static_cast<double>(i % channels);
    }

    std::array<double, 32> tapers{};
    std::array<double, 4> concentrations{};
    neurale::signal::multitap(8, 2.5, 4, tapers, concentrations);
    constexpr std::array<std::size_t, 2> band_bins{0, 1};
    constexpr std::array lmp_cutoff{4.0};
    const auto lmp_sos =
        neurale::signal::iir_sos("butterworth", 2, lmp_cutoff, "lowpass", 1'000.0, 0.5, 30.0);
    neurale::features::LmpProcessor lmp{8, 4, channels, lmp_sos, lmp_sos.size() / 6};
    neurale::features::BandpowerProcessor pmtm{8,
                                               4,
                                               channels,
                                               8,
                                               1'000.0,
                                               tapers,
                                               4,
                                               concentrations,
                                               neurale::signal::MultitaperWeighting::adaptive,
                                               band_bins,
                                               125.0,
                                               neurale::features::Detrend::none,
                                               neurale::signal::SpectralBackend::automatic};
    constexpr std::array hilbert_cutoff{8.0, 12.0};
    const auto hilbert_sos =
        neurale::signal::iir_sos("butterworth", 2, hilbert_cutoff, "bandpass", 1'000.0, 0.5, 30.0);
    neurale::features::HilbertEnvelopeProcessor hilbert{
        8,        4,
        channels, hilbert_sos,
        1,        hilbert_sos.size() / 6,
        8,        neurale::signal::SpectralBackend::automatic};

    std::vector<double> expected;
    const std::array<std::size_t, 2> chunks{8, 4};
    std::size_t sample_offset{};
    for (std::size_t frame_index = 0; frame_index < chunks.size(); ++frame_index)
    {
        const auto chunk = chunks[frame_index];
        const auto values =
            std::span<const double>{input}.subspan(sample_offset * channels, chunk * channels);
        const auto rows = lmp.output_count(chunk);
        CHECK(pmtm.output_count(chunk) == rows);
        CHECK(hilbert.output_count(chunk) == rows);
        std::vector<double> lmp_values(rows * channels);
        std::vector<double> pmtm_values(rows * channels);
        std::vector<double> hilbert_values(rows * channels);
        lmp.process(values, chunk, lmp_values);
        pmtm.process(values, chunk, pmtm_values);
        hilbert.process(values, chunk, hilbert_values);
        for (std::size_t row = 0; row < rows; ++row)
        {
            for (const auto* branch : {&lmp_values, &pmtm_values, &hilbert_values})
            {
                expected.insert(expected.end(), branch->begin() + row * channels,
                                branch->begin() + (row + 1) * channels);
            }
        }

        CHECK(fill_frame(lease.frame(), schema, values, frame_index + 1, sample_offset) ==
              StreamStatus::ok);
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
        sample_offset += chunk;
    }
    CHECK(nearly_equal(sink.values(), expected));
    CHECK(sink.frame_count() == 2);
    CHECK(sink.block(0).observation_time_start_ns == 14'000'000);
    CHECK(sink.block(1).observation_time_start_ns == 18'000'000);

    CHECK(adapter.reset() == StreamStatus::ok);
    sink.clear();
    CHECK(fill_frame(lease.frame(), schema, std::span<const double>{input}.first(16), 3, 0) ==
          StreamStatus::ok);
    const auto before = allocations.load(std::memory_order_relaxed);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(adapter.flush(sink) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == before);
    return 0;
}

[[nodiscard]] int test_reset_and_discontinuity_barriers()
{
    auto schema = make_schema();
    neurale::pipeline::FeatureStackAdapter adapter{stack_config()};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    const std::array<double, 6> partial{1, 2, 3, 4, 5, 6};
    const std::array<double, 16> complete{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    CHECK(fill_frame(lease.frame(), schema, partial, 1, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.frame_count() == 0);
    CHECK(adapter.reset() == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, complete, 2, 100) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.frame_count() == 1);
    CHECK(sink.block(0).observation_time_start_ns == 114'000'000);

    CHECK(adapter.reset() == StreamStatus::ok);
    sink.clear();
    CHECK(fill_frame(lease.frame(), schema, partial, 3, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    const Discontinuity gap{.session_id = 5,
                            .previous_frame_sequence = 3,
                            .actual_frame_sequence = 7,
                            .reason = GapReason::source_gap};
    CHECK(adapter.handle_discontinuity(gap) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, complete, 7, 200) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.frame_count() == 1);
    CHECK(sink.block(0).observation_time_start_ns == 214'000'000);
    return 0;
}

} // namespace

int main()
{
    if (const auto status = test_descriptor_and_order(); status != 0)
    {
        return status;
    }
    if (const auto status = test_channel_name_fallback(); status != 0)
    {
        return status;
    }
    if (const auto status = test_lmp_unit_is_derived_from_input(); status != 0)
    {
        return status;
    }
    if (const auto status = test_filter_design_validation(); status != 0)
    {
        return status;
    }
    if (const auto status = test_numerical_parity_chunking_and_allocation(); status != 0)
    {
        return status;
    }
    return test_reset_and_discontinuity_barriers();
}
