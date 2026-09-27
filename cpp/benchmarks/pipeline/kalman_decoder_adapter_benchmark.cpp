/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

// Per-frame latency and per-observation throughput of the native Kalman
// decoder adapter, at a geometry a cursor decoder actually runs: a few hundred
// selected features driving a handful of decoded target channels. The dominant
// per-observation cost is the factorization of the observation-dimension
// innovation covariance (cubic in the number of selected features), with the
// state-dimension covariance propagation (cubic in the state dimension) a
// smaller term when the observation dimension is much larger than the state
// dimension. The whole cost of a frame is paid once per observation in it, so
// the block size is a regime rather than a detail, and each is reported
// separately.
//
// One JSON Lines record per regime, with p50/p95/p99 alongside the environment
// that produced them: a latency number without the build type, the math
// backend, and the host is not a measurement anyone can act on.

#include "benchmark_options.h"
#include "kalman_decoder_adapter.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <neurale/models/state_space.h>
#include <neurale/runtime/runtime_info.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame.h>
#include <neurale/streaming/frame_emitter.h>
#include <neurale/streaming/schema.h>

namespace
{
using neurale::benchmark::parse_size;

using Clock = std::chrono::steady_clock;
using namespace neurale::streaming;

constexpr std::uint64_t kShiftNs = 20'000'000; // 50 Hz observations

struct Options
{
    std::size_t features{128};
    std::size_t selected{96};
    std::size_t state{3};
    std::size_t max_observations{32};
    std::size_t warmups{64};
    std::size_t repetitions{1'024};
};

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view argument{argv[i]};
        if (i + 1 >= argc)
        {
            throw std::invalid_argument("missing Kalman decoder benchmark option value");
        }
        const auto* next = argv[++i];
        if (argument == "--features")
        {
            options.features = parse_size(next, "features must be positive");
        }
        else if (argument == "--selected")
        {
            options.selected = parse_size(next, "selected must be positive");
        }
        else if (argument == "--state")
        {
            options.state = parse_size(next, "state must be positive");
        }
        else if (argument == "--max-observations")
        {
            options.max_observations = parse_size(next, "max-observations must be positive");
        }
        else if (argument == "--warmups")
        {
            options.warmups = parse_size(next, "warmups must be positive");
        }
        else if (argument == "--repetitions")
        {
            options.repetitions = parse_size(next, "repetitions must be positive");
        }
        else
        {
            throw std::invalid_argument("unknown Kalman decoder benchmark option");
        }
    }
    if (options.selected > options.features)
    {
        throw std::invalid_argument("selected must not exceed features");
    }
    return options;
}

std::size_t percentile_index(std::size_t count, std::size_t numerator) noexcept
{
    return ((count - 1) * numerator + 9'999) / 10'000;
}

/// A well-conditioned fitted model of the requested geometry.
///
/// The point is a realistic *shape* and a numerically healthy recursion, not a
/// realistic brain: the gain solve has to succeed on every step or the timings
/// would measure the fault path.
neurale::models::LinearGaussianModelState make_model(const Options& options)
{
    const auto k = options.state;
    const auto m = options.selected;
    neurale::models::LinearGaussianModelState state;
    state.state_dim = k;
    state.observation_dim = m;
    state.transition.assign(k * k, 0.0);
    state.process_covariance.assign(k * k, 0.0);
    state.initial_covariance.assign(k * k, 0.0);
    for (std::size_t row = 0; row < k; ++row)
    {
        state.transition[row * k + row] = 0.95;
        state.process_covariance[row * k + row] = 0.02;
        state.initial_covariance[row * k + row] = 0.05;
        if (row + 1 < k)
        {
            state.transition[row * k + row + 1] = 0.03;
        }
    }
    state.transition_offset.assign(k, 0.01);
    state.initial_state.assign(k, 0.0);
    state.observation.assign(m * k, 0.0);
    state.observation_offset.assign(m, 0.0);
    state.observation_covariance.assign(m * m, 0.0);
    std::mt19937_64 engine{20260804};
    std::uniform_real_distribution<double> weights{-1.0, 1.0};
    for (std::size_t row = 0; row < m; ++row)
    {
        for (std::size_t col = 0; col < k; ++col)
        {
            state.observation[row * k + col] = weights(engine);
        }
        state.observation_covariance[row * m + row] = 0.25;
    }
    return state;
}

class BenchmarkSink final : public FrameEmitter
{
  public:
    explicit BenchmarkSink(std::size_t payload_bytes) : pool_(1, payload_bytes, 1) {}

    [[nodiscard]] StreamStatus try_acquire_frame(MutableFrame*& frame) noexcept override
    {
        const auto status = pool_.try_acquire(lease_);
        frame = status == StreamStatus::ok ? &lease_.frame() : nullptr;
        return status;
    }

    [[nodiscard]] StreamStatus publish_acquired_frame() noexcept override
    {
        ++count_;
        return lease_.reset();
    }

    [[nodiscard]] StreamStatus publish_input() noexcept override
    {
        return StreamStatus::ok;
    }

    [[nodiscard]] std::size_t count() const noexcept
    {
        return count_;
    }

  private:
    [[nodiscard]] StreamStatus publish_owned(FrameLease lease) noexcept override
    {
        return lease.reset();
    }

    FramePool pool_;
    FrameLease lease_{};
    std::size_t count_{};
};

void fill_frame(MutableFrame& frame, const Options& options, std::size_t observations,
                std::uint64_t sequence, SampleIndex observation_start)
{
    frame.header() = FrameHeader{
        .session_id = 1,
        .sequence = sequence,
        .host_received_ns = 0,
        .source_tick = 0,
        .valid_until_ns = 0,
        .schema_id = 29,
        .source_clock_domain = 41,
        .flags = FrameFlags::none,
    };
    const auto payload_bytes = observations * options.features * sizeof(double);
    frame.block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = observation_start,
        .device_tick_start = 0,
        .observation_time_start_ns = observation_start * kShiftNs,
        .payload_offset = 0,
        .payload_byte_count = payload_bytes,
        .signal_id = 11,
        .n_samples = static_cast<std::uint32_t>(observations),
        .clock_sync =
            ClockSyncSnapshot{
                .device_tick_reference = 0,
                .host_time_reference_ns = 0,
                .device_tick_rate = {50, 1},
                .uncertainty_ns = 0,
                .clock_domain = 41,
                .generation = 1,
                .flags = ClockSyncFlags::synchronized,
            },
    };
    static_cast<void>(frame.set_used_sizes(1, payload_bytes));
}

struct RegimeResult
{
    std::uint64_t min{};
    std::uint64_t median{};
    std::uint64_t p95{};
    std::uint64_t p99{};
    std::uint64_t max{};
    std::size_t published_frames{};
};

RegimeResult run_regime(neurale::pipeline::KalmanDecoderAdapter& adapter, FrameLease& input,
                        BenchmarkSink& sink, const Options& options, std::size_t observations)
{
    static_cast<void>(adapter.reset());
    SampleIndex observation_start = 0;
    std::uint64_t sequence = 1;
    for (std::size_t warmup = 0; warmup < options.warmups; ++warmup)
    {
        fill_frame(input.frame(), options, observations, sequence++, observation_start);
        if (adapter.process(input.frame(), sink) != StreamStatus::ok)
        {
            throw std::runtime_error("adapter process failed during warmup");
        }
        observation_start += observations;
    }

    const auto before_published = sink.count();
    std::vector<std::uint64_t> durations;
    durations.reserve(options.repetitions);
    for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition)
    {
        fill_frame(input.frame(), options, observations, sequence++, observation_start);
        const auto start = Clock::now();
        const auto status = adapter.process(input.frame(), sink);
        const auto stop = Clock::now();
        if (status != StreamStatus::ok)
        {
            throw std::runtime_error("adapter process failed during measurement");
        }
        durations.push_back(static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()));
        observation_start += observations;
    }
    std::sort(durations.begin(), durations.end());
    RegimeResult result;
    result.min = durations.front();
    result.median = durations[percentile_index(durations.size(), 5'000)];
    result.p95 = durations[percentile_index(durations.size(), 9'500)];
    result.p99 = durations[percentile_index(durations.size(), 9'900)];
    result.max = durations.back();
    result.published_frames = sink.count() - before_published;
    return result;
}

// A field the runtime could not determine is reported as "unknown" rather than
// omitted, so every record carries the same keys.
std::string or_unknown(const std::optional<std::string>& value)
{
    return value.value_or("unknown");
}

void emit_record(const RegimeResult& result, const Options& options, std::size_t observations)
{
    const auto build = neurale::runtime::build_info();
    const auto cpu = neurale::runtime::cpu_info();
    const auto threading = neurale::runtime::threading_info();
    const auto per_observation =
        static_cast<double>(result.median) / static_cast<double>(observations);
    const auto throughput = result.median > 0 ? 1.0e9 * static_cast<double>(observations) /
                                                    static_cast<double>(result.median)
                                              : 0.0;
    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"benchmark\":\"pipeline_kalman_decoder_adapter\","
              << "\"regime\":\"" << observations << "_observations_per_frame\","
              << "\"metric\":\"per_frame_process_latency\",\"unit\":\"ns\","
              << "\"samples\":" << options.repetitions << ",\"min\":" << result.min
              << ",\"median\":" << result.median << ",\"p95\":" << result.p95
              << ",\"p99\":" << result.p99 << ",\"max\":" << result.max
              << ",\"median_per_observation_ns\":" << per_observation
              << ",\"throughput_observations_per_second\":" << throughput
              << ",\"warmups\":" << options.warmups << ",\"repetitions\":" << options.repetitions
              << ",\"features\":" << options.features
              << ",\"selected_features\":" << options.selected << ",\"state_dim\":" << options.state
              << ",\"observations_per_frame\":" << observations
              << ",\"published_frames\":" << result.published_frames
              << ",\"allocation_tracking\":\"not_measured\""
              << ",\"provider\":\"native_cpu\",\"dtype\":\"float64\""
              << ",\"version\":\"" << build.version << "\",\"compiler\":\"" << build.compiler
              << "\",\"build_type\":\"" << build.build_type << "\",\"cpu_math_backend\":\""
              << build.cpu_math_backend << "\",\"threading_backend\":\"" << threading.backend
              << "\",\"threads\":" << threading.num_threads << ",\"cpu_model\":\""
              << or_unknown(cpu.model) << "\",\"logical_cores\":" << cpu.logical_cores << "}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto options = parse_options(argc, argv);

        std::vector<FeatureSetDescriptor> descriptors{FeatureSetDescriptor{
            .id = 47,
            .feature_names = {},
            .unit_ids = std::vector<UnitId>(options.features, 53),
            .source_stream_id = 11,
            .source_stream = "units",
            .algorithm_name = "rate",
            .algorithm_version = "1",
            .window_length_ns = 2 * kShiftNs,
            .shift_ns = kShiftNs,
            .timestamp_reference = FeatureTimestampReference::window_center,
        }};
        descriptors.front().feature_names.reserve(options.features);
        for (std::size_t i = 0; i < options.features; ++i)
        {
            descriptors.front().feature_names.push_back("rate_" + std::to_string(i));
        }
        const std::vector<UnitDescriptor> units{
            UnitDescriptor{.id = 53, .symbol = "Hz", .description = "hertz"}};
        const std::array signals{SignalSchema{
            11,
            SignalDType::float64,
            static_cast<std::uint32_t>(options.features),
            static_cast<std::uint32_t>(options.max_observations),
            static_cast<std::uint32_t>(options.max_observations),
            {50, 1},
            41,
            SignalLayout::sample_major,
            DeviceTickTracking::unavailable,
            PhysicalUnit::unspecified,
            17,
            19,
            23,
            SignalKind::feature,
            47,
            ObservationTiming::regular,
        }};
        const StreamSchema input_schema{29, signals, descriptors, units};

        // The selection is the first ``options.selected`` features, so the
        // gather is a contiguous prefix -- not an identity, but simple enough to
        // keep the benchmark about the filter recursion rather than the gather.
        neurale::pipeline::KalmanDecoderAdapterConfig config{
            .output_schema_id = 37,
            .output_signal_id = 43,
            .output_channel_set_id = 59,
            .output_physical_unit = PhysicalUnit::dimensionless,
            .feature_set_id = 47,
            .selection = {},
            .selected_feature_names = {},
            .fitted_feature_contract =
                neurale::pipeline::FittedFeatureContract{
                    .feature_names = descriptors.front().feature_names,
                    .feature_unit_symbols = std::vector<std::string>(options.features, "Hz"),
                    .observation_rate = {50, 1},
                    .window_length_ns = 2 * kShiftNs,
                    .shift_ns = kShiftNs,
                    .algorithm_name = "rate",
                    .algorithm_version = "1",
                    .source_stream = "units",
                    .timestamp_reference = FeatureTimestampReference::window_center,
                },
            .scaling = neurale::pipeline::FeatureScaling::standard,
            .scaler_center = std::vector<double>(options.selected, 0.5),
            .scaler_scale = std::vector<double>(options.selected, 2.0),
            .model = make_model(options),
            .innovation_jitter = 1e-9,
            .missing = neurale::pipeline::KalmanMissingPolicy::error,
        };
        for (std::size_t i = 0; i < options.selected; ++i)
        {
            config.selection.push_back(i);
            config.selected_feature_names.push_back(descriptors.front().feature_names[i]);
        }

        neurale::pipeline::KalmanDecoderAdapter adapter{config};
        const auto contract = adapter.prepare({input_schema, 1, 1, 1});
        const auto output_bytes = contract.output_schema.signals().front().max_block_bytes;

        const auto input_bytes = options.max_observations * options.features * sizeof(double);
        FramePool input_pool{1, input_bytes, 1};
        FrameLease input{};
        if (input_pool.try_acquire(input) != StreamStatus::ok)
        {
            throw std::runtime_error("could not acquire input frame pool lease");
        }
        std::vector<double> payload(options.max_observations * options.features);
        std::mt19937_64 engine{20260805};
        std::normal_distribution<double> noise{0.5, 1.0};
        for (auto& value : payload)
        {
            value = noise(engine);
        }
        std::memcpy(input.frame().payload_storage().data(), payload.data(),
                    payload.size() * sizeof(double));
        BenchmarkSink sink{output_bytes};

        // One record per distinct block size: a geometry whose maximum already
        // is one of the smaller regimes is reported once, not twice.
        std::vector<std::size_t> regimes{1, 8, options.max_observations};
        std::sort(regimes.begin(), regimes.end());
        regimes.erase(std::unique(regimes.begin(), regimes.end()), regimes.end());
        for (const auto observations : regimes)
        {
            if (observations > options.max_observations)
            {
                continue;
            }
            const auto result = run_regime(adapter, input, sink, options, observations);
            emit_record(result, options, observations);
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Kalman decoder adapter benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
