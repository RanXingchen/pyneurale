/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

// Per-frame latency of the online spike-detector adapter across sparse and
// dense regimes. The adapter reconstructs a ``FixedCapacitySpikeBlock`` over
// the output payload on every ``process()`` call, and that constructor clears
// the full fixed-capacity payload with ``memset`` -- a capacity-proportional
// cost that is paid even by an empty spike frame. This benchmark measures the
// real per-frame cost at 0, 1, and full spikes-per-frame so the empty/sparse
// case (dominated by the clear) is not hidden behind burst/full-block evidence.

#include "benchmark_options.h"
#include "spike_detector_adapter.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <neurale/runtime/runtime_info.h>
#include <neurale/sorting/online_detection.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame.h>
#include <neurale/streaming/frame_emitter.h>
#include <neurale/streaming/schema.h>

namespace
{
using neurale::benchmark::parse_size;

using Clock = std::chrono::steady_clock;

struct Options
{
    std::size_t channels{32};
    std::size_t capacity{64};
    std::size_t pre_samples{30};
    std::size_t post_samples{30};
    std::size_t alignment_search_radius{4};
    std::size_t refractory_samples{10};
    std::size_t max_input_samples{256};
    std::size_t warmups{32};
    std::size_t repetitions{512};
};

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view argument{argv[i]};
        if (i + 1 >= argc)
        {
            throw std::invalid_argument("missing spike detector adapter benchmark option value");
        }
        const auto next = argv[++i];
        if (argument == "--channels")
        {
            options.channels = parse_size(next, "channels must be positive");
        }
        else if (argument == "--capacity")
        {
            options.capacity = parse_size(next, "capacity must be positive");
        }
        else if (argument == "--pre-samples")
        {
            options.pre_samples = parse_size(next, "pre-samples must be positive");
        }
        else if (argument == "--post-samples")
        {
            options.post_samples = parse_size(next, "post-samples must be positive");
        }
        else if (argument == "--alignment-search-radius")
        {
            options.alignment_search_radius =
                parse_size(next, "alignment-search-radius must be a non-negative integer", true);
        }
        else if (argument == "--refractory-samples")
        {
            options.refractory_samples =
                parse_size(next, "refractory-samples must be a non-negative integer", true);
        }
        else if (argument == "--max-input-samples")
        {
            options.max_input_samples = parse_size(next, "max-input-samples must be positive");
        }
        else if (argument == "--warmups")
        {
            options.warmups = parse_size(next, "warmups must be a non-negative integer", true);
        }
        else if (argument == "--repetitions")
        {
            options.repetitions = parse_size(next, "repetitions must be positive");
        }
        else
        {
            throw std::invalid_argument("unknown spike detector adapter benchmark option");
        }
    }
    if (options.alignment_search_radius > options.pre_samples)
    {
        throw std::invalid_argument("alignment-search-radius must not exceed pre-samples");
    }
    return options;
}

std::size_t percentile_index(std::size_t count, std::size_t numerator) noexcept
{
    return ((count - 1) * numerator + 9'999) / 10'000;
}

// Minimal output emitter: acquire one frame from a single-slot pool, then
// release it on publish. The benchmark does not inspect the emitted spike
// block; it only drives the adapter through its real acquire/memset/detect/
// publish path.
class BenchmarkSink final : public neurale::streaming::FrameEmitter
{
  public:
    explicit BenchmarkSink(std::size_t payload_bytes) : pool_(1, payload_bytes, 1) {}

    [[nodiscard]] neurale::streaming::StreamStatus
    try_acquire_frame(neurale::streaming::MutableFrame*& frame) noexcept override
    {
        const auto status = pool_.try_acquire(lease_);
        frame = status == neurale::streaming::StreamStatus::ok ? &lease_.frame() : nullptr;
        return status;
    }

    [[nodiscard]] neurale::streaming::StreamStatus publish_acquired_frame() noexcept override
    {
        ++count_;
        return lease_.reset();
    }

    [[nodiscard]] neurale::streaming::StreamStatus publish_input() noexcept override
    {
        return neurale::streaming::StreamStatus::ok;
    }

    [[nodiscard]] std::size_t count() const noexcept
    {
        return count_;
    }

  private:
    [[nodiscard]] neurale::streaming::StreamStatus
    publish_owned(neurale::streaming::FrameLease lease) noexcept override
    {
        return lease.reset();
    }

    neurale::streaming::FramePool pool_;
    neurale::streaming::FrameLease lease_;
    std::size_t count_{};
};

// Build the per-frame input payload pattern that produces ``crossings_per_frame``
// completed spikes. Each onset is a single-sample excursion to
// ``-spike_amp`` surrounded by zeros. Onsets are distributed round-robin
// across channels (one group per channel), so a single channel never holds more
// than ``ceil(crossings / channels)`` onsets; onsets on the same channel are
// spaced ``spacing`` samples apart (>= refractory) and early enough for the
// post-tail to complete within the frame.
void build_payload(std::vector<double>& payload, const Options& options,
                   std::size_t crossings_per_frame)
{
    const auto n = options.max_input_samples;
    payload.assign(n * options.channels, 0.0);
    if (crossings_per_frame == 0)
    {
        return;
    }
    const auto per_channel = (crossings_per_frame + options.channels - 1) / options.channels;
    const auto spacing = n / (per_channel + 1);
    const double spike_amp = -5.0;
    for (std::size_t i = 0; i < crossings_per_frame; ++i)
    {
        const auto channel = i % options.channels;
        const auto onset = i / options.channels;
        const auto offset = spacing * (onset + 1);
        if (offset == 0 || offset + options.alignment_search_radius + options.post_samples >= n)
        {
            throw std::invalid_argument(
                "input frame too small for the requested crossings per frame");
        }
        payload[offset * options.channels + channel] = spike_amp;
    }
}

void fill_header(neurale::streaming::MutableFrame& frame, const Options& options,
                 neurale::streaming::SignalId signal_id, neurale::streaming::SchemaId schema_id,
                 std::uint64_t sequence, neurale::streaming::SampleIndex sample_idx_start)
{
    using namespace neurale::streaming;
    frame.header() = FrameHeader{
        .session_id = 1,
        .sequence = sequence,
        .host_received_ns = 0,
        .source_tick = 0,
        .valid_until_ns = 0,
        .schema_id = schema_id,
        .source_clock_domain = 41,
        .flags = FrameFlags::none,
    };
    frame.block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = sample_idx_start,
        .device_tick_start = static_cast<DeviceTick>(sample_idx_start),
        .payload_offset = 0,
        .payload_byte_count = options.max_input_samples * options.channels * sizeof(double),
        .signal_id = signal_id,
        .n_samples = static_cast<std::uint32_t>(options.max_input_samples),
        .clock_sync =
            ClockSyncSnapshot{
                .device_tick_reference = 0,
                .host_time_reference_ns = 0,
                .device_tick_rate = {1'000, 1},
                .uncertainty_ns = 0,
                .clock_domain = 41,
                .generation = 1,
                .flags = ClockSyncFlags::synchronized,
            },
    };
    static_cast<void>(
        frame.set_used_sizes(1, options.max_input_samples * options.channels * sizeof(double)));
}

struct Regime
{
    const char* name;
    std::size_t crossings_per_frame;
};

struct RegimeResult
{
    std::uint64_t min{};
    std::uint64_t median{};
    std::uint64_t p95{};
    std::uint64_t p99{};
    std::uint64_t max{};
    std::size_t published_frames{};
};

RegimeResult run_regime(neurale::pipeline::SpikeDetectorAdapter& adapter,
                        neurale::streaming::FrameLease& input, BenchmarkSink& sink,
                        const Options& options, neurale::streaming::SignalId signal_id,
                        neurale::streaming::SchemaId schema_id, std::size_t crossings_per_frame,
                        std::uint64_t first_sequence)
{
    std::vector<double> payload;
    build_payload(payload, options, crossings_per_frame);
    std::memcpy(input.frame().payload_storage().data(), payload.data(),
                payload.size() * sizeof(double));

    // Each regime is independent: reset the adapter so the stateful detector
    // re-anchors at sample_idx_start 0 instead of carrying expected_next over
    // from the previous regime.
    static_cast<void>(adapter.reset());

    auto sample_idx_start = static_cast<neurale::streaming::SampleIndex>(0);
    auto sequence = first_sequence;
    for (std::size_t warmup = 0; warmup < options.warmups; ++warmup)
    {
        fill_header(input.frame(), options, signal_id, schema_id, sequence, sample_idx_start);
        static_cast<void>(adapter.process(input.frame(), sink));
        sample_idx_start += static_cast<neurale::streaming::SampleIndex>(options.max_input_samples);
        ++sequence;
    }

    const auto before_published = sink.count();
    std::vector<std::uint64_t> durations;
    durations.reserve(options.repetitions);
    for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition)
    {
        fill_header(input.frame(), options, signal_id, schema_id, sequence, sample_idx_start);
        const auto start = Clock::now();
        const auto status = adapter.process(input.frame(), sink);
        const auto stop = Clock::now();
        if (status != neurale::streaming::StreamStatus::ok)
        {
            throw std::runtime_error("adapter process failed during measurement");
        }
        durations.push_back(static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()));
        sample_idx_start += static_cast<neurale::streaming::SampleIndex>(options.max_input_samples);
        ++sequence;
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

void emit_record(const char* benchmark, const char* regime, const RegimeResult& result,
                 const Options& options, std::size_t crossings_per_frame, std::size_t payload_bytes)
{
    const auto build = neurale::runtime::build_info();
    const auto throughput = result.median > 0 ? 1.0e9 / static_cast<double>(result.median) : 0.0;
    std::cout << std::setprecision(17) << "{\"schema_version\":1,\"benchmark\":\"" << benchmark
              << "\","
              << "\"regime\":\"" << regime << "\","
              << "\"metric\":\"per_frame_process_latency\",\"unit\":\"ns\","
              << "\"samples\":" << options.repetitions << ",\"min\":" << result.min
              << ",\"median\":" << result.median << ",\"p95\":" << result.p95
              << ",\"p99\":" << result.p99 << ",\"max\":" << result.max
              << ",\"warmups\":" << options.warmups << ",\"repetitions\":" << options.repetitions
              << ",\"channels\":" << options.channels << ",\"capacity\":" << options.capacity
              << ",\"waveform_samples\":" << (options.pre_samples + 1 + options.post_samples)
              << ",\"pre_samples\":" << options.pre_samples
              << ",\"post_samples\":" << options.post_samples
              << ",\"alignment_search_radius\":" << options.alignment_search_radius
              << ",\"refractory_samples\":" << options.refractory_samples
              << ",\"max_input_samples\":" << options.max_input_samples
              << ",\"crossings_per_frame\":" << crossings_per_frame
              << ",\"published_frames\":" << result.published_frames
              << ",\"spike_block_payload_bytes\":" << payload_bytes
              << ",\"throughput_frames_per_second\":" << throughput
              << ",\"provider\":\"native_cpu\",\"dtype\":\"float64\","
              << "\"build_type\":\"" << build.build_type << "\",\"cpu_math_backend\":\""
              << build.cpu_math_backend << "\"}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto options = parse_options(argc, argv);
        using namespace neurale::streaming;
        const std::array signals{SignalSchema{11,
                                              SignalDType::float64,
                                              static_cast<std::uint32_t>(options.channels),
                                              static_cast<std::uint32_t>(options.max_input_samples),
                                              static_cast<std::uint32_t>(options.max_input_samples),
                                              {1'000, 1},
                                              41,
                                              SignalLayout::sample_major,
                                              DeviceTickTracking::unavailable,
                                              PhysicalUnit::volts,
                                              17,
                                              19,
                                              23}};
        const StreamSchema input_schema{29, signals};

        std::vector<double> centers(options.channels, 0.0);
        std::vector<double> thresholds(options.channels, 1.0);
        neurale::pipeline::SpikeDetectorAdapterConfig adapter_config{
            .output_schema_id = 37,
            .output_signal_id = 43,
            .block_capacity = options.capacity,
            .refractory_samples = options.refractory_samples,
            .alignment_search_radius = options.alignment_search_radius,
            .pre_samples = options.pre_samples,
            .post_samples = options.post_samples,
            .polarity = neurale::sorting::DetectionPolarity::Negative,
            .boundary_behavior = neurale::sorting::BoundaryBehavior::Drop,
            .overflow_policy = neurale::sorting::SpikeBlockOverflowPolicy::drop_newest,
            .channel_centers = centers,
            .channel_thresholds = thresholds,
        };
        neurale::pipeline::SpikeDetectorAdapter adapter{adapter_config};
        const auto contract = adapter.prepare({input_schema, 1, 1, 1});
        const auto payload_bytes = contract.output_schema.signals().front().max_block_bytes;

        FramePool input_pool{1, options.max_input_samples * options.channels * sizeof(double), 1};
        FrameLease input{};
        if (input_pool.try_acquire(input) != StreamStatus::ok)
        {
            throw std::runtime_error("could not acquire input frame pool lease");
        }
        BenchmarkSink sink{payload_bytes};

        const Regime regimes[] = {
            {"0_spikes_per_frame", 0},
            {"1_spike_per_frame", 1},
            {"full_block_per_frame", options.capacity},
        };
        const auto full = regimes[2].crossings_per_frame;
        if (full == 0 || full > options.capacity)
        {
            throw std::invalid_argument("capacity must be positive for the full-block regime");
        }

        std::uint64_t sequence = 0;
        for (const auto& regime : regimes)
        {
            const auto result = run_regime(adapter, input, sink, options, /*signal_id=*/11,
                                           /*schema_id=*/29, regime.crossings_per_frame, sequence);
            emit_record("pipeline_spike_detector_adapter", regime.name, result, options,
                        regime.crossings_per_frame, payload_bytes);
            sequence += options.warmups + options.repetitions + 1;
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "spike detector adapter benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
