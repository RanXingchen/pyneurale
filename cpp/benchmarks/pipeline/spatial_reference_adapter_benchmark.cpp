/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

// Per-frame latency of the common-reference adapter, mean against median.
//
// The mean path is a single pass over the reference channels; the median path
// gathers them into a preallocated workspace and runs std::nth_element, whose
// comparison count depends on the data. That difference is the whole reason
// this benchmark exists: the physical-chain benchmark fixes its first stage at
// mean and cannot answer what median costs, and the strict-realtime contract
// test proves median allocates nothing but says nothing about its tail.
//
// Read the max/median ratio, not just the median. If median's tail turns out to
// be unacceptable, the alternative on the table is a data-oblivious selection
// network, and the threshold at which it would win has to come from these
// numbers rather than from a guess.

#include "benchmark_options.h"
#include "spatial_reference_adapter.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <neurale/runtime/runtime_info.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame.h>
#include <neurale/streaming/frame_emitter.h>
#include <neurale/streaming/schema.h>

namespace
{
using neurale::benchmark::parse_size;
using namespace neurale::streaming;

using Clock = std::chrono::steady_clock;

// Four samples per frame is the 1 ms frame of the 4 kHz physical chain, so a
// per-frame number here is comparable with that benchmark's geometry.
constexpr std::size_t kDefaultSamplesPerFrame = 4;
constexpr std::array<std::size_t, 5> kChannelCounts{2, 16, 64, 128, 256};

struct Options
{
    std::size_t samples_per_frame{kDefaultSamplesPerFrame};
    std::size_t warmups{64};
    std::size_t repetitions{2'048};
};

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view argument{argv[i]};
        const auto next = [&]
        {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument("missing value for " + std::string{argument});
            }
            return argv[++i];
        };
        if (argument == "--samples-per-frame")
        {
            options.samples_per_frame = parse_size(next(), "--samples-per-frame must be a count");
        }
        else if (argument == "--warmups")
        {
            options.warmups = parse_size(next(), "--warmups must be a count", true);
        }
        else if (argument == "--repetitions")
        {
            options.repetitions = parse_size(next(), "--repetitions must be a count");
        }
        else
        {
            throw std::invalid_argument("unknown common-reference benchmark option: " +
                                        std::string{argument});
        }
    }
    return options;
}

std::size_t percentile_index(std::size_t count, std::size_t numerator) noexcept
{
    return ((count - 1) * numerator + 9'999) / 10'000;
}

// The adapter forwards its input rather than acquiring an output, so the sink
// only has to accept publish_input(). Counting is all it does: anything heavier
// here would land inside the timed region and measure the sink instead.
class CountingSink final : public FrameEmitter
{
  public:
    void begin(MutableFrame& input) noexcept
    {
        input_ = &input;
    }

    StreamStatus try_acquire_frame(MutableFrame*& frame) noexcept override
    {
        frame = nullptr;
        return StreamStatus::invalid_state;
    }

    StreamStatus publish_acquired_frame() noexcept override
    {
        return StreamStatus::invalid_state;
    }

    StreamStatus publish_input() noexcept override
    {
        if (input_ == nullptr)
        {
            return StreamStatus::invalid_state;
        }
        ++count_;
        return StreamStatus::ok;
    }

    [[nodiscard]] std::size_t count() const noexcept
    {
        return count_;
    }

  private:
    StreamStatus publish_owned(FrameLease) noexcept override
    {
        ++count_;
        return StreamStatus::ok;
    }

    MutableFrame* input_{};
    std::size_t count_{};
};

[[nodiscard]] StreamSchema make_schema(std::size_t channels, std::size_t samples_per_frame)
{
    const std::array signals{
        SignalSchema{11,
                     SignalDType::float64,
                     static_cast<std::uint32_t>(channels),
                     static_cast<std::uint32_t>(samples_per_frame),
                     static_cast<std::uint32_t>(samples_per_frame),
                     {4'000, 1},
                     41,
                     SignalLayout::sample_major,
                     DeviceTickTracking::unavailable,
                     PhysicalUnit::volts,
                     17,
                     19,
                     23},
    };
    return StreamSchema{29, signals};
}

// A shared drift plus a per-channel offset and a channel-dependent phase: the
// reference channels are neither already sorted nor already equal, so the
// selection does real work rather than hitting a degenerate partition.
void fill_payload(std::vector<double>& payload, std::size_t channels, std::size_t samples_per_frame)
{
    payload.resize(channels * samples_per_frame);
    for (std::size_t sample = 0; sample < samples_per_frame; ++sample)
    {
        for (std::size_t channel = 0; channel < channels; ++channel)
        {
            const auto index = static_cast<double>(sample * channels + channel);
            payload[sample * channels + channel] = std::sin(index * 0.37) +
                                                   0.25 * static_cast<double>(channel % 7) +
                                                   0.01 * static_cast<double>(sample);
        }
    }
}

void fill_header(MutableFrame& frame, std::size_t channels, std::size_t samples_per_frame,
                 std::uint64_t sequence, SampleIndex sample_idx_start)
{
    const auto payload_bytes = channels * samples_per_frame * sizeof(double);
    frame.header() = FrameHeader{
        .session_id = 5,
        .sequence = sequence,
        .host_received_ns = 900 + sequence,
        .source_tick = 700 + sequence,
        .valid_until_ns = 1'900 + sequence,
        .schema_id = 29,
        .source_clock_domain = 41,
        .flags = FrameFlags::source_tick | FrameFlags::valid_until | FrameFlags::source_received,
    };
    frame.block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = sample_idx_start,
        .device_tick_start = 0,
        .payload_offset = 0,
        .payload_byte_count = payload_bytes,
        .signal_id = 11,
        .n_samples = static_cast<std::uint32_t>(samples_per_frame),
        .clock_sync = ClockSyncSnapshot{},
    };
    static_cast<void>(frame.set_used_sizes(1, payload_bytes));
}

struct CaseResult
{
    std::uint64_t min{};
    std::uint64_t median{};
    std::uint64_t p95{};
    std::uint64_t p99{};
    std::uint64_t max{};
    std::size_t published_frames{};
};

[[nodiscard]] CaseResult run_case(std::size_t channels, std::span<const std::size_t> subset,
                                  neurale::pipeline::SpatialReferenceStatistic statistic,
                                  const Options& options)
{
    auto schema = make_schema(channels, options.samples_per_frame);
    neurale::pipeline::SpatialReferenceAdapter adapter{subset, statistic};
    static_cast<void>(adapter.prepare({schema, 1, 0, 1}));

    const auto payload_bytes = channels * options.samples_per_frame * sizeof(double);
    FramePool pool{1, payload_bytes, 1};
    FrameLease lease;
    if (pool.try_acquire(lease) != StreamStatus::ok)
    {
        throw std::runtime_error("common-reference benchmark could not acquire a frame");
    }
    std::vector<double> payload;
    fill_payload(payload, channels, options.samples_per_frame);

    CountingSink sink;
    auto sequence = std::uint64_t{1};
    auto sample_idx_start = SampleIndex{0};
    const auto drive = [&]
    {
        // The adapter rewrites the payload in place, so every iteration has to
        // start from the same input; otherwise the values decay toward zero and
        // the selection stops being representative.
        std::memcpy(lease.frame().payload_storage().data(), payload.data(), payload_bytes);
        fill_header(lease.frame(), channels, options.samples_per_frame, sequence, sample_idx_start);
        sink.begin(lease.frame());
        ++sequence;
        sample_idx_start += static_cast<SampleIndex>(options.samples_per_frame);
    };

    for (std::size_t warmup = 0; warmup < options.warmups; ++warmup)
    {
        drive();
        if (adapter.process(lease.frame(), sink) != StreamStatus::ok)
        {
            throw std::runtime_error("common-reference benchmark warm-up failed");
        }
    }

    const auto before_published = sink.count();
    std::vector<std::uint64_t> durations;
    durations.reserve(options.repetitions);
    for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition)
    {
        drive();
        const auto start = Clock::now();
        const auto status = adapter.process(lease.frame(), sink);
        const auto stop = Clock::now();
        if (status != StreamStatus::ok)
        {
            throw std::runtime_error("common-reference benchmark process failed");
        }
        durations.push_back(static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()));
    }
    std::sort(durations.begin(), durations.end());
    return {
        .min = durations.front(),
        .median = durations[percentile_index(durations.size(), 5'000)],
        .p95 = durations[percentile_index(durations.size(), 9'500)],
        .p99 = durations[percentile_index(durations.size(), 9'900)],
        .max = durations.back(),
        .published_frames = sink.count() - before_published,
    };
}

void emit_record(const CaseResult& result, const Options& options, std::size_t channels,
                 std::size_t reference_channels, const char* selection, const char* statistic)
{
    const auto build = neurale::runtime::build_info();
    const auto per_sample =
        static_cast<double>(result.median) / static_cast<double>(options.samples_per_frame);
    const auto tail_ratio =
        result.median > 0 ? static_cast<double>(result.max) / static_cast<double>(result.median)
                          : 0.0;
    std::cout << std::setprecision(17) << "{\"schema_version\":1,"
              << "\"benchmark\":\"pipeline_spatial_reference_adapter\","
              << "\"statistic\":\"" << statistic << "\","
              << "\"reference_selection\":\"" << selection << "\","
              << "\"metric\":\"per_frame_process_latency\",\"unit\":\"ns\","
              << "\"channels\":" << channels << ",\"reference_channels\":" << reference_channels
              << ",\"samples_per_frame\":" << options.samples_per_frame << ",\"min\":" << result.min
              << ",\"median\":" << result.median << ",\"p95\":" << result.p95
              << ",\"p99\":" << result.p99 << ",\"max\":" << result.max
              << ",\"per_sample_median_ns\":" << per_sample << ",\"max_over_median\":" << tail_ratio
              << ",\"warmups\":" << options.warmups << ",\"repetitions\":" << options.repetitions
              << ",\"published_frames\":" << result.published_frames
              << ",\"provider\":\"native_cpu\",\"dtype\":\"float64\","
              << "\"allocation_tracking\":\"not_measured\","
              << "\"build_type\":\"" << build.build_type << "\",\"cpu_math_backend\":\""
              << build.cpu_math_backend << "\"}\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto options = parse_options(argc, argv);
        using neurale::pipeline::SpatialReferenceStatistic;
        for (const auto channels : kChannelCounts)
        {
            // Every other channel: the subset path pays an index indirection
            // per term that the whole-set path does not.
            std::vector<std::size_t> half;
            half.reserve(channels / 2);
            for (std::size_t channel = 0; channel < channels; channel += 2)
            {
                half.push_back(channel);
            }
            for (const auto statistic :
                 {SpatialReferenceStatistic::mean, SpatialReferenceStatistic::median})
            {
                const auto* name =
                    statistic == SpatialReferenceStatistic::median ? "median" : "mean";
                emit_record(run_case(channels, std::span<const std::size_t>{}, statistic, options),
                            options, channels, channels, "all_channels", name);
                emit_record(run_case(channels, half, statistic, options), options, channels,
                            half.size(), "every_other_channel", name);
            }
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
