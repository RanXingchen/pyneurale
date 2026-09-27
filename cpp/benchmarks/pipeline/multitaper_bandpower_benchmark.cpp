/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "allocation_tracker.h"
#include "multitaper_bandpower_adapter.h"

#include <neurale/runtime/runtime_info.h>
#include <neurale/streaming/buffer_pool.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{

using namespace neurale::streaming;
using Clock = std::chrono::steady_clock;

constexpr std::size_t channels = 4;
constexpr std::size_t samples = 256;

struct Options
{
    std::size_t warmups{100};
    std::size_t repetitions{1'000};
};

class ReleasingEmitter final : public FrameEmitter
{
  public:
    explicit ReleasingEmitter(const StreamSchema& schema)
        : pool_(1, schema.signals().front().max_block_bytes, 1)
    {
    }

    StreamStatus try_acquire_frame(MutableFrame*& frame) noexcept override
    {
        frame = nullptr;
        if (acquired_)
        {
            return StreamStatus::invalid_state;
        }
        const auto status = pool_.try_acquire(acquired_);
        if (status == StreamStatus::ok)
        {
            frame = &acquired_.frame();
        }
        return status;
    }

    StreamStatus publish_acquired_frame() noexcept override
    {
        return acquired_.reset();
    }

    StreamStatus publish_input() noexcept override
    {
        return StreamStatus::invalid_state;
    }

  private:
    StreamStatus publish_owned(FrameLease lease) noexcept override
    {
        return lease.reset();
    }

    FramePool pool_;
    FrameLease acquired_{};
};

[[nodiscard]] StreamSchema make_schema()
{
    const std::array signals{SignalSchema{
        11,
        SignalDType::float64,
        channels,
        samples,
        samples,
        {2'000, 1},
        41,
        SignalLayout::sample_major,
        DeviceTickTracking::sample_counter,
        PhysicalUnit::volts,
        17,
        19,
        23,
    }};
    return StreamSchema{29, signals};
}

[[nodiscard]] StreamStatus fill_frame(MutableFrame& frame, std::span<const double> values) noexcept
{
    frame.header() = FrameHeader{
        .session_id = 5,
        .sequence = 1,
        .host_received_ns = 100'000'000,
        .source_tick = 5'000,
        .valid_until_ns = 200'000'000,
        .schema_id = 29,
        .source_clock_domain = 41,
        .flags = FrameFlags::source_tick | FrameFlags::valid_until | FrameFlags::source_received,
    };
    frame.block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = 0,
        .device_tick_start = 5'000,
        .payload_offset = 0,
        .payload_byte_count = values.size_bytes(),
        .signal_id = 11,
        .n_samples = samples,
        .clock_sync =
            ClockSyncSnapshot{
                .device_tick_reference = 5'000,
                .host_time_reference_ns = 10'000'000,
                .device_tick_rate = {2'000, 1},
                .uncertainty_ns = 3,
                .clock_domain = 41,
                .generation = 2,
                .flags = ClockSyncFlags::synchronized,
            },
    };
    std::memcpy(frame.payload_storage().data(), values.data(), values.size_bytes());
    return frame.set_used_sizes(1, values.size_bytes());
}

[[nodiscard]] std::size_t percentile_index(std::size_t size, std::size_t numerator) noexcept
{
    return ((size - 1) * numerator + 9'999) / 10'000;
}

void emit_json(const Options& options, std::vector<std::uint64_t>& durations,
               std::uint64_t allocations)
{
    std::sort(durations.begin(), durations.end());
    const auto build = neurale::runtime::build_info();
    const auto median = durations[percentile_index(durations.size(), 5'000)];
    const auto p95 = durations[percentile_index(durations.size(), 9'500)];
    const auto p99 = durations[percentile_index(durations.size(), 9'900)];
    const auto provider =
        build.fft_backend == "mkl" ? std::string_view{"mkl"} : std::string_view{"builtin"};
    std::cout << "{\"schema_version\":1,"
              << "\"benchmark\":\"pipeline_multitaper_bandpower\","
              << "\"metric\":\"process_latency\",\"unit\":\"ns\","
              << "\"samples\":" << durations.size() << ",\"min\":" << durations.front()
              << ",\"median\":" << median << ",\"p95\":" << p95 << ",\"p99\":" << p99
              << ",\"max\":" << durations.back() << ",\"warmups\":" << options.warmups
              << ",\"repetitions\":" << options.repetitions << ",\"allocations\":" << allocations
              << ",\"allocation_backend\":\"" << neurale::benchmark::allocation_tracking_backend()
              << "\",\"dtype\":\"float64\",\"shape\":[" << samples << ',' << channels
              << "],\"window_samples\":256,\"shift_samples\":128,"
              << "\"fft_length\":256,\"n_tapers\":5,"
              << "\"weighting\":\"adaptive\",\"n_bands\":3,"
              << "\"provider_request\":\"automatic\",\"provider\":\"" << provider
              << "\",\"build_type\":\"" << build.build_type << "\",\"cpu_math_backend\":\""
              << build.cpu_math_backend << "\",\"passed\":" << (allocations == 0 ? "true" : "false")
              << "}\n";
}

[[nodiscard]] Options parse_options(int argc, char** argv)
{
    Options result;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view argument{argv[i]};
        if (argument == "--verify")
        {
            result.warmups = 2;
            result.repetitions = 8;
        }
        else if (argument == "--warmups" && i + 1 < argc)
        {
            result.warmups = std::strtoull(argv[++i], nullptr, 10);
        }
        else if (argument == "--repetitions" && i + 1 < argc)
        {
            result.repetitions = std::strtoull(argv[++i], nullptr, 10);
        }
        else
        {
            throw std::invalid_argument("unknown bandpower benchmark option");
        }
    }
    if (result.repetitions == 0)
    {
        throw std::invalid_argument("bandpower benchmark repetitions must be positive");
    }
    return result;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto options = parse_options(argc, argv);
        auto schema = make_schema();
        neurale::pipeline::MultitaperBandpowerAdapter adapter{{
            .output_schema_id = 37,
            .output_signal_id = 43,
            .feature_set_id = 47,
            .feature_unit_id = 53,
            .bands = {{"low", 0.0, 30.0}, {"mid", 30.0, 100.0}, {"high", 100.0, 500.0}},
            .channel_names = {"c0", "c1", "c2", "c3"},
            .source_stream = "benchmark-voltage",
            .algorithm_version = "1",
            .window_samples = 256,
            .shift_samples = 128,
            .time_bandwidth = 3.5,
            .n_tapers = 5,
            .fft_length = 256,
            .weighting = neurale::signal::MultitaperWeighting::adaptive,
        }};
        const auto contract = adapter.prepare({
            .input_schema = schema,
            .max_process_outputs = 1,
            .max_flush_outputs = 0,
            .available_frame_pool_leases = 1,
        });
        ReleasingEmitter emitter{contract.output_schema};
        FramePool input_pool{1, schema.signals().front().max_block_bytes, 1};
        FrameLease lease;
        if (input_pool.try_acquire(lease) != StreamStatus::ok)
        {
            return 2;
        }
        std::array<double, samples * channels> values{};
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            values[i] = std::sin(static_cast<double>(i) * 0.013);
        }
        if (fill_frame(lease.frame(), values) != StreamStatus::ok)
        {
            return 3;
        }
        for (std::size_t i = 0; i < options.warmups; ++i)
        {
            if (adapter.reset() != StreamStatus::ok ||
                adapter.process(lease.frame(), emitter) != StreamStatus::ok)
            {
                return 4;
            }
        }

        std::vector<std::uint64_t> durations(options.repetitions);
        neurale::benchmark::AllocationScope allocations;
        for (std::size_t i = 0; i < options.repetitions; ++i)
        {
            if (adapter.reset() != StreamStatus::ok)
            {
                return 5;
            }
            const auto start = Clock::now();
            if (adapter.process(lease.frame(), emitter) != StreamStatus::ok)
            {
                return 6;
            }
            durations[i] = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
        }
        allocations.stop();
        const auto count = allocations.count();
        emit_json(options, durations, count);
        return count == 0 ? 0 : 7;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
