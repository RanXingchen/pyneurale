/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/runtime/runtime_info.h>
#include <neurale/sorting/valley_seeking.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "benchmark_options.h"

namespace
{
using neurale::benchmark::parse_size;

using Clock = std::chrono::steady_clock;

struct Options
{
    std::size_t observations{2'000};
    std::size_t features{8};
    std::size_t clusters{8};
    std::size_t warmups{3};
    std::size_t repetitions{20};
    std::size_t max_iterations{100};
    double radius{1.0};
};

std::atomic<std::int64_t> benchmark_sink{};

Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view argument{argv[i]};
        if (i + 1 >= argc)
        {
            throw std::invalid_argument("missing Valley Seeking benchmark option value");
        }
        if (argument == "--observations")
        {
            options.observations = parse_size(argv[++i], "observations must be positive");
        }
        else if (argument == "--features")
        {
            options.features = parse_size(argv[++i], "features must be positive");
        }
        else if (argument == "--clusters")
        {
            options.clusters = parse_size(argv[++i], "clusters must be positive");
        }
        else if (argument == "--warmups")
        {
            options.warmups = parse_size(argv[++i], "warmups must be a non-negative integer", true);
        }
        else if (argument == "--repetitions")
        {
            options.repetitions = parse_size(argv[++i], "repetitions must be positive");
        }
        else if (argument == "--max-iterations")
        {
            options.max_iterations = parse_size(argv[++i], "max-iterations must be positive");
        }
        else if (argument == "--radius")
        {
            char* end = nullptr;
            options.radius = std::strtod(argv[++i], &end);
            if (end == argv[i] || *end != '\0' || !std::isfinite(options.radius) ||
                options.radius <= 0.0)
            {
                throw std::invalid_argument("radius must be finite and positive");
            }
        }
        else
        {
            throw std::invalid_argument("unknown Valley Seeking benchmark option");
        }
    }
    if (options.clusters > options.observations)
    {
        throw std::invalid_argument("clusters must not exceed observations");
    }
    if (options.observations > neurale::sorting::valley_seeking_max_observations ||
        options.features > neurale::sorting::valley_seeking_max_features ||
        options.clusters > neurale::sorting::valley_seeking_max_labels)
    {
        throw std::invalid_argument("benchmark shape exceeds native Valley Seeking limits");
    }
    return options;
}

std::size_t percentile_index(std::size_t count, std::size_t numerator) noexcept
{
    return ((count - 1) * numerator + 9'999) / 10'000;
}

void make_input(const Options& options, std::vector<double>& features,
                std::vector<std::int64_t>& labels)
{
    std::mt19937_64 random{0x56'53'45'45'4bULL};
    std::normal_distribution<double> noise{0.0, 0.15};
    features.resize(options.observations * options.features);
    labels.resize(options.observations);
    for (std::size_t observation = 0; observation < options.observations; ++observation)
    {
        const auto cluster = observation % options.clusters;
        for (std::size_t feature = 0; feature < options.features; ++feature)
        {
            const auto center = feature == 0 ? static_cast<double>(cluster) * 3.0 : 0.0;
            features[observation * options.features + feature] = center + noise(random);
        }
        const auto assigned = observation % 11 == 0 ? (cluster + 1) % options.clusters : cluster;
        labels[observation] = static_cast<std::int64_t>(assigned);
    }
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto options = parse_options(argc, argv);
        std::vector<double> features;
        std::vector<std::int64_t> labels;
        make_input(options, features, labels);

        neurale::sorting::ValleySeekingResult last;
        for (std::size_t warmup = 0; warmup < options.warmups; ++warmup)
        {
            last =
                neurale::sorting::valley_seeking(features, options.observations, options.features,
                                                 labels, options.radius, options.max_iterations);
            benchmark_sink.fetch_xor(last.labels.front(), std::memory_order_relaxed);
        }

        std::vector<std::uint64_t> durations;
        durations.reserve(options.repetitions);
        for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition)
        {
            const auto start = Clock::now();
            last =
                neurale::sorting::valley_seeking(features, options.observations, options.features,
                                                 labels, options.radius, options.max_iterations);
            const auto stop = Clock::now();
            durations.push_back(static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count()));
            benchmark_sink.fetch_xor(last.labels.front(), std::memory_order_relaxed);
        }
        std::sort(durations.begin(), durations.end());
        const auto p50 = durations[percentile_index(durations.size(), 5'000)];
        const auto p95 = durations[percentile_index(durations.size(), 9'500)];
        const auto p99 = durations[percentile_index(durations.size(), 9'900)];
        const auto throughput =
            static_cast<double>(options.observations) * 1.0e9 / static_cast<double>(p50);
        const auto build = neurale::runtime::build_info();

        std::cout << std::setprecision(17)
                  << "{\"schema_version\":1,\"benchmark\":\"sorting_valley_seeking\","
                  << "\"metric\":\"whole_kernel_latency\",\"unit\":\"ns\","
                  << "\"samples\":" << durations.size() << ",\"min\":" << durations.front()
                  << ",\"median\":" << p50 << ",\"p95\":" << p95 << ",\"p99\":" << p99
                  << ",\"max\":" << durations.back() << ",\"warmups\":" << options.warmups
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"observations\":" << options.observations
                  << ",\"features\":" << options.features << ",\"clusters\":" << options.clusters
                  << ",\"radius\":" << options.radius
                  << ",\"max_iterations\":" << options.max_iterations
                  << ",\"resolved_iterations\":" << last.iterations
                  << ",\"converged\":" << (last.converged ? "true" : "false")
                  << ",\"neighbor_pairs\":" << last.neighbor_pairs
                  << ",\"workspace_bytes\":" << last.workspace_bytes << ",\"max_neighbor_pairs\":"
                  << neurale::sorting::valley_seeking_max_neighbor_pairs
                  << ",\"throughput_observations_per_second\":" << throughput
                  << ",\"provider\":\"native_cpu\",\"dtype\":\"float64\","
                  << "\"layout\":\"observation_major\",\"build_type\":\"" << build.build_type
                  << "\",\"cpu_math_backend\":\"" << build.cpu_math_backend << "\"}\n";
        return last.converged ? 0 : 2;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Valley Seeking benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
