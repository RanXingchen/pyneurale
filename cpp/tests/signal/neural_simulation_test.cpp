/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/neural_simulation.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "allocation_counter.h"
#include "check_returns.h"

namespace
{

namespace ns = neurale::signal::simulation;

template <typename Callable> [[nodiscard]] bool rejects(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

[[nodiscard]] ns::NeuralSignalConfig test_config()
{
    ns::NeuralSignalConfig config;
    config.n_channels = 3;
    config.fs = 2'000.0;
    config.units_per_channel = 2;
    config.seed = 73;
    config.spike_duration_seconds = 0.002;
    config.lfp_frequency_hz = 20.0;
    config.nonstationarity_time_constant_seconds = 2.0;
    return config;
}

int run()
{
    {
        auto config = test_config();
        config.n_channels = 0;
        CHECK(rejects([&] { ns::NeuralSignalGenerator generator{config}; }));
        config = test_config();
        config.lfp_frequency_hz = config.fs / 2.0;
        CHECK(rejects([&] { ns::NeuralSignalGenerator generator{config}; }));
        config = test_config();
        config.refractory_seconds = std::numeric_limits<double>::max();
        CHECK(rejects([&] { ns::NeuralSignalGenerator generator{config}; }));
        config = test_config();
        config.drift_per_unit_rotation_std_degrees = 1'000.0;
        config.drift_per_unit_rotation_limit_degrees = 0.1;
        CHECK(rejects([&] { ns::NeuralSignalGenerator generator{config}; }));
    }
    {
        const auto config = test_config();
        constexpr std::size_t sample_count = 257;
        constexpr std::size_t split = 83;
        const ns::Intent2D intent{0.75, -0.25};

        ns::NeuralSignalGenerator whole_generator{config};
        std::vector<double> whole(sample_count * config.n_channels);
        std::vector<std::uint8_t> whole_spikes(sample_count * config.n_channels *
                                               config.units_per_channel);
        CHECK(whole_generator.generate(sample_count, intent, 0.4, whole, whole_spikes) ==
              ns::NeuralGenerationStatus::ok);
        CHECK(whole_generator.sample_index() == sample_count);

        ns::NeuralSignalGenerator chunked_generator{config};
        std::vector<double> first(split * config.n_channels);
        std::vector<double> second((sample_count - split) * config.n_channels);
        std::vector<std::uint8_t> first_spikes(split * config.n_channels *
                                               config.units_per_channel);
        std::vector<std::uint8_t> second_spikes((sample_count - split) * config.n_channels *
                                                config.units_per_channel);
        CHECK(chunked_generator.generate(split, intent, 0.4, first, first_spikes) ==
              ns::NeuralGenerationStatus::ok);
        CHECK(chunked_generator.generate(sample_count - split, intent, 0.4, second,
                                         second_spikes) == ns::NeuralGenerationStatus::ok);
        CHECK(std::equal(first.begin(), first.end(), whole.begin()));
        CHECK(std::equal(second.begin(), second.end(), whole.begin() + first.size()));
        CHECK(std::equal(first_spikes.begin(), first_spikes.end(), whole_spikes.begin()));
        CHECK(std::equal(second_spikes.begin(), second_spikes.end(),
                         whole_spikes.begin() + first_spikes.size()));

        chunked_generator.reset();
        std::vector<double> repeated(sample_count * config.n_channels);
        CHECK(chunked_generator.generate(sample_count, intent, 0.4, repeated) ==
              ns::NeuralGenerationStatus::ok);
        CHECK(repeated == whole);
    }
    {
        const auto config = test_config();
        ns::NeuralSignalGenerator generator{config};
        constexpr std::size_t count = 64;
        std::vector<double> output(count * config.n_channels);
        std::vector<std::uint8_t> spikes(count * config.n_channels * config.units_per_channel);
        CHECK(generator.generate(count, {1.0, 0.0}, 0.0, output, spikes) ==
              ns::NeuralGenerationStatus::ok);
        const auto before = allocations.load(std::memory_order_relaxed);
        CHECK(generator.generate(count, {1.0, 0.0}, 0.0, output, spikes) ==
              ns::NeuralGenerationStatus::ok);
        CHECK(allocations.load(std::memory_order_relaxed) == before);
    }
    {
        auto config = test_config();
        config.drift_rotation_degrees = 0.0;
        config.drift_per_unit_rotation_std_degrees = 30.0;
        config.drift_per_unit_rotation_limit_degrees = 60.0;
        ns::NeuralSignalGenerator first{config};
        ns::NeuralSignalGenerator repeated{config};
        CHECK(first.drift_fingerprint() == repeated.drift_fingerprint());
        ++config.seed;
        ns::NeuralSignalGenerator different{config};
        CHECK(first.drift_fingerprint() != different.drift_fingerprint());
    }
    return 0;
}

} // namespace

int main()
{
    return run();
}
