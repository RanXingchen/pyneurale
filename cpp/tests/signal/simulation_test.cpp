/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/simulation.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
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

int run()
{
    {
        const auto generator = ns::SignalGenerator::zeros(2, 1000.0);
        std::array<double, 6> output = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        CHECK(generator.generate(17, 3, output) == ns::GenerationStatus::ok);
        CHECK((output == std::array<double, 6>{}));
    }
    {
        const std::array values = {2.0, -3.0};
        const auto generator = ns::SignalGenerator::constant(2, 500.0, values);
        std::array<double, 4> output{};
        CHECK(generator.generate(9, 2, output) == ns::GenerationStatus::ok);
        CHECK((output == std::array<double, 4>{2.0, -3.0, 2.0, -3.0}));
    }
    {
        const std::array freqs = {10.0, 20.0, 30.0, 40.0};
        const std::array amps = {1.0, 2.0, 0.5, 0.25};
        const std::array phases = {0.0, 0.1, 0.2, 0.3};
        const auto generator = ns::SignalGenerator::tones(2, 200.0, 2, freqs, amps, phases);
        std::array<double, 10> whole{};
        std::array<double, 4> first{};
        std::array<double, 6> second{};
        CHECK(generator.generate(7, 5, whole) == ns::GenerationStatus::ok);
        CHECK(generator.generate(7, 2, first) == ns::GenerationStatus::ok);
        CHECK(generator.generate(9, 3, second) == ns::GenerationStatus::ok);
        CHECK(std::equal(first.begin(), first.end(), whole.begin()));
        CHECK(std::equal(second.begin(), second.end(), whole.begin() + 4));
        const auto expected =
            amps[0] * std::sin(phases[0] + 2.0 * std::acos(-1.0) * freqs[0] * 7.0 / 200.0) +
            amps[2] * std::sin(phases[2] + 2.0 * std::acos(-1.0) * freqs[2] * 7.0 / 200.0);
        CHECK(std::abs(whole[0] - expected) < 1e-15);
    }
    {
        const auto generator = ns::SignalGenerator::noise(3, 1000.0, 42, -2.0, 4.0);
        std::array<double, 12> first{};
        std::array<double, 12> second{};
        CHECK(generator.generate(100, 4, first) == ns::GenerationStatus::ok);
        CHECK(generator.generate(100, 4, second) == ns::GenerationStatus::ok);
        CHECK(first == second);
        CHECK(first[0] != first[1]);
    }
    {
        const std::array source = {1.0, 2.0, 3.0, 4.0};
        const auto finite = ns::SignalGenerator::samples(2, 100.0, 2, source, false);
        const auto repeating = ns::SignalGenerator::samples(2, 100.0, 2, source, true);
        std::array<double, 4> output{};
        CHECK(finite.generate(1, 2, output) == ns::GenerationStatus::source_exhausted);
        CHECK(repeating.generate(1, 2, output) == ns::GenerationStatus::ok);
        CHECK((output == std::array<double, 4>{3.0, 4.0, 1.0, 2.0}));
    }
    {
        const std::array value = {1.0};
        CHECK(rejects([] { static_cast<void>(ns::SignalGenerator::zeros(0, 1.0)); }));
        CHECK(rejects([] { static_cast<void>(ns::SignalGenerator::zeros(1, 0.0)); }));
        CHECK(rejects([&] { static_cast<void>(ns::SignalGenerator::constant(2, 1.0, value)); }));
        CHECK(rejects(
            []
            {
                const std::array freq = {501.0};
                const std::array amp = {1.0};
                const std::array phase = {0.0};
                static_cast<void>(ns::SignalGenerator::tones(1, 1000.0, 1, freq, amp, phase));
            }));
        CHECK(rejects(
            [] { static_cast<void>(ns::SignalGenerator::noise(1, 1000.0, 1, -1e308, 1e308)); }));
        const auto generator = ns::SignalGenerator::constant(1, 1.0, value);
        std::array<double, 2> output{};
        CHECK(generator.generate(0, 1, output) == ns::GenerationStatus::invalid_output);
    }
    {
        const std::array freqs = {17.0, 23.0};
        const std::array amps = {1.0, 2.0};
        const std::array phases = {0.0, 0.5};
        const auto generator = ns::SignalGenerator::tones(2, 1000.0, 1, freqs, amps, phases);
        std::array<double, 128> output{};
        CHECK(generator.generate(0, 64, output) == ns::GenerationStatus::ok);
        const auto before = allocations.load(std::memory_order_relaxed);
        CHECK(generator.generate(64, 64, output) == ns::GenerationStatus::ok);
        CHECK(allocations.load(std::memory_order_relaxed) == before);
    }
    return 0;
}

} // namespace

int main()
{
    return run();
}
