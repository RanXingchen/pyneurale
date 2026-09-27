/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "check_returns.h"
#include "fault_injection.h"

#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/continuity.h>
#include <neurale/streaming/discontinuity_pool.h>
#include <neurale/streaming/runtime.h>
#include <neurale/streaming/spsc_ring.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <thread>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>

#include <psapi.h>
#elif defined(__linux__)
#include <fstream>
#include <unistd.h>
#endif

using namespace neurale::streaming;
using namespace neurale::streaming::test;

namespace
{

struct SoakOptions
{
    std::string_view mode{"ci"};
    std::size_t runtime_cycles{2'000};
    std::uint64_t ring_iterations{2'000'000};
    std::uint64_t pool_iterations{1'000'000};
    std::chrono::seconds duration{};
};

[[nodiscard]] std::uint64_t environment_count(const char* name, std::uint64_t fallback)
{
    const auto* value = std::getenv(name);
    return value == nullptr ? fallback : std::strtoull(value, nullptr, 10);
}

[[nodiscard]] SoakOptions load_options()
{
    SoakOptions options;
    if (const auto* mode = std::getenv("NEURALE_STREAMING_SOAK_MODE"))
    {
        options.mode = mode;
    }
    if (options.mode == "nightly")
    {
        options.runtime_cycles = 20'000;
        options.ring_iterations = 50'000'000;
        options.pool_iterations = 20'000'000;
    }
    else if (options.mode == "soak")
    {
        options.runtime_cycles = 20'000;
        options.ring_iterations = 100'000'000;
        options.pool_iterations = 50'000'000;
        options.duration = std::chrono::seconds{3'600};
    }
    options.runtime_cycles = static_cast<std::size_t>(
        environment_count("NEURALE_STREAMING_SOAK_CYCLES", options.runtime_cycles));
    options.ring_iterations =
        environment_count("NEURALE_STREAMING_SOAK_RING_ITERATIONS", options.ring_iterations);
    options.pool_iterations =
        environment_count("NEURALE_STREAMING_SOAK_POOL_ITERATIONS", options.pool_iterations);
    options.duration = std::chrono::seconds{
        environment_count("NEURALE_STREAMING_SOAK_SECONDS", options.duration.count())};
    return options;
}

[[nodiscard]] std::size_t resident_memory_bytes() noexcept
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                             sizeof(counters)) == 0)
    {
        return 0;
    }
    return static_cast<std::size_t>(counters.WorkingSetSize);
#elif defined(__linux__)
    std::ifstream status{"/proc/self/statm"};
    std::size_t pages{};
    std::size_t resident{};
    if (!(status >> pages >> resident))
    {
        return 0;
    }
    return resident * static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
#else
    return 0;
#endif
}

int test_runtime_start_stop_soak(const SoakOptions& options, std::size_t& memory_growth)
{
    const FaultInjectionConfig injection{FaultPoint::none, 0, 1};
    InjectedSource source{injection};
    InjectedProcessor processor{injection};
    InjectedActuator actuator{injection};
    source.pace_with(actuator.applied_counter());
    FaultSafetyController safety;
    NativeStreamRunner runtime{make_fault_schema(),
                               make_fault_config(),
                               source,
                               processor,
                               actuator,
                               default_native_clock(),
                               safety};

    const auto warmup_cycle = std::min<std::size_t>(100, options.runtime_cycles / 4);
    std::size_t baseline_memory{};
    const auto started = std::chrono::steady_clock::now();
    std::size_t cycles{};
    while (cycles < options.runtime_cycles ||
           (options.duration.count() != 0 &&
            std::chrono::steady_clock::now() - started < options.duration))
    {
        CHECK(runtime.prepare() == StreamStatus::ok);
        CHECK(runtime.arm() == StreamStatus::ok);
        CHECK(runtime.run() == StreamStatus::ok);
        CHECK(runtime.state() == RuntimeState::stopped);
        CHECK(runtime.stats().frames_acquired == 1);
        CHECK(runtime.stats().frames_consumed == 1);
        CHECK(runtime.outstanding_frames() == 0);
        CHECK(runtime.outstanding_discontinuities() == 0);
        CHECK(!runtime.primary_fault().has_value());
        CHECK(runtime.reset() == StreamStatus::ok);
        CHECK(runtime.state() == RuntimeState::created);
        ++cycles;
        if (cycles == warmup_cycle)
        {
            baseline_memory = resident_memory_bytes();
        }
    }

    const auto final_memory = resident_memory_bytes();
    memory_growth = baseline_memory == 0 || final_memory <= baseline_memory
                        ? 0
                        : final_memory - baseline_memory;
#ifdef NEURALE_SANITIZED_BUILD
    constexpr std::size_t allowed_growth = 256ULL * 1024 * 1024;
#else
    constexpr std::size_t allowed_growth = 64ULL * 1024 * 1024;
#endif
    CHECK(memory_growth <= allowed_growth);
    return 0;
}

struct SoakMessage
{
    std::uint64_t sequence{};

    SoakMessage() noexcept = default;
    explicit SoakMessage(std::uint64_t value) noexcept : sequence(value) {}
    SoakMessage(const SoakMessage&) = delete;
    SoakMessage& operator=(const SoakMessage&) = delete;
    SoakMessage(SoakMessage&& other) noexcept
        : sequence(std::exchange(other.sequence, std::numeric_limits<std::uint64_t>::max()))
    {
    }
    SoakMessage& operator=(SoakMessage&& other) noexcept
    {
        sequence = std::exchange(other.sequence, std::numeric_limits<std::uint64_t>::max());
        return *this;
    }
};

int test_long_ring_wrap(std::uint64_t iterations)
{
    SpscRing<SoakMessage> ring{257};
    std::atomic<bool> start{};
    std::atomic<bool> failed{};

    std::thread producer{
        [&]
        {
            start.wait(false, std::memory_order_acquire);
            for (std::uint64_t sequence = 0; sequence < iterations; ++sequence)
            {
                SoakMessage message{sequence};
                while (ring.try_push(std::move(message)) == StreamStatus::queue_overflow)
                {
                    std::this_thread::yield();
                }
            }
            ring.close();
        }};
    std::thread consumer{[&]
                         {
                             start.wait(false, std::memory_order_acquire);
                             SoakMessage message;
                             for (std::uint64_t expected = 0; expected < iterations;)
                             {
                                 const auto status = ring.try_pop(message);
                                 if (status == StreamStatus::ok)
                                 {
                                     if (message.sequence != expected)
                                     {
                                         failed.store(true, std::memory_order_relaxed);
                                         break;
                                     }
                                     ++expected;
                                 }
                                 else if (status != StreamStatus::would_block)
                                 {
                                     failed.store(true, std::memory_order_relaxed);
                                     break;
                                 }
                             }
                         }};
    start.store(true, std::memory_order_release);
    start.notify_all();
    producer.join();
    consumer.join();
    CHECK(!failed.load(std::memory_order_relaxed));
    CHECK(ring.empty());
    return 0;
}

int test_long_slot_generation(std::uint64_t iterations)
{
    FramePool pool{1, 64, 1};
    for (std::uint64_t generation = 0; generation < iterations; ++generation)
    {
        FrameLease lease;
        CHECK(pool.try_acquire(lease) == StreamStatus::ok);
        CHECK(lease.token().slot == 0);
        CHECK(lease.token().generation == generation);
        FrameLease owner{std::move(lease)};
        CHECK(!lease);
        CHECK(owner.reset() == StreamStatus::ok);
        CHECK(pool.outstanding() == 0);
    }
    return 0;
}

int test_frame_sequence_overflow_is_explicitly_fatal()
{
    auto schema = make_fault_schema();
    ContinuityChecker checker{schema};
    FramePool frames{2, 64, 1};
    DiscontinuityPool discontinuities{1, 1};

    FrameLease first;
    CHECK(frames.try_acquire(first) == StreamStatus::ok);
    CHECK(fill_fault_frame(first.frame(), std::numeric_limits<std::uint64_t>::max() - 1, 0) ==
          StreamStatus::ok);
    auto checked = checker.check(std::move(first), discontinuities);
    CHECK(checked.status == ContinuityStatus::continuous);
    checked.messages[0] = StreamMessage{};

    FrameLease overflow;
    CHECK(frames.try_acquire(overflow) == StreamStatus::ok);
    CHECK(fill_fault_frame(overflow.frame(), std::numeric_limits<std::uint64_t>::max(), 4) ==
          StreamStatus::ok);
    checked = checker.check(std::move(overflow), discontinuities);
    CHECK(checked.status == ContinuityStatus::fatal);
    CHECK(checked.error == ContinuityError::frame_sequence_overflow);
    CHECK(frames.outstanding() == 0);
    return 0;
}

} // namespace

int main()
{
    const auto options = load_options();
    if (options.runtime_cycles == 0 || options.ring_iterations == 0 || options.pool_iterations == 0)
    {
        std::cerr << "soak counts must be positive\n";
        return 2;
    }

    std::size_t memory_growth{};
    if (const auto result = test_runtime_start_stop_soak(options, memory_growth); result != 0)
    {
        return result;
    }
    if (const auto result = test_long_ring_wrap(options.ring_iterations); result != 0)
    {
        return result;
    }
    if (const auto result = test_long_slot_generation(options.pool_iterations); result != 0)
    {
        return result;
    }
    if (const auto result = test_frame_sequence_overflow_is_explicitly_fatal(); result != 0)
    {
        return result;
    }

    std::cout << "{\"mode\":\"" << options.mode
              << "\",\"runtime_cycles\":" << options.runtime_cycles
              << ",\"ring_iterations\":" << options.ring_iterations
              << ",\"pool_iterations\":" << options.pool_iterations
              << ",\"resident_growth_bytes\":" << memory_growth << "}\n";
    return 0;
}
