/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/spsc_ring.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <thread>
#include <type_traits>
#include <utility>

#include "allocation_counter.h"
#include "check_returns.h"

namespace
{

using neurale::streaming::SpscRing;
using neurale::streaming::StreamStatus;

struct Message
{
    std::uint64_t sequence{};

    explicit Message(std::uint64_t value = 0) noexcept : sequence(value) {}
    Message(const Message&) = delete;
    Message& operator=(const Message&) = delete;

    Message(Message&& other) noexcept : sequence(std::exchange(other.sequence, moved_from)) {}

    Message& operator=(Message&& other) noexcept
    {
        sequence = std::exchange(other.sequence, moved_from);
        return *this;
    }

    static constexpr std::uint64_t moved_from = std::numeric_limits<std::uint64_t>::max();
};

static_assert(!std::is_copy_constructible_v<Message>);
static_assert(std::is_nothrow_move_constructible_v<Message>);

struct TrackedMessage
{
    static inline std::atomic<int> live_count{};

    explicit TrackedMessage(std::uint64_t value = 0) noexcept : sequence(value)
    {
        live_count.fetch_add(1, std::memory_order_relaxed);
    }

    TrackedMessage(const TrackedMessage&) = delete;
    TrackedMessage& operator=(const TrackedMessage&) = delete;

    TrackedMessage(TrackedMessage&& other) noexcept
        : sequence(std::exchange(other.sequence, Message::moved_from))
    {
        live_count.fetch_add(1, std::memory_order_relaxed);
    }

    TrackedMessage& operator=(TrackedMessage&& other) noexcept
    {
        sequence = std::exchange(other.sequence, Message::moved_from);
        return *this;
    }

    ~TrackedMessage() noexcept
    {
        live_count.fetch_sub(1, std::memory_order_relaxed);
    }

    std::uint64_t sequence{};
};

int test_construction_validation()
{
    bool rejected_zero = false;
    try
    {
        SpscRing<Message> ring{0};
    }
    catch (const std::invalid_argument&)
    {
        rejected_zero = true;
    }
    CHECK(rejected_zero);

    bool rejected_overflow = false;
    try
    {
        SpscRing<Message> ring{std::numeric_limits<std::size_t>::max()};
    }
    catch (const std::length_error&)
    {
        rejected_overflow = true;
    }
    CHECK(rejected_overflow);
    return 0;
}

int test_capacity_and_wraparound()
{
    SpscRing<Message> ring{3};
    Message first{1};
    Message second{2};
    Message third{3};
    Message rejected{4};

    CHECK(ring.capacity() == 3);
    CHECK(ring.empty());
    CHECK(!ring.full());
    CHECK(ring.try_push(std::move(first)) == StreamStatus::ok);
    CHECK(ring.try_push(std::move(second)) == StreamStatus::ok);
    CHECK(ring.try_push(std::move(third)) == StreamStatus::ok);
    CHECK(ring.full());
    CHECK(ring.approximate_size() == 3);
    CHECK(ring.try_push(std::move(rejected)) == StreamStatus::queue_overflow);
    CHECK(rejected.sequence == 4);

    Message output;
    CHECK(ring.try_pop(output) == StreamStatus::ok);
    CHECK(output.sequence == 1);
    CHECK(ring.try_push(std::move(rejected)) == StreamStatus::ok);

    for (const auto expected : {2U, 3U, 4U})
    {
        CHECK(ring.try_pop(output) == StreamStatus::ok);
        CHECK(output.sequence == expected);
    }
    CHECK(ring.empty());
    CHECK(ring.try_pop(output) == StreamStatus::would_block);

    for (std::uint64_t sequence = 0; sequence < 10'000; ++sequence)
    {
        Message input{sequence};
        CHECK(ring.try_push(std::move(input)) == StreamStatus::ok);
        CHECK(ring.try_pop(output) == StreamStatus::ok);
        CHECK(output.sequence == sequence);
    }
    return 0;
}

int test_close_and_drain()
{
    SpscRing<Message> ring{2};
    Message first{1};
    Message second{2};
    Message rejected{3};
    Message output;

    CHECK(ring.try_push(std::move(first)) == StreamStatus::ok);
    CHECK(ring.try_push(std::move(second)) == StreamStatus::ok);
    ring.close();
    CHECK(ring.closed());
    CHECK(ring.try_push(std::move(rejected)) == StreamStatus::stopped);
    CHECK(rejected.sequence == 3);
    CHECK(ring.try_pop(output) == StreamStatus::ok);
    CHECK(output.sequence == 1);
    CHECK(ring.try_pop(output) == StreamStatus::ok);
    CHECK(output.sequence == 2);
    CHECK(ring.try_pop(output) == StreamStatus::stopped);
    return 0;
}

int test_destructor_releases_pending_messages()
{
    CHECK(TrackedMessage::live_count.load(std::memory_order_relaxed) == 0);
    {
        SpscRing<TrackedMessage> ring{2};
        TrackedMessage first{1};
        TrackedMessage second{2};
        CHECK(ring.try_push(std::move(first)) == StreamStatus::ok);
        CHECK(ring.try_push(std::move(second)) == StreamStatus::ok);
        CHECK(TrackedMessage::live_count.load(std::memory_order_relaxed) == 4);
    }
    CHECK(TrackedMessage::live_count.load(std::memory_order_relaxed) == 0);
    return 0;
}

int test_steady_state_does_not_allocate()
{
    SpscRing<Message> ring{8};
    Message output;
    const auto allocations_before = allocations.load(std::memory_order_relaxed);

    for (std::uint64_t sequence = 0; sequence < 100'000; ++sequence)
    {
        Message input{sequence};
        CHECK(ring.try_push(std::move(input)) == StreamStatus::ok);
        CHECK(ring.try_pop(output) == StreamStatus::ok);
    }

    CHECK(allocations.load(std::memory_order_relaxed) == allocations_before);
    return 0;
}

int run_concurrent_stress(bool slow_producer)
{
    constexpr std::uint64_t iterations = 1'000'000;
    SpscRing<Message> ring{1'024};
    std::atomic<bool> start{};
    std::atomic<bool> failed{};

    std::thread producer{
        [&]
        {
            while (!start.load(std::memory_order_acquire))
            {
            }
            for (std::uint64_t sequence = 0; sequence < iterations; ++sequence)
            {
                if (slow_producer && (sequence & 7U) == 0U)
                {
                    for (int spin = 0; spin < 32; ++spin)
                    {
                        std::atomic_signal_fence(std::memory_order_seq_cst);
                    }
                }
                Message message{sequence};
                while (ring.try_push(std::move(message)) == StreamStatus::queue_overflow)
                {
                }
                if (message.sequence != Message::moved_from)
                {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
            }
            ring.close();
        }};

    std::thread consumer{[&]
                         {
                             while (!start.load(std::memory_order_acquire))
                             {
                             }
                             Message message;
                             for (std::uint64_t expected = 0; expected < iterations;)
                             {
                                 const auto status = ring.try_pop(message);
                                 if (status == StreamStatus::ok)
                                 {
                                     if (message.sequence != expected)
                                     {
                                         failed.store(true, std::memory_order_relaxed);
                                         return;
                                     }
                                     ++expected;
                                     if (!slow_producer && (expected & 7U) == 0U)
                                     {
                                         for (int spin = 0; spin < 32; ++spin)
                                         {
                                             std::atomic_signal_fence(std::memory_order_seq_cst);
                                         }
                                     }
                                 }
                                 else if (status == StreamStatus::stopped)
                                 {
                                     failed.store(true, std::memory_order_relaxed);
                                     return;
                                 }
                             }
                             for (;;)
                             {
                                 const auto status = ring.try_pop(message);
                                 if (status == StreamStatus::stopped)
                                 {
                                     break;
                                 }
                                 if (status != StreamStatus::would_block)
                                 {
                                     failed.store(true, std::memory_order_relaxed);
                                     return;
                                 }
                             }
                         }};

    start.store(true, std::memory_order_release);
    producer.join();
    consumer.join();
    CHECK(!failed.load(std::memory_order_relaxed));
    CHECK(ring.empty());
    CHECK(ring.approximate_size() == 0);
    return 0;
}

int test_concurrent_stress()
{
    CHECK(run_concurrent_stress(false) == 0);
    CHECK(run_concurrent_stress(true) == 0);
    return 0;
}

} // namespace

int main()
{
    for (const auto test : {
             test_construction_validation,
             test_capacity_and_wraparound,
             test_close_and_drain,
             test_destructor_releases_pending_messages,
             test_steady_state_does_not_allocate,
             test_concurrent_stress,
         })
    {
        if (const auto line = test(); line != 0)
        {
            return line;
        }
    }
    return 0;
}
