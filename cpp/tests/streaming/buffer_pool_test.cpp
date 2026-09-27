/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/discontinuity_pool.h>
#include <neurale/streaming/spsc_ring.h>
#include <neurale/streaming/stream_message.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#include "allocation_counter.h"
#include "check_returns.h"

namespace
{

using namespace neurale::streaming;

template <typename Exception, typename Callable> [[nodiscard]] bool throws(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (const Exception&)
    {
        return true;
    }
    catch (...)
    {
        return false;
    }
    return false;
}

int test_construction_and_capacity()
{
    CHECK(throws<std::invalid_argument>([] { FramePool{0, 16, 1}; }));
    CHECK(throws<std::invalid_argument>([] { FramePool{1, 0, 1}; }));
    CHECK(throws<std::invalid_argument>([] { FramePool{1, 16, 0}; }));
    CHECK(throws<std::length_error>([]
                                    { FramePool{2, std::numeric_limits<std::size_t>::max(), 1}; }));
    CHECK(throws<std::length_error>(
        [] { DiscontinuityPool{2, std::numeric_limits<std::size_t>::max()}; }));

    FramePool pool{2, 128, 3};
    CHECK(pool.capacity() == 2);
    CHECK(pool.buffer_size() == 128);
    CHECK(pool.max_signal_blocks() == 3);
    FrameLease first;
    FrameLease second;
    FrameLease exhausted;
    CHECK(pool.try_acquire(first) == StreamStatus::ok);
    CHECK(pool.try_acquire(second) == StreamStatus::ok);
    CHECK(first.frame().payload_storage().data() != second.frame().payload_storage().data());
    CHECK(first.frame().block_storage().data() != second.frame().block_storage().data());
    CHECK(pool.outstanding() == 2);
    CHECK(pool.try_acquire(exhausted) == StreamStatus::buffer_exhausted);
    return 0;
}

int test_move_only_generation_and_zero_copy_publication()
{
    static_assert(!std::is_copy_constructible_v<FrameLease>);
    static_assert(!std::is_copy_constructible_v<MutableFrame>);
    static_assert(!std::is_copy_constructible_v<StreamMessage>);

    FramePool pool{1, 64, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    const auto first_token = lease.token();
    auto* const payload = lease.frame().payload_storage().data();
    payload[0] = std::byte{0x2a};
    CHECK(lease.frame().set_used_sizes(0, 1) == StreamStatus::ok);

    auto message = StreamMessage::from_frame(std::move(lease));
    CHECK(!lease);
    CHECK(message.frame().payload.data() == payload);
    CHECK(message.frame().payload[0] == std::byte{0x2a});
    CHECK(pool.outstanding() == 1);
    FrameLease unavailable;
    CHECK(pool.try_acquire(unavailable) == StreamStatus::buffer_exhausted);

    message = StreamMessage::end_of_stream();
    CHECK(pool.try_acquire(unavailable) == StreamStatus::ok);
    CHECK(unavailable.token().slot == first_token.slot);
    CHECK(unavailable.token().generation == first_token.generation + 1);
    CHECK(unavailable.reset() == StreamStatus::ok);
    CHECK(pool.outstanding() == 0);
    CHECK(unavailable.reset() == StreamStatus::stopped);
    return 0;
}

int test_queue_owns_frame_until_consumed()
{
    FramePool pool{1, 64, 1};
    SpscRing<StreamMessage> ring{1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    auto* const payload = lease.frame().payload_storage().data();
    payload[0] = std::byte{0x7f};
    CHECK(lease.frame().set_used_sizes(0, 1) == StreamStatus::ok);
    auto message = StreamMessage::from_frame(std::move(lease));
    CHECK(ring.try_push(std::move(message)) == StreamStatus::ok);

    FrameLease blocked;
    CHECK(pool.try_acquire(blocked) == StreamStatus::buffer_exhausted);
    StreamMessage received;
    CHECK(ring.try_pop(received) == StreamStatus::ok);
    CHECK(received.frame().payload.data() == payload);
    CHECK(received.frame().payload[0] == std::byte{0x7f});
    CHECK(pool.try_acquire(blocked) == StreamStatus::buffer_exhausted);
    received = StreamMessage::end_of_stream();
    CHECK(pool.try_acquire(blocked) == StreamStatus::ok);
    return 0;
}

int test_steady_state_does_not_allocate()
{
    FramePool frames{2, 256, 4};
    DiscontinuityPool discontinuities{2, 4};
    const auto before = allocations.load(std::memory_order_relaxed);
    for (std::size_t iteration = 0; iteration < 100'000; ++iteration)
    {
        FrameLease frame;
        DiscontinuityLease discontinuity;
        CHECK(frames.try_acquire(frame) == StreamStatus::ok);
        CHECK(discontinuities.try_acquire(discontinuity) == StreamStatus::ok);
        CHECK(frame.reset() == StreamStatus::ok);
        CHECK(discontinuity.reset() == StreamStatus::ok);
    }
    CHECK(allocations.load(std::memory_order_relaxed) == before);
    return 0;
}

int test_concurrent_acquire_release()
{
    constexpr std::size_t capacity = 8;
    constexpr std::size_t n_workers = 4;
    constexpr std::size_t iterations = 50'000;
    FramePool pool{capacity, 64, 1};
    std::array<std::atomic<int>, capacity> active{};
    std::atomic<bool> start{};
    std::atomic<bool> failed{};
    std::array<std::thread, n_workers> workers;

    for (auto& worker : workers)
    {
        worker =
            std::thread{[&]
                        {
                            while (!start.load(std::memory_order_acquire))
                            {
                            }
                            for (std::size_t iteration = 0; iteration < iterations; ++iteration)
                            {
                                FrameLease lease;
                                while (pool.try_acquire(lease) == StreamStatus::buffer_exhausted)
                                {
                                }
                                if (!lease)
                                {
                                    failed.store(true, std::memory_order_relaxed);
                                    return;
                                }
                                const auto slot = lease.token().slot;
                                if (active[slot].fetch_add(1, std::memory_order_relaxed) != 0)
                                {
                                    failed.store(true, std::memory_order_relaxed);
                                }
                                active[slot].fetch_sub(1, std::memory_order_relaxed);
                                if (lease.reset() != StreamStatus::ok)
                                {
                                    failed.store(true, std::memory_order_relaxed);
                                    return;
                                }
                            }
                        }};
    }
    start.store(true, std::memory_order_release);
    for (auto& worker : workers)
    {
        worker.join();
    }
    CHECK(!failed.load(std::memory_order_relaxed));
    return 0;
}

} // namespace

int main()
{
    for (const auto test : {
             test_construction_and_capacity,
             test_move_only_generation_and_zero_copy_publication,
             test_queue_owns_frame_until_consumed,
             test_steady_state_does_not_allocate,
             test_concurrent_acquire_release,
         })
    {
        if (const auto line = test(); line != 0)
        {
            return line;
        }
    }
    return 0;
}
