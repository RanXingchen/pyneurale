/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>

#include <neurale/streaming/fault.h>

namespace neurale::execution
{

using namespace neurale::experiments;

/// Private single-producer/single-consumer handoff used by the three paradigm
/// controllers. Storage is fixed by prepare(); reset() is a control-plane
/// operation performed only after both participants have stopped.
template <typename T> class BoundedTraceQueue
{
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::atomic<std::size_t>::is_always_lock_free);

  public:
    BoundedTraceQueue() noexcept = default;

    BoundedTraceQueue(const BoundedTraceQueue&) = delete;
    BoundedTraceQueue& operator=(const BoundedTraceQueue&) = delete;
    BoundedTraceQueue(BoundedTraceQueue&&) = delete;
    BoundedTraceQueue& operator=(BoundedTraceQueue&&) = delete;

    void prepare(std::size_t capacity)
    {
        if (capacity == 0)
            throw std::invalid_argument("trace capacity must be positive");
        if (capacity > (std::numeric_limits<std::size_t>::max)() / sizeof(T))
            throw std::length_error("trace storage size overflows size_t");
        if (storage_ != nullptr)
        {
            if (capacity != capacity_)
                throw std::invalid_argument("trace capacity cannot change after prepare");
            reset();
            return;
        }
        storage_ = std::make_unique_for_overwrite<T[]>(capacity);
        capacity_ = capacity;
        reset();
    }

    [[nodiscard]] streaming::StreamStatus try_push(const T& value) noexcept
    {
        if (closed_.load(std::memory_order_acquire))
        {
            note_dropped(1);
            return streaming::StreamStatus::stopped;
        }
        const auto produced = produced_.load(std::memory_order_relaxed);
        const auto consumed = consumed_.load(std::memory_order_acquire);
        if (produced - consumed == capacity_)
        {
            note_dropped(1);
            return streaming::StreamStatus::queue_overflow;
        }
        storage_[produced % capacity_] = value;
        produced_.store(produced + 1, std::memory_order_release);
        return streaming::StreamStatus::ok;
    }

    [[nodiscard]] streaming::StreamStatus try_pop(T& value) noexcept
    {
        const auto consumed = consumed_.load(std::memory_order_relaxed);
        const auto produced = produced_.load(std::memory_order_acquire);
        if (consumed == produced)
            return closed_.load(std::memory_order_acquire) ? streaming::StreamStatus::stopped
                                                           : streaming::StreamStatus::would_block;
        value = storage_[consumed % capacity_];
        consumed_.store(consumed + 1, std::memory_order_release);
        return streaming::StreamStatus::ok;
    }

    [[nodiscard]] bool can_push(std::size_t count = 1) const noexcept
    {
        if (closed_.load(std::memory_order_acquire) || count > capacity_)
            return false;
        const auto produced = produced_.load(std::memory_order_acquire);
        const auto consumed = consumed_.load(std::memory_order_acquire);
        return produced - consumed <= capacity_ - count;
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        const auto produced = produced_.load(std::memory_order_acquire);
        const auto consumed = consumed_.load(std::memory_order_acquire);
        return produced - consumed;
    }

    [[nodiscard]] std::size_t capacity() const noexcept
    {
        return capacity_;
    }

    /// Count trace records the producer decided not to enqueue.
    ///
    /// A producer that checks can_push() before building a record never reaches
    /// try_push(), so the queue cannot see that refusal on its own; the producer
    /// says how many records the refusal cost. Both paths end in the same
    /// counter because a consumer cannot tell them apart and should not have to.
    void note_dropped(std::size_t count) noexcept
    {
        dropped_.fetch_add(static_cast<std::uint64_t>(count), std::memory_order_relaxed);
    }

    /// Trace records lost since this queue was constructed.
    ///
    /// Monotonic for the queue's whole lifetime, and deliberately **not**
    /// cleared by reset(): it is read by comparing successive samples, and a
    /// counter that went backwards would make a loss look like a gain. reset()
    /// is a control-plane operation between runs; the losses of the run before
    /// it still happened.
    [[nodiscard]] std::uint64_t dropped() const noexcept
    {
        return dropped_.load(std::memory_order_relaxed);
    }

    void close() noexcept
    {
        closed_.store(true, std::memory_order_release);
    }

    void reset() noexcept
    {
        consumed_.store(0, std::memory_order_relaxed);
        produced_.store(0, std::memory_order_relaxed);
        closed_.store(false, std::memory_order_release);
    }

  private:
    std::unique_ptr<T[]> storage_{};
    std::size_t capacity_{};
    alignas(64) std::atomic<std::size_t> produced_{};
    alignas(64) std::atomic<std::size_t> consumed_{};
    alignas(64) std::atomic<bool> closed_{};
    alignas(64) std::atomic<std::uint64_t> dropped_{};
};

} // namespace neurale::execution
