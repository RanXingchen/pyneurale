/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <neurale/streaming/fault.h>

namespace neurale::streaming
{

class NativeStreamRunner;

/// Fixed-capacity queue for exactly one producer and one consumer.
/// Construction and destruction belong to the control plane; destruction
/// requires both participants to have stopped.
template <typename T> class SpscRing
{
    static_assert(std::is_nothrow_move_constructible_v<T>);
    static_assert(std::is_nothrow_move_assignable_v<T>);
    static_assert(std::is_nothrow_destructible_v<T>);
    static_assert(std::atomic<std::size_t>::is_always_lock_free);
    static_assert(std::atomic<bool>::is_always_lock_free);

  public:
    explicit SpscRing(std::size_t capacity) : capacity_(capacity)
    {
        if (capacity_ == 0)
        {
            throw std::invalid_argument("SPSC ring capacity must be positive");
        }
        if (capacity_ > std::numeric_limits<std::size_t>::max() / sizeof(Slot))
        {
            throw std::length_error("SPSC ring storage size overflows size_t");
        }
        slots_ = std::make_unique_for_overwrite<Slot[]>(capacity_);
    }

    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;
    SpscRing(SpscRing&&) = delete;
    SpscRing& operator=(SpscRing&&) = delete;

    ~SpscRing() noexcept
    {
        auto consumed = consumer_.sequence.load(std::memory_order_relaxed);
        const auto produced = producer_.sequence.load(std::memory_order_relaxed);
        while (consumed != produced)
        {
            std::destroy_at(slots_[slot_index(consumed)].value());
            ++consumed;
        }
    }

    /// Publish one message, or return queue_overflow/stopped without consuming it.
    [[nodiscard]] StreamStatus try_push(T&& value) noexcept
    {
        if (closed_.load(std::memory_order_acquire))
        {
            return StreamStatus::stopped;
        }

        const auto produced = producer_.sequence.load(std::memory_order_relaxed);
        // Acquire the consumer's release before reusing a destroyed slot.
        const auto consumed = consumer_.sequence.load(std::memory_order_acquire);
        if (produced - consumed == capacity_)
        {
            return StreamStatus::queue_overflow;
        }

        std::construct_at(slots_[slot_index(produced)].storage_address(), std::move(value));
        // Publish the fully constructed message to the consumer.
        producer_.sequence.store(produced + 1, std::memory_order_release);
        return StreamStatus::ok;
    }

    /// Consume one message, or return would_block/stopped without modifying output.
    [[nodiscard]] StreamStatus try_pop(T& output) noexcept
    {
        const auto consumed = consumer_.sequence.load(std::memory_order_relaxed);
        // Acquire the producer's release before reading a published slot.
        auto produced = producer_.sequence.load(std::memory_order_acquire);
        if (consumed == produced)
        {
            if (!closed_.load(std::memory_order_acquire))
            {
                return StreamStatus::would_block;
            }
            // close() follows the producer's final publication. Reload after
            // acquiring close so stopped always means fully drained.
            produced = producer_.sequence.load(std::memory_order_acquire);
            if (consumed == produced)
            {
                return StreamStatus::stopped;
            }
        }

        auto* value = slots_[slot_index(consumed)].value();
        output = std::move(*value);
        std::destroy_at(value);
        // Publish slot destruction before the producer can reuse it.
        consumer_.sequence.store(consumed + 1, std::memory_order_release);
        return StreamStatus::ok;
    }

    /// Producer-side terminal operation; it must not race another try_push call.
    void close() noexcept
    {
        closed_.store(true, std::memory_order_release);
    }

    [[nodiscard]] std::size_t capacity() const noexcept
    {
        return capacity_;
    }

    /// Diagnostic snapshot; concurrent progress may make it immediately stale.
    [[nodiscard]] std::size_t approximate_size() const noexcept
    {
        const auto consumed = consumer_.sequence.load(std::memory_order_acquire);
        const auto produced = producer_.sequence.load(std::memory_order_acquire);
        return std::min(produced - consumed, capacity_);
    }

    /// Diagnostic snapshot; exact for the consumer and advisory for other threads.
    [[nodiscard]] bool empty() const noexcept
    {
        const auto consumed = consumer_.sequence.load(std::memory_order_acquire);
        const auto produced = producer_.sequence.load(std::memory_order_acquire);
        return consumed == produced;
    }

    void prefault() noexcept
    {
        for (std::size_t i = 0; i < capacity_; ++i)
        {
            std::fill_n(slots_[i].storage, sizeof(slots_[i].storage), std::byte{});
        }
    }

    /// Diagnostic snapshot; exact for the producer and advisory for other threads.
    [[nodiscard]] bool full() const noexcept
    {
        const auto produced = producer_.sequence.load(std::memory_order_acquire);
        const auto consumed = consumer_.sequence.load(std::memory_order_acquire);
        return produced - consumed >= capacity_;
    }

    [[nodiscard]] bool closed() const noexcept
    {
        return closed_.load(std::memory_order_acquire);
    }

  private:
    friend class NativeStreamRunner;

    static constexpr std::size_t cache_line_size = 64;

    struct Slot
    {
        alignas(T) std::byte storage[sizeof(T)];

        [[nodiscard]] T* storage_address() noexcept
        {
            return reinterpret_cast<T*>(storage);
        }

        [[nodiscard]] T* value() noexcept
        {
            return std::launder(storage_address());
        }
    };

    struct alignas(cache_line_size) Cursor
    {
        std::atomic<std::size_t> sequence{};
    };

    static_assert(alignof(Cursor) == cache_line_size);
    static_assert(sizeof(Cursor) >= cache_line_size);

    [[nodiscard]] std::size_t slot_index(std::size_t sequence) const noexcept
    {
        return sequence % capacity_;
    }

    [[nodiscard]] std::size_t producer_sequence() const noexcept
    {
        return producer_.sequence.load(std::memory_order_acquire);
    }

    [[nodiscard]] std::size_t consumer_sequence() const noexcept
    {
        return consumer_.sequence.load(std::memory_order_acquire);
    }

    const std::size_t capacity_;
    std::unique_ptr<Slot[]> slots_;
    Cursor producer_{};
    Cursor consumer_{};
    alignas(cache_line_size) std::atomic<bool> closed_{};
};

} // namespace neurale::streaming
