/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <neurale/streaming/fault.h>

namespace neurale::streaming
{

/// Bounded observer edge. Its producer may evict the oldest item while the
/// observer remains the normal consumer; the dequeue cursor arbitrates that
/// race and each slot carries a generation-aware read/write phase.
template <typename T> class ObserverQueue
{
    static_assert(std::is_nothrow_move_constructible_v<T>);
    static_assert(std::is_nothrow_move_assignable_v<T>);
    static_assert(std::is_nothrow_destructible_v<T>);

  public:
    explicit ObserverQueue(std::size_t capacity) : capacity_(capacity)
    {
        if (capacity == 0)
        {
            throw std::invalid_argument("observer queue capacity must be positive");
        }
        if (capacity > std::numeric_limits<std::size_t>::max() / sizeof(Slot))
        {
            throw std::length_error("observer queue storage overflows size_t");
        }
        slots_ = std::make_unique_for_overwrite<Slot[]>(capacity);
        for (std::size_t i = 0; i < capacity; ++i)
        {
            slots_[i].sequence.store(i * 2, std::memory_order_relaxed);
        }
    }

    ObserverQueue(const ObserverQueue&) = delete;
    ObserverQueue& operator=(const ObserverQueue&) = delete;

    ~ObserverQueue() noexcept
    {
        T value;
        while (try_pop(value) == StreamStatus::ok)
        {
            value = T{};
        }
    }

    [[nodiscard]] StreamStatus try_push(T&& value) noexcept
    {
        if (closed_.load(std::memory_order_acquire))
        {
            return StreamStatus::stopped;
        }
        const auto pos = enqueue_.load(std::memory_order_relaxed);
        auto& slot = slots_[pos % capacity_];
        const auto writable_sequence = pos * 2;
        if (slot.sequence.load(std::memory_order_acquire) != writable_sequence)
        {
            return StreamStatus::queue_overflow;
        }
        std::construct_at(slot.value(), std::move(value));
        slot.sequence.store(writable_sequence + 1, std::memory_order_release);
        enqueue_.store(pos + 1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    [[nodiscard]] StreamStatus try_pop(T& output) noexcept
    {
        auto pos = dequeue_.load(std::memory_order_relaxed);
        for (;;)
        {
            auto& slot = slots_[pos % capacity_];
            const auto readable_sequence = pos * 2 + 1;
            if (slot.sequence.load(std::memory_order_acquire) != readable_sequence)
            {
                return closed_.load(std::memory_order_acquire) && empty()
                           ? StreamStatus::stopped
                           : StreamStatus::would_block;
            }
            if (dequeue_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed,
                                               std::memory_order_relaxed))
            {
                auto* value = slot.value();
                output = std::move(*value);
                std::destroy_at(value);
                slot.sequence.store((pos + capacity_) * 2, std::memory_order_release);
                return StreamStatus::ok;
            }
        }
    }

    void close() noexcept
    {
        closed_.store(true, std::memory_order_release);
    }

    [[nodiscard]] std::size_t capacity() const noexcept
    {
        return capacity_;
    }
    [[nodiscard]] bool closed() const noexcept
    {
        return closed_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::size_t approximate_size() const noexcept
    {
        const auto enqueued = enqueue_.load(std::memory_order_acquire);
        const auto dequeued = dequeue_.load(std::memory_order_acquire);
        return enqueued >= dequeued ? std::min(enqueued - dequeued, capacity_) : 0;
    }
    [[nodiscard]] bool empty() const noexcept
    {
        return approximate_size() == 0;
    }

    void prefault() noexcept
    {
        for (std::size_t i = 0; i < capacity_; ++i)
        {
            std::fill_n(slots_[i].storage, sizeof(slots_[i].storage), std::byte{});
        }
    }

  private:
    struct Slot
    {
        std::atomic<std::size_t> sequence{};
        alignas(T) std::byte storage[sizeof(T)];

        [[nodiscard]] T* value() noexcept
        {
            return std::launder(reinterpret_cast<T*>(storage));
        }
    };

    const std::size_t capacity_;
    std::unique_ptr<Slot[]> slots_;
    alignas(64) std::atomic<std::size_t> enqueue_{};
    alignas(64) std::atomic<std::size_t> dequeue_{};
    alignas(64) std::atomic<bool> closed_{};
};

} // namespace neurale::streaming
