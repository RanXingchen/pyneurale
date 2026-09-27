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
#include <stdexcept>

#include <neurale/streaming/fault.h>
#include <neurale/streaming/frame.h>

namespace neurale::streaming::detail
{

class SlotRegistry
{
  public:
    explicit SlotRegistry(std::size_t capacity) : capacity_(capacity)
    {
        static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
        static_assert(std::atomic<std::size_t>::is_always_lock_free);
        if (capacity == 0 || capacity > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::invalid_argument("slot capacity is outside uint32 range");
        }
        states_ = std::make_unique_for_overwrite<SlotState[]>(capacity);
    }

    [[nodiscard]] StreamStatus try_acquire(BufferToken& token) noexcept
    {
        const auto start = next_hint_.fetch_add(1, std::memory_order_relaxed) % capacity_;
        for (std::size_t offset = 0; offset < capacity_; ++offset)
        {
            const auto slot = (start + offset) % capacity_;
            auto state = states_[slot].value.load(std::memory_order_relaxed);
            if ((state & leased_bit) != 0)
            {
                continue;
            }
            if (states_[slot].value.compare_exchange_strong(state, state | leased_bit,
                                                            std::memory_order_acquire,
                                                            std::memory_order_relaxed))
            {
                token = BufferToken{
                    .slot = static_cast<std::uint32_t>(slot),
                    .generation = state >> generation_shift,
                };
                outstanding_.fetch_add(1, std::memory_order_relaxed);
                return StreamStatus::ok;
            }
        }
        return StreamStatus::buffer_exhausted;
    }

    [[nodiscard]] StreamStatus release(BufferToken token) noexcept
    {
        if (token.slot >= capacity_ || token.generation >= max_generation)
        {
            return StreamStatus::invalid_frame;
        }
        auto expected = (token.generation << generation_shift) | leased_bit;
        const auto next = (token.generation + 1) << generation_shift;
        if (!states_[token.slot].value.compare_exchange_strong(
                expected, next, std::memory_order_release, std::memory_order_relaxed))
        {
            return StreamStatus::invalid_frame;
        }
        outstanding_.fetch_sub(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    [[nodiscard]] std::size_t capacity() const noexcept
    {
        return capacity_;
    }
    [[nodiscard]] std::size_t outstanding() const noexcept
    {
        return outstanding_.load(std::memory_order_acquire);
    }

  private:
    static constexpr std::uint64_t leased_bit = 1;
    static constexpr unsigned generation_shift = 1;
    static constexpr std::uint64_t max_generation =
        std::numeric_limits<std::uint64_t>::max() >> generation_shift;

    struct SlotState
    {
        std::atomic<std::uint64_t> value{};
    };

    const std::size_t capacity_;
    std::unique_ptr<SlotState[]> states_;
    std::atomic<std::size_t> next_hint_{};
    std::atomic<std::size_t> outstanding_{};
};

} // namespace neurale::streaming::detail
