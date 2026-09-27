/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <type_traits>

#include <neurale/streaming/discontinuity.h>
#include <neurale/streaming/fault.h>
#include <neurale/streaming/frame.h>

namespace neurale::streaming
{

class DiscontinuityPool;

/// Exclusive ownership of one preallocated discontinuity record.
class DiscontinuityLease
{
  public:
    DiscontinuityLease() noexcept = default;
    ~DiscontinuityLease() noexcept;

    DiscontinuityLease(const DiscontinuityLease&) = delete;
    DiscontinuityLease& operator=(const DiscontinuityLease&) = delete;
    DiscontinuityLease(DiscontinuityLease&& other) noexcept;
    DiscontinuityLease& operator=(DiscontinuityLease&& other) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return owner_ != nullptr;
    }

    [[nodiscard]] std::size_t gap_capacity() const noexcept
    {
        return gap_storage_.size();
    }

    [[nodiscard]] StreamStatus assign(SessionId session_id, std::uint64_t previous_frame_sequence,
                                      std::uint64_t actual_frame_sequence, GapReason reason,
                                      std::span<const SignalGap> gaps) noexcept;

    [[nodiscard]] Discontinuity view() const noexcept
    {
        return value_;
    }
    [[nodiscard]] StreamStatus reset() noexcept;

  private:
    friend class DiscontinuityPool;

    DiscontinuityPool* owner_{};
    BufferToken token_{};
    std::span<SignalGap> gap_storage_{};
    Discontinuity value_{};
};

/// Fixed-capacity storage for discontinuities waiting on ordered edges.
class DiscontinuityPool
{
  public:
    DiscontinuityPool(std::size_t capacity, std::size_t gaps_per_record);
    ~DiscontinuityPool();

    DiscontinuityPool(const DiscontinuityPool&) = delete;
    DiscontinuityPool& operator=(const DiscontinuityPool&) = delete;
    DiscontinuityPool(DiscontinuityPool&&) = delete;
    DiscontinuityPool& operator=(DiscontinuityPool&&) = delete;

    [[nodiscard]] StreamStatus try_acquire(DiscontinuityLease& lease) noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t gaps_per_record() const noexcept;
    [[nodiscard]] std::size_t outstanding() const noexcept;
    void prefault() noexcept;

  private:
    friend class DiscontinuityLease;
    struct Impl;

    [[nodiscard]] StreamStatus release(BufferToken token) noexcept;

    std::unique_ptr<Impl> impl_;
};

static_assert(!std::is_copy_constructible_v<DiscontinuityLease>);
static_assert(std::is_nothrow_move_constructible_v<DiscontinuityLease>);
static_assert(std::is_nothrow_move_assignable_v<DiscontinuityLease>);

} // namespace neurale::streaming
