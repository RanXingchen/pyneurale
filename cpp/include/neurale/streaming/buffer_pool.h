/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

#include <neurale/streaming/fault.h>
#include <neurale/streaming/frame.h>

namespace neurale::streaming
{

class FramePool;

/// Exclusive ownership of one preallocated frame-pool slot.
class FrameLease
{
  public:
    FrameLease() noexcept = default;
    ~FrameLease() noexcept;

    FrameLease(const FrameLease&) = delete;
    FrameLease& operator=(const FrameLease&) = delete;
    FrameLease(FrameLease&& other) noexcept;
    FrameLease& operator=(FrameLease&& other) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return owner_ != nullptr;
    }

    [[nodiscard]] MutableFrame& frame() noexcept
    {
        return frame_;
    }
    [[nodiscard]] const MutableFrame& frame() const noexcept
    {
        return frame_;
    }
    [[nodiscard]] FrameView view() const noexcept
    {
        return frame_.view();
    }
    [[nodiscard]] BufferToken token() const noexcept
    {
        return frame_.token_;
    }

    /// Release the slot immediately. Destruction performs the same operation.
    [[nodiscard]] StreamStatus reset() noexcept;

  private:
    friend class FramePool;

    FramePool* owner_{};
    MutableFrame frame_{};
};

/// Fixed-capacity frame storage allocated before realtime startup.
class FramePool
{
  public:
    FramePool(std::size_t capacity, std::size_t payload_bytes, std::size_t max_signal_blocks);
    ~FramePool();

    FramePool(const FramePool&) = delete;
    FramePool& operator=(const FramePool&) = delete;
    FramePool(FramePool&&) = delete;
    FramePool& operator=(FramePool&&) = delete;

    [[nodiscard]] StreamStatus try_acquire(FrameLease& lease) noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t buffer_size() const noexcept;
    [[nodiscard]] std::size_t max_signal_blocks() const noexcept;
    [[nodiscard]] std::size_t outstanding() const noexcept;
    void prefault() noexcept;

  private:
    friend class FrameLease;
    struct Impl;

    [[nodiscard]] StreamStatus release(BufferToken token) noexcept;

    std::unique_ptr<Impl> impl_;
};

static_assert(!std::is_copy_constructible_v<FrameLease>);
static_assert(std::is_nothrow_move_constructible_v<FrameLease>);
static_assert(std::is_nothrow_move_assignable_v<FrameLease>);

} // namespace neurale::streaming
