/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

#include <neurale/streaming/fault.h>
#include <neurale/streaming/frame.h>

namespace neurale::streaming
{

class FrameLease;
class LinearProcessorChain;
class NativeFrameProcessor;
class FrameBorrowScope;

namespace detail
{

struct FrameBorrowState
{
    MutableFrame* frame{};
    std::uint64_t epoch{};
    bool active{};
};

} // namespace detail

/// Move-only, callback-scoped access to a mutable frame.
///
/// The emitter owns the frame lease. A borrow becomes stale when its frame is
/// published or when the processor callback ends. `get()` then returns null;
/// direct accessor use after invalidation terminates instead of accessing a
/// recycled pool slot.
class FrameBorrow final
{
  public:
    FrameBorrow() noexcept = default;
    FrameBorrow(const FrameBorrow&) = delete;
    FrameBorrow& operator=(const FrameBorrow&) = delete;

    FrameBorrow(FrameBorrow&& other) noexcept
        : state_(std::exchange(other.state_, nullptr)), epoch_(std::exchange(other.epoch_, 0))
    {
    }

    FrameBorrow& operator=(FrameBorrow&& other) noexcept
    {
        if (this != &other)
        {
            state_ = std::exchange(other.state_, nullptr);
            epoch_ = std::exchange(other.epoch_, 0);
        }
        return *this;
    }

    [[nodiscard]] MutableFrame* get() noexcept
    {
        return valid() ? state_->frame : nullptr;
    }

    [[nodiscard]] const MutableFrame* get() const noexcept
    {
        return valid() ? state_->frame : nullptr;
    }

    [[nodiscard]] bool valid() const noexcept
    {
        return state_ != nullptr && state_->active && state_->epoch == epoch_ &&
               state_->frame != nullptr;
    }

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return valid();
    }

    [[nodiscard]] MutableFrame* operator->() noexcept
    {
        return &checked();
    }

    [[nodiscard]] const MutableFrame* operator->() const noexcept
    {
        return &checked();
    }

    [[nodiscard]] MutableFrame& operator*() noexcept
    {
        return checked();
    }

    [[nodiscard]] const MutableFrame& operator*() const noexcept
    {
        return checked();
    }

    [[nodiscard]] FrameHeader& header() noexcept
    {
        return checked().header();
    }
    [[nodiscard]] const FrameHeader& header() const noexcept
    {
        return checked().header();
    }
    [[nodiscard]] std::span<SignalBlockHeader> block_storage() noexcept
    {
        return checked().block_storage();
    }
    [[nodiscard]] std::span<std::byte> payload_storage() noexcept
    {
        return checked().payload_storage();
    }
    [[nodiscard]] std::span<SignalBlockHeader> blocks() noexcept
    {
        return checked().blocks();
    }
    [[nodiscard]] std::span<std::byte> payload() noexcept
    {
        return checked().payload();
    }
    [[nodiscard]] StreamStatus set_used_sizes(std::size_t n_blocks,
                                              std::size_t payload_size) noexcept
    {
        auto* frame = get();
        return frame == nullptr ? StreamStatus::invalid_state
                                : frame->set_used_sizes(n_blocks, payload_size);
    }
    [[nodiscard]] FrameView view() const noexcept
    {
        return checked().view();
    }

  private:
    friend class FrameEmitter;
    friend class FrameBorrowScope;

    FrameBorrow(detail::FrameBorrowState& state, std::uint64_t epoch) noexcept
        : state_(&state), epoch_(epoch)
    {
    }

    [[nodiscard]] MutableFrame& checked() noexcept
    {
        auto* frame = get();
        if (frame == nullptr)
        {
            std::terminate();
        }
        return *frame;
    }

    [[nodiscard]] const MutableFrame& checked() const noexcept
    {
        const auto* frame = get();
        if (frame == nullptr)
        {
            std::terminate();
        }
        return *frame;
    }

    detail::FrameBorrowState* state_{};
    std::uint64_t epoch_{};
};

static_assert(!std::is_copy_constructible_v<FrameBorrow>);
static_assert(std::is_nothrow_move_constructible_v<FrameBorrow>);
static_assert(std::is_nothrow_move_assignable_v<FrameBorrow>);

/// RAII boundary for directly invoking a processor with an existing frame.
class FrameBorrowScope final
{
  public:
    ~FrameBorrowScope() noexcept
    {
        state_->active = false;
        state_->frame = nullptr;
    }

    FrameBorrowScope(const FrameBorrowScope&) = delete;
    FrameBorrowScope& operator=(const FrameBorrowScope&) = delete;
    FrameBorrowScope(FrameBorrowScope&&) = delete;
    FrameBorrowScope& operator=(FrameBorrowScope&&) = delete;

    [[nodiscard]] bool valid() const noexcept
    {
        return borrow_.valid();
    }

    [[nodiscard]] FrameBorrow& borrow() noexcept
    {
        return borrow_;
    }

  private:
    friend class NativeFrameProcessor;

    FrameBorrowScope(detail::FrameBorrowState& state, MutableFrame& frame) noexcept : state_(&state)
    {
        if (!state.active && state.epoch != std::numeric_limits<std::uint64_t>::max())
        {
            ++state.epoch;
            state.frame = &frame;
            state.active = true;
            borrow_ = FrameBorrow{state, state.epoch};
        }
    }

    detail::FrameBorrowState* state_{};
    FrameBorrow borrow_{};
};

/// Callback-scoped publication surface for a native frame processor.
class FrameEmitter
{
  public:
    virtual ~FrameEmitter() = default;

    FrameEmitter(const FrameEmitter&) = delete;
    FrameEmitter& operator=(const FrameEmitter&) = delete;
    FrameEmitter(FrameEmitter&&) = delete;
    FrameEmitter& operator=(FrameEmitter&&) = delete;

    /// Borrow one emitter-owned writable frame until publication or callback end.
    [[nodiscard]] StreamStatus try_acquire(FrameBorrow& frame) noexcept
    {
        frame = {};
        if (acquired_borrow_state_.active)
        {
            return StreamStatus::invalid_state;
        }
        MutableFrame* mutable_frame{};
        const auto status = try_acquire_frame(mutable_frame);
        if (status != StreamStatus::ok)
        {
            return status;
        }
        if (mutable_frame == nullptr)
        {
            return StreamStatus::invalid_state;
        }
        return activate_borrow(acquired_borrow_state_, *mutable_frame, frame);
    }

    /// Transfer the currently acquired output to the emitter's destination.
    [[nodiscard]] StreamStatus publish_acquired() noexcept
    {
        invalidate_borrow(acquired_borrow_state_);
        return publish_acquired_frame();
    }

    /// Transfer the callback input to the emitter's destination without copying.
    [[nodiscard]] virtual StreamStatus publish_input() noexcept = 0;

  protected:
    FrameEmitter() noexcept = default;

    [[nodiscard]] static StreamStatus activate_borrow(detail::FrameBorrowState& state,
                                                      MutableFrame& frame,
                                                      FrameBorrow& borrow) noexcept
    {
        if (state.active || state.epoch == std::numeric_limits<std::uint64_t>::max())
        {
            return StreamStatus::invalid_state;
        }
        ++state.epoch;
        state.frame = &frame;
        state.active = true;
        borrow = FrameBorrow{state, state.epoch};
        return StreamStatus::ok;
    }

    static void invalidate_borrow(detail::FrameBorrowState& state) noexcept
    {
        state.active = false;
        state.frame = nullptr;
    }

    void invalidate_acquired_borrow() noexcept
    {
        invalidate_borrow(acquired_borrow_state_);
    }

    [[nodiscard]] virtual StreamStatus try_acquire_frame(MutableFrame*& frame) noexcept = 0;
    [[nodiscard]] virtual StreamStatus publish_acquired_frame() noexcept = 0;

  private:
    friend class LinearProcessorChain;

    /// Transfer an emitter-owned lease without exposing it to processors.
    [[nodiscard]] virtual StreamStatus publish_owned(FrameLease lease) noexcept = 0;

    detail::FrameBorrowState acquired_borrow_state_{};
};

} // namespace neurale::streaming
