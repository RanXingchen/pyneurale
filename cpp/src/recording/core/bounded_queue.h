/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// A bounded single-producer / single-consumer queue of fixed-size byte slots.
///
/// Why not `neurale::streaming::SpscRing<T>`: that ring moves a `T` into a
/// slot, which is exactly right for a move-only lease and exactly wrong here.
/// What the critical callback hands over is a *copy* of the item -- the frame
/// lease goes back to the runtime the moment the callback returns -- so the
/// bytes have to land in storage the recorder already owns. A ring of
/// `std::vector` payloads would move a pointer and leave the buffers wherever
/// they were last allocated; this ring hands the producer a span into storage
/// allocated once, at `reserve()`.
///
/// The producer writes into the slot it acquired and then publishes; nothing
/// the consumer can see changes until the release-store of `head_`. There is
/// no lock, no notification, and no syscall on the producer side, which is
/// what lets the critical callback satisfy "no blocking wait".
///
/// Saturation is reported, never absorbed: `try_acquire()` returns an empty
/// span when the queue is full and the caller decides what that means. For a
/// critical recorder it means a fault (contract section 4.3) -- this class has
/// no drop policy to configure, because there is none to choose from.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace neurale::recording
{

class BoundedSlotQueue
{
  public:
    BoundedSlotQueue() = default;

    BoundedSlotQueue(const BoundedSlotQueue&) = delete;
    BoundedSlotQueue& operator=(const BoundedSlotQueue&) = delete;
    BoundedSlotQueue(BoundedSlotQueue&&) = delete;
    BoundedSlotQueue& operator=(BoundedSlotQueue&&) = delete;

    /// Allocate every byte the queue will ever use. This is the only
    /// allocation, and it belongs to `prepare()`.
    void reserve(std::size_t n_slots, std::size_t slot_bytes)
    {
        storage_.assign(n_slots * slot_bytes, std::byte{0});
        lengths_.assign(n_slots, 0);
        n_slots_ = n_slots;
        slot_bytes_ = slot_bytes;
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        high_water_mark_.store(0, std::memory_order_relaxed);
    }

    /// Give every byte back. Called from exactly one transition into `closed`
    /// or `failed`, after both participants have stopped.
    void release_storage() noexcept
    {
        storage_.clear();
        storage_.shrink_to_fit();
        lengths_.clear();
        lengths_.shrink_to_fit();
        n_slots_ = 0;
        slot_bytes_ = 0;
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

    // --- producer side, callable from the critical callback ----------------

    /// The next free slot, or an empty span when the queue is full.
    [[nodiscard]] std::span<std::byte> try_acquire() noexcept
    {
        const auto head = head_.load(std::memory_order_relaxed);
        // Acquire the consumer's release before reusing a slot it has freed.
        const auto tail = tail_.load(std::memory_order_acquire);
        const auto pending = head - tail;
        if (n_slots_ == 0 || pending >= n_slots_)
        {
            return {};
        }
        const auto idx = static_cast<std::size_t>(head % n_slots_);
        return std::span<std::byte>{storage_.data() + idx * slot_bytes_, slot_bytes_};
    }

    /// Publish the slot returned by the last `try_acquire()`, *used* bytes of
    /// it. Nothing the consumer reads was visible before this store.
    void publish(std::size_t used) noexcept
    {
        const auto head = head_.load(std::memory_order_relaxed);
        const auto idx = static_cast<std::size_t>(head % n_slots_);
        lengths_[idx] = used;
        head_.store(head + 1, std::memory_order_release);

        const auto pending = head + 1 - tail_.load(std::memory_order_relaxed);
        if (pending > high_water_mark_.load(std::memory_order_relaxed))
        {
            high_water_mark_.store(pending, std::memory_order_relaxed);
        }
    }

    // --- consumer side, the worker thread ----------------------------------

    /// The oldest published slot, or an empty span when the queue is empty.
    [[nodiscard]] std::span<const std::byte> try_peek() const noexcept
    {
        const auto tail = tail_.load(std::memory_order_relaxed);
        const auto head = head_.load(std::memory_order_acquire);
        if (tail == head)
        {
            return {};
        }
        const auto idx = static_cast<std::size_t>(tail % n_slots_);
        return std::span<const std::byte>{storage_.data() + idx * slot_bytes_, lengths_[idx]};
    }

    /// Free the slot `try_peek()` returned.
    void release() noexcept
    {
        tail_.store(tail_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    }

    // --- observation -------------------------------------------------------

    [[nodiscard]] std::uint64_t pending() const noexcept
    {
        const auto head = head_.load(std::memory_order_acquire);
        const auto tail = tail_.load(std::memory_order_acquire);
        return head - tail;
    }

    [[nodiscard]] std::uint64_t high_water_mark() const noexcept
    {
        return high_water_mark_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::size_t capacity() const noexcept
    {
        return n_slots_;
    }

    [[nodiscard]] std::size_t slot_bytes() const noexcept
    {
        return slot_bytes_;
    }

  private:
    std::vector<std::byte> storage_{};
    std::vector<std::size_t> lengths_{};
    std::size_t n_slots_{};
    std::size_t slot_bytes_{};

    // 64-bit counters that only ever increase: the difference is the depth,
    // and neither wraps in any run this hardware will see.
    std::atomic<std::uint64_t> head_{0};
    std::atomic<std::uint64_t> tail_{0};
    std::atomic<std::uint64_t> high_water_mark_{0};

    static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                  "the critical callback must not take a lock to enqueue");
};

/// A claim returned by `BoundedMpscSlotQueue::try_acquire()`. Multiple
/// producers each get a distinct claim; `publish()` needs the position to mark
/// the cell published, so it travels with the slot.
struct MpscSlotClaim
{
    std::span<std::byte> slot{};
    std::uint64_t pos{};
    [[nodiscard]] explicit operator bool() const noexcept
    {
        return !slot.empty();
    }
};

/// A bounded multi-producer / single-consumer queue of fixed-size byte slots.
///
/// The control plane is produced by the experiment or application, which is
/// not a single thread the way the critical observer edge is, so the control
/// queue is MPSC where the data queue is SPSC. Several `submit_control()`
/// callers may enqueue at once: each claims a distinct cell through a
/// compare-and-swap on the producer counter, writes into it, and publishes with
/// a release-store of that cell's sequence. The single worker consumes in
/// claim order. There is no lock on the producer path -- a CAS and two atomic
/// loads are the whole cost -- because the control plane is not the realtime
/// observer hot path but it is still a path a caller should not have to
/// serialize by hand.
///
/// The bounded full check is a sequence protocol: a cell is free at position
/// *p* when its sequence equals *p*, published when it equals *p + 1*, and
/// freed for the next lap when the consumer sets it to *p + capacity*.
/// Saturation is reported, never absorbed, exactly as for the SPSC queue: a
/// full control queue on a critical recorder is a fault.
class BoundedMpscSlotQueue
{
  public:
    BoundedMpscSlotQueue() = default;

    BoundedMpscSlotQueue(const BoundedMpscSlotQueue&) = delete;
    BoundedMpscSlotQueue& operator=(const BoundedMpscSlotQueue&) = delete;
    BoundedMpscSlotQueue(BoundedMpscSlotQueue&&) = delete;
    BoundedMpscSlotQueue& operator=(BoundedMpscSlotQueue&&) = delete;

    /// Allocate every byte and every sequence the queue will ever use. The
    /// only allocation, and it belongs to `prepare()`.
    void reserve(std::size_t n_slots, std::size_t slot_bytes)
    {
        storage_.assign(n_slots * slot_bytes, std::byte{0});
        lengths_.assign(n_slots, 0);
        sequences_ = std::make_unique<std::atomic<std::uint64_t>[]>(n_slots);
        for (std::size_t i = 0; i < n_slots; ++i)
        {
            sequences_[i].store(static_cast<std::uint64_t>(i), std::memory_order_relaxed);
        }
        n_slots_ = n_slots;
        slot_bytes_ = slot_bytes;
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        high_water_mark_.store(0, std::memory_order_relaxed);
    }

    /// Give every byte back. Called from exactly one transition into `closed`
    /// or `failed`, after the worker has stopped.
    void release_storage() noexcept
    {
        storage_.clear();
        storage_.shrink_to_fit();
        lengths_.clear();
        lengths_.shrink_to_fit();
        sequences_.reset();
        n_slots_ = 0;
        slot_bytes_ = 0;
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

    // --- producer side, callable from any control-plane thread -------------

    /// Claim the next free slot for one producer, or an empty claim when the
    /// queue is full. The claim's `slot` is storage this queue owns; the
    /// producer writes into it and then calls `publish()` with the same claim.
    [[nodiscard]] MpscSlotClaim try_acquire() noexcept
    {
        auto pos = head_.load(std::memory_order_acquire);
        for (;;)
        {
            if (n_slots_ == 0)
            {
                return {};
            }
            const auto idx = static_cast<std::size_t>(pos % n_slots_);
            const auto seq = sequences_[idx].load(std::memory_order_acquire);
            // `seq == pos` marks the cell free at this position; `seq < pos`
            // (read signed) means a previous lap is still in flight -- the
            // queue is full; `seq > pos` means another producer claimed ahead,
            // so reload and retry.
            const auto diff = static_cast<std::int64_t>(seq - pos);
            if (diff == 0)
            {
                if (head_.compare_exchange_weak(pos, pos + 1, std::memory_order_acq_rel,
                                                std::memory_order_acquire))
                {
                    return MpscSlotClaim{
                        .slot =
                            std::span<std::byte>{storage_.data() + idx * slot_bytes_, slot_bytes_},
                        .pos = pos};
                }
                // `pos` was reloaded by the failed CAS; retry.
                continue;
            }
            if (diff < 0)
            {
                return {};
            }
            pos = head_.load(std::memory_order_acquire);
        }
    }

    /// Publish the cell *claim* returned, *used* bytes of it. Nothing the
    /// consumer reads was visible before this store.
    void publish(const MpscSlotClaim& claim, std::size_t used) noexcept
    {
        const auto idx = static_cast<std::size_t>(claim.pos % n_slots_);
        lengths_[idx] = used;
        sequences_[idx].store(claim.pos + 1, std::memory_order_release);
        const auto depth =
            head_.load(std::memory_order_relaxed) - tail_.load(std::memory_order_relaxed);
        if (depth > high_water_mark_.load(std::memory_order_relaxed))
        {
            high_water_mark_.store(depth, std::memory_order_relaxed);
        }
    }

    // --- consumer side, the single worker thread ----------------------------

    /// The oldest published slot, or an empty span when the queue is empty.
    [[nodiscard]] std::span<const std::byte> try_peek() const noexcept
    {
        const auto tail = tail_.load(std::memory_order_relaxed);
        const auto idx = static_cast<std::size_t>(tail % n_slots_);
        const auto seq = sequences_[idx].load(std::memory_order_acquire);
        if (seq != tail + 1)
        {
            return {};
        }
        return std::span<const std::byte>{storage_.data() + idx * slot_bytes_, lengths_[idx]};
    }

    /// Free the slot `try_peek()` returned.
    void release() noexcept
    {
        const auto tail = tail_.load(std::memory_order_relaxed);
        const auto idx = static_cast<std::size_t>(tail % n_slots_);
        sequences_[idx].store(tail + n_slots_, std::memory_order_release);
        tail_.store(tail + 1, std::memory_order_release);
    }

    // --- observation -------------------------------------------------------

    [[nodiscard]] std::uint64_t pending() const noexcept
    {
        const auto head = head_.load(std::memory_order_acquire);
        const auto tail = tail_.load(std::memory_order_acquire);
        return head - tail;
    }

    [[nodiscard]] std::uint64_t high_water_mark() const noexcept
    {
        return high_water_mark_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::size_t capacity() const noexcept
    {
        return n_slots_;
    }

    [[nodiscard]] std::size_t slot_bytes() const noexcept
    {
        return slot_bytes_;
    }

  private:
    std::vector<std::byte> storage_{};
    std::vector<std::size_t> lengths_{};
    std::unique_ptr<std::atomic<std::uint64_t>[]> sequences_{};
    std::size_t n_slots_{};
    std::size_t slot_bytes_{};

    std::atomic<std::uint64_t> head_{0};
    std::atomic<std::uint64_t> tail_{0};
    std::atomic<std::uint64_t> high_water_mark_{0};

    static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                  "a control-plane producer must not take a lock to enqueue");
};

} // namespace neurale::recording
