/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/buffer_pool.h>

#include "slot_registry.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace neurale::streaming
{
namespace
{

[[nodiscard]] std::size_t checked_product(std::size_t left, std::size_t right)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::length_error("frame pool storage size overflows size_t");
    }
    return left * right;
}

} // namespace

struct FramePool::Impl
{
    Impl(std::size_t n_slots, std::size_t bytes_per_slot, std::size_t blocks_per_slot)
        : registry(n_slots), payload_bytes(bytes_per_slot), signal_blocks(blocks_per_slot)
    {
        if (payload_bytes == 0 || signal_blocks == 0)
        {
            throw std::invalid_argument("frame pool payload and block capacities must be positive");
        }
        payloads =
            std::make_unique_for_overwrite<std::byte[]>(checked_product(n_slots, payload_bytes));
        blocks = std::make_unique_for_overwrite<SignalBlockHeader[]>(
            checked_product(n_slots, signal_blocks));
    }

    detail::SlotRegistry registry;
    std::size_t payload_bytes{};
    std::size_t signal_blocks{};
    std::unique_ptr<std::byte[]> payloads;
    std::unique_ptr<SignalBlockHeader[]> blocks;
};

FrameLease::~FrameLease() noexcept
{
    static_cast<void>(reset());
}

FrameLease::FrameLease(FrameLease&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), frame_(std::move(other.frame_))
{
    other.frame_.clear();
}

FrameLease& FrameLease::operator=(FrameLease&& other) noexcept
{
    if (this != &other)
    {
        static_cast<void>(reset());
        owner_ = std::exchange(other.owner_, nullptr);
        frame_ = std::move(other.frame_);
        other.frame_.clear();
    }
    return *this;
}

StreamStatus FrameLease::reset() noexcept
{
    if (owner_ == nullptr)
    {
        return StreamStatus::stopped;
    }
    auto* owner = std::exchange(owner_, nullptr);
    const auto token = frame_.token_;
    frame_.clear();
    return owner->release(token);
}

FramePool::FramePool(std::size_t capacity, std::size_t payload_bytes, std::size_t max_signal_blocks)
    : impl_(std::make_unique<Impl>(capacity, payload_bytes, max_signal_blocks))
{
}

FramePool::~FramePool()
{
    if (impl_->registry.outstanding() != 0)
    {
        std::terminate();
    }
}

StreamStatus FramePool::try_acquire(FrameLease& lease) noexcept
{
    if (lease)
    {
        return StreamStatus::invalid_frame;
    }
    BufferToken token{};
    const auto status = impl_->registry.try_acquire(token);
    if (status != StreamStatus::ok)
    {
        return status;
    }
    const auto slot = static_cast<std::size_t>(token.slot);
    lease.owner_ = this;
    const std::span<std::byte> payload{
        impl_->payloads.get() + slot * impl_->payload_bytes,
        impl_->payload_bytes,
    };
    const std::span<SignalBlockHeader> blocks{
        impl_->blocks.get() + slot * impl_->signal_blocks,
        impl_->signal_blocks,
    };
    lease.frame_.attach(token, blocks, payload);
    return StreamStatus::ok;
}

std::size_t FramePool::capacity() const noexcept
{
    return impl_->registry.capacity();
}

std::size_t FramePool::buffer_size() const noexcept
{
    return impl_->payload_bytes;
}

std::size_t FramePool::max_signal_blocks() const noexcept
{
    return impl_->signal_blocks;
}

std::size_t FramePool::outstanding() const noexcept
{
    return impl_->registry.outstanding();
}

void FramePool::prefault() noexcept
{
    std::fill_n(impl_->payloads.get(), impl_->registry.capacity() * impl_->payload_bytes,
                std::byte{});
    std::fill_n(impl_->blocks.get(), impl_->registry.capacity() * impl_->signal_blocks,
                SignalBlockHeader{});
}

StreamStatus FramePool::release(BufferToken token) noexcept
{
    return impl_->registry.release(token);
}

} // namespace neurale::streaming
