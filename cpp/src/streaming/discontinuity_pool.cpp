/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/discontinuity_pool.h>

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

[[nodiscard]] std::size_t checked_gap_count(std::size_t capacity, std::size_t gaps_per_record)
{
    if (gaps_per_record == 0 ||
        capacity > std::numeric_limits<std::size_t>::max() / gaps_per_record)
    {
        throw std::length_error("discontinuity pool size overflows size_t");
    }
    return capacity * gaps_per_record;
}

} // namespace

struct DiscontinuityPool::Impl
{
    Impl(std::size_t n_slots, std::size_t gap_capacity)
        : registry(n_slots), gaps_per_record(gap_capacity),
          gaps(
              std::make_unique_for_overwrite<SignalGap[]>(checked_gap_count(n_slots, gap_capacity)))
    {
    }

    detail::SlotRegistry registry;
    std::size_t gaps_per_record{};
    std::unique_ptr<SignalGap[]> gaps;
};

DiscontinuityLease::~DiscontinuityLease() noexcept
{
    static_cast<void>(reset());
}

DiscontinuityLease::DiscontinuityLease(DiscontinuityLease&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), token_(other.token_),
      gap_storage_(other.gap_storage_), value_(other.value_)
{
    other.gap_storage_ = {};
    other.value_ = {};
}

DiscontinuityLease& DiscontinuityLease::operator=(DiscontinuityLease&& other) noexcept
{
    if (this != &other)
    {
        static_cast<void>(reset());
        owner_ = std::exchange(other.owner_, nullptr);
        token_ = other.token_;
        gap_storage_ = other.gap_storage_;
        value_ = other.value_;
        other.gap_storage_ = {};
        other.value_ = {};
    }
    return *this;
}

StreamStatus DiscontinuityLease::assign(SessionId session_id, std::uint64_t previous_frame_sequence,
                                        std::uint64_t actual_frame_sequence, GapReason reason,
                                        std::span<const SignalGap> gaps) noexcept
{
    if (owner_ == nullptr || gaps.size() > gap_storage_.size())
    {
        return StreamStatus::invalid_frame;
    }
    std::copy(gaps.begin(), gaps.end(), gap_storage_.begin());
    value_ = Discontinuity{
        .session_id = session_id,
        .previous_frame_sequence = previous_frame_sequence,
        .actual_frame_sequence = actual_frame_sequence,
        .signal_gaps = gap_storage_.first(gaps.size()),
        .reason = reason,
    };
    return StreamStatus::ok;
}

StreamStatus DiscontinuityLease::reset() noexcept
{
    if (owner_ == nullptr)
    {
        return StreamStatus::stopped;
    }
    auto* owner = std::exchange(owner_, nullptr);
    const auto token = token_;
    gap_storage_ = {};
    value_ = {};
    return owner->release(token);
}

DiscontinuityPool::DiscontinuityPool(std::size_t capacity, std::size_t gaps_per_record)
    : impl_(std::make_unique<Impl>(capacity, gaps_per_record))
{
}

DiscontinuityPool::~DiscontinuityPool()
{
    if (impl_->registry.outstanding() != 0)
    {
        std::terminate();
    }
}

StreamStatus DiscontinuityPool::try_acquire(DiscontinuityLease& lease) noexcept
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
    lease.token_ = token;
    lease.gap_storage_ = {
        impl_->gaps.get() + slot * impl_->gaps_per_record,
        impl_->gaps_per_record,
    };
    return StreamStatus::ok;
}

std::size_t DiscontinuityPool::capacity() const noexcept
{
    return impl_->registry.capacity();
}

std::size_t DiscontinuityPool::gaps_per_record() const noexcept
{
    return impl_->gaps_per_record;
}

std::size_t DiscontinuityPool::outstanding() const noexcept
{
    return impl_->registry.outstanding();
}

void DiscontinuityPool::prefault() noexcept
{
    std::fill_n(impl_->gaps.get(), impl_->registry.capacity() * impl_->gaps_per_record,
                SignalGap{});
}

StreamStatus DiscontinuityPool::release(BufferToken token) noexcept
{
    return impl_->registry.release(token);
}

} // namespace neurale::streaming
