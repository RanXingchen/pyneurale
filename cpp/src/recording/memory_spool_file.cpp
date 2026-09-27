/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "memory_spool_file.h"

#include <algorithm>
#include <cstddef>

namespace neurale::recording
{

MemorySpoolFile::MemorySpoolFile(std::size_t capacity_bytes) : capacity_(capacity_bytes)
{
    // Reserve the whole store now, on the caller's thread, so no later append
    // can reallocate. `reserve` is what makes every subsequent `insert` a plain
    // copy into memory that already exists.
    data_.reserve(capacity_bytes);
}

SpoolIoResult MemorySpoolFile::append(std::span<const std::byte> data) noexcept
{
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto remaining = capacity_ > data_.size() ? capacity_ - data_.size() : std::size_t{0};
    const auto take = std::min(data.size(), remaining);
    data_.insert(data_.end(), data.begin(), data.begin() + static_cast<std::ptrdiff_t>(take));
    if (take < data.size())
    {
        // The same shape a full disk reports: what was transferred, and why it
        // stopped. The writer decides what to do with a partial append; this
        // store does not pretend the rest fit.
        return SpoolIoResult{
            .status = SpoolIoStatus::out_of_space, .transferred = take, .platform_error = 0};
    }
    return SpoolIoResult{.status = SpoolIoStatus::ok, .transferred = take, .platform_error = 0};
}

SpoolIoResult MemorySpoolFile::read_at(std::uint64_t offset, std::span<std::byte> out) noexcept
{
    const std::lock_guard<std::mutex> guard(mutex_);
    if (offset >= data_.size())
    {
        return SpoolIoResult{
            .status = SpoolIoStatus::incomplete, .transferred = 0, .platform_error = 0};
    }
    const auto available = data_.size() - static_cast<std::size_t>(offset);
    const auto take = std::min(out.size(), available);
    std::copy_n(data_.begin() + static_cast<std::ptrdiff_t>(offset), take, out.begin());
    return SpoolIoResult{.status =
                             take == out.size() ? SpoolIoStatus::ok : SpoolIoStatus::incomplete,
                         .transferred = take,
                         .platform_error = 0};
}

SpoolIoResult MemorySpoolFile::sync() noexcept
{
    // Nothing to flush. See the header: this returning `ok` is not a durability
    // claim, and the store must be driven under `buffered` so the writer never
    // turns it into one.
    return SpoolIoResult{};
}

SpoolIoResult MemorySpoolFile::truncate(std::uint64_t bytes) noexcept
{
    const std::lock_guard<std::mutex> guard(mutex_);
    if (bytes < data_.size())
    {
        data_.resize(static_cast<std::size_t>(bytes));
    }
    return SpoolIoResult{};
}

std::uint64_t MemorySpoolFile::size() const noexcept
{
    const std::lock_guard<std::mutex> guard(mutex_);
    return data_.size();
}

std::vector<std::byte> MemorySpoolFile::snapshot() const
{
    const std::lock_guard<std::mutex> guard(mutex_);
    return data_;
}

} // namespace neurale::recording
