/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

#include <neurale/streaming/fault.h>
#include <neurale/streaming/schema.h>

namespace neurale::pipeline
{

inline constexpr std::size_t kMaxCapturedDecoderFeatures = 256;

struct DecoderTrainingObservation
{
    streaming::SampleIndex sample_idx{};
    std::uint32_t n_features{};
    std::array<double, kMaxCapturedDecoderFeatures> values{};
};

class DecoderTrainingCapture final
{
  public:
    void prepare(std::size_t capacity, std::size_t n_features)
    {
        if (capacity == 0 || n_features == 0 || n_features > kMaxCapturedDecoderFeatures)
            throw std::invalid_argument("invalid decoder training capture shape");
        if (capacity >
            (std::numeric_limits<std::size_t>::max)() / sizeof(DecoderTrainingObservation))
            throw std::length_error("decoder training capture storage overflows size_t");
        storage_ = std::make_unique_for_overwrite<DecoderTrainingObservation[]>(capacity);
        capacity_ = capacity;
        n_features_ = n_features;
    }

    [[nodiscard]] bool can_push(std::size_t count) const noexcept
    {
        if (storage_ == nullptr || count > capacity_)
            return false;
        const auto produced = produced_.load(std::memory_order_acquire);
        const auto consumed = consumed_.load(std::memory_order_acquire);
        return produced - consumed <= capacity_ - count;
    }

    [[nodiscard]] streaming::StreamStatus try_push(streaming::SampleIndex sample_idx,
                                                   std::span<const double> values) noexcept
    {
        if (storage_ == nullptr || values.size() != n_features_)
            return streaming::StreamStatus::invalid_state;
        const auto produced = produced_.load(std::memory_order_relaxed);
        const auto consumed = consumed_.load(std::memory_order_acquire);
        if (produced - consumed == capacity_)
        {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return streaming::StreamStatus::queue_overflow;
        }
        auto& output = storage_[produced % capacity_];
        output.sample_idx = sample_idx;
        output.n_features = static_cast<std::uint32_t>(n_features_);
        std::copy(values.begin(), values.end(), output.values.begin());
        produced_.store(produced + 1, std::memory_order_release);
        return streaming::StreamStatus::ok;
    }

    [[nodiscard]] streaming::StreamStatus try_pop(DecoderTrainingObservation& value) noexcept
    {
        const auto consumed = consumed_.load(std::memory_order_relaxed);
        const auto produced = produced_.load(std::memory_order_acquire);
        if (consumed == produced)
            return streaming::StreamStatus::would_block;
        value = storage_[consumed % capacity_];
        consumed_.store(consumed + 1, std::memory_order_release);
        return streaming::StreamStatus::ok;
    }

    void note_dropped(std::size_t count) noexcept
    {
        dropped_.fetch_add(count, std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t dropped() const noexcept
    {
        return dropped_.load(std::memory_order_relaxed);
    }

  private:
    std::unique_ptr<DecoderTrainingObservation[]> storage_{};
    std::size_t capacity_{};
    std::size_t n_features_{};
    alignas(64) std::atomic<std::size_t> produced_{};
    alignas(64) std::atomic<std::size_t> consumed_{};
    alignas(64) std::atomic<std::uint64_t> dropped_{};
};

} // namespace neurale::pipeline
