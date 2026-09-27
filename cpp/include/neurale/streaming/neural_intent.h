/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <cstdint>

namespace neurale::streaming
{

inline constexpr char kNeuralIntentSourceCapsule[] = "neurale.NeuralIntentSource.v1";

struct NeuralIntentSnapshot
{
    std::uint64_t sequence{};
    double intent_x{};
    double intent_y{};
    std::uint64_t context_ordinal{};
    std::uint64_t source_time_ns{};
    std::uint64_t source_frame_sequence{};
    std::uint64_t source_sample_index{};
    bool valid{};
};

class NeuralIntentSource
{
  public:
    virtual ~NeuralIntentSource() = default;
    [[nodiscard]] virtual NeuralIntentSnapshot read_intent() const noexcept = 0;
};

/** Single-writer, multi-reader allocation-free intent handoff. */
class NeuralIntentState final : public NeuralIntentSource
{
  public:
    void publish(const NeuralIntentSnapshot& snapshot) noexcept
    {
        revision_.fetch_add(1, std::memory_order_acq_rel);
        std::atomic_thread_fence(std::memory_order_release);
        sequence_.store(snapshot.sequence, std::memory_order_relaxed);
        intent_x_.store(snapshot.intent_x, std::memory_order_relaxed);
        intent_y_.store(snapshot.intent_y, std::memory_order_relaxed);
        context_ordinal_.store(snapshot.context_ordinal, std::memory_order_relaxed);
        source_time_ns_.store(snapshot.source_time_ns, std::memory_order_relaxed);
        source_frame_sequence_.store(snapshot.source_frame_sequence, std::memory_order_relaxed);
        source_sample_index_.store(snapshot.source_sample_index, std::memory_order_relaxed);
        valid_.store(snapshot.valid, std::memory_order_relaxed);
        revision_.fetch_add(1, std::memory_order_release);
    }

    [[nodiscard]] NeuralIntentSnapshot read_intent() const noexcept override
    {
        NeuralIntentSnapshot result{};
        for (;;)
        {
            const auto before = revision_.load(std::memory_order_acquire);
            if ((before & 1U) != 0U)
                continue;
            result.sequence = sequence_.load(std::memory_order_relaxed);
            result.intent_x = intent_x_.load(std::memory_order_relaxed);
            result.intent_y = intent_y_.load(std::memory_order_relaxed);
            result.context_ordinal = context_ordinal_.load(std::memory_order_relaxed);
            result.source_time_ns = source_time_ns_.load(std::memory_order_relaxed);
            result.source_frame_sequence = source_frame_sequence_.load(std::memory_order_relaxed);
            result.source_sample_index = source_sample_index_.load(std::memory_order_relaxed);
            result.valid = valid_.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (revision_.load(std::memory_order_acquire) == before)
                return result;
        }
    }

    void reset() noexcept
    {
        publish({});
    }

  private:
    std::atomic<std::uint64_t> revision_{};
    std::atomic<std::uint64_t> sequence_{};
    std::atomic<double> intent_x_{};
    std::atomic<double> intent_y_{};
    std::atomic<std::uint64_t> context_ordinal_{};
    std::atomic<std::uint64_t> source_time_ns_{};
    std::atomic<std::uint64_t> source_frame_sequence_{};
    std::atomic<std::uint64_t> source_sample_index_{};
    std::atomic<bool> valid_{};
};

} // namespace neurale::streaming
