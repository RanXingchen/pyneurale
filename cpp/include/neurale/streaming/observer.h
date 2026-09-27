/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <neurale/streaming/discontinuity.h>
#include <neurale/streaming/fault.h>
#include <neurale/streaming/frame.h>

namespace neurale::streaming
{

using ObserverId = std::uint32_t;

enum class ObserverDropPolicy : std::uint8_t
{
    drop_oldest,
    latest_value,
    drop_newest,
    /// Lossless until fault. Required for a critical observer edge.
    fault,
};

struct ObserverEdgeConfig
{
    ObserverId id{};
    std::size_t capacity{};
    std::size_t drop_history_capacity{8};
    ObserverDropPolicy drop_policy{ObserverDropPolicy::drop_newest};
    bool critical_recorder{};
};

struct ObserverDropRange
{
    std::uint64_t first_sequence{};
    std::uint64_t last_sequence{};
};

struct ObserverEdgeStats
{
    ObserverId id{};
    std::uint64_t enqueued{};
    std::uint64_t delivered{};
    std::uint64_t discontinuities{};
    std::uint64_t dropped{};
    std::uint64_t drop_range_count{};
    std::uint64_t drop_history_dropped{};
    std::uint64_t failures{};
    std::uint64_t high_water_mark{};
    std::uint64_t last_enqueued_sequence{};
    std::uint64_t last_delivered_sequence{};
    bool detached{};
};

/// Noncritical native endpoint reached only through its own bounded edge.
class NativeObserver
{
  public:
    virtual ~NativeObserver() = default;

    virtual StreamStatus observe(FrameView frame) noexcept = 0;
    virtual StreamStatus handle_discontinuity(const Discontinuity& discontinuity) noexcept = 0;
    virtual StreamStatus flush() noexcept = 0;
    virtual StreamStatus reset() noexcept = 0;
    virtual void cancel() noexcept = 0;
};

} // namespace neurale::streaming
