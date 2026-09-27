/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <span>
#include <type_traits>

#include <neurale/streaming/clock.h>
#include <neurale/streaming/schema.h>

namespace neurale::streaming
{

/// Primary explanation for a native continuity break.
enum class GapReason : std::uint8_t
{
    frame_sequence_gap,
    sample_gap,
    device_tick_gap,
    device_restart,
    source_gap,
    buffer_exhausted,
    queue_overflow,
};

enum class SignalGapFlags : std::uint8_t
{
    none = 0,
    missing_samples_known = 1U << 0U,
    device_ticks_available = 1U << 1U,
};

[[nodiscard]] constexpr SignalGapFlags operator|(SignalGapFlags left, SignalGapFlags right) noexcept
{
    return static_cast<SignalGapFlags>(static_cast<std::uint8_t>(left) |
                                       static_cast<std::uint8_t>(right));
}

[[nodiscard]] constexpr bool has_flag(SignalGapFlags value, SignalGapFlags flag) noexcept
{
    return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(flag)) != 0U;
}

/// Sample-centered continuity information for one affected signal.
struct SignalGap
{
    SampleIndex expected_sample_idx{};
    SampleIndex actual_sample_idx{};
    std::uint64_t missing_samples{};
    DeviceTick expected_device_tick{};
    DeviceTick actual_device_tick{};
    SignalId signal_id{};
    GapReason reason{GapReason::frame_sequence_gap};
    SignalGapFlags flags{SignalGapFlags::none};
};

/// Ordered view backed by one leased discontinuity-pool slot.
struct Discontinuity
{
    SessionId session_id{};
    std::uint64_t previous_frame_sequence{};
    std::uint64_t actual_frame_sequence{};
    std::span<const SignalGap> signal_gaps{};
    GapReason reason{GapReason::frame_sequence_gap};
};

static_assert(std::is_trivially_copyable_v<SignalGap>);
static_assert(std::is_trivially_copyable_v<Discontinuity>);

} // namespace neurale::streaming
