/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace neurale::streaming
{

/// Stable identifier for one native streaming session.
using SessionId = std::uint64_t;

/// Stable identifier for one device or host clock domain.
using ClockDomainId = std::uint32_t;

/// Monotonic device counter supplied by a native source.
using DeviceTick = std::uint64_t;

/// Absolute sample position within one stream schema.
using SampleIndex = std::uint64_t;

/// Monotonic host timestamp represented as nanoseconds in the runtime clock.
using HostTimeNs = std::uint64_t;

/// Exact positive rate used for sample and device-clock domains.
struct RationalRate
{
    std::uint64_t numerator{};
    std::uint64_t denominator{1};
};

enum class ClockSyncFlags : std::uint32_t
{
    none = 0,
    synchronized = 1U << 0U,
};

[[nodiscard]] constexpr bool has_flag(ClockSyncFlags value, ClockSyncFlags flag) noexcept
{
    return (static_cast<std::uint32_t>(value) & static_cast<std::uint32_t>(flag)) != 0U;
}

/// Fixed snapshot mapping one device-clock reference point to host time.
struct ClockSyncSnapshot
{
    DeviceTick device_tick_reference{};
    HostTimeNs host_time_reference_ns{};
    RationalRate device_tick_rate{};
    HostTimeNs uncertainty_ns{};
    ClockDomainId clock_domain{};
    std::uint32_t generation{};
    ClockSyncFlags flags{ClockSyncFlags::none};
};

/// Duration type used by native streaming configuration and diagnostics.
using RealtimeDuration = std::chrono::nanoseconds;

/// Monotonic native clock used by runtime stages and the watchdog.
class NativeClock
{
  public:
    virtual ~NativeClock() = default;

    [[nodiscard]] virtual HostTimeNs now_ns() noexcept = 0;
    virtual void wait_until(HostTimeNs deadline_ns) noexcept = 0;
    virtual void wake() noexcept = 0;
};

/// Default host-monotonic clock. Waiting is only used off realtime callbacks.
class SteadyNativeClock final : public NativeClock
{
  public:
    [[nodiscard]] HostTimeNs now_ns() noexcept override;
    void wait_until(HostTimeNs deadline_ns) noexcept override;
    void wake() noexcept override;

  private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<std::uint64_t> revision_{};
};

[[nodiscard]] NativeClock& default_native_clock() noexcept;

} // namespace neurale::streaming
