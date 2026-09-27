/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/clock.h>

namespace neurale::streaming
{

HostTimeNs SteadyNativeClock::now_ns() noexcept
{
    return static_cast<HostTimeNs>(std::chrono::duration_cast<RealtimeDuration>(
                                       std::chrono::steady_clock::now().time_since_epoch())
                                       .count());
}

void SteadyNativeClock::wait_until(HostTimeNs deadline_ns) noexcept
{
    const auto deadline = std::chrono::steady_clock::time_point{RealtimeDuration{deadline_ns}};
    std::unique_lock lock(mutex_);
    const auto revision = revision_.load(std::memory_order_acquire);
    condition_.wait_until(lock, deadline, [this, revision]
                          { return revision_.load(std::memory_order_acquire) != revision; });
}

void SteadyNativeClock::wake() noexcept
{
    revision_.fetch_add(1, std::memory_order_release);
    condition_.notify_all();
}

NativeClock& default_native_clock() noexcept
{
    static SteadyNativeClock clock;
    return clock;
}

} // namespace neurale::streaming
