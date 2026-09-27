/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace neurale::streaming
{

enum class RealtimeConfigMode : std::uint8_t
{
    disabled,
    best_effort,
    strict,
};

enum class RealtimeSchedulingPolicy : std::uint8_t
{
    normal,
    fifo,
    round_robin,
};

enum class RealtimeThreadRole : std::uint8_t
{
    acquisition,
    processing,
    actuator,
    observer_dispatch,
    watchdog,
    count,
};

enum class RealtimeFeature : std::uint32_t
{
    none = 0,
    thread_name = 1U << 0U,
    cpu_affinity = 1U << 1U,
    scheduling = 1U << 2U,
    priority = 1U << 3U,
    stack_size = 1U << 4U,
    memory_lock = 1U << 5U,
    stack_prefault = 1U << 6U,
    pool_prefault = 1U << 7U,
};

[[nodiscard]] constexpr std::uint32_t feature_mask(RealtimeFeature feature)
{
    return static_cast<std::uint32_t>(feature);
}

struct RealtimeThreadConfig
{
    static constexpr std::size_t name_capacity = 16;

    std::array<char, name_capacity> name{};
    std::uint64_t cpu_affinity_mask{};
    RealtimeSchedulingPolicy scheduling_policy{RealtimeSchedulingPolicy::normal};
    std::int32_t priority{};
    std::size_t stack_size{};
    std::size_t prefault_stack_bytes{};

    void set_name(const char* value);
    [[nodiscard]] std::uint32_t requested_features() const noexcept;
    void validate() const;
};

struct RealtimePlatformConfig
{
    RealtimeConfigMode mode{RealtimeConfigMode::disabled};
    bool lock_memory{};
    bool prefault_pools{};
    RealtimeThreadConfig acquisition{};
    RealtimeThreadConfig processing{};
    RealtimeThreadConfig actuator{};
    RealtimeThreadConfig observer_dispatch{};
    RealtimeThreadConfig watchdog{};

    [[nodiscard]] const RealtimeThreadConfig& thread(RealtimeThreadRole role) const noexcept;
    void validate() const;
};

struct RealtimePlatformCapabilities
{
    std::uint32_t supported_features{};
    std::uint32_t logical_cpu_count{};
    std::uint32_t max_thread_name_length{};
    bool linux{};
    bool windows{};
    bool hard_realtime_guaranteed{};

    [[nodiscard]] bool supports(RealtimeFeature feature) const noexcept
    {
        return (supported_features & feature_mask(feature)) != 0;
    }
};

struct RealtimeApplyResult
{
    std::uint32_t requested{};
    std::uint32_t applied{};
    std::uint32_t unsupported{};
    std::uint32_t failed{};
    std::int32_t native_error{};

    [[nodiscard]] bool ok() const noexcept
    {
        return unsupported == 0 && failed == 0;
    }
};

struct RealtimeConfigurationStatus
{
    RealtimeApplyResult memory{};
    std::array<RealtimeApplyResult, static_cast<std::size_t>(RealtimeThreadRole::count)> threads{};

    [[nodiscard]] bool ok() const noexcept;
    [[nodiscard]] std::uint32_t warning_count() const noexcept;
    [[nodiscard]] const RealtimeApplyResult& thread(RealtimeThreadRole role) const noexcept
    {
        return threads[static_cast<std::size_t>(role)];
    }
};

class RealtimePlatform
{
  public:
    virtual ~RealtimePlatform() = default;

    [[nodiscard]] virtual RealtimePlatformCapabilities capabilities() const noexcept = 0;
    [[nodiscard]] virtual RealtimeApplyResult
    configure_process_memory(const RealtimePlatformConfig& config) noexcept = 0;
    [[nodiscard]] virtual RealtimeApplyResult
    configure_current_thread(const RealtimeThreadConfig& config) noexcept = 0;
};

[[nodiscard]] RealtimePlatform& default_realtime_platform() noexcept;
[[nodiscard]] RealtimePlatformCapabilities realtime_platform_capabilities() noexcept;

} // namespace neurale::streaming
