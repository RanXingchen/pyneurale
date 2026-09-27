/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/realtime_thread.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <thread>

#if defined(__linux__)
#include <alloca.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#elif defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

namespace neurale::streaming
{
namespace
{

constexpr auto all_thread_features =
    feature_mask(RealtimeFeature::thread_name) | feature_mask(RealtimeFeature::cpu_affinity) |
    feature_mask(RealtimeFeature::scheduling) | feature_mask(RealtimeFeature::priority) |
    feature_mask(RealtimeFeature::stack_size) | feature_mask(RealtimeFeature::stack_prefault);

void mark_unsupported(RealtimeApplyResult& result, std::uint32_t supported)
{
    result.unsupported |= result.requested & ~supported;
}

#if defined(__linux__)
class NativeRealtimePlatform final : public RealtimePlatform
{
  public:
    [[nodiscard]] RealtimePlatformCapabilities capabilities() const noexcept override
    {
        return {
            .supported_features = feature_mask(RealtimeFeature::thread_name) |
                                  feature_mask(RealtimeFeature::cpu_affinity) |
                                  feature_mask(RealtimeFeature::scheduling) |
                                  feature_mask(RealtimeFeature::priority) |
                                  feature_mask(RealtimeFeature::memory_lock) |
                                  feature_mask(RealtimeFeature::stack_prefault) |
                                  feature_mask(RealtimeFeature::pool_prefault),
            .logical_cpu_count = std::thread::hardware_concurrency(),
            .max_thread_name_length = 15,
            .linux = true,
        };
    }

    [[nodiscard]] RealtimeApplyResult
    configure_process_memory(const RealtimePlatformConfig& config) noexcept override
    {
        RealtimeApplyResult result{};
        if (config.lock_memory)
        {
            result.requested |= feature_mask(RealtimeFeature::memory_lock);
            if (::mlockall(MCL_CURRENT | MCL_FUTURE) == 0)
            {
                result.applied |= feature_mask(RealtimeFeature::memory_lock);
            }
            else
            {
                result.failed |= feature_mask(RealtimeFeature::memory_lock);
                result.native_error = errno;
            }
        }
        if (config.prefault_pools)
        {
            result.requested |= feature_mask(RealtimeFeature::pool_prefault);
            result.applied |= feature_mask(RealtimeFeature::pool_prefault);
        }
        return result;
    }

    [[nodiscard]] RealtimeApplyResult
    configure_current_thread(const RealtimeThreadConfig& config) noexcept override
    {
        RealtimeApplyResult result{.requested = config.requested_features()};
        mark_unsupported(result, capabilities().supported_features);

        if (config.name[0] != '\0')
        {
            const auto error = ::pthread_setname_np(pthread_self(), config.name.data());
            record(result, RealtimeFeature::thread_name, error);
        }
        if (config.cpu_affinity_mask != 0)
        {
            cpu_set_t cpus;
            CPU_ZERO(&cpus);
            for (std::size_t cpu = 0; cpu < 64; ++cpu)
            {
                if ((config.cpu_affinity_mask & (std::uint64_t{1} << cpu)) != 0)
                {
                    CPU_SET(cpu, &cpus);
                }
            }
            record(result, RealtimeFeature::cpu_affinity,
                   ::pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus));
        }
        if (config.scheduling_policy != RealtimeSchedulingPolicy::normal || config.priority != 0)
        {
            const auto policy =
                config.scheduling_policy == RealtimeSchedulingPolicy::fifo          ? SCHED_FIFO
                : config.scheduling_policy == RealtimeSchedulingPolicy::round_robin ? SCHED_RR
                                                                                    : SCHED_OTHER;
            sched_param parameters{.sched_priority = config.priority};
            const auto error = ::pthread_setschedparam(pthread_self(), policy, &parameters);
            if (config.scheduling_policy != RealtimeSchedulingPolicy::normal)
            {
                record(result, RealtimeFeature::scheduling, error);
            }
            if (config.priority != 0)
            {
                record(result, RealtimeFeature::priority, error);
            }
        }
        if (config.prefault_stack_bytes != 0)
        {
            auto* memory = static_cast<volatile std::byte*>(alloca(config.prefault_stack_bytes));
            for (std::size_t offset = 0; offset < config.prefault_stack_bytes; offset += 4096)
            {
                memory[offset] = std::byte{};
            }
            memory[config.prefault_stack_bytes - 1] = std::byte{};
            result.applied |= feature_mask(RealtimeFeature::stack_prefault);
        }
        return result;
    }

  private:
    static void record(RealtimeApplyResult& result, RealtimeFeature feature, int error) noexcept
    {
        const auto mask = feature_mask(feature);
        if ((result.unsupported & mask) != 0)
        {
            return;
        }
        if (error == 0)
        {
            result.applied |= mask;
        }
        else
        {
            result.failed |= mask;
            if (result.native_error == 0)
            {
                result.native_error = error;
            }
        }
    }
};
#elif defined(_WIN32)
class NativeRealtimePlatform final : public RealtimePlatform
{
  public:
    [[nodiscard]] RealtimePlatformCapabilities capabilities() const noexcept override
    {
        return {
            .supported_features = feature_mask(RealtimeFeature::thread_name) |
                                  feature_mask(RealtimeFeature::cpu_affinity) |
                                  feature_mask(RealtimeFeature::priority) |
                                  feature_mask(RealtimeFeature::stack_prefault) |
                                  feature_mask(RealtimeFeature::pool_prefault),
            .logical_cpu_count = std::thread::hardware_concurrency(),
            .max_thread_name_length =
                static_cast<std::uint32_t>(RealtimeThreadConfig::name_capacity - 1),
            .windows = true,
        };
    }

    [[nodiscard]] RealtimeApplyResult
    configure_process_memory(const RealtimePlatformConfig& config) noexcept override
    {
        RealtimeApplyResult result{};
        if (config.lock_memory)
        {
            result.requested |= feature_mask(RealtimeFeature::memory_lock);
            result.unsupported |= feature_mask(RealtimeFeature::memory_lock);
        }
        if (config.prefault_pools)
        {
            result.requested |= feature_mask(RealtimeFeature::pool_prefault);
            result.applied |= feature_mask(RealtimeFeature::pool_prefault);
        }
        return result;
    }

    [[nodiscard]] RealtimeApplyResult
    configure_current_thread(const RealtimeThreadConfig& config) noexcept override
    {
        RealtimeApplyResult result{.requested = config.requested_features()};
        mark_unsupported(result, capabilities().supported_features);

        if (config.name[0] != '\0')
        {
            wchar_t name[RealtimeThreadConfig::name_capacity]{};
            std::size_t idx = 0;
            for (; idx + 1 < RealtimeThreadConfig::name_capacity && config.name[idx] != '\0'; ++idx)
            {
                name[idx] = static_cast<unsigned char>(config.name[idx]);
            }
            record(result, RealtimeFeature::thread_name,
                   ::SetThreadDescription(::GetCurrentThread(), name) == S_OK);
        }
        if (config.cpu_affinity_mask != 0)
        {
            record(result, RealtimeFeature::cpu_affinity,
                   ::SetThreadAffinityMask(::GetCurrentThread(),
                                           static_cast<DWORD_PTR>(config.cpu_affinity_mask)) != 0);
        }
        if (config.priority != 0)
        {
            record(result, RealtimeFeature::priority,
                   ::SetThreadPriority(::GetCurrentThread(), config.priority) != 0);
        }
        if (config.prefault_stack_bytes != 0)
        {
            auto* memory = static_cast<volatile std::byte*>(_alloca(config.prefault_stack_bytes));
            for (std::size_t offset = 0; offset < config.prefault_stack_bytes; offset += 4096)
            {
                memory[offset] = std::byte{};
            }
            memory[config.prefault_stack_bytes - 1] = std::byte{};
            result.applied |= feature_mask(RealtimeFeature::stack_prefault);
        }
        return result;
    }

  private:
    static void record(RealtimeApplyResult& result, RealtimeFeature feature, bool success) noexcept
    {
        const auto mask = feature_mask(feature);
        if ((result.unsupported & mask) != 0)
        {
            return;
        }
        if (success)
        {
            result.applied |= mask;
        }
        else
        {
            result.failed |= mask;
            if (result.native_error == 0)
            {
                result.native_error = static_cast<std::int32_t>(::GetLastError());
            }
        }
    }
};
#else
class NativeRealtimePlatform final : public RealtimePlatform
{
  public:
    [[nodiscard]] RealtimePlatformCapabilities capabilities() const noexcept override
    {
        return {.logical_cpu_count = std::thread::hardware_concurrency()};
    }

    [[nodiscard]] RealtimeApplyResult
    configure_process_memory(const RealtimePlatformConfig& config) noexcept override
    {
        RealtimeApplyResult result{};
        if (config.lock_memory)
        {
            result.requested |= feature_mask(RealtimeFeature::memory_lock);
        }
        if (config.prefault_pools)
        {
            result.requested |= feature_mask(RealtimeFeature::pool_prefault);
        }
        result.unsupported = result.requested;
        return result;
    }

    [[nodiscard]] RealtimeApplyResult
    configure_current_thread(const RealtimeThreadConfig& config) noexcept override
    {
        RealtimeApplyResult result{.requested = config.requested_features()};
        result.unsupported = result.requested & all_thread_features;
        return result;
    }
};
#endif

} // namespace

void RealtimeThreadConfig::set_name(const char* value)
{
    if (value == nullptr)
    {
        throw std::invalid_argument("thread name cannot be null");
    }
    const auto length = std::strlen(value);
    if (length >= name.size())
    {
        throw std::invalid_argument("thread name exceeds fixed platform limit");
    }
    name.fill('\0');
    std::copy_n(value, length, name.begin());
}

std::uint32_t RealtimeThreadConfig::requested_features() const noexcept
{
    std::uint32_t result{};
    if (name[0] != '\0')
    {
        result |= feature_mask(RealtimeFeature::thread_name);
    }
    if (cpu_affinity_mask != 0)
    {
        result |= feature_mask(RealtimeFeature::cpu_affinity);
    }
    if (scheduling_policy != RealtimeSchedulingPolicy::normal)
    {
        result |= feature_mask(RealtimeFeature::scheduling);
    }
    if (priority != 0)
    {
        result |= feature_mask(RealtimeFeature::priority);
    }
    if (stack_size != 0)
    {
        result |= feature_mask(RealtimeFeature::stack_size);
    }
    if (prefault_stack_bytes != 0)
    {
        result |= feature_mask(RealtimeFeature::stack_prefault);
    }
    return result;
}

void RealtimeThreadConfig::validate() const
{
    if (name.back() != '\0')
    {
        throw std::invalid_argument("thread name must be null terminated");
    }
    if (prefault_stack_bytes > stack_size && stack_size != 0)
    {
        throw std::invalid_argument("stack prefault size exceeds requested stack size");
    }
}

const RealtimeThreadConfig& RealtimePlatformConfig::thread(RealtimeThreadRole role) const noexcept
{
    switch (role)
    {
    case RealtimeThreadRole::acquisition:
        return acquisition;
    case RealtimeThreadRole::processing:
        return processing;
    case RealtimeThreadRole::actuator:
        return actuator;
    case RealtimeThreadRole::observer_dispatch:
        return observer_dispatch;
    case RealtimeThreadRole::watchdog:
    case RealtimeThreadRole::count:
        return watchdog;
    }
    return watchdog;
}

void RealtimePlatformConfig::validate() const
{
    acquisition.validate();
    processing.validate();
    actuator.validate();
    observer_dispatch.validate();
    watchdog.validate();
}

bool RealtimeConfigurationStatus::ok() const noexcept
{
    if (!memory.ok())
    {
        return false;
    }
    return std::all_of(threads.begin(), threads.end(),
                       [](const auto& result) { return result.ok(); });
}

std::uint32_t RealtimeConfigurationStatus::warning_count() const noexcept
{
    auto count = static_cast<std::uint32_t>(!memory.ok());
    for (const auto& result : threads)
    {
        count += static_cast<std::uint32_t>(!result.ok());
    }
    return count;
}

RealtimePlatform& default_realtime_platform() noexcept
{
    static NativeRealtimePlatform platform;
    return platform;
}

RealtimePlatformCapabilities realtime_platform_capabilities() noexcept
{
    return default_realtime_platform().capabilities();
}

} // namespace neurale::streaming
