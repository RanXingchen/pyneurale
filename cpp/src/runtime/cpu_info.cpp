/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/runtime/runtime_info.h>

#include <array>
#include <cstring>
#include <thread>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#include <cpuid.h>
#endif

namespace neurale::runtime
{
namespace
{

std::string architecture()
{
#if defined(_M_X64) || defined(__x86_64__)
    return "x86_64";
#elif defined(_M_IX86) || defined(__i386__)
    return "x86";
#elif defined(_M_ARM64) || defined(__aarch64__)
    return "arm64";
#elif defined(_M_ARM) || defined(__arm__)
    return "arm";
#else
    return "unknown";
#endif
}

bool cpuid(std::uint32_t leaf, std::array<std::uint32_t, 4>& registers)
{
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    std::array<int, 4> values{};
    __cpuid(values.data(), static_cast<int>(leaf));
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        registers[i] = static_cast<std::uint32_t>(values[i]);
    }
    return true;
#elif (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
    return __get_cpuid(leaf, &registers[0], &registers[1], &registers[2], &registers[3]) != 0;
#else
    (void)leaf;
    (void)registers;
    return false;
#endif
}

std::optional<std::string> cpu_vendor()
{
    std::array<std::uint32_t, 4> registers{};
    if (!cpuid(0, registers))
    {
        return std::nullopt;
    }

    std::array<char, 13> vendor{};
    std::memcpy(vendor.data(), &registers[1], 4);
    std::memcpy(vendor.data() + 4, &registers[3], 4);
    std::memcpy(vendor.data() + 8, &registers[2], 4);
    return std::string(vendor.data());
}

std::optional<std::string> cpu_model()
{
    std::array<std::uint32_t, 4> maximum{};
    if (!cpuid(0x80000000U, maximum) || maximum[0] < 0x80000004U)
    {
        return std::nullopt;
    }

    std::array<char, 49> brand{};
    for (std::uint32_t leaf = 0; leaf < 3; ++leaf)
    {
        std::array<std::uint32_t, 4> registers{};
        if (!cpuid(0x80000002U + leaf, registers))
        {
            return std::nullopt;
        }
        std::memcpy(brand.data() + leaf * 16, registers.data(), 16);
    }

    std::string result(brand.data());
    const auto first = result.find_first_not_of(' ');
    const auto last = result.find_last_not_of(' ');
    if (first == std::string::npos)
    {
        return std::nullopt;
    }
    return result.substr(first, last - first + 1);
}

} // namespace

CpuInfo cpu_info()
{
    const auto cores = std::thread::hardware_concurrency();
    return CpuInfo{
        architecture(),
        cpu_vendor(),
        cpu_model(),
        cores == 0 ? 1U : cores,
    };
}

} // namespace neurale::runtime
