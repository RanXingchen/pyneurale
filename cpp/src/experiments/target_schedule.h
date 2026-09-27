// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <neurale/experiments/schedule.h>

namespace neurale::experiments::detail
{
/// Shared Center-Out/SSVEP Fisher-Yates cycle. Preserve the original draw addressing.
template <std::size_t Capacity>
[[nodiscard]] ContractStatus balanced_cycle_index(ScheduleSeed seed, DrawStream stream,
                                                  TrialOrdinal trial, std::uint8_t count,
                                                  std::uint8_t& index) noexcept
{
    if (count == 0 || count > Capacity)
        return ContractStatus::target_set_invalid;
    std::array<std::uint8_t, Capacity> order{};
    for (std::uint8_t slot = 0; slot < count; ++slot)
        order[slot] = slot;
    for (std::uint64_t remaining = count; remaining > 1; --remaining)
    {
        std::uint64_t selected = 0;
        const DrawKey key{seed, stream, trial / count,
                          static_cast<std::uint32_t>(count - remaining)};
        if (const auto status = sample_index(key, remaining, selected);
            status != ContractStatus::ok)
            return status;
        const auto last = static_cast<std::size_t>(remaining - 1);
        const auto chosen = static_cast<std::size_t>(selected);
        const auto temporary = order[last];
        order[last] = order[chosen];
        order[chosen] = temporary;
    }
    index = order[static_cast<std::size_t>(trial % count)];
    return ContractStatus::ok;
}
} // namespace neurale::experiments::detail
