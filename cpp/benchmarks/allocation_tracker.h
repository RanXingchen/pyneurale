/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <string_view>

namespace neurale::benchmark
{

void reset_allocation_count() noexcept;
void set_allocation_tracking(bool enabled) noexcept;
[[nodiscard]] std::uint64_t allocation_count() noexcept;
[[nodiscard]] std::string_view allocation_tracking_backend() noexcept;

class AllocationScope
{
  public:
    AllocationScope() noexcept;
    ~AllocationScope() noexcept;

    AllocationScope(const AllocationScope&) = delete;
    AllocationScope& operator=(const AllocationScope&) = delete;

    [[nodiscard]] std::uint64_t count() const noexcept;
    void stop() noexcept;

  private:
    bool active_{true};
};

} // namespace neurale::benchmark
