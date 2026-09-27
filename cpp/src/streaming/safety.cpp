/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/safety.h>

namespace neurale::streaming
{
namespace
{

class NoopSafetyController final : public SafetyController
{
  public:
    StreamStatus inhibit(SafetyReason) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus release() noexcept override
    {
        return StreamStatus::ok;
    }
};

} // namespace

SafetyController& default_safety_controller() noexcept
{
    static NoopSafetyController controller;
    return controller;
}

} // namespace neurale::streaming
