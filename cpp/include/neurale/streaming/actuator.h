/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include <neurale/streaming/fault.h>
#include <neurale/streaming/frame.h>

namespace neurale::streaming
{

using ActuatorCommandId = std::uint64_t;

struct ActuatorCommand
{
    ActuatorCommandId id{};
    std::uint64_t sequence{};
    HostTimeNs generated_at_ns{};
    HostTimeNs valid_until_ns{};
    SessionId session_id{};
    std::uint64_t runtime_generation{};
    FrameView payload{};

    [[nodiscard]] bool expired(HostTimeNs now_ns) const noexcept
    {
        return now_ns >= valid_until_ns;
    }
};

/// Safety-critical native endpoint. submit() enforces command expiry before
/// entering the device-specific write implementation.
class NativeActuator
{
  public:
    virtual ~NativeActuator() = default;

    [[nodiscard]] StreamStatus submit(const ActuatorCommand& command, HostTimeNs now_ns) noexcept
    {
        return command.expired(now_ns) ? StreamStatus::deadline_exceeded : write(command);
    }

    virtual StreamStatus flush() noexcept = 0;
    virtual StreamStatus reset() noexcept = 0;
    virtual void cancel() noexcept = 0;

  private:
    virtual StreamStatus write(const ActuatorCommand& command) noexcept = 0;
};

static_assert(std::is_trivially_copyable_v<ActuatorCommand>);

} // namespace neurale::streaming
