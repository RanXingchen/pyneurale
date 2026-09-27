/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <neurale/streaming/actuator.h>
#include <neurale/streaming/discontinuity.h>
#include <neurale/streaming/fault.h>
#include <neurale/streaming/frame.h>

namespace neurale::streaming
{

class NativeClock;

class NativeFrameConsumer : public NativeActuator
{
  public:
    virtual ~NativeFrameConsumer() = default;

    /// Bind the clock used by the runtime that will call this consumer.
    /// Consumers without host-timing evidence need not override this hook.
    virtual void bind_runtime_clock(NativeClock&) noexcept {}

    virtual StreamStatus consume(FrameView frame) noexcept = 0;

    virtual StreamStatus handle_discontinuity(const Discontinuity& discontinuity) noexcept = 0;

    virtual StreamStatus flush() noexcept = 0;

    virtual StreamStatus reset() noexcept = 0;

    void cancel() noexcept override {}

  private:
    StreamStatus write(const ActuatorCommand& command) noexcept final
    {
        return consume(command.payload);
    }
};

} // namespace neurale::streaming
