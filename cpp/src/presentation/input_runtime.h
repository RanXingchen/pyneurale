// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <neurale/experiments/contract.h>

#include "surface_types.h"
#include "time_mapper.h"

namespace neurale::experiment_presentation
{

enum class InputEventKind : std::uint8_t
{
    pointer_moved = 0,
    mouse_button,
    key,
    window_close,
    framebuffer_resized,
};

enum class InputAction : std::uint8_t
{
    none = 0,
    press,
    release,
    repeat,
};

struct InputEvent
{
    std::uint64_t ordinal{};
    RendererTimeNs renderer_time_ns{};
    experiments::ExperimentTimeNs experiment_time_ns{};
    InputEventKind kind{InputEventKind::pointer_moved};
    InputAction action{InputAction::none};
    int code{};
    int modifiers{};
    Point2d pointer{};
    Size2i size{};
    bool pointer_inside{};
};

struct PendingInputEvent
{
    RendererTimeNs renderer_time_ns{};
    InputEventKind kind{InputEventKind::pointer_moved};
    InputAction action{InputAction::none};
    int code{};
    int modifiers{};
    Point2d pointer{};
    Size2i size{};
    bool pointer_inside{};
};

/// Per-presenter input admission policy, checked in the GLFW callbacks before an
/// event ever reaches the FixedInputQueue. A single glfwPollEvents() runs every
/// registered callback synchronously, so a pointer burst can overflow the queue
/// (a sticky input_overflow fault) before the presenter gets to drain. Admission
/// drops the kinds a presenter does not consume right at the callback, so a Speech
/// run, which has no pointer/selection semantics, cannot be faulted by pointer
/// motion regardless of how the orchestrator schedules drains. Resize never
/// enters the queue at all: the framebuffer-size callback only refreshes the
/// coordinate mapper, it does not enqueue, since no concrete presenter consumes
/// resize as orchestration evidence. The bits align with InputEventKind so admits
/// maps a kind to its bit by a single shift.
enum class InputAdmission : std::uint8_t
{
    none = 0,
    pointer_moved = 1 << 0,
    mouse_button = 1 << 1,
    key = 1 << 2,
    window_close = 1 << 3,
    // pointer_moved | mouse_button | key | window_close, written as a literal so
    // the enum initializer does not depend on operator| being declared yet.
    all = 0x0F,
};

[[nodiscard]] constexpr InputAdmission operator|(InputAdmission left, InputAdmission right) noexcept
{
    return static_cast<InputAdmission>(static_cast<std::uint8_t>(left) |
                                       static_cast<std::uint8_t>(right));
}

[[nodiscard]] constexpr InputAdmission operator&(InputAdmission left, InputAdmission right) noexcept
{
    return static_cast<InputAdmission>(static_cast<std::uint8_t>(left) &
                                       static_cast<std::uint8_t>(right));
}

[[nodiscard]] constexpr bool admits(InputAdmission mask, InputEventKind kind) noexcept
{
    return (static_cast<std::uint8_t>(mask) &
            (std::uint8_t{1} << static_cast<std::uint8_t>(kind))) != 0;
}

class FixedInputQueue
{
  public:
    [[nodiscard]] SurfaceStatus prepare(std::size_t capacity);
    [[nodiscard]] SurfaceStatus push(const PendingInputEvent& event,
                                     RendererTimeMapper& time_mapper) noexcept;
    [[nodiscard]] bool pop(InputEvent& event) noexcept;
    void reset() noexcept;
    void close() noexcept;

    [[nodiscard]] SurfaceStatus status() const noexcept
    {
        return status_;
    }
    [[nodiscard]] std::size_t size() const noexcept
    {
        return size_;
    }
    [[nodiscard]] std::size_t capacity() const noexcept
    {
        return storage_.size();
    }

  private:
    std::vector<InputEvent> storage_{};
    std::size_t head_{};
    std::size_t size_{};
    std::uint64_t next_ordinal_{};
    SurfaceStatus status_{SurfaceStatus::invalid_state};
};

} // namespace neurale::experiment_presentation
