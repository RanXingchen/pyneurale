// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "input_runtime.h"

namespace neurale::experiment_presentation
{

SurfaceStatus FixedInputQueue::prepare(std::size_t capacity)
{
    if (capacity == 0)
    {
        return SurfaceStatus::invalid_configuration;
    }
    storage_.assign(capacity, InputEvent{});
    reset();
    return SurfaceStatus::ok;
}

SurfaceStatus FixedInputQueue::push(const PendingInputEvent& pending,
                                    RendererTimeMapper& time_mapper) noexcept
{
    if (status_ != SurfaceStatus::ok)
    {
        return status_;
    }
    if (size_ == storage_.size())
    {
        status_ = SurfaceStatus::input_overflow;
        return status_;
    }
    experiments::ExperimentTimeNs experiment_time_ns{};
    const auto mapping_status = time_mapper.map(pending.renderer_time_ns, experiment_time_ns);
    if (mapping_status != SurfaceStatus::ok)
    {
        status_ = mapping_status;
        return status_;
    }
    const auto tail = (head_ + size_) % storage_.size();
    storage_[tail] = InputEvent{
        .ordinal = next_ordinal_++,
        .renderer_time_ns = pending.renderer_time_ns,
        .experiment_time_ns = experiment_time_ns,
        .kind = pending.kind,
        .action = pending.action,
        .code = pending.code,
        .modifiers = pending.modifiers,
        .pointer = pending.pointer,
        .size = pending.size,
        .pointer_inside = pending.pointer_inside,
    };
    ++size_;
    return SurfaceStatus::ok;
}

bool FixedInputQueue::pop(InputEvent& event) noexcept
{
    if (size_ == 0)
    {
        return false;
    }
    event = storage_[head_];
    head_ = (head_ + 1) % storage_.size();
    --size_;
    return true;
}

void FixedInputQueue::reset() noexcept
{
    head_ = 0;
    size_ = 0;
    next_ordinal_ = 0;
    status_ = storage_.empty() ? SurfaceStatus::invalid_state : SurfaceStatus::ok;
}

void FixedInputQueue::close() noexcept
{
    storage_.clear();
    storage_.shrink_to_fit();
    head_ = 0;
    size_ = 0;
    next_ordinal_ = 0;
    status_ = SurfaceStatus::invalid_state;
}

} // namespace neurale::experiment_presentation
