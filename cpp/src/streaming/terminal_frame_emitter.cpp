/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "terminal_frame_emitter.h"

#include <utility>

namespace neurale::streaming
{

TerminalFrameEmitter::TerminalFrameEmitter(FramePool& pool, FrameValidator& validator,
                                           SpscRing<StreamMessage>& critical_edge,
                                           RuntimeCounters& stats, NativeClock& clock,
                                           std::atomic<std::uint64_t>& critical_epoch) noexcept
    : pool_(pool), validator_(validator), critical_edge_(critical_edge), stats_(stats),
      clock_(clock), critical_epoch_(critical_epoch)
{
}

void TerminalFrameEmitter::begin(FrameLease input, std::size_t output_limit, bool input_forwarding,
                                 bool flushing) noexcept
{
    input_ = std::move(input);
    invalidate_borrow(input_borrow_state_);
    invalidate_acquired_borrow();
    static_cast<void>(acquired_.reset());
    output_limit_ = output_limit;
    published_count_ = 0;
    failure_ = StreamStatus::ok;
    failure_code_ = FaultCode::none;
    active_ = true;
    input_forwarding_ = input_forwarding;
    flushing_ = flushing;
    input_borrow_ = {};
    if (input_)
    {
        const auto status = activate_borrow(input_borrow_state_, input_.frame(), input_borrow_);
        if (status != StreamStatus::ok)
        {
            remember_failure(status, FaultCode::output_contract);
        }
    }
}

FrameBorrow& TerminalFrameEmitter::input() noexcept
{
    return input_borrow_;
}

void TerminalFrameEmitter::remember_failure(StreamStatus status, FaultCode code) noexcept
{
    if (failure_ == StreamStatus::ok)
    {
        failure_ = status;
        failure_code_ = code;
    }
}

StreamStatus TerminalFrameEmitter::try_acquire_frame(MutableFrame*& frame) noexcept
{
    frame = nullptr;
    if (!active_ || acquired_)
    {
        remember_failure(StreamStatus::invalid_state, FaultCode::output_contract);
        return StreamStatus::invalid_state;
    }
    if (failure_ != StreamStatus::ok)
    {
        return failure_;
    }
    if (published_count_ >= output_limit_)
    {
        remember_failure(StreamStatus::output_limit, FaultCode::output_contract);
        return StreamStatus::output_limit;
    }
    const auto status = pool_.try_acquire(acquired_);
    if (status != StreamStatus::ok)
    {
        if (status == StreamStatus::buffer_exhausted)
        {
            stats_.pool_exhaustion();
        }
        remember_failure(status, FaultCode::output_pool_exhausted);
        return status;
    }
    frame = &acquired_.frame();
    return StreamStatus::ok;
}

StreamStatus TerminalFrameEmitter::publish(FrameLease& lease) noexcept
{
    if (!active_ || !lease)
    {
        remember_failure(StreamStatus::invalid_state, FaultCode::output_contract);
        return StreamStatus::invalid_state;
    }
    if (failure_ != StreamStatus::ok)
    {
        static_cast<void>(lease.reset());
        return failure_;
    }
    if (published_count_ >= output_limit_)
    {
        static_cast<void>(lease.reset());
        remember_failure(StreamStatus::output_limit, FaultCode::output_contract);
        return StreamStatus::output_limit;
    }
    if (validator_.validate(lease.view()) != FrameValidationError::none)
    {
        static_cast<void>(lease.reset());
        remember_failure(StreamStatus::invalid_frame, FaultCode::output_validation);
        return StreamStatus::invalid_frame;
    }

    const auto generated_at = clock_.now_ns();
    auto message = StreamMessage::from_frame(std::move(lease), generated_at);
    const auto status = critical_edge_.try_push(std::move(message));
    if (status != StreamStatus::ok)
    {
        remember_failure(status, FaultCode::actuator_queue_overrun);
        return status;
    }
    ++published_count_;
    stats_.published(flushing_);
    stats_.actuator_enqueued();
    stats_.observe_actuator_size(critical_edge_.approximate_size());
    critical_epoch_.fetch_add(1, std::memory_order_release);
    critical_epoch_.notify_one();
    return StreamStatus::ok;
}

StreamStatus TerminalFrameEmitter::publish_acquired_frame() noexcept
{
    return publish(acquired_);
}

StreamStatus TerminalFrameEmitter::publish_input() noexcept
{
    if (!input_forwarding_)
    {
        remember_failure(StreamStatus::invalid_state, FaultCode::output_contract);
        return StreamStatus::invalid_state;
    }
    invalidate_borrow(input_borrow_state_);
    return publish(input_);
}

StreamStatus TerminalFrameEmitter::publish_owned(FrameLease lease) noexcept
{
    return publish(lease);
}

StreamStatus TerminalFrameEmitter::finish() noexcept
{
    if (!active_)
    {
        return StreamStatus::invalid_state;
    }
    if (acquired_)
    {
        invalidate_acquired_borrow();
        static_cast<void>(acquired_.reset());
        remember_failure(StreamStatus::invalid_state, FaultCode::output_contract);
    }
    invalidate_borrow(input_borrow_state_);
    static_cast<void>(input_.reset());
    active_ = false;
    return failure_;
}

} // namespace neurale::streaming
