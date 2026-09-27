/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <utility>

#include <neurale/streaming/discontinuity.h>
#include <neurale/streaming/fault.h>
#include <neurale/streaming/frame.h>
#include <neurale/streaming/frame_emitter.h>
#include <neurale/streaming/schema.h>

namespace neurale::streaming
{

/// Bounded resources retained or used concurrently by a prepared processor.
struct ProcessorResourceBounds
{
    std::size_t workspace_bytes{};
    std::size_t frame_pool_leases{};
};

/// Runtime capacities visible while a processor declares its prepared contract.
struct ProcessorPrepareContext
{
    const StreamSchema& input_schema;
    std::size_t max_process_outputs{};
    std::size_t max_flush_outputs{};
    std::size_t available_frame_pool_leases{};
};

/// Immutable, owning schema and output bounds returned by processor prepare.
struct PreparedProcessorContract
{
    StreamSchema accepted_input_schema;
    StreamSchema output_schema;
    std::size_t max_process_outputs_per_input{};
    std::size_t max_flush_outputs{};
    bool can_forward_input{};
    ProcessorResourceBounds required_resources{};
};

/// Fixed-shape native processor prepared before realtime startup.
class NativeFrameProcessor
{
  public:
    virtual ~NativeFrameProcessor() = default;

    /// Allocate control-plane state and declare the immutable data-plane bounds.
    [[nodiscard]] virtual PreparedProcessorContract
    prepare(const ProcessorPrepareContext& context) = 0;

    /// Process one callback-scoped frame borrow. Retaining the borrow or any
    /// derived view, span, or pointer beyond this call violates the contract.
    virtual StreamStatus process(FrameBorrow& frame, FrameEmitter& emitter) noexcept = 0;

    /// Establish a callback scope when invoking a processor directly.
    [[nodiscard]] StreamStatus process(MutableFrame& frame, FrameEmitter& emitter) noexcept
    {
        FrameBorrowScope scope{direct_borrow_state_, frame};
        if (!scope.valid())
        {
            return StreamStatus::invalid_state;
        }
        return process(scope.borrow(), emitter);
    }

    virtual StreamStatus handle_discontinuity(const Discontinuity& discontinuity) noexcept = 0;

    virtual StreamStatus flush(FrameEmitter& emitter) noexcept = 0;

    virtual StreamStatus reset() noexcept = 0;

  private:
    detail::FrameBorrowState direct_borrow_state_{};
};

} // namespace neurale::streaming
