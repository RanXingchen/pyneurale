/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include <neurale/streaming/frame.h>
#include <neurale/streaming/frame_emitter.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/schema.h>

/// The sampled in-place adapters' shared data plane.
///
/// FIR, IIR, SOS and common referencing differ in what arithmetic they apply and
/// in nothing else on the data plane: each validates the frame, reinterprets the
/// single block's payload as a mutable span of doubles, runs its kernel over it
/// and forwards the input frame unchanged.
///
/// This header is separate from `adapter_support.h` so the boundary that matters
/// most here is a file boundary: **everything in `adapter_support.h` is
/// prepare-time and may throw; everything below runs on the real-time thread and
/// may not.** It mirrors `feature_adapter_stream.h`, which draws the same line
/// for the windowed-feature adapters.
///
/// The contract that follows from that:
///
/// * `process_inplace_frame` is a template, not a virtual call. The kernel
///   invocation is supplied by the caller as an invocable and resolved at
///   compile time -- no vtable, no type erasure and no `std::function` is
///   introduced by sharing this code. The only indirect call in the compiled
///   body is `FrameEmitter::publish_input`, which is pure virtual on the
///   caller-supplied emitter; nothing here adds indirection beyond that.
/// * It allocates nothing. It writes through the frame it was handed and hands
///   the same frame on. This is covered by
///   `neurale_pipeline_strict_realtime_contract_test --allocation-only`.
/// * It is `noexcept` and reports every ordinary outcome as a `StreamStatus`.
///   The kernel call is guarded, because a kernel's refusal is a result here,
///   not an error path -- `SosRealtimeProcessor::process` and
///   `IirRealtimeProcessor::process` are not `noexcept`, and an exception
///   escaping a `noexcept` adapter would terminate the process rather than fail
///   the frame.
///
/// :class:`InplaceStreamState` is a plain aggregate with no virtual functions.
/// Each adapter's `Impl` inherits it so that prepare-time code keeps naming the
/// fields directly; inheritance here is field reuse, not dispatch.
namespace neurale::pipeline::adapter_support
{

/// The per-stream state an in-place adapter's data plane reads.
///
/// Holds no processor: the kernel type differs per adapter and is reached
/// through the invocable passed to `process_inplace_frame`, which is what keeps
/// this struct free of any template parameter or virtual function.
struct InplaceStreamState
{
    std::unique_ptr<streaming::FrameValidator> validator;
    std::unique_ptr<streaming::StreamSchema> prepared_schema;
    std::size_t n_channels{};
};

/// Run one input frame's samples through *invoke* in place and forward the frame.
///
/// *invoke* is called as `invoke(std::span<double> samples, std::size_t
/// n_samples)`, with `samples.size() == n_samples * state.n_channels`. It is
/// where the per-adapter call shape lives: the FIR adapter drives its kernel one
/// sample at a time and the others hand it the whole block, and that difference
/// is a property of the kernel, not of the streaming archetype.
///
/// *prepared* is the adapter's own answer to "has prepare() run": each owns its
/// processor, and this header deliberately cannot see it.
template <typename Invoke>
[[nodiscard]] streaming::StreamStatus
process_inplace_frame(const InplaceStreamState& state, bool prepared, streaming::FrameBorrow& frame,
                      streaming::FrameEmitter& emitter, Invoke&& invoke) noexcept
{
    if (!prepared || state.validator == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    if (state.validator->validate(frame.view()) != streaming::FrameValidationError::none)
    {
        return streaming::StreamStatus::invalid_frame;
    }
    const auto& block = frame.blocks().front();
    auto* samples = reinterpret_cast<double*>(frame.payload().data() + block.payload_offset);
    const auto n_samples = static_cast<std::size_t>(block.n_samples);
    try
    {
        invoke(std::span<double>{samples, n_samples * state.n_channels}, n_samples);
    }
    catch (...)
    {
        return streaming::StreamStatus::processor_failure;
    }
    return emitter.publish_input();
}

} // namespace neurale::pipeline::adapter_support
