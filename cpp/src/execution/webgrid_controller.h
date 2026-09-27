/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include <neurale/experiments/abnormal.h>
#include <neurale/experiments/webgrid.h>
#include <neurale/streaming/fault.h>

#include "abnormal_reporter.h"
#include "bounded_trace_queue.h"
#include "presentation_evidence.h"

namespace neurale::execution
{

using namespace neurale::experiments;

enum class WebGridTraceKind : std::uint8_t
{
    session_start,
    pointer_update,
    selection,
    /// A condition the task rules do not describe, and what was decided about
    /// it.
    abnormal,
};

struct WebGridHeadlessTrace
{
    WebGridTraceKind kind{WebGridTraceKind::pointer_update};
    /// Stable producer-assigned identity for a pointer/selection update.
    SequenceOrdinal pointer_update_ordinal{};
    webgrid::PointerPosition pointer{};
    webgrid::WebGridStepResult step{};
    /// Meaningful for ::WebGridTraceKind::abnormal.
    AbnormalEvent abnormal{};
    /// Whether the target in flight when this record was produced had already
    /// been invalidated by an abnormal condition.
    ///
    /// Carried on every record rather than only on the one that decided a
    /// trial, because a reader scanning pointer samples is entitled to know
    /// which of them belong to a target whose timing is no longer a
    /// measurement.
    bool trial_invalidated{};
    /// Whether the acquisition interval this record contributes to is still a
    /// continuous observation of the pointer.
    ///
    /// WebGrid's published metric is time from target onset to selection, and
    /// that number means nothing across an interval in which the pointer was
    /// not being observed. It is false for exactly the targets an ::input_gap
    /// crossed.
    bool acquisition_timing_valid{true};
    /// Presenter callback identity, when this task input came from the native
    /// presentation runtime. Headless inputs deliberately leave it absent.
    WebGridPresentationInputEvidence presentation_input{};
};

struct WebGridControllerConfig
{
    /// The task itself.
    webgrid::WebGridConfig task{};
    /// How severely this run treats each class of abnormal condition.
    AbnormalPolicySet abnormal{};
    /// Longest interval between accepted pointer observations that an
    /// acquisition time survives. Zero means the caller declares no bound.
    ///
    /// Unlike Center-Out there is nothing being integrated here, so a gap does
    /// not corrupt a position -- it corrupts a *duration*. A target whose onset
    /// is separated from its selection by an interval in which nobody was
    /// watching the pointer has an acquisition time that is not the quantity
    /// the metric is defined as.
    DurationNs max_pointer_interval_ns{};
    std::size_t trace_capacity{};
};

/// Noncritical, renderer-free WebGrid controller. A combined call always
/// applies the pointer observation first and evaluates the SelectionEvent at
/// that same position; WebGridMachine remains the sole owner of selection and
/// target semantics.
class WebGridHeadlessController final
{
  public:
    WebGridHeadlessController() noexcept = default;

    WebGridHeadlessController(const WebGridHeadlessController&) = delete;
    WebGridHeadlessController& operator=(const WebGridHeadlessController&) = delete;

    [[nodiscard]] streaming::StreamStatus prepare(const WebGridControllerConfig& config);
    /// Prepare a run that declares no abnormal policy and no pointer-interval
    /// bound, which is the defaulted ::WebGridControllerConfig.
    [[nodiscard]] streaming::StreamStatus prepare(const webgrid::WebGridConfig& config,
                                                  std::size_t trace_capacity)
    {
        return prepare(WebGridControllerConfig{.task = config, .trace_capacity = trace_capacity});
    }
    [[nodiscard]] streaming::StreamStatus start(ParadigmId paradigm,
                                                ExperimentTimeNs time_ns) noexcept;
    [[nodiscard]] streaming::StreamStatus process(ExperimentTimeNs time_ns,
                                                  const webgrid::PointerPosition& pointer) noexcept;
    [[nodiscard]] streaming::StreamStatus process(ExperimentTimeNs time_ns,
                                                  const webgrid::PointerPosition& pointer,
                                                  const SelectionEvent& selection) noexcept;
    [[nodiscard]] streaming::StreamStatus
    process(ExperimentTimeNs time_ns, const webgrid::PointerPosition& pointer,
            const WebGridPresentationInputEvidence& presentation_input) noexcept;
    [[nodiscard]] streaming::StreamStatus
    process(ExperimentTimeNs time_ns, const webgrid::PointerPosition& pointer,
            const SelectionEvent& selection,
            const WebGridPresentationInputEvidence& presentation_input) noexcept;
    [[nodiscard]] streaming::StreamStatus reset() noexcept;
    void cancel() noexcept;
    void close() noexcept;

    /// Report a gap in the pointer data this task is being driven from.
    ///
    /// The caller owns the pointer source and is the only thing that can know
    /// its data stopped arriving, so this is reported to the task rather than
    /// detected by it -- except for the interval bound, which the task does
    /// check for itself on every step.
    [[nodiscard]] streaming::StreamStatus note_input_gap(ExperimentTimeNs time_ns,
                                                         std::uint32_t detail) noexcept;

    /// Report a drop on an observation or rendering path that carried no task
    /// input.
    ///
    /// Deliberately a different call from ::note_input_gap, and not a flag on
    /// it. A dropped monitoring or UI frame did not change what the task was
    /// given; recording it through the same entry point as a pointer gap would
    /// make the two indistinguishable in the record, and every dropped frame
    /// would then invalidate a perfectly good acquisition time. This one can
    /// never invalidate anything, whatever the configuration says.
    [[nodiscard]] streaming::StreamStatus note_observer_drop(ExperimentTimeNs time_ns,
                                                             std::uint32_t detail) noexcept;

    /// Refuse all further task input from this instant. See
    /// CenterOutController::halt for why this is not cancel(), and why it only
    /// latches.
    void halt(ExperimentTimeNs time_ns, AbnormalCondition condition) noexcept;

    /// Produce the record a latched halt owes. See
    /// CenterOutController::finish_halt.
    void finish_halt() noexcept;

    [[nodiscard]] bool
    enqueue_presentation_failure(const PresentationFailureEvidence& evidence) noexcept
    {
        return presentation_failures_.try_push(evidence) == streaming::StreamStatus::ok;
    }
    /// Apply queued failures on the task-owner thread.
    ///
    /// @param budget Maximum number applied by this call.
    ///        ::kUnboundedPresentationFailureDrain drains all of them and is for
    ///        shutdown; the input path passes
    ///        ::kPresentationFailureDrainBudget so one pointer/selection call
    ///        cannot absorb a whole queue's worth of abnormal decisions.
    [[nodiscard]] streaming::StreamStatus
    apply_presentation_failures(std::size_t budget = kUnboundedPresentationFailureDrain) noexcept;

    [[nodiscard]] bool latest_pointer_update_ordinal(SequenceOrdinal& ordinal) const noexcept
    {
        if (!has_pointer_update_ordinal_)
            return false;
        ordinal = last_pointer_update_ordinal_;
        return true;
    }

    /// Latch the terminal state, recording nothing. See
    /// CenterOutController::stop_accepting for what an `abort_session`
    /// response *is*, why this is distinct from halt(), and why it writes no
    /// record of its own.
    void stop_accepting() noexcept
    {
        halted_.store(true, std::memory_order_release);
    }

    /// Whether input is being refused because the run reached a terminal state.
    [[nodiscard]] bool halted() const noexcept
    {
        return halted_.load(std::memory_order_acquire);
    }

    /// Abnormal conditions this controller has met, ever. Monotonic.
    [[nodiscard]] AbnormalSummary abnormal_summary() const noexcept
    {
        return abnormal_.summary();
    }

    [[nodiscard]] streaming::StreamStatus try_pop_trace(WebGridHeadlessTrace& trace) noexcept
    {
        return traces_.try_pop(trace);
    }
    [[nodiscard]] webgrid::WebGridSnapshot snapshot() const noexcept
    {
        return machine_.snapshot();
    }
    [[nodiscard]] bool prepared() const noexcept
    {
        return prepared_;
    }

    /// Trace records this controller could not enqueue, ever. Monotonic.
    [[nodiscard]] std::uint64_t dropped_trace_count() const noexcept
    {
        return traces_.dropped() + presentation_failures_.dropped();
    }

    /// The configuration prepare() froze. See CenterOutController for why the
    /// recording bridge reads it from here rather than from its own copy.
    [[nodiscard]] const webgrid::WebGridConfig& configuration() const noexcept
    {
        return config_.task;
    }
    /// The whole prepared configuration, including the abnormal policies.
    [[nodiscard]] const WebGridControllerConfig& controller_configuration() const noexcept
    {
        return config_;
    }

  private:
    [[nodiscard]] streaming::StreamStatus
    process_impl(ExperimentTimeNs time_ns, const webgrid::PointerPosition& pointer,
                 const SelectionEvent* selection,
                 const WebGridPresentationInputEvidence* presentation_input) noexcept;
    [[nodiscard]] streaming::StreamStatus report_abnormal(AbnormalCondition condition,
                                                          ExperimentTimeNs time_ns,
                                                          AbnormalPolicy floor, bool input_refused,
                                                          std::uint32_t detail) noexcept;
    [[nodiscard]] streaming::StreamStatus
    note_presentation_failure(const PresentationFailureEvidence& evidence) noexcept;
    [[nodiscard]] streaming::StreamStatus refuse_after_halt(ExperimentTimeNs time_ns) noexcept;

    WebGridControllerConfig config_{};
    AbnormalReporter abnormal_{};
    webgrid::WebGridMachine machine_{};
    BoundedTraceQueue<WebGridHeadlessTrace> traces_{};
    BoundedTraceQueue<PresentationFailureEvidence> presentation_failures_{};
    ExperimentTimeNs last_pointer_ns_{};
    bool has_pointer_{};
    /// Whether an abnormal condition has invalidated the target in flight.
    /// Cleared when the machine decides a trial and the next target begins.
    bool target_invalidated_{};
    SequenceOrdinal next_pointer_update_ordinal_{};
    SequenceOrdinal last_pointer_update_ordinal_{};
    bool has_pointer_update_ordinal_{};
    bool prepared_{};
    bool running_{};
    bool closed_{};
    bool halt_refusal_recorded_{};
    ExperimentTimeNs halt_time_ns_{};
    AbnormalCondition halt_condition_{AbnormalCondition::unspecified};
    bool halt_pending_{};
    std::atomic<bool> cancelled_{};
    std::atomic<bool> halted_{};
};

} // namespace neurale::execution
