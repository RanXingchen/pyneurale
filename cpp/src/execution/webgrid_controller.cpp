/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "webgrid_controller.h"

#include "contract_status.h"

namespace neurale::execution
{

using namespace neurale::experiments;

streaming::StreamStatus WebGridHeadlessController::prepare(const WebGridControllerConfig& config)
{
    if (prepared_ || closed_)
        return streaming::StreamStatus::invalid_state;
    // A severity nothing declared is refused where the run freezes it, like
    // every other persisted enumeration in the contract.
    if (config.trace_capacity == 0 || validate(config.abnormal) != ContractStatus::ok ||
        webgrid::validate(config.task) != ContractStatus::ok)
        return streaming::StreamStatus::realtime_configuration_failed;
    traces_.prepare(config.trace_capacity);
    presentation_failures_.prepare(64);
    config_ = config;
    prepared_ = true;
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus WebGridHeadlessController::start(ParadigmId paradigm,
                                                         ExperimentTimeNs time_ns) noexcept
{
    if (!prepared_ || running_ || closed_)
        return streaming::StreamStatus::invalid_state;
    traces_.reset();
    presentation_failures_.reset();
    machine_.reset();
    cancelled_.store(false, std::memory_order_release);
    halted_.store(false, std::memory_order_release);
    abnormal_.configure(config_.abnormal, paradigm);
    abnormal_.restart();
    last_pointer_ns_ = time_ns;
    has_pointer_ = false;
    target_invalidated_ = false;
    next_pointer_update_ordinal_ = 0;
    last_pointer_update_ordinal_ = 0;
    has_pointer_update_ordinal_ = false;
    halt_refusal_recorded_ = false;
    halt_pending_ = false;
    webgrid::WebGridStepResult result{};
    const auto status = machine_.start(paradigm, config_.task, time_ns, result);
    if (status != ContractStatus::ok)
        return map_contract_status(status);
    WebGridHeadlessTrace trace{};
    trace.kind = WebGridTraceKind::session_start;
    trace.step = result;
    const auto push_status = traces_.try_push(trace);
    if (push_status != streaming::StreamStatus::ok)
    {
        machine_.reset();
        return push_status;
    }
    running_ = true;
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus
WebGridHeadlessController::process(ExperimentTimeNs time_ns,
                                   const webgrid::PointerPosition& pointer) noexcept
{
    return process_impl(time_ns, pointer, nullptr, nullptr);
}

streaming::StreamStatus WebGridHeadlessController::process(ExperimentTimeNs time_ns,
                                                           const webgrid::PointerPosition& pointer,
                                                           const SelectionEvent& selection) noexcept
{
    return process_impl(time_ns, pointer, &selection, nullptr);
}

streaming::StreamStatus WebGridHeadlessController::process(
    ExperimentTimeNs time_ns, const webgrid::PointerPosition& pointer,
    const WebGridPresentationInputEvidence& presentation_input) noexcept
{
    return process_impl(time_ns, pointer, nullptr, &presentation_input);
}

streaming::StreamStatus WebGridHeadlessController::process(
    ExperimentTimeNs time_ns, const webgrid::PointerPosition& pointer,
    const SelectionEvent& selection,
    const WebGridPresentationInputEvidence& presentation_input) noexcept
{
    return process_impl(time_ns, pointer, &selection, &presentation_input);
}

streaming::StreamStatus WebGridHeadlessController::process_impl(
    ExperimentTimeNs time_ns, const webgrid::PointerPosition& pointer,
    const SelectionEvent* selection,
    const WebGridPresentationInputEvidence* presentation_input) noexcept
{
    if (!prepared_ || closed_)
        return streaming::StreamStatus::invalid_state;
    if (presentation_input != nullptr &&
        (!presentation_input->available || presentation_input->experiment_time_ns != time_ns))
        return streaming::StreamStatus::invalid_frame;
    if (cancelled_.load(std::memory_order_acquire))
        return streaming::StreamStatus::stopped;
    // Input that arrives after the run reached a terminal state is refused, and
    // that includes a selection. A selection accepted after a fault would be
    // counted, scored, and recorded as though the run were still the run.
    if (halted_.load(std::memory_order_acquire))
        return refuse_after_halt(time_ns);
    // Budgeted: this runs on the input/task path, not at shutdown.
    const auto presentation_status = apply_presentation_failures(kPresentationFailureDrainBudget);
    if (presentation_status != streaming::StreamStatus::ok)
        return presentation_status;
    if (!running_)
        return streaming::StreamStatus::invalid_state;
    if (machine_.complete())
        return streaming::StreamStatus::stopped;

    // A pointer that stopped arriving for longer than this run will accept
    // breaks the only continuity WebGrid measures. Checked before the step, so
    // the record of the gap precedes the record of the observation that
    // revealed it.
    const bool crossed_gap = config_.max_pointer_interval_ns != 0 && has_pointer_ &&
                             time_ns >= last_pointer_ns_ &&
                             time_ns - last_pointer_ns_ > config_.max_pointer_interval_ns;
    // Room for everything this input will produce, checked before it changes
    // anything. An input that crosses a gap produces two records, and reserving
    // one would let the gap be reported -- counted, and the target marked
    // inadmissible -- and then the step refused for want of room. The caller
    // would see `queue_overflow` for an input the run had already reacted to,
    // and retrying it would report the same gap a second time, because nothing
    // that advances `last_pointer_ns_` had run yet.
    const std::size_t required = crossed_gap ? 2 : 1;
    if (!traces_.can_push(required))
    {
        // Refused before the machine is stepped, so the step never happens and
        // the records it would have produced are the records that are lost.
        traces_.note_dropped(required);
        return streaming::StreamStatus::queue_overflow;
    }

    if (crossed_gap)
    {
        const auto gap =
            report_abnormal(AbnormalCondition::input_gap, time_ns, AbnormalPolicy::record, false,
                            static_cast<std::uint32_t>(time_ns - last_pointer_ns_));
        if (gap != streaming::StreamStatus::ok)
            return gap;
    }

    webgrid::WebGridStepResult result{};
    const auto status = selection == nullptr ? machine_.step(time_ns, pointer, result)
                                             : machine_.step(time_ns, pointer, *selection, result);
    if (status != ContractStatus::ok)
        return map_contract_status(status);
    WebGridHeadlessTrace trace{};
    trace.kind =
        selection == nullptr ? WebGridTraceKind::pointer_update : WebGridTraceKind::selection;
    trace.pointer_update_ordinal = next_pointer_update_ordinal_;
    trace.pointer = pointer;
    trace.step = result;
    trace.trial_invalidated = target_invalidated_;
    trace.acquisition_timing_valid = !target_invalidated_;
    if (presentation_input != nullptr)
    {
        trace.presentation_input = *presentation_input;
    }
    const auto push_status = traces_.try_push(trace);
    if (push_status != streaming::StreamStatus::ok)
        return push_status;
    last_pointer_ns_ = time_ns;
    has_pointer_ = true;
    last_pointer_update_ordinal_ = next_pointer_update_ordinal_;
    has_pointer_update_ordinal_ = true;
    ++next_pointer_update_ordinal_;
    // The next target starts clean. What was invalidated was this target's
    // acquisition interval, and the interval ended here.
    if (result.trial_decided)
        target_invalidated_ = false;
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus WebGridHeadlessController::report_abnormal(AbnormalCondition condition,
                                                                   ExperimentTimeNs time_ns,
                                                                   AbnormalPolicy floor,
                                                                   bool input_refused,
                                                                   std::uint32_t detail) noexcept
{
    if (!traces_.can_push())
    {
        traces_.note_dropped(1);
        return streaming::StreamStatus::queue_overflow;
    }
    const auto snapshot = machine_.snapshot();
    const bool in_trial = running_ && !machine_.complete();

    WebGridHeadlessTrace trace{};
    trace.kind = WebGridTraceKind::abnormal;
    // WebGridMachine owns when a target ends, and it offers no way to abandon
    // one. So an `abort_trial` policy here invalidates the target in flight
    // rather than ending it, and the record says `trial_invalidated` -- which
    // is the truth, and is what stops the eventual selection from being read as
    // an ordinary success.
    trace.abnormal = abnormal_.observe(condition, time_ns, in_trial ? &snapshot.trial : nullptr,
                                       detail, floor, /*can_end_trial=*/false, input_refused);
    if (trace.abnormal.response == AbnormalResponse::session_aborted)
        // See CenterOutController::apply_abnormal: the latch is what ends the
        // run, and the fatal status returned below only tells a caller about it.
        stop_accepting();
    if (trace.abnormal.response == AbnormalResponse::trial_invalidated)
        target_invalidated_ = true;
    trace.trial_invalidated = target_invalidated_;
    trace.acquisition_timing_valid = !target_invalidated_;
    const auto push = traces_.try_push(trace);
    if (push != streaming::StreamStatus::ok)
        return push;
    return trace.abnormal.response == AbnormalResponse::session_aborted
               ? streaming::StreamStatus::consumer_failure
               : streaming::StreamStatus::ok;
}

streaming::StreamStatus WebGridHeadlessController::note_input_gap(ExperimentTimeNs time_ns,
                                                                  std::uint32_t detail) noexcept
{
    if (!prepared_ || closed_)
        return streaming::StreamStatus::invalid_state;
    if (cancelled_.load(std::memory_order_acquire))
        return streaming::StreamStatus::stopped;
    if (halted_.load(std::memory_order_acquire))
        return refuse_after_halt(time_ns);
    if (!running_)
        return streaming::StreamStatus::invalid_state;
    return report_abnormal(AbnormalCondition::input_gap, time_ns, AbnormalPolicy::record, false,
                           detail);
}

streaming::StreamStatus WebGridHeadlessController::note_observer_drop(ExperimentTimeNs time_ns,
                                                                      std::uint32_t detail) noexcept
{
    if (!prepared_ || closed_)
        return streaming::StreamStatus::invalid_state;
    if (cancelled_.load(std::memory_order_acquire))
        return streaming::StreamStatus::stopped;
    if (!running_)
        return streaming::StreamStatus::invalid_state;
    // No floor, and policy_for() gives this condition `record` whatever the
    // configuration says. It is recorded and it changes nothing.
    return report_abnormal(AbnormalCondition::observer_frame_drop, time_ns, AbnormalPolicy::record,
                           false, detail);
}

streaming::StreamStatus
WebGridHeadlessController::refuse_after_halt(ExperimentTimeNs time_ns) noexcept
{
    if (!halt_refusal_recorded_)
    {
        halt_refusal_recorded_ = true;
        static_cast<void>(report_abnormal(AbnormalCondition::input_after_terminal, time_ns,
                                          AbnormalPolicy::record, true, 0));
    }
    return streaming::StreamStatus::stopped;
}

void WebGridHeadlessController::halt(ExperimentTimeNs time_ns, AbnormalCondition condition) noexcept
{
    if (closed_ || halted_.exchange(true, std::memory_order_acq_rel))
        return;
    halt_time_ns_ = time_ns;
    halt_condition_ = condition;
    halt_pending_ = true;
}

void WebGridHeadlessController::finish_halt() noexcept
{
    if (!halt_pending_)
        return;
    halt_pending_ = false;
    if (!running_)
        return;
    static_cast<void>(
        report_abnormal(halt_condition_, halt_time_ns_, AbnormalPolicy::abort_trial, true, 0));
    running_ = false;
}

streaming::StreamStatus WebGridHeadlessController::note_presentation_failure(
    const PresentationFailureEvidence& evidence) noexcept
{
    if (!prepared_ || closed_ || !running_)
        return streaming::StreamStatus::invalid_state;
    if (cancelled_.load(std::memory_order_acquire))
        return streaming::StreamStatus::stopped;
    if (halted_.load(std::memory_order_acquire))
        return refuse_after_halt(evidence.time_ns);
    const auto snapshot = machine_.snapshot();
    if (machine_.complete() || !same_trial(snapshot.trial, evidence.trial))
        return streaming::StreamStatus::invalid_frame;
    return report_abnormal(AbnormalCondition::presentation_failed, evidence.time_ns,
                           AbnormalPolicy::record, false, evidence.implementation_status);
}

streaming::StreamStatus
WebGridHeadlessController::apply_presentation_failures(std::size_t budget) noexcept
{
    return drain_presentation_failures(presentation_failures_, budget,
                                       [this](const PresentationFailureEvidence& evidence) noexcept
                                       { return note_presentation_failure(evidence); });
}

streaming::StreamStatus WebGridHeadlessController::reset() noexcept
{
    if (!prepared_ || closed_)
        return streaming::StreamStatus::invalid_state;
    machine_.reset();
    traces_.reset();
    presentation_failures_.reset();
    running_ = false;
    last_pointer_ns_ = 0;
    has_pointer_ = false;
    target_invalidated_ = false;
    next_pointer_update_ordinal_ = 0;
    last_pointer_update_ordinal_ = 0;
    has_pointer_update_ordinal_ = false;
    halt_refusal_recorded_ = false;
    halt_pending_ = false;
    cancelled_.store(false, std::memory_order_release);
    halted_.store(false, std::memory_order_release);
    return streaming::StreamStatus::ok;
}

void WebGridHeadlessController::cancel() noexcept
{
    cancelled_.store(true, std::memory_order_release);
}

void WebGridHeadlessController::close() noexcept
{
    if (closed_)
        return;
    cancel();
    running_ = false;
    traces_.close();
    presentation_failures_.close();
    closed_ = true;
}

} // namespace neurale::execution
