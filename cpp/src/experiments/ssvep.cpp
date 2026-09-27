// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "target_schedule.h"
#include <cmath>
#include <neurale/experiments/ssvep.h>

namespace neurale::experiments::ssvep
{
ContractStatus validate(const SSVEPTarget& target) noexcept
{
    if (target.id == kUnsetTargetId)
        return ContractStatus::identity_missing;
    if (!std::isfinite(target.frequency_hz))
        return ContractStatus::value_not_finite;
    return target.frequency_hz > 0 ? ContractStatus::ok : ContractStatus::parameter_out_of_range;
}

ContractStatus validate(const SSVEPConfig& config) noexcept
{
    if (config.n_targets < 2 || config.n_targets > kMaxSSVEPTargets)
        return ContractStatus::target_set_invalid;
    if (config.stimulus_id == kUnsetStimulusId)
        return ContractStatus::identity_missing;
    if (config.n_trials == 0 || config.cue_duration_ns == 0 ||
        config.stimulation_duration_ns == 0 || config.decision_timeout_ns == 0 ||
        config.feedback_duration_ns == 0 || config.inter_trial_ns == 0)
        return ContractStatus::range_empty;
    for (std::size_t i = 0; i < config.n_targets; ++i)
    {
        if (const auto status = validate(config.targets[i]); status != ContractStatus::ok)
            return status;
        for (std::size_t j = 0; j < i; ++j)
            if (config.targets[i].id == config.targets[j].id ||
                config.targets[i].frequency_hz == config.targets[j].frequency_hz)
                return ContractStatus::target_set_invalid;
    }
    // Unknown sampler versions remain readable; preparation refuses regeneration.
    return ContractStatus::ok;
}

ContractStatus validate(const SSVEPTrialSchedule& schedule) noexcept
{
    return schedule.target_id == kUnsetTargetId ? ContractStatus::identity_missing
                                                : ContractStatus::ok;
}

bool contains_target(const SSVEPConfig& config, TargetId target) noexcept
{
    if (config.n_targets > kMaxSSVEPTargets)
        return false;
    for (std::size_t i = 0; i < config.n_targets; ++i)
        if (config.targets[i].id == target)
            return true;
    return false;
}

ContractStatus prepare_trial(const SSVEPConfig& config, TrialOrdinal ordinal,
                             SSVEPTrialSchedule& schedule) noexcept
{
    if (const auto status = validate(config); status != ContractStatus::ok)
        return status;
    if (ordinal >= config.n_trials)
        return ContractStatus::outcome_invalid;
    if (!sampler_version_supported(config.sampler_version))
        return ContractStatus::version_unsupported;
    std::uint8_t index{};
    if (const auto status = detail::balanced_cycle_index<kMaxSSVEPTargets>(
            config.seed, kTargetSelectionStream, ordinal, config.n_targets, index);
        status != ContractStatus::ok)
        return status;
    schedule = {ordinal, config.targets[index].id};
    return ContractStatus::ok;
}

// State machine.
namespace
{
bool running(SSVEPState state) noexcept
{
    return state != SSVEPState::idle && state != SSVEPState::complete;
}
SSVEPPhase phase_of(SSVEPState state) noexcept
{
    return running(state) ? static_cast<SSVEPPhase>(state) : SSVEPPhase::none;
}
SSVEPMarker marker_of(SSVEPState state, bool onset) noexcept
{
    return static_cast<SSVEPMarker>(2 * (static_cast<unsigned>(phase_of(state)) - 1) +
                                    (onset ? 1 : 2));
}
bool identical(const TrialIdentity& a, const TrialIdentity& b) noexcept
{
    return same_trial(a, b) && a.key == b.key && a.target_id == b.target_id &&
           a.stimulus_id == b.stimulus_id;
}
} // namespace

ContractStatus validate(const SSVEPPresentationRequest& value) noexcept
{
    const auto& request = value.request;
    if (const auto status = neurale::experiments::validate(request); status != ContractStatus::ok)
        return status;
    if (!trial_outcome_declared(value.outcome))
        return ContractStatus::enum_undeclared;
    const auto phase = static_cast<SSVEPPhase>(request.phase);
    if (phase > SSVEPPhase::inter_trial)
        return ContractStatus::enum_undeclared;
    const bool targets = phase == SSVEPPhase::cue || phase == SSVEPPhase::stimulation ||
                         phase == SSVEPPhase::feedback;
    if (request.cue != (targets ? CueKind::ssvep_targets : CueKind::black) ||
        request.trial.target_id == kUnsetTargetId ||
        (targets && request.stimulus_id != request.trial.stimulus_id))
        return ContractStatus::presentation_invalid;
    if (phase != SSVEPPhase::feedback)
        return value.selected_id == kUnsetTargetId && value.outcome == TrialOutcome::pending
                   ? ContractStatus::ok
                   : ContractStatus::presentation_invalid;
    if (value.outcome == TrialOutcome::timeout)
        return value.selected_id == kUnsetTargetId ? ContractStatus::ok
                                                   : ContractStatus::presentation_invalid;
    if ((value.outcome != TrialOutcome::success && value.outcome != TrialOutcome::failure) ||
        value.selected_id == kUnsetTargetId ||
        (value.outcome == TrialOutcome::success) != (value.selected_id == request.trial.target_id))
        return ContractStatus::presentation_invalid;
    return ContractStatus::ok;
}

ContractStatus validate(const SSVEPTrial& trial) noexcept
{
    if (const auto status = neurale::experiments::validate(trial.record);
        status != ContractStatus::ok)
        return status;
    if (const auto status = validate(trial.schedule); status != ContractStatus::ok)
        return status;
    if (!trial_ended(trial.record.outcome) ||
        trial.record.trial.ordinal != trial.schedule.ordinal ||
        trial.record.trial.target_id != trial.schedule.target_id ||
        trial.record.trial.stimulus_id == kUnsetStimulusId || trial.n_phases == 0 ||
        trial.n_phases > 4)
        return ContractStatus::outcome_invalid;
    const bool aborted = trial.record.outcome == TrialOutcome::aborted;
    ExperimentTimeNs cursor = trial.record.interval.start_ns;
    SSVEPPhase previous = SSVEPPhase::none;
    TimeInterval stimulation{};
    ExperimentTimeNs feedback_start{};
    for (std::size_t i = 0; i < trial.n_phases; ++i)
    {
        const auto& entry = trial.phases[i];
        const bool next =
            (previous == SSVEPPhase::none && entry.phase == SSVEPPhase::cue) ||
            (previous == SSVEPPhase::cue && entry.phase == SSVEPPhase::stimulation) ||
            (previous == SSVEPPhase::stimulation &&
             (entry.phase == SSVEPPhase::await_decision || entry.phase == SSVEPPhase::feedback)) ||
            (previous == SSVEPPhase::await_decision && entry.phase == SSVEPPhase::feedback);
        if (!next || entry.interval.start_ns != cursor || !entry.interval.well_formed() ||
            (entry.interval.empty() && !(aborted && i + 1 == trial.n_phases)))
            return ContractStatus::outcome_invalid;
        if (entry.phase == SSVEPPhase::stimulation)
            stimulation = entry.interval;
        if (entry.phase == SSVEPPhase::feedback)
            feedback_start = entry.interval.start_ns;
        cursor = entry.interval.end_ns;
        previous = entry.phase;
    }
    if (cursor != trial.record.interval.end_ns || (!aborted && previous != SSVEPPhase::feedback))
        return ContractStatus::outcome_invalid;
    for (std::size_t i = trial.n_phases; i < trial.phases.size(); ++i)
        if (trial.phases[i].phase != SSVEPPhase::none || trial.phases[i].interval.start_ns != 0 ||
            trial.phases[i].interval.end_ns != 0)
            return ContractStatus::outcome_invalid;
    if (trial.has_selection)
    {
        if (const auto status = neurale::experiments::validate(trial.selection);
            status != ContractStatus::ok)
            return status;
        if (!identical(trial.selection.trial, trial.record.trial) ||
            trial.selection.paradigm != trial.record.paradigm ||
            trial.selection.kind != SelectionKind::discrete ||
            trial.selection.intended_id != trial.schedule.target_id ||
            trial.selection.selected_id == kUnsetTargetId || trial.n_phases < 2 ||
            trial.selection.time_ns < stimulation.start_ns ||
            trial.selection.time_ns > trial.record.interval.end_ns ||
            (trial.has_decision && trial.selection.time_ns > trial.decision_ns))
            return ContractStatus::outcome_invalid;
    }
    else if (trial.selection.time_ns != 0 || trial.selection.sequence != 0 ||
             trial.selection.paradigm != 0 || trial.selection.selected_id != 0 ||
             trial.selection.intended_id != 0 || trial.selection.correct ||
             trial.selection.kind != SelectionKind::discrete || trial.selection.dwell_ns != 0 ||
             !identical(trial.selection.trial, TrialIdentity{}))
        return ContractStatus::outcome_invalid;
    if ((trial.has_decision &&
         (previous != SSVEPPhase::feedback || trial.decision_ns != feedback_start)) ||
        (!trial.has_decision && (trial.decision_ns != 0 || previous == SSVEPPhase::feedback)))
        return ContractStatus::outcome_invalid;
    const auto reason = static_cast<SSVEPReason>(trial.record.reason);
    if (aborted)
        return reason == SSVEPReason::stopped ? ContractStatus::ok
                                              : ContractStatus::outcome_invalid;
    if (!trial.has_decision)
        return ContractStatus::outcome_invalid;
    if (trial.record.outcome == TrialOutcome::timeout)
        return !trial.has_selection && trial.n_phases == 4 &&
                       reason == SSVEPReason::decision_timeout
                   ? ContractStatus::ok
                   : ContractStatus::outcome_invalid;
    const bool correct = trial.record.outcome == TrialOutcome::success;
    if (trial.has_selection &&
        trial.decision_ns != (trial.selection.time_ns > stimulation.end_ns ? trial.selection.time_ns
                                                                           : stimulation.end_ns))
        return ContractStatus::outcome_invalid;
    return trial.has_selection && trial.selection.correct == correct &&
                   reason ==
                       (correct ? SSVEPReason::correct_selection : SSVEPReason::incorrect_selection)
               ? ContractStatus::ok
               : ContractStatus::outcome_invalid;
}

ContractStatus SSVEPMachine::event(Run& run, ExperimentTimeNs time, ExperimentEventKind kind,
                                   SSVEPStepResult& out, SSVEPMarker marker) const noexcept
{
    if (out.n_events >= out.events.size())
        return ContractStatus::numerical_failure;
    ExperimentEvent value{};
    if (const auto status = run.sequence.issue(value.sequence); status != ContractStatus::ok)
        return status;
    value.time_ns = time;
    value.trial = run.trial.record.trial;
    value.paradigm = paradigm_;
    value.kind = kind;
    value.code = static_cast<std::uint32_t>(marker);
    if (kind == ExperimentEventKind::trial_outcome)
        value.value = static_cast<std::int64_t>(run.trial.record.outcome);
    out.events[out.n_events++] = value;
    return ContractStatus::ok;
}

ContractStatus SSVEPMachine::transition(Run& run, SSVEPState to, ExperimentTimeNs time,
                                        SSVEPCause cause, SSVEPStepResult& out) const noexcept
{
    if (out.n_transitions >= out.transitions.size())
        return ContractStatus::numerical_failure;
    StateTransition value{};
    if (const auto status = run.sequence.issue(value.sequence); status != ContractStatus::ok)
        return status;
    value.time_ns = time;
    value.trial = run.trial.record.trial;
    value.paradigm = paradigm_;
    value.from_state = static_cast<StateId>(run.state);
    value.to_state = static_cast<StateId>(to);
    value.cause = static_cast<std::uint32_t>(cause);
    out.transitions[out.n_transitions++] = value;
    run.state = to;
    return ContractStatus::ok;
}

ContractStatus SSVEPMachine::request(Run& run, SSVEPStepResult& out) const noexcept
{
    if (out.n_requests >= out.requests.size())
        return ContractStatus::numerical_failure;
    SSVEPPresentationRequest value{};
    auto& request = value.request;
    if (const auto status = run.sequence.issue(request.sequence); status != ContractStatus::ok)
        return status;
    request.requested_ns = request.onset_ns = run.active.start_ns;
    // Inspect the candidate run, never the not-yet-committed machine.
    request.valid_until_ns = run.state == SSVEPState::complete ? kNoExpiryNs : run.active.end_ns;
    request.duration_ns = run.active.duration_ns();
    request.trial = run.trial.record.trial;
    request.paradigm = paradigm_;
    request.phase = static_cast<PhaseId>(phase_of(run.state));
    const bool targets = run.state == SSVEPState::cue || run.state == SSVEPState::stimulation ||
                         run.state == SSVEPState::feedback;
    request.cue = targets ? CueKind::ssvep_targets : CueKind::black;
    request.stimulus_id = targets ? config_.stimulus_id : kUnsetStimulusId;
    if (run.state == SSVEPState::feedback)
    {
        value.selected_id =
            run.trial.has_selection ? run.trial.selection.selected_id : kUnsetTargetId;
        value.outcome = run.trial.record.outcome;
    }
    out.requests[out.n_requests++] = value;
    return ContractStatus::ok;
}

ContractStatus SSVEPMachine::enter(Run& run, SSVEPState state, ExperimentTimeNs time,
                                   DurationNs duration, SSVEPCause cause,
                                   SSVEPStepResult& out) const noexcept
{
    if (const auto status = interval_from_duration(time, duration, run.active);
        status != ContractStatus::ok)
        return status;
    if (const auto status = transition(run, state, time, cause, out); status != ContractStatus::ok)
        return status;
    if (const auto status =
            event(run, time, ExperimentEventKind::paradigm_marker, out, marker_of(state, true));
        status != ContractStatus::ok)
        return status;
    return request(run, out);
}

ContractStatus SSVEPMachine::leave(Run& run, ExperimentTimeNs time,
                                   SSVEPStepResult& out) const noexcept
{
    if (run.state != SSVEPState::inter_trial &&
        !(run.state == SSVEPState::await_decision && time == run.active.start_ns))
    {
        if (run.trial.n_phases >= run.trial.phases.size())
            return ContractStatus::numerical_failure;
        run.trial.phases[run.trial.n_phases++] = {phase_of(run.state), {run.active.start_ns, time}};
    }
    return event(run, time, ExperimentEventKind::paradigm_marker, out, marker_of(run.state, false));
}

ContractStatus SSVEPMachine::begin_trial(Run& run, ExperimentTimeNs time,
                                         SSVEPStepResult& out) const noexcept
{
    SSVEPTrial trial{};
    TrialOrdinal ordinal{};
    if (const auto status = run.trials.issue(ordinal); status != ContractStatus::ok)
        return status;
    if (const auto status = prepare_trial(config_, ordinal, trial.schedule);
        status != ContractStatus::ok)
        return status;
    trial.record.trial = {ordinal, kUnsetTrialKey, 0, trial.schedule.target_id,
                          config_.stimulus_id};
    trial.record.paradigm = paradigm_;
    trial.record.interval = {time, time};
    run.trial = trial;
    TimeInterval cue{}, wait{}, feedback_interval{}, rest{};
    if (const auto status = interval_from_duration(time, config_.cue_duration_ns, cue);
        status != ContractStatus::ok)
        return status;
    if (const auto status =
            interval_from_duration(cue.end_ns, config_.stimulation_duration_ns, run.stimulation);
        status != ContractStatus::ok)
        return status;
    if (const auto status =
            interval_from_duration(run.stimulation.end_ns, config_.decision_timeout_ns, wait);
        status != ContractStatus::ok)
        return status;
    run.deadline_ns = wait.end_ns;
    // Reserve the latest possible feedback/rest too, so a valid trial can always finish.
    if (const auto status =
            interval_from_duration(wait.end_ns, config_.feedback_duration_ns, feedback_interval);
        status != ContractStatus::ok)
        return status;
    if (const auto status =
            interval_from_duration(feedback_interval.end_ns, config_.inter_trial_ns, rest);
        status != ContractStatus::ok)
        return status;
    if (ordinal == 0)
        if (const auto status = event(run, time, ExperimentEventKind::session_start, out);
            status != ContractStatus::ok)
            return status;
    if (const auto status = event(run, time, ExperimentEventKind::trial_start, out);
        status != ContractStatus::ok)
        return status;
    return enter(run, SSVEPState::cue, time, config_.cue_duration_ns,
                 ordinal == 0 ? SSVEPCause::session_started : SSVEPCause::rest_elapsed, out);
}

ContractStatus SSVEPMachine::feedback(Run& run, ExperimentTimeNs time,
                                      SSVEPStepResult& out) const noexcept
{
    // A result observed exactly at stimulus end precedes the outcome it causes.
    if (out.selection_disposition == SSVEPSelectionDisposition::accepted &&
        run.trial.has_selection && run.trial.selection.time_ns == time)
        if (const auto status = event(run, time, ExperimentEventKind::selection, out);
            status != ContractStatus::ok)
            return status;
    run.trial.has_decision = true;
    run.trial.decision_ns = time;
    run.trial.record.outcome = !run.trial.has_selection      ? TrialOutcome::timeout
                               : run.trial.selection.correct ? TrialOutcome::success
                                                             : TrialOutcome::failure;
    run.trial.record.reason = static_cast<std::uint32_t>(
        !run.trial.has_selection      ? SSVEPReason::decision_timeout
        : run.trial.selection.correct ? SSVEPReason::correct_selection
                                      : SSVEPReason::incorrect_selection);
    if (const auto status = event(run, time, ExperimentEventKind::trial_outcome, out);
        status != ContractStatus::ok)
        return status;
    return enter(run, SSVEPState::feedback, time, config_.feedback_duration_ns,
                 run.trial.has_selection ? SSVEPCause::selection_received
                                         : SSVEPCause::decision_timeout,
                 out);
}

ContractStatus SSVEPMachine::hold_inter_trial_until(ExperimentTimeNs time) noexcept
{
    if (run_.state != SSVEPState::inter_trial)
        return ContractStatus::not_running;
    if (time < run_.active.start_ns)
        return ContractStatus::time_regressed;
    if (time > run_.active.end_ns)
        run_.active.end_ns = time;
    return ContractStatus::ok;
}

ContractStatus SSVEPMachine::finish_trial(Run& run, ExperimentTimeNs time,
                                          SSVEPStepResult& out) const noexcept
{
    run.trial.record.interval.end_ns = time;
    if (const auto status = validate(run.trial); status != ContractStatus::ok)
        return status;
    ++run.completed; // Bounded by positive config.n_trials; a trial finishes once.
    out.trial = run.trial;
    out.trial_decided = true;
    return event(run, time, ExperimentEventKind::trial_stop, out);
}

ContractStatus SSVEPMachine::terminate(Run& run, ExperimentTimeNs time, SSVEPCause cause,
                                       SSVEPStepResult& out) const noexcept
{
    if (const auto status = transition(run, SSVEPState::complete, time, cause, out);
        status != ContractStatus::ok)
        return status;
    run.active = {time, time};
    if (const auto status = event(run, time, ExperimentEventKind::session_stop, out);
        status != ContractStatus::ok)
        return status;
    out.settled = true;
    return request(run, out);
}

ContractStatus SSVEPMachine::advance(Run& run, ExperimentTimeNs time,
                                     SSVEPStepResult& out) const noexcept
{
    for (std::size_t guard = 0; guard < kMaxStepTransitions; ++guard)
    {
        if (!running(run.state) || !run.active.elapsed_at(time))
        {
            out.settled = true;
            return ContractStatus::ok;
        }
        const auto boundary = run.active.end_ns;
        const auto previous = run.state;
        if (const auto status = leave(run, boundary, out); status != ContractStatus::ok)
            return status;
        ContractStatus status{};
        switch (previous)
        {
        case SSVEPState::cue:
            status = enter(run, SSVEPState::stimulation, boundary, config_.stimulation_duration_ns,
                           SSVEPCause::cue_elapsed, out);
            break;
        case SSVEPState::stimulation:
            status = run.trial.has_selection
                         ? feedback(run, boundary, out)
                         : enter(run, SSVEPState::await_decision, boundary,
                                 config_.decision_timeout_ns, SSVEPCause::stimulation_elapsed, out);
            break;
        case SSVEPState::await_decision:
            status = feedback(run, boundary, out);
            break;
        case SSVEPState::feedback:
            if (const auto ended = finish_trial(run, boundary, out); ended != ContractStatus::ok)
                return ended;
            status = enter(run, SSVEPState::inter_trial, boundary, config_.inter_trial_ns,
                           SSVEPCause::feedback_elapsed, out);
            out.settled = !run.active.elapsed_at(time);
            return status; // At most one completed trial per call.
        case SSVEPState::inter_trial:
            if (run.completed == config_.n_trials)
                return terminate(run, boundary, SSVEPCause::trial_limit_reached, out);
            status = begin_trial(run, boundary, out);
            break;
        default:
            return ContractStatus::outcome_invalid;
        }
        if (status != ContractStatus::ok)
            return status;
    }
    return ContractStatus::numerical_failure;
}

SSVEPSnapshot SSVEPMachine::build_snapshot(const Run& run) const noexcept
{
    return {run.gate.last_ns(),       run.state,       run.trial.record.trial,  run.active,
            run.stimulation,          run.deadline_ns, run.trial.has_selection, run.trial.selection,
            run.trial.record.outcome, run.completed};
}
SSVEPSnapshot SSVEPMachine::snapshot() const noexcept
{
    return build_snapshot(run_);
}
void SSVEPMachine::reset() noexcept
{
    *this = SSVEPMachine{};
}

ContractStatus SSVEPMachine::start(ParadigmId paradigm, const SSVEPConfig& config,
                                   ExperimentTimeNs time, SSVEPStepResult& result) noexcept
{
    if (run_.state != SSVEPState::idle)
        return ContractStatus::already_running;
    if (paradigm == kUnsetParadigmId)
        return ContractStatus::identity_missing;
    if (const auto status = validate(config); status != ContractStatus::ok)
        return status;
    SSVEPMachine machine{};
    machine.config_ = config;
    machine.paradigm_ = paradigm;
    machine.run_.gate.reset(time);
    SSVEPStepResult local{};
    if (const auto status = machine.begin_trial(machine.run_, time, local);
        status != ContractStatus::ok)
        return status;
    local.snapshot = machine.snapshot();
    local.settled = true;
    *this = machine;
    result = local;
    return ContractStatus::ok;
}

ContractStatus SSVEPMachine::step_impl(ExperimentTimeNs time, const SelectionEvent* selection,
                                       SSVEPStepResult& result) noexcept
{
    if (!running(run_.state))
        return ContractStatus::not_running;
    Run run = run_;
    if (const auto status = run.gate.accept(time); status != ContractStatus::ok)
        return status;
    SSVEPStepResult local{};
    if (selection)
    {
        if (const auto status = neurale::experiments::validate(*selection);
            status != ContractStatus::ok)
            return status;
        if (selection->kind != SelectionKind::discrete || selection->time_ns != time ||
            selection->paradigm != paradigm_ ||
            !identical(selection->trial, run.trial.record.trial) ||
            selection->intended_id != run.trial.schedule.target_id ||
            !contains_target(config_, selection->selected_id) || time < run.stimulation.start_ns ||
            run.trial.has_selection ||
            (run.has_last_selection && selection->sequence <= run.last_selection_sequence))
            return ContractStatus::outcome_invalid;
        if (time >= run.deadline_ns)
            local.selection_disposition = SSVEPSelectionDisposition::expired;
        else
        {
            run.trial.has_selection = true;
            run.trial.selection = *selection;
            run.has_last_selection = true;
            run.last_selection_sequence = selection->sequence;
            local.selection_disposition = SSVEPSelectionDisposition::accepted;
        }
    }
    // Advance before applying an observed result: a result arriving during the wait
    // must not be backdated to the end of stimulation by a sparse polling caller.
    const bool accepted = local.selection_disposition == SSVEPSelectionDisposition::accepted;
    if (accepted && time > run.stimulation.end_ns)
        run.trial.has_selection = false;
    if (const auto status = advance(run, time, local); status != ContractStatus::ok)
        return status;
    if (accepted)
    {
        run.trial.has_selection = true;
        if (run.state == SSVEPState::await_decision)
        {
            if (const auto status = leave(run, time, local); status != ContractStatus::ok)
                return status;
            if (const auto status = feedback(run, time, local); status != ContractStatus::ok)
                return status;
        }
        else if (run.state == SSVEPState::stimulation)
            if (const auto status = event(run, time, ExperimentEventKind::selection, local);
                status != ContractStatus::ok)
                return status;
    }
    local.snapshot = build_snapshot(run);
    run_ = run;
    result = local;
    return ContractStatus::ok;
}
ContractStatus SSVEPMachine::step(ExperimentTimeNs time, SSVEPStepResult& result) noexcept
{
    return step_impl(time, nullptr, result);
}
ContractStatus SSVEPMachine::step(ExperimentTimeNs time, const SelectionEvent& selection,
                                  SSVEPStepResult& result) noexcept
{
    return step_impl(time, &selection, result);
}

ContractStatus SSVEPMachine::stop(ExperimentTimeNs time, SSVEPStepResult& result) noexcept
{
    if (!running(run_.state))
        return ContractStatus::not_running;
    Run run = run_;
    if (const auto status = run.gate.accept(time); status != ContractStatus::ok)
        return status;
    SSVEPStepResult local{};
    // Stop wins over unobserved elapsed boundaries; it does not invent new trials.
    if (const auto status = leave(run, time, local); status != ContractStatus::ok)
        return status;
    if (run.state != SSVEPState::inter_trial)
    {
        run.trial.record.outcome = TrialOutcome::aborted;
        run.trial.record.reason = static_cast<std::uint32_t>(SSVEPReason::stopped);
        if (const auto status = event(run, time, ExperimentEventKind::trial_outcome, local);
            status != ContractStatus::ok)
            return status;
        if (const auto status = finish_trial(run, time, local); status != ContractStatus::ok)
            return status;
    }
    if (const auto status = terminate(run, time, SSVEPCause::stopped, local);
        status != ContractStatus::ok)
        return status;
    local.snapshot = build_snapshot(run);
    run_ = run;
    result = local;
    return ContractStatus::ok;
}
} // namespace neurale::experiments::ssvep
