/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "replay_verdict.h"

#include <neurale/experiments/webgrid_replay.h>

namespace neurale::experiments::webgrid
{
namespace
{
void compare_selection(ReplayComparator& comparator, const SelectionEvent& recorded,
                       const SelectionEvent& regenerated) noexcept
{
    comparator.integer("time_ns", recorded.time_ns, regenerated.time_ns);
    comparator.integer("sequence", recorded.sequence, regenerated.sequence);
    comparator.integer("paradigm", recorded.paradigm, regenerated.paradigm);
    comparator.integer("kind", static_cast<std::uint64_t>(recorded.kind),
                       static_cast<std::uint64_t>(regenerated.kind));
    comparator.flag("correct", recorded.correct, regenerated.correct);
    comparator.integer("selected_id", recorded.selected_id, regenerated.selected_id);
    comparator.integer("intended_id", recorded.intended_id, regenerated.intended_id);
    comparator.integer("dwell_ns", recorded.dwell_ns, regenerated.dwell_ns);
    comparator.identity("trial", recorded.trial, regenerated.trial);
}

void compare_record(ReplayComparator& comparator, const TrialRecord& recorded,
                    const TrialRecord& regenerated) noexcept
{
    comparator.identity("trial", recorded.trial, regenerated.trial);
    comparator.integer("start_ns", recorded.interval.start_ns, regenerated.interval.start_ns);
    comparator.integer("end_ns", recorded.interval.end_ns, regenerated.interval.end_ns);
    comparator.integer("paradigm", recorded.paradigm, regenerated.paradigm);
    comparator.integer("reason", recorded.reason, regenerated.reason);
}

[[nodiscard]] bool valid_replay_input(const WebGridReplayInput& input) noexcept
{
    if (!webgrid_replay_input_kind_declared(input.kind))
    {
        return false;
    }
    switch (input.kind)
    {
    case WebGridReplayInputKind::pointer:
    case WebGridReplayInputKind::selection:
        return input.condition == AbnormalCondition::unspecified &&
               input.response == AbnormalResponse::recorded;
    case WebGridReplayInputKind::pointer_gap:
        return input.condition == AbnormalCondition::input_gap &&
               abnormal_response_declared(input.response);
    case WebGridReplayInputKind::observer_drop:
        return input.condition == AbnormalCondition::observer_frame_drop &&
               abnormal_response_declared(input.response);
    case WebGridReplayInputKind::halt:
        return abnormal_condition_declared(input.condition) &&
               input.condition != AbnormalCondition::unspecified &&
               abnormal_response_declared(input.response);
    }
    return false;
}
} // namespace

struct WebGridReplay::Cursors
{
    std::size_t pointer{};
    std::size_t selections{};
    std::size_t trials{};
    std::size_t targets{};
    bool exhausted{};
};

ContractStatus WebGridReplay::prepare(const WebGridReplayConfig& config) noexcept
{
    if (config.paradigm == kUnsetParadigmId)
    {
        return ContractStatus::identity_missing;
    }
    if (const auto status = validate(config.task); status != ContractStatus::ok)
    {
        return status;
    }
    if (const auto status = experiments::validate(config.abnormal); status != ContractStatus::ok)
    {
        return status;
    }
    config_ = config;
    prepared_ = true;
    reset();
    return ContractStatus::ok;
}

void WebGridReplay::reset() noexcept
{
    machine_.reset();
    metrics_ = WebGridMetrics{};
    invalidated_targets_ = 0;
    last_onset_ns_ = 0;
    last_target_ = kUnsetTargetId;
    target_invalidated_ = false;
    has_target_ = false;
    running_ = false;
    terminal_ = false;
}

ReplayProvenance WebGridReplay::provenance() const noexcept
{
    const auto configuration = configuration_fingerprint(config_.task);
    const ScheduleIdentity identity{
        .seed = config_.task.seed,
        .sampler_version = config_.task.sampler_version,
        .configuration_fingerprint = configuration,
        // WebGrid presents cells rather than stimuli and has no catalog.
        .catalog_fingerprint = 0,
    };
    return ReplayProvenance{
        .paradigm = config_.paradigm,
        .experiment_version = kWebGridRecordVersion,
        .configuration_fingerprint = configuration,
        .schedule_fingerprint = schedule_fingerprint(identity),
        .realized_schedule_fingerprint = 0,
        .seed = config_.task.seed,
        .sampler_version = config_.task.sampler_version,
        .metric_version = config_.task.metric_version,
        .policy_version = 0,
    };
}

void WebGridReplay::compare_target(const WebGridSnapshot& snapshot,
                                   const WebGridReplayRecording& recording, Cursors& cursors,
                                   ReplayComparator& comparator) noexcept
{
    // The bridge's own suppression: a cell that stayed up across a step has not
    // had a second onset, and the onset instant is part of what makes it one.
    if (snapshot.active_target == kUnsetTargetId)
    {
        return;
    }
    if (has_target_ && snapshot.active_target == last_target_ &&
        snapshot.target_onset_ns == last_onset_ns_)
    {
        return;
    }
    has_target_ = true;
    last_target_ = snapshot.active_target;
    last_onset_ns_ = snapshot.target_onset_ns;
    if (!recording.expected.has_targets || comparator.differed())
    {
        return;
    }
    if (cursors.targets >= recording.expected.targets.size())
    {
        cursors.exhausted = true;
        return;
    }
    const auto& recorded = recording.expected.targets[cursors.targets];
    comparator.begin(ReplayItem::target, cursors.targets, recorded.onset_ns, recorded.trial);
    comparator.integer("onset_ns", recorded.onset_ns, snapshot.target_onset_ns);
    comparator.integer("target_id", recorded.target_id, snapshot.active_target);
    comparator.integer("completed", recorded.completed, snapshot.completed);
    comparator.identity("trial", recorded.trial, snapshot.trial);
    ++cursors.targets;
}

ContractStatus WebGridReplay::replay_pointer(const WebGridReplayInput& input, bool with_selection,
                                             const WebGridReplayRecording& recording,
                                             Cursors& cursors,
                                             ReplayComparator& comparator) noexcept
{
    if (terminal_ || !running_ || machine_.complete())
    {
        // The run refused it, so it produced nothing to compare. A pointer
        // sample after a completed session is not a difference between two
        // runs; it is an input the recording kept and the task declined.
        return ContractStatus::ok;
    }
    WebGridStepResult step{};
    const auto status = with_selection
                            ? machine_.step(input.time_ns, input.pointer, input.selection, step)
                            : machine_.step(input.time_ns, input.pointer, step);
    if (status != ContractStatus::ok)
    {
        return status;
    }
    metrics_ = step.snapshot.metrics;

    const auto& expected = recording.expected;
    if (expected.has_pointer && !comparator.differed())
    {
        if (cursors.pointer >= expected.pointer.size())
        {
            cursors.exhausted = true;
        }
        else
        {
            const auto& recorded = expected.pointer[cursors.pointer];
            comparator.begin(ReplayItem::cursor, cursors.pointer, recorded.time_ns, recorded.trial);
            comparator.integer("time_ns", recorded.time_ns, step.snapshot.time_ns);
            comparator.real("x", recorded.pointer.x, input.pointer.x);
            comparator.real("y", recorded.pointer.y, input.pointer.y);
            comparator.integer("state", static_cast<std::uint64_t>(recorded.state),
                               static_cast<std::uint64_t>(step.snapshot.state));
            comparator.integer("active_target", recorded.active_target,
                               step.snapshot.active_target);
            comparator.identity("trial", recorded.trial, step.snapshot.trial);
            ++cursors.pointer;
        }
    }
    if (step.selection_processed && expected.has_selections && !comparator.differed())
    {
        if (cursors.selections >= expected.selections.size())
        {
            cursors.exhausted = true;
        }
        else
        {
            const auto& recorded = expected.selections[cursors.selections];
            comparator.begin(ReplayItem::selection, cursors.selections,
                             recorded.record.event.time_ns, recorded.record.event.trial);
            compare_selection(comparator, recorded.record.event, step.selection.event);
            comparator.integer("target_onset_ns", recorded.record.target_onset_ns,
                               step.selection.target_onset_ns);
            comparator.integer("elapsed_since_target_onset_ns",
                               recorded.record.elapsed_since_target_onset_ns,
                               step.selection.elapsed_since_target_onset_ns);
            comparator.flag("acquisition_timing_valid", recorded.acquisition_timing_valid,
                            !target_invalidated_);
            ++cursors.selections;
        }
    }
    if (step.trial_decided && expected.has_trials && !comparator.differed())
    {
        if (cursors.trials >= expected.trials.size())
        {
            cursors.exhausted = true;
        }
        else
        {
            const auto& recorded = expected.trials[cursors.trials];
            comparator.begin(ReplayItem::trial, cursors.trials,
                             recorded.trial.record.interval.end_ns, recorded.trial.record.trial);
            compare_record(comparator, recorded.trial.record, step.trial.record);
            // The machine's own verdict and the recording's column, separately.
            // A correct selection over an invalidated interval is written as
            // aborted with the verdict kept beside it, and a replay comparing
            // only one of the two would miss whichever it dropped.
            comparator.integer("task_outcome",
                               static_cast<std::uint64_t>(recorded.trial.record.outcome),
                               static_cast<std::uint64_t>(step.trial.record.outcome));
            comparator.integer(
                "recorded_outcome", static_cast<std::uint64_t>(recorded.recorded_outcome),
                static_cast<std::uint64_t>(target_invalidated_ ? TrialOutcome::aborted
                                                               : step.trial.record.outcome));
            comparator.integer("acquisition_ns", recorded.trial.acquisition_ns,
                               step.trial.acquisition_ns);
            compare_selection(comparator, recorded.trial.selection.event,
                              step.trial.selection.event);
            comparator.flag("acquisition_timing_valid", recorded.acquisition_timing_valid,
                            !target_invalidated_);
            ++cursors.trials;
        }
    }
    if (step.trial_decided)
    {
        // The next target starts clean: what was invalidated was this target's
        // acquisition interval, and the interval ended here.
        target_invalidated_ = false;
    }
    compare_target(step.snapshot, recording, cursors, comparator);
    return ContractStatus::ok;
}

void WebGridReplay::replay_decision(const WebGridReplayInput& input,
                                    ReplayComparator& comparator) noexcept
{
    const auto snapshot = machine_.snapshot();
    const bool in_trial = running_ && !machine_.complete();
    const auto policy = policy_for(config_.abnormal, input.condition);
    // WebGridMachine owns when a target ends and offers no way to abandon one,
    // so `abort_trial` invalidates the target in flight rather than ending it.
    const auto response =
        abnormal_response_for(policy, in_trial, /*can_end_trial=*/false, /*input_refused=*/false);
    if (!comparator.differed())
    {
        comparator.begin(ReplayItem::trial, 0, input.time_ns, snapshot.trial);
        comparator.integer("response", static_cast<std::uint64_t>(input.response),
                           static_cast<std::uint64_t>(response));
    }
    if (response == AbnormalResponse::trial_invalidated)
    {
        target_invalidated_ = true;
        ++invalidated_targets_;
    }
    if (response == AbnormalResponse::session_aborted)
    {
        terminal_ = true;
        running_ = false;
    }
}

ReplayReport WebGridReplay::run(const WebGridReplayRecording& recording,
                                ReplayIncompletePolicy policy) noexcept
{
    ReplayReport report{};
    if (!prepared_)
    {
        report.rejection = ReplayRejection::configuration_invalid;
        return report;
    }
    const auto rejection = check_provenance(recording.provenance, provenance());
    if (rejection != ReplayRejection::none)
    {
        report.rejection = rejection;
        return report;
    }
    ExperimentTimeNs previous = recording.origin_ns;
    for (const auto& input : recording.inputs)
    {
        if (!valid_replay_input(input) || input.time_ns < previous)
        {
            report.rejection = ReplayRejection::evidence_invalid;
            return report;
        }
        previous = input.time_ns;
    }

    const auto& expected = recording.expected;
    if (!expected.has_pointer || !expected.has_selections || !expected.has_trials ||
        !expected.has_targets || !expected.has_metrics)
    {
        report.completeness = ReplayCompleteness::stream_absent;
    }
    else if (recording.trace_loss_recorded)
    {
        report.completeness = ReplayCompleteness::trace_loss_recorded;
    }
    else if (!recording.run_end_recorded)
    {
        report.completeness = ReplayCompleteness::run_end_missing;
    }

    if (report.completeness != ReplayCompleteness::complete &&
        policy == ReplayIncompletePolicy::refuse)
    {
        report.verdict = ReplayVerdict::incomplete;
        return report;
    }

    reset();
    ReplayComparator comparator;
    Cursors cursors{};

    WebGridStepResult opening{};
    const auto opening_status =
        machine_.start(config_.paradigm, config_.task, recording.origin_ns, opening);
    if (opening_status != ContractStatus::ok)
    {
        if (opening_status == ContractStatus::version_unsupported)
        {
            report.rejection = config_.task.metric_version != kCurrentMetricVersion
                                   ? ReplayRejection::metric_version_unsupported
                                   : ReplayRejection::sampler_version_unsupported;
        }
        else
        {
            report.rejection = ReplayRejection::configuration_invalid;
        }
        return report;
    }
    running_ = true;
    metrics_ = opening.snapshot.metrics;
    compare_target(opening.snapshot, recording, cursors, comparator);

    ContractStatus regeneration_status = ContractStatus::ok;
    for (const auto& input : recording.inputs)
    {
        if (comparator.differed())
        {
            break;
        }
        ++report.inputs_replayed;
        switch (input.kind)
        {
        case WebGridReplayInputKind::pointer:
            regeneration_status =
                replay_pointer(input, /*with_selection=*/false, recording, cursors, comparator);
            break;
        case WebGridReplayInputKind::selection:
            regeneration_status =
                replay_pointer(input, /*with_selection=*/true, recording, cursors, comparator);
            break;
        case WebGridReplayInputKind::pointer_gap:
        case WebGridReplayInputKind::observer_drop:
            replay_decision(input, comparator);
            break;
        case WebGridReplayInputKind::halt:
            replay_decision(input, comparator);
            terminal_ = true;
            running_ = false;
            break;
        }
        if (regeneration_status != ContractStatus::ok)
        {
            break;
        }
    }

    if (expected.has_metrics && !comparator.differed() && regeneration_status == ContractStatus::ok)
    {
        // The published metric set, and the count that says whether it is
        // provisional. Compared last because it is the run's own summary of
        // everything above it: a metric that differs while every selection
        // agreed is a metric formula that changed, and reporting it before the
        // selections would hide which of the two happened.
        comparator.begin(ReplayItem::metrics, 0, recording.origin_ns, TrialIdentity{});
        comparator.integer("metric_version", expected.metrics.metric_version,
                           metrics_.metric_version);
        comparator.integer("correct_selections", expected.metrics.correct_selections,
                           metrics_.correct_selections);
        comparator.integer("incorrect_selections", expected.metrics.incorrect_selections,
                           metrics_.incorrect_selections);
        comparator.integer("elapsed_active_ns", expected.metrics.elapsed_active_ns,
                           metrics_.elapsed_active_ns);
        comparator.flag("rates_defined", expected.metrics.rates_defined, metrics_.rates_defined);
        comparator.real("correct_targets_per_minute", expected.metrics.correct_targets_per_minute,
                        metrics_.correct_targets_per_minute);
        comparator.real("net_correct_targets_per_minute",
                        expected.metrics.net_correct_targets_per_minute,
                        metrics_.net_correct_targets_per_minute);
        comparator.integer("acquisition_count", expected.metrics.n_acquisitions,
                           metrics_.n_acquisitions);
        comparator.integer("total_acquisition_ns", expected.metrics.total_acquisition_ns,
                           metrics_.total_acquisition_ns);
        comparator.integer("minimum_acquisition_ns", expected.metrics.minimum_acquisition_ns,
                           metrics_.minimum_acquisition_ns);
        comparator.integer("maximum_acquisition_ns", expected.metrics.maximum_acquisition_ns,
                           metrics_.maximum_acquisition_ns);
        comparator.integer("mean_acquisition_ns", expected.metrics.mean_acquisition_ns,
                           metrics_.mean_acquisition_ns);
        comparator.integer("invalidated_targets", expected.invalidated_targets,
                           invalidated_targets_);
    }

    if (note_regeneration_outcome(report, comparator, regeneration_status))
    {
        return report;
    }
    if (cursors.exhausted && report.completeness != ReplayCompleteness::trace_loss_recorded)
    {
        report.verdict = ReplayVerdict::mismatch;
        report.first_mismatch.item = ReplayItem::stream_length;
        report.first_mismatch.field = "recorded_short";
        return report;
    }
    if (note_recorded_extra(report, expected.has_pointer, expected.pointer.size(),
                            cursors.pointer) ||
        note_recorded_extra(report, expected.has_selections, expected.selections.size(),
                            cursors.selections) ||
        note_recorded_extra(report, expected.has_trials, expected.trials.size(), cursors.trials) ||
        note_recorded_extra(report, expected.has_targets, expected.targets.size(), cursors.targets))
    {
        return report;
    }
    report.verdict = report.completeness == ReplayCompleteness::complete
                         ? ReplayVerdict::match
                         : ReplayVerdict::incomplete;
    return report;
}

} // namespace neurale::experiments::webgrid
