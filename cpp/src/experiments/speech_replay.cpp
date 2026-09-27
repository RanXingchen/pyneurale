/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "replay_verdict.h"

#include <neurale/experiments/speech_replay.h>

#include <algorithm>

namespace neurale::experiments::speech
{
namespace
{
void compare_transition(ReplayComparator& comparator, const StateTransition& recorded,
                        const StateTransition& regenerated) noexcept
{
    comparator.integer("time_ns", recorded.time_ns, regenerated.time_ns);
    comparator.integer("sequence", recorded.sequence, regenerated.sequence);
    comparator.integer("paradigm", recorded.paradigm, regenerated.paradigm);
    comparator.integer("from_state", recorded.from_state, regenerated.from_state);
    comparator.integer("to_state", recorded.to_state, regenerated.to_state);
    comparator.integer("cause", recorded.cause, regenerated.cause);
    comparator.identity("trial", recorded.trial, regenerated.trial);
}

void compare_event(ReplayComparator& comparator, const ExperimentEvent& recorded,
                   const ExperimentEvent& regenerated) noexcept
{
    comparator.integer("time_ns", recorded.time_ns, regenerated.time_ns);
    comparator.integer("sequence", recorded.sequence, regenerated.sequence);
    comparator.integer("kind", static_cast<std::uint64_t>(recorded.kind),
                       static_cast<std::uint64_t>(regenerated.kind));
    comparator.integer("paradigm", recorded.paradigm, regenerated.paradigm);
    comparator.integer("code", recorded.code, regenerated.code);
    comparator.integer("value", static_cast<std::uint64_t>(recorded.value),
                       static_cast<std::uint64_t>(regenerated.value));
    comparator.identity("trial", recorded.trial, regenerated.trial);
}

void compare_schedule(ReplayComparator& comparator, const SpeechTrialSchedule& recorded,
                      const SpeechTrialSchedule& regenerated) noexcept
{
    comparator.integer("ordinal", recorded.ordinal, regenerated.ordinal);
    comparator.integer("black_duration_ns", recorded.black_duration_ns,
                       regenerated.black_duration_ns);
    comparator.integer("cross_duration_ns", recorded.cross_duration_ns,
                       regenerated.cross_duration_ns);
    comparator.integer("content_duration_ns", recorded.content_duration_ns,
                       regenerated.content_duration_ns);
    comparator.integer("stimulus_id", recorded.stimulus_id, regenerated.stimulus_id);
    comparator.integer("sampler_version", recorded.sampler_version, regenerated.sampler_version);
    comparator.flag("cross_enabled", recorded.cross_enabled, regenerated.cross_enabled);
}

[[nodiscard]] bool valid_replay_input(const SpeechReplayInput& input) noexcept
{
    if (!speech_replay_input_kind_declared(input.kind))
    {
        return false;
    }
    switch (input.kind)
    {
    case SpeechReplayInputKind::advance:
    case SpeechReplayInputKind::presentation_report:
        return input.condition == AbnormalCondition::unspecified &&
               input.response == AbnormalResponse::recorded;
    case SpeechReplayInputKind::halt:
        return abnormal_condition_declared(input.condition) &&
               input.condition != AbnormalCondition::unspecified &&
               abnormal_response_declared(input.response);
    }
    return false;
}
} // namespace

struct SpeechReplay::Cursors
{
    std::size_t transitions{};
    std::size_t events{};
    std::size_t requests{};
    std::size_t phases{};
    std::size_t trials{};
    std::size_t reports{};
    std::size_t started_trials{};
    bool exhausted{};
};

ContractStatus SpeechReplay::prepare(const SpeechReplayConfig& config)
{
    if (config.paradigm == kUnsetParadigmId)
    {
        return ContractStatus::identity_missing;
    }
    if (const auto status = validate(config.task); status != ContractStatus::ok)
    {
        return status;
    }
    if (const auto status = validate(config.catalog); status != ContractStatus::ok)
    {
        return status;
    }
    if (const auto status = validate_against(config.task, config.catalog);
        status != ContractStatus::ok)
    {
        return status;
    }
    if (const auto status = experiments::validate(config.abnormal); status != ContractStatus::ok)
    {
        return status;
    }
    const auto count = static_cast<std::size_t>(config.task.n_trials);
    if (static_cast<TrialOrdinal>(count) != config.task.n_trials)
    {
        return ContractStatus::parameter_out_of_range;
    }
    auto requests = std::make_unique<EmittedRequest[]>(count);
    std::unique_ptr<SpeechTrialSchedule[]> prepared_schedules{};
    const auto authority = replay_authority(config.task);
    const bool configuration_supplies_schedule =
        config.task.schedule == SpeechScheduleKind::explicit_sequence ||
        authority == ReplayAuthority::regenerate;
    if (configuration_supplies_schedule)
    {
        prepared_schedules = std::make_unique<SpeechTrialSchedule[]>(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            const auto status =
                prepare_trial(config.task, static_cast<TrialOrdinal>(i), prepared_schedules[i]);
            if (status != ContractStatus::ok)
            {
                return status;
            }
        }
    }
    config_ = config;
    authority_ = authority;
    requests_ = std::move(requests);
    prepared_schedules_ = std::move(prepared_schedules);
    n_trials_ = count;
    prepared_ = true;
    reset();
    return ContractStatus::ok;
}

void SpeechReplay::reset() noexcept
{
    machine_.reset();
    std::fill_n(requests_.get(), n_trials_, EmittedRequest{});
    last_trial_ = 0;
    has_trial_ = false;
    running_ = false;
    terminal_ = false;
}

ReplayAuthority SpeechReplay::authority() const noexcept
{
    return authority_;
}

ReplayProvenance
SpeechReplay::provenance(std::span<const SpeechTrialSchedule> realized) const noexcept
{
    const auto identity = schedule_identity(config_.task, config_.catalog);
    // Absorbed field by field, in trial order, exactly as the recording bridge
    // digests it. Two legal realizations of one seeded configuration have to be
    // distinguishable, and a digest taken over the struct's bytes would depend
    // on its padding instead.
    FingerprintAccumulator accumulator;
    accumulator.absorb(realized.size());
    for (const auto& schedule : realized)
    {
        accumulator.absorb(schedule.ordinal);
        accumulator.absorb(schedule.black_duration_ns);
        accumulator.absorb(schedule.cross_duration_ns);
        accumulator.absorb(schedule.content_duration_ns);
        accumulator.absorb(schedule.stimulus_id);
        accumulator.absorb(schedule.sampler_version);
        accumulator.absorb(schedule.cross_enabled ? 1U : 0U);
    }
    return ReplayProvenance{
        .paradigm = config_.paradigm,
        .experiment_version = kSpeechRecordVersion,
        .configuration_fingerprint = identity.configuration_fingerprint,
        .schedule_fingerprint = schedule_fingerprint(identity),
        .realized_schedule_fingerprint = accumulator.value(),
        .seed = identity.seed,
        .sampler_version = identity.sampler_version,
        .metric_version = 0,
        .policy_version = static_cast<std::uint32_t>(config_.task.stimulus_order),
    };
}

ContractStatus SpeechReplay::schedule_for(TrialOrdinal ordinal,
                                          const SpeechReplayRecording& recording,
                                          SpeechTrialSchedule& schedule) const noexcept
{
    if (ordinal >= n_trials_)
    {
        return ContractStatus::range_empty;
    }
    if (prepared_schedules_ != nullptr)
    {
        schedule = prepared_schedules_[static_cast<std::size_t>(ordinal)];
        return ContractStatus::ok;
    }
    // Only an unsupported seeded sampler reaches this branch. Its recorded
    // realized schedule is fallback input, requested one trial at a time so an
    // early terminal input does not require schedules for trials never started.
    if (!recording.expected.has_schedules || ordinal >= recording.expected.schedules.size())
    {
        return ContractStatus::version_unsupported;
    }
    schedule = recording.expected.schedules[ordinal];
    return ContractStatus::ok;
}

void SpeechReplay::compare_step(const SpeechStepResult& step,
                                const SpeechReplayRecording& recording, Cursors& cursors,
                                ReplayComparator& comparator) noexcept
{
    const auto& expected = recording.expected;
    if (expected.has_transitions)
    {
        for (std::uint8_t i = 0; i < step.n_transitions && !comparator.differed(); ++i)
        {
            if (cursors.transitions >= expected.transitions.size())
            {
                cursors.exhausted = true;
                break;
            }
            const auto& recorded = expected.transitions[cursors.transitions];
            comparator.begin(ReplayItem::transition, cursors.transitions, recorded.time_ns,
                             recorded.trial);
            compare_transition(comparator, recorded, step.transitions[i]);
            ++cursors.transitions;
        }
    }
    if (expected.has_events)
    {
        for (std::uint8_t i = 0; i < step.n_events && !comparator.differed(); ++i)
        {
            if (cursors.events >= expected.events.size())
            {
                cursors.exhausted = true;
                break;
            }
            const auto& recorded = expected.events[cursors.events];
            comparator.begin(ReplayItem::event, cursors.events, recorded.time_ns, recorded.trial);
            compare_event(comparator, recorded, step.events[i]);
            ++cursors.events;
        }
    }
    for (std::uint8_t i = 0; i < step.n_requests; ++i)
    {
        const auto& request = step.requests[i];
        if (request.cue == CueKind::text_content)
        {
            // Remembered exactly as the recording bridge remembers it, because
            // the whole of the matching rule is that a report has to name the
            // request its own trial made.
            auto& slot = requests_[static_cast<std::size_t>(request.trial.ordinal)];
            slot.ordinal = request.trial.ordinal;
            slot.sequence = request.sequence;
            slot.requested_ns = request.requested_ns;
            slot.trial = request.trial;
            slot.stimulus_id = request.stimulus_id;
            slot.occupied = true;
            slot.emitted = true;
            slot.answered = false;
            slot.invalidated = false;
        }
        if (!expected.has_requests || comparator.differed())
        {
            continue;
        }
        if (cursors.requests >= expected.requests.size())
        {
            cursors.exhausted = true;
            continue;
        }
        const auto& recorded = expected.requests[cursors.requests];
        comparator.begin(ReplayItem::presentation_request, cursors.requests, recorded.requested_ns,
                         recorded.trial);
        comparator.integer("sequence", recorded.sequence, request.sequence);
        comparator.integer("requested_ns", recorded.requested_ns, request.requested_ns);
        comparator.integer("onset_ns", recorded.onset_ns, request.onset_ns);
        comparator.integer("valid_until_ns", recorded.valid_until_ns, request.valid_until_ns);
        comparator.integer("duration_ns", recorded.duration_ns, request.duration_ns);
        comparator.integer("paradigm", recorded.paradigm, request.paradigm);
        comparator.integer("phase", recorded.phase, request.phase);
        comparator.integer("cue", static_cast<std::uint64_t>(recorded.cue),
                           static_cast<std::uint64_t>(request.cue));
        comparator.integer("stimulus_id", recorded.stimulus_id, request.stimulus_id);
        comparator.identity("trial", recorded.trial, request.trial);
        ++cursors.requests;
    }
    if (step.trial_decided && expected.has_trials && !comparator.differed())
    {
        if (cursors.trials >= expected.trials.size())
        {
            cursors.exhausted = true;
        }
        else
        {
            const auto& slot = requests_[static_cast<std::size_t>(step.trial.record.trial.ordinal)];
            const bool invalidated = slot.occupied &&
                                     slot.ordinal == step.trial.record.trial.ordinal &&
                                     slot.invalidated;
            const auto& recorded = expected.trials[cursors.trials];
            comparator.begin(ReplayItem::trial, cursors.trials,
                             recorded.trial.record.interval.end_ns, recorded.trial.record.trial);
            comparator.identity("trial", recorded.trial.record.trial, step.trial.record.trial);
            comparator.integer("start_ns", recorded.trial.record.interval.start_ns,
                               step.trial.record.interval.start_ns);
            comparator.integer("end_ns", recorded.trial.record.interval.end_ns,
                               step.trial.record.interval.end_ns);
            comparator.integer("paradigm", recorded.trial.record.paradigm,
                               step.trial.record.paradigm);
            comparator.integer("reason", recorded.trial.record.reason, step.trial.record.reason);
            comparator.integer("task_outcome",
                               static_cast<std::uint64_t>(recorded.trial.record.outcome),
                               static_cast<std::uint64_t>(step.trial.record.outcome));
            comparator.integer("recorded_outcome",
                               static_cast<std::uint64_t>(recorded.recorded_outcome),
                               static_cast<std::uint64_t>(invalidated ? TrialOutcome::aborted
                                                                      : step.trial.record.outcome));
            comparator.flag("invalidated", recorded.invalidated, invalidated);
            compare_schedule(comparator, recorded.trial.schedule, step.trial.schedule);
            comparator.integer("next_trial_start_ns", recorded.trial.timeline.next_trial_start_ns,
                               step.trial.timeline.next_trial_start_ns);
            ++cursors.trials;
        }
    }
}

void SpeechReplay::compare_trial_start(const SpeechSnapshot& snapshot,
                                       const SpeechReplayRecording& recording, Cursors& cursors,
                                       ReplayComparator& comparator) noexcept
{
    // The bridge writes a trial's realized schedule and its intended phases once
    // per trial, at the first step that is inside it. Suppressed the same way
    // here, because a replay that emitted them per step would disagree with
    // every correct recording.
    if (snapshot.state == SpeechState::idle || snapshot.state == SpeechState::complete)
    {
        return;
    }
    if (has_trial_ && snapshot.trial.ordinal == last_trial_)
    {
        return;
    }
    has_trial_ = true;
    last_trial_ = snapshot.trial.ordinal;
    ++cursors.started_trials;
    const auto& expected = recording.expected;

    if (expected.has_schedules && prepared_schedules_ != nullptr && !comparator.differed())
    {
        // For explicit and supported-seeded configurations the schedule used
        // to execute came from the configuration, not this output stream.
        if (snapshot.trial.ordinal >= expected.schedules.size())
        {
            cursors.exhausted = true;
        }
        else
        {
            comparator.begin(ReplayItem::schedule, snapshot.trial.ordinal,
                             snapshot.timeline.trial.start_ns, snapshot.trial);
            compare_schedule(comparator, expected.schedules[snapshot.trial.ordinal],
                             snapshot.schedule);
        }
    }
    if (!expected.has_phases || comparator.differed())
    {
        return;
    }
    for (std::uint8_t i = 0; i < snapshot.timeline.count && !comparator.differed(); ++i)
    {
        if (cursors.phases >= expected.phases.size())
        {
            cursors.exhausted = true;
            break;
        }
        const auto& recorded = expected.phases[cursors.phases];
        const auto& regenerated = snapshot.timeline.phases[i];
        comparator.begin(ReplayItem::phase, cursors.phases, recorded.phase.interval.start_ns,
                         recorded.trial);
        comparator.integer("phase", static_cast<std::uint64_t>(recorded.phase.phase),
                           static_cast<std::uint64_t>(regenerated.phase));
        comparator.integer("cue", static_cast<std::uint64_t>(recorded.phase.cue),
                           static_cast<std::uint64_t>(regenerated.cue));
        comparator.integer("stimulus_id", recorded.phase.stimulus_id, regenerated.stimulus_id);
        comparator.integer("start_ns", recorded.phase.interval.start_ns,
                           regenerated.interval.start_ns);
        comparator.integer("end_ns", recorded.phase.interval.end_ns, regenerated.interval.end_ns);
        comparator.identity("trial", recorded.trial, snapshot.trial);
        ++cursors.phases;
    }
}

ContractStatus SpeechReplay::replay_advance(const SpeechReplayInput& input,
                                            const SpeechReplayRecording& recording,
                                            Cursors& cursors, ReplayComparator& comparator) noexcept
{
    if (terminal_ || !running_ || machine_.complete())
    {
        return ContractStatus::ok;
    }
    const auto snapshot = machine_.snapshot();
    const auto next_ordinal = snapshot.trial.ordinal + 1;
    SpeechStepResult step{};
    ContractStatus status{};
    if (next_ordinal < config_.task.n_trials)
    {
        SpeechTrialSchedule next{};
        if (const auto schedule_status = schedule_for(next_ordinal, recording, next);
            schedule_status != ContractStatus::ok)
        {
            return schedule_status;
        }
        status = machine_.step(input.time_ns, next, step);
    }
    else
    {
        status = machine_.step(input.time_ns, step);
    }
    if (status != ContractStatus::ok)
    {
        return status;
    }
    compare_step(step, recording, cursors, comparator);
    compare_trial_start(step.snapshot, recording, cursors, comparator);
    return ContractStatus::ok;
}

void SpeechReplay::replay_report(const SpeechReplayInput& input,
                                 const SpeechReplayRecording& recording, Cursors& cursors,
                                 ReplayComparator& comparator) noexcept
{
    // Nothing about the report's own content is regenerated. What is derived is
    // the run's decision: could this run attribute the report to a CONTENT
    // request it emitted, and what did the configured severity make of it.
    const bool in_range = input.outcome.trial.ordinal < n_trials_;
    EmittedRequest* slot =
        in_range ? &requests_[static_cast<std::size_t>(input.outcome.trial.ordinal)] : nullptr;
    const bool addressable =
        slot != nullptr && slot->occupied && slot->ordinal == input.outcome.trial.ordinal;
    bool matched = experiments::validate(input.outcome) == ContractStatus::ok && addressable &&
                   slot->emitted && !slot->answered &&
                   input.outcome.request_sequence == slot->sequence &&
                   input.outcome.requested_ns == slot->requested_ns &&
                   input.outcome.stimulus_id == slot->stimulus_id &&
                   same_trial(input.outcome.trial, slot->trial);
    if (matched)
    {
        slot->answered = true;
    }

    AbnormalCondition condition{AbnormalCondition::presentation_report_unmatched};
    bool has_trial = false;
    if (matched && input.outcome.status == PresentationStatus::presented)
    {
        // A presentation that happened as asked is evidence and nothing else.
        // It is recorded as supplying the trial's evidence, and no condition is
        // met, so there is no decision here to regenerate beyond the match.
        condition = AbnormalCondition::unspecified;
    }
    else if (matched)
    {
        condition = AbnormalCondition::presentation_failed;
        has_trial = true;
    }
    auto response = AbnormalResponse::recorded;
    if (condition != AbnormalCondition::unspecified)
    {
        const auto policy = policy_for(config_.abnormal, condition);
        // SpeechMachine owns when a trial ends -- a trial ends when its CONTENT
        // phase reaches its end instant -- so `abort_trial` marks the trial in
        // flight inadmissible rather than ending it.
        response = abnormal_response_for(policy, has_trial, /*can_end_trial=*/false,
                                         /*input_refused=*/!matched);
    }
    if (response == AbnormalResponse::trial_invalidated && addressable)
    {
        slot->invalidated = true;
    }
    if (response == AbnormalResponse::session_aborted)
    {
        terminal_ = true;
        running_ = false;
    }

    if (!recording.expected.has_reports || comparator.differed())
    {
        return;
    }
    if (cursors.reports >= recording.expected.reports.size())
    {
        cursors.exhausted = true;
        return;
    }
    const auto& recorded = recording.expected.reports[cursors.reports];
    comparator.begin(ReplayItem::presentation_request, cursors.reports, input.outcome.requested_ns,
                     input.outcome.trial);
    comparator.flag("matched_request", recorded.matched, matched);
    comparator.integer("condition", static_cast<std::uint64_t>(recorded.condition),
                       static_cast<std::uint64_t>(condition));
    comparator.integer("response", static_cast<std::uint64_t>(recorded.response),
                       static_cast<std::uint64_t>(response));
    ++cursors.reports;
}

ReplayReport SpeechReplay::run(const SpeechReplayRecording& recording,
                               ReplayIncompletePolicy policy) noexcept
{
    ReplayReport report{};
    if (!prepared_)
    {
        report.rejection = ReplayRejection::configuration_invalid;
        return report;
    }
    const bool recorded_fallback = prepared_schedules_ == nullptr;
    auto candidate_provenance = provenance(
        recorded_fallback
            ? recording.expected.schedules
            : std::span<const SpeechTrialSchedule>{prepared_schedules_.get(), n_trials_});
    if (recorded_fallback && recording.expected.schedules.size() != n_trials_)
    {
        // The recording holds one schedule per trial that actually started,
        // while its session provenance fingerprints the full schedule prepared
        // by the original build. This build cannot reconstruct that unseen
        // suffix. The used prefix remains validated by SpeechMachine as
        // fallback input.
        candidate_provenance.realized_schedule_fingerprint =
            recording.provenance.realized_schedule_fingerprint;
    }
    const auto rejection = check_provenance(recording.provenance, candidate_provenance);
    if (rejection != ReplayRejection::none)
    {
        report.rejection = rejection;
        return report;
    }
    ExperimentTimeNs previous = recording.origin_ns;
    for (const auto& input : recording.inputs)
    {
        if (!valid_replay_input(input))
        {
            report.rejection = ReplayRejection::evidence_invalid;
            return report;
        }
        // Only an advance carries the run's own clock. A presenter's report is
        // stamped with the instant its request was made, which is behind the
        // run by construction, and requiring it to be monotonic here would
        // reject exactly the recordings this replay exists for.
        if (input.kind == SpeechReplayInputKind::advance)
        {
            if (input.time_ns < previous)
            {
                report.rejection = ReplayRejection::evidence_invalid;
                return report;
            }
            previous = input.time_ns;
        }
    }

    const auto& expected = recording.expected;
    if (!expected.has_transitions || !expected.has_events || !expected.has_requests ||
        !expected.has_phases || !expected.has_schedules || !expected.has_trials ||
        !expected.has_reports)
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

    SpeechTrialSchedule first{};
    const auto first_schedule_status = schedule_for(0, recording, first);
    if (first_schedule_status != ContractStatus::ok)
    {
        report.rejection = first_schedule_status == ContractStatus::version_unsupported
                               ? ReplayRejection::sampler_version_unsupported
                               : ReplayRejection::evidence_invalid;
        return report;
    }
    SpeechStepResult opening{};
    if (machine_.start(config_.paradigm, config_.task, first, recording.origin_ns, opening) !=
        ContractStatus::ok)
    {
        report.rejection = ReplayRejection::configuration_invalid;
        return report;
    }
    running_ = true;
    compare_step(opening, recording, cursors, comparator);
    compare_trial_start(opening.snapshot, recording, cursors, comparator);

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
        case SpeechReplayInputKind::advance:
            regeneration_status = replay_advance(input, recording, cursors, comparator);
            break;
        case SpeechReplayInputKind::presentation_report:
            replay_report(input, recording, cursors, comparator);
            break;
        case SpeechReplayInputKind::halt:
        {
            const auto snapshot = machine_.snapshot();
            const bool in_trial = running_ && !machine_.complete();
            const auto derived =
                abnormal_response_for(escalate(policy_for(config_.abnormal, input.condition),
                                               AbnormalPolicy::abort_trial),
                                      in_trial, /*can_end_trial=*/false, /*input_refused=*/true);
            if (!comparator.differed())
            {
                comparator.begin(ReplayItem::trial, cursors.trials, input.time_ns, snapshot.trial);
                comparator.integer("response", static_cast<std::uint64_t>(input.response),
                                   static_cast<std::uint64_t>(derived));
            }
            terminal_ = true;
            running_ = false;
            break;
        }
        }
        if (regeneration_status != ContractStatus::ok)
        {
            break;
        }
    }

    if (note_regeneration_outcome(report, comparator, regeneration_status))
    {
        return report;
    }
    if (expected.has_schedules && expected.schedules.size() < cursors.started_trials &&
        report.completeness != ReplayCompleteness::trace_loss_recorded)
    {
        report.verdict = ReplayVerdict::mismatch;
        report.first_mismatch.item = ReplayItem::stream_length;
        report.first_mismatch.field = "recorded_short";
        report.first_mismatch.recorded = expected.schedules.size();
        report.first_mismatch.regenerated = cursors.started_trials;
        return report;
    }
    if (cursors.exhausted && report.completeness != ReplayCompleteness::trace_loss_recorded)
    {
        report.verdict = ReplayVerdict::mismatch;
        report.first_mismatch.item = ReplayItem::stream_length;
        report.first_mismatch.field = "recorded_short";
        return report;
    }
    if (note_recorded_extra(report, expected.has_transitions, expected.transitions.size(),
                            cursors.transitions) ||
        note_recorded_extra(report, expected.has_events, expected.events.size(), cursors.events) ||
        note_recorded_extra(report, expected.has_requests, expected.requests.size(),
                            cursors.requests) ||
        note_recorded_extra(report, expected.has_phases, expected.phases.size(), cursors.phases) ||
        note_recorded_extra(report, expected.has_schedules, expected.schedules.size(),
                            cursors.started_trials) ||
        note_recorded_extra(report, expected.has_trials, expected.trials.size(), cursors.trials) ||
        note_recorded_extra(report, expected.has_reports, expected.reports.size(), cursors.reports))
    {
        return report;
    }
    report.verdict = report.completeness == ReplayCompleteness::complete
                         ? ReplayVerdict::match
                         : ReplayVerdict::incomplete;
    return report;
}

} // namespace neurale::experiments::speech
