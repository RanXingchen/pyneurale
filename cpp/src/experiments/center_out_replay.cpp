/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "replay_verdict.h"

#include <neurale/experiments/center_out_replay.h>

#include <algorithm>
#include <cmath>

namespace neurale::experiments::center_out
{
namespace
{
/// The version of the transform that produced the assisted velocity, by method.
///
/// The same lookup the recording bridge does, and it has to be: `policy_version`
/// is what a replay compares its own transform against, so the two must derive
/// it from the method the same way.
[[nodiscard]] std::uint32_t assistance_version(assistance::AssistanceMethod method) noexcept
{
    switch (method)
    {
    case assistance::AssistanceMethod::linear_blend:
        return assistance::kLinearBlendVersion1;
    case assistance::AssistanceMethod::ortho_impedance:
        return assistance::kOrthoImpedanceVersion1;
    case assistance::AssistanceMethod::none:
        break;
    }
    return 0;
}

void compare_velocity(ReplayComparator& comparator, std::string_view field,
                      const assistance::VelocityVector& recorded,
                      const assistance::VelocityVector& regenerated) noexcept
{
    comparator.integer("space", recorded.space, regenerated.space);
    comparator.integer("dimension", recorded.dim, regenerated.dim);
    const auto width = recorded.dim < regenerated.dim ? recorded.dim : regenerated.dim;
    for (std::uint8_t i = 0; i < width; ++i)
    {
        comparator.real(field, recorded.values[i], regenerated.values[i]);
    }
}

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

void compare_record(ReplayComparator& comparator, const TrialRecord& recorded,
                    const TrialRecord& regenerated) noexcept
{
    comparator.identity("trial", recorded.trial, regenerated.trial);
    comparator.integer("start_ns", recorded.interval.start_ns, regenerated.interval.start_ns);
    comparator.integer("end_ns", recorded.interval.end_ns, regenerated.interval.end_ns);
    comparator.integer("paradigm", recorded.paradigm, regenerated.paradigm);
    comparator.integer("outcome", static_cast<std::uint64_t>(recorded.outcome),
                       static_cast<std::uint64_t>(regenerated.outcome));
    comparator.integer("reason", recorded.reason, regenerated.reason);
}

[[nodiscard]] bool valid_replay_input(const CenterOutReplayInput& input) noexcept
{
    if (!center_out_replay_input_kind_declared(input.kind))
    {
        return false;
    }
    if (input.kind == CenterOutReplayInputKind::observation)
    {
        return input.condition == AbnormalCondition::unspecified &&
               input.response == AbnormalResponse::recorded && std::isfinite(input.decoded[0]) &&
               std::isfinite(input.decoded[1]);
    }
    return abnormal_condition_declared(input.condition) &&
           input.condition != AbnormalCondition::unspecified &&
           abnormal_response_declared(input.response);
}
} // namespace

CenterOutConditionHandling center_out_condition_handling(AbnormalCondition condition) noexcept
{
    switch (condition)
    {
    case AbnormalCondition::input_schema_mismatch:
        // Not the stream the run prepared against, and the next frame is wrong
        // in exactly the same way. A task that restarted its segment per
        // malformed frame would restart forever while recording nothing usable.
        return CenterOutConditionHandling{AbnormalPolicy::abort_session, true};
    case AbnormalCondition::decoded_command_invalid:
        // The value never reaches the cursor whatever the configuration says,
        // which is the refusal; how severely the run treats it is the
        // configuration's to choose, so there is no floor.
        return CenterOutConditionHandling{AbnormalPolicy::record, true};
    case AbnormalCondition::input_stale:
    case AbnormalCondition::source_discontinuity:
        // A hold and a reach are accumulated across consecutive observations,
        // and there are none across a gap. What the configuration still chooses
        // is whether the run continues at all.
        return CenterOutConditionHandling{AbnormalPolicy::abort_trial, false};
    case AbnormalCondition::emergency_stop:
        // The trial in flight ended rather than being left to look unfinished,
        // and every input after the stop is refused.
        return CenterOutConditionHandling{AbnormalPolicy::abort_trial, true};
    case AbnormalCondition::input_after_terminal:
        // The run it would have ended has already ended.
        return CenterOutConditionHandling{AbnormalPolicy::record, true};
    default:
        break;
    }
    // A condition Center-Out does not raise itself imposes no floor and refuses
    // nothing: it is the configuration's alone to decide.
    return CenterOutConditionHandling{AbnormalPolicy::record, false};
}

struct CenterOutReplay::Cursors
{
    std::size_t transitions{};
    std::size_t events{};
    std::size_t trials{};
    std::size_t aborts{};
    std::size_t targets{};
    std::size_t cursor{};
    std::size_t vel{};
    std::size_t guidance{};
    /// Whether a recorded stream ran out before the replay stopped producing.
    ///
    /// Latched rather than reported per stream: a recording that lost records
    /// says so once, and the first stream to run dry is the whole finding.
    bool exhausted{};
};

ContractStatus CenterOutReplay::prepare(const CenterOutReplayConfig& config) noexcept
{
    if (config.paradigm == kUnsetParadigmId)
    {
        return ContractStatus::identity_missing;
    }
    if (const auto status = validate(config.task); status != ContractStatus::ok)
    {
        return status;
    }
    if (const auto status = experiments::validate(config.velocity_space);
        status != ContractStatus::ok)
    {
        return status;
    }
    if (const auto status = experiments::validate(config.abnormal); status != ContractStatus::ok)
    {
        return status;
    }
    if (!std::isfinite(config.cursor_min.x) || !std::isfinite(config.cursor_min.y) ||
        !std::isfinite(config.cursor_max.x) || !std::isfinite(config.cursor_max.y) ||
        config.cursor_min.x > config.cursor_max.x || config.cursor_min.y > config.cursor_max.y ||
        config.initial_position.x < config.cursor_min.x ||
        config.initial_position.x > config.cursor_max.x ||
        config.initial_position.y < config.cursor_min.y ||
        config.initial_position.y > config.cursor_max.y)
    {
        return ContractStatus::parameter_out_of_range;
    }
    if (config.assistance_method == assistance::AssistanceMethod::linear_blend)
    {
        if (const auto status = assistance::validate(config.linear_assistance);
            status != ContractStatus::ok)
        {
            return status;
        }
    }
    // An unconfigured guidance is not a broken one: the run simply blended
    // against no reference velocity. Configuring the guidance object with it is
    // still how the replay reaches the same state the run's guidance was in.
    if (config.guidance.geometry_unit != GeometryUnit::unspecified)
    {
        if (const auto status = validate(config.guidance); status != ContractStatus::ok)
        {
            return status;
        }
        if (const auto status = validate_against(config.guidance, config.task);
            status != ContractStatus::ok)
        {
            return status;
        }
    }
    if (guidance_.configure(config.guidance) != ContractStatus::ok)
    {
        return ContractStatus::parameter_out_of_range;
    }
    config_ = config;
    prepared_ = true;
    reset();
    return ContractStatus::ok;
}

void CenterOutReplay::reset() noexcept
{
    machine_.reset();
    guidance_.reset();
    position_ = config_.initial_position;
    last_time_ns_ = 0;
    trial_started_ns_ = 0;
    last_target_ = kUnsetTargetId;
    has_last_time_ = false;
    has_trial_ = false;
    has_target_ = false;
    pending_restart_ = false;
    running_ = false;
    terminal_ = false;
}

ReplayProvenance CenterOutReplay::provenance() const noexcept
{
    const auto configuration = configuration_fingerprint(config_.task);
    // The configuration's sampler version, not this build's, for the same
    // reason the bridge records it that way: a configuration may legally name a
    // sampler this build cannot regenerate, and a provenance that substituted
    // the build's would contradict the fingerprint beside it.
    const ScheduleIdentity identity{
        .seed = config_.task.seed,
        .sampler_version = config_.task.sampler_version,
        .configuration_fingerprint = configuration,
        .catalog_fingerprint = 0,
    };
    return ReplayProvenance{
        .paradigm = config_.paradigm,
        .experiment_version = kCenterOutRecordVersion,
        .configuration_fingerprint = configuration,
        .schedule_fingerprint = schedule_fingerprint(identity),
        // Center-Out draws its targets as the run goes rather than freezing a
        // schedule beforehand, so there is no realized schedule to digest.
        .realized_schedule_fingerprint = 0,
        .seed = config_.task.seed,
        .sampler_version = config_.task.sampler_version,
        .metric_version = 0,
        .policy_version = assistance_version(config_.assistance_method),
    };
}

ContractStatus CenterOutReplay::begin_segment(ExperimentTimeNs time_ns,
                                              CenterOutStepResult& result) noexcept
{
    machine_.reset();
    guidance_.reset();
    has_trial_ = false;
    trial_started_ns_ = time_ns;
    const auto status = machine_.start(config_.paradigm, config_.task, time_ns, result);
    if (status == ContractStatus::ok)
    {
        note_trial_boundaries(result);
    }
    return status;
}

void CenterOutReplay::note_trial_boundaries(const CenterOutStepResult& result) noexcept
{
    for (std::uint8_t i = 0; i < result.n_events; ++i)
    {
        const auto& event = result.events[i];
        if (event.kind == ExperimentEventKind::trial_start)
        {
            trial_started_ns_ = event.time_ns;
            has_trial_ = true;
        }
        else if (event.kind == ExperimentEventKind::trial_stop)
        {
            has_trial_ = false;
        }
    }
}

void CenterOutReplay::forget_target() noexcept
{
    has_target_ = false;
    last_target_ = kUnsetTargetId;
}

void CenterOutReplay::compare_step(const CenterOutStepResult& step,
                                   const CenterOutReplayRecording& recording, Cursors& cursors,
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
    if (step.trial_decided && expected.has_trials && !comparator.differed())
    {
        if (cursors.trials >= expected.trials.size())
        {
            cursors.exhausted = true;
        }
        else
        {
            const auto& recorded = expected.trials[cursors.trials];
            comparator.begin(ReplayItem::trial, cursors.trials, recorded.record.interval.end_ns,
                             recorded.record.trial);
            compare_record(comparator, recorded.record, step.trial.record);
            comparator.integer("outward_target", recorded.outward_target,
                               step.trial.outward_target);
            comparator.integer("outward_index", recorded.outward_idx, step.trial.outward_idx);
            comparator.integer("decided_phase", static_cast<std::uint64_t>(recorded.decided_phase),
                               static_cast<std::uint64_t>(step.trial.decided_phase));
            comparator.integer("center_acquire_ns", recorded.center_acquire_ns,
                               step.trial.center_acquire_ns);
            comparator.integer("outward_acquire_ns", recorded.outward_acquire_ns,
                               step.trial.outward_acquire_ns);
            ++cursors.trials;
        }
    }
}

void CenterOutReplay::compare_target(const CenterOutSnapshot& snapshot, ExperimentTimeNs time_ns,
                                     const CenterOutReplayRecording& recording, Cursors& cursors,
                                     ReplayComparator& comparator) noexcept
{
    // The same suppression the bridge applies: a target that stayed up across a
    // step has not had a second onset. A replay that recorded one per step
    // would disagree with every correct recording.
    if (snapshot.active_target == kUnsetTargetId)
    {
        return;
    }
    if (has_target_ && snapshot.active_target == last_target_)
    {
        return;
    }
    has_target_ = true;
    last_target_ = snapshot.active_target;
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
    comparator.begin(ReplayItem::target, cursors.targets, recorded.time_ns, recorded.trial);
    comparator.integer("time_ns", recorded.time_ns, time_ns);
    comparator.integer("target_id", recorded.target_id, snapshot.active_target);
    comparator.real("x", recorded.pos.x, snapshot.active_position.x);
    comparator.real("y", recorded.pos.y, snapshot.active_position.y);
    comparator.integer("phase", static_cast<std::uint64_t>(recorded.phase),
                       static_cast<std::uint64_t>(snapshot.phase));
    comparator.integer("outward_target", recorded.outward_target, snapshot.outward_target);
    comparator.integer("outward_index", recorded.outward_idx, snapshot.outward_idx);
    comparator.identity("trial", recorded.trial, snapshot.trial);
    ++cursors.targets;
}

ContractStatus CenterOutReplay::replay_observation(const CenterOutReplayInput& input,
                                                   const CenterOutReplayRecording& recording,
                                                   Cursors& cursors,
                                                   ReplayComparator& comparator) noexcept
{
    if (terminal_ || machine_.complete())
    {
        // Input the run refused because it had already ended produced no
        // semantic output, so there is nothing to regenerate and nothing to
        // compare. The refusal itself was recorded as an abnormal condition,
        // which is not a stream this replay reproduces.
        return ContractStatus::ok;
    }

    CenterOutStepResult segment_start{};
    const bool restarting = pending_restart_;
    if (restarting)
    {
        if (const auto status = begin_segment(input.time_ns, segment_start);
            status != ContractStatus::ok)
        {
            return status;
        }
        last_time_ns_ = input.time_ns;
        has_last_time_ = true;
        pending_restart_ = false;
        forget_target();
        compare_step(segment_start, recording, cursors, comparator);
        compare_target(segment_start.snapshot, input.time_ns, recording, cursors, comparator);
    }

    const DurationNs dt_ns = has_last_time_ ? input.time_ns - last_time_ns_ : 0;
    const auto before = position_;

    assistance::VelocityVector decoded{};
    decoded.space = config_.velocity_space.id;
    decoded.dim = 2;
    decoded.values[0] = input.decoded[0];
    decoded.values[1] = input.decoded[1];

    assistance::VelocityVector reference{};
    reference.space = config_.velocity_space.id;
    reference.dim = 2;
    CenterOutGuidanceSample guidance_sample{};
    const auto snapshot_before = machine_.snapshot();
    if (snapshot_before.active_target != kUnsetTargetId)
    {
        const TargetPlacement target{snapshot_before.active_target,
                                     snapshot_before.active_position};
        if (const auto status = guidance_.update(target, position_, dt_ns, guidance_sample);
            status != ContractStatus::ok)
        {
            return status;
        }
        reference.values[0] = guidance_sample.vel.x;
        reference.values[1] = guidance_sample.vel.y;
    }
    else
    {
        guidance_.reset();
    }

    assistance::VelocityVector assisted = decoded;
    if (config_.assistance_method == assistance::AssistanceMethod::linear_blend)
    {
        if (const auto status = assistance::blend_velocity(
                config_.velocity_space, decoded, reference, config_.linear_assistance, assisted);
            status != ContractStatus::ok)
        {
            return status;
        }
    }

    const auto dt_seconds = static_cast<double>(dt_ns) / static_cast<double>(kNanosecondsPerSecond);
    const auto next_x = position_.x + assisted.values[0] * dt_seconds;
    const auto next_y = position_.y + assisted.values[1] * dt_seconds;
    if (!std::isfinite(next_x) || !std::isfinite(next_y))
    {
        return ContractStatus::numerical_failure;
    }
    position_ = WorkspacePoint{std::clamp(next_x, config_.cursor_min.x, config_.cursor_max.x),
                               std::clamp(next_y, config_.cursor_min.y, config_.cursor_max.y)};

    CenterOutStepResult step{};
    if (const auto status = machine_.step(input.time_ns, position_, step);
        status != ContractStatus::ok)
    {
        position_ = before;
        return status;
    }
    note_trial_boundaries(step);
    last_time_ns_ = input.time_ns;
    has_last_time_ = true;

    // Compared in the order the run computed them, so a report names the value
    // a difference started at rather than the first place it showed up. A
    // cursor that drifted makes every transition after it differ, and a reader
    // told about the transition would have to work backwards to the cursor.
    const auto& expected = recording.expected;
    if (expected.has_cursor && !comparator.differed())
    {
        if (cursors.cursor >= expected.cursor.size())
        {
            cursors.exhausted = true;
        }
        else
        {
            const auto& recorded = expected.cursor[cursors.cursor];
            comparator.begin(ReplayItem::cursor, cursors.cursor, recorded.time_ns, recorded.trial);
            comparator.integer("time_ns", recorded.time_ns, input.time_ns);
            comparator.real("before_x", recorded.before.x, before.x);
            comparator.real("before_y", recorded.before.y, before.y);
            comparator.real("after_x", recorded.after.x, position_.x);
            comparator.real("after_y", recorded.after.y, position_.y);
            comparator.integer("dt_ns", recorded.dt_ns, dt_ns);
            comparator.integer("frame_sequence", recorded.frame_sequence, input.frame_sequence);
            comparator.integer("sample_index", recorded.sample_idx, input.sample_idx);
            comparator.integer("state", static_cast<std::uint64_t>(recorded.state),
                               static_cast<std::uint64_t>(step.snapshot.state));
            comparator.identity("trial", recorded.trial, step.snapshot.trial);
            ++cursors.cursor;
        }
    }
    if (expected.has_velocity && !comparator.differed())
    {
        if (cursors.vel >= expected.vel.size())
        {
            cursors.exhausted = true;
        }
        else
        {
            const auto& recorded = expected.vel[cursors.vel];
            comparator.begin(ReplayItem::assisted_velocity, cursors.vel, recorded.time_ns,
                             recorded.trial);
            comparator.integer("time_ns", recorded.time_ns, input.time_ns);
            compare_velocity(comparator, "decoded", recorded.decoded, decoded);
            compare_velocity(comparator, "guidance", recorded.guidance, reference);
            compare_velocity(comparator, "assisted", recorded.assisted, assisted);
            comparator.identity("trial", recorded.trial, step.snapshot.trial);
            ++cursors.vel;
        }
    }
    if (expected.has_guidance && !comparator.differed())
    {
        if (cursors.guidance >= expected.guidance.size())
        {
            cursors.exhausted = true;
        }
        else
        {
            const auto& recorded = expected.guidance[cursors.guidance];
            comparator.begin(ReplayItem::guidance_sample, cursors.guidance, recorded.time_ns,
                             step.snapshot.trial);
            comparator.integer("time_ns", recorded.time_ns, input.time_ns);
            comparator.real("vx", recorded.sample.vel.x, guidance_sample.vel.x);
            comparator.real("vy", recorded.sample.vel.y, guidance_sample.vel.y);
            comparator.real("distance", recorded.sample.distance, guidance_sample.distance);
            comparator.real("speed_towards_target", recorded.sample.speed_towards_target,
                            guidance_sample.speed_towards_target);
            comparator.flag("retargeted", recorded.sample.retargeted, guidance_sample.retargeted);
            comparator.integer("phase", static_cast<std::uint64_t>(recorded.sample.state.phase),
                               static_cast<std::uint64_t>(guidance_sample.state.phase));
            comparator.integer("state_target", recorded.sample.state.target,
                               guidance_sample.state.target);
            ++cursors.guidance;
        }
    }
    compare_step(step, recording, cursors, comparator);
    compare_target(step.snapshot, input.time_ns, recording, cursors, comparator);
    return ContractStatus::ok;
}

void CenterOutReplay::replay_decision(const CenterOutReplayInput& input,
                                      const CenterOutReplayRecording& recording, Cursors& cursors,
                                      ReplayComparator& comparator) noexcept
{
    const auto snapshot = machine_.snapshot();
    const bool in_trial = running_ && !machine_.complete();
    const auto handling = center_out_condition_handling(input.condition);
    const auto policy = escalate(policy_for(config_.abnormal, input.condition), handling.floor);
    // Center-Out can genuinely end the trial in flight, so `can_end_trial` is
    // true here for the same reason it is true in the controller: the machine
    // is reset and a new segment begins at the next accepted observation.
    const auto response =
        abnormal_response_for(policy, in_trial, /*can_end_trial=*/true, handling.input_refused);

    const bool ended = response == AbnormalResponse::trial_aborted ||
                       response == AbnormalResponse::session_aborted;
    if (ended && in_trial && has_trial_)
    {
        TrialRecord regenerated{};
        regenerated.trial = snapshot.trial;
        regenerated.interval =
            TimeInterval{trial_started_ns_,
                         input.time_ns < trial_started_ns_ ? trial_started_ns_ : input.time_ns};
        regenerated.paradigm = config_.paradigm;
        regenerated.outcome = TrialOutcome::aborted;
        if (recording.expected.has_aborts && !comparator.differed())
        {
            if (cursors.aborts >= recording.expected.aborts.size())
            {
                cursors.exhausted = true;
            }
            else
            {
                const auto& recorded = recording.expected.aborts[cursors.aborts];
                comparator.begin(ReplayItem::trial, cursors.aborts, recorded.record.interval.end_ns,
                                 recorded.record.trial);
                compare_record(comparator, recorded.record, regenerated);
                comparator.integer("condition", static_cast<std::uint64_t>(recorded.condition),
                                   static_cast<std::uint64_t>(input.condition));
                comparator.integer("response", static_cast<std::uint64_t>(recorded.response),
                                   static_cast<std::uint64_t>(response));
                ++cursors.aborts;
            }
        }
    }
    else if (!comparator.differed())
    {
        // A decision that ended no trial still has a response, and the response
        // is the thing the configured severity decides. Compared on its own so
        // that a replay under a different policy set disagrees even when the
        // run had no trial in flight to lose.
        comparator.begin(ReplayItem::trial, cursors.aborts, input.time_ns, snapshot.trial);
        comparator.integer("response", static_cast<std::uint64_t>(input.response),
                           static_cast<std::uint64_t>(response));
    }

    if (response == AbnormalResponse::trial_aborted)
    {
        machine_.reset();
        guidance_.reset();
        has_last_time_ = false;
        has_trial_ = false;
        pending_restart_ = true;
        forget_target();
    }
    if (response == AbnormalResponse::session_aborted)
    {
        terminal_ = true;
        running_ = false;
        forget_target();
    }
}

ReplayReport CenterOutReplay::run(const CenterOutReplayRecording& recording,
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
    // The inputs have to be a timeline the run could have consumed before any
    // of them is replayed. A regressed instant is not a difference between two
    // runs; it is evidence that cannot describe one.
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
    if (!expected.has_transitions || !expected.has_events || !expected.has_trials ||
        !expected.has_aborts || !expected.has_targets || !expected.has_cursor ||
        !expected.has_velocity || !expected.has_guidance)
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

    CenterOutStepResult opening{};
    const auto opening_status = begin_segment(recording.origin_ns, opening);
    if (opening_status != ContractStatus::ok)
    {
        report.rejection = opening_status == ContractStatus::version_unsupported
                               ? ReplayRejection::sampler_version_unsupported
                               : ReplayRejection::configuration_invalid;
        return report;
    }
    position_ = config_.initial_position;
    last_time_ns_ = recording.origin_ns;
    has_last_time_ = true;
    running_ = true;
    compare_step(opening, recording, cursors, comparator);
    compare_target(opening.snapshot, recording.origin_ns, recording, cursors, comparator);

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
        case CenterOutReplayInputKind::observation:
            regeneration_status = replay_observation(input, recording, cursors, comparator);
            break;
        case CenterOutReplayInputKind::decision:
            replay_decision(input, recording, cursors, comparator);
            break;
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
    if (cursors.exhausted && report.completeness != ReplayCompleteness::trace_loss_recorded)
    {
        // A recording that claims to be whole and holds fewer outputs than the
        // run produced disagrees with the run, and there is no item to say so
        // on -- which is what ::ReplayItem::stream_length exists for.
        report.verdict = ReplayVerdict::mismatch;
        report.first_mismatch.item = ReplayItem::stream_length;
        report.first_mismatch.field = "recorded_short";
        return report;
    }
    if (note_recorded_extra(report, expected.has_transitions, expected.transitions.size(),
                            cursors.transitions) ||
        note_recorded_extra(report, expected.has_events, expected.events.size(), cursors.events) ||
        note_recorded_extra(report, expected.has_trials, expected.trials.size(), cursors.trials) ||
        note_recorded_extra(report, expected.has_aborts, expected.aborts.size(), cursors.aborts) ||
        note_recorded_extra(report, expected.has_targets, expected.targets.size(),
                            cursors.targets) ||
        note_recorded_extra(report, expected.has_cursor, expected.cursor.size(), cursors.cursor) ||
        note_recorded_extra(report, expected.has_velocity, expected.vel.size(), cursors.vel) ||
        note_recorded_extra(report, expected.has_guidance, expected.guidance.size(),
                            cursors.guidance))
    {
        return report;
    }
    report.verdict = report.completeness == ReplayCompleteness::complete
                         ? ReplayVerdict::match
                         : ReplayVerdict::incomplete;
    return report;
}

} // namespace neurale::experiments::center_out
