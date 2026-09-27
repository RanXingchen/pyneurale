// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "target_schedule.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <neurale/experiments/center_out.h>
#include <numbers>

namespace neurale::experiments::center_out
{
namespace
{

[[nodiscard]] ContractStatus check_finite(double value) noexcept
{
    return std::isfinite(value) ? ContractStatus::ok : ContractStatus::value_not_finite;
}

[[nodiscard]] ContractStatus check_point(const WorkspacePoint& point) noexcept
{
    if (const ContractStatus status = check_finite(point.x); status != ContractStatus::ok)
        return status;
    return check_finite(point.y);
}

// A slot past the declared count must be untouched, exactly as command values
// are. Two layouts that place the same targets then also have the same bytes,
// which is what lets layout_fingerprint() digest the whole array without
// consulting the count first.
[[nodiscard]] bool placement_is_unset(const TargetPlacement& target) noexcept
{
    return target.id == kUnsetTargetId && target.pos.x == 0.0 && target.pos.y == 0.0;
}

// Negative zero is folded onto zero so that geometry in the same place digests
// the same. A non-finite coordinate is absorbed by its bits and is not folded:
// validate() rejects one before it can reach a persisted fingerprint, and
// inventing an ordering among NaN payloads here would hide that.
void absorb_double(FingerprintAccumulator& accumulator, double value) noexcept
{
    const double folded = value == 0.0 ? 0.0 : value;
    accumulator.absorb(std::bit_cast<std::uint64_t>(folded));
}

void absorb_placement(FingerprintAccumulator& accumulator, const TargetPlacement& target) noexcept
{
    accumulator.absorb(target.id);
    absorb_double(accumulator, target.pos.x);
    absorb_double(accumulator, target.pos.y);
}

void absorb_durations(FingerprintAccumulator& accumulator, const PhaseDurations& durations) noexcept
{
    accumulator.absorb(durations.to_center);
    accumulator.absorb(durations.to_out);
}

} // namespace

double radial_spoke_step(std::uint32_t count) noexcept
{
    const std::uint32_t divisor = count > kMinRadialSpokes ? count : kMinRadialSpokes;
    return 2.0 * std::numbers::pi / static_cast<double>(divisor);
}

ContractStatus radial_position(double radius, std::uint32_t count, std::uint32_t spoke,
                               WorkspacePoint& pos) noexcept
{
    if (const ContractStatus status = check_finite(radius); status != ContractStatus::ok)
        return status;
    if (radius <= 0.0)
        return ContractStatus::parameter_out_of_range;
    const double theta = radial_spoke_step(count) * static_cast<double>(spoke);
    pos = WorkspacePoint{radius * std::cos(theta), radius * std::sin(theta)};
    return ContractStatus::ok;
}

ContractStatus build_radial_layout(const RadialLayoutRequest& request,
                                   CenterOut2DLayout& layout) noexcept
{
    if (const ContractStatus status = validate(request); status != ContractStatus::ok)
        return status;

    // Built into a local and copied out only on success, so a caller's layout is
    // never left half-written by a failure partway through the ring.
    CenterOut2DLayout built{};
    built.center = TargetPlacement{request.center_id, WorkspacePoint{0.0, 0.0}};
    built.count = request.count;
    for (std::size_t i = 0; i < request.count; ++i)
    {
        WorkspacePoint pos{};
        if (const ContractStatus status =
                radial_position(request.radius, request.count, request.spokes[i], pos);
            status != ContractStatus::ok)
            return status;
        built.surrounding[i] = TargetPlacement{request.ids[i], pos};
    }
    if (const ContractStatus status = validate(built); status != ContractStatus::ok)
        return status;
    layout = built;
    return ContractStatus::ok;
}

ContractStatus contains_cursor(const AcceptanceRegion& region,
                               const CursorGeometry& cursor_geometry, const WorkspacePoint& target,
                               const WorkspacePoint& cursor, bool& inside) noexcept
{
    if (const ContractStatus status = validate(region); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(cursor_geometry); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = check_point(target); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = check_point(cursor); status != ContractStatus::ok)
        return status;

    // Keep edge comparisons in this order; regrouping changes ulp-boundary cases.
    //
    // A region narrower than the cursor then fails on both edges of that axis
    // and the answer is false everywhere, which is the honest result of the
    // question rather than an error -- the caller that configured it that way is
    // refused by validate(const CenterOut2DConfig&).
    const double extent = cursor_geometry.extent;
    inside = cursor.x - extent >= target.x - region.half_extent_x &&
             cursor.x + extent <= target.x + region.half_extent_x &&
             cursor.y - extent >= target.y - region.half_extent_y &&
             cursor.y + extent <= target.y + region.half_extent_y;
    return ContractStatus::ok;
}

ContractStatus contains_point(const AcceptanceRegion& region, const WorkspacePoint& target,
                              const WorkspacePoint& cursor, bool& inside) noexcept
{
    return contains_cursor(region, CursorGeometry{0.0}, target, cursor, inside);
}

ContractStatus select_outward_target(const CenterOut2DConfig& config, TrialOrdinal trial,
                                     std::uint64_t successes, std::uint8_t& idx) noexcept
{
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;

    const std::uint64_t count = config.layout.count;
    if (config.selection == TargetSelectionPolicy::repeat_until_success)
    {
        // No draw is made, so the sampler version is not consulted: this branch
        // reproduces on any build that can read the configuration at all.
        idx = static_cast<std::uint8_t>(successes % count);
        return ContractStatus::ok;
    }

    // A build that cannot evaluate the recorded sampler must not answer with a
    // value from a different one. Reported here rather than in validate(), which
    // follows validate(const ScheduleIdentity&) in accepting a version it cannot
    // regenerate, because such a session still replays from its recorded
    // schedule.
    if (!sampler_version_supported(config.sampler_version))
        return ContractStatus::version_unsupported;

    if (config.selection == TargetSelectionPolicy::balanced_shuffled_cycles)
    {
        return detail::balanced_cycle_index<kMaxSurroundingTargets>(
            config.seed, kTargetSelectionStream, trial, config.layout.count, idx);
    }
    else
    {
        std::uint64_t value = 0;
        const DrawKey key{config.seed, kTargetSelectionStream, trial, 0};
        if (const ContractStatus status = sample_index(key, count, value);
            status != ContractStatus::ok)
            return status;
        idx = static_cast<std::uint8_t>(value);
    }
    return ContractStatus::ok;
}

ContractStatus validate(const TargetPlacement& target) noexcept
{
    if (target.id == kUnsetTargetId)
        return ContractStatus::identity_missing;
    return check_point(target.pos);
}

ContractStatus validate(const CenterOut2DLayout& layout) noexcept
{
    if (layout.count == 0 || layout.count > kMaxSurroundingTargets)
        return ContractStatus::target_set_invalid;
    if (const ContractStatus status = validate(layout.center); status != ContractStatus::ok)
        return status;

    for (std::size_t i = 0; i < layout.count; ++i)
    {
        const TargetPlacement& target = layout.surrounding[i];
        if (const ContractStatus status = validate(target); status != ContractStatus::ok)
            return status;
        // A surrounding target that reuses the centre's identifier would make a
        // record naming that identifier unreadable, and would reintroduce by the
        // back door the one thing the split centre field exists to prevent.
        if (target.id == layout.center.id)
            return ContractStatus::target_set_invalid;
        for (std::size_t other = 0; other < i; ++other)
            if (layout.surrounding[other].id == target.id)
                return ContractStatus::target_set_invalid;
    }
    // Two targets at the same coordinates under different identifiers are not
    // rejected: nothing in the contract says a layout may not stack them, and a
    // paradigm that wants them distinct can say so itself.
    for (std::size_t i = layout.count; i < kMaxSurroundingTargets; ++i)
        if (!placement_is_unset(layout.surrounding[i]))
            return ContractStatus::target_set_invalid;
    return ContractStatus::ok;
}

ContractStatus validate(const AcceptanceRegion& region) noexcept
{
    if (const ContractStatus status = check_finite(region.half_extent_x);
        status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = check_finite(region.half_extent_y);
        status != ContractStatus::ok)
        return status;
    // A region of zero width accepts nothing, not even a point cursor sitting
    // exactly on the target, once the cursor has any extent at all. Requiring it
    // to be positive keeps "the region exists" from depending on the cursor.
    if (region.half_extent_x <= 0.0 || region.half_extent_y <= 0.0)
        return ContractStatus::parameter_out_of_range;
    return ContractStatus::ok;
}

ContractStatus validate(const CursorGeometry& cursor) noexcept
{
    if (const ContractStatus status = check_finite(cursor.extent); status != ContractStatus::ok)
        return status;
    if (cursor.extent < 0.0)
        return ContractStatus::parameter_out_of_range;
    return ContractStatus::ok;
}

ContractStatus validate(const RadialLayoutRequest& request) noexcept
{
    if (const ContractStatus status = check_finite(request.radius); status != ContractStatus::ok)
        return status;
    if (request.radius <= 0.0)
        return ContractStatus::parameter_out_of_range;
    if (request.count == 0 || request.count > kMaxSurroundingTargets)
        return ContractStatus::target_set_invalid;
    if (request.center_id == kUnsetTargetId)
        return ContractStatus::identity_missing;

    for (std::size_t i = 0; i < request.count; ++i)
    {
        const TargetId id = request.ids[i];
        if (id == kUnsetTargetId)
            return ContractStatus::identity_missing;
        if (id == request.center_id)
            return ContractStatus::target_set_invalid;
        for (std::size_t other = 0; other < i; ++other)
            if (request.ids[other] == id)
                return ContractStatus::target_set_invalid;
    }
    for (std::size_t i = request.count; i < kMaxSurroundingTargets; ++i)
        if (request.ids[i] != kUnsetTargetId || request.spokes[i] != 0)
            return ContractStatus::target_set_invalid;
    return ContractStatus::ok;
}

ContractStatus validate(const CenterOut2DConfig& config) noexcept
{
    // Enumerated fields first, as everywhere else in the contract: a validator
    // that accepts a number naming no enumerator tells a reader that a record it
    // cannot interpret is fine.
    if (!geometry_unit_declared(config.geometry_unit))
        return ContractStatus::enum_undeclared;
    if (config.geometry_unit == GeometryUnit::unspecified)
        return ContractStatus::identity_missing;
    if (!target_selection_policy_declared(config.selection))
        return ContractStatus::enum_undeclared;
    if (config.selection == TargetSelectionPolicy::unspecified)
        return ContractStatus::identity_missing;

    if (const ContractStatus status = validate(config.layout); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(config.acceptance); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(config.cursor); status != ContractStatus::ok)
        return status;

    // Legal field by field, impossible together: no cursor position satisfies
    // contains_cursor() when the cursor is wider than the region, so every trial
    // of such a session would time out.
    if (config.cursor.extent > config.acceptance.half_extent_x ||
        config.cursor.extent > config.acceptance.half_extent_y)
        return ContractStatus::parameter_out_of_range;
    // A leg with no time cannot be completed either.
    if (config.movement_timeout.to_center == 0 || config.movement_timeout.to_out == 0)
        return ContractStatus::parameter_out_of_range;
    // The hold duration is deliberately not checked against the timeouts: how
    // the two compose is task semantics, and this header does not own those.

    if (config.sampler_version == 0)
        return ContractStatus::identity_missing;
    return ContractStatus::ok;
}

std::uint64_t layout_fingerprint(const CenterOut2DLayout& layout) noexcept
{
    FingerprintAccumulator accumulator;
    absorb_placement(accumulator, layout.center);
    accumulator.absorb(layout.count);
    for (std::size_t i = 0; i < layout.count; ++i)
        absorb_placement(accumulator, layout.surrounding[i]);
    return accumulator.value();
}

std::uint64_t configuration_fingerprint(const CenterOut2DConfig& config) noexcept
{
    FingerprintAccumulator accumulator;
    accumulator.absorb(static_cast<std::uint64_t>(config.geometry_unit));
    accumulator.absorb(layout_fingerprint(config.layout));
    absorb_double(accumulator, config.acceptance.half_extent_x);
    absorb_double(accumulator, config.acceptance.half_extent_y);
    absorb_double(accumulator, config.cursor.extent);
    absorb_durations(accumulator, config.movement_timeout);
    accumulator.absorb(config.hold_ns);
    absorb_durations(accumulator, config.reward_dwell);
    absorb_durations(accumulator, config.punish_dwell);
    accumulator.absorb(static_cast<std::uint64_t>(config.selection));
    accumulator.absorb(config.seed);
    accumulator.absorb(config.sampler_version);
    accumulator.absorb(config.trial_limit);
    return accumulator.value();
}

// State machine.
namespace
{

// The trial the current records belong to. The outward target rides along
// because a Center-Out record read back without it says which trial it came
// from and not what that trial asked for.
[[nodiscard]] TrialIdentity identity_of(TrialOrdinal ordinal, TargetId outward) noexcept
{
    TrialIdentity identity{};
    identity.ordinal = ordinal;
    identity.target_id = outward;
    return identity;
}

} // namespace

ContractStatus validate(const CenterOutTrial& trial) noexcept
{
    if (const ContractStatus status = validate(trial.record); status != ContractStatus::ok)
        return status;
    if (!center_out_phase_declared(trial.decided_phase))
        return ContractStatus::enum_undeclared;
    if (trial.outward_target == kUnsetTargetId)
        return ContractStatus::identity_missing;
    if (trial.record.trial.target_id != trial.outward_target)
        return ContractStatus::outcome_invalid;
    // No layout holds an index at or past the surrounding capacity, so a trial
    // carrying one cannot be a valid decided Center-Out trial even though the
    // number is a legal uint8. The validator has no configuration, so it cannot
    // check that the index names the outward target -- but the contract bound is
    // enough to keep "valid CenterOutTrial" from describing a trial no layout
    // can reproduce.
    if (trial.outward_idx >= kMaxSurroundingTargets)
        return ContractStatus::outcome_invalid;
    // A decided trial is exactly one of these three combinations; anything else
    // -- a pending, failed, or aborted outcome, or a phase and reason that do
    // not match it -- is outcome_invalid. The shared record validator already
    // reported an undeclared outcome as enum_undeclared.
    const bool success =
        trial.record.outcome == TrialOutcome::success &&
        trial.decided_phase == CenterOutPhase::to_out &&
        trial.record.reason == static_cast<std::uint32_t>(CenterOutReason::outward_acquired);
    const bool center_timeout =
        trial.record.outcome == TrialOutcome::timeout &&
        trial.decided_phase == CenterOutPhase::to_center &&
        trial.record.reason == static_cast<std::uint32_t>(CenterOutReason::center_movement_timeout);
    const bool outward_timeout =
        trial.record.outcome == TrialOutcome::timeout &&
        trial.decided_phase == CenterOutPhase::to_out &&
        trial.record.reason ==
            static_cast<std::uint32_t>(CenterOutReason::outward_movement_timeout);
    if (!(success || center_timeout || outward_timeout))
        return ContractStatus::outcome_invalid;
    // The two acquisition durations must be consistent with the trial interval
    // and with the leg that decided the trial. A hand-built trial whose
    // interval runs one nanosecond yet claims a 201 ns time to target would
    // otherwise enter the mean, describing a trial that never happened; this is
    // a combination of fields rather than one out-of-range parameter, so it is
    // outcome_invalid. These are the strongest checks that do not need the
    // configuration: the machine could only ever have produced a trial that
    // satisfies them, because every movement window is strictly positive
    // (validate(const CenterOut2DConfig&) rejects a zero timeout) and the
    // outward leg cannot begin before the centre is acquired.
    const DurationNs duration = trial.record.interval.duration_ns();
    if (success)
    {
        // Each duration is bounded by the trial it sits in, and the two cannot
        // overlap: the outward leg starts no earlier than centre acquisition, so
        // the trial must be at least as long as their sum. A centre reward dwell
        // only makes it longer. Both may be zero (a zero hold, an overlapping
        // geometry), so only upper bounds -- not positive lower bounds -- are
        // enforced, including a zero-duration success when both legs complete at
        // the same instant. The sum is checked by subtraction so a pair of large
        // durations cannot overflow the addition.
        if (trial.center_acquire_ns > duration || trial.outward_acquire_ns > duration)
            return ContractStatus::outcome_invalid;
        if (trial.outward_acquire_ns > duration - trial.center_acquire_ns)
            return ContractStatus::outcome_invalid;
    }
    else if (center_timeout)
    {
        // The centre never acquired, so neither duration is meaningful. A centre
        // timeout also cannot be a zero-length trial: the centre movement window
        // is strictly positive, so the machine always runs it for some time
        // before it times out.
        if (duration == 0 || trial.center_acquire_ns != 0 || trial.outward_acquire_ns != 0)
            return ContractStatus::outcome_invalid;
    }
    else // outward_timeout
    {
        // The centre was acquired and the outward leg was not. The outward
        // movement window is strictly positive and begins no earlier than centre
        // acquisition, so the trial must end strictly after the centre was
        // acquired -- a centre acquisition that fills the whole trial leaves no
        // room for a positive outward window to time out in.
        if (duration == 0 || trial.center_acquire_ns >= duration || trial.outward_acquire_ns != 0)
            return ContractStatus::outcome_invalid;
    }
    return ContractStatus::ok;
}

ContractStatus summarize(std::span<const CenterOutTrial> trials,
                         CenterOutStatistics& statistics) noexcept
{
    CenterOutStatistics summary{};
    for (const CenterOutTrial& trial : trials)
    {
        if (const ContractStatus status = validate(trial); status != ContractStatus::ok)
            return status;
        ++summary.decided;
        if (trial.record.outcome == TrialOutcome::success)
        {
            ++summary.successes;
            if (!time_fits(summary.total_time_to_target_ns, trial.outward_acquire_ns))
                return ContractStatus::duration_overflow;
            summary.total_time_to_target_ns += trial.outward_acquire_ns;
        }
        else if (trial.decided_phase == CenterOutPhase::to_center)
            ++summary.center_timeouts;
        else
            ++summary.outward_timeouts;
    }
    // Report the truncated integer mean in nanoseconds.
    summary.mean_time_to_target_ns =
        summary.successes == 0 ? 0 : summary.total_time_to_target_ns / summary.successes;
    statistics = summary;
    return ContractStatus::ok;
}

ContractStatus CenterOutMachine::emit_transition(Run& run, CenterOutStepResult& result,
                                                 ExperimentTimeNs time_ns, CenterOutState to,
                                                 CenterOutCause cause) const noexcept
{
    // Unreachable while the chain stops at one decided trial: the longest run is
    // six. Checked anyway, because a future state that lengthened the chain must
    // fail loudly rather than write past the array.
    if (result.n_transitions >= kMaxStepTransitions)
        return ContractStatus::numerical_failure;
    SequenceOrdinal sequence = 0;
    if (const ContractStatus status = run.sequence.issue(sequence); status != ContractStatus::ok)
        return status;

    StateTransition transition{};
    transition.time_ns = time_ns;
    transition.sequence = sequence;
    transition.trial = identity_of(run.ordinal, run.outward_target);
    transition.paradigm = paradigm_;
    transition.from_state = static_cast<StateId>(run.state);
    transition.to_state = static_cast<StateId>(to);
    transition.cause = static_cast<std::uint32_t>(cause);
    result.transitions[result.n_transitions++] = transition;
    run.state = to;
    return ContractStatus::ok;
}

ContractStatus CenterOutMachine::emit_event(Run& run, CenterOutStepResult& result,
                                            ExperimentTimeNs time_ns,
                                            ExperimentEventKind kind) const noexcept
{
    if (result.n_events >= kMaxStepEvents)
        return ContractStatus::numerical_failure;
    SequenceOrdinal sequence = 0;
    if (const ContractStatus status = run.sequence.issue(sequence); status != ContractStatus::ok)
        return status;

    ExperimentEvent event{};
    event.time_ns = time_ns;
    event.sequence = sequence;
    event.trial = identity_of(run.ordinal, run.outward_target);
    event.kind = kind;
    event.paradigm = paradigm_;
    result.events[result.n_events++] = event;
    return ContractStatus::ok;
}

ContractStatus CenterOutMachine::begin_trial(Run& run, CenterOutStepResult& result,
                                             ExperimentTimeNs time_ns) const noexcept
{
    TrialOrdinal ordinal = 0;
    if (const ContractStatus status = run.trials.issue(ordinal); status != ContractStatus::ok)
        return status;

    // The outward target comes from the schedule and from nowhere else. It is
    // chosen when the trial begins rather than when its outward leg does, so
    // the whole trial knows what it is for; the success count cannot change
    // in between, so the answer is the same either way.
    std::uint8_t idx = 0;
    if (const ContractStatus status = select_outward_target(config_, ordinal, run.successes, idx);
        status != ContractStatus::ok)
        return status;

    TimeInterval leg{};
    if (const ContractStatus status =
            interval_from_duration(time_ns, config_.movement_timeout.to_center, leg);
        status != ContractStatus::ok)
        return status;

    run.ordinal = ordinal;
    run.outward_idx = idx;
    run.outward_target = config_.layout.surrounding[idx].id;
    run.trial_started_ns = time_ns;
    run.center_acquired = false;
    run.center_acquired_ns = 0;
    run.contained = false;
    run.leg = leg;
    run.hold = TimeInterval{};
    run.dwell = TimeInterval{};
    return emit_event(run, result, time_ns, ExperimentEventKind::trial_start);
}

ContractStatus CenterOutMachine::decide_trial(Run& run, CenterOutStepResult& result,
                                              TrialOutcome outcome, CenterOutReason reason,
                                              CenterOutPhase phase,
                                              ExperimentTimeNs time_ns) const noexcept
{
    const DurationNs dwell_ns = outcome == TrialOutcome::success
                                    ? phase_duration(config_.reward_dwell, phase)
                                    : phase_duration(config_.punish_dwell, phase);
    TimeInterval dwell{};
    if (const ContractStatus status = interval_from_duration(time_ns, dwell_ns, dwell);
        status != ContractStatus::ok)
        return status;

    CenterOutTrial trial{};
    trial.record.trial = identity_of(run.ordinal, run.outward_target);
    trial.record.interval = TimeInterval{run.trial_started_ns, time_ns};
    trial.record.paradigm = paradigm_;
    trial.record.outcome = outcome;
    trial.record.reason = static_cast<std::uint32_t>(reason);
    trial.outward_target = run.outward_target;
    trial.outward_idx = run.outward_idx;
    trial.decided_phase = phase;
    // Which of these is meaningful follows from the outcome and the deciding
    // leg, which is why neither carries a validity flag of its own.
    trial.center_acquire_ns =
        run.center_acquired ? run.center_acquired_ns - run.trial_started_ns : DurationNs{0};
    // `run.leg` is still the outward leg here, so its start is where the time to
    // target is measured from.
    trial.outward_acquire_ns =
        outcome == TrialOutcome::success ? time_ns - run.leg.start_ns : DurationNs{0};

    run.dwell = dwell;
    run.hold = TimeInterval{};
    run.contained = false;
    ++run.completed;
    if (outcome == TrialOutcome::success)
        ++run.successes;

    result.trial = trial;
    result.trial_decided = true;
    return emit_event(run, result, time_ns, ExperimentEventKind::trial_stop);
}

ContractStatus CenterOutMachine::advance(Run& run, ExperimentTimeNs time_ns,
                                         const WorkspacePoint& cursor,
                                         CenterOutStepResult& result) const noexcept
{
    for (std::size_t guard = 0; guard <= kMaxStepTransitions; ++guard)
    {
        const CenterOutState state = run.state;
        const CenterOutPhase phase = center_out_phase_of(state);

        if (center_out_state_is_leg(state))
        {
            const TargetPlacement& placement = phase == CenterOutPhase::to_center
                                                   ? config_.layout.center
                                                   : config_.layout.surrounding[run.outward_idx];
            const bool holding =
                state == CenterOutState::hold_center || state == CenterOutState::hold_out;

            // A hold that completed strictly before the movement window ended
            // wins outright: it happened first, whatever this observation says
            // and however late this observation is. The half-open convention is
            // what makes "strictly before" the right comparison -- at the
            // window's end instant the window is already over.
            if (holding && run.hold.elapsed_at(time_ns) && run.hold.end_ns < run.leg.end_ns)
            {
                const ExperimentTimeNs acquired_ns = run.hold.end_ns;
                if (phase == CenterOutPhase::to_center)
                {
                    TimeInterval dwell{};
                    if (const ContractStatus status = interval_from_duration(
                            acquired_ns, config_.reward_dwell.to_center, dwell);
                        status != ContractStatus::ok)
                        return status;
                    run.center_acquired = true;
                    run.center_acquired_ns = acquired_ns;
                    run.dwell = dwell;
                    run.hold = TimeInterval{};
                    run.contained = false;
                    if (const ContractStatus status = emit_transition(
                            run, result, acquired_ns, CenterOutState::center_success_dwell,
                            CenterOutCause::hold_completed);
                        status != ContractStatus::ok)
                        return status;
                    continue;
                }
                if (const ContractStatus status =
                        decide_trial(run, result, TrialOutcome::success,
                                     CenterOutReason::outward_acquired, phase, acquired_ns);
                    status != ContractStatus::ok)
                    return status;
                if (const ContractStatus status =
                        emit_transition(run, result, acquired_ns, CenterOutState::out_success_dwell,
                                        CenterOutCause::hold_completed);
                    status != ContractStatus::ok)
                    return status;
                result.settled = !run.dwell.elapsed_at(time_ns);
                return ContractStatus::ok;
            }

            // The movement window ended. Checked before this observation is
            // consulted, because the window's end instant is at or before it.
            if (run.leg.elapsed_at(time_ns))
            {
                const ExperimentTimeNs failed_ns = run.leg.end_ns;
                const CenterOutReason reason = phase == CenterOutPhase::to_center
                                                   ? CenterOutReason::center_movement_timeout
                                                   : CenterOutReason::outward_movement_timeout;
                if (const ContractStatus status =
                        decide_trial(run, result, TrialOutcome::timeout, reason, phase, failed_ns);
                    status != ContractStatus::ok)
                    return status;
                const CenterOutState dwell_state = phase == CenterOutPhase::to_center
                                                       ? CenterOutState::center_failure_dwell
                                                       : CenterOutState::out_failure_dwell;
                if (const ContractStatus status = emit_transition(
                        run, result, failed_ns, dwell_state, CenterOutCause::movement_timed_out);
                    status != ContractStatus::ok)
                    return status;
                result.settled = !run.dwell.elapsed_at(time_ns);
                return ContractStatus::ok;
            }

            bool inside = false;
            if (const ContractStatus status = contains_cursor(config_.acceptance, config_.cursor,
                                                              placement.pos, cursor, inside);
                status != ContractStatus::ok)
                return status;
            run.contained = inside;

            if (holding && !inside)
            {
                // The hold is discarded, not paused. A later containment starts
                // a new one from the instant it is observed; the movement window
                // is untouched, because leaving the target does not buy time.
                run.hold = TimeInterval{};
                const CenterOutState move_state = phase == CenterOutPhase::to_center
                                                      ? CenterOutState::move_to_center
                                                      : CenterOutState::move_to_out;
                if (const ContractStatus status = emit_transition(run, result, time_ns, move_state,
                                                                  CenterOutCause::containment_lost);
                    status != ContractStatus::ok)
                    return status;
                continue;
            }
            if (!holding && inside)
            {
                TimeInterval hold{};
                if (const ContractStatus status =
                        interval_from_duration(time_ns, config_.hold_ns, hold);
                    status != ContractStatus::ok)
                    return status;
                run.hold = hold;
                const CenterOutState hold_state = phase == CenterOutPhase::to_center
                                                      ? CenterOutState::hold_center
                                                      : CenterOutState::hold_out;
                if (const ContractStatus status = emit_transition(
                        run, result, time_ns, hold_state, CenterOutCause::containment_gained);
                    status != ContractStatus::ok)
                    return status;
                continue;
            }

            result.settled = true;
            return ContractStatus::ok;
        }

        if (center_out_state_is_dwell(state))
        {
            if (!run.dwell.elapsed_at(time_ns))
            {
                result.settled = true;
                return ContractStatus::ok;
            }
            const ExperimentTimeNs resumed_ns = run.dwell.end_ns;

            if (state == CenterOutState::center_success_dwell)
            {
                TimeInterval leg{};
                if (const ContractStatus status =
                        interval_from_duration(resumed_ns, config_.movement_timeout.to_out, leg);
                    status != ContractStatus::ok)
                    return status;
                run.leg = leg;
                run.hold = TimeInterval{};
                run.dwell = TimeInterval{};
                run.contained = false;
                if (const ContractStatus status =
                        emit_transition(run, result, resumed_ns, CenterOutState::move_to_out,
                                        CenterOutCause::dwell_elapsed);
                    status != ContractStatus::ok)
                    return status;
                continue;
            }

            // The three trial-ending dwells. The dwell runs whether or not the
            // session is over: it is the interval the subject experiences after
            // an outcome, and cutting it short on the last trial would make the
            // last trial a different trial.
            run.dwell = TimeInterval{};
            if (trial_limit_reached(config_, run.completed))
            {
                if (const ContractStatus status =
                        emit_event(run, result, resumed_ns, ExperimentEventKind::session_stop);
                    status != ContractStatus::ok)
                    return status;
                if (const ContractStatus status =
                        emit_transition(run, result, resumed_ns, CenterOutState::complete,
                                        CenterOutCause::trial_limit_reached);
                    status != ContractStatus::ok)
                    return status;
                result.settled = true;
                return ContractStatus::ok;
            }
            if (const ContractStatus status = begin_trial(run, result, resumed_ns);
                status != ContractStatus::ok)
                return status;
            if (const ContractStatus status =
                    emit_transition(run, result, resumed_ns, CenterOutState::move_to_center,
                                    CenterOutCause::dwell_elapsed);
                status != ContractStatus::ok)
                return status;
            continue;
        }

        // ::idle and ::complete are refused by step() before the chain runs.
        result.settled = true;
        return ContractStatus::ok;
    }
    return ContractStatus::numerical_failure;
}

CenterOutSnapshot CenterOutMachine::build_snapshot(const Run& run) const noexcept
{
    CenterOutSnapshot snapshot{};
    snapshot.time_ns = run.gate.last_ns();
    snapshot.state = run.state;
    snapshot.phase = center_out_phase_of(run.state);
    snapshot.trial = identity_of(run.ordinal, run.outward_target);
    snapshot.outward_target = run.outward_target;
    snapshot.outward_idx = run.outward_idx;
    snapshot.contained = run.contained;
    snapshot.leg = run.leg;
    snapshot.hold = run.hold;
    snapshot.dwell = run.dwell;
    snapshot.completed = run.completed;
    snapshot.successes = run.successes;
    if (center_out_state_is_leg(run.state))
    {
        const TargetPlacement& placement = snapshot.phase == CenterOutPhase::to_center
                                               ? config_.layout.center
                                               : config_.layout.surrounding[run.outward_idx];
        snapshot.active_target = placement.id;
        snapshot.active_position = placement.pos;
    }
    return snapshot;
}

ContractStatus CenterOutMachine::start(ParadigmId paradigm, const CenterOut2DConfig& config,
                                       ExperimentTimeNs time_ns,
                                       CenterOutStepResult& result) noexcept
{
    // A session already occupies this machine. Starting again would silently
    // replace it -- dropping the running trial, clearing the ordinals, and
    // emitting a fresh session_start -- so a new session needs reset() first.
    // The lifecycle is frozen as idle -> start -> running -> ... -> complete,
    // and only idle -> start.
    if (run_.state != CenterOutState::idle)
        return ContractStatus::already_running;
    if (paradigm == kUnsetParadigmId)
        return ContractStatus::identity_missing;
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;

    // Built whole and committed only on success, so a refused start leaves a
    // machine that was already running exactly as it was.
    CenterOutMachine started{};
    started.paradigm_ = paradigm;
    started.config_ = config;
    Run& run = started.run_;
    if (const ContractStatus status = run.gate.accept(time_ns); status != ContractStatus::ok)
        return status;

    CenterOutStepResult local{};
    if (const ContractStatus status =
            started.emit_event(run, local, time_ns, ExperimentEventKind::session_start);
        status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = started.begin_trial(run, local, time_ns);
        status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = started.emit_transition(
            run, local, time_ns, CenterOutState::move_to_center, CenterOutCause::session_started);
        status != ContractStatus::ok)
        return status;

    local.snapshot = started.build_snapshot(run);
    local.settled = true;
    *this = started;
    result = local;
    return ContractStatus::ok;
}

void CenterOutMachine::reset() noexcept
{
    *this = CenterOutMachine{};
}

ContractStatus CenterOutMachine::step(ExperimentTimeNs time_ns, const WorkspacePoint& cursor,
                                      CenterOutStepResult& result) noexcept
{
    if (run_.state == CenterOutState::idle || run_.state == CenterOutState::complete)
        return ContractStatus::not_running;

    Run run = run_;
    if (const ContractStatus status = run.gate.accept(time_ns); status != ContractStatus::ok)
        return status;

    CenterOutStepResult local{};
    if (const ContractStatus status = advance(run, time_ns, cursor, local);
        status != ContractStatus::ok)
        return status;

    local.snapshot = build_snapshot(run);
    run_ = run;
    result = local;
    return ContractStatus::ok;
}

CenterOutSnapshot CenterOutMachine::snapshot() const noexcept
{
    return build_snapshot(run_);
}
} // namespace neurale::experiments::center_out
