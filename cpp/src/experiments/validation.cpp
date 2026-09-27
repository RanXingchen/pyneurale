/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/experiments/command.h>
#include <neurale/experiments/events.h>
#include <neurale/experiments/presentation.h>
#include <neurale/experiments/replay.h>

#include <cmath>
#include <cstddef>

namespace neurale::experiments
{
namespace
{

// A record that does not say which paradigm produced it cannot be read back
// against that paradigm's own state, phase, and reason enumerations, so every
// runtime record requires one.
[[nodiscard]] ContractStatus require_paradigm(ParadigmId paradigm) noexcept
{
    return paradigm == kUnsetParadigmId ? ContractStatus::identity_missing : ContractStatus::ok;
}

// Expiry is optional; when present it must leave a nonempty window.
[[nodiscard]] ContractStatus check_expiry(ExperimentTimeNs from_ns,
                                          ExperimentTimeNs valid_until_ns) noexcept
{
    if (valid_until_ns == kNoExpiryNs)
        return ContractStatus::ok;
    return valid_until_ns > from_ns ? ContractStatus::ok : ContractStatus::expiry_before_generation;
}

// A cue that stands for content needs an identifier to resolve; a cue that does
// not must not carry one, so that "which stimulus is showing" has one answer.
[[nodiscard]] ContractStatus check_cue_stimulus(CueKind cue, StimulusId stimulus_id) noexcept
{
    if (!cue_kind_declared(cue))
        return ContractStatus::enum_undeclared;
    if (cue == CueKind::text_content || cue == CueKind::ssvep_targets)
        return stimulus_id == kUnsetStimulusId ? ContractStatus::identity_missing
                                               : ContractStatus::ok;
    return stimulus_id == kUnsetStimulusId ? ContractStatus::ok
                                           : ContractStatus::presentation_invalid;
}

} // namespace

ContractStatus validate(const ExperimentEvent& event) noexcept
{
    // The enumerated field is checked before anything else in every validator
    // below. These types are persisted and replay-visible, so a validator that
    // accepts a number naming no enumerator is telling a reader that a record it
    // cannot interpret is fine.
    if (!experiment_event_kind_declared(event.kind))
        return ContractStatus::enum_undeclared;
    if (event.kind == ExperimentEventKind::unspecified)
        return ContractStatus::identity_missing;
    return require_paradigm(event.paradigm);
}

ContractStatus validate(const StateTransition& transition) noexcept
{
    // A self-transition is not rejected: re-entering a state is a decision some
    // paradigms take, and the contract has no basis to call it wrong.
    return require_paradigm(transition.paradigm);
}

ContractStatus validate(const SelectionEvent& selection) noexcept
{
    if (!selection_kind_declared(selection.kind))
        return ContractStatus::enum_undeclared;
    if (const ContractStatus status = require_paradigm(selection.paradigm);
        status != ContractStatus::ok)
        return status;
    if (selection.intended_id == kUnsetTargetId)
        return ContractStatus::identity_missing;
    const bool dwell = selection.kind == SelectionKind::dwell;
    if (dwell != (selection.dwell_ns > 0))
        return ContractStatus::outcome_invalid;
    if (selection.correct != (selection.selected_id == selection.intended_id))
        return ContractStatus::outcome_invalid;
    if (selection.correct && selection.selected_id == kUnsetTargetId)
        return ContractStatus::outcome_invalid;
    return ContractStatus::ok;
}

ContractStatus validate(const TrialRecord& record) noexcept
{
    if (!trial_outcome_declared(record.outcome))
        return ContractStatus::enum_undeclared;
    if (const ContractStatus status = require_paradigm(record.paradigm);
        status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(record.interval); status != ContractStatus::ok)
        return status;
    // A pending trial has no end yet, and an empty interval is how it says so.
    if (!trial_ended(record.outcome) && !record.interval.empty())
        return ContractStatus::outcome_invalid;
    return ContractStatus::ok;
}

ContractStatus validate(const PresentationRequest& request) noexcept
{
    if (const ContractStatus status = require_paradigm(request.paradigm);
        status != ContractStatus::ok)
        return status;
    if (request.onset_ns < request.requested_ns)
        return ContractStatus::time_regressed;
    if (!time_fits(request.onset_ns, request.duration_ns))
        return ContractStatus::duration_overflow;
    if (const ContractStatus status = check_expiry(request.requested_ns, request.valid_until_ns);
        status != ContractStatus::ok)
        return status;
    if (request.valid_until_ns != kNoExpiryNs && request.valid_until_ns < request.onset_ns)
        return ContractStatus::expiry_before_generation;
    return check_cue_stimulus(request.cue, request.stimulus_id);
}

ContractStatus validate(const PresentationState& state) noexcept
{
    if (const ContractStatus status = require_paradigm(state.paradigm);
        status != ContractStatus::ok)
        return status;
    return check_cue_stimulus(state.cue, state.stimulus_id);
}

ContractStatus validate(const PresentationOutcome& outcome) noexcept
{
    if (!presentation_status_declared(outcome.status))
        return ContractStatus::enum_undeclared;
    switch (outcome.status)
    {
    case PresentationStatus::presented:
        // Only a presenter can report this, and it cannot be earlier than the
        // request it reports on.
        return outcome.presented_ns < outcome.requested_ns ? ContractStatus::time_regressed
                                                           : ContractStatus::ok;
    case PresentationStatus::not_reported:
    case PresentationStatus::skipped:
    case PresentationStatus::expired:
        // No presentation reached a software observation point, so there is no
        // presenter-reported time to carry.
        return outcome.presented_ns == 0 ? ContractStatus::ok
                                         : ContractStatus::presentation_invalid;
    }
    // Unreachable: the domain check above already rejected everything the switch
    // does not name. Present so the function has a return on every path.
    return ContractStatus::enum_undeclared;
}

ContractStatus validate(const CommandSpace& space) noexcept
{
    if (space.id == kUnsetCommandSpaceId)
        return ContractStatus::identity_missing;
    if (space.dim == 0 || space.dim > kMaxCommandDim)
        return ContractStatus::dimension_invalid;
    if (!command_frame_declared(space.frame))
        return ContractStatus::enum_undeclared;
    if (space.frame == CommandFrame::unspecified)
        return ContractStatus::identity_missing;
    for (std::size_t axis = 0; axis < kMaxCommandDim; ++axis)
    {
        if (!command_axis_name_declared(space.axes[axis].name) ||
            !command_unit_declared(space.axes[axis].unit))
            return ContractStatus::enum_undeclared;
        const bool in_use = axis < space.dim;
        const bool described = space.axes[axis].name != CommandAxisName::unspecified &&
                               space.axes[axis].unit != CommandUnit::unspecified;
        const bool untouched = space.axes[axis].name == CommandAxisName::unspecified &&
                               space.axes[axis].unit == CommandUnit::unspecified;
        if (in_use && !described)
            return ContractStatus::identity_missing;
        if (!in_use && !untouched)
            return ContractStatus::dimension_invalid;
    }
    return ContractStatus::ok;
}

ContractStatus validate(const CommandRequest& request) noexcept
{
    if (request.space == kUnsetCommandSpaceId)
        return ContractStatus::identity_missing;
    if (request.dim == 0 || request.dim > kMaxCommandDim)
        return ContractStatus::dimension_invalid;
    if (const ContractStatus status = check_expiry(request.generated_ns, request.valid_until_ns);
        status != ContractStatus::ok)
        return status;
    for (std::size_t axis = 0; axis < kMaxCommandDim; ++axis)
    {
        if (axis < request.dim)
        {
            if (!std::isfinite(request.values[axis]))
                return ContractStatus::value_not_finite;
        }
        // An unused slot is required to be exactly zero: two commands that mean
        // the same thing then also have the same bytes, which is what lets a
        // record be compared and fingerprinted without its dimension.
        else if (request.values[axis] != 0.0)
        {
            return ContractStatus::dimension_invalid;
        }
    }
    return ContractStatus::ok;
}

ContractStatus validate_against(const CommandRequest& request, const CommandSpace& space) noexcept
{
    if (const ContractStatus status = validate(space); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(request); status != ContractStatus::ok)
        return status;
    if (request.space != space.id)
        return ContractStatus::identity_missing;
    if (request.dim != space.dim)
        return ContractStatus::dimension_invalid;
    return ContractStatus::ok;
}

ContractStatus validate(const CommandOutcome& outcome) noexcept
{
    if (!command_application_declared(outcome.application))
        return ContractStatus::enum_undeclared;
    if (outcome.submitted_ns < outcome.generated_ns)
        return ContractStatus::time_regressed;
    if (outcome.application != CommandApplication::not_submitted)
        return ContractStatus::ok;
    // Nothing was submitted, so there is no submission instant and no runtime
    // status to report.
    return outcome.submitted_ns == outcome.generated_ns && outcome.status_code == 0
               ? ContractStatus::ok
               : ContractStatus::outcome_invalid;
}

ContractStatus validate(const ExperimentSnapshot& snapshot) noexcept
{
    if (const ContractStatus status = require_paradigm(snapshot.paradigm);
        status != ContractStatus::ok)
        return status;
    return validate(snapshot.schedule);
}

ContractStatus validate(const DecisionSnapshot& snapshot) noexcept
{
    if (const ContractStatus status = require_paradigm(snapshot.paradigm);
        status != ContractStatus::ok)
        return status;
    return validate(snapshot.schedule);
}

} // namespace neurale::experiments
