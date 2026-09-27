/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/experiments/abnormal.h>

namespace neurale::experiments
{
namespace
{
/// How severe a response is, on the same scale as AbnormalPolicy.
///
/// A response is admissible under a policy when it is no more severe than what
/// that policy authorises. Refusing an input is not on the scale: a paradigm
/// refuses a value it cannot apply whatever the policy says, because applying
/// it is not an option the configuration gets to choose.
[[nodiscard]] AbnormalPolicy severity_of(AbnormalResponse response) noexcept
{
    switch (response)
    {
    case AbnormalResponse::recorded:
    case AbnormalResponse::input_refused:
        return AbnormalPolicy::record;
    case AbnormalResponse::trial_invalidated:
    case AbnormalResponse::trial_aborted:
        return AbnormalPolicy::abort_trial;
    case AbnormalResponse::session_aborted:
        return AbnormalPolicy::abort_session;
    }
    return AbnormalPolicy::abort_session;
}

[[nodiscard]] bool names_a_trial(AbnormalResponse response) noexcept
{
    return response == AbnormalResponse::trial_invalidated ||
           response == AbnormalResponse::trial_aborted;
}
} // namespace

AbnormalPolicy policy_for(const AbnormalPolicySet& policies, AbnormalCondition condition) noexcept
{
    switch (condition)
    {
    case AbnormalCondition::decoded_command_invalid:
    case AbnormalCondition::input_schema_mismatch:
        return policies.decoded_command_invalid;
    case AbnormalCondition::input_stale:
    case AbnormalCondition::source_discontinuity:
    case AbnormalCondition::input_gap:
    case AbnormalCondition::deadline_missed:
        return policies.input_discontinuity;
    case AbnormalCondition::presentation_failed:
        return policies.presentation_failed;
    case AbnormalCondition::presentation_evidence_missing:
        return policies.presentation_evidence_missing;
    case AbnormalCondition::recorder_fault:
    case AbnormalCondition::actuator_fault:
    case AbnormalCondition::runtime_fault:
        return policies.acquisition_fault;
    case AbnormalCondition::observer_frame_drop:
    case AbnormalCondition::input_after_terminal:
    case AbnormalCondition::presentation_report_unmatched:
        // None of them consults the set. The first did not touch the task's
        // input, the second arrived at a run that had already ended -- there is
        // nothing left for a severity to end -- and the third is a report the
        // run cannot attribute to any request it made, so there is no trial it
        // is entitled to speak for.
        return AbnormalPolicy::record;
    case AbnormalCondition::emergency_stop:
        return AbnormalPolicy::abort_session;
    case AbnormalCondition::unspecified:
        break;
    }
    // An unset or undeclared condition is the one case where the conservative
    // answer is the severe one: nothing knows what it was.
    return AbnormalPolicy::abort_session;
}

AbnormalResponse abnormal_response_for(AbnormalPolicy policy, bool has_trial, bool can_end_trial,
                                       bool input_refused) noexcept
{
    if (policy == AbnormalPolicy::abort_session)
    {
        return AbnormalResponse::session_aborted;
    }
    if (policy == AbnormalPolicy::abort_trial && has_trial)
    {
        // A paradigm that cannot end the trial in flight says so, and the
        // record says so too. Writing `trial_aborted` for a trial that actually
        // ran to its normal end would be claiming a termination that never
        // happened.
        return can_end_trial ? AbnormalResponse::trial_aborted
                             : AbnormalResponse::trial_invalidated;
    }
    // Refusing an input is not on the severity scale: a paradigm refuses a
    // value it cannot apply whatever the policy says, so this is what is left
    // once the two ending severities have had their say.
    return input_refused ? AbnormalResponse::input_refused : AbnormalResponse::recorded;
}

const char* abnormal_condition_name(AbnormalCondition condition) noexcept
{
    switch (condition)
    {
    case AbnormalCondition::unspecified:
        return "unspecified";
    case AbnormalCondition::decoded_command_invalid:
        return "decoded-command-invalid";
    case AbnormalCondition::input_schema_mismatch:
        return "input-schema-mismatch";
    case AbnormalCondition::input_stale:
        return "input-stale";
    case AbnormalCondition::source_discontinuity:
        return "source-discontinuity";
    case AbnormalCondition::input_gap:
        return "input-gap";
    case AbnormalCondition::deadline_missed:
        return "deadline-missed";
    case AbnormalCondition::observer_frame_drop:
        return "observer-frame-drop";
    case AbnormalCondition::recorder_fault:
        return "recorder-fault";
    case AbnormalCondition::actuator_fault:
        return "actuator-fault";
    case AbnormalCondition::runtime_fault:
        return "runtime-fault";
    case AbnormalCondition::presentation_failed:
        return "presentation-failed";
    case AbnormalCondition::presentation_evidence_missing:
        return "presentation-evidence-missing";
    case AbnormalCondition::input_after_terminal:
        return "input-after-terminal";
    case AbnormalCondition::emergency_stop:
        return "emergency-stop";
    case AbnormalCondition::presentation_report_unmatched:
        return "presentation-report-unmatched";
    }
    return "undeclared";
}

ContractStatus validate(const AbnormalPolicySet& policies) noexcept
{
    if (!abnormal_policy_declared(policies.decoded_command_invalid) ||
        !abnormal_policy_declared(policies.input_discontinuity) ||
        !abnormal_policy_declared(policies.presentation_failed) ||
        !abnormal_policy_declared(policies.presentation_evidence_missing) ||
        !abnormal_policy_declared(policies.acquisition_fault))
        return ContractStatus::enum_undeclared;
    return ContractStatus::ok;
}

ContractStatus validate(const AbnormalEvent& event) noexcept
{
    if (!abnormal_condition_declared(event.condition) || !abnormal_policy_declared(event.policy) ||
        !abnormal_response_declared(event.response))
        return ContractStatus::enum_undeclared;
    if (event.paradigm == kUnsetParadigmId)
        return ContractStatus::identity_missing;
    if (event.condition == AbnormalCondition::unspecified)
        // Every other field can be read; this one is the record saying it does
        // not know what it is about, which is not a record.
        return ContractStatus::identity_missing;
    if (names_a_trial(event.response) && !event.has_trial)
        // A response that says a trial was ended or invalidated, on a record
        // that carries no trial, names nothing. A reader cannot act on it and
        // cannot tell which trial to distrust.
        return ContractStatus::outcome_invalid;
    if (static_cast<std::uint8_t>(severity_of(event.response)) >
        static_cast<std::uint8_t>(event.policy))
        // The run did more than the policy beside it authorises. Either the
        // policy recorded is not the one that was in force, or the response is
        // not the one that happened; both make the record evidence of nothing.
        return ContractStatus::outcome_invalid;
    return ContractStatus::ok;
}

} // namespace neurale::experiments
