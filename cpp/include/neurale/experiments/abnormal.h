/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

#include <neurale/experiments/contract.h>
#include <neurale/experiments/identity.h>

/**
 * @file
 * @brief What an experiment does when the run stops being normal.
 *
 * A paradigm meets conditions its task rules say nothing about: a decoded value
 * that is not a number, a break in the data it is measuring against, a
 * presenter that could not present, a recorder that stopped taking records. The
 * task state machines do not model any of them, and deliberately so -- they are
 * pure functions of explicit time and task input, and a machine that also
 * modelled acquisition failure would be a machine whose replay depended on how
 * the hardware behaved.
 *
 * This is the vocabulary for the layer above them. It says what was observed,
 * how severely the run treats that class of observation, and what actually
 * happened to the trial in flight. All three are recorded: an abnormal
 * condition that changed the run and left no record is the failure this
 * vocabulary exists to prevent.
 *
 * What it does **not** do is touch a device. Inhibiting and releasing an
 * actuator path belongs to the streaming runtime and its `SafetyController`,
 * which decide it from the runtime's own fault state; nothing here has a second
 * opinion about it, and a paradigm that wants an actuator stopped ends the run
 * and lets the runtime do what it already does. Experiment and trial semantics
 * after a fault are this vocabulary's; actuator state is not.
 */
namespace neurale::experiments
{

/// What abnormal condition was observed.
///
/// Append-only, like every other persisted enumeration in the contract:
/// existing values are never renumbered.
enum class AbnormalCondition : std::uint16_t
{
    /// Unset.
    unspecified = 0,
    /// A decoded value the paradigm may not apply, because it is not finite.
    ///
    /// Distinct from ::input_schema_mismatch, which reports a frame that is
    /// malformed rather than a number that is unusable: one is the decoder
    /// producing something a task cannot integrate, the other is the stream not
    /// being the stream the run prepared for, and they are not the same
    /// person's problem.
    decoded_command_invalid,
    /// A frame whose schema, shape, or ordering is not the one the run prepared
    /// against.
    input_schema_mismatch,
    /// More time passed since the last accepted input than the run will apply a
    /// command across.
    ///
    /// Not a regenerated command: it is the refusal to treat one decoded value
    /// as if it had been held for an interval nobody observed it over.
    input_stale,
    /// The source reported a break in the data the paradigm consumes.
    source_discontinuity,
    /// The paradigm's own task input carried a gap.
    ///
    /// Distinct from ::source_discontinuity because a paradigm may consume no
    /// stream at all and still lose its input -- a pointer that stopped
    /// arriving is a gap in the task's input without being a discontinuity in
    /// anything the runtime is carrying.
    input_gap,
    /// A deadline the runtime enforces was missed.
    deadline_missed,
    /// A drop on an observation or presentation path that carries no task
    /// input.
    ///
    /// Recorded and never anything more. A dropped monitoring or rendering
    /// frame did not change what the task was given, and treating it as an
    /// input gap would invalidate measurements that are perfectly good. If a
    /// drop *did* change the task's input then it is an ::input_gap, and the
    /// paradigm reports it as one.
    observer_frame_drop,
    /// The recorder could not take what the run produced.
    recorder_fault,
    /// The terminal actuator or consumer edge failed.
    actuator_fault,
    /// The runtime reported a fault of its own.
    runtime_fault,
    /// A presenter reports that a requested presentation did not happen.
    presentation_failed,
    /// A presentation the run's claims depend on has no reported evidence.
    ///
    /// Reported rather than resolved. The intended onset is not evidence of
    /// presentation, and a record that filled the missing instant with it would
    /// be manufacturing the measurement it was supposed to be reporting the
    /// absence of.
    presentation_evidence_missing,
    /// Task input arrived after the run had already reached a terminal state.
    input_after_terminal,
    /// The run was stopped immediately from outside the task.
    emergency_stop,
    /// A presenter reported on a request this run never made.
    ///
    /// Recorded and never anything more. A report whose identity does not match
    /// a request the run actually emitted is a statement about the presenter,
    /// not about a trial, and letting it stand in for evidence would let an
    /// unrelated report satisfy -- or destroy -- a trial's presentation
    /// evidence. The trial it named keeps whatever evidence it actually has.
    presentation_report_unmatched,
};

/// Whether @p condition is one of the declared AbnormalCondition values.
[[nodiscard]] constexpr bool abnormal_condition_declared(AbnormalCondition condition) noexcept
{
    return static_cast<std::uint16_t>(condition) <=
           static_cast<std::uint16_t>(AbnormalCondition::presentation_report_unmatched);
}

/// How severely a run treats a class of abnormal condition.
///
/// Configuration, not observation. What a policy produced in one concrete
/// situation is ::AbnormalResponse, and the two are recorded side by side
/// because a policy of `abort_trial` does not always end a trial: a paradigm
/// whose task state machine owns trial termination and offers no way to
/// abandon a trial can only mark the trial in flight inadmissible, and a reader
/// is entitled to see which of the two happened.
enum class AbnormalPolicy : std::uint8_t
{
    /// Record it and carry on. Never silent, and never the absence of a policy:
    /// it is the statement that this condition does not invalidate a
    /// measurement.
    record = 0,
    /// End or invalidate the trial in flight, and carry on with the run.
    abort_trial,
    /// End the run.
    abort_session,
};

/// Whether @p policy is one of the declared AbnormalPolicy values.
[[nodiscard]] constexpr bool abnormal_policy_declared(AbnormalPolicy policy) noexcept
{
    return static_cast<std::uint8_t>(policy) <=
           static_cast<std::uint8_t>(AbnormalPolicy::abort_session);
}

/// The more severe of @p left and @p right.
///
/// Used where a paradigm has a *floor* under a configured policy: a condition
/// after which the task's own accumulation demonstrably cannot continue is not
/// made continuable by configuring it as `record`.
[[nodiscard]] constexpr AbnormalPolicy escalate(AbnormalPolicy left, AbnormalPolicy right) noexcept
{
    return static_cast<std::uint8_t>(left) >= static_cast<std::uint8_t>(right) ? left : right;
}

/// What actually happened to the run because of one abnormal condition.
enum class AbnormalResponse : std::uint8_t
{
    /// It was recorded and nothing else. No input was refused and no
    /// measurement was invalidated.
    recorded = 0,
    /// The input was refused before it could reach the task.
    input_refused,
    /// The trial in flight kept running and is no longer admissible.
    ///
    /// The measurement it produces will still be recorded -- raw data is not
    /// deleted because it turned out to be unusable -- but it is recorded as
    /// inadmissible, and a trial marked this way is never reported as a
    /// success.
    trial_invalidated,
    /// The trial in flight was ended, and is recorded with
    /// ::TrialOutcome::aborted.
    trial_aborted,
    /// The run was ended.
    session_aborted,
};

/// Whether @p response is one of the declared AbnormalResponse values.
[[nodiscard]] constexpr bool abnormal_response_declared(AbnormalResponse response) noexcept
{
    return static_cast<std::uint8_t>(response) <=
           static_cast<std::uint8_t>(AbnormalResponse::session_aborted);
}

/// Whether @p response leaves the trial it names admissible.
///
/// The rule behind "no silent success": a trial an abnormal condition
/// invalidated or aborted cannot be reported as ::TrialOutcome::success, and a
/// writer that has this answer does not have to rediscover it per paradigm.
[[nodiscard]] constexpr bool abnormal_response_admits_trial(AbnormalResponse response) noexcept
{
    return response == AbnormalResponse::recorded || response == AbnormalResponse::input_refused;
}

/// The severity a run applies to each class of abnormal condition.
///
/// Grouped by what a paradigm can actually distinguish rather than one field
/// per condition, because a configuration with a knob nothing turns is a
/// configuration that reads as more considered than it is.
///
/// The defaults are the conservative ones: anything that breaks the continuity
/// a trial's measurement rests on ends that trial, anything that breaks the
/// run's ability to record ends the run, and nothing that left the task's input
/// untouched invalidates anything.
struct AbnormalPolicySet
{
    /// A decoded value that may not be applied, or a frame that is not the one
    /// the run prepared against.
    AbnormalPolicy decoded_command_invalid{AbnormalPolicy::abort_trial};
    /// A break in the data the paradigm consumes, in its own task input, or in
    /// the interval it is willing to apply a command across.
    AbnormalPolicy input_discontinuity{AbnormalPolicy::abort_trial};
    /// A presenter reporting that a requested presentation did not happen.
    AbnormalPolicy presentation_failed{AbnormalPolicy::abort_trial};
    /// A presentation with no reported evidence, where the run asked for
    /// evidence.
    ///
    /// `record` by default, because the absence of a report is a statement
    /// about the presenter and not about the trial: whether it invalidates the
    /// claim depends on whether the claim needed display timing, which is the
    /// caller's to say.
    AbnormalPolicy presentation_evidence_missing{AbnormalPolicy::record};
    /// The recorder, the acquisition path, or the runtime failing under the
    /// run, **as a paradigm observed it**.
    ///
    /// Deliberately narrower than it first reads. When the streaming runtime
    /// faults, or when the recorder refuses what a session produced, the
    /// reaction is not decided here: the runtime's own fault path decides the
    /// first (and is the only thing that inhibits an actuator), and
    /// `TraceLossPolicy` decides the second. Both were settled before this
    /// vocabulary existed and are not duplicated by it -- two policies for one
    /// decision would be one policy too many.
    ///
    /// What this governs is a paradigm that *itself* observes one of those
    /// conditions and reports it as an AbnormalEvent. No shipped paradigm
    /// does so, so on the shipped paths this field is inert; it is here
    /// because the conditions it maps are part of the vocabulary and a
    /// reporter needs an answer for them.
    AbnormalPolicy acquisition_fault{AbnormalPolicy::abort_session};
};

/// Validate a policy set.
///
/// @return ContractStatus::ok, or ContractStatus::enum_undeclared when any
///         field names no declared ::AbnormalPolicy. Called where a run freezes
///         its configuration, for the same reason every other persisted
///         enumeration is checked there: a severity nothing declared cannot be
///         compared, recorded, or replayed.
[[nodiscard]] ContractStatus validate(const AbnormalPolicySet& policies) noexcept;

/// The policy @p policies applies to @p condition.
///
/// Three conditions do not consult the set at all, and say so here rather than
/// in each paradigm: ::AbnormalCondition::observer_frame_drop is always
/// `record` because it did not touch the task's input,
/// ::AbnormalCondition::input_after_terminal is always `record` because the run
/// it would have ended has already ended, and
/// ::AbnormalCondition::emergency_stop is always `abort_session` because that
/// is the whole of what it means. ::AbnormalCondition::presentation_report_unmatched
/// joins them at `record`: a report the run cannot match to a request it made
/// is not evidence about any trial, and must not be allowed to end one.
[[nodiscard]] AbnormalPolicy policy_for(const AbnormalPolicySet& policies,
                                        AbnormalCondition condition) noexcept;

/// The response a condition decided under @p policy produces.
///
/// The whole of the rule, in one place, because two places would be two rules.
/// A live run applies it when it meets a condition; a deterministic replay
/// applies it again when it re-derives what the run did, and a replay that
/// re-implemented the mapping would be checking a recording against a second
/// opinion rather than against the rule the run obeyed.
///
/// @param policy         Severity in force, already escalated to any floor the
///                       paradigm imposed.
/// @param has_trial      Whether a trial was in flight to end or distrust.
/// @param can_end_trial  Whether the paradigm's own state machine is able to
///                       *end* the trial in flight, rather than only mark it
///                       inadmissible. A property of the paradigm, not of the
///                       condition: Center-Out can, WebGrid and Speech cannot.
/// @param input_refused  Whether the input carrying the condition was refused
///                       before it reached the task.
[[nodiscard]] AbnormalResponse abnormal_response_for(AbnormalPolicy policy, bool has_trial,
                                                     bool can_end_trial,
                                                     bool input_refused) noexcept;

/// One abnormal condition, as the run recorded it.
///
/// Fixed-size, trivially copyable, and free of pointers and text, so a realtime
/// path can produce one into a bounded queue exactly as it produces every other
/// trace record.
struct AbnormalEvent
{
    /// Experiment time the condition was observed, supplied by the caller.
    ExperimentTimeNs time_ns{};
    /// Emission ordinal within the reporter that produced it.
    ///
    /// Reporter-local and not session-global: one run may hold more than one
    /// reporter -- Speech holds two, because the scheduler and the presenter
    /// bridge are two producers on two threads -- so two events of one session
    /// can legitimately carry the same ordinal. Ordering across a session comes
    /// from the control records themselves, which is where it already was.
    SequenceOrdinal sequence{};
    /// Trial the condition applies to, or an unset identity when it applies to
    /// no trial in particular.
    TrialIdentity trial{};
    /// Which paradigm observed it.
    ParadigmId paradigm{kUnsetParadigmId};
    /// What was observed.
    AbnormalCondition condition{AbnormalCondition::unspecified};
    /// The severity in force for that class of condition.
    AbnormalPolicy policy{AbnormalPolicy::record};
    /// What the run actually did.
    AbnormalResponse response{AbnormalResponse::recorded};
    /// Whether `trial` names a trial at all.
    ///
    /// Explicit rather than inferred from an unset ordinal, because trial
    /// ordinal zero is a perfectly good trial and a reader must not have to
    /// guess whether the first trial or no trial was meant.
    bool has_trial{};
    /// The reporting domain's own status or fault enumerator, as its integer
    /// value.
    ///
    /// The experiment contract must not depend on the streaming domain, so it
    /// records the number rather than the type -- the same rule
    /// ::CommandOutcome::status_code follows. The layer that produced it is the
    /// only one that knows which enumeration it came from.
    std::uint32_t detail{};
};

/// Stable ASCII name of @p condition.
///
/// The returned pointer has static storage duration, so naming a condition
/// allocates nothing. These are names rather than numbers because an abnormal
/// condition is *shared* contract vocabulary, unlike a paradigm's own state,
/// cause, and reason enumerators -- which the records carry as opaque numbers
/// precisely because this layer does not own them.
[[nodiscard]] const char* abnormal_condition_name(AbnormalCondition condition) noexcept;

/// Validate an abnormal event.
///
/// @return ContractStatus::ok; ContractStatus::enum_undeclared for a condition,
///         policy, or response naming no declared enumerator;
///         ContractStatus::identity_missing when `paradigm` is unset; or
///         ContractStatus::outcome_invalid when the fields contradict one
///         another -- a response that names a trial the event does not carry,
///         or a response more severe than the policy that produced it allows.
[[nodiscard]] ContractStatus validate(const AbnormalEvent& event) noexcept;

} // namespace neurale::experiments
