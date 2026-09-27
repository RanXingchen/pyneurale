/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

#include <neurale/experiments/contract.h>
#include <neurale/experiments/identity.h>
#include <neurale/experiments/time.h>

/**
 * @file
 * @brief Runtime experiment events, state transitions, selections, and trials.
 *
 * These are the runtime records the paradigms emit. They are fixed-size,
 * trivially copyable, and carry no strings and no pointers, so producing one
 * allocates nothing and a bounded trace buffer is one array.
 *
 * They are not a second event or trial data model. `neurale.data.Event`,
 * `EventSeries`, `Trial`, and `TrialTable` remain the offline analysis
 * containers, in floating-point seconds. Converting a runtime record into one
 * of those is an export boundary, and it is the only place where integer
 * nanoseconds become seconds.
 */
namespace neurale::experiments
{

/// Kind of a semantic experiment event.
///
/// The set is shared and append-only: existing values are never renumbered, and
/// a paradigm that needs an event of its own emits
/// ExperimentEventKind::paradigm_marker with its own code rather than extending
/// this enumeration.
enum class ExperimentEventKind : std::uint16_t
{
    /// Unset.
    unspecified = 0,
    /// The experiment session began.
    session_start,
    /// The experiment session ended.
    session_stop,
    /// A block of trials began.
    block_start,
    /// A block of trials ended.
    block_stop,
    /// A trial began.
    trial_start,
    /// A trial ended.
    trial_stop,
    /// A paradigm state machine changed state.
    state_transition,
    /// A cue was requested for presentation.
    presentation_request,
    /// The believed presentation state changed.
    presentation_state,
    /// A discrete or dwell selection was made.
    selection,
    /// A command was generated for the runtime.
    command_request,
    /// The runtime reported what it could prove about a command.
    command_outcome,
    /// A randomized value was realized.
    schedule_draw,
    /// A trial outcome was decided.
    trial_outcome,
    /// A paradigm-owned marker carrying a paradigm-owned code.
    paradigm_marker,
};

/// Whether @p kind is one of the declared ExperimentEventKind values.
///
/// A range test, valid because the set is contiguous and append-only.
[[nodiscard]] constexpr bool experiment_event_kind_declared(ExperimentEventKind kind) noexcept
{
    return static_cast<std::uint16_t>(kind) <=
           static_cast<std::uint16_t>(ExperimentEventKind::paradigm_marker);
}

/// One compact semantic experiment event.
struct ExperimentEvent
{
    /// Experiment time the event is stamped with, supplied by the caller.
    ExperimentTimeNs time_ns{};
    /// Emission ordinal within the session.
    SequenceOrdinal sequence{};
    /// Trial the event belongs to.
    TrialIdentity trial{};
    /// What happened.
    ExperimentEventKind kind{ExperimentEventKind::unspecified};
    /// Which paradigm emitted it.
    ParadigmId paradigm{kUnsetParadigmId};
    /// Paradigm-owned code. Meaningful for ExperimentEventKind::paradigm_marker.
    std::uint32_t code{};
    /// Paradigm-owned integer payload. There is no floating-point and no string
    /// payload: a value whose meaning needs text is carried by an identifier
    /// resolved outside the hot path.
    std::int64_t value{};
};

/// One semantic state transition of one paradigm state machine.
///
/// `from_state` and `to_state` are the paradigm's own enumerators widened for
/// storage. The contract records them against ::ParadigmId and never interprets
/// them: there is no shared state enumeration, because the three paradigms do
/// not share states.
struct StateTransition
{
    /// Experiment time the transition was taken.
    ExperimentTimeNs time_ns{};
    /// Emission ordinal within the session.
    SequenceOrdinal sequence{};
    /// Trial the transition belongs to.
    TrialIdentity trial{};
    /// Paradigm whose state enumeration `from_state` and `to_state` belong to.
    ParadigmId paradigm{kUnsetParadigmId};
    /// State left.
    StateId from_state{};
    /// State entered.
    StateId to_state{};
    /// Paradigm-owned transition cause code.
    std::uint32_t cause{};
};

/// How a selection was made.
enum class SelectionKind : std::uint8_t
{
    /// An explicit discrete selection input.
    discrete = 0,
    /// Continuous occupancy of one item for a configured duration.
    dwell,
};

/// Whether @p kind is one of the declared SelectionKind values.
[[nodiscard]] constexpr bool selection_kind_declared(SelectionKind kind) noexcept
{
    return static_cast<std::uint8_t>(kind) <= static_cast<std::uint8_t>(SelectionKind::dwell);
}

/// One discrete user selection.
///
/// Required by WebGrid, which must distinguish a correct selection from a
/// misclick and both from a trial that reached its timeout with no selection at
/// all -- the last of which produces no SelectionEvent, which is exactly how it
/// stays distinguishable.
struct SelectionEvent
{
    /// Experiment time the selection was made.
    ExperimentTimeNs time_ns{};
    /// Emission ordinal within the session.
    SequenceOrdinal sequence{};
    /// Trial the selection belongs to.
    TrialIdentity trial{};
    /// Which paradigm produced it.
    ParadigmId paradigm{kUnsetParadigmId};
    /// How the selection was made.
    SelectionKind kind{SelectionKind::discrete};
    /// Whether the selected item was the intended one.
    bool correct{};
    /// Item selected, or ::kUnsetTargetId when nothing selectable was under the pointer.
    TargetId selected_id{kUnsetTargetId};
    /// Item the trial intended.
    TargetId intended_id{kUnsetTargetId};
    /// Occupancy that produced a dwell selection. Zero for a discrete selection.
    DurationNs dwell_ns{};
};

/// The runtime record of one completed or in-flight trial.
///
/// This is not `neurale.data.Trial`. That type describes an offline trial
/// interval in seconds and cannot represent a pending outcome or a
/// paradigm-owned reason code; this one is the runtime value, and conversion
/// between them happens at export.
struct TrialRecord
{
    /// Trial this record describes.
    TrialIdentity trial{};
    /// Half-open trial interval. `end_ns` equals `start_ns` while pending.
    TimeInterval interval{};
    /// Which paradigm ran the trial.
    ParadigmId paradigm{kUnsetParadigmId};
    /// How the trial ended.
    TrialOutcome outcome{TrialOutcome::pending};
    /// Paradigm-owned outcome reason code.
    std::uint32_t reason{};
};

/// Validate an experiment event.
[[nodiscard]] ContractStatus validate(const ExperimentEvent& event) noexcept;

/// Validate a state transition.
[[nodiscard]] ContractStatus validate(const StateTransition& transition) noexcept;

/// Validate a selection event.
[[nodiscard]] ContractStatus validate(const SelectionEvent& selection) noexcept;

/// Validate a trial record.
[[nodiscard]] ContractStatus validate(const TrialRecord& record) noexcept;

} // namespace neurale::experiments
