/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <limits>

/**
 * @file
 * @brief Shared vocabulary of the experiment value contract.
 *
 * This header declares the scalar types and the failure vocabulary that every
 * other experiment contract header builds on. It depends on nothing: not on
 * streaming, not on any algorithm domain, and not on Python. That is what lets
 * `neurale.experiments.assistance` use the contract without acquiring a
 * dependency on the streaming runtime.
 *
 * The contract carries values only. It defines no paradigm base class, no task
 * registry, and no scheduler; each paradigm header owns its own configuration,
 * state enumerations, and transition logic.
 */
namespace neurale::experiments
{

/// Monotonic experiment time as an integer nanosecond count.
///
/// There are no floating-point seconds anywhere in the experiment contract. A
/// state machine never samples this value: every time in the contract was
/// supplied by the caller.
using ExperimentTimeNs = std::uint64_t;

/// Non-negative experiment duration as an integer nanosecond count.
using DurationNs = std::uint64_t;

/// Session-local trial ordinal: deterministic, monotonic, and zero-based.
using TrialOrdinal = std::uint64_t;

/// Stable trial identifier that survives across sessions. Zero means unset.
///
/// Separate from ::TrialOrdinal on purpose: an ordinal answers "which trial of
/// this session", a key answers "which trial", and conflating them makes a
/// resumed or re-blocked session unreadable.
using TrialKey = std::uint64_t;

/// Session-local block ordinal.
using BlockOrdinal = std::uint32_t;

/// Emission ordinal of a semantic record within one session. Provenance only.
using SequenceOrdinal = std::uint64_t;

/// Semantic target or selectable-item identifier. Zero means unset.
using TargetId = std::uint32_t;

/// Semantic stimulus identifier, resolved against a prepared catalog. Zero
/// means unset. The identifier is all a hot path ever carries; the content it
/// stands for lives in the owning paradigm's immutable catalog.
using StimulusId = std::uint32_t;

/// Identifies which paradigm produced a record. Zero means unset.
///
/// Provenance, never dispatch. Nothing in the contract selects behaviour from
/// this value; it exists so that a persisted ::StateId can be read back
/// against the right paradigm's own state enumeration.
using ParadigmId = std::uint16_t;

/// One paradigm's own state enumerator, widened for storage.
///
/// The contract stores it and never interprets it.
using StateId = std::uint16_t;

/// One paradigm's own phase enumerator, widened for storage.
using PhaseId = std::uint16_t;

/// Sentinel `valid_until_ns` meaning "this request does not expire".
inline constexpr ExperimentTimeNs kNoExpiryNs = (std::numeric_limits<std::uint64_t>::max)();

/// Unset ::TrialKey.
inline constexpr TrialKey kUnsetTrialKey = 0;

/// Unset ::TargetId.
inline constexpr TargetId kUnsetTargetId = 0;

/// Unset ::StimulusId.
inline constexpr StimulusId kUnsetStimulusId = 0;

/// Unset ::ParadigmId.
inline constexpr ParadigmId kUnsetParadigmId = 0;

/// Exact reason one contract value was rejected.
///
/// Validation returns a reason rather than a bool so that a caller reports what
/// was wrong instead of guessing, and so that no validation path needs to throw
/// on a realtime thread.
enum class ContractStatus : std::uint8_t
{
    /// The value satisfies the contract.
    ok = 0,
    /// Supplied time moved backwards relative to the last accepted time.
    time_regressed,
    /// An interval ends before it starts.
    interval_inverted,
    /// A start plus a duration does not fit in an integer nanosecond count.
    duration_overflow,
    /// A command dimension is zero or exceeds the fixed command capacity.
    dimension_invalid,
    /// A realtime command value is NaN or infinite.
    value_not_finite,
    /// A request expires no later than it was generated.
    expiry_before_generation,
    /// A sampling range contains no admissible value.
    range_empty,
    /// A bounded draw rejected every one of its attempts.
    ///
    /// Each attempt is accepted with probability above one half, so this has
    /// probability below 2^-64 and is reported rather than worked around: a
    /// value folded into range after exhaustion would be biased, and a sampler
    /// that is uniform except on a rare path is not a uniform sampler.
    sampling_exhausted,
    /// An ordinal sequence has no unissued ordinal left.
    ordinal_exhausted,
    /// A field the record cannot mean anything without is unset.
    identity_missing,
    /// An enumerated field holds a number that names no declared enumerator.
    ///
    /// Distinct from ::identity_missing, which reports a field that is validly
    /// *unset*. This one reports a field that is corrupt or was written by a
    /// build that declared more values than this one does, and treating the two
    /// alike would let an unreadable record pass as an incomplete one.
    enum_undeclared,
    /// Individually legal presentation fields describe no possible presentation.
    ///
    /// A cue that carries a payload its kind does not have, or a status that
    /// contradicts the times reported beside it. Every field passes on its own;
    /// what fails is the state they jointly describe. An undeclared enumerator
    /// is ::enum_undeclared instead, because a corrupt field and a contradictory
    /// record are read back by different people for different reasons.
    presentation_invalid,
    /// Individually legal fields of any other record contradict one another.
    ///
    /// The counterpart of ::presentation_invalid outside presentation: a trial
    /// outcome, a selection, or a command application that disagrees with the
    /// record carrying it.
    outcome_invalid,
    /// An assistance parameter lies outside the interval its method declares.
    ///
    /// Reported rather than clipped. A transform that quietly moved a gain of
    /// 1.5 back to 1.0 would return a velocity nobody asked for, and the caller
    /// that computed 1.5 would keep computing it.
    assistance_out_of_range,
    /// A domain mask selects no axis, or selects one the command does not have.
    ///
    /// Separate from ::dimension_invalid because the dimension may be perfectly
    /// good: what is wrong is the subset of it that was asked for.
    domain_mask_invalid,
    /// An assistance trace record names an algorithm version this build does not
    /// replay.
    ///
    /// Distinct from ::identity_missing, which reports an unset identifier, and
    /// from ::enum_undeclared, which reports an enumerator this build did not
    /// declare. A record carrying version 999 is structurally readable -- every
    /// field is individually valid -- but the transform that produced it is not
    /// one this build can recompute, so the record is not valid provenance. The
    /// schedule's sampler version is handled differently on purpose: an
    /// unsupported sampler version still has a replay path through the recorded
    /// schedule, so it is ::ok; an assistance record has no such fallback.
    version_unsupported,
    /// A bounded procedure ran to its iteration budget without meeting its
    /// termination criterion.
    ///
    /// Reported rather than returning a result built on an unknown numerical
    /// state. The orthogonal-impedance projector factorizes a Gram matrix with
    /// a bounded Jacobi rotation; if the rotation does not converge within its
    /// sweep cap the projector is not trustworthy, and the transform reports
    /// this rather than handing back a velocity that might be anything.
    ///
    /// The Center-Out step chain uses it for the same reason: it advances to a
    /// fixed point under a transition budget, and reaching that budget means the
    /// machine did not settle where its own rules say it must.
    numerical_failure,
    /// A configuration parameter is finite but outside the interval its role
    /// declares.
    ///
    /// Distinct from ::value_not_finite, which reports a NaN or infinite value:
    /// a tolerance of -1.0 is finite but inadmissible, and reporting it as
    /// "not finite" would misname the problem. Also distinct from
    /// ::assistance_out_of_range, which is reserved for the [0, 1] assistance
    /// and impedance parameters. The orthogonal-impedance rank and conditioning
    /// tolerances use this: a negative tolerance, or a conditioning tolerance
    /// above one, is a parameter this build will not apply.
    parameter_out_of_range,
    /// A set of identified items is empty, oversized, or names one item twice.
    ///
    /// Distinct from ::dimension_invalid, which reports the width of one value:
    /// this reports the membership of a collection. Distinct from
    /// ::outcome_invalid too, because a duplicated target identifier is not two
    /// fields disagreeing -- each entry is perfectly consistent, and what is
    /// wrong is that the set cannot say which one a record referring to that
    /// identifier meant.
    target_set_invalid,
    /// A stateful procedure was asked to advance before it was started, or after
    /// it had finished.
    ///
    /// Distinct from ::outcome_invalid, which reports a record whose fields
    /// disagree: nothing here is malformed. The procedure simply has no step to
    /// take, and answering with a snapshot of a session that never began -- or
    /// of one that ended -- would let a caller mistake "not started" for
    /// "started and doing nothing".
    not_running,
    /// A procedure that holds one session was asked to begin another without
    /// first returning to its idle state.
    ///
    /// The Center-Out machine freezes its lifecycle as idle -> start ->
    /// running -> ... -> complete, and a second start() on a non-idle machine
    /// would silently replace the session in progress -- dropping its trial,
    /// clearing its ordinals, and emitting a fresh session_start -- rather than
    /// begin a new one. A new session needs reset() first. Distinct from
    /// ::not_running, which is the inverse: that reports step() on a machine
    /// with no active session.
    already_running,
};

/// Stable human-readable message for @p status.
///
/// The returned pointer has static storage duration, so reporting a rejection
/// allocates nothing and is safe to call from a bounded context.
[[nodiscard]] const char* contract_status_message(ContractStatus status) noexcept;

} // namespace neurale::experiments
