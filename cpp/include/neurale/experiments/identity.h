/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include <neurale/experiments/contract.h>

/**
 * @file
 * @brief Trial identity, deterministic ordinals, and trial outcomes.
 *
 * Every identifier here is a plain integer. No object address, no pointer, and
 * no hash of one participates in a persisted identifier: a session recorded on
 * one run must be readable against a session recorded on another, and an
 * address is not reproducible.
 */
namespace neurale::experiments
{

/// Which trial a record belongs to.
///
/// The ordinal is session-local and deterministic; the key is the optional
/// stable identity that survives across sessions. Target and stimulus
/// identifiers are carried here because most records are meaningless without
/// them and because carrying them costs nothing in a fixed-size record.
struct TrialIdentity
{
    /// Session-local, zero-based, deterministic trial ordinal.
    TrialOrdinal ordinal{};
    /// Stable cross-session trial identifier, or ::kUnsetTrialKey.
    TrialKey key{kUnsetTrialKey};
    /// Session-local block ordinal.
    BlockOrdinal block{};
    /// Semantic target for this trial, or ::kUnsetTargetId.
    TargetId target_id{kUnsetTargetId};
    /// Semantic stimulus for this trial, or ::kUnsetStimulusId.
    StimulusId stimulus_id{kUnsetStimulusId};
};

static_assert(std::is_trivially_copyable_v<TrialIdentity>);

/// Whether two identities refer to the same trial of the same session.
[[nodiscard]] constexpr bool same_trial(const TrialIdentity& left,
                                        const TrialIdentity& right) noexcept
{
    return left.ordinal == right.ordinal && left.block == right.block;
}

/// How one trial ended.
///
/// Deliberately small. Finer reasons -- which target was missed, which cell was
/// clicked, why a hold was broken -- are paradigm-owned and travel in the
/// `reason` code of the record that carries the outcome, not as new enumerators
/// here.
enum class TrialOutcome : std::uint8_t
{
    /// The trial has not ended.
    pending = 0,
    /// The trial met its task-defined success criterion.
    success,
    /// The trial ended without meeting it, for a paradigm-owned reason.
    failure,
    /// The trial reached a configured time limit with no decision.
    timeout,
    /// The trial was ended by something outside the task.
    aborted,
};

/// Whether @p outcome is one of the declared TrialOutcome values.
///
/// Every persisted enumeration in the contract has one of these. A record read
/// back from a file, a wire, or an older build can hold a number that names no
/// enumerator, and a validator that only asks "is this the pending one" would
/// wave it through as a perfectly good terminal outcome.
///
/// The check is a range test because the set is contiguous and append-only:
/// enumerators are never renumbered, so the last one is the upper bound.
[[nodiscard]] constexpr bool trial_outcome_declared(TrialOutcome outcome) noexcept
{
    return static_cast<std::uint8_t>(outcome) <= static_cast<std::uint8_t>(TrialOutcome::aborted);
}

/// Whether @p outcome is terminal, that is, anything other than pending.
///
/// Undeclared values are not terminal: an outcome that names no enumerator does
/// not get to end a trial. Callers should reject it through
/// trial_outcome_declared() rather than treat this `false` as "still running".
[[nodiscard]] constexpr bool trial_ended(TrialOutcome outcome) noexcept
{
    return trial_outcome_declared(outcome) && outcome != TrialOutcome::pending;
}

/// Deterministic monotonic ordinal source with explicit reset.
///
/// Neither a clock nor a random source: the next value is a pure function of
/// how many values were issued since the last reset, so a replay that issues
/// the same number of values sees the same ordinals.
///
/// @tparam Ordinal Unsigned integer ordinal type.
/// @tparam Tag Empty type distinguishing one ordinal sequence from another, so
///         that a trial counter and a record counter are different types even
///         when they count with the same integer.
template <typename Ordinal, typename Tag> class OrdinalCounter
{
    static_assert(std::is_unsigned_v<Ordinal>, "an ordinal counter must be unsigned");

  public:
    /// Construct a counter whose first issued value is @p origin.
    constexpr explicit OrdinalCounter(Ordinal origin = 0) noexcept : origin_(origin), next_(origin)
    {
    }

    /// Value the next issue() will return, without issuing it.
    [[nodiscard]] constexpr Ordinal peek() const noexcept
    {
        return next_;
    }

    /// Whether the sequence has no ordinal left to issue.
    ///
    /// The maximum representable ordinal is reserved as this boundary and is
    /// never issued. Spending one value out of the range buys an issued() count
    /// that is exact and cannot wrap, and it makes exhaustion a state the
    /// counter can be *in* rather than one it can only be about to enter.
    [[nodiscard]] constexpr bool exhausted() const noexcept
    {
        return next_ == (std::numeric_limits<Ordinal>::max)();
    }

    /// Issue the next ordinal and advance.
    ///
    /// An exhausted counter issues nothing. It does not saturate onto its last
    /// value: an ordinal is an identity, so handing the same one out twice would
    /// silently break the uniqueness that every persisted record depends on,
    /// which is worse than failing.
    ///
    /// @param value Receives the ordinal on ContractStatus::ok, and is left
    ///              unmodified otherwise.
    /// @return ContractStatus::ok, or ContractStatus::ordinal_exhausted.
    [[nodiscard]] constexpr ContractStatus issue(Ordinal& value) noexcept
    {
        if (exhausted())
            return ContractStatus::ordinal_exhausted;
        value = next_;
        ++next_;
        return ContractStatus::ok;
    }

    /// Number of ordinals issued since construction or the last reset.
    [[nodiscard]] constexpr Ordinal issued() const noexcept
    {
        return static_cast<Ordinal>(next_ - origin_);
    }

    /// Origin this counter restarts from.
    [[nodiscard]] constexpr Ordinal origin() const noexcept
    {
        return origin_;
    }

    /// Restart from the construction origin.
    ///
    /// This restarts the session-local ordinal sequence, so it is a session
    /// boundary. Resetting mid-session re-issues ordinals that were already
    /// persisted and makes the session unreadable.
    constexpr void reset() noexcept
    {
        next_ = origin_;
    }

    /// Restart from an explicit new origin, which becomes the next issued value.
    constexpr void reset(Ordinal origin) noexcept
    {
        origin_ = origin;
        next_ = origin;
    }

  private:
    Ordinal origin_{};
    Ordinal next_{};
};

/// Tag distinguishing the trial ordinal sequence.
struct TrialOrdinalTag;

/// Tag distinguishing the record emission ordinal sequence.
struct SequenceOrdinalTag;

/// Deterministic session-local trial ordinal source.
using TrialCounter = OrdinalCounter<TrialOrdinal, TrialOrdinalTag>;

/// Deterministic session-local record emission ordinal source.
using SequenceCounter = OrdinalCounter<SequenceOrdinal, SequenceOrdinalTag>;

} // namespace neurale::experiments
