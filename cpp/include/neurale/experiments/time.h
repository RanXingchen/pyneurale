/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <limits>

#include <neurale/experiments/contract.h>

/**
 * @file
 * @brief Explicit experiment time, half-open intervals, and the monotonic gate.
 *
 * Two rules are fixed here and are not restated by the paradigms:
 *
 * - every phase and trial interval is half-open, `[start_ns, end_ns)`, so the
 *   exact end instant belongs to the following phase and never to both;
 * - supplied time is monotonic non-decreasing, and a state machine has no
 *   notion of "now" it did not receive as an argument.
 *
 * Nothing in this header reads a clock.
 */
namespace neurale::experiments
{

/// Nanoseconds in one second.
///
/// Written down once because it is an *integration factor*, not a formatting
/// convenience: a velocity is applied to a cursor by multiplying it by an
/// interval expressed in seconds, and a deterministic replay of that
/// multiplication has to divide by exactly the same constant the run did. Two
/// spellings of it in two translation units is two chances for that to stop
/// being true.
inline constexpr std::uint64_t kNanosecondsPerSecond = 1'000'000'000ULL;

/// Half-open experiment interval `[start_ns, end_ns)`.
///
/// The exact end instant is *not* contained: a phase that ends at `t` and a
/// phase that starts at `t` do not both own `t`. An interval with
/// `end_ns == start_ns` is empty and contains nothing at all.
struct TimeInterval
{
    /// First instant in the interval.
    ExperimentTimeNs start_ns{};
    /// First instant after the interval.
    ExperimentTimeNs end_ns{};

    /// Whether the interval is well formed, that is, does not end before it starts.
    [[nodiscard]] constexpr bool well_formed() const noexcept
    {
        return end_ns >= start_ns;
    }

    /// Whether the interval contains no instant.
    [[nodiscard]] constexpr bool empty() const noexcept
    {
        return end_ns <= start_ns;
    }

    /// Length of the interval. Zero for an ill-formed or empty interval.
    [[nodiscard]] constexpr DurationNs duration_ns() const noexcept
    {
        return end_ns > start_ns ? end_ns - start_ns : DurationNs{0};
    }

    /// Half-open containment of @p time_ns.
    [[nodiscard]] constexpr bool contains(ExperimentTimeNs time_ns) const noexcept
    {
        return time_ns >= start_ns && time_ns < end_ns;
    }

    /// Whether @p time_ns has reached or passed the end of the interval.
    ///
    /// This is the predicate a timeout or a phase advance uses, and it is the
    /// exact complement of contains() above `start_ns`: at `end_ns` the
    /// interval is over, which is what makes the boundary unambiguous.
    [[nodiscard]] constexpr bool elapsed_at(ExperimentTimeNs time_ns) const noexcept
    {
        return time_ns >= end_ns;
    }
};

/// Whether `start_ns + duration_ns` fits in an integer nanosecond count.
[[nodiscard]] constexpr bool time_fits(ExperimentTimeNs start_ns, DurationNs duration_ns) noexcept
{
    return duration_ns <= (std::numeric_limits<ExperimentTimeNs>::max)() - start_ns;
}

/// Build `[start_ns, start_ns + duration_ns)`, reporting overflow.
///
/// @param start_ns First instant of the interval.
/// @param duration_ns Interval length.
/// @param interval Receives the interval when the result is ContractStatus::ok.
/// @return ContractStatus::ok, or ContractStatus::duration_overflow.
[[nodiscard]] constexpr ContractStatus interval_from_duration(ExperimentTimeNs start_ns,
                                                              DurationNs duration_ns,
                                                              TimeInterval& interval) noexcept
{
    if (!time_fits(start_ns, duration_ns))
        return ContractStatus::duration_overflow;
    interval = TimeInterval{start_ns, start_ns + duration_ns};
    return ContractStatus::ok;
}

/// Validate that an interval is well formed.
[[nodiscard]] constexpr ContractStatus validate(const TimeInterval& interval) noexcept
{
    return interval.well_formed() ? ContractStatus::ok : ContractStatus::interval_inverted;
}

/// Guard for the "supplied time is monotonic non-decreasing" precondition.
///
/// The gate never samples a clock and never advances on its own. It remembers
/// the last accepted instant so that a caller handing a paradigm an out-of-order
/// time is told so, instead of the paradigm computing a negative elapsed
/// duration in unsigned arithmetic and silently producing a very large one.
///
/// Equal times are accepted: two semantic decisions may share an instant.
class MonotonicTimeGate
{
  public:
    /// Construct an unstarted gate that accepts any first instant.
    constexpr MonotonicTimeGate() noexcept = default;

    /// Construct a gate that has already accepted @p origin_ns.
    constexpr explicit MonotonicTimeGate(ExperimentTimeNs origin_ns) noexcept
        : last_ns_(origin_ns), started_(true)
    {
    }

    /// Accept @p time_ns when it does not move backwards.
    ///
    /// @return ContractStatus::ok, or ContractStatus::time_regressed, in which
    ///         case the gate is left unchanged.
    [[nodiscard]] constexpr ContractStatus accept(ExperimentTimeNs time_ns) noexcept
    {
        if (started_ && time_ns < last_ns_)
            return ContractStatus::time_regressed;
        last_ns_ = time_ns;
        started_ = true;
        return ContractStatus::ok;
    }

    /// Last accepted instant. Zero while the gate is unstarted.
    [[nodiscard]] constexpr ExperimentTimeNs last_ns() const noexcept
    {
        return last_ns_;
    }

    /// Whether any instant has been accepted.
    [[nodiscard]] constexpr bool started() const noexcept
    {
        return started_;
    }

    /// Return to the unstarted state, so the next instant is accepted whatever
    /// it is. This is a session boundary, not a trial boundary.
    constexpr void reset() noexcept
    {
        last_ns_ = 0;
        started_ = false;
    }

    /// Restart from an explicit origin, which counts as already accepted.
    constexpr void reset(ExperimentTimeNs origin_ns) noexcept
    {
        last_ns_ = origin_ns;
        started_ = true;
    }

  private:
    ExperimentTimeNs last_ns_{};
    bool started_{};
};

} // namespace neurale::experiments
