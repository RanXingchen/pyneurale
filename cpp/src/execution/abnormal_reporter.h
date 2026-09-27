/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The abnormal-condition bookkeeping all three paradigm controllers do
/// identically.
///
/// What is shared here is arithmetic, not semantics. Deciding *that* the
/// pointer stopped arriving, or that a decoded value is not a number, is each
/// paradigm's own; turning that observation into a policy decision, a response,
/// and a set of counters is the same operation everywhere, and three copies of
/// it would be three chances for one paradigm to answer "was this trial still
/// admissible" differently from another.
///
/// Every operation is bounded, allocation-free, and callable from a realtime
/// thread: a Center-Out observation meets its abnormal conditions on the
/// runtime's own path, and a reporter that allocated there would be a reporter
/// that could not be used where it is most needed.

#include <cstdint>

#include <neurale/experiments/abnormal.h>
#include <neurale/experiments/identity.h>

namespace neurale::execution
{

using namespace neurale::experiments;

/// What a paradigm reports about the abnormal conditions it has met.
///
/// Two kinds of field, with two different lifetimes, and the difference is not
/// cosmetic:
///
/// - The **counts** are for the source's whole lifetime and never decrease,
///   exactly like ::ExperimentTraceSource::dropped_trace_count. A controller
///   that is reset and started again keeps its totals, so a session reads them
///   as *positions* and reports only what accrued after it started.
/// - `session_aborted`, `primary`, and `primary_policy` are about the run
///   currently executing, and are cleared when one starts. They have to be:
///   a latch and a severity cannot be recovered by subtracting a baseline the
///   way a count can, so a lifetime latch would make every later run of one
///   controller inherit the first run's abort and the first run's primary
///   condition -- and a session reading them has no way to tell that from a
///   run that really did abort.
struct AbnormalSummary
{
    /// Abnormal conditions observed.
    std::uint64_t observed{};
    /// Of those, the ones that ended or invalidated a trial.
    std::uint64_t trials_affected{};
    /// Of those, the ones whose only response was to record them.
    std::uint64_t recorded_only{};
    /// Whether one of them demanded that the run end.
    ///
    /// A latch, not a count: a run does not become more ended.
    bool session_aborted{};
    /// The first condition observed at the highest severity reached.
    ///
    /// The first rather than the last, because the interesting condition is the
    /// one that broke the run, and everything after it may well be a
    /// consequence of it.
    AbnormalCondition primary{AbnormalCondition::unspecified};
    /// The severity `primary` was decided under, so that two summaries can be
    /// merged without having to re-derive it.
    AbnormalPolicy primary_policy{AbnormalPolicy::record};
};

/// Combine two summaries from two reporters of one run.
///
/// Speech is why this exists: the scheduler meets its own conditions on the
/// thread that advances it, and the presenter's reports are drained on the
/// bridge thread, so one run has two single-producer reporters rather than one
/// shared object with a lock on the path a realtime producer uses.
[[nodiscard]] inline AbnormalSummary merge(const AbnormalSummary& left,
                                           const AbnormalSummary& right) noexcept
{
    AbnormalSummary merged{};
    merged.observed = left.observed + right.observed;
    merged.trials_affected = left.trials_affected + right.trials_affected;
    merged.recorded_only = left.recorded_only + right.recorded_only;
    merged.session_aborted = left.session_aborted || right.session_aborted;
    // The more severe primary wins, and a tie keeps the left one. Neither
    // summary knows which was observed first, so ordering between them is not
    // claimed -- what is claimed is the severity, which is what a reader is
    // looking for.
    const bool take_right = left.primary == AbnormalCondition::unspecified ||
                            static_cast<std::uint8_t>(right.primary_policy) >
                                static_cast<std::uint8_t>(left.primary_policy);
    merged.primary = take_right && right.primary != AbnormalCondition::unspecified ? right.primary
                                                                                   : left.primary;
    merged.primary_policy = take_right && right.primary != AbnormalCondition::unspecified
                                ? right.primary_policy
                                : left.primary_policy;
    return merged;
}

/// Turns one observation into a decision, a record, and a set of counters.
class AbnormalReporter
{
  public:
    AbnormalReporter() noexcept = default;

    /// Fix the severities and the paradigm identity. Called from prepare().
    void configure(const AbnormalPolicySet& policies, ParadigmId paradigm) noexcept
    {
        policies_ = policies;
        paradigm_ = paradigm;
    }

    /// Begin a new run: clear what is about this run, keep what is a total.
    ///
    /// The counts are deliberately **not** restarted. They are a lifetime
    /// counter by contract, and the session that reads them takes its own
    /// baseline; clearing them here would make a controller that ran twice
    /// report the second run's conditions as though the first had none, and the
    /// session would have no way to tell that from a controller that really was
    /// fresh.
    ///
    /// The abort latch and the primary condition are restarted, for the exact
    /// opposite reason: they are not differences, so no baseline recovers them.
    /// A controller reused after a run that was emergency-stopped would
    /// otherwise report the next, entirely clean, run as aborted by the
    /// previous run's stop.
    void restart() noexcept
    {
        sequence_ = 0;
        session_aborted_ = false;
        primary_ = AbnormalCondition::unspecified;
        primary_policy_ = AbnormalPolicy::record;
    }

    /// Decide and account one abnormal condition.
    ///
    /// @param condition   What was observed.
    /// @param time_ns     Experiment instant it was observed at, supplied by
    ///                    the caller like every other instant in the contract.
    /// @param trial       Trial in flight, or nullptr when none is.
    /// @param detail      The reporting domain's own enumerator, as a number.
    /// @param floor       Severity the paradigm will not go below whatever the
    ///                    configuration says, because the accumulation a lower
    ///                    severity implies demonstrably cannot continue.
    /// @param can_end_trial Whether this paradigm is able to *end* the trial in
    ///                    flight, rather than only mark it inadmissible. It is
    ///                    a property of the paradigm's own state machine and
    ///                    not of the condition.
    /// @param input_refused Whether the input that carried the condition was
    ///                    refused before it reached the task.
    [[nodiscard]] AbnormalEvent observe(AbnormalCondition condition, ExperimentTimeNs time_ns,
                                        const TrialIdentity* trial, std::uint32_t detail,
                                        AbnormalPolicy floor, bool can_end_trial,
                                        bool input_refused) noexcept
    {
        const auto policy = escalate(policy_for(policies_, condition), floor);

        AbnormalEvent event{};
        event.time_ns = time_ns;
        event.sequence = sequence_++;
        event.paradigm = paradigm_;
        event.condition = condition;
        event.policy = policy;
        event.detail = detail;
        if (trial != nullptr)
        {
            event.trial = *trial;
            event.has_trial = true;
        }

        // The mapping itself belongs to the contract, not to this class. A
        // deterministic replay re-derives the same response from the same
        // recorded condition, and two copies of the rule would be two rules.
        event.response =
            abnormal_response_for(policy, trial != nullptr, can_end_trial, input_refused);

        ++observed_;
        switch (event.response)
        {
        case AbnormalResponse::trial_invalidated:
        case AbnormalResponse::trial_aborted:
            ++trials_affected_;
            break;
        case AbnormalResponse::session_aborted:
            session_aborted_ = true;
            break;
        case AbnormalResponse::recorded:
        case AbnormalResponse::input_refused:
            ++recorded_only_;
            break;
        }
        if (static_cast<std::uint8_t>(policy) > static_cast<std::uint8_t>(primary_policy_) ||
            primary_ == AbnormalCondition::unspecified)
        {
            primary_policy_ = policy;
            primary_ = condition;
        }
        return event;
    }

    [[nodiscard]] AbnormalSummary summary() const noexcept
    {
        return AbnormalSummary{
            .observed = observed_,
            .trials_affected = trials_affected_,
            .recorded_only = recorded_only_,
            .session_aborted = session_aborted_,
            .primary = primary_,
            .primary_policy = primary_policy_,
        };
    }

    [[nodiscard]] const AbnormalPolicySet& policies() const noexcept
    {
        return policies_;
    }

  private:
    AbnormalPolicySet policies_{};
    ParadigmId paradigm_{kUnsetParadigmId};
    SequenceOrdinal sequence_{};
    std::uint64_t observed_{};
    std::uint64_t trials_affected_{};
    std::uint64_t recorded_only_{};
    bool session_aborted_{};
    AbnormalCondition primary_{AbnormalCondition::unspecified};
    AbnormalPolicy primary_policy_{AbnormalPolicy::record};
};

} // namespace neurale::execution
