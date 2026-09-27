/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <bit>
#include <cstdint>
#include <string_view>

#include <neurale/experiments/contract.h>
#include <neurale/experiments/identity.h>
#include <neurale/experiments/schedule.h>

/**
 * @file
 * @brief Replay anchors, and the verdict vocabulary a semantic replay reports in.
 *
 * A recorded trace says what happened. The two anchors say what the decisions
 * depended on, which is what makes them re-derivable rather than merely
 * re-readable. The rest of this header defines the shape of the answer a
 * deterministic semantic replay gives.
 *
 * # What a semantic replay is, and is not
 *
 * It re-executes the *paradigm's* decisions -- the state machine, the schedule
 * draws, the guidance and assistance transforms -- from recorded semantic
 * inputs, and checks that they come out the same. It does not replay the data
 * plane: recorded frames and discontinuities belong to the streaming layer's
 * own replay, and a discontinuity reaches this layer as the decision the run
 * took about it rather than as a stream event to be re-detected.
 *
 * Two facts therefore enter a replay as **inputs**, not as things to regenerate:
 *
 * - Anything the run decided against conditions that no longer exist. A gap in
 *   a decoded stream, a stale interval, an operator's stop. The stream is gone;
 *   what a replay verifies is that *given* those decisions, the same states,
 *   trials, and outcomes follow.
 * - Anything an external party reported. A presenter's actual display timing is
 *   the presentation renderer's measurement, and the paradigm layer cannot
 *   regenerate it from anything. It is evidence about a run, and a replay that
 *   "reproduced" it would be reproducing its own assumption.
 *
 * # Verdicts are not booleans
 *
 * A replay answers with a ReplayVerdict, and ReplayVerdict::match is only
 * one of four. Evidence that does not cover the run cannot produce it: the
 * whole failure mode this vocabulary exists to prevent is a recording missing
 * the outputs a replay would have disagreed with, and a boolean answer would
 * report that as success.
 */
namespace neurale::experiments
{

/// What a replay concluded.
///
/// Four values, and the reason there are four rather than two is that the two
/// interesting failures are not "the outputs differed". A recording that never
/// held the outputs, and a recording of a different experiment, are both
/// answerable and neither is a match.
enum class ReplayVerdict : std::uint8_t
{
    /// Every regenerated output agreed with the recorded one, over evidence
    /// that covered the run.
    match = 0,
    /// A regenerated output differed. ::ReplayReport::first_mismatch names the
    /// earliest one in replay order.
    mismatch,
    /// Everything compared agreed, and the evidence did not cover the run.
    ///
    /// Deliberately not a match. A recording missing the stream a replay would
    /// have disagreed with looks exactly like a recording of a run that agreed,
    /// and only this verdict distinguishes them.
    incomplete,
    /// Replay could not complete a semantic comparison.
    ///
    /// Identity and evidence rejection normally happen before comparison. An
    /// unsupported version may instead be discovered only after a verified
    /// prefix reaches the first operation that requires it.
    rejected,
};

/// Whether @p verdict is one of the declared ReplayVerdict values.
[[nodiscard]] constexpr bool replay_verdict_declared(ReplayVerdict verdict) noexcept
{
    return static_cast<std::uint8_t>(verdict) <= static_cast<std::uint8_t>(ReplayVerdict::rejected);
}

/// Why a replay could not complete a semantic comparison.
///
/// Identity disagreements are checked before the first input is replayed.
/// Unsupported semantics that are only exercised by a later input are reported
/// at the first step that requires them, before output from that step is
/// compared.
enum class ReplayRejection : std::uint8_t
{
    /// Nothing was rejected.
    none = 0,
    /// The recording names a different paradigm.
    paradigm_mismatch,
    /// The recording names a different version of this paradigm's records.
    experiment_version_mismatch,
    /// The configuration offered digests differently from the recorded one.
    configuration_fingerprint_mismatch,
    /// The schedule identity differs: seed, sampler, configuration, or catalog.
    schedule_fingerprint_mismatch,
    /// The seeds differ.
    seed_mismatch,
    /// The sampler versions differ.
    sampler_version_mismatch,
    /// The recorded sampler is one a paradigm must execute but this build
    /// cannot regenerate, and that paradigm has no authoritative realized
    /// schedule to use instead.
    ///
    /// Separate from ::sampler_version_mismatch because nothing disagrees: the
    /// recording is internally consistent and this build simply is not the one
    /// that can reproduce it. ScheduleIdentity says so too -- an unsupported
    /// version is legal, and ::ReplayAuthority::recorded_schedule is the route
    /// through it, which needs a recorded schedule to exist.
    sampler_version_unsupported,
    /// The realized schedules differ: two legal realizations of one seeded
    /// configuration are still two different runs.
    realized_schedule_mismatch,
    /// The metric versions differ, so derived metrics are not comparable.
    metric_version_mismatch,
    /// The policy versions differ -- for Center-Out, the assistance transform's.
    policy_version_mismatch,
    /// The provenance itself is unusable: an unset paradigm, or no sampler.
    provenance_incomplete,
    /// The configuration handed to the replay is not a configuration this
    /// paradigm can run.
    configuration_invalid,
    /// The recorded inputs are not a timeline this paradigm could have consumed.
    evidence_invalid,
    /// The configured WebGrid metric formula is not implemented by this build.
    metric_version_unsupported,
};

/// Whether @p rejection is one of the declared ReplayRejection values.
[[nodiscard]] constexpr bool replay_rejection_declared(ReplayRejection rejection) noexcept
{
    return static_cast<std::uint8_t>(rejection) <=
           static_cast<std::uint8_t>(ReplayRejection::metric_version_unsupported);
}

/// Which recorded stream an item belongs to.
///
/// Named rather than numbered because a mismatch report is read by a person
/// deciding whether a run is trustworthy, and "transition 41 differs" is the
/// smallest sentence that helps them.
enum class ReplayItem : std::uint8_t
{
    /// Unset.
    none = 0,
    /// A state transition.
    transition,
    /// A semantic event.
    event,
    /// A decided trial.
    trial,
    /// A target onset.
    target,
    /// A cursor position and the interval it moved over.
    cursor,
    /// An assisted velocity and its inputs.
    assisted_velocity,
    /// A guidance sample.
    guidance_sample,
    /// A selection.
    selection,
    /// A derived metric set.
    metrics,
    /// An intended phase of a Speech trial.
    phase,
    /// A presentation request the paradigm made.
    presentation_request,
    /// A realized per-trial schedule.
    schedule,
    /// The number of items a stream held.
    ///
    /// A recording holding more of something than the replay produced -- or
    /// fewer -- differs in a way no per-item comparison would reach, because
    /// there is no item to compare it against.
    stream_length,
};

/// Whether @p item is one of the declared ReplayItem values.
[[nodiscard]] constexpr bool replay_item_declared(ReplayItem item) noexcept
{
    return static_cast<std::uint8_t>(item) <= static_cast<std::uint8_t>(ReplayItem::stream_length);
}

/// What was missing from the evidence.
///
/// A replay reports the *first* of these it meets, in the order the enumeration
/// declares, so two recordings incomplete in the same way answer the same way.
enum class ReplayCompleteness : std::uint8_t
{
    /// The evidence covers the run from its first input to its recorded end.
    complete = 0,
    /// An output stream this paradigm produces is absent from the evidence.
    ///
    /// Not the same as an empty one. A run that produced no selection and a
    /// recording that did not keep selections are different facts, and only a
    /// reader that distinguishes them can be told the difference.
    stream_absent,
    /// The recording says trace records were lost.
    ///
    /// Whatever else is present, a stream with a hole in it cannot establish
    /// that the run agreed: the records a replay would have disagreed with are
    /// exactly the ones that might be missing.
    trace_loss_recorded,
    /// The evidence ends before the run did.
    run_end_missing,
};

/// Whether @p completeness is one of the declared ReplayCompleteness values.
[[nodiscard]] constexpr bool replay_completeness_declared(ReplayCompleteness completeness) noexcept
{
    return static_cast<std::uint8_t>(completeness) <=
           static_cast<std::uint8_t>(ReplayCompleteness::run_end_missing);
}

/// What a replay does when the evidence does not cover the run.
enum class ReplayIncompletePolicy : std::uint8_t
{
    /// Replay and compare what the evidence does cover, and answer
    /// ::ReplayVerdict::incomplete however well the prefix agreed.
    ///
    /// The default. A partial verification is worth having, and the verdict is
    /// what stops it being read as a whole one.
    verify_available = 0,
    /// Compare nothing and answer ::ReplayVerdict::incomplete immediately.
    ///
    /// For a caller whose question is "is this recording replayable", where a
    /// prefix that agrees is not an answer.
    refuse,
};

/// Whether @p policy is one of the declared ReplayIncompletePolicy values.
[[nodiscard]] constexpr bool
replay_incomplete_policy_declared(ReplayIncompletePolicy policy) noexcept
{
    return static_cast<std::uint8_t>(policy) <=
           static_cast<std::uint8_t>(ReplayIncompletePolicy::refuse);
}

/// The one output a replay found differing, and where.
///
/// Values travel as both an integer and a real, with `real_valued` saying which
/// one to read. A single pair of doubles would round a 64-bit identity on its
/// way into a report about identities not matching.
struct ReplayMismatch
{
    /// Stream the differing item belongs to.
    ReplayItem item{ReplayItem::none};
    /// Index of the item within its own stream, counted from zero.
    std::uint64_t idx{};
    /// Experiment instant the *recorded* item carries.
    ExperimentTimeNs time_ns{};
    /// Trial the recorded item belongs to.
    TrialIdentity trial{};
    /// Which field differed.
    ///
    /// A view of a string literal with static storage duration. It is never
    /// owned, never allocated, and outlives every report that carries it.
    std::string_view field{};
    /// Recorded value, for an integral field.
    std::uint64_t recorded{};
    /// Regenerated value, for an integral field.
    std::uint64_t regenerated{};
    /// Recorded value, for a real field.
    double recorded_real{};
    /// Regenerated value, for a real field.
    double regenerated_real{};
    /// Whether the real pair, rather than the integral pair, is the value.
    bool real_valued{};
};

/// What one replay established.
struct ReplayReport
{
    /// The answer.
    ///
    /// Defaults to ::ReplayVerdict::rejected: a report nobody filled in has not
    /// verified anything, and a default of `match` would make forgetting to run
    /// the replay look like running it.
    ReplayVerdict verdict{ReplayVerdict::rejected};
    /// Why semantic comparison could not complete, when rejected.
    ReplayRejection rejection{ReplayRejection::none};
    /// What the evidence lacked.
    ReplayCompleteness completeness{ReplayCompleteness::complete};
    /// The earliest differing item, when one differed.
    ReplayMismatch first_mismatch{};
    /// Recorded semantic inputs consumed.
    std::uint64_t inputs_replayed{};
    /// Individual output items compared.
    std::uint64_t items_compared{};
    /// Fields compared across those items.
    ///
    /// Reported because "nothing differed" over two fields and over two hundred
    /// are different statements, and a report that only counted items would
    /// make a shallow comparison look like a deep one.
    std::uint64_t fields_compared{};
};

/// Whether @p report is a verified reproduction.
///
/// One function rather than a comparison every caller writes, so that a caller
/// cannot accidentally spell it `verdict != mismatch` -- which is true for a
/// rejected replay (including after a verified prefix) and for an incomplete
/// one.
[[nodiscard]] constexpr bool replay_reproduced(const ReplayReport& report) noexcept
{
    return report.verdict == ReplayVerdict::match;
}

/// Stable name of @p verdict, for a report a person reads.
[[nodiscard]] const char* replay_verdict_name(ReplayVerdict verdict) noexcept;
/// Stable name of @p rejection.
[[nodiscard]] const char* replay_rejection_name(ReplayRejection rejection) noexcept;
/// Stable name of @p item.
[[nodiscard]] const char* replay_item_name(ReplayItem item) noexcept;
/// Stable name of @p completeness.
[[nodiscard]] const char* replay_completeness_name(ReplayCompleteness completeness) noexcept;

/// Everything a replay checks before it regenerates anything.
///
/// The recorded side comes out of the recording's session metadata; the
/// regenerated side is computed from the configuration the caller is about to
/// replay with. They are compared field by field in the order
/// check_provenance() declares, so two callers holding the same disagreement
/// are told about the same one.
struct ReplayProvenance
{
    /// Paradigm the run was recorded under.
    ParadigmId paradigm{kUnsetParadigmId};
    /// Version of the paradigm's own record layout.
    std::uint32_t experiment_version{};
    /// Digest of the immutable configuration.
    std::uint64_t configuration_fingerprint{};
    /// Digest of the schedule identity.
    std::uint64_t schedule_fingerprint{};
    /// Digest of the realized schedule, or zero for a paradigm that freezes none.
    std::uint64_t realized_schedule_fingerprint{};
    /// Schedule seed.
    ScheduleSeed seed{};
    /// Sampler that produced the run's randomized values.
    SamplerVersion sampler_version{};
    /// Version of the derived metric formula, or zero where there is none.
    std::uint32_t metric_version{};
    /// Version of the paradigm's own policy -- for Center-Out, the assistance
    /// transform; for Speech, the stimulus order.
    std::uint32_t policy_version{};
};

/// Whether the two provenances describe the same experiment.
///
/// @param recorded  What the recording says the run was.
/// @param candidate What the replay is about to regenerate.
/// This function checks identity agreement only. Whether the named sampler must
/// actually be executed is a paradigm/schedule-policy decision made by the
/// replay engine that knows which values need drawing.
///
/// @return ::ReplayRejection::none when the identities agree, and otherwise the
///         first disagreement in declaration order.
[[nodiscard]] ReplayRejection check_provenance(const ReplayProvenance& recorded,
                                               const ReplayProvenance& candidate) noexcept;

/// Accumulates one replay's findings, keeping the first mismatch and nothing else.
///
/// The engines are streaming: they regenerate one step at a time and compare it
/// against the recorded evidence immediately, so nothing here stores an output.
/// What it stores is a latch, and the latch is the whole reason this is one
/// class rather than three copies of the same `if (first) { ... }`.
///
/// Real fields are compared by their bit patterns, not by `==`. A replay is a
/// determinism claim, and a determinism claim that admitted two different
/// doubles as equal would be claiming something weaker than it says. Recorded
/// doubles round-trip exactly -- they are written in the shortest form that
/// reads back unchanged -- so the strict comparison is one a correct recording
/// passes.
class ReplayComparator
{
  public:
    ReplayComparator() noexcept = default;

    /// Name the item the following field comparisons belong to.
    void begin(ReplayItem item, std::uint64_t idx, ExperimentTimeNs time_ns,
               const TrialIdentity& trial) noexcept;

    /// Compare one integral field.
    void integer(std::string_view field, std::uint64_t recorded,
                 std::uint64_t regenerated) noexcept;
    /// Compare one real field, by bit pattern.
    void real(std::string_view field, double recorded, double regenerated) noexcept;
    /// Compare one boolean field.
    void flag(std::string_view field, bool recorded, bool regenerated) noexcept;
    /// Compare a whole trial identity, field by field.
    void identity(std::string_view field, const TrialIdentity& recorded,
                  const TrialIdentity& regenerated) noexcept;

    /// Whether a mismatch has been latched. Comparison may stop here.
    [[nodiscard]] bool differed() const noexcept
    {
        return differed_;
    }
    /// The first mismatch, meaningful only when differed().
    [[nodiscard]] const ReplayMismatch& first_mismatch() const noexcept
    {
        return first_;
    }
    /// Items begun.
    [[nodiscard]] std::uint64_t items_compared() const noexcept
    {
        return items_;
    }
    /// Fields compared.
    [[nodiscard]] std::uint64_t fields_compared() const noexcept
    {
        return fields_;
    }

  private:
    void latch(std::string_view field) noexcept;

    ReplayMismatch first_{};
    ReplayItem item_{ReplayItem::none};
    std::uint64_t idx_{};
    ExperimentTimeNs time_ns_{};
    TrialIdentity trial_{};
    std::uint64_t items_{};
    std::uint64_t fields_{};
    bool differed_{};
};

/// Session-scoped replay anchor, persisted once before the first trial.
///
/// Everything a reader needs in order to regenerate the session's randomized
/// values, and to place its ordinals: a session that resumed from a nonzero
/// ordinal is unreadable without knowing where it started.
struct ExperimentSnapshot
{
    /// Experiment time the session's own timeline starts from.
    ExperimentTimeNs origin_ns{};
    /// Schedule identity, including seed and sampler version.
    ScheduleIdentity schedule{};
    /// First trial ordinal this session will issue.
    TrialOrdinal first_trial_ordinal{};
    /// First emission ordinal this session will issue.
    SequenceOrdinal first_sequence{};
    /// First block ordinal this session will issue.
    BlockOrdinal first_block{};
    /// Paradigm that owns the session.
    ParadigmId paradigm{kUnsetParadigmId};
};

/// Replay anchor for the randomized state one semantic decision was taken in.
///
/// It pins where the decision sat in the schedule, so the values it drew are
/// regenerable without replaying every draw before it: `draw_cursor` says how
/// many draws the trial had consumed, which makes the next DrawKey
/// reconstructible.
///
/// It is deliberately *not* the complete input to a decision. It carries no
/// cursor position, no decoded velocity, no selection input, no assistance
/// input, and no external event; those belong to the paradigm that owns the
/// decision, and a paradigm that needs them replayable records them itself. A
/// reader holding only this record can re-derive the random draws, not rerun
/// the state machine.
struct DecisionSnapshot
{
    /// Experiment time the decision was taken.
    ExperimentTimeNs time_ns{};
    /// Emission ordinal within the session.
    SequenceOrdinal sequence{};
    /// Trial the decision belongs to.
    TrialIdentity trial{};
    /// Schedule the decision sampled from.
    ScheduleIdentity schedule{};
    /// Draw sequence the cursor refers to.
    DrawStream stream{};
    /// Draws consumed in `(stream, trial)` before this decision.
    std::uint32_t draw_cursor{};
    /// Paradigm whose state enumeration `state` belongs to.
    ParadigmId paradigm{kUnsetParadigmId};
    /// Paradigm state at the moment of the decision.
    StateId state{};
};

/// Draw coordinates the next draw of @p snapshot would use.
[[nodiscard]] constexpr DrawKey next_draw_key(const DecisionSnapshot& snapshot) noexcept
{
    return DrawKey{snapshot.schedule.seed, snapshot.stream, snapshot.trial.ordinal,
                   snapshot.draw_cursor};
}

/// Validate a session replay anchor.
[[nodiscard]] ContractStatus validate(const ExperimentSnapshot& snapshot) noexcept;

/// Validate a decision replay anchor.
[[nodiscard]] ContractStatus validate(const DecisionSnapshot& snapshot) noexcept;

} // namespace neurale::experiments
