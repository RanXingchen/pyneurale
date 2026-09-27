// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <neurale/experiments/contract.h>
#include <neurale/experiments/events.h>
#include <neurale/experiments/identity.h>
#include <neurale/experiments/presentation.h>
#include <neurale/experiments/schedule.h>
#include <neurale/experiments/time.h>

/**
 * @file
 * @brief Speech cue configuration, stimulus catalog, timing schedule, and state machine.
 *
 * The Speech paradigm is a timed cue paradigm. One trial presents a blank
 * field, optionally a fixation cross, and then textual content:
 *
 * ```text
 * BLACK -> CROSS (optional, configuration-controlled) -> CONTENT
 * ```
 *
 * This header owns what a Speech session *is*: its immutable configuration, the
 * immutable stimulus catalog, the deterministic per-trial schedule, and the
 * intended phase timeline that follows from one realized schedule, plus the
 * explicit-time state machine. It owns no clock, renderer, microphone, audio, or
 * decoder. Nothing here presents anything, and nothing here decodes speech.
 *
 * # Whether there is a cross is configuration, never a draw
 *
 * SpeechCueConfig::cross_enabled decides it. When it is set, the cross duration
 * is sampled from an open range and is therefore *strictly positive*; when it
 * is clear, no cross draw is made at all, the realized
 * SpeechTrialSchedule::cross_duration_ns is exactly zero, and the timeline has
 * no cross phase in it -- not a cross phase of no length.
 *
 * That is what makes a recorded zero unambiguous: it always means "no cross
 * phase," never "an unlucky draw of length zero."
 *
 * # The ranges are open at both ends
 *
 * Every sampled duration lies strictly inside its configured bound: it is never
 * zero, and it never equals the bound. In integer nanoseconds that is
 *
 * ```text
 * black_duration_ns   in [1, t1 - 1]
 * cross_duration_ns   in [1, t2 - 1]   when cross_enabled, and 0 otherwise
 * content_duration_ns in [1, t3 - 1]
 * ```
 *
 * so a bound is a value the timeline approaches and never realizes. A bound
 * that admits no such integer is rejected by validate(const SpeechCueConfig&)
 * as ContractStatus::range_empty rather than clamped: a session that quietly
 * ran with a phase length nobody configured is not the session that was asked
 * for.
 *
 * # Determinism
 *
 * Every draw is addressed rather than advanced, through the shared stateless
 * sampler in `schedule.h`. A trial's values are a pure function of
 *
 * ```text
 * (seed, stream, trial ordinal, draw ordinal, sampler version)
 * ```
 *
 * with no global state, no `std::random_device`, no `<random>` distribution,
 * and no host entropy anywhere. Preparing trial 900 does not require having
 * prepared trials 0 to 899, so a replay reconstructs any trial on its own and a
 * long session costs no memory at all.
 *
 * The draw ordinals within kTrialTimingStream are fixed per phase --
 * kBlackDraw, kCrossDraw, kContentDraw -- rather than allocated in the
 * order the draws happen to be made. Toggling SpeechCueConfig::cross_enabled
 * therefore changes only whether the cross value is realized, and leaves every
 * black and content duration in the session exactly where it was.
 *
 * # Two authorities, and only ever one at a time
 *
 * A seeded configuration regenerates its schedule. An explicit configuration
 * carries the realized schedule itself, sampling nothing. Fields that would
 * change a realized schedule are required to be *absent* on the side that does
 * not decide it -- an explicit configuration carries no stimulus set, no
 * ordering policy, and no cross bound when the cross is off -- so a reader
 * never has to work out which of two disagreeing answers was used. The seed is
 * the one exception, and deliberately: it identifies where a frozen schedule
 * came from, and identifying provenance is not deciding a value.
 */
namespace neurale::experiments::speech
{
/// Largest stimulus catalog, and largest ordered stimulus set, this build holds.
///
/// A fixed capacity keeps the catalog a trivially copyable value that can be
/// compared and fingerprinted without owning storage, which is what lets a
/// session's catalog identity be persisted beside its seed.
inline constexpr std::size_t kMaxSpeechStimuli = 128;

/// Largest explicit schedule this build carries inside a configuration.
///
/// An explicit schedule is for a frozen block -- a calibration set, or a
/// recorded session being replayed exactly. A long session is described by a
/// seed instead, and a seeded session has no length limit at all.
inline constexpr std::size_t kMaxSpeechExplicitTrials = 128;

/// Largest stimulus text this build stores, in bytes.
inline constexpr std::size_t kMaxSpeechTextBytes = 64;

/// Phases one realized trial can hold: black, cross, content, and the gap.
inline constexpr std::size_t kMaxSpeechPhases = 4;

/// Draw stream the per-trial phase durations are sampled from.
///
/// Stream tags are paradigm-local, and the paradigms do not share a seed:
/// Center-Out uses 1 and WebGrid uses 2 under their own seeds, and a session
/// running more than one paradigm gives each its own ::ScheduleSeed rather than
/// trying to partition one stream space between them.
inline constexpr DrawStream kTrialTimingStream = 3;

/// Draw stream the stimulus order is sampled from.
inline constexpr DrawStream kStimulusOrderStream = 4;

/// Draw ordinal of the black duration within ::kTrialTimingStream.
inline constexpr std::uint32_t kBlackDraw = 0;

/// Draw ordinal of the cross duration within ::kTrialTimingStream.
///
/// Reserved whether or not the cross is enabled. Nothing reads this coordinate
/// when it is disabled, and nothing else is ever given it either.
inline constexpr std::uint32_t kCrossDraw = 1;

/// Draw ordinal of the content duration within ::kTrialTimingStream.
inline constexpr std::uint32_t kContentDraw = 2;

/// Unset semantic class label.
inline constexpr std::uint32_t kUnsetSpeechLabel = 0;

/// Semantic content type of one stimulus.
///
/// Deliberately narrow. Version 1 presents text and nothing else: images,
/// audio, and phoneme-specific presentation are later extensions that add an
/// enumerator and the fields it needs, not something a caller smuggles through
/// an opaque payload. The enumeration is append-only.
enum class SpeechContentKind : std::uint8_t
{
    /// Unset. Rejected by validate(const SpeechStimulus&).
    unspecified = 0,
    /// A text prompt, held in SpeechStimulus::text.
    text,
};

/// Whether @p value is one of the declared SpeechContentKind values.
[[nodiscard]] constexpr bool speech_content_kind_declared(SpeechContentKind value) noexcept
{
    return static_cast<std::uint8_t>(value) <= static_cast<std::uint8_t>(SpeechContentKind::text);
}

/// Where a session's per-trial schedule comes from.
enum class SpeechScheduleKind : std::uint8_t
{
    /// Unset. Rejected by validate(const SpeechCueConfig&); the two below are
    /// different experiments and picking one silently would decide that for the
    /// caller.
    unspecified = 0,
    /// Regenerated from the seed, the trial ordinal, and the sampler version.
    seeded,
    /// Carried verbatim in SpeechCueConfig::explicit_schedule. Nothing is sampled.
    explicit_sequence,
};

/// Whether @p value is one of the declared SpeechScheduleKind values.
[[nodiscard]] constexpr bool speech_schedule_kind_declared(SpeechScheduleKind value) noexcept
{
    return static_cast<std::uint8_t>(value) <=
           static_cast<std::uint8_t>(SpeechScheduleKind::explicit_sequence);
}

/// How each trial's stimulus is drawn from the ordered stimulus set.
///
/// This is the repetition policy: how often a session revisits a stimulus, and
/// whether it is allowed to repeat one before showing the others.
enum class StimulusOrderPolicy : std::uint8_t
{
    /// Unset. Required to be this, and only this, for an explicit schedule,
    /// which names each trial's stimulus itself.
    unspecified = 0,
    /// The set in order, wrapping: trial `i` presents entry `i % count`.
    ///
    /// Every stimulus appears equally often and the order never varies, which
    /// is what a fixed calibration block wants and what a psychophysics block
    /// does not.
    sequential,
    /// Independently sampled per trial, uniformly and with replacement.
    ///
    /// A stimulus may follow itself, and over a short session the counts are
    /// not balanced. That is what "with replacement" means, and it is stated
    /// here because a caller expecting balance would not notice the difference
    /// until the analysis.
    random_with_replacement,
    /// A fresh seeded permutation of the whole set every `count` trials.
    ///
    /// Within one block each stimulus appears exactly once, so the counts are
    /// balanced at every block boundary. Two consecutive trials across a block
    /// boundary may still present the same stimulus: the permutations are
    /// independent, and forbidding that would make them non-uniform.
    shuffled_blocks,
};

/// Whether @p value is one of the declared StimulusOrderPolicy values.
[[nodiscard]] constexpr bool stimulus_order_policy_declared(StimulusOrderPolicy value) noexcept
{
    return static_cast<std::uint8_t>(value) <=
           static_cast<std::uint8_t>(StimulusOrderPolicy::shuffled_blocks);
}

/// One phase of a realized trial.
///
/// A paradigm-owned enumeration, widened into ::PhaseId for storage in the
/// shared presentation records.
enum class SpeechPhase : std::uint8_t
{
    /// A blank field.
    black = 0,
    /// A fixation cross. Present only when the cross is enabled.
    cross,
    /// The textual stimulus content.
    content,
    /// The gap before the next trial's black phase. Present only when
    /// SpeechCueConfig::inter_trial_ns is nonzero.
    inter_trial,
};

/// Whether @p value is one of the declared SpeechPhase values.
[[nodiscard]] constexpr bool speech_phase_declared(SpeechPhase value) noexcept
{
    return static_cast<std::uint8_t>(value) <= static_cast<std::uint8_t>(SpeechPhase::inter_trial);
}

/// One catalog entry: what a ::StimulusId means.
///
/// The hot path and every persisted runtime record carry the identifier alone.
/// The content it stands for is resolved here, outside the hot path, which is
/// why this record may hold text at all.
struct SpeechStimulus
{
    /// Identifier the runtime records carry. ::kUnsetStimulusId is rejected.
    StimulusId id{kUnsetStimulusId};
    /// Semantic class or label, or ::kUnsetSpeechLabel.
    ///
    /// Optional on purpose: a free-recall prompt has no class, and a zero that
    /// meant "class zero" would make the two indistinguishable.
    std::uint32_t label{kUnsetSpeechLabel};
    /// Opaque application-owned tag, stored and fingerprinted, never interpreted.
    ///
    /// Small and immutable by construction, and *not* an extension point: it is
    /// a fixed-width integer precisely so that it cannot grow into an arbitrary
    /// object payload. A stimulus type that needs more than an integer needs a
    /// SpeechContentKind enumerator and its own fields.
    std::uint64_t metadata{};
    /// What kind of content this entry holds.
    SpeechContentKind content{SpeechContentKind::unspecified};
    /// Bytes of `text` that are in use.
    std::uint8_t text_length{};
    /// The prompt, as canonical UTF-8 bytes. Not NUL-terminated; `text_length`
    /// is the length in bytes.
    ///
    /// The contract imposes exactly one encoding -- UTF-8 -- and validates it:
    /// the first `text_length` bytes must form a complete, well-formed UTF-8
    /// sequence, with no overlong form, no lone surrogate, and no truncated
    /// continuation. A task that presents a phrase in one script and one that
    /// presents it in another are still the same task here; what they share is a
    /// UTF-8 byte string, and a presenter shapes that string rather than
    /// re-encoding it. Bytes beyond `text_length` are required to be zero, so
    /// that two entries with the same prompt have the same bytes and therefore
    /// the same fingerprint.
    std::array<char, kMaxSpeechTextBytes> text{};
};

/// The immutable semantic stimulus catalog of one session.
///
/// Prepared before a session starts and unchanged during one. It is owned here
/// and not by a renderer: a recorded session carrying identifiers whose meaning
/// lived only in the presentation layer would not be reproducible.
struct SpeechCatalog
{
    /// Entries in use.
    std::uint16_t count{};
    /// The entries. Slots at or beyond `count` are required to be default.
    std::array<SpeechStimulus, kMaxSpeechStimuli> entries{};
};

/// One trial's realized schedule: the replay authority for that trial.
///
/// Every field is a value that was decided before the trial began. The state
/// machine that will run this trial samples nothing.
struct SpeechTrialSchedule
{
    /// Session-local trial ordinal this schedule belongs to.
    TrialOrdinal ordinal{};
    /// Realized black duration. Strictly positive.
    DurationNs black_duration_ns{};
    /// Realized cross duration. Strictly positive when `cross_enabled`, and
    /// exactly zero otherwise, where it means "there was no cross phase" and
    /// can mean nothing else.
    DurationNs cross_duration_ns{};
    /// Realized content duration. Strictly positive.
    DurationNs content_duration_ns{};
    /// Stimulus presented during the content phase.
    StimulusId stimulus_id{kUnsetStimulusId};
    /// Sampler that produced the durations, recorded so the entry is readable
    /// on its own rather than only beside the configuration it came from.
    ///
    /// Zero -- unset -- by default rather than the current sampler, because the
    /// default value of this record is also the required value of every unused
    /// slot of SpeechCueConfig::explicit_schedule, and those slots are absorbed
    /// into configuration_fingerprint(). A padding default that tracked the
    /// build's current sampler would move the fingerprint of an unchanged
    /// configuration the first time that constant is bumped. A realized
    /// schedule always carries a version: prepare_trial() writes the
    /// configuration's, and an explicit entry is required to state it.
    SamplerVersion sampler_version{};
    /// Whether this trial has a cross phase. Configuration, never a draw.
    bool cross_enabled{};
};

/// One phase of a built timeline.
struct SpeechPhaseInterval
{
    /// Half-open `[start_ns, end_ns)`. The exact end instant belongs to the
    /// following phase and never to both.
    TimeInterval interval{};
    /// Which phase this is.
    SpeechPhase phase{SpeechPhase::black};
    /// What is semantically presented during it.
    CueKind cue{CueKind::none};
    /// Stimulus to resolve, or ::kUnsetStimulusId when the cue needs none.
    StimulusId stimulus_id{kUnsetStimulusId};
};

/// The intended phase timeline of one realized trial.
///
/// Intended, not observed: these are the instants the paradigm decided on, and
/// nothing here reports when anything appeared on a display.
///
/// A disabled cross and a zero inter-trial gap are *absent* from `phases`
/// rather than present with an empty interval, which is what makes `count` the
/// answer to "which phases does this trial have".
struct SpeechTimeline
{
    /// Phases in presentation order.
    std::array<SpeechPhaseInterval, kMaxSpeechPhases> phases{};
    /// The trial itself: black start to content end, excluding any gap.
    TimeInterval trial{};
    /// First instant of the next trial's black phase.
    ///
    /// Equal to `trial.end_ns` when there is no gap. This is the value a caller
    /// chains trials with, rather than re-deriving the gap at each call site.
    ExperimentTimeNs next_trial_start_ns{};
    /// Entries of `phases` in use: two, three, or four.
    std::uint8_t count{};
};

/// One trial's realized durations, as shared replay provenance.
///
/// The same values as the SpeechTrialSchedule they came from, in the shared
/// ScheduleDraw vocabulary, so a reader that knows nothing about this
/// paradigm can still check a session's draws against a regeneration.
struct SpeechTrialDraws
{
    /// The records, in phase order.
    std::array<ScheduleDraw, 3> draws{};
    /// Records in use: two when the cross is disabled, three when it is enabled.
    std::uint8_t count{};
};

/// Immutable Speech cue v1 session configuration.
///
/// The three bounds are exclusive upper bounds on the sampled phase durations,
/// named `t1`, `t2` and `t3` in the requirements. They are validated whichever
/// schedule kind is in use: an explicit schedule is checked against them too,
/// so a supplied schedule is the same experiment as the seeded one it replaces
/// rather than an unrelated set of numbers wearing its configuration.
struct SpeechCueConfig
{
    /// Exclusive upper bound `t1` on the black duration. At least two
    /// nanoseconds, so that the open range holds an integer.
    DurationNs black_bound_ns{};
    /// Exclusive upper bound `t2` on the cross duration.
    ///
    /// Required to be zero when `cross_enabled` is clear. A bound naming a
    /// sampling domain that nothing samples is a field no reader can interpret,
    /// and leaving one behind is how a session ends up believed to have had a
    /// cross phase it never presented.
    DurationNs cross_bound_ns{};
    /// Exclusive upper bound `t3` on the content duration.
    DurationNs content_bound_ns{};
    /// Gap between the end of one trial's content and the next trial's black.
    ///
    /// Exact rather than sampled, and separate from the black phase: black is a
    /// task phase with a randomized length, and the gap is the space between
    /// trials. Zero means the next trial's black phase begins immediately, and
    /// the timeline then has no inter-trial entry at all.
    DurationNs inter_trial_ns{};
    /// Seed of this session's schedule.
    ///
    /// Provenance for an explicit schedule, which regenerates nothing: it
    /// records where a frozen schedule came from, and is not required to be
    /// zero there.
    ScheduleSeed seed{};
    /// Trials in the session. Strictly positive.
    TrialOrdinal n_trials{};
    /// Where the schedule comes from.
    SpeechScheduleKind schedule{SpeechScheduleKind::unspecified};
    /// How the stimulus of each trial is drawn from `stimuli`.
    ///
    /// Required to be StimulusOrderPolicy::unspecified for an explicit
    /// schedule, which decides each trial's stimulus itself.
    StimulusOrderPolicy stimulus_order{StimulusOrderPolicy::unspecified};
    /// Sampler the schedule's draws are made with.
    SamplerVersion sampler_version{kCurrentSamplerVersion};
    /// Whether trials have a cross phase. Never a random outcome.
    bool cross_enabled{};
    /// Ordered stimulus set the seeded schedule draws from.
    ///
    /// Required to be empty for an explicit schedule. Order matters: it is the
    /// sequence StimulusOrderPolicy::sequential steps through and the domain
    /// the permutation is taken over.
    std::uint16_t n_stimuli{};
    /// The set, in schedule order. Slots at or beyond `stimulus_count` are
    /// required to be ::kUnsetStimulusId.
    std::array<StimulusId, kMaxSpeechStimuli> stimuli{};
    /// Entries of `explicit_schedule` in use. Zero for a seeded schedule.
    std::uint16_t n_explicit{};
    /// The frozen schedule, one entry per trial, in trial order.
    std::array<SpeechTrialSchedule, kMaxSpeechExplicitTrials> explicit_schedule{};
};

/// Validate one catalog entry.
///
/// @return ContractStatus::ok, ContractStatus::identity_missing for an unset
///         identifier or an empty text, ContractStatus::enum_undeclared,
///         ContractStatus::parameter_out_of_range for a length beyond
///         ::kMaxSpeechTextBytes, or ContractStatus::outcome_invalid when the
///         bytes beyond the length are not zero or the in-use bytes are not a
///         complete, well-formed UTF-8 sequence.
[[nodiscard]] ContractStatus validate(const SpeechStimulus& stimulus) noexcept;

/// Validate a stimulus catalog.
///
/// @return ContractStatus::ok, ContractStatus::target_set_invalid when the
///         catalog is empty, oversized, names one identifier twice, or holds a
///         non-default slot beyond its count, or whatever
///         validate(const SpeechStimulus&) reported for an entry.
[[nodiscard]] ContractStatus validate(const SpeechCatalog& catalog) noexcept;

/// Validate a realized trial schedule on its own.
///
/// Structural only: the bounds a duration was drawn against live in the
/// configuration, and are checked by
/// validate_against(const SpeechTrialSchedule&, const SpeechCueConfig&).
///
/// @return ContractStatus::ok, ContractStatus::identity_missing for an unset
///         stimulus or an absent sampler version,
///         ContractStatus::parameter_out_of_range for a black or content
///         duration of zero, or ContractStatus::outcome_invalid when the cross
///         duration contradicts `cross_enabled`.
[[nodiscard]] ContractStatus validate(const SpeechTrialSchedule& schedule) noexcept;

/// Validate a Speech configuration.
///
/// @return ContractStatus::ok, ContractStatus::enum_undeclared,
///         ContractStatus::identity_missing for an absent sampler version or an
///         unset schedule kind, ContractStatus::range_empty for a timing bound
///         admitting no duration or a zero trial count,
///         ContractStatus::target_set_invalid for a malformed stimulus set,
///         ContractStatus::outcome_invalid when a field that decides a realized
///         schedule is set on the side that does not decide it, or whatever
///         validate(const SpeechTrialSchedule&) reported for an explicit entry.
[[nodiscard]] ContractStatus validate(const SpeechCueConfig& config) noexcept;

/// Check a realized schedule against the configuration it claims to belong to.
///
/// This is what an externally supplied schedule is admitted by, and it is
/// deliberately an ownership check rather than a plausibility check. Being
/// individually well-formed and inside the bounds is not evidence that a
/// schedule came from this session, so on top of that:
///
/// - the sampler version must be the configuration's, because
///   ScheduleIdentity::sampler_version is taken from the configuration and a
///   session cannot name two samplers;
/// - under SpeechScheduleKind::seeded the stimulus must be one the
///   configuration can present, so a schedule cannot introduce a stimulus the
///   deterministic order would never have selected;
/// - under SpeechScheduleKind::explicit_sequence the schedule must equal
///   `config.explicit_schedule[schedule.ordinal]` field for field, because that
///   entry *is* the replay authority for the trial and an alternative that
///   merely fits the bounds is a second answer to a question already answered.
///
/// What it does not do is re-run the sampler to confirm a seeded schedule's
/// realized values. Admitting a schedule and verifying that a recorded session
/// reproduces are different questions, and the second one belongs to replay
/// verification rather than to the gate every timeline passes through.
///
/// @return ContractStatus::ok, ContractStatus::outcome_invalid when
///         `cross_enabled` or the sampler version disagrees with the
///         configuration or an explicit schedule differs from the entry it
///         claims to be, ContractStatus::range_empty when the ordinal names no
///         trial of the session, ContractStatus::parameter_out_of_range when a
///         duration reaches or passes its bound,
///         ContractStatus::target_set_invalid when a seeded schedule names a
///         stimulus outside the configured set, or whatever either validate()
///         reported.
[[nodiscard]] ContractStatus validate_against(const SpeechTrialSchedule& schedule,
                                              const SpeechCueConfig& config) noexcept;

/// Check that every stimulus a configuration can present is in the catalog.
///
/// @return ContractStatus::ok, ContractStatus::identity_missing when a
///         referenced identifier is absent from the catalog, or whatever either
///         validate() reported.
[[nodiscard]] ContractStatus validate_against(const SpeechCueConfig& config,
                                              const SpeechCatalog& catalog) noexcept;

/// Resolve one catalog entry.
///
/// @param catalog Prepared catalog.
/// @param id Identifier to resolve.
/// @param stimulus Receives the entry on ContractStatus::ok, and is left
///                 unmodified otherwise.
/// @return ContractStatus::ok, ContractStatus::identity_missing when the
///         catalog holds no such entry, or whatever validate() reported.
[[nodiscard]] ContractStatus find_stimulus(const SpeechCatalog& catalog, StimulusId id,
                                           SpeechStimulus& stimulus) noexcept;

/// Resolve the stimulus of trial @p ordinal.
///
/// @param config Session configuration.
/// @param ordinal Session-local trial ordinal.
/// @param stimulus_id Receives the identifier on ContractStatus::ok.
/// @return ContractStatus::ok, ContractStatus::range_empty when the session has
///         no such trial, ContractStatus::version_unsupported when a seeded
///         draw is needed and this build cannot make it, or whatever
///         validate(const SpeechCueConfig&) reported.
[[nodiscard]] ContractStatus select_stimulus(const SpeechCueConfig& config, TrialOrdinal ordinal,
                                             StimulusId& stimulus_id) noexcept;

/// Realize the schedule of trial @p ordinal.
///
/// For a seeded configuration this samples the enabled durations at their fixed
/// draw coordinates; for an explicit one it returns the frozen entry. Either
/// way the result is the same pure function of the configuration and the
/// ordinal, so calling it twice, out of order, or on another machine gives the
/// same answer, and preparing one trial does not require having prepared the
/// ones before it.
///
/// @param config Session configuration.
/// @param ordinal Session-local trial ordinal.
/// @param schedule Receives the realized schedule on ContractStatus::ok, and is
///                 left unmodified otherwise.
/// @return ContractStatus::ok, ContractStatus::range_empty when the session has
///         no such trial, ContractStatus::version_unsupported when a seeded
///         draw is needed and this build cannot make it,
///         ContractStatus::sampling_exhausted from the bounded sampler, or
///         whatever validate(const SpeechCueConfig&) reported.
[[nodiscard]] ContractStatus prepare_trial(const SpeechCueConfig& config, TrialOrdinal ordinal,
                                           SpeechTrialSchedule& schedule) noexcept;

/// Total length of one trial, excluding any inter-trial gap.
///
/// @param schedule Realized schedule.
/// @param duration_ns Receives the length on ContractStatus::ok.
/// @return ContractStatus::ok, ContractStatus::duration_overflow, or whatever
///         validate(const SpeechTrialSchedule&) reported.
[[nodiscard]] ContractStatus trial_duration(const SpeechTrialSchedule& schedule,
                                            DurationNs& duration_ns) noexcept;

/// Build the intended phase timeline of one realized trial.
///
/// The phases are half-open and contiguous: each begins exactly where the last
/// ended, so no instant belongs to two of them and none belongs to neither. A
/// disabled cross contributes no entry, and neither does a zero gap.
///
/// @param config Session configuration the schedule belongs to.
/// @param schedule Realized schedule.
/// @param start_ns First instant of the black phase.
/// @param timeline Receives the timeline on ContractStatus::ok, and is left
///                 unmodified otherwise.
/// @return ContractStatus::ok, ContractStatus::duration_overflow when the
///         timeline does not fit in an integer nanosecond count, or whatever
///         validate_against(const SpeechTrialSchedule&, const SpeechCueConfig&)
///         reported.
[[nodiscard]] ContractStatus build_timeline(const SpeechCueConfig& config,
                                            const SpeechTrialSchedule& schedule,
                                            ExperimentTimeNs start_ns,
                                            SpeechTimeline& timeline) noexcept;

/// Emit one trial's realized durations as shared ScheduleDraw provenance.
///
/// The identity has to be the schedule's own. These records are what a replay
/// reads to learn which trial a realized value belongs to, so attributing a
/// value drawn for one trial to the identity of another does not produce a
/// slightly wrong record -- it produces provenance that is wrong in the one
/// field it exists to carry. The identity's ordinal and stimulus must therefore
/// match the schedule's, and are checked rather than trusted.
///
/// @param schedule Realized schedule.
/// @param trial Trial identity the records are attributed to. Its `ordinal` and
///              `stimulus_id` must be the schedule's.
/// @param time_ns Experiment time the values were realized.
/// @param draws Receives the records on ContractStatus::ok, and is left
///              unmodified otherwise.
/// @return ContractStatus::ok, ContractStatus::identity_missing when the
///         identity carries no stimulus, ContractStatus::outcome_invalid when
///         its ordinal or stimulus is not the schedule's, or whatever
///         validate(const SpeechTrialSchedule&) reported.
[[nodiscard]] ContractStatus make_schedule_draws(const SpeechTrialSchedule& schedule,
                                                 const TrialIdentity& trial,
                                                 ExperimentTimeNs time_ns,
                                                 SpeechTrialDraws& draws) noexcept;

/// Digest of a Speech configuration, for ScheduleIdentity.
///
/// Every field participates, including `cross_enabled`, the sampler version,
/// and the whole of both fixed-capacity arrays. Two configurations that would
/// produce different schedules therefore have different fingerprints.
///
/// The converse is what validate() protects: unused array slots are absorbed
/// too, so validate() requires each of them to hold one exact value. Without
/// that, two configurations that schedule identically could be told apart by a
/// value nothing ever reads.
[[nodiscard]] std::uint64_t configuration_fingerprint(const SpeechCueConfig& config) noexcept;

/// Digest of a stimulus catalog, for ScheduleIdentity.
///
/// Content participates, not just identifiers: a catalog that reassigned an
/// identifier to different text would otherwise be mistaken for the one a
/// session actually presented.
[[nodiscard]] std::uint64_t catalog_fingerprint(const SpeechCatalog& catalog) noexcept;

/// Everything a reader needs in order to reproduce a session's random values.
[[nodiscard]] ScheduleIdentity schedule_identity(const SpeechCueConfig& config,
                                                 const SpeechCatalog& catalog) noexcept;

/// Which artefact replays this configuration.
///
/// An explicit schedule is always its own authority: there is nothing to
/// regenerate, so the sampler version does not enter into it.
[[nodiscard]] constexpr ReplayAuthority replay_authority(const SpeechCueConfig& config) noexcept
{
    return config.schedule == SpeechScheduleKind::explicit_sequence
               ? ReplayAuthority::recorded_schedule
               : neurale::experiments::replay_authority(config.sampler_version);
}

/**
 * @brief The deterministic Speech cue trial state machine.
 *
 * The machine is a pure function of what it is given: an explicit `time_ns`, the
 * immutable SpeechCueConfig captured at start(), a prepared SpeechTrialSchedule
 * per trial, and its own state. It reads no clock, starts no timer, spawns no
 * thread, touches no microphone, decodes nothing, draws nothing, writes no
 * recording, and allocates nothing. Two machines given the same configuration
 * and the same schedules produce byte-identical transitions, events, requests,
 * and trials.
 *
 * # It does not sample
 *
 * Not "it samples reproducibly" -- it does not sample at all. Every realized
 * duration and the stimulus reach the machine inside a SpeechTrialSchedule that
 * was decided before the trial began, and the state machine does not get to
 * reinterpret that. That is why start() and step() take schedules rather
 * than ordinals: a machine holding a seed could regenerate one, and then the
 * question of which artefact decided a session would have two answers.
 *
 * # It runs the timeline it was given
 *
 * Every phase boundary is read out of the SpeechTimeline that
 * speech::build_timeline() produced for the trial, computed once when the trial
 * begins. The machine does not re-derive a boundary from a duration, so the
 * timeline a caller can inspect and the boundaries the machine actually takes
 * cannot drift apart -- they are the same values. build_timeline() runs
 * validate_against(), so its schedule-ownership checks gate every trial this
 * machine will run: a schedule that does not belong to the configuration never
 * reaches a transition.
 *
 * # Time
 *
 * Every phase is a half-open TimeInterval, following the rule frozen in
 * `time.h`: the exact end instant belongs to the next phase and never to both.
 * BLACK runs `[trial_start, trial_start + black_duration_ns)`; CROSS, when the
 * trial has one, begins exactly at BLACK's end; CONTENT begins at CROSS's end,
 * or at BLACK's end when the trial has no cross; and the trial completes at
 * CONTENT's end instant. Every realized duration is strictly positive, so no
 * phase is ever entered and left at the same instant.
 *
 * A phase advances because `time_ns` reached the instant its window ends, never
 * because step() was called some number of times. A caller polling at 10 Hz and
 * one polling at 1 kHz cross the same boundaries at the same instants, and a
 * caller that stopped polling for a whole trial's worth of time crosses all of
 * that trial's boundaries in one step, each exactly once.
 *
 * # A disabled cross is absent, not empty
 *
 * When the trial's schedule says `cross_enabled == false`, there is no
 * SpeechState::fixation_cross state, no cross transition, no cross onset or
 * offset marker, no cross PresentationRequest, and no cross entry in the
 * recorded timeline. CONTENT begins exactly at BLACK's end. Nothing anywhere
 * carries a zero-length cross, because a zero-length cross and a configured-away
 * cross would then be the same record, and the whole enabled/disabled
 * distinction rests on those two never looking alike.
 *
 * # What one step does
 *
 * step() advances repeatedly while the active phase has already ended at
 * `time_ns`, and stops when it has not -- or when one trial has been decided,
 * whichever comes first. Deciding at most one trial per step is what bounds the
 * work: a caller that stopped polling for several trials' worth of time would
 * otherwise have a single call infer an unbounded run of trials, and would have
 * to have supplied all of their schedules at once.
 *
 * A step that ends a trial also begins the next one, at the same instant --
 * CONTENT's end *is* the next BLACK's start, and half-open intervals leave no
 * gap between them for the machine to be in. That is why the next trial's
 * schedule is an argument to the step that crosses the boundary: there is no
 * position between two trials, only a missing input. A step that would cross it
 * without one reports ContractStatus::identity_missing and leaves the machine
 * exactly as it was, so the caller retries with the schedule and loses nothing.
 *
 * SpeechStepResult::settled says whether anything is still pending. When it is
 * true -- the case for any caller polling faster than one phase -- step() is
 * idempotent: repeating it at the same `time_ns` produces no transition, no
 * event, no request, and no trial.
 *
 * # Semantic time is not presentation time
 *
 * Every PresentationRequest this machine issues carries the instant the paradigm
 * *decided* to present, and never the instant anything appeared. The machine
 * produces no PresentationOutcome and cannot: only a presenter can report a
 * software presentation observation point, and that point is not physical
 * display onset. A headless run of this machine is evidence about semantic
 * timing and carries none about a monitor, a vsync, or a photodiode.
 */
/// Transitions one call to SpeechMachine::step() can emit.
///
/// The longest chain a single step can take is: leave BLACK for CROSS, leave
/// CROSS for CONTENT, and leave CONTENT for the next trial's BLACK or for
/// ::SpeechState::complete -- three, after which the step stops because a trial
/// has been decided. The capacity is stated above that so a future state cannot
/// silently overflow it.
inline constexpr std::size_t kMaxStepTransitions = 4;

/// Semantic events one call to SpeechMachine::step() can emit.
///
/// The longest chain emits a BLACK offset, a CROSS onset and offset, a CONTENT
/// onset and offset, ExperimentEventKind::trial_stop, and then either
/// ExperimentEventKind::session_stop or a ExperimentEventKind::trial_start with
/// the next trial's BLACK onset -- eight.
inline constexpr std::size_t kMaxStepEvents = 12;

/// Presentation requests one call to SpeechMachine::step() can issue.
///
/// One per phase onset: a CROSS, a CONTENT, and the next trial's BLACK.
inline constexpr std::size_t kMaxStepRequests = 4;

/// The states of one Speech cue session.
///
/// There is no inter-trial state, and the machine refuses a configuration that
/// asks for a gap: the next trial begins at the instant the previous one
/// completes, so its BLACK *is* the blank period between them. See
/// SpeechMachine::start().
enum class SpeechState : std::uint8_t
{
    /// No session. SpeechMachine::step() refuses; only start() leaves it.
    idle = 0,
    /// The blank field that opens every trial.
    black,
    /// The fixation cross. Entered only when the trial's schedule enables it.
    fixation_cross,
    /// The textual stimulus content.
    content,
    /// The session reached its configured trial count. Terminal.
    complete,
};

/// Whether @p state is one of the declared SpeechState values.
[[nodiscard]] constexpr bool speech_state_declared(SpeechState state) noexcept
{
    return static_cast<std::uint8_t>(state) <= static_cast<std::uint8_t>(SpeechState::complete);
}

/// Whether @p state is one in which a phase of a trial is running.
[[nodiscard]] constexpr bool speech_state_is_phase(SpeechState state) noexcept
{
    return state == SpeechState::black || state == SpeechState::fixation_cross ||
           state == SpeechState::content;
}

/// Which phase @p state presents.
///
/// ::SpeechState::idle and ::SpeechState::complete present no phase; they answer
/// SpeechPhase::black because the enumeration has no value for "none", and a
/// caller must not read a phase out of them. speech_state_is_phase() is the
/// question to ask first.
[[nodiscard]] constexpr SpeechPhase speech_phase_of(SpeechState state) noexcept
{
    return state == SpeechState::fixation_cross ? SpeechPhase::cross
           : state == SpeechState::content      ? SpeechPhase::content
                                                : SpeechPhase::black;
}

/// What is presented during @p phase.
[[nodiscard]] constexpr CueKind speech_cue_of(SpeechPhase phase) noexcept
{
    return phase == SpeechPhase::black     ? CueKind::black
           : phase == SpeechPhase::cross   ? CueKind::fixation_cross
           : phase == SpeechPhase::content ? CueKind::text_content
                                           : CueKind::none;
}

/// Why a transition was taken, stored in `StateTransition::cause`.
enum class SpeechCause : std::uint32_t
{
    /// Unset.
    unspecified = 0,
    /// A session began.
    session_started,
    /// The BLACK phase reached its end instant.
    black_elapsed,
    /// The CROSS phase reached its end instant.
    cross_elapsed,
    /// The CONTENT phase reached its end instant, ending the trial.
    content_elapsed,
    /// The configured trial count was reached.
    trial_limit_reached,
};

/// Paradigm-owned marker codes, carried in `ExperimentEvent::code`.
///
/// These are the phase onsets and offsets. They exist because ExperimentEvent is
/// the stream a recording stores, and PresentationRequest is not one of them: a
/// session whose event series held only session and trial events would have no
/// record of when its phases began and ended. Every onset is co-instant with the
/// PresentationRequest issued for the same phase, and every offset is co-instant
/// with the next onset -- or, for CONTENT, with
/// ExperimentEventKind::trial_stop. A disabled cross produces neither of its
/// two.
enum class SpeechMarker : std::uint32_t
{
    /// Unset.
    unspecified = 0,
    /// The blank field began.
    black_onset,
    /// The blank field ended.
    black_offset,
    /// The fixation cross began.
    cross_onset,
    /// The fixation cross ended.
    cross_offset,
    /// The stimulus content began.
    content_onset,
    /// The stimulus content ended.
    content_offset,
};

/// Why a trial ended, stored in `TrialRecord::reason`.
///
/// One value, and deliberately so. This machine observes nothing -- no
/// microphone, no decoder, no response -- so a trial that was presented in
/// full is the only outcome it can produce, and inventing a failure code for a
/// failure it cannot detect would be a vocabulary for something that never
/// happens. Richer outcomes arrive with acquisition, as new enumerators.
enum class SpeechReason : std::uint32_t
{
    /// Unset.
    unspecified = 0,
    /// The CONTENT phase ran to its end instant.
    content_elapsed,
};

/// One completed Speech cue trial.
///
/// `record` is a canonical experiment-side TrialRecord. The schedule beside it
/// is the trial's replay authority, and the timeline is the intervals the
/// machine actually ran -- with no cross entry when the trial had no cross.
/// Together they let a reader reconstruct the whole trial without the
/// configuration.
struct SpeechTrial
{
    /// The trial as the shared contract records it.
    TrialRecord record{};
    /// Realized schedule this trial ran.
    SpeechTrialSchedule schedule{};
    /// Phase intervals as run. `timeline.trial` equals `record.interval`.
    SpeechTimeline timeline{};
};

/// Everything a caller can read about a session between steps.
///
/// A value, produced on demand from the machine's own state. Nothing reads it
/// back, and holding one does not pin the machine.
struct SpeechSnapshot
{
    /// Most recent instant the machine accepted.
    ExperimentTimeNs time_ns{};
    /// Where the machine is.
    SpeechState state{SpeechState::idle};
    /// Phase the state presents. Meaningless in ::idle and ::complete.
    SpeechPhase phase{SpeechPhase::black};
    /// Trial in progress, or the last one completed once the session ended.
    TrialIdentity trial{};
    /// Realized schedule of that trial.
    SpeechTrialSchedule schedule{};
    /// Phase intervals of that trial.
    SpeechTimeline timeline{};
    /// Interval of the running phase. Empty in ::idle and ::complete.
    TimeInterval active{};
    /// What the experiment believes is being presented.
    ///
    /// A belief, not an observation: it advances when the machine issues a
    /// request, not when a presenter reports one. Its `cue` is CueKind::none in
    /// ::idle and ::complete, which is the machine saying it is asking for
    /// nothing rather than saying a blank field is up.
    PresentationState presentation{};
    /// Trials completed so far.
    TrialOrdinal completed{};
};

/// What one call to SpeechMachine::step() or start() produced.
///
/// Fixed capacity and no ownership, so a step allocates nothing. The counts say
/// how much of each array is meaningful; entries beyond them are unset.
struct SpeechStepResult
{
    /// State after the call.
    SpeechSnapshot snapshot{};
    /// Whether no phase remained ended when the call stopped.
    ///
    /// False only when the call stopped because it had decided a trial while
    /// more had already elapsed. Stepping again at the same `time_ns` continues.
    bool settled{true};
    /// Meaningful entries in `transitions`.
    std::uint8_t n_transitions{};
    /// Transitions taken, in order.
    std::array<StateTransition, kMaxStepTransitions> transitions{};
    /// Meaningful entries in `events`.
    std::uint8_t n_events{};
    /// Semantic events emitted, in order.
    std::array<ExperimentEvent, kMaxStepEvents> events{};
    /// Meaningful entries in `requests`.
    std::uint8_t n_requests{};
    /// Presentation requests issued, in order.
    std::array<PresentationRequest, kMaxStepRequests> requests{};
    /// Whether a trial was completed.
    bool trial_decided{};
    /// The completed trial. Meaningful only when `trial_decided`.
    SpeechTrial trial{};
};

/// Validate a completed Speech trial.
///
/// The shared TrialRecord validator runs first, then the schedule's own. What is
/// checked beyond them is that the three artefacts describe one trial: the
/// record's identity is the schedule's ordinal and stimulus; the outcome is the
/// only one this paradigm produces; the timeline holds exactly the phases the
/// schedule's `cross_enabled` calls for, contiguous, in order, with the cues and
/// durations the schedule realized; and `record.interval` is `timeline.trial`.
///
/// @return ContractStatus::ok, ContractStatus::outcome_invalid when the three
///         disagree, or whatever validate(const TrialRecord&) or
///         validate(const SpeechTrialSchedule&) reported.
[[nodiscard]] ContractStatus validate(const SpeechTrial& trial) noexcept;

/// The Speech cue trial state machine.
///
/// Value semantics: copyable, no pointer, no owned storage, and no hidden state.
/// Copying a machine copies a session, and a copy stepped with the same times
/// and schedules stays identical to its original. That is the whole production
/// core; the Python binding is a thin wrapper over it and reimplements none of
/// it.
class SpeechMachine
{
  public:
    /// Construct an ::SpeechState::idle machine with no session.
    SpeechMachine() noexcept = default;

    /// Begin a session on the first trial's prepared schedule.
    ///
    /// The configuration is captured, not referenced: it cannot change under a
    /// running session, and a machine is readable without the caller having kept
    /// the value it was started from.
    ///
    /// Emits ExperimentEventKind::session_start,
    /// ExperimentEventKind::trial_start, the BLACK onset marker, the transition
    /// into BLACK, and the BLACK presentation request.
    ///
    /// A nonzero `SpeechCueConfig::inter_trial_ns` is refused with
    /// ContractStatus::outcome_invalid. The machine freezes the trial at three
    /// phases and has no state to occupy during a separate gap, so the
    /// next trial begins at the instant the previous one completes and its BLACK
    /// is the blank period between them. A machine that accepted the field and
    /// ignored it would run a session that speech::build_timeline() describes
    /// differently -- two answers to when the next trial starts, one of which
    /// nobody would notice. Honouring a gap means adding the state, not adding a
    /// silent behaviour.
    ///
    /// @param paradigm Provenance identifier stamped on every record. Never
    ///        dispatch: nothing selects behaviour from it.
    /// @param config Session configuration.
    /// @param schedule Prepared schedule of trial zero. Its ordinal must be
    ///        zero, and it must belong to @p config.
    /// @param time_ns Instant the session begins, and the first BLACK onset.
    /// @param result Receives the transitions, events, requests, and snapshot on
    ///        ContractStatus::ok. Left unmodified otherwise.
    /// @return ContractStatus::ok, ContractStatus::already_running when the
    ///         machine already holds a session (any state other than ::idle; a
    ///         new session needs reset() first), ContractStatus::identity_missing
    ///         when @p paradigm is unset, ContractStatus::outcome_invalid for a
    ///         nonzero inter-trial gap or a schedule whose ordinal is not zero,
    ///         or whatever validate(const SpeechCueConfig&) or
    ///         build_timeline() reported. A refused start leaves the machine and
    ///         @p result unmodified.
    [[nodiscard]] ContractStatus start(ParadigmId paradigm, const SpeechCueConfig& config,
                                       const SpeechTrialSchedule& schedule,
                                       ExperimentTimeNs time_ns, SpeechStepResult& result) noexcept;

    /// Return to the state a default-constructed machine has.
    ///
    /// The configuration goes too. A reset session is a *new* session rather
    /// than a rewound one, so start() decides everything again from what it is
    /// given; that is what makes "reset then replay" reproduce a run exactly
    /// instead of resuming into ordinals that were already emitted.
    void reset() noexcept;

    /// Advance the session to @p time_ns without a next-trial schedule.
    ///
    /// Use this while a trial is in flight, and on the session's last trial,
    /// where no schedule follows. A step that would cross a trial boundary with
    /// trials still to run reports ContractStatus::identity_missing and leaves
    /// the machine untouched: the boundary is not a place the machine can stop,
    /// so the missing schedule is a missing input rather than a state to sit in.
    ///
    /// @param time_ns Instant to advance to. Must not move backwards.
    /// @param result Receives everything the step produced on
    ///        ContractStatus::ok. Left unmodified otherwise, as is the machine.
    /// @return ContractStatus::ok, ContractStatus::not_running in
    ///         ::SpeechState::idle or ::SpeechState::complete,
    ///         ContractStatus::time_regressed,
    ///         ContractStatus::identity_missing when a further trial's schedule
    ///         was needed and none was supplied, a propagated ordinal or
    ///         overflow failure, or ContractStatus::numerical_failure if the
    ///         chain ever exceeded its transition budget -- which the
    ///         one-trial-per-step rule makes unreachable, and which is checked
    ///         rather than assumed.
    [[nodiscard]] ContractStatus step(ExperimentTimeNs time_ns, SpeechStepResult& result) noexcept;

    /// Advance the session to @p time_ns, with the next trial's schedule to hand.
    ///
    /// @p next is used only if this step crosses a trial boundary; if it does
    /// not, nothing reads it. It is validated whenever it is used: its ordinal
    /// must be the current trial's plus one, and it must belong to the
    /// configuration.
    ///
    /// @param time_ns Instant to advance to. Must not move backwards.
    /// @param next Prepared schedule of the trial after the one in flight.
    /// @param result Receives everything the step produced on
    ///        ContractStatus::ok. Left unmodified otherwise, as is the machine.
    /// @return As the one-argument overload, plus ContractStatus::outcome_invalid
    ///         when @p next does not name the trial that follows, or whatever
    ///         build_timeline() reported for it.
    [[nodiscard]] ContractStatus step(ExperimentTimeNs time_ns, const SpeechTrialSchedule& next,
                                      SpeechStepResult& result) noexcept;

    /// Everything readable about the session right now.
    [[nodiscard]] SpeechSnapshot snapshot() const noexcept;

    /// The configuration this session was started with.
    ///
    /// Default-constructed while ::SpeechState::idle.
    [[nodiscard]] const SpeechCueConfig& configuration() const noexcept
    {
        return config_;
    }

    /// Provenance identifier stamped on this session's records.
    [[nodiscard]] ParadigmId paradigm() const noexcept
    {
        return paradigm_;
    }

    /// Where the machine is.
    [[nodiscard]] SpeechState state() const noexcept
    {
        return run_.state;
    }

    /// Whether the session has reached ::SpeechState::complete.
    [[nodiscard]] bool complete() const noexcept
    {
        return run_.state == SpeechState::complete;
    }

  private:
    /// The mutable half of a session.
    ///
    /// Split out so that step() can work on a copy and commit only on success:
    /// a step that fails partway leaves the machine exactly as it was, and
    /// copying this is cheap in a way that copying the configuration beside it
    /// would not be.
    struct Run;

    [[nodiscard]] ContractStatus emit_transition(Run& run, SpeechStepResult& result,
                                                 ExperimentTimeNs time_ns, SpeechState to,
                                                 SpeechCause cause) const noexcept;
    [[nodiscard]] ContractStatus emit_event(Run& run, SpeechStepResult& result,
                                            ExperimentTimeNs time_ns,
                                            ExperimentEventKind kind) const noexcept;
    [[nodiscard]] ContractStatus emit_marker(Run& run, SpeechStepResult& result,
                                             ExperimentTimeNs time_ns,
                                             SpeechMarker marker) const noexcept;
    [[nodiscard]] ContractStatus emit_request(Run& run, SpeechStepResult& result,
                                              SpeechPhase phase) const noexcept;
    [[nodiscard]] ContractStatus enter_phase(Run& run, SpeechStepResult& result, SpeechPhase phase,
                                             SpeechCause cause) const noexcept;
    [[nodiscard]] ContractStatus begin_trial(Run& run, SpeechStepResult& result,
                                             const SpeechTrialSchedule& schedule,
                                             ExperimentTimeNs time_ns, SpeechCause cause,
                                             bool first) const noexcept;
    [[nodiscard]] ContractStatus complete_trial(Run& run, SpeechStepResult& result,
                                                ExperimentTimeNs time_ns) const noexcept;
    [[nodiscard]] ContractStatus advance(Run& run, ExperimentTimeNs time_ns,
                                         const SpeechTrialSchedule* next,
                                         SpeechStepResult& result) const noexcept;
    [[nodiscard]] ContractStatus step_impl(ExperimentTimeNs time_ns,
                                           const SpeechTrialSchedule* next,
                                           SpeechStepResult& result) noexcept;
    [[nodiscard]] SpeechSnapshot build_snapshot(const Run& run) const noexcept;

    struct Run
    {
        MonotonicTimeGate gate{};
        TrialCounter trials{};
        SequenceCounter sequence{};
        SpeechState state{SpeechState::idle};
        TrialOrdinal ordinal{};
        SpeechTrialSchedule schedule{};
        SpeechTimeline timeline{};
        TimeInterval active{};
        PresentationState presentation{};
        TrialOrdinal completed{};
    };

    ParadigmId paradigm_{kUnsetParadigmId};
    SpeechCueConfig config_{};
    Run run_{};
};
} // namespace neurale::experiments::speech
