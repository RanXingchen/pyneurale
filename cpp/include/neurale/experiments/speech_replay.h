/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include <neurale/experiments/abnormal.h>
#include <neurale/experiments/presentation.h>
#include <neurale/experiments/replay.h>
#include <neurale/experiments/speech.h>

/**
 * @file
 * @brief Deterministic semantic replay of a recorded Speech cue run.
 *
 * Speech time is semantic: the machine advances through half-open phases from
 * explicit instants, so the recorded semantic input is nothing more than the
 * instants the run was advanced to. Everything else -- the per-trial BLACK,
 * CROSS, and CONTENT durations, the phase sequence, the onsets and ends, the
 * stimulus identifiers, the presentation requests, the completed trials -- is
 * regenerated from the configuration and its selected schedule authority.
 *
 * # The presentation boundary, kept
 *
 * A presenter's report of what it actually displayed is **evidence**. The
 * paradigm layer has nothing to regenerate it from, and a replay that
 * reproduced `presented_ns` would be reproducing the number it was handed. So
 * a report enters this replay as a recorded input, and what is regenerated is
 * everything the paradigm layer decided *about* it: whether the run could
 * attribute it to a CONTENT request the run itself emitted, which condition
 * that produced, and which response the configured severity gave the
 * condition. The report's own instants are never compared against a
 * regenerated instant, because there is no such thing.
 *
 * # Which schedule is authoritative
 *
 * A supported seeded configuration regenerates from its determining tuple. An
 * explicit configuration reads its schedule from the configuration itself and
 * never executes a sampler. Only a seeded configuration whose sampler this
 * build does not support reads the recorded realized schedule as fallback
 * input. In the first two cases the recorded schedule remains an output stream
 * checked only for trials that actually started.
 */
namespace neurale::experiments::speech
{

/// Version of the Speech record layout a replay understands.
///
/// The single source of truth for it; the recording bridge takes its
/// `experiment_version` from here.
inline constexpr std::uint32_t kSpeechRecordVersion = 1;

/// Everything a Speech replay re-executes under.
struct SpeechReplayConfig
{
    /// Provenance identifier the run was recorded under.
    ParadigmId paradigm{kUnsetParadigmId};
    /// Cue configuration, including the bounds, the seed, and the sampler.
    SpeechCueConfig task{};
    /// The immutable stimulus catalog.
    SpeechCatalog catalog{};
    /// Severities the run treated each class of abnormal condition under.
    AbnormalPolicySet abnormal{};
};

/// What one recorded semantic input to a Speech run was.
enum class SpeechReplayInputKind : std::uint8_t
{
    /// The run was advanced to an instant.
    advance = 0,
    /// A presenter reported on a presentation.
    ///
    /// Evidence, not an output. What the replay regenerates is the run's
    /// decision about it.
    presentation_report,
    /// The run was stopped from outside the task.
    halt,
};

/// Whether @p kind is one of the declared SpeechReplayInputKind values.
[[nodiscard]] constexpr bool speech_replay_input_kind_declared(SpeechReplayInputKind kind) noexcept
{
    return static_cast<std::uint8_t>(kind) <=
           static_cast<std::uint8_t>(SpeechReplayInputKind::halt);
}

/// One recorded semantic input.
struct SpeechReplayInput
{
    /// Which kind of input this is.
    SpeechReplayInputKind kind{SpeechReplayInputKind::advance};
    /// Experiment instant the run was advanced to, for ::SpeechReplayInputKind::advance.
    ExperimentTimeNs time_ns{};
    /// The presenter's report, for ::SpeechReplayInputKind::presentation_report.
    PresentationOutcome outcome{};
    /// Condition the run recorded for a halt.
    AbnormalCondition condition{AbnormalCondition::unspecified};
    /// Response the run recorded for a halt, compared against the derived one.
    AbnormalResponse response{AbnormalResponse::recorded};
};

/// What the run decided about one presenter report.
///
/// Every field is the paradigm layer's own decision. None of them is the
/// report's content: the instants a presenter reported are not regenerated,
/// and are not compared against anything a replay produced.
struct SpeechReplayReportDecision
{
    /// Whether the run attributed the report to a CONTENT request it emitted.
    bool matched{};
    /// The condition the run recorded about it.
    AbnormalCondition condition{AbnormalCondition::unspecified};
    /// The response the condition was decided under.
    AbnormalResponse response{AbnormalResponse::recorded};
};

/// One recorded intended phase of one trial.
struct SpeechReplayPhase
{
    /// The phase as the timeline described it.
    SpeechPhaseInterval phase{};
    /// Trial it belonged to.
    TrialIdentity trial{};
};

/// One recorded decided trial.
struct SpeechReplayTrial
{
    /// The trial as the machine decided it, with the machine's own verdict.
    SpeechTrial trial{};
    /// The outcome the recording's own column carries.
    ///
    /// Not always the machine's: a trial whose CONTENT a presenter says was
    /// never shown is written as TrialOutcome::aborted with the machine's
    /// verdict kept beside it.
    TrialOutcome recorded_outcome{TrialOutcome::pending};
    /// Whether a presenter's report made the trial inadmissible.
    bool invalidated{};
};

/// The recorded semantic outputs a replay is checked against.
struct SpeechReplayExpectation
{
    /// State transitions, in emission order.
    std::span<const StateTransition> transitions{};
    /// Semantic events, in emission order.
    std::span<const ExperimentEvent> events{};
    /// Presentation requests the paradigm made, in emission order.
    std::span<const PresentationRequest> requests{};
    /// Intended phases, in trial order and then phase order.
    std::span<const SpeechReplayPhase> phases{};
    /// Realized per-trial schedules, indexed by trial ordinal.
    std::span<const SpeechTrialSchedule> schedules{};
    /// Decided trials, in decision order.
    std::span<const SpeechReplayTrial> trials{};
    /// What the run decided about each presenter report, in report order.
    std::span<const SpeechReplayReportDecision> reports{};

    /// Whether the recording carried state transitions.
    bool has_transitions{};
    /// Whether it carried semantic events.
    bool has_events{};
    /// Whether it carried presentation requests.
    bool has_requests{};
    /// Whether it carried intended phases.
    bool has_phases{};
    /// Whether it carried realized per-trial schedules.
    ///
    /// A recording without them is replayable when the configuration itself
    /// supplies the schedules (explicit) or this build can regenerate them.
    /// An unsupported seeded sampler requires one record whenever replay first
    /// reaches the corresponding trial.
    bool has_schedules{};
    /// Whether it carried decided trials.
    bool has_trials{};
    /// Whether it carried the decisions taken about presenter reports.
    bool has_reports{};
};

/// One recorded Speech run, as a reader recovers it.
struct SpeechReplayRecording
{
    /// What the recording says the run was.
    ReplayProvenance provenance{};
    /// Experiment instant the run started.
    ExperimentTimeNs origin_ns{};
    /// Recorded semantic inputs, in the order the run consumed them.
    std::span<const SpeechReplayInput> inputs{};
    /// Recorded semantic outputs.
    SpeechReplayExpectation expected{};
    /// Whether the recording holds the run's own end.
    bool run_end_recorded{};
    /// Whether the recording reports that trace records were lost.
    bool trace_loss_recorded{};
};

/// Re-executes a recorded Speech cue run and checks it against what was recorded.
class SpeechReplay
{
  public:
    SpeechReplay() noexcept = default;

    /// Fix the configuration this replay regenerates under.
    [[nodiscard]] ContractStatus prepare(const SpeechReplayConfig& config);

    /// Return to the state a prepared replay starts each run from.
    void reset() noexcept;

    /// Replay @p recording and report what it established.
    [[nodiscard]] ReplayReport
    run(const SpeechReplayRecording& recording,
        ReplayIncompletePolicy policy = ReplayIncompletePolicy::verify_available) noexcept;

    /// The provenance this configuration regenerates under.
    ///
    /// Its `realized_schedule_fingerprint` is taken over @p realized, using the
    /// same field order as the recording bridge. Live metadata passes the full
    /// prepared schedule even though the output stream contains only trials
    /// that actually started.
    [[nodiscard]] ReplayProvenance
    provenance(std::span<const SpeechTrialSchedule> realized) const noexcept;

    /// Which artefact this configuration's sampler makes authoritative.
    [[nodiscard]] ReplayAuthority authority() const noexcept;

    /// Whether a configuration has been accepted.
    [[nodiscard]] bool prepared() const noexcept
    {
        return prepared_;
    }

    /// The configuration prepare() accepted.
    [[nodiscard]] const SpeechReplayConfig& configuration() const noexcept
    {
        return config_;
    }

  private:
    struct Cursors;

    [[nodiscard]] ContractStatus schedule_for(TrialOrdinal ordinal,
                                              const SpeechReplayRecording& recording,
                                              SpeechTrialSchedule& schedule) const noexcept;
    [[nodiscard]] ContractStatus replay_advance(const SpeechReplayInput& input,
                                                const SpeechReplayRecording& recording,
                                                Cursors& cursors,
                                                ReplayComparator& comparator) noexcept;
    void replay_report(const SpeechReplayInput& input, const SpeechReplayRecording& recording,
                       Cursors& cursors, ReplayComparator& comparator) noexcept;
    void compare_step(const SpeechStepResult& step, const SpeechReplayRecording& recording,
                      Cursors& cursors, ReplayComparator& comparator) noexcept;
    void compare_trial_start(const SpeechSnapshot& snapshot, const SpeechReplayRecording& recording,
                             Cursors& cursors, ReplayComparator& comparator) noexcept;

    /// One trial's emitted CONTENT request, kept so a presenter's report can be
    /// matched against it exactly as the run matched it.
    struct EmittedRequest
    {
        TrialOrdinal ordinal{};
        SequenceOrdinal sequence{};
        ExperimentTimeNs requested_ns{};
        TrialIdentity trial{};
        StimulusId stimulus_id{kUnsetStimulusId};
        /// Whether this slot describes the trial `ordinal` names.
        bool occupied{};
        /// Whether that trial emitted its CONTENT request.
        bool emitted{};
        /// Whether a report has already been accepted for it.
        bool answered{};
        /// Whether a report made the trial inadmissible.
        bool invalidated{};
    };

    SpeechReplayConfig config_{};
    SpeechMachine machine_{};
    /// One request/evidence slot per configured trial, allocated by prepare().
    /// The live SpeechTraceWriter owns the same cardinality, so offline replay
    /// imposes no extra report-latency window on a valid recording.
    std::unique_ptr<EmittedRequest[]> requests_{};
    /// Schedules fixed by the configuration at prepare time: explicit entries
    /// or values regenerated by a supported seeded sampler. Null only for the
    /// unsupported-seeded fallback that must read schedules on demand.
    std::unique_ptr<SpeechTrialSchedule[]> prepared_schedules_{};
    std::size_t n_trials_{};
    ReplayAuthority authority_{ReplayAuthority::regenerate};
    TrialOrdinal last_trial_{};
    bool has_trial_{};
    bool running_{};
    bool terminal_{};
    bool prepared_{};
};

} // namespace neurale::experiments::speech
