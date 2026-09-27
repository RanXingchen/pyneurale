/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The Speech side of the recording bridge.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include <memory>

#include <neurale/experiments/abnormal.h>
#include <neurale/experiments/presentation.h>
#include <neurale/experiments/speech.h>
#include <neurale/experiments/speech_replay.h>

#include "abnormal_reporter.h"
#include "presentation_evidence.h"
#include "session.h"
#include "speech_controller.h"

namespace neurale::execution
{

using namespace neurale::experiments;

inline constexpr std::string_view kSpeechExperimentType{"speech_cue"};
inline constexpr std::uint32_t kSpeechExperimentVersion = speech::kSpeechRecordVersion;

/// The identity of the CONTENT presentation request one trial emitted.
///
/// Enough of the request to decide whether a later report is about it: the
/// sequence that names it, the instant it was made at, and the stimulus and
/// trial it was for. A report that agrees with all of them is evidence about
/// that trial; one that does not is a statement about the presenter.
struct EmittedContentRequest
{
    SequenceOrdinal sequence{};
    ExperimentTimeNs requested_ns{};
    TrialIdentity trial{};
    StimulusId stimulus_id{kUnsetStimulusId};
    /// Whether the request was made at all.
    bool emitted{};
    /// Whether a report has already been accepted for it. A second report on
    /// one request is not a second presentation.
    bool answered{};
};

/// Identity of one BLACK, CROSS, or CONTENT request emitted by a trial.
/// The scheduler owns the fixed producer-side slots; this is the narrow value
/// copied out for callers that need to correlate presentation evidence.
struct EmittedPresentationRequest
{
    SequenceOrdinal sequence{};
    ExperimentTimeNs requested_ns{};
    TrialIdentity trial{};
    StimulusId stimulus_id{kUnsetStimulusId};
    CueKind cue{CueKind::none};
    bool emitted{};
    bool answered{};
};

/// Turns SpeechHeadlessScheduler trace records into control records.
///
/// The intended and the actual are two records, never one. Everything the
/// paradigm decides -- the phase transitions, the presentation requests, the
/// believed presentation state -- is stamped with the experiment time the
/// paradigm decided at. What a presenter later reports it actually did arrives
/// through report_presentation() and is written as its own record. Folding the
/// reported instant back into the intended one would destroy the only evidence
/// that they differed, which is the measurement this writer exists to make.
class SpeechTraceWriter final : public ExperimentTraceSource
{
  public:
    /// Bind to the scheduler that will run, and to the paradigm identity the
    /// run is recorded under.
    ///
    /// The configuration, catalog, and prepared schedule are read from the
    /// scheduler rather than handed in again. Handing them in a second time
    /// would make it possible to record one prepared schedule while the
    /// scheduler advances another -- and for Speech that is not a theoretical
    /// risk, because a seeded schedule is accepted on membership and bounds
    /// alone, so two different realizations of one configuration are both
    /// legal.
    /// Not noexcept: it sizes the outcome queue, which allocates. It was never
    /// noexcept in fact -- the queue's prepare() has always allocated -- and a
    /// constructor that says otherwise is a claim the next reader has to check
    /// against the body to disbelieve.
    SpeechTraceWriter(SpeechHeadlessScheduler& scheduler, ParadigmId paradigm);

    /// Accept one presenter report. Bounded and lock-free; the presenter's
    /// thread is the only producer. `false` means the bounded outcome queue was
    /// full, which the caller reports to its session as a trace loss.
    [[nodiscard]] bool report_presentation(const PresentationOutcome& outcome) noexcept;
    /// Preserve the complete software-timing evidence beside the semantic
    /// outcome. The outcome remains the only value used for policy.
    [[nodiscard]] bool report_presentation(const SpeechPresentationEvidence& evidence) noexcept;
    /// Bounded implementation evidence for surface lifecycle events that do
    /// not answer a semantic PresentationRequest.
    [[nodiscard]] bool report_lifecycle(const PresentationSoftwareEvidence& evidence) noexcept
    {
        return lifecycle_.try_push(evidence) == streaming::StreamStatus::ok;
    }
    /// Record the frozen Speech presentation configuration as provenance,
    /// including the font identity (path, face index, and the SHA-256 of the
    /// bytes FreeType consumed). Stored and written once at configuration time;
    /// duplicate or post-configuration reports are rejected.
    [[nodiscard]] bool report_presentation_config(SpeechPresentationConfigRecord config) noexcept;

    /// Fingerprint of the exact catalog frozen by the bound scheduler.
    [[nodiscard]] std::uint64_t catalog_fingerprint() const noexcept
    {
        return speech::catalog_fingerprint(scheduler_.catalog());
    }

    /// Hand one exact runtime failure to the scheduler this writer is bound to.
    /// Keeping the association here prevents a caller from pairing a writer
    /// with a different Speech task owner.
    [[nodiscard]] bool
    enqueue_presentation_failure(const PresentationFailureEvidence& evidence) noexcept
    {
        return scheduler_.enqueue_presentation_failure(evidence);
    }

    /// Size the outcome queue.
    ///
    /// Optional. The constructor installs a default-sized queue, and -- when
    /// the scheduler it is given is already prepared, which is the supported
    /// order -- the per-trial presentation evidence as well. Nothing about
    /// matching a presenter's report to the request it answers depends on this
    /// call; what depends on it is only how many reports can be in flight at
    /// once.
    ///
    /// Legal before the session starts, and only then, which is now answered
    /// rather than assumed: it replaces the queue rather than resizing it, and
    /// a presenter pushing into the old one while it is replaced is a race no
    /// ordering here could fix. Returns ::ContractStatus::already_running when
    /// called on a live run, and changes nothing.
    ///
    /// A zero capacity keeps the queue that is installed. Whatever the
    /// capacity, this also sizes the evidence if the constructor could not --
    /// which is the case only for a writer built before its scheduler was
    /// prepared, an order ::validate refuses a run for.
    [[nodiscard]] ContractStatus prepare_outcomes(std::size_t capacity);

    /// Require a presenter's report before this run's trials count as evidence
    /// of what was shown.
    ///
    /// Off by default, because a headless-only session has no presenter and
    /// every trial would report missing evidence. It is turned on by a caller
    /// whose scientific claim rests on actual display timing -- and the point of the
    /// flag is that the run then says so, rather than the intended onset
    /// quietly standing in for a measurement nobody made.
    void require_presentation_evidence(bool required) noexcept
    {
        require_evidence_ = required;
    }

    /// The CONTENT presentation request trial @p ordinal has emitted, if any.
    ///
    /// What a presenter has to report against. A presenter learns a request by
    /// receiving it, but a bridge that has to correlate reports back -- and a
    /// test standing in for one -- needs the run's own record of what was
    /// asked, because a report is only evidence about a trial when it names the
    /// request that trial actually made. Returns false before the request is
    /// emitted, and for an ordinal this run has no trial for.
    [[nodiscard]] bool emitted_content_request(TrialOrdinal ordinal,
                                               EmittedContentRequest& request) const noexcept;

    /// Query one semantic request after the scheduler producer emitted it.
    /// Recording drain order is irrelevant.
    [[nodiscard]] bool
    emitted_presentation_request(TrialOrdinal ordinal, speech::SpeechPhase phase,
                                 EmittedPresentationRequest& request) const noexcept;

    [[nodiscard]] ContractStatus validate() const noexcept override;
    [[nodiscard]] streaming::StreamStatus
    start_execution(ExperimentTimeNs time_ns) noexcept override;
    void stop_execution(bool aborting) noexcept override;
    [[nodiscard]] std::uint64_t dropped_trace_count() const noexcept override;
    [[nodiscard]] SessionMetadata metadata() const noexcept override;
    void write_configuration(ExperimentTraceSink& sink, ExperimentTimeNs time_ns) noexcept override;
    [[nodiscard]] std::size_t drain(ExperimentTraceSink& sink,
                                    std::size_t budget) noexcept override;
    void write_summary(ExperimentTraceSink& sink, ExperimentTimeNs time_ns) noexcept override;
    [[nodiscard]] AbnormalSummary abnormal_summary() const noexcept override;
    void halt(ExperimentTimeNs time_ns, AbnormalCondition condition) noexcept override;

  private:
    /// Allocate per-trial recording evidence from the scheduler's prepared
    /// schedule. Idempotent, and a no-op until there is a prepared schedule to
    /// size against.
    void size_evidence();
    void write_trace(ExperimentTraceSink& sink, const SpeechHeadlessTrace& trace) noexcept;
    void write_abnormal(ExperimentTraceSink& sink, const AbnormalEvent& event) noexcept;
    /// Account one presenter report against the trial it names, and report the
    /// abnormal condition when it says the presentation did not happen.
    void note_outcome(ExperimentTraceSink& sink, const PresentationOutcome& outcome,
                      bool matched) noexcept;
    /// Account that the recorder has consumed a presentable request. Producer
    /// identity is owned by the scheduler; this only updates recording-summary
    /// evidence.
    void note_presentation_request(const PresentationRequest& request) noexcept;
    /// Whether @p outcome reports on a request this run actually made.
    ///
    /// The whole of what makes a presenter's report evidence *about a trial*.
    /// Without it, a report is attributed on the trial ordinal it names, and a
    /// report naming any legal ordinal could satisfy -- or destroy -- that
    /// trial's presentation evidence while reporting on a request that was
    /// never emitted.
    [[nodiscard]] bool reports_on_emitted_request(const PresentationOutcome& outcome,
                                                  CueKind& cue) noexcept;
    [[nodiscard]] bool trial_invalidated(TrialOrdinal ordinal) const noexcept;
    void write_trial_start(ExperimentTraceSink& sink,
                           const speech::SpeechSnapshot& snapshot) noexcept;
    void write_outcome(ExperimentTraceSink& sink, const SpeechPresentationEvidence& evidence,
                       bool matched) noexcept;
    void write_lifecycle(ExperimentTraceSink& sink,
                         const PresentationSoftwareEvidence& evidence) noexcept;
    void write_presentation_config(ExperimentTraceSink& sink, ExperimentTimeNs time_ns) noexcept;

    /// What the scheduler froze at prepare(). The only copies there are.
    [[nodiscard]] const speech::SpeechCueConfig& config() const noexcept
    {
        return scheduler_.configuration();
    }
    [[nodiscard]] const speech::SpeechCatalog& catalog() const noexcept
    {
        return scheduler_.catalog();
    }
    [[nodiscard]] std::span<const speech::SpeechTrialSchedule> schedules() const noexcept
    {
        return scheduler_.schedules();
    }

    SpeechHeadlessScheduler& scheduler_;
    ParadigmId paradigm_{kUnsetParadigmId};
    /// Held indirectly so that ::prepare_outcomes can install a different size
    /// before the run. The queue itself is fixed-capacity once installed, and
    /// the extra indirection is one load on a path that is bounded but not
    /// realtime -- the presenter's, not a controller's.
    std::unique_ptr<BoundedTraceQueue<SpeechPresentationEvidence>> outcomes_{};
    BoundedTraceQueue<PresentationSoftwareEvidence> lifecycle_{};
    /// This writer's own reporter, separate from the scheduler's.
    ///
    /// Two reporters for one run, because they have two producers: the
    /// scheduler meets its conditions on whichever thread advances it, and a
    /// presenter's reports are drained here on the bridge thread. One shared
    /// reporter would need a lock on a path a realtime producer uses, and
    /// ::abnormal_summary merges the two instead.
    AbnormalReporter abnormal_{};
    /// Per prepared trial: whether its CONTENT was requested, whether a
    /// presenter reported presenting it, and whether an abnormal condition made
    /// the trial inadmissible. Sized once, never grown, and indexed by trial
    /// ordinal.
    std::unique_ptr<std::uint8_t[]> evidence_{};
    std::size_t n_evidence_{};
    std::uint64_t presentation_reports_{};
    std::uint64_t presentations_failed_{};
    std::uint64_t invalidated_trials_{};
    /// Presenter reports this run could not attribute to a request it made.
    std::uint64_t unmatched_reports_{};
    bool require_evidence_{};
    SpeechPresentationConfigRecord presentation_config_{};
    bool has_presentation_config_{};
    bool configuration_written_{};
    /// Whether a session is executing against this writer. Read by
    /// ::prepare_outcomes, which must not replace anything a presenter thread
    /// may be pushing into.
    bool running_{};
    bool has_trial_{};
    TrialOrdinal last_trial_{};
    TrialOrdinal completed_{};
};

} // namespace neurale::execution
