/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "speech_trace_writer.h"

#include <utility>

#include <neurale/experiments/schedule.h>

namespace neurale::execution
{

using namespace neurale::experiments;
namespace
{
/// Outcome slots a session is given when the caller does not size the queue.
/// One report per presentation request in flight is the shape the presenter
/// produces; a deeper queue only delays the moment a stalled presenter is
/// noticed.
constexpr std::size_t kDefaultOutcomeCapacity = 64;

/// Bits held per prepared trial in the evidence array.
constexpr std::uint8_t kContentRequested = 1U << 0U;
constexpr std::uint8_t kContentPresented = 1U << 1U;
constexpr std::uint8_t kTrialInvalidated = 1U << 2U;

/// Lowercase hex of a 32-byte digest into a fixed 64-char buffer, returned as a
/// view for the control writer's string field. The font SHA-256 is provenance,
/// recorded verbatim rather than truncated or rehashed.
[[nodiscard]] std::string_view hex_sha256(const std::array<std::uint8_t, 32>& bytes,
                                          std::array<char, 64>& out) noexcept
{
    static constexpr char kHex[] = "0123456789abcdef";
    for (std::size_t i = 0; i < bytes.size(); ++i)
    {
        out[i * 2] = kHex[(bytes[i] >> 4) & 0x0F];
        out[i * 2 + 1] = kHex[bytes[i] & 0x0F];
    }
    return {out.data(), out.size()};
}

/// Digest of the realized schedule itself.
///
/// Every field of every entry, in trial order, so that two legal realizations
/// of one configuration are distinguishable. Absorbed field by field rather
/// than as bytes because the struct has padding and a byte digest would depend
/// on it.
[[nodiscard]] std::uint64_t
realized_fingerprint(std::span<const speech::SpeechTrialSchedule> schedules) noexcept
{
    FingerprintAccumulator accumulator;
    accumulator.absorb(schedules.size());
    for (const auto& schedule : schedules)
    {
        accumulator.absorb(schedule.ordinal);
        accumulator.absorb(schedule.black_duration_ns);
        accumulator.absorb(schedule.cross_duration_ns);
        accumulator.absorb(schedule.content_duration_ns);
        accumulator.absorb(schedule.stimulus_id);
        accumulator.absorb(schedule.sampler_version);
        accumulator.absorb(schedule.cross_enabled ? 1U : 0U);
    }
    return accumulator.value();
}
} // namespace

SpeechTraceWriter::SpeechTraceWriter(SpeechHeadlessScheduler& scheduler, ParadigmId paradigm)
    : scheduler_(scheduler), paradigm_(paradigm),
      outcomes_(std::make_unique<BoundedTraceQueue<SpeechPresentationEvidence>>())
{
    outcomes_->prepare(kDefaultOutcomeCapacity);
    lifecycle_.prepare(kDefaultOutcomeCapacity);
    size_evidence();
}

void SpeechTraceWriter::size_evidence()
{
    const auto trials = scheduler_.schedules().size();
    if (trials == 0 || n_evidence_ == trials)
    {
        return;
    }
    // One byte per prepared trial, allocated here and never again. Producer-side
    // request identities are owned by the scheduler so their visibility cannot
    // depend on this writer's drain progress.
    evidence_ = std::make_unique<std::uint8_t[]>(trials);
    n_evidence_ = trials;
}

ContractStatus SpeechTraceWriter::prepare_outcomes(std::size_t capacity)
{
    if (running_)
    {
        // Replacing the queue under a live run is a use-after-free waiting to
        // happen -- a presenter can be pushing into the old one from its own
        // thread while the bridge drains it -- so the precondition is enforced
        // rather than written down and hoped for.
        return ContractStatus::already_running;
    }
    if (capacity != 0)
    {
        auto replacement = std::make_unique<BoundedTraceQueue<SpeechPresentationEvidence>>();
        replacement->prepare(capacity);
        outcomes_ = std::move(replacement);
    }
    // Sized again in case this writer was built before the scheduler was
    // prepared. It is not the supported order -- ::validate refuses a run whose
    // evidence does not cover the schedule -- but a caller that does prepare
    // the scheduler afterwards and then calls this has a usable writer.
    size_evidence();
    return ContractStatus::ok;
}

bool SpeechTraceWriter::report_presentation(const PresentationOutcome& outcome) noexcept
{
    SpeechPresentationEvidence evidence{};
    evidence.outcome = outcome;
    return report_presentation(evidence);
}

bool SpeechTraceWriter::report_presentation(const SpeechPresentationEvidence& evidence) noexcept
{
    return outcomes_->try_push(evidence) == streaming::StreamStatus::ok;
}

bool SpeechTraceWriter::report_presentation_config(SpeechPresentationConfigRecord config) noexcept
{
    if (configuration_written_ || has_presentation_config_)
    {
        return false;
    }
    presentation_config_ = std::move(config);
    has_presentation_config_ = true;
    return true;
}

ContractStatus SpeechTraceWriter::validate() const noexcept
{
    if (!scheduler_.prepared())
    {
        return ContractStatus::not_running;
    }
    if (paradigm_ == kUnsetParadigmId)
    {
        return ContractStatus::identity_missing;
    }
    if (const auto status = speech::validate(config()); status != ContractStatus::ok)
    {
        return status;
    }
    if (const auto status = speech::validate(catalog()); status != ContractStatus::ok)
    {
        return status;
    }
    if (const auto status = speech::validate_against(config(), catalog());
        status != ContractStatus::ok)
    {
        return status;
    }
    // The prepared schedule is one of the two things this session validates
    // before anything is armed. A schedule that does not cover the configured
    // trials, or whose ordinals are not the session's own, is not a schedule
    // this run can be replayed from.
    if (schedules().size() != static_cast<std::size_t>(config().n_trials))
    {
        return ContractStatus::identity_missing;
    }
    // Evidence and request identities cover every prepared trial, or this run
    // does not start. Without them a report cannot be matched to the request it
    // claims to answer, `presentation_evidence_required` would be recorded
    // beside a count of zero missing trials -- a run that tracked nothing
    // claiming everything was in order -- and every correct presenter report
    // would be filed as unattributable. A writer built before its scheduler was
    // prepared is the way to get there, and it is refused before anything is
    // armed rather than discovered in the recording afterwards.
    if (n_evidence_ != schedules().size())
    {
        return ContractStatus::identity_missing;
    }
    const auto prepared = schedules();
    for (std::size_t i = 0; i < prepared.size(); ++i)
    {
        const auto& schedule = prepared[i];
        if (const auto status = speech::validate(schedule); status != ContractStatus::ok)
        {
            return status;
        }
        if (const auto status = speech::validate_against(schedule, config());
            status != ContractStatus::ok)
        {
            return status;
        }
        if (schedule.ordinal != static_cast<TrialOrdinal>(i))
        {
            return ContractStatus::identity_missing;
        }
    }
    return ContractStatus::ok;
}

streaming::StreamStatus SpeechTraceWriter::start_execution(ExperimentTimeNs time_ns) noexcept
{
    const auto status = scheduler_.start(paradigm_, time_ns);
    if (status != streaming::StreamStatus::ok)
    {
        return status;
    }
    // This writer's reporter is configured from the same policy set and the
    // same paradigm the scheduler froze at prepare(), and from nowhere else.
    // Left unconfigured it would decide a presenter's report under the default
    // severities rather than the run's, and stamp the record with an unset
    // paradigm -- an event that ::validate rejects, written into a recording
    // that never asked it.
    abnormal_.configure(scheduler_.abnormal_policies(), paradigm_);
    abnormal_.restart();
    for (std::size_t i = 0; i < n_evidence_; ++i)
    {
        evidence_[i] = 0;
    }
    presentation_reports_ = 0;
    presentations_failed_ = 0;
    invalidated_trials_ = 0;
    unmatched_reports_ = 0;
    lifecycle_.reset();
    running_ = true;
    return status;
}

void SpeechTraceWriter::stop_execution(bool) noexcept
{
    // Once presentation producers are quiet, apply their accepted bounded
    // handoff before cancellation. This preserves a final runtime failure even
    // when no later scheduler advance occurs.
    static_cast<void>(scheduler_.apply_presentation_failures());
    running_ = false;
    scheduler_.finish_halt();
    scheduler_.cancel();
}

AbnormalSummary SpeechTraceWriter::abnormal_summary() const noexcept
{
    return merge(scheduler_.abnormal_summary(), abnormal_.summary());
}

void SpeechTraceWriter::halt(ExperimentTimeNs time_ns, AbnormalCondition condition) noexcept
{
    scheduler_.halt(time_ns, condition);
}

bool SpeechTraceWriter::trial_invalidated(TrialOrdinal ordinal) const noexcept
{
    return ordinal < n_evidence_ && (evidence_[ordinal] & kTrialInvalidated) != 0U;
}

void SpeechTraceWriter::note_presentation_request(const PresentationRequest& request) noexcept
{
    if (request.cue == CueKind::none || request.trial.ordinal >= n_evidence_ ||
        request.phase >= speech::kMaxSpeechPhases)
    {
        return;
    }
    if (request.cue == CueKind::text_content)
    {
        evidence_[request.trial.ordinal] |= kContentRequested;
    }
}

bool SpeechTraceWriter::emitted_content_request(TrialOrdinal ordinal,
                                                EmittedContentRequest& request) const noexcept
{
    EmittedPresentationRequest emitted{};
    if (!emitted_presentation_request(ordinal, speech::SpeechPhase::content, emitted))
    {
        return false;
    }
    request = {.sequence = emitted.sequence,
               .requested_ns = emitted.requested_ns,
               .trial = emitted.trial,
               .stimulus_id = emitted.stimulus_id,
               .emitted = emitted.emitted,
               .answered = emitted.answered};
    return true;
}

bool SpeechTraceWriter::emitted_presentation_request(
    TrialOrdinal ordinal, speech::SpeechPhase phase,
    EmittedPresentationRequest& request) const noexcept
{
    PresentationRequest emitted{};
    bool answered{};
    if (!scheduler_.emitted_presentation_request(ordinal, phase, emitted, answered))
        return false;
    request = {.sequence = emitted.sequence,
               .requested_ns = emitted.requested_ns,
               .trial = emitted.trial,
               .stimulus_id = emitted.stimulus_id,
               .cue = emitted.cue,
               .emitted = true,
               .answered = answered};
    return true;
}

bool SpeechTraceWriter::reports_on_emitted_request(const PresentationOutcome& outcome,
                                                   CueKind& cue) noexcept
{
    return scheduler_.claim_presentation_report(outcome, cue);
}

void SpeechTraceWriter::write_abnormal(ExperimentTraceSink& sink,
                                       const AbnormalEvent& event) noexcept
{
    static_cast<void>(sink.abnormal(event));
    static_cast<void>(sink.commit());
    if (event.response == AbnormalResponse::trial_invalidated && event.has_trial &&
        event.trial.ordinal < n_evidence_ &&
        (evidence_[event.trial.ordinal] & kTrialInvalidated) == 0U)
    {
        evidence_[event.trial.ordinal] |= kTrialInvalidated;
        ++invalidated_trials_;
    }
}

void SpeechTraceWriter::note_outcome(ExperimentTraceSink& sink, const PresentationOutcome& outcome,
                                     bool matched) noexcept
{
    ++presentation_reports_;
    if (!matched)
    {
        // A report this run cannot attribute to a request it made changes no
        // trial's evidence -- neither by supplying it nor by destroying it --
        // and is recorded as what it is. Marking the trial it named would let a
        // presenter decide, by naming an ordinal, whether a trial the run never
        // asked about counts as presented.
        ++unmatched_reports_;
        const auto event = abnormal_.observe(AbnormalCondition::presentation_report_unmatched,
                                             outcome.requested_ns, nullptr,
                                             static_cast<std::uint32_t>(outcome.status),
                                             AbnormalPolicy::record, /*can_end_trial=*/false,
                                             /*input_refused=*/true);
        write_abnormal(sink, event);
        return;
    }
    if (outcome.status == PresentationStatus::presented)
    {
        // Only CONTENT satisfies the trial-level scientific evidence contract;
        // BLACK/CROSS outcomes remain auditable presentation records.
        // `matched` is true here, so the request lookup performed by drain has
        // already established which request this is.
        return;
    }
    // A presenter reporting that it skipped, or that the request expired before
    // it could be shown, is a statement that the stimulus this trial is about
    // was not presented as asked. Whether that ends the trial's admissibility
    // is the configured policy's to say -- but it is never nothing, and it is
    // never resolved by pretending the intended onset was a presentation.
    ++presentations_failed_;
    const auto event = abnormal_.observe(
        AbnormalCondition::presentation_failed, outcome.requested_ns, &outcome.trial,
        static_cast<std::uint32_t>(outcome.status), AbnormalPolicy::record,
        /*can_end_trial=*/false, /*input_refused=*/false);
    write_abnormal(sink, event);
    if (event.response == AbnormalResponse::session_aborted)
    {
        // The condition was met here, on the thread that drains presenter
        // reports, but the run that has to stop is the scheduler's. Latched
        // rather than halted: the event just written is the record this abort
        // owes, and halt() would add a second one naming the same condition.
        scheduler_.stop_accepting();
    }
}

std::uint64_t SpeechTraceWriter::dropped_trace_count() const noexcept
{
    // Presenter reports are experiment trace too: an outcome that did not fit
    // is a run whose presenter-reported software presentation instant was never
    // recorded, and that is the one measurement this writer exists to make.
    return scheduler_.dropped_trace_count() + outcomes_->dropped() + lifecycle_.dropped();
}

SessionMetadata SpeechTraceWriter::metadata() const noexcept
{
    const auto identity = speech::schedule_identity(config(), catalog());
    return SessionMetadata{
        .experiment_type = kSpeechExperimentType,
        .experiment_version = kSpeechExperimentVersion,
        .configuration_fingerprint = identity.configuration_fingerprint,
        .schedule_fingerprint = schedule_fingerprint(identity),
        .realized_schedule_fingerprint = realized_fingerprint(schedules()),
        .schedule_seed = identity.seed,
        .sampler_version = identity.sampler_version,
        .input_schema_fingerprint = 0,
        .output_schema_fingerprint = 0,
        .metric_version = 0,
        .policy_version = static_cast<std::uint32_t>(config().stimulus_order),
    };
}

void SpeechTraceWriter::write_configuration(ExperimentTraceSink& sink,
                                            ExperimentTimeNs time_ns) noexcept
{
    configuration_written_ = true;
    const auto identity = speech::schedule_identity(config(), catalog());
    auto& text = sink.named(ControlKind::task_variables, "speech.configuration", time_ns);
    text.u64("configuration_fingerprint", identity.configuration_fingerprint);
    text.u64("catalog_fingerprint", identity.catalog_fingerprint);
    text.u64("schedule_fingerprint", schedule_fingerprint(identity));
    text.u64("realized_schedule_fingerprint", realized_fingerprint(schedules()));
    text.u64("paradigm", paradigm_);
    text.u64("seed", config().seed);
    text.u64("sampler_version", config().sampler_version);
    text.u64("trial_count", config().n_trials);
    text.u64("black_bound_ns", config().black_bound_ns);
    text.u64("cross_bound_ns", config().cross_bound_ns);
    text.u64("content_bound_ns", config().content_bound_ns);
    text.u64("inter_trial_ns", config().inter_trial_ns);
    text.boolean("cross_enabled", config().cross_enabled);
    text.u64("schedule", static_cast<std::uint64_t>(config().schedule));
    text.u64("stimulus_order", static_cast<std::uint64_t>(config().stimulus_order));
    text.u64("stimulus_count", config().n_stimuli);
    text.u64("prepared_trials", schedules().size());
    static_cast<void>(sink.commit());

    write_presentation_config(sink, time_ns);
}

std::size_t SpeechTraceWriter::drain(ExperimentTraceSink& sink, std::size_t budget) noexcept
{
    std::size_t taken = 0;
    PresentationSoftwareEvidence lifecycle{};
    while (taken < budget && lifecycle_.try_pop(lifecycle) == streaming::StreamStatus::ok)
    {
        write_lifecycle(sink, lifecycle);
        ++taken;
    }
    // Presenter reports may arrive before the recorder consumes their request
    // trace. Matching uses the scheduler's producer-side registry, so either
    // queue can be drained first without changing attribution.
    SpeechPresentationEvidence evidence{};
    while (taken < budget && outcomes_->try_pop(evidence) == streaming::StreamStatus::ok)
    {
        const auto& outcome = evidence.outcome;
        // Decided once, and told to both writers. Deciding it twice would let
        // the record say the report was attributed while the accounting said it
        // was not -- and the second call would find the request already
        // answered by the first.
        CueKind matched_cue{CueKind::none};
        const bool matched = reports_on_emitted_request(outcome, matched_cue);
        if (matched && matched_cue == CueKind::text_content &&
            outcome.status == PresentationStatus::presented)
        {
            evidence_[outcome.trial.ordinal] |= kContentPresented;
        }
        write_outcome(sink, evidence, matched);
        note_outcome(sink, outcome, matched);
        ++taken;
    }
    SpeechHeadlessTrace trace{};
    while (taken < budget && scheduler_.try_pop_trace(trace) == streaming::StreamStatus::ok)
    {
        write_trace(sink, trace);
        ++taken;
    }
    return taken;
}

void SpeechTraceWriter::write_lifecycle(ExperimentTraceSink& sink,
                                        const PresentationSoftwareEvidence& evidence) noexcept
{
    auto& text = sink.named(ControlKind::experiment_states, "speech.presentation_lifecycle",
                            evidence.time_ns);
    text.u64("event", static_cast<std::uint64_t>(evidence.event));
    text.u64("implementation_status", evidence.implementation_status);
    text.u64("update_ordinal", evidence.update_ordinal);
    text.boolean("has_trial", evidence.has_trial);
    if (evidence.has_trial)
        text.trial("trial", evidence.trial);
    static_cast<void>(sink.commit());

    if (evidence.event != PresentationLifecycleEvent::faulted)
        return;
    if (evidence.policy_delegated)
        return;
    const auto* trial = evidence.has_trial ? &evidence.trial : nullptr;
    const auto event = abnormal_.observe(AbnormalCondition::presentation_failed, evidence.time_ns,
                                         trial, evidence.implementation_status,
                                         AbnormalPolicy::record, /*can_end_trial=*/false,
                                         /*input_refused=*/false);
    write_abnormal(sink, event);
    if (event.response == AbnormalResponse::session_aborted)
        scheduler_.stop_accepting();
}

void SpeechTraceWriter::write_trace(ExperimentTraceSink& sink,
                                    const SpeechHeadlessTrace& trace) noexcept
{
    if (trace.is_abnormal)
    {
        write_abnormal(sink, trace.abnormal);
        return;
    }
    const auto& step = trace.step;
    for (std::uint8_t i = 0; i < step.n_transitions; ++i)
    {
        const auto& transition = step.transitions[i];
        auto& text =
            sink.named(ControlKind::experiment_states, "speech.transition", transition.time_ns);
        text.u64("sequence", transition.sequence);
        text.u64("paradigm", transition.paradigm);
        text.u64("from_state", transition.from_state);
        text.u64("to_state", transition.to_state);
        text.u64("cause", transition.cause);
        text.trial("trial", transition.trial);
        static_cast<void>(sink.value(static_cast<std::uint64_t>(transition.to_state)));
        static_cast<void>(sink.commit());
    }
    for (std::uint8_t i = 0; i < step.n_events; ++i)
    {
        const auto& event = step.events[i];
        auto& text = sink.named(ControlKind::events, "speech.event", event.time_ns);
        text.u64("sequence", event.sequence);
        text.u64("kind", static_cast<std::uint64_t>(event.kind));
        text.u64("paradigm", event.paradigm);
        text.u64("code", event.code);
        text.i64("value", event.value);
        text.trial("trial", event.trial);
        static_cast<void>(sink.value(static_cast<std::uint64_t>(event.kind)));
        static_cast<void>(sink.commit());
    }

    write_trial_start(sink, step.snapshot);

    for (std::uint8_t i = 0; i < trace.n_presentations; ++i)
    {
        const auto& prepared = trace.presentations[i];
        const auto& request = prepared.request;
        auto& text =
            sink.named(ControlKind::commands, "speech.presentation_request", request.requested_ns);
        text.u64("sequence", request.sequence);
        text.u64("onset_ns", request.onset_ns);
        text.u64("valid_until_ns", request.valid_until_ns);
        text.u64("duration_ns", request.duration_ns);
        text.u64("paradigm", request.paradigm);
        text.u64("phase", request.phase);
        text.u64("cue", static_cast<std::uint64_t>(request.cue));
        text.u64("stimulus_id", request.stimulus_id);
        text.boolean("payload_resolved", prepared.has_payload);
        if (prepared.has_payload)
        {
            text.u64("content", static_cast<std::uint64_t>(prepared.payload.content));
            text.u64("label", prepared.payload.label);
            text.u64("metadata", prepared.payload.metadata);
        }
        text.trial("trial", request.trial);
        static_cast<void>(sink.commit());
        note_presentation_request(request);
    }

    const auto& presentation = step.snapshot.presentation;
    auto& state =
        sink.named(ControlKind::labels, "speech.presentation_state", presentation.since_ns);
    state.u64("paradigm", presentation.paradigm);
    state.u64("phase", presentation.phase);
    state.u64("cue", static_cast<std::uint64_t>(presentation.cue));
    state.u64("stimulus_id", presentation.stimulus_id);
    state.trial("trial", presentation.trial);
    static_cast<void>(sink.commit());

    if (step.trial_decided)
    {
        const auto& trial = step.trial;
        auto record = trial.record;
        const bool invalidated = trial_invalidated(trial.record.trial.ordinal);
        if (invalidated)
        {
            // The machine ran the trial to its CONTENT end, which is the only
            // outcome it can produce. What it cannot know is that the
            // presentation the trial is about did not happen as asked. Writing
            // the machine's verdict unchanged would report a completed
            // presentation trial in which nothing was presented.
            record.outcome = TrialOutcome::aborted;
        }
        auto& label = sink.trial(record);
        label.boolean("invalidated", invalidated);
        if (invalidated)
        {
            label.u64("task_outcome", static_cast<std::uint64_t>(trial.record.outcome));
            label.u64("task_reason", trial.record.reason);
        }
        label.trial("trial", trial.record.trial);
        label.u64("paradigm", trial.record.paradigm);
        label.u64("black_duration_ns", trial.schedule.black_duration_ns);
        label.u64("cross_duration_ns", trial.schedule.cross_duration_ns);
        label.u64("content_duration_ns", trial.schedule.content_duration_ns);
        label.u64("stimulus_id", trial.schedule.stimulus_id);
        label.u64("sampler_version", trial.schedule.sampler_version);
        label.boolean("cross_enabled", trial.schedule.cross_enabled);
        label.u64("next_trial_start_ns", trial.timeline.next_trial_start_ns);
        static_cast<void>(sink.commit());
        completed_ = trial.record.trial.ordinal + 1;
    }
}

void SpeechTraceWriter::write_trial_start(ExperimentTraceSink& sink,
                                          const speech::SpeechSnapshot& snapshot) noexcept
{
    if (snapshot.state == speech::SpeechState::idle ||
        snapshot.state == speech::SpeechState::complete)
        return;
    if (has_trial_ && snapshot.trial.ordinal == last_trial_)
    {
        return;
    }
    has_trial_ = true;
    last_trial_ = snapshot.trial.ordinal;

    // The realized schedule and the intended timeline, once per trial, taken
    // from the same snapshot. Reading the two from different steps could pair
    // one trial's schedule with another's timeline, and a step begins at most
    // one trial.
    auto& schedule =
        sink.named(ControlKind::labels, "speech.schedule", snapshot.timeline.trial.start_ns);
    schedule.u64("black_duration_ns", snapshot.schedule.black_duration_ns);
    schedule.u64("cross_duration_ns", snapshot.schedule.cross_duration_ns);
    schedule.u64("content_duration_ns", snapshot.schedule.content_duration_ns);
    schedule.u64("stimulus_id", snapshot.schedule.stimulus_id);
    schedule.u64("sampler_version", snapshot.schedule.sampler_version);
    schedule.boolean("cross_enabled", snapshot.schedule.cross_enabled);
    schedule.interval("trial", snapshot.timeline.trial);
    schedule.u64("next_trial_start_ns", snapshot.timeline.next_trial_start_ns);
    schedule.u64("phase_count", snapshot.timeline.count);
    schedule.trial("trial_identity", snapshot.trial);
    static_cast<void>(sink.commit());

    for (std::uint8_t i = 0; i < snapshot.timeline.count; ++i)
    {
        const auto& phase = snapshot.timeline.phases[i];
        auto& text = sink.named(ControlKind::experiment_states, "speech.intended_phase",
                                phase.interval.start_ns);
        text.u64("phase", static_cast<std::uint64_t>(phase.phase));
        text.u64("cue", static_cast<std::uint64_t>(phase.cue));
        text.u64("stimulus_id", phase.stimulus_id);
        text.interval("interval", phase.interval);
        text.trial("trial", snapshot.trial);
        static_cast<void>(sink.value(static_cast<std::uint64_t>(phase.phase)));
        static_cast<void>(sink.commit());
    }

    auto& stimulus =
        sink.named(ControlKind::targets, "speech.stimulus", snapshot.timeline.trial.start_ns);
    stimulus.u64("stimulus_id", snapshot.trial.stimulus_id);
    stimulus.trial("trial", snapshot.trial);
    static_cast<void>(sink.value(static_cast<std::uint64_t>(snapshot.trial.stimulus_id)));
    static_cast<void>(sink.commit());
}

void SpeechTraceWriter::write_outcome(ExperimentTraceSink& sink,
                                      const SpeechPresentationEvidence& evidence,
                                      bool matched) noexcept
{
    const auto& outcome = evidence.outcome;
    // Stamped with the reported instant, and carrying the requested one beside
    // it. The record it reports on is named by request_sequence, never by a
    // time: the contract allows two requests at the same instant.
    //
    // `presented_ns` is meaningful only for PresentationStatus::presented. For
    // any other status it is written as null and the record is stamped with the
    // requested instant instead. The alternative -- writing the field's zero,
    // or stamping the record at zero -- puts an instant in a column a reader is
    // entitled to read as a measurement, for a presentation that never
    // happened. Missing evidence is recorded as missing.
    const bool presented = outcome.status == PresentationStatus::presented;
    auto& text = sink.named(ControlKind::labels, "speech.presentation_outcome",
                            presented ? outcome.presented_ns : outcome.requested_ns);
    text.u64("sequence", outcome.sequence);
    text.u64("request_sequence", outcome.request_sequence);
    text.u64("requested_ns", outcome.requested_ns);
    text.boolean("software_evidence_available", evidence.software.has_software_times);
    if (evidence.software.has_software_times)
    {
        text.u64("intended_ns", evidence.software.intended_ns);
        text.u64("submitted_renderer_ns", evidence.software.submitted_renderer_ns);
        text.u64("submitted_ns", evidence.software.submitted_ns);
        text.u64("presented_renderer_ns", evidence.software.presented_renderer_ns);
        text.u64("implementation_status", evidence.software.implementation_status);
        // The presenter's swap-return experiment-time observation, kept distinct
        // from the semantic `presented_ns` below. A swap that crossed the
        // deadline yields status=expired with `presented_ns` null (it was never
        // a valid presentation) but the software swap still physically happened
        // at this instant -- the measurement the presenter exists to make is
        // preserved here, not dropped at the recording boundary.
        text.u64("software_presented_ns", evidence.software.presented_ns);
    }
    if (presented)
    {
        text.u64("presented_ns", outcome.presented_ns);
    }
    else
    {
        text.null("presented_ns");
    }
    text.u64("status", static_cast<std::uint64_t>(outcome.status));
    text.u64("stimulus_id", outcome.stimulus_id);
    // Whether this report was matched to a request the run emitted. A reader
    // must be able to tell a measurement from a report the run kept but could
    // not attribute; the record is kept either way, because a presenter saying
    // something unexpected is itself worth having.
    text.boolean("matched_request", matched);
    text.trial("trial", outcome.trial);
    static_cast<void>(sink.value(static_cast<std::uint64_t>(outcome.status)));
    static_cast<void>(sink.commit());
}

void SpeechTraceWriter::write_presentation_config(ExperimentTraceSink& sink,
                                                  ExperimentTimeNs time_ns) noexcept
{
    if (!has_presentation_config_)
    {
        return;
    }
    const auto& c = presentation_config_;
    auto& text = sink.named(ControlKind::task_variables, "speech.presentation_config", time_ns);
    text.u64("pixel_height", c.pixel_height);
    text.f64("logical_left", c.logical_left);
    text.f64("logical_bottom", c.logical_bottom);
    text.f64("logical_width", c.logical_width);
    text.f64("logical_height", c.logical_height);
    text.f64("fixation_center_x", c.fixation_center_x);
    text.f64("fixation_center_y", c.fixation_center_y);
    text.f64("fixation_half_extent", c.fixation_half_extent);
    text.f64("text_baseline_x", c.text_baseline_x);
    text.f64("text_baseline_y", c.text_baseline_y);
    text.u64("aspect_policy", c.aspect_policy);
    static_cast<void>(sink.commit());

    auto& font = sink.named(ControlKind::task_variables, "speech.presentation_font", time_ns);
    // The font identity the presenter computed (path, face index, and SHA-256 of
    // the bytes FreeType consumed via FT_New_Memory_Face) is separate from
    // geometry so both bounded records remain below the session control limit.
    font.text("font_path", c.font_path);
    std::array<char, 64> hex{};
    font.text("font_sha256", hex_sha256(c.font_sha256, hex));
    font.i64("face_index", c.face_idx);
    static_cast<void>(sink.commit());

    auto& window = sink.named(ControlKind::task_variables, "speech.presentation_window", time_ns);
    window.u64("window_width", static_cast<std::uint64_t>(c.window_width));
    window.u64("window_height", static_cast<std::uint64_t>(c.window_height));
    window.i64("monitor_index", c.monitor_idx);
    window.i64("swap_interval", c.swap_interval);
    window.boolean("fullscreen", c.fullscreen);
    window.boolean("resizable", c.resizable);
    window.boolean("visible", c.visible);
    static_cast<void>(sink.commit());

    auto& style = sink.named(ControlKind::task_variables, "speech.presentation_style", time_ns);
    const auto write_colour =
        [&style](std::string_view name, const PresentationConfigColour& colour) noexcept
    {
        const std::array<double, 4> components{colour.red, colour.green, colour.blue, colour.alpha};
        style.array_f64(name, components);
    };
    write_colour("background", c.background);
    write_colour("fixation", c.fixation);
    write_colour("text", c.text);
    static_cast<void>(sink.commit());
}

void SpeechTraceWriter::write_summary(ExperimentTraceSink& sink, ExperimentTimeNs time_ns) noexcept
{
    // Trials that asked for their content to be presented and for which no
    // presenter ever said it was. Counted here, at the end, because a report
    // may legitimately arrive after the trial it is about has completed;
    // deciding per trial as it ended would report a missing report that was
    // merely late.
    std::uint64_t missing_evidence = 0;
    for (std::size_t i = 0; i < n_evidence_; ++i)
    {
        const auto bits = evidence_[i];
        if ((bits & kContentRequested) != 0U && (bits & kContentPresented) == 0U)
        {
            ++missing_evidence;
        }
    }
    if (require_evidence_ && missing_evidence != 0)
    {
        // Reported once, for the run, and not resolved. There is no instant to
        // supply for a presentation nobody observed, and the intended onset is
        // not one: it is what the paradigm asked for, which is the very thing a
        // timing claim would be trying to check against.
        const auto event =
            abnormal_.observe(AbnormalCondition::presentation_evidence_missing, time_ns, nullptr,
                              static_cast<std::uint32_t>(missing_evidence), AbnormalPolicy::record,
                              /*can_end_trial=*/false,
                              /*input_refused=*/false);
        auto& fault = sink.abnormal(event);
        fault.u64("trials_without_evidence", missing_evidence);
        fault.u64("presentation_reports", presentation_reports_);
        fault.u64("unmatched_reports", unmatched_reports_);
        static_cast<void>(sink.commit());
    }

    auto& text = sink.named(ControlKind::task_variables, "speech.summary", time_ns);
    text.u64("completed_trials", completed_);
    text.u64("configured_trials", config().n_trials);
    text.u64("prepared_trials", schedules().size());
    text.u64("presentation_reports", presentation_reports_);
    text.u64("presentations_failed", presentations_failed_);
    text.u64("unmatched_presentation_reports", unmatched_reports_);
    text.u64("invalidated_trials", invalidated_trials_);
    text.u64("trials_without_presentation_evidence", missing_evidence);
    text.boolean("presentation_evidence_required", require_evidence_);
    static_cast<void>(sink.commit());
}

} // namespace neurale::execution
