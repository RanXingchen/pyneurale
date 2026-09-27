/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "speech_controller.h"

#include <limits>

#include "contract_status.h"

namespace neurale::execution
{

using namespace neurale::experiments;

streaming::StreamStatus
SpeechHeadlessScheduler::prepare(const speech::SpeechCueConfig& config,
                                 const speech::SpeechCatalog& catalog,
                                 std::span<const speech::SpeechTrialSchedule> schedules,
                                 std::size_t trace_capacity, const AbnormalPolicySet& abnormal)
{
    if (prepared_ || closed_)
        return streaming::StreamStatus::invalid_state;
    if (trace_capacity == 0 || validate(abnormal) != ContractStatus::ok ||
        speech::validate(config) != ContractStatus::ok ||
        speech::validate(catalog) != ContractStatus::ok ||
        speech::validate_against(config, catalog) != ContractStatus::ok ||
        config.n_trials != schedules.size() || schedules.empty())
        return streaming::StreamStatus::realtime_configuration_failed;
    if (schedules.size() >
            (std::numeric_limits<std::size_t>::max)() / sizeof(speech::SpeechTrialSchedule) ||
        schedules.size() >
            (std::numeric_limits<std::size_t>::max)() / sizeof(speech::SpeechStimulus) ||
        schedules.size() > (std::numeric_limits<std::size_t>::max)() / speech::kMaxSpeechPhases ||
        schedules.size() * speech::kMaxSpeechPhases >
            (std::numeric_limits<std::size_t>::max)() / sizeof(PresentationRequestSlot))
        return streaming::StreamStatus::realtime_configuration_failed;

    auto prepared_schedules =
        std::make_unique_for_overwrite<speech::SpeechTrialSchedule[]>(schedules.size());
    auto prepared_payloads =
        std::make_unique_for_overwrite<speech::SpeechStimulus[]>(schedules.size());
    auto prepared_requests =
        std::make_unique<PresentationRequestSlot[]>(schedules.size() * speech::kMaxSpeechPhases);
    for (std::size_t i = 0; i < schedules.size(); ++i)
    {
        if (schedules[i].ordinal != i ||
            speech::validate_against(schedules[i], config) != ContractStatus::ok ||
            speech::find_stimulus(catalog, schedules[i].stimulus_id, prepared_payloads[i]) !=
                ContractStatus::ok)
            return streaming::StreamStatus::realtime_configuration_failed;
        prepared_schedules[i] = schedules[i];
    }

    traces_.prepare(trace_capacity);
    presentation_failures_.prepare(trace_capacity);
    config_ = config;
    catalog_ = catalog;
    schedules_ = std::move(prepared_schedules);
    payloads_ = std::move(prepared_payloads);
    presentation_requests_ = std::move(prepared_requests);
    n_schedules_ = schedules.size();
    abnormal_policies_pending_ = abnormal;
    prepared_ = true;
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus SpeechHeadlessScheduler::start(ParadigmId paradigm,
                                                       ExperimentTimeNs time_ns) noexcept
{
    if (!prepared_ || running_ || closed_)
        return streaming::StreamStatus::invalid_state;
    traces_.reset();
    presentation_failures_.reset();
    reset_presentation_requests();
    machine_.reset();
    cancelled_.store(false, std::memory_order_release);
    halted_.store(false, std::memory_order_release);
    halt_refusal_recorded_ = false;
    halt_pending_ = false;
    abnormal_.configure(abnormal_policies_pending_, paradigm);
    abnormal_.restart();
    speech::SpeechStepResult result{};
    const auto status = machine_.start(paradigm, config_, schedules_[0], time_ns, result);
    if (status != ContractStatus::ok)
        return map_contract_status(status);
    const auto publish_status = publish(result);
    if (publish_status != streaming::StreamStatus::ok)
    {
        machine_.reset();
        return publish_status;
    }
    running_ = true;
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus SpeechHeadlessScheduler::advance(ExperimentTimeNs time_ns) noexcept
{
    if (!prepared_ || closed_)
        return streaming::StreamStatus::invalid_state;
    if (cancelled_.load(std::memory_order_acquire))
        return streaming::StreamStatus::stopped;
    if (halted_.load(std::memory_order_acquire))
        return refuse_after_halt(time_ns);
    // Budgeted: advance() is the stepping path, not shutdown.
    if (const auto status = apply_presentation_failures(kPresentationFailureDrainBudget);
        status != streaming::StreamStatus::ok)
        return status;
    if (!running_)
        return streaming::StreamStatus::invalid_state;
    if (machine_.complete())
        return streaming::StreamStatus::stopped;
    if (!traces_.can_push())
    {
        traces_.note_dropped(1);
        return streaming::StreamStatus::queue_overflow;
    }

    // Nothing here treats a large step as abnormal. Speech time is semantic:
    // the machine advances through half-open phases from explicit instants, and
    // a caller that advances by a whole trial's worth of nanoseconds gets every
    // phase boundary in between, in order, exactly as if it had been stepped
    // through them. A jump is a jump in the caller's clock, not evidence that
    // anything went wrong, and reporting one as a condition would fill the
    // record with faults on every deliberately coarse advance.
    speech::SpeechStepResult result{};
    const auto snapshot = machine_.snapshot();
    const auto next_ordinal = snapshot.trial.ordinal + 1;
    const auto status = next_ordinal < n_schedules_
                            ? machine_.step(time_ns, schedules_[next_ordinal], result)
                            : machine_.step(time_ns, result);
    if (status != ContractStatus::ok)
        return map_contract_status(status);
    const auto publish_status = publish(result);
    if (publish_status != streaming::StreamStatus::ok)
        return publish_status;
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus
SpeechHeadlessScheduler::publish(const speech::SpeechStepResult& result) noexcept
{
    SpeechHeadlessTrace trace{};
    trace.step = result;
    trace.n_presentations = result.n_requests;
    for (std::uint8_t i = 0; i < result.n_requests; ++i)
    {
        const auto& request = result.requests[i];
        auto& presentation = trace.presentations[i];
        presentation.request = request;
        if (request.stimulus_id == kUnsetStimulusId)
            continue;
        if (request.trial.ordinal >= n_schedules_ ||
            schedules_[request.trial.ordinal].stimulus_id != request.stimulus_id)
            return streaming::StreamStatus::consumer_failure;
        presentation.payload = payloads_[request.trial.ordinal];
        presentation.has_payload = true;
    }
    if (const auto status = register_presentation_requests(result);
        status != streaming::StreamStatus::ok)
        return status;
    return traces_.try_push(trace);
}

void SpeechHeadlessScheduler::reset_presentation_requests() noexcept
{
    for (std::size_t i = 0; i < n_schedules_ * speech::kMaxSpeechPhases; ++i)
    {
        presentation_requests_[i].answered.store(false, std::memory_order_relaxed);
        presentation_requests_[i].emitted.store(false, std::memory_order_relaxed);
        presentation_requests_[i].request = {};
    }
}

streaming::StreamStatus SpeechHeadlessScheduler::register_presentation_requests(
    const speech::SpeechStepResult& result) noexcept
{
    for (std::uint8_t i = 0; i < result.n_requests; ++i)
    {
        const auto& request = result.requests[i];
        if (experiments::validate(request) != ContractStatus::ok ||
            request.trial.ordinal >= n_schedules_ || request.phase >= speech::kMaxSpeechPhases)
            return streaming::StreamStatus::consumer_failure;
        auto& slot = presentation_requests_[request.trial.ordinal * speech::kMaxSpeechPhases +
                                            request.phase];
        if (slot.emitted.load(std::memory_order_acquire))
            return streaming::StreamStatus::consumer_failure;
        slot.request = request;
        slot.answered.store(false, std::memory_order_relaxed);
        slot.emitted.store(true, std::memory_order_release);
    }
    return streaming::StreamStatus::ok;
}

bool SpeechHeadlessScheduler::emitted_presentation_request(TrialOrdinal ordinal,
                                                           speech::SpeechPhase phase,
                                                           PresentationRequest& request,
                                                           bool& answered) const noexcept
{
    const auto phase_idx = static_cast<std::size_t>(phase);
    if (presentation_requests_ == nullptr || ordinal >= n_schedules_ ||
        phase_idx >= speech::kMaxSpeechPhases)
        return false;
    const auto& slot = presentation_requests_[ordinal * speech::kMaxSpeechPhases + phase_idx];
    if (!slot.emitted.load(std::memory_order_acquire))
        return false;
    request = slot.request;
    answered = slot.answered.load(std::memory_order_acquire);
    return true;
}

bool SpeechHeadlessScheduler::claim_presentation_report(const PresentationOutcome& outcome,
                                                        CueKind& cue) noexcept
{
    if (experiments::validate(outcome) != ContractStatus::ok ||
        outcome.trial.ordinal >= n_schedules_)
        return false;
    const auto base = outcome.trial.ordinal * speech::kMaxSpeechPhases;
    for (std::size_t phase = 0; phase < speech::kMaxSpeechPhases; ++phase)
    {
        auto& slot = presentation_requests_[base + phase];
        if (!slot.emitted.load(std::memory_order_acquire))
            continue;
        const auto& request = slot.request;
        if (outcome.request_sequence != request.sequence ||
            outcome.requested_ns != request.requested_ns ||
            outcome.stimulus_id != request.stimulus_id || !same_trial(outcome.trial, request.trial))
            continue;
        bool expected = false;
        if (!slot.answered.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
            return false;
        cue = request.cue;
        return true;
    }
    return false;
}

streaming::StreamStatus SpeechHeadlessScheduler::note_presentation_failure(
    const PresentationFailureEvidence& evidence) noexcept
{
    if (!prepared_ || closed_ || !running_)
        return streaming::StreamStatus::invalid_state;
    if (cancelled_.load(std::memory_order_acquire))
        return streaming::StreamStatus::stopped;
    if (halted_.load(std::memory_order_acquire))
        return refuse_after_halt(evidence.time_ns);
    const auto current = machine_.snapshot();
    if (machine_.complete() || !same_trial(current.trial, evidence.trial))
        return streaming::StreamStatus::invalid_frame;
    return report_abnormal(AbnormalCondition::presentation_failed, evidence.time_ns,
                           AbnormalPolicy::record, false, evidence.implementation_status);
}

streaming::StreamStatus
SpeechHeadlessScheduler::apply_presentation_failures(std::size_t budget) noexcept
{
    return drain_presentation_failures(presentation_failures_, budget,
                                       [this](const PresentationFailureEvidence& evidence) noexcept
                                       { return note_presentation_failure(evidence); });
}

streaming::StreamStatus SpeechHeadlessScheduler::reset() noexcept
{
    if (!prepared_ || closed_)
        return streaming::StreamStatus::invalid_state;
    machine_.reset();
    traces_.reset();
    presentation_failures_.reset();
    reset_presentation_requests();
    running_ = false;
    halt_refusal_recorded_ = false;
    halt_pending_ = false;
    cancelled_.store(false, std::memory_order_release);
    halted_.store(false, std::memory_order_release);
    return streaming::StreamStatus::ok;
}

void SpeechHeadlessScheduler::cancel() noexcept
{
    cancelled_.store(true, std::memory_order_release);
}

void SpeechHeadlessScheduler::close() noexcept
{
    if (closed_)
        return;
    cancel();
    running_ = false;
    traces_.close();
    presentation_failures_.close();
    closed_ = true;
}

streaming::StreamStatus SpeechHeadlessScheduler::report_abnormal(AbnormalCondition condition,
                                                                 ExperimentTimeNs time_ns,
                                                                 AbnormalPolicy floor,
                                                                 bool input_refused,
                                                                 std::uint32_t detail) noexcept
{
    if (!traces_.can_push())
    {
        traces_.note_dropped(1);
        return streaming::StreamStatus::queue_overflow;
    }
    const auto snapshot = machine_.snapshot();
    const bool in_trial = running_ && !machine_.complete();

    SpeechHeadlessTrace trace{};
    trace.is_abnormal = true;
    // SpeechMachine owns when a trial ends -- a trial ends when its CONTENT
    // phase reaches its end instant, and there is no other way out of one. So a
    // policy of `abort_trial` marks the trial in flight inadmissible rather
    // than ending it, and the record says which of the two happened.
    trace.abnormal = abnormal_.observe(condition, time_ns, in_trial ? &snapshot.trial : nullptr,
                                       detail, floor, /*can_end_trial=*/false, input_refused);
    if (trace.abnormal.response == AbnormalResponse::session_aborted)
        // See CenterOutController::apply_abnormal. Speech is stepped by its
        // caller in every configuration, so here the latch is the only thing
        // that ends the run at all.
        stop_accepting();
    const auto push = traces_.try_push(trace);
    if (push != streaming::StreamStatus::ok)
        return push;
    return trace.abnormal.response == AbnormalResponse::session_aborted
               ? streaming::StreamStatus::consumer_failure
               : streaming::StreamStatus::ok;
}

streaming::StreamStatus
SpeechHeadlessScheduler::refuse_after_halt(ExperimentTimeNs time_ns) noexcept
{
    if (!halt_refusal_recorded_)
    {
        halt_refusal_recorded_ = true;
        static_cast<void>(report_abnormal(AbnormalCondition::input_after_terminal, time_ns,
                                          AbnormalPolicy::record, true, 0));
    }
    return streaming::StreamStatus::stopped;
}

void SpeechHeadlessScheduler::halt(ExperimentTimeNs time_ns, AbnormalCondition condition) noexcept
{
    if (closed_ || halted_.exchange(true, std::memory_order_acq_rel))
        return;
    halt_time_ns_ = time_ns;
    halt_condition_ = condition;
    halt_pending_ = true;
}

void SpeechHeadlessScheduler::finish_halt() noexcept
{
    if (!halt_pending_)
        return;
    halt_pending_ = false;
    if (!running_)
        return;
    static_cast<void>(
        report_abnormal(halt_condition_, halt_time_ns_, AbnormalPolicy::abort_trial, true, 0));
    running_ = false;
}

} // namespace neurale::execution
