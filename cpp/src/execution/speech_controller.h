/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include <neurale/experiments/abnormal.h>
#include <neurale/experiments/speech.h>
#include <neurale/streaming/fault.h>

#include "abnormal_reporter.h"
#include "bounded_trace_queue.h"
#include "presentation_evidence.h"

namespace neurale::execution
{

using namespace neurale::experiments;

struct PreparedPresentation
{
    PresentationRequest request{};
    speech::SpeechStimulus payload{};
    bool has_payload{};
};

struct SpeechHeadlessTrace
{
    speech::SpeechStepResult step{};
    std::array<PreparedPresentation, speech::kMaxStepRequests> presentations{};
    std::uint8_t n_presentations{};
    /// True when this record carries an abnormal condition rather than a step.
    ///
    /// A flag rather than a kind enumeration because Speech has exactly one
    /// other kind of record, and `step` is unset on an abnormal one.
    bool is_abnormal{};
    /// Meaningful when `is_abnormal`.
    AbnormalEvent abnormal{};
};

/// Explicit-time headless Speech scheduler. prepare() freezes every trial
/// schedule and resolves every content payload; advance() performs no sampling,
/// catalog search, wall-clock wait, allocation, rendering, or I/O.
class SpeechHeadlessScheduler final
{
  public:
    SpeechHeadlessScheduler() noexcept = default;

    SpeechHeadlessScheduler(const SpeechHeadlessScheduler&) = delete;
    SpeechHeadlessScheduler& operator=(const SpeechHeadlessScheduler&) = delete;

    [[nodiscard]] streaming::StreamStatus
    prepare(const speech::SpeechCueConfig& config, const speech::SpeechCatalog& catalog,
            std::span<const speech::SpeechTrialSchedule> schedules, std::size_t trace_capacity,
            const AbnormalPolicySet& abnormal = {});
    [[nodiscard]] streaming::StreamStatus start(ParadigmId paradigm,
                                                ExperimentTimeNs time_ns) noexcept;
    [[nodiscard]] streaming::StreamStatus advance(ExperimentTimeNs time_ns) noexcept;
    [[nodiscard]] streaming::StreamStatus reset() noexcept;
    void cancel() noexcept;
    void close() noexcept;

    /// Refuse all further advances from this instant. See
    /// CenterOutController::halt for why this is not cancel(), and why it only
    /// latches.
    void halt(ExperimentTimeNs time_ns, AbnormalCondition condition) noexcept;

    /// Produce the record a latched halt owes. See
    /// CenterOutController::finish_halt.
    void finish_halt() noexcept;

    /// Bounded presentation-thread handoff. The task state remains owned by
    /// the scheduler thread and is not changed by enqueueing evidence.
    [[nodiscard]] bool
    enqueue_presentation_failure(const PresentationFailureEvidence& evidence) noexcept
    {
        return presentation_failures_.try_push(evidence) == streaming::StreamStatus::ok;
    }

    /// Apply queued presentation failures on the scheduler/task-owner thread.
    ///
    /// @param budget Maximum number applied by this call.
    ///        ::kUnboundedPresentationFailureDrain drains all of them and is for
    ///        shutdown; advance() passes ::kPresentationFailureDrainBudget so a
    ///        single step cannot absorb a whole queue's worth of abnormal
    ///        decisions.
    [[nodiscard]] streaming::StreamStatus
    apply_presentation_failures(std::size_t budget = kUnboundedPresentationFailureDrain) noexcept;

    /// Query a request at the point where the Speech producer emitted it.
    /// This registry is independent of when the recording consumer drains the
    /// corresponding trace.
    [[nodiscard]] bool emitted_presentation_request(TrialOrdinal ordinal, speech::SpeechPhase phase,
                                                    PresentationRequest& request,
                                                    bool& answered) const noexcept;

    /// Atomically accept the first report that exactly answers one request
    /// emitted by this scheduler. Duplicate or malformed reports are rejected.
    [[nodiscard]] bool claim_presentation_report(const PresentationOutcome& outcome,
                                                 CueKind& cue) noexcept;

    /// Latch the terminal state, recording nothing. See
    /// CenterOutController::stop_accepting for what an `abort_session`
    /// response *is*, why this is distinct from halt(), and why it writes no
    /// record of its own.
    void stop_accepting() noexcept
    {
        halted_.store(true, std::memory_order_release);
    }

    /// Whether advances are being refused because the run reached a terminal
    /// state.
    [[nodiscard]] bool halted() const noexcept
    {
        return halted_.load(std::memory_order_acquire);
    }

    /// Abnormal conditions this scheduler has met, ever. Monotonic.
    ///
    /// A presenter's reports are not among them: those reach the run through
    /// SpeechTraceWriter, which keeps its own reporter, and the two are merged
    /// where the session reads them.
    [[nodiscard]] AbnormalSummary abnormal_summary() const noexcept
    {
        return abnormal_.summary();
    }

    /// The abnormal policies prepare() froze.
    [[nodiscard]] const AbnormalPolicySet& abnormal_policies() const noexcept
    {
        return abnormal_.policies();
    }

    [[nodiscard]] streaming::StreamStatus try_pop_trace(SpeechHeadlessTrace& trace) noexcept
    {
        return traces_.try_pop(trace);
    }
    [[nodiscard]] speech::SpeechSnapshot snapshot() const noexcept
    {
        return machine_.snapshot();
    }
    [[nodiscard]] bool prepared() const noexcept
    {
        return prepared_;
    }

    /// Trace records this scheduler could not enqueue, ever. Monotonic.
    [[nodiscard]] std::uint64_t dropped_trace_count() const noexcept
    {
        return traces_.dropped() + presentation_failures_.dropped();
    }

    /// The configuration prepare() froze. See CenterOutController for why the
    /// recording bridge reads it from here rather than from its own copy.
    [[nodiscard]] const speech::SpeechCueConfig& configuration() const noexcept
    {
        return config_;
    }
    /// The catalog prepare() froze.
    [[nodiscard]] const speech::SpeechCatalog& catalog() const noexcept
    {
        return catalog_;
    }
    /// The realized schedule prepare() froze, in trial order.
    ///
    /// This is the schedule the run is executing -- not a schedule the caller
    /// says it prepared -- which is what makes a fingerprint taken over it
    /// evidence rather than a restatement of the configuration.
    [[nodiscard]] std::span<const speech::SpeechTrialSchedule> schedules() const noexcept
    {
        return std::span<const speech::SpeechTrialSchedule>{schedules_.get(), n_schedules_};
    }

  private:
    struct PresentationRequestSlot
    {
        PresentationRequest request{};
        std::atomic<bool> emitted{};
        std::atomic<bool> answered{};
    };
    static_assert(std::atomic<bool>::is_always_lock_free);

    [[nodiscard]] streaming::StreamStatus publish(const speech::SpeechStepResult& result) noexcept;
    void reset_presentation_requests() noexcept;
    [[nodiscard]] streaming::StreamStatus
    register_presentation_requests(const speech::SpeechStepResult& result) noexcept;
    [[nodiscard]] streaming::StreamStatus
    note_presentation_failure(const PresentationFailureEvidence& evidence) noexcept;
    [[nodiscard]] streaming::StreamStatus report_abnormal(AbnormalCondition condition,
                                                          ExperimentTimeNs time_ns,
                                                          AbnormalPolicy floor, bool input_refused,
                                                          std::uint32_t detail) noexcept;
    [[nodiscard]] streaming::StreamStatus refuse_after_halt(ExperimentTimeNs time_ns) noexcept;

    speech::SpeechCueConfig config_{};
    speech::SpeechCatalog catalog_{};
    std::unique_ptr<speech::SpeechTrialSchedule[]> schedules_{};
    std::unique_ptr<speech::SpeechStimulus[]> payloads_{};
    std::unique_ptr<PresentationRequestSlot[]> presentation_requests_{};
    std::size_t n_schedules_{};
    speech::SpeechMachine machine_{};
    BoundedTraceQueue<SpeechHeadlessTrace> traces_{};
    BoundedTraceQueue<PresentationFailureEvidence> presentation_failures_{};
    AbnormalReporter abnormal_{};
    /// The policies prepare() froze, held until start() knows the paradigm
    /// identity the reporter has to stamp records with.
    AbnormalPolicySet abnormal_policies_pending_{};
    bool prepared_{};
    bool running_{};
    bool closed_{};
    bool halt_refusal_recorded_{};
    ExperimentTimeNs halt_time_ns_{};
    AbnormalCondition halt_condition_{AbnormalCondition::unspecified};
    bool halt_pending_{};
    std::atomic<bool> cancelled_{};
    std::atomic<bool> halted_{};
};

} // namespace neurale::execution
