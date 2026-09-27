/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// Headless session orchestration and the experiment recording bridge.
///
/// What this is: the order in which a recorded experiment is brought up and
/// taken down, and the one thread that turns the paradigm controllers' bounded
/// trace records into control records.
///
/// What this is **not**, deliberately: there is no experiment base class, no
/// registry, no factory, no paradigm graph, and no scheduler. It constructs no
/// source, processor, decoder, actuator, recorder, spool, or clock -- the
/// caller owns every one of those and hands over references. A session is a
/// sequence, not a framework.
///
/// ### Bring-up order
///
/// 1. validate the paradigm configuration and its prepared deterministic
///    schedule (::ExperimentTraceSource::validate);
/// 2. configure and prepare the recorder, when recording is enabled;
/// 3. prepare the runtime, which leaves safety inhibited;
/// 4. pass the recorder's readiness gate directly when it is control-only;
///    a runtime-attached recorder is checked by `NativeStreamRunner::arm()`;
/// 5. arm, and only through `NativeStreamRunner::arm()` -- the session never
///    touches a `SafetyController` itself, because the runtime's arm is where
///    the readiness of every critical edge is checked before safety is
///    released, and a second path to release would be a second policy;
/// 6. start the paradigm's own execution
///    (::ExperimentTraceSource::start_execution);
/// 7. start the recorder, write the session's opening records, and drain
///    synchronously what the paradigm's own start already produced;
/// 8. start the bridge, and only then start the runtime.
///
/// Steps 6 to 8 are in that order because the runtime is what delivers frames
/// and the paradigm is what consumes them. Starting the runtime first would let
/// the first frame arrive at a controller that has not started, and starting
/// the paradigm before step 5 would put execution in front of the safety
/// release. A `start()` that returns `ok` therefore means the paradigm is
/// running, not that it is ready to be run: the caller has nothing left to
/// start.
///
/// The runtime is last, and the synchronous drain in step 7 is why. A paradigm
/// that has started has already produced its opening trace, and the smallest
/// legal trace capacity is one slot; a runtime started while that slot is still
/// occupied can overflow the queue on its very first frame, before any drain
/// has had a chance to run. Emptying the queue on this thread makes the initial
/// trace's departure a fact rather than a race that a larger capacity merely
/// makes rarer, and starting the bridge before the runtime is the same
/// argument for every frame after the first.
///
/// A failed start unwinds everything it reached, and `arm()` is the part that
/// is easy to miss: it releases safety without moving the runtime out of
/// `prepared`, so a runtime that was armed and never started is aborted here
/// rather than left released.
///
/// ### Shutdown order
///
/// 1. latch the stop or abort intent;
/// 2. ask the runtime to stop -- safety inhibition on the actuator path is the
///    runtime's own, and happens there;
/// 3. join the runtime workers, then stop the paradigm's execution, which
///    together are what make the trace producer quiet;
/// 4. drain every accepted trace record, with the recorder still accepting,
///    and write the summary;
/// 5. settle the terminal intent **after** that final drain, then stop or abort
///    the recorder accordingly, and finalize or expose the failure;
/// 6. repeated stop, abort, and close change nothing and report the same thing.
///
/// Step 5 comes after step 4 rather than before it because the last drain is
/// where a loss is most likely to be discovered: a queue that was full, a
/// record the recorder refuses, or a summary that does not fit. A terminal
/// intent frozen before the drain would close a recording as a clean stop while
/// the session's own outcome said the trace has a hole in it, and the recording
/// would then be the more optimistic of the two artefacts.
///
/// A control-only recorder is driven entirely by this session. The formal
/// closed-loop path may instead attach the same recorder to the runtime's
/// synchronous input critical tap. Its adapter deliberately treats runtime
/// terminal as "data producer quiet" rather than sealing the recorder; this
/// session remains the sole owner of the final trace drain, summary, terminal,
/// and finalization ordering. That is how acquisition and experiment evidence
/// share one recording without racing the control tail against runtime exit.
///
/// ### Finalization
///
/// The native recorder core does not finalize: `finalize()` lives in the
/// offline Python finalizer. `close()` here therefore reports
/// `finalization_required` honestly rather than claiming a sealed NRF session
/// that nothing wrote. An interrupted session leaves the spool exactly as the
/// recovery mechanisms expect to find it.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string_view>
#include <thread>

#include <neurale/experiments/abnormal.h>
#include <neurale/experiments/contract.h>
#include <neurale/experiments/events.h>
#include <neurale/experiments/schedule.h>
#include <neurale/streaming/fault.h>

#include "abnormal_reporter.h"
#include "recorder.h"
#include "recorder_status.h"
#include "recording_plan.h"
#include "records.h"
#include "spool_file.h"

namespace neurale::streaming
{
class NativeStreamRunner;
}

namespace neurale::execution
{

using namespace neurale::experiments;

/// Version of the record layout this session writes. It travels in the session
/// metadata so a reader never has to infer the layout from the records.
inline constexpr std::uint32_t kExperimentTraceVersion = 1;

/// The fault code a lost experiment trace record is reported under.
inline constexpr std::string_view kTraceLossFaultCode{"experiment-trace-loss"};
/// The stage that fault belongs to.
inline constexpr std::string_view kTraceLossFaultStage{"experiment"};
/// The stage an abnormal experiment condition is reported under. Its *code* is
/// ::abnormal_condition_name of the condition, so the vocabulary a reader sees
/// in a recording is the contract's own rather than a second one invented here.
inline constexpr std::string_view kAbnormalFaultStage{"experiment"};
/// The fault code the runtime's own preserved primary fault is written under.
inline constexpr std::string_view kRuntimeFaultCode{"runtime-primary-fault"};
/// The stage that fault belongs to. Not `experiment`: the runtime produced it.
inline constexpr std::string_view kRuntimeFaultStage{"runtime"};
/// The fault code an operator's emergency stop is *requested* under.
///
/// Deliberately not `emergency-stop`. That code belongs to the paradigm's own
/// ::AbnormalEvent -- what the task did about the stop, which trial it ended,
/// and under which policy -- and two records sharing it would make "how many
/// emergency stops did this run have" unanswerable from the recording. This one
/// is the other fact: that a stop was asked for from outside, and the reason
/// the operator gave. It is written even when the paradigm's own record cannot
/// be, which is why it is not folded into it.
inline constexpr std::string_view kEmergencyStopRequestCode{"emergency-stop-requested"};
/// The fault code the session's reaction to a paradigm's abort is written under.
///
/// For the same reason ::kEmergencyStopRequestCode is not `emergency-stop`.
/// The condition itself is already a record: the paradigm's own ::AbnormalEvent
/// names it, times it, and says which trial it ended. This is the second, and
/// different, fact -- that the session saw the run declare itself unable to
/// continue and reacted, once, by faulting and (where there is one) aborting
/// the runtime. Written under the condition's own name, the two would be
/// indistinguishable, and a reader counting `presentation-failed` records would
/// count one failed presentation as two. The condition is carried in the
/// record's fields instead, where it is data rather than an identity.
inline constexpr std::string_view kSessionAbortedCode{"experiment-session-aborted"};

/// What a session records about the run as a whole.
///
/// Every field is a fact the caller already fixed: a fingerprint of an
/// immutable configuration, the seed and sampler that made the schedule
/// deterministic, the schema fingerprints of what went in and came out, and the
/// versions of the metric and policy the run was decided under. None of it is
/// sampled, and none of it is derived here.
struct SessionMetadata
{
    /// Stable ASCII name of the paradigm, owned by its trace writer.
    std::string_view experiment_type{};
    std::uint32_t experiment_version{1};
    std::uint64_t configuration_fingerprint{};
    /// Fingerprint of the schedule's *identity* -- seed, sampler, configuration
    /// and catalog. Two different legal schedules drawn under the same identity
    /// share it, which is exactly what makes it insufficient on its own.
    std::uint64_t schedule_fingerprint{};
    /// Fingerprint of the schedule the run actually executed, or zero for a
    /// paradigm that prepares no schedule ahead of time.
    ///
    /// Taken over the realized entries themselves rather than over what
    /// produced them. A seeded schedule is only checked for membership, bounds,
    /// and sampler version when it is accepted, so a configuration admits many
    /// legal realizations; without this, all of them record identically and a
    /// replay cannot tell which one ran.
    std::uint64_t realized_schedule_fingerprint{};
    ScheduleSeed schedule_seed{};
    SamplerVersion sampler_version{};
    /// Fingerprint of the schema the paradigm consumed, or zero when it
    /// consumed no stream.
    std::uint64_t input_schema_fingerprint{};
    /// Fingerprint of the schema the paradigm produced, or zero.
    std::uint64_t output_schema_fingerprint{};
    /// Version of the metric formula the run is summarised under, or zero.
    std::uint32_t metric_version{};
    /// Version of the assistance, selection, or presentation policy in force,
    /// or zero when the paradigm declares none.
    std::uint32_t policy_version{};
};

/// Where an encoded record goes, and the only place a record is counted.
///
/// The sink is what makes "recorder disabled" cost the same as "recorder
/// enabled" everywhere except the submission itself: the writers build exactly
/// the same records either way, and a sink with no recorder discards them after
/// building them rather than skipping the build. A paradigm therefore cannot
/// behave differently because recording is off.
class ExperimentTraceSink
{
  public:
    ExperimentTraceSink() noexcept = default;

    ExperimentTraceSink(const ExperimentTraceSink&) = delete;
    ExperimentTraceSink& operator=(const ExperimentTraceSink&) = delete;

    /// Begin one of the seven uniform kinds. Returns the nested `text` writer.
    JsonWriter& named(ControlKind kind, std::string_view name, ExperimentTimeNs time_ns) noexcept
    {
        return writer_.named(kind, name, time_ns);
    }
    /// Begin a `trials` record. Returns the nested `label` writer.
    JsonWriter& trial(const TrialRecord& record) noexcept
    {
        return writer_.trial(record);
    }
    /// Begin a `faults` record. Returns its nested `text` writer.
    JsonWriter& fault(std::string_view code, std::string_view stage,
                      ExperimentTimeNs time_ns) noexcept
    {
        return writer_.fault(code, stage, time_ns);
    }
    /// Begin the `faults` record for one abnormal condition, with the shared
    /// fields already written. Returns the nested `text` writer so a paradigm
    /// can add what only it knows.
    ///
    /// One shape for all three paradigms, because a reader looking for "what
    /// went wrong in this run" must not have to know which paradigm produced
    /// the record in order to find the answer.
    JsonWriter& abnormal(const AbnormalEvent& event) noexcept;
    /// Attach the record's float64 `value`.
    bool value(double quantity) noexcept
    {
        return writer_.value(quantity);
    }
    bool value(std::uint64_t quantity) noexcept
    {
        return writer_.value(quantity);
    }

    /// Close the record under construction and offer it. `false` means it was
    /// not recorded, which is a trace loss whatever the reason -- an oversized
    /// body, a refusing recorder, or a recorder that has stopped.
    [[nodiscard]] bool commit() noexcept;

    [[nodiscard]] std::uint64_t encoded() const noexcept
    {
        return encoded_;
    }
    [[nodiscard]] std::uint64_t recorded() const noexcept
    {
        return recorded_;
    }
    [[nodiscard]] std::uint64_t refused() const noexcept
    {
        return refused_;
    }
    /// The experiment instant of the last record offered, or zero before the
    /// first. A loss noticed during a drain is stamped with it: the loss
    /// happened at the record it was about, and stamping a fault row with an
    /// instant the session never reached would put it before the session began.
    [[nodiscard]] ExperimentTimeNs last_time_ns() const noexcept
    {
        return last_time_ns_;
    }

  private:
    friend class ExperimentSession;

    void bind(recording::NativeRecorderCore* recorder, std::uint32_t clock_domain,
              std::size_t max_body_bytes) noexcept;

    ControlRecordWriter writer_{};
    recording::NativeRecorderCore* recorder_{};
    std::uint32_t clock_domain_{};
    std::size_t max_body_bytes_{kMaxControlBodyBytes};
    /// One identity counter per control kind, indexed by the registry number.
    std::array<std::uint64_t, 12> identities_{};
    std::uint64_t encoded_{};
    std::uint64_t recorded_{};
    std::uint64_t refused_{};
    ExperimentTimeNs last_time_ns_{};
};

/// The paradigm side of the bridge.
///
/// One implementation per concrete paradigm. It owns the mapping from that
/// paradigm's trace records onto control records and nothing else: it does not
/// know whether recording is on, does not own a lifecycle, and never touches
/// the runtime.
class ExperimentTraceSource
{
  public:
    virtual ~ExperimentTraceSource() = default;

    /// Validate the paradigm configuration and its prepared schedule, *as the
    /// execution owner holds them*.
    ///
    /// An implementation reads what it validates from the controller or
    /// scheduler that will run it, never from a second copy of its own. That is
    /// the whole point: a bridge that keeps its own configuration can pass this
    /// call while the controller runs a different one, and the recording would
    /// then describe a session that never happened. It also fails when the
    /// execution owner is not prepared at all.
    [[nodiscard]] virtual ContractStatus validate() const noexcept = 0;

    /// Start the paradigm's own execution at *time_ns*.
    ///
    /// Called as bring-up step 6, after arm and before the runtime is started.
    /// A session whose start() succeeded has already made this call, so the
    /// caller has nothing left to start and cannot get the order wrong.
    [[nodiscard]] virtual streaming::StreamStatus
    start_execution(ExperimentTimeNs time_ns) noexcept = 0;

    /// Stop the paradigm's execution, as shutdown step 3.
    ///
    /// It must leave the producer quiet and must not discard what the producer
    /// already queued -- the drain that follows is what delivers it. The
    /// session does not close the controller: the caller owns it, and closing
    /// something the caller may still want to inspect is not this layer's to
    /// decide.
    virtual void stop_execution(bool aborting) noexcept = 0;

    /// Trace records the paradigm could not enqueue, counted since the source
    /// was constructed and never decreasing.
    ///
    /// This is how a loss reaches the session when the caller never sees the
    /// status that reported it -- a controller running inside a
    /// `NativeStreamRunner` returns `queue_overflow` to the runtime, not to
    /// anyone who could report it here. The session samples this and turns each
    /// increase into a loss. Inferring loss from a runtime fault instead would
    /// be wrong in both directions: the runtime faults for many reasons, and a
    /// full queue does not always fault it.
    [[nodiscard]] virtual std::uint64_t dropped_trace_count() const noexcept = 0;

    /// The session metadata this paradigm's configuration fixes.
    [[nodiscard]] virtual SessionMetadata metadata() const noexcept = 0;

    /// Write the records that describe the configuration itself.
    virtual void write_configuration(ExperimentTraceSink& sink,
                                     ExperimentTimeNs time_ns) noexcept = 0;

    /// Move at most *budget* trace records out of the paradigm's bounded queue
    /// and encode them into *sink*. Returns how many were taken.
    [[nodiscard]] virtual std::size_t drain(ExperimentTraceSink& sink,
                                            std::size_t budget) noexcept = 0;

    /// Write the records that summarise the run.
    virtual void write_summary(ExperimentTraceSink& sink, ExperimentTimeNs time_ns) noexcept = 0;

    /// Abnormal conditions the paradigm has met, counted since the source was
    /// constructed and never decreasing.
    ///
    /// Sampled by the session exactly like ::dropped_trace_count, and for the
    /// same reason: a paradigm running inside a `NativeStreamRunner` reports
    /// what it decided to the runtime, not to anyone who could act on it here.
    /// A session that learned about abnormal conditions only from what a caller
    /// passed back would report a clean run for one that ended a trial on every
    /// frame.
    [[nodiscard]] virtual AbnormalSummary abnormal_summary() const noexcept = 0;

    /// Refuse all further task input from this instant, and end or invalidate
    /// the trial in flight.
    ///
    /// This is the task side of an emergency stop, and it is deliberately not
    /// ::stop_execution. That one is shutdown step 3 and runs *after* the
    /// runtime has been stopped and joined; this one has to take effect
    /// immediately, before anything else is asked to wind down, because the
    /// whole point is that no further command is applied. It must be bounded,
    /// allocation-free, callable from any thread, and idempotent.
    ///
    /// It does not inhibit an actuator, and must not try to. Whether a device
    /// is released or inhibited is the runtime's and its `SafetyController`'s,
    /// decided from the runtime's own fault state; the session aborts the
    /// runtime and lets that path do what it already does.
    virtual void halt(ExperimentTimeNs time_ns, AbnormalCondition condition) noexcept = 0;
};

/// What a lost experiment trace record does to the session.
enum class TraceLossPolicy : std::uint8_t
{
    /// Latch the loss and abort the runtime immediately. The default, because a
    /// paradigm whose trace has a hole is a paradigm whose run cannot be
    /// replayed, and continuing produces a longer recording of a session that
    /// is already unusable.
    ///
    /// It does not end the session on the caller's behalf, and a caller-stepped
    /// paradigm has no runtime to abort at all. What it guarantees is that the
    /// loss is latched: whenever the session is then stopped, the recording is
    /// ended as an abort rather than a clean stop
    /// (::SessionOutcome::terminal_abort).
    fault = 0,
    /// Mark the trace incomplete and keep running. Legal, and never silent:
    /// the loss is recorded as a fault row and the outcome says the trace is
    /// incomplete.
    mark_incomplete = 1,
};

enum class SessionState : std::uint8_t
{
    created = 0,
    started = 1,
    stopped = 2,
    closed = 3,
    failed = 4,
};

/// Everything the session is prepared to say about itself.
struct SessionOutcome
{
    SessionState state{SessionState::created};
    /// Whether a recorder was attached at all.
    bool recorded{};
    /// Whether every required trace record reached the recorder.
    ///
    /// This is **not** a second session verdict. The recorder owns whether the
    /// recording is complete; this says only whether the experiment trace has a hole in
    /// it, and it is false the moment one record is lost.
    bool experiment_trace_complete{true};
    std::uint64_t records_encoded{};
    std::uint64_t records_recorded{};
    std::uint64_t records_refused{};
    /// Trace records lost, refusals and producer drops together.
    std::uint64_t trace_losses{};
    /// What the paradigm's own bounded queue reports it could not keep. A
    /// subset of trace_losses, kept separately so a hole on the producer's side
    /// is not confused with a recorder that refused.
    std::uint64_t producer_trace_drops{};
    /// Whether a loss escalated the session under ::TraceLossPolicy::fault.
    bool loss_faulted{};
    /// Abnormal conditions this session's paradigm reported.
    ///
    /// This session's, not the paradigm's total: the source counts for its own
    /// lifetime, and the session takes a baseline at start for the same reason
    /// it does for producer drops.
    std::uint64_t abnormal_conditions{};
    /// Of those, the ones that ended or invalidated a trial.
    std::uint64_t abnormal_trials_affected{};
    /// Whether one of them demanded that the run end.
    bool abnormal_session_aborted{};
    /// The most severe condition the paradigm reported.
    AbnormalCondition primary_abnormal{AbnormalCondition::unspecified};
    /// Whether the runtime recorded a fault of its own.
    bool has_runtime_fault{};
    /// The runtime's own first fault, taken from it rather than inferred.
    ///
    /// Preserved across shutdown: a stop or abort issued *because* of a fault
    /// returns its own status, and a session that only kept the last thing the
    /// runtime said would report the shutdown and lose what caused it.
    streaming::FaultRecord runtime_fault{};
    /// Whether the recording was ended as an abort rather than a clean stop.
    ///
    /// Settled after the final drain, so it is true for a session that looked
    /// graceful right up to the moment its last records could not be written.
    bool terminal_abort{};
    /// What the runtime's shutdown reported, or `ok` when there was none.
    streaming::StreamStatus runtime_status{streaming::StreamStatus::ok};
    /// What the recorder's close reported, or `ok` when there was none.
    recording::RecorderStatusCode recorder_status{recording::RecorderStatusCode::ok};
    /// The recorder's own status, taken once after close. Meaningful only when
    /// recorded.
    recording::RecorderStatus recorder{};
};

/// What the session needs to bring a recorder up. Every pointer is the
/// caller's; the session stores them and outlives none of them.
struct RecorderAttachment
{
    recording::NativeRecorderCore* recorder{};
    const recording::NativeRecordingPlan* plan{};
    recording::SpoolFile* spool{};
    recording::UnixClock* clock{};
    /// The same recorder is attached to the runtime through an observer that
    /// leaves the control tail open at runtime terminal.
    bool runtime_critical_edge{};
};

struct ExperimentSessionConfig
{
    /// Trace records one drain pass moves. Bounds how long the bridge holds a
    /// pass, and how much a manual pump does per call.
    ///
    /// Must be positive. Zero is refused by ::ExperimentSession::start before
    /// anything is prepared or armed: every drain loop is bounded by this
    /// budget, so a budget of zero is a session that never moves a record while
    /// reporting that it drained -- including the synchronous drain that is
    /// supposed to empty the paradigm's opening trace.
    std::size_t drain_budget{64};
    /// How long the bridge sleeps between drain passes. Zero means there is no
    /// bridge thread at all and the caller pumps, which is what a paradigm
    /// driven step by step from its own control thread wants -- and which
    /// ::ExperimentSession::attach_runtime refuses, because a runtime produces
    /// on threads no caller's pump is ordered against.
    std::uint64_t bridge_poll_nanos{200000};
    /// The clock domain the experiment's own time belongs to.
    ///
    /// Experiment time is the paradigm's monotonic nanosecond count, supplied
    /// by the caller at every step. Domain 0 is host monotonic and is the right
    /// answer only when the caller's experiment clock *is* host monotonic.
    std::uint32_t control_clock_domain{};
    /// Largest control body this session will offer. The caller sets it from
    /// the plan's `max_control_payload_bytes`; a larger body is a plan
    /// violation, and faulting the recorder over one this layer built would be
    /// this layer's bug reported as the recording's failure.
    std::size_t max_control_body_bytes{kMaxControlBodyBytes};
    TraceLossPolicy trace_loss_policy{TraceLossPolicy::fault};
};

#if defined(NEURALE_EXPERIMENTS_TEST_HOOKS)
/// Deterministic failure injection for the one bring-up step that fails by
/// throwing rather than by returning a status.
///
/// Creating the bridge thread is that step. It exists as a hook for the same
/// reason `neurale_streaming` has one for its own thread creation: a failure
/// that cannot be provoked is a failure whose unwind is only asserted by
/// reading the code. The surface and its branch are absent when native tests
/// are not built.
struct SessionTestHooks
{
    /// Throw `std::system_error` instead of creating the bridge thread.
    bool bridge_thread_fails{};
};
#endif

/// The sequence itself.
class ExperimentSession
{
  public:
    ExperimentSession(ExperimentTraceSource& source, ExperimentSessionConfig config) noexcept;
    ~ExperimentSession();

    ExperimentSession(const ExperimentSession&) = delete;
    ExperimentSession& operator=(const ExperimentSession&) = delete;

    /// Enable recording. Legal only before start().
    [[nodiscard]] streaming::StreamStatus
    enable_recording(const RecorderAttachment& attachment) noexcept;
#if defined(NEURALE_EXPERIMENTS_TEST_HOOKS)
    /// Install failure injection. The caller owns the hooks and outlives this.
    void set_test_hooks(SessionTestHooks* hooks) noexcept
    {
        test_hooks_ = hooks;
    }
#endif

    /// Drive a runtime. Legal only before start(). Without one the session
    /// orders a paradigm the caller steps itself.
    ///
    /// Refused with `realtime_configuration_failed` when
    /// ::ExperimentSessionConfig::bridge_poll_nanos is zero: a runtime produces
    /// trace asynchronously, and a session with no bridge has no drain running
    /// while it does.
    [[nodiscard]] streaming::StreamStatus
    attach_runtime(streaming::NativeStreamRunner& runner) noexcept;

    /// Bring the session up in the documented order.
    ///
    /// May throw: bringing a session up allocates and creates a thread, and
    /// both can fail that way. When it does, the session is left `failed` with
    /// everything it had already brought up unwound -- the same unwind a
    /// status-returning failure gets -- and the exception is rethrown rather
    /// than reported as a status it does not mean.
    [[nodiscard]] streaming::StreamStatus start(ExperimentTimeNs time_ns);

    /// Move whatever the paradigm has queued into the recorder. Returns how
    /// many trace records were taken. Legal at any time; the bridge thread
    /// calls it when one is configured.
    std::size_t pump() noexcept;

    /// Report what a paradigm step returned, and the experiment instant it was
    /// stepped at.
    ///
    /// `queue_overflow` is the paradigm saying its bounded trace queue could
    /// not take a record, which is exactly the loss this session may not report
    /// as a complete trace. What is *counted* is the source's own drop counter,
    /// never this call: a step reports one status however many records the
    /// refusal cost, and a session that also counted the status would count the
    /// same loss twice. *time_ns* is what the resulting fault row is stamped
    /// with; the caller has it because it just supplied it to the step. The
    /// status is returned unchanged so a caller can keep using it.
    streaming::StreamStatus note_step(streaming::StreamStatus status,
                                      ExperimentTimeNs time_ns) noexcept;

    /// Stop the run immediately because something outside the task demanded it.
    ///
    /// The order is what makes this different from ::abort. The paradigm is
    /// halted **first**, before the runtime is touched: an emergency stop is a
    /// statement about command application, and asking the runtime to wind down
    /// first would leave the task applying commands for as long as the wind-down
    /// took. Only then is the runtime aborted -- which is also what inhibits the
    /// actuator path, because that decision is the runtime's and its
    /// `SafetyController`'s and this layer does not get a second one -- and only
    /// then is the session shut down as an abort.
    ///
    /// The trial in flight is recorded as ended or invalidated with
    /// ::AbnormalCondition::emergency_stop, and no new task input is accepted
    /// after the halt. Idempotent, and never upgraded back to a graceful stop.
    [[nodiscard]] streaming::StreamStatus emergency_stop(ExperimentTimeNs time_ns,
                                                         std::string_view reason);

    /// Graceful shutdown. Idempotent.
    [[nodiscard]] streaming::StreamStatus stop(ExperimentTimeNs time_ns, std::string_view reason);
    /// Abort. Idempotent, and never upgraded back to a graceful stop.
    [[nodiscard]] streaming::StreamStatus abort(ExperimentTimeNs time_ns, std::string_view reason);
    /// Release everything and take the recorder's final status. Idempotent.
    ///
    /// From ::SessionState::started this first shuts the session down as an
    /// **abort**, and writes no summary. A summary record needs the experiment
    /// time the run actually ended at, that time belongs to the caller, and a
    /// layer that invented one would be recording a fact nobody supplied.
    [[nodiscard]] recording::RecorderStatusCode close();

    [[nodiscard]] SessionOutcome outcome() const noexcept;
    [[nodiscard]] SessionState state() const noexcept
    {
        return state_;
    }

  private:
    void bridge_main() noexcept;
    void drain_to_empty() noexcept;
    void register_loss(ExperimentTimeNs time_ns, std::string_view detail,
                       std::uint64_t count) noexcept;
    void account_refusals(ExperimentTimeNs time_ns, std::string_view detail) noexcept;
    void account_producer_drops(ExperimentTimeNs time_ns) noexcept;
    void account_abnormal(ExperimentTimeNs time_ns) noexcept;
    void write_metadata(ExperimentTimeNs time_ns) noexcept;
    void write_runtime_fault() noexcept;
    /// Keep the *first* non-ok status. A shutdown issued because of a fault
    /// returns its own status, and overwriting would report the reaction and
    /// lose the cause.
    void latch_runtime_status(streaming::StreamStatus status) noexcept;
    void latch_recorder_status(recording::RecorderStatusCode status) noexcept;
    [[nodiscard]] streaming::StreamStatus
    shut_down(ExperimentTimeNs time_ns, std::string_view reason, bool aborting, bool summarize);
    void unwind_failed_start() noexcept;
    /// Steps 1 to 8, with the lifecycle lock already held.
    [[nodiscard]] streaming::StreamStatus bring_up(ExperimentTimeNs time_ns);

    ExperimentTraceSource& source_;
    ExperimentSessionConfig config_;
    ExperimentTraceSink sink_{};

    recording::NativeRecorderCore* recorder_{};
    const recording::NativeRecordingPlan* plan_{};
    recording::SpoolFile* spool_{};
    recording::UnixClock* clock_{};
    bool runtime_critical_recorder_{};
    streaming::NativeStreamRunner* runner_{};

    std::thread bridge_{};
    std::atomic<bool> bridge_stop_{false};
    /// Serializes start, stop, abort, and close against each other. The bridge
    /// never takes it: it only drains, and the drain is safe beside a shutdown
    /// that has not yet stopped the recorder.
    mutable std::mutex lifecycle_mutex_{};
    /// Guards the sink, which the bridge thread and the shutdown both write.
    mutable std::mutex sink_mutex_{};

    SessionState state_{SessionState::created};
    ExperimentTimeNs start_time_ns_{};
    bool shutdown_latched_{};
    /// Written by whichever thread observes a loss -- the bridge, a manual
    /// pump, or the caller reporting a step -- and read by outcome() from
    /// another. Atomic because the loss path deliberately takes no lifecycle
    /// lock: it may run on the bridge thread while a shutdown holds that lock,
    /// and a loss that had to wait for a shutdown would be a loss reported
    /// after the recorder it belonged to had stopped.
    std::atomic<bool> trace_complete_{true};
    std::atomic<bool> loss_faulted_{false};
    std::atomic<std::uint64_t> trace_losses_{0};
    /// Whether the paradigm reported a condition that demands the run end.
    /// Latched, and read by the shutdown to decide the terminal intent, exactly
    /// as ::loss_faulted_ is.
    std::atomic<bool> abnormal_faulted_{false};
    /// The paradigm's abnormal totals where this session found them, and what
    /// accrued after. Guarded by sink_mutex_. Positions and counts are kept
    /// apart here for the same reason as for producer drops: the source counts
    /// for its own lifetime, and a controller reused across sessions would
    /// otherwise open by charging a clean run for a previous run's conditions.
    AbnormalSummary accounted_abnormal_{};
    std::uint64_t abnormal_conditions_{};
    std::uint64_t abnormal_trials_affected_{};
    AbnormalCondition primary_abnormal_{AbnormalCondition::unspecified};
    AbnormalPolicy primary_abnormal_policy_{AbnormalPolicy::record};
    /// Refusals already turned into a loss report. Guarded by sink_mutex_.
    std::uint64_t accounted_refusals_{};
    /// The last sample taken of the source's monotonic drop counter. Guarded
    /// by sink_mutex_. This is a *position* in that counter, not a count: the
    /// counter runs for the source's whole lifetime, so only the difference
    /// between two samples belongs to this session.
    std::uint64_t accounted_drops_{};
    /// Producer drops this session is responsible for, accumulated from those
    /// differences. Guarded by sink_mutex_.
    ///
    /// Kept apart from ::accounted_drops_ because a source may outlive a
    /// session: a controller reset and started again reports the drops of every
    /// run it has had, and reporting that total as this session's would charge
    /// a clean run for a previous run's losses.
    std::uint64_t producer_drops_{};
    /// Whether start_execution() succeeded, so shutdown knows whether there is
    /// an execution to stop. Guarded by lifecycle_mutex_.
    bool execution_started_{};
    /// Whether the recorder was started, so a failed start can end it as the
    /// abort it is. Guarded by lifecycle_mutex_.
    bool recorder_started_{};
    streaming::StreamStatus runtime_status_{streaming::StreamStatus::ok};
    /// The runtime's own first fault, taken from it before this session's own
    /// shutdown status can obscure it. Guarded by lifecycle_mutex_.
    streaming::FaultRecord runtime_fault_{};
    bool has_runtime_fault_{};
    /// Whether an emergency stop has already halted the paradigm.
    bool halted_{};
    /// The intent the recorder was actually ended under. Guarded by
    /// lifecycle_mutex_.
    bool terminal_abort_{};
#if defined(NEURALE_EXPERIMENTS_TEST_HOOKS)
    SessionTestHooks* test_hooks_{};
#endif
    recording::RecorderStatusCode recorder_status_{recording::RecorderStatusCode::ok};
    recording::RecorderStatus recorder_snapshot_{};
    bool closed_{};
};

} // namespace neurale::execution
