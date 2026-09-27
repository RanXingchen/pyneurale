/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "session.h"

#include <chrono>
#include <span>
#include <system_error>

#include <neurale/streaming/runtime.h>

#include "contract_status.h"
#include "spool_layout.h"

namespace neurale::execution
{

using namespace neurale::experiments;
namespace
{
using recording::ProducerIdentityKind;

/// The record layer repeats the recording core's producer-identity numbers so
/// that it stays compilable against the experiment contract alone. These are
/// what keep the repetition from drifting into a second registry.
static_assert(static_cast<std::uint32_t>(ControlKind::events) ==
              static_cast<std::uint32_t>(ProducerIdentityKind::events));
static_assert(static_cast<std::uint32_t>(ControlKind::trials) ==
              static_cast<std::uint32_t>(ProducerIdentityKind::trials));
static_assert(static_cast<std::uint32_t>(ControlKind::experiment_states) ==
              static_cast<std::uint32_t>(ProducerIdentityKind::experiment_states));
static_assert(static_cast<std::uint32_t>(ControlKind::commands) ==
              static_cast<std::uint32_t>(ProducerIdentityKind::commands));
static_assert(static_cast<std::uint32_t>(ControlKind::targets) ==
              static_cast<std::uint32_t>(ProducerIdentityKind::targets));
static_assert(static_cast<std::uint32_t>(ControlKind::labels) ==
              static_cast<std::uint32_t>(ProducerIdentityKind::labels));
static_assert(static_cast<std::uint32_t>(ControlKind::assistance) ==
              static_cast<std::uint32_t>(ProducerIdentityKind::assistance));
static_assert(static_cast<std::uint32_t>(ControlKind::faults) ==
              static_cast<std::uint32_t>(ProducerIdentityKind::faults));
static_assert(static_cast<std::uint32_t>(ControlKind::task_variables) ==
              static_cast<std::uint32_t>(ProducerIdentityKind::task_variables));

/// Bound on the shutdown drain. The producer is quiet by then, so the loop ends
/// on an empty queue; the bound is here so that a source which reported
/// progress it did not make cannot turn a shutdown into a hang.
constexpr std::size_t kMaxDrainPasses = 1u << 20;

} // namespace

// --- ExperimentTraceSink ----------------------------------------------------

void ExperimentTraceSink::bind(recording::NativeRecorderCore* recorder, std::uint32_t clock_domain,
                               std::size_t max_body_bytes) noexcept
{
    recorder_ = recorder;
    clock_domain_ = clock_domain;
    max_body_bytes_ = max_body_bytes == 0 ? kMaxControlBodyBytes : max_body_bytes;
}

bool ExperimentTraceSink::commit() noexcept
{
    const auto record = writer_.finish();
    ++encoded_;
    if (!record.complete || record.body.size() > max_body_bytes_)
    {
        // The record was built and did not fit. Offering the bytes anyway would
        // either fault the recorder over a plan violation this layer caused, or
        // -- worse -- record a truncated document that parses into something
        // else. It is counted as refused, which is a trace loss.
        ++refused_;
        last_time_ns_ = record.time_ns;
        return false;
    }
    if (recorder_ == nullptr)
    {
        last_time_ns_ = record.time_ns;
        return true;
    }
    const auto idx = static_cast<std::size_t>(record.kind);
    // The identity advances whether or not the record is accepted, exactly as
    // M6's own submitter does it: identity fixes order within a kind, and a gap
    // in that order is how a reader sees that something was offered and refused
    // rather than never offered at all.
    const auto identity = identities_[idx]++;
    const recording::ControlSubmission submission{
        .kind = static_cast<ProducerIdentityKind>(record.kind),
        .identity = identity,
        .clock_domain = clock_domain_,
        .time_ns = record.time_ns,
        .body = std::as_bytes(std::span<const char>{record.body.data(), record.body.size()}),
    };
    last_time_ns_ = record.time_ns;
    if (!recorder_->submit_control(submission))
    {
        ++refused_;
        return false;
    }
    ++recorded_;
    last_time_ns_ = record.time_ns;
    return true;
}

JsonWriter& ExperimentTraceSink::abnormal(const AbnormalEvent& event) noexcept
{
    // A `faults` record, with the condition's own contract name as the fault
    // code. There is no new record kind here and no new body shape: the
    // recording finalizer accepts three, and an abnormal condition is a fault
    // by every reading of what that column means.
    auto& text =
        writer_.fault(abnormal_condition_name(event.condition), kAbnormalFaultStage, event.time_ns);
    text.u64("sequence", event.sequence);
    text.u64("paradigm", event.paradigm);
    text.u64("condition", static_cast<std::uint64_t>(event.condition));
    text.u64("policy", static_cast<std::uint64_t>(event.policy));
    text.u64("response", static_cast<std::uint64_t>(event.response));
    text.u64("detail", event.detail);
    // The policy and the response are both recorded, and they are not the same
    // fact: `abort_trial` produces `trial_aborted` where a paradigm can end a
    // trial and `trial_invalidated` where its state machine owns termination.
    // A record carrying only one of them could not tell those apart.
    if (event.has_trial)
    {
        text.trial("trial", event.trial);
    }
    return text;
}

// --- ExperimentSession ------------------------------------------------------

ExperimentSession::ExperimentSession(ExperimentTraceSource& source,
                                     ExperimentSessionConfig config) noexcept
    : source_(source), config_(config)
{
}

ExperimentSession::~ExperimentSession()
{
    bool running = false;
    {
        const std::lock_guard lock(lifecycle_mutex_);
        running = state_ == SessionState::started;
    }
    if (running)
    {
        // A session that owns a bring-up order owns the unwinding of it. Ending
        // only the bridge would leave the paradigm running, the runtime
        // producing, and the recorder open -- and would leave that runtime
        // producing into a queue whose only consumer had just been joined.
        //
        // close() is the cleanup contract that already exists for this: an
        // abort with no fabricated end time, because there is no real one to
        // use. A destructor cannot ask the caller for one.
        try
        {
            static_cast<void>(close());
        }
        catch (...)
        {
            // Nothing here can report, and there is no caller left to report
            // to. What matters is that the threads below are joined rather than
            // left running past the object they point into.
        }
    }
    bridge_stop_.store(true, std::memory_order_release);
    if (bridge_.joinable())
    {
        bridge_.join();
    }
}

streaming::StreamStatus
ExperimentSession::enable_recording(const RecorderAttachment& attachment) noexcept
{
    const std::lock_guard lock(lifecycle_mutex_);
    if (state_ != SessionState::created)
    {
        return streaming::StreamStatus::invalid_state;
    }
    if (attachment.recorder == nullptr || attachment.plan == nullptr ||
        attachment.spool == nullptr || attachment.clock == nullptr)
    {
        return streaming::StreamStatus::invalid_frame;
    }
    recorder_ = attachment.recorder;
    plan_ = attachment.plan;
    spool_ = attachment.spool;
    clock_ = attachment.clock;
    runtime_critical_recorder_ = attachment.runtime_critical_edge;
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus
ExperimentSession::attach_runtime(streaming::NativeStreamRunner& runner) noexcept
{
    const std::lock_guard lock(lifecycle_mutex_);
    if (state_ != SessionState::created)
    {
        return streaming::StreamStatus::invalid_state;
    }
    if (config_.bridge_poll_nanos == 0)
    {
        // An attached runtime produces trace on its own threads, so the session
        // must own a drain that runs while it does. Without a bridge the only
        // drain is a pump the caller makes, and between two such pumps the
        // producer's bounded queue is unattended -- which is a loss the caller
        // cannot see coming and this layer cannot bound. Refusing here says so
        // at attach time rather than in a recording.
        return streaming::StreamStatus::realtime_configuration_failed;
    }
    runner_ = &runner;
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus ExperimentSession::start(ExperimentTimeNs time_ns)
{
    const std::lock_guard lock(lifecycle_mutex_);
    if (state_ != SessionState::created)
    {
        return streaming::StreamStatus::invalid_state;
    }

    // 0. This session's own configuration, before anything is touched. A zero
    //    drain budget is the one that matters: every drain loop is bounded by
    //    it, so a zero budget is a session that never moves a record while
    //    reporting that it drained -- including the synchronous drain that is
    //    what makes the opening trace's departure a fact rather than a hope.
    //    Refusing it here is refusing it before a recorder is prepared, a
    //    runtime is armed, or a paradigm is started.
    if (config_.drain_budget == 0)
    {
        state_ = SessionState::failed;
        return streaming::StreamStatus::realtime_configuration_failed;
    }

    try
    {
        return bring_up(time_ns);
    }
    catch (...)
    {
        // Bringing a session up allocates and creates a thread, and both fail
        // by throwing rather than by returning a status. The invariant this
        // layer states -- a failure at any step unwinds what came up -- has to
        // hold for those too, or the one bring-up failure nobody wrote a status
        // for would be the one that leaves safety released, a paradigm running,
        // and a recorder open.
        state_ = SessionState::failed;
        unwind_failed_start();
        throw;
    }
}

streaming::StreamStatus ExperimentSession::bring_up(ExperimentTimeNs time_ns)
{
    // 1. The paradigm configuration and its prepared schedule, before anything
    //    is allocated, attached, or armed.
    const auto contract = source_.validate();
    if (contract != ContractStatus::ok)
    {
        state_ = SessionState::failed;
        return map_contract_status(contract);
    }

    sink_.bind(recorder_, config_.control_clock_domain, config_.max_control_body_bytes);
    {
        // Take the producer's drop counter where this session found it. That
        // counter is monotonic for the source's whole lifetime by design, so a
        // session that starts from zero would charge itself for every drop a
        // previous run of the same controller had already reported.
        const std::lock_guard guard(sink_mutex_);
        accounted_drops_ = source_.dropped_trace_count();
        producer_drops_ = 0;
        accounted_abnormal_ = source_.abnormal_summary();
        abnormal_conditions_ = 0;
        abnormal_trials_affected_ = 0;
        primary_abnormal_ = AbnormalCondition::unspecified;
        primary_abnormal_policy_ = AbnormalPolicy::record;
    }

    // 2. Configure the recorder. prepare() writes no durable byte, so a failure
    //    here leaves nothing behind to clean up.
    if (recorder_ != nullptr)
    {
        latch_recorder_status(recorder_->prepare(*plan_, *spool_, *clock_));
        if (recorder_status_ != recording::RecorderStatusCode::ok)
        {
            state_ = SessionState::failed;
            return streaming::StreamStatus::consumer_failure;
        }
    }

    // 3. Prepare the runtime. Safety stays inhibited across this call; the
    //    runtime is what guarantees that, and arm() refuses if it is not.
    if (runner_ != nullptr)
    {
        const auto prepared = runner_->prepare();
        if (prepared != streaming::StreamStatus::ok)
        {
            state_ = SessionState::failed;
            latch_runtime_status(prepared);
            unwind_failed_start();
            return prepared;
        }
    }

    // 4. A control-only recorder owns its readiness gate here. A recorder on
    //    the runtime input tap is checked by runner.arm(), alongside every
    //    other critical edge, before safety is released.
    if (recorder_ != nullptr && !runtime_critical_recorder_)
    {
        latch_recorder_status(recorder_->pass_readiness_gate(true));
        if (recorder_status_ != recording::RecorderStatusCode::ok)
        {
            state_ = SessionState::failed;
            unwind_failed_start();
            return streaming::StreamStatus::consumer_failure;
        }
    }

    // 5. Arm, and only here, and only through the runtime.
    if (runner_ != nullptr)
    {
        const auto armed = runner_->arm();
        if (armed != streaming::StreamStatus::ok)
        {
            state_ = SessionState::failed;
            latch_runtime_status(armed);
            unwind_failed_start();
            return armed;
        }
    }

    // 6. Begin execution. The paradigm starts before the runtime does: the
    //    runtime is what delivers frames, and a frame that arrives at a
    //    controller which has not started is a frame the run cannot account
    //    for. It is also what makes a successful start() mean the experiment is
    //    running rather than merely startable.
    {
        const auto executing = source_.start_execution(time_ns);
        if (executing != streaming::StreamStatus::ok)
        {
            state_ = SessionState::failed;
            unwind_failed_start();
            return executing;
        }
        execution_started_ = true;
    }

    // 7. Start the recorder and write the session's opening records, then take
    //    what the paradigm's own start already produced -- synchronously, on
    //    this thread, before anything else can produce.
    if (recorder_ != nullptr)
    {
        latch_recorder_status(recorder_->start());
        if (recorder_status_ != recording::RecorderStatusCode::ok)
        {
            state_ = SessionState::failed;
            unwind_failed_start();
            return streaming::StreamStatus::consumer_failure;
        }
        recorder_started_ = true;
    }
    state_ = SessionState::started;
    start_time_ns_ = time_ns;
    {
        const std::lock_guard guard(sink_mutex_);
        write_metadata(time_ns);
        source_.write_configuration(sink_, time_ns);
    }
    // The opening trace leaves the producer's queue here rather than whenever a
    // drain first happens to run. A paradigm's start already occupies part of a
    // queue whose smallest legal capacity is one slot, so a runtime started
    // before that slot was freed could overflow it on its very first frame --
    // and a larger capacity only lowers the odds of that, it does not remove
    // the race.
    drain_to_empty();

    // 8. The drain has a home before the runtime has a producer. For an
    //    attached runtime that home is the bridge thread, which is started
    //    first; for a caller-stepped paradigm it is the caller's own pump, and
    //    there is no asynchronous producer to lose a race to.
    if (config_.bridge_poll_nanos != 0)
    {
        bridge_stop_.store(false, std::memory_order_release);
#if defined(NEURALE_EXPERIMENTS_TEST_HOOKS)
        if (test_hooks_ != nullptr && test_hooks_->bridge_thread_fails)
        {
            throw std::system_error(
                std::make_error_code(std::errc::resource_unavailable_try_again));
        }
#endif
        bridge_ = std::thread(&ExperimentSession::bridge_main, this);
    }
    if (runner_ != nullptr)
    {
        const auto started = runner_->start();
        if (started != streaming::StreamStatus::ok)
        {
            state_ = SessionState::failed;
            latch_runtime_status(started);
            write_runtime_fault();
            unwind_failed_start();
            return started;
        }
    }
    return streaming::StreamStatus::ok;
}

void ExperimentSession::unwind_failed_start() noexcept
{
    // The bridge may already be draining by the time a later step fails. It is
    // stopped first so that nothing is still writing into the sink while the
    // recorder below is closed.
    bridge_stop_.store(true, std::memory_order_release);
    if (bridge_.joinable())
    {
        bridge_.join();
    }
    if (runner_ != nullptr)
    {
        const auto runtime_state = runner_->state();
        // `prepared` is not a resting state once arm() has run: arm() releases
        // safety and leaves the runtime in `prepared`, so a start that fails
        // after arming leaves an actuator path released with nothing running on
        // it. abort() is what re-inhibits it, and asking only about `running`
        // is how that release survives a failed start.
        if (runtime_state == streaming::RuntimeState::prepared ||
            runtime_state == streaming::RuntimeState::running ||
            runtime_state == streaming::RuntimeState::stopping)
        {
            static_cast<void>(runner_->abort());
        }
        // join() refuses a runtime that never started workers, so it is asked
        // only of one that had them.
        if (runtime_state == streaming::RuntimeState::running ||
            runtime_state == streaming::RuntimeState::stopping)
        {
            static_cast<void>(runner_->join());
        }
    }
    // A start that got as far as starting the paradigm and then failed leaves
    // the paradigm running unless it is stopped here. Nothing else will: the
    // caller never started it and has no reason to believe it needs stopping.
    if (execution_started_)
    {
        source_.stop_execution(true);
        execution_started_ = false;
    }
    if (recorder_ != nullptr)
    {
        if (recorder_started_)
        {
            // A recorder that was already accepting is ended as an abort. Its
            // own close() would end it as a normal stop, which would leave a
            // recording that reads as a run that finished behind a start the
            // caller was told had failed.
            static_cast<void>(recorder_->abort("start_failed"));
            recorder_started_ = false;
        }
        static_cast<void>(recorder_->close());
        recorder_snapshot_ = recorder_->status();
        closed_ = true;
    }
}

void ExperimentSession::bridge_main() noexcept
{
    const std::chrono::nanoseconds idle{config_.bridge_poll_nanos};
    while (!bridge_stop_.load(std::memory_order_acquire))
    {
        if (pump() == 0)
        {
            std::this_thread::sleep_for(idle);
        }
    }
}

std::size_t ExperimentSession::pump() noexcept
{
    const std::lock_guard guard(sink_mutex_);
    const auto taken = source_.drain(sink_, config_.drain_budget);
    account_producer_drops(sink_.last_time_ns());
    account_abnormal(sink_.last_time_ns());
    account_refusals(sink_.last_time_ns(), "control_refused");
    return taken;
}

void ExperimentSession::account_producer_drops(ExperimentTimeNs time_ns) noexcept
{
    const auto dropped = source_.dropped_trace_count();
    if (dropped <= accounted_drops_)
    {
        return;
    }
    const auto fresh = dropped - accounted_drops_;
    accounted_drops_ = dropped;
    producer_drops_ += fresh;
    // One report for the whole increase rather than one per record. A queue
    // that refused a frame refused every observation in it, and a fault row per
    // lost record would make a recorder that is already behind fall further
    // behind for no extra information.
    register_loss(time_ns, "trace_queue_overflow", fresh);
    accounted_refusals_ = sink_.refused();
}

void ExperimentSession::account_abnormal(ExperimentTimeNs time_ns) noexcept
{
    const auto summary = source_.abnormal_summary();
    // Differences against the baseline, for the same reason the drop counter is
    // read that way: these are lifetime totals of a source that may outlive the
    // session reading them.
    if (summary.observed > accounted_abnormal_.observed)
    {
        abnormal_conditions_ += summary.observed - accounted_abnormal_.observed;
    }
    if (summary.trials_affected > accounted_abnormal_.trials_affected)
    {
        abnormal_trials_affected_ += summary.trials_affected - accounted_abnormal_.trials_affected;
    }
    if (summary.primary != AbnormalCondition::unspecified &&
        (primary_abnormal_ == AbnormalCondition::unspecified ||
         static_cast<std::uint8_t>(summary.primary_policy) >
             static_cast<std::uint8_t>(primary_abnormal_policy_)))
    {
        primary_abnormal_ = summary.primary;
        primary_abnormal_policy_ = summary.primary_policy;
    }
    accounted_abnormal_ = summary;

    if (!summary.session_aborted || abnormal_faulted_.exchange(true, std::memory_order_acq_rel))
    {
        return;
    }
    // The paradigm decided the run cannot continue. Reported here rather than
    // returned to a caller, because the paradigm that decided it may be running
    // inside the runtime, where the only thing that sees its status is the
    // runtime -- exactly the asymmetry the drop counter exists to close.
    //
    // The escalation goes through the runtime, not through this session's own
    // shutdown, for the reason ::TraceLossPolicy::fault does: this can run on
    // the bridge thread while a caller holds the lifecycle lock, and a bridge
    // that waited for that lock would stop draining when the shutdown needs it
    // most. Aborting the runtime is also what inhibits the actuator path -- by
    // the runtime's own decision, which is the only one there is.
    auto& text = sink_.fault(kSessionAbortedCode, kAbnormalFaultStage, time_ns);
    text.text("primary_condition", abnormal_condition_name(summary.primary));
    text.u64("condition", static_cast<std::uint64_t>(summary.primary));
    text.u64("observed", summary.observed);
    text.u64("trials_affected", summary.trials_affected);
    static_cast<void>(sink_.commit());
    if (runner_ != nullptr)
    {
        static_cast<void>(runner_->abort());
    }
}

void ExperimentSession::latch_runtime_status(streaming::StreamStatus status) noexcept
{
    if (runtime_status_ == streaming::StreamStatus::ok)
    {
        runtime_status_ = status;
    }
}

void ExperimentSession::latch_recorder_status(recording::RecorderStatusCode status) noexcept
{
    if (recorder_status_ == recording::RecorderStatusCode::ok)
    {
        recorder_status_ = status;
    }
}

void ExperimentSession::write_runtime_fault() noexcept
{
    if (runner_ == nullptr || has_runtime_fault_)
    {
        return;
    }
    const auto fault = runner_->primary_fault();
    if (!fault.has_value())
    {
        return;
    }
    has_runtime_fault_ = true;
    runtime_fault_ = *fault;
}

void ExperimentSession::account_refusals(ExperimentTimeNs time_ns, std::string_view detail) noexcept
{
    const auto refused = sink_.refused();
    while (accounted_refusals_ < refused)
    {
        ++accounted_refusals_;
        register_loss(time_ns, detail, 1);
    }
    // A loss report that was itself refused is already counted in the sink and
    // in the trace's incompleteness. Absorbing it here is what stops a
    // permanently refusing recorder from turning one loss into an unbounded run
    // of reports about reports.
    accounted_refusals_ = sink_.refused();
}

void ExperimentSession::drain_to_empty() noexcept
{
    for (std::size_t pass = 0; pass < kMaxDrainPasses; ++pass)
    {
        if (pump() == 0)
        {
            return;
        }
    }
}

void ExperimentSession::register_loss(ExperimentTimeNs time_ns, std::string_view detail,
                                      std::uint64_t count) noexcept
{
    if (count == 0)
    {
        return;
    }
    trace_complete_.store(false, std::memory_order_release);
    trace_losses_.fetch_add(count, std::memory_order_relaxed);

    // Reported, always. A fault row is data rather than a verdict, and writing
    // one is what keeps "the trace is incomplete" from being a claim the
    // session makes about a recording that says nothing of the kind. It is
    // offered like any other record: if the recorder refuses this one too, the
    // refusal is counted and the trace stays incomplete either way.
    auto& text = sink_.fault(kTraceLossFaultCode, kTraceLossFaultStage, time_ns);
    text.text("detail", detail);
    text.u64("lost", count);
    text.u64("losses", trace_losses_.load(std::memory_order_relaxed));
    if (!sink_.commit())
    {
        // Deliberately not recursive: a refused loss report is already counted
        // in the sink and the trace is already incomplete.
    }

    if (config_.trace_loss_policy != TraceLossPolicy::fault)
    {
        return;
    }
    loss_faulted_.store(true, std::memory_order_release);
    // Escalate through the runtime rather than through this session's own
    // shutdown. The loss can be observed on the bridge thread while a caller
    // holds the lifecycle lock, and a bridge that waited for that lock would be
    // a bridge that stopped draining exactly when the shutdown needs it most.
    if (runner_ != nullptr)
    {
        static_cast<void>(runner_->abort());
    }
}

streaming::StreamStatus ExperimentSession::note_step(streaming::StreamStatus status,
                                                     ExperimentTimeNs time_ns) noexcept
{
    if (status == streaming::StreamStatus::queue_overflow)
    {
        // The step already told the source's counter how much it cost. Reading
        // that counter here is what keeps one loss from being counted twice --
        // once by the caller who saw the status and once by the bridge that
        // samples the same counter a moment later.
        const std::lock_guard guard(sink_mutex_);
        account_producer_drops(time_ns);
    }
    return status;
}

void ExperimentSession::write_metadata(ExperimentTimeNs time_ns) noexcept
{
    const auto metadata = source_.metadata();
    auto& text = sink_.named(ControlKind::task_variables, "experiment.metadata", time_ns);
    text.text("type", metadata.experiment_type);
    text.u64("version", metadata.experiment_version);
    text.u64("trace_version", kExperimentTraceVersion);
    text.u64("configuration_fingerprint", metadata.configuration_fingerprint);
    text.u64("schedule_fingerprint", metadata.schedule_fingerprint);
    text.u64("realized_schedule_fingerprint", metadata.realized_schedule_fingerprint);
    text.u64("schedule_seed", metadata.schedule_seed);
    text.u64("sampler_version", metadata.sampler_version);
    text.u64("input_schema_fingerprint", metadata.input_schema_fingerprint);
    text.u64("output_schema_fingerprint", metadata.output_schema_fingerprint);
    text.u64("metric_version", metadata.metric_version);
    text.u64("policy_version", metadata.policy_version);
    static_cast<void>(sink_.commit());
}

streaming::StreamStatus ExperimentSession::emergency_stop(ExperimentTimeNs time_ns,
                                                          std::string_view reason)
{
    {
        const std::lock_guard lock(lifecycle_mutex_);
        if (state_ == SessionState::started && !halted_)
        {
            halted_ = true;
            // First, and before anything is asked to wind down. An emergency
            // stop is a statement about command application: halting the
            // paradigm is what stops the next command from being produced, and
            // it takes effect the moment the latch is visible. Stopping the
            // runtime first would leave the task producing commands for as long
            // as the wind-down took.
            source_.halt(time_ns, AbnormalCondition::emergency_stop);
            abnormal_faulted_.store(true, std::memory_order_release);
            // Then the runtime -- which is also what inhibits the actuator
            // path. That decision belongs to the runtime and its
            // SafetyController, and this layer does not have a second one: it
            // aborts the runtime and the existing fault path does what it
            // already does.
            if (runner_ != nullptr)
            {
                static_cast<void>(runner_->abort());
            }
            {
                // The request, not the reaction. What the task did about the
                // stop is the paradigm's own abnormal record, written under the
                // condition's own name when its trace is drained; this one says
                // a stop was asked for and what the operator called it. Writing
                // both under one code would make them indistinguishable, and a
                // reader counting stops would count this run twice.
                const std::lock_guard guard(sink_mutex_);
                auto& text = sink_.fault(kEmergencyStopRequestCode, kAbnormalFaultStage, time_ns);
                text.text("reason", reason);
                static_cast<void>(sink_.commit());
            }
        }
    }
    // And only then the ordinary abort. It is a separate call because it takes
    // the lifecycle lock, and because everything above had to happen before it.
    return shut_down(time_ns, reason, true, true);
}

streaming::StreamStatus ExperimentSession::stop(ExperimentTimeNs time_ns, std::string_view reason)
{
    return shut_down(time_ns, reason, false, true);
}

streaming::StreamStatus ExperimentSession::abort(ExperimentTimeNs time_ns, std::string_view reason)
{
    return shut_down(time_ns, reason, true, true);
}

streaming::StreamStatus ExperimentSession::shut_down(ExperimentTimeNs time_ns,
                                                     std::string_view reason, bool aborting,
                                                     bool summarize)
{
    const std::lock_guard lock(lifecycle_mutex_);
    if (shutdown_latched_)
    {
        return runtime_status_;
    }
    if (state_ != SessionState::started)
    {
        return streaming::StreamStatus::invalid_state;
    }
    // 1. Latch once. Everything below runs exactly once for this session, and a
    //    second stop or abort finds the latch and reports what the first did.
    //    This is the intent the *runtime* is taken down under; the recorder's
    //    own terminal intent is settled at step 5, after the final drain.
    shutdown_latched_ = true;
    const bool runtime_abort = aborting || loss_faulted_.load(std::memory_order_acquire) ||
                               abnormal_faulted_.load(std::memory_order_acquire);

    // 2 and 3. The runtime stops its own way: safety inhibition on an actuator
    //    path is the runtime's, and join() is what makes the trace producer
    //    quiet -- which is the precondition the drain below depends on.
    if (runner_ != nullptr)
    {
        // Sampled before the stop, because the stop changes it. join() waits
        // for workers, and a runtime that never started any has none to wait
        // for -- it would wait for a shutdown that nothing is going to signal.
        // A started session always has a started runtime, so this guard costs
        // nothing on the path that matters and bounds the one that would not
        // otherwise be bounded.
        const auto runtime_state = runner_->state();
        // Taken before the stop and again after it, and kept only the first
        // time. A runtime that faulted is stopped *because* it faulted, and the
        // stop reports on the stop; a session that overwrote what it already
        // knew would hand back "the abort succeeded" as the explanation for a
        // run that failed.
        write_runtime_fault();
        latch_runtime_status(runtime_abort ? runner_->abort() : runner_->stop());
        if (runtime_state != streaming::RuntimeState::created &&
            runtime_state != streaming::RuntimeState::prepared)
        {
            latch_runtime_status(runner_->join());
        }
        write_runtime_fault();
    }
    bridge_stop_.store(true, std::memory_order_release);
    if (bridge_.joinable())
    {
        bridge_.join();
    }
    // 3, continued. A paradigm the caller steps itself has no runtime to join,
    // so stopping its execution is the only thing that makes it quiet. It does
    // not discard what is already queued; the drain below is what delivers it.
    if (execution_started_)
    {
        source_.stop_execution(runtime_abort);
        execution_started_ = false;
    }

    // 4. Drain what the paradigm accepted, while the recorder still accepts,
    //    and write the summary into that same still-open window.
    drain_to_empty();
    {
        const std::lock_guard guard(sink_mutex_);
        if (summarize)
        {
            source_.write_summary(sink_, time_ns);
        }
        // The summary is the last thing offered, and it is offered like any
        // other record -- so a summary the recorder refuses is a lost record
        // too. Accounting only inside pump() would let exactly that one slip
        // through and leave the session reporting a complete trace whose final
        // record was never written.
        account_producer_drops(sink_.last_time_ns());
        account_abnormal(sink_.last_time_ns());
        // The runtime's preserved fault goes into the recording too, under the
        // runtime's own stage rather than the experiment's. A recording that
        // held the experiment's view of a failed run and not the runtime's
        // would be missing the half that says why.
        if (has_runtime_fault_)
        {
            auto& text = sink_.fault(kRuntimeFaultCode, kRuntimeFaultStage, sink_.last_time_ns());
            text.u64("code", static_cast<std::uint64_t>(runtime_fault_.code));
            text.u64("status", static_cast<std::uint64_t>(runtime_fault_.status));
            text.u64("stage", static_cast<std::uint64_t>(runtime_fault_.stage));
            text.u64("component_id", runtime_fault_.component_id);
            text.u64("detail", runtime_fault_.detail);
            text.u64("frame_sequence", runtime_fault_.frame_sequence);
            text.u64("detected_at_ns", runtime_fault_.detected_at_ns);
            static_cast<void>(sink_.commit());
        }
        account_refusals(sink_.last_time_ns(), "control_refused");
    }

    // 5. Settle the terminal intent now, not before the drain. A loss first
    //    seen in step 4 -- a queue that had overflowed, a refused record, a
    //    summary that did not fit -- must still reach the recording: closing it
    //    as a clean stop would make the recording claim less than the session
    //    knows.
    const bool terminal_abort = runtime_abort || loss_faulted_.load(std::memory_order_acquire) ||
                                abnormal_faulted_.load(std::memory_order_acquire);

    // Stop the recorder. Finalization happens offline; close() reports that
    // rather than claiming it.
    if (recorder_ != nullptr)
    {
        latch_recorder_status(terminal_abort ? recorder_->abort(reason) : recorder_->stop(reason));
    }
    terminal_abort_ = terminal_abort;
    state_ = SessionState::stopped;
    return runtime_status_;
}

recording::RecorderStatusCode ExperimentSession::close()
{
    bool running = false;
    {
        const std::lock_guard lock(lifecycle_mutex_);
        running = state_ == SessionState::started;
    }
    if (running)
    {
        // A running session is closed by ending it, and ending it without an
        // end time is an abort with no summary.
        static_cast<void>(shut_down(start_time_ns_, "close", true, false));
    }
    const std::lock_guard lock(lifecycle_mutex_);
    if (closed_)
    {
        return recorder_status_;
    }
    closed_ = true;
    if (recorder_ != nullptr)
    {
        latch_recorder_status(recorder_->close());
        recorder_snapshot_ = recorder_->status();
    }
    if (state_ != SessionState::failed)
    {
        state_ = SessionState::closed;
    }
    return recorder_status_;
}

SessionOutcome ExperimentSession::outcome() const noexcept
{
    const std::lock_guard lock(lifecycle_mutex_);
    const std::lock_guard guard(sink_mutex_);
    return SessionOutcome{
        .state = state_,
        .recorded = recorder_ != nullptr,
        .experiment_trace_complete = trace_complete_.load(std::memory_order_acquire),
        .records_encoded = sink_.encoded(),
        .records_recorded = sink_.recorded(),
        .records_refused = sink_.refused(),
        .trace_losses = trace_losses_.load(std::memory_order_relaxed),
        .producer_trace_drops = producer_drops_,
        .loss_faulted = loss_faulted_.load(std::memory_order_acquire),
        .abnormal_conditions = abnormal_conditions_,
        .abnormal_trials_affected = abnormal_trials_affected_,
        .abnormal_session_aborted = abnormal_faulted_.load(std::memory_order_acquire),
        .primary_abnormal = primary_abnormal_,
        .has_runtime_fault = has_runtime_fault_,
        .runtime_fault = runtime_fault_,
        .terminal_abort = terminal_abort_,
        .runtime_status = runtime_status_,
        .recorder_status = recorder_status_,
        .recorder = recorder_snapshot_,
    };
}

} // namespace neurale::execution
