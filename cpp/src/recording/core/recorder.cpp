/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "recorder.h"

#include "byte_order.h"

#include <algorithm>
#include <chrono>
#include <limits>

namespace neurale::recording
{
namespace
{

// --- the in-queue slot framing ---------------------------------------------
//
// A slot holds one whole item -- one frame with every block the plan records,
// or one discontinuity with every gap, or one control record -- already
// encoded into the spool record payloads it will become. The callback does the
// encoding because it is bounded work over data that is about to be handed
// back to the runtime; the worker then only has to frame and write it.
//
// An item is never split across slots or across transactions. A torn spool
// tail that committed a frame without its blocks would leave the item neither
// committed nor lost, and the contract has no fractional item anywhere.

inline constexpr std::uint32_t kSlotItemFrame = 1;
inline constexpr std::uint32_t kSlotItemDiscontinuity = 2;
inline constexpr std::uint32_t kSlotItemControl = 3;

inline constexpr std::size_t kSlotHeaderBytes = 16;
inline constexpr std::size_t kSlotRecordHeaderBytes = 8;

namespace slot_header
{
inline constexpr std::size_t kItemKind = 0;
inline constexpr std::size_t kRecordCount = 4;
inline constexpr std::size_t kLogicalOrdinal = 8;
} // namespace slot_header

namespace slot_record
{
inline constexpr std::size_t kKind = 0;
inline constexpr std::size_t kReserved = 2;
inline constexpr std::size_t kPayloadBytes = 4;
inline constexpr std::size_t kPayload = 8;
} // namespace slot_record

[[nodiscard]] std::size_t data_slot_bytes(const NativeRecordingPlan& plan) noexcept
{
    const auto blocks = static_cast<std::size_t>(plan.max_blocks_per_frame);
    const std::size_t frame_slot = kSlotHeaderBytes + (blocks + 1) * kSlotRecordHeaderBytes +
                                   kFramePayloadBytes + blocks * kSignalBlockHeaderPayloadBytes +
                                   static_cast<std::size_t>(plan.max_frame_payload_bytes);
    const auto gaps = static_cast<std::size_t>(plan.max_signal_gaps_per_discontinuity);
    const std::size_t discontinuity_slot = kSlotHeaderBytes + (gaps + 1) * kSlotRecordHeaderBytes +
                                           kDiscontinuityPayloadBytes +
                                           gaps * kSignalGapPayloadBytes;
    return std::max(frame_slot, discontinuity_slot);
}

[[nodiscard]] std::size_t control_slot_bytes(const NativeRecordingPlan& plan) noexcept
{
    return kSlotHeaderBytes + kSlotRecordHeaderBytes + kControlHeaderPayloadBytes +
           plan.max_control_payload_bytes;
}

/// Bytes one record costs inside a spool transaction, padding included.
[[nodiscard]] std::size_t transaction_cost(std::size_t payload_bytes) noexcept
{
    return kRecordHeaderBytes + static_cast<std::size_t>(spool_padded_length(payload_bytes));
}

[[nodiscard]] bool is_control_plane_identity(ProducerIdentityKind kind) noexcept
{
    const auto value = static_cast<std::uint32_t>(kind);
    return value >= 3U && value <= 11U;
}

[[nodiscard]] streaming::StreamStatus to_stream_status(RecorderStatusCode status) noexcept
{
    switch (status)
    {
    case RecorderStatusCode::ok:
        return streaming::StreamStatus::ok;
    case RecorderStatusCode::drain_timed_out:
        return streaming::StreamStatus::deadline_exceeded;
    case RecorderStatusCode::writer_failed:
        return streaming::StreamStatus::consumer_failure;
    case RecorderStatusCode::wrong_state:
    case RecorderStatusCode::invalid_plan:
    case RecorderStatusCode::prepare_failed:
    case RecorderStatusCode::not_ready:
    case RecorderStatusCode::not_implemented:
        return streaming::StreamStatus::invalid_state;
    }
    return streaming::StreamStatus::consumer_failure;
}

} // namespace

std::uint64_t SystemUnixClock::unix_nanos() noexcept
{
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

NativeRecorderCore::~NativeRecorderCore()
{
    // A destructor that left a thread running would outlive the queues it
    // reads. `close()` is idempotent, so calling it here is safe whatever the
    // caller already did.
    static_cast<void>(close());
    // Destruction cannot outlive borrowed spool/clock storage. Explicit close
    // is bounded and retryable; destruction waits for outstanding disk I/O.
    if (worker_.joinable())
    {
        worker_.join();
        static_cast<void>(release_resources());
    }
}

RecorderStatusCode NativeRecorderCore::prepare(const NativeRecordingPlan& plan, SpoolFile& spool,
                                               UnixClock& clock)
{
    const std::lock_guard<std::mutex> guard(lifecycle_mutex_);
    if (state_.load(std::memory_order_acquire) != RecorderLifecycleState::created)
    {
        return RecorderStatusCode::wrong_state;
    }
    if (plan.validate() != RecordingPlanStatus::ok)
    {
        return RecorderStatusCode::invalid_plan;
    }

    // `prepare()` allocates and validates only; it writes no durable byte, so
    // any failure -- a returned status or a thrown bad_alloc -- is rolled back
    // to exactly `created`, and the object may be prepared again. The spool
    // superblock is committed at the readiness gate, not here, so a run that
    // never passes the gate leaves no artifact (contract section 3.1).
    const auto rollback = [&]() noexcept
    {
        data_queue_.release_storage();
        control_queue_.release_storage();
        data_queue_capacity_.store(0, std::memory_order_release);
        control_queue_capacity_.store(0, std::memory_order_release);
        session_id_storage_.clear();
        plan_document_storage_.clear();
        plan_signals_.clear();
        spool_ = nullptr;
        clock_ = nullptr;
    };

    try
    {
        // Copy every borrowed range into storage the recorder owns. The
        // caller's plan may be a temporary, and the superblock is built from
        // these bytes at the readiness gate, long after `prepare()` returned.
        session_id_storage_.assign(plan.session_id.begin(), plan.session_id.end());
        plan_document_storage_.assign(plan.plan_document.begin(), plan.plan_document.end());
        plan_signals_.assign(plan.recorded_signals.begin(), plan.recorded_signals.end());

        plan_ = plan;
        plan_.session_id = std::string_view{session_id_storage_.data(), session_id_storage_.size()};
        plan_.plan_document = plan_document_storage_;
        plan_.recorded_signals = plan_signals_;

        spool_ = &spool;
        clock_ = &clock;

        data_queue_.reserve(plan_.frame_queue_capacity, data_slot_bytes(plan_));
        control_queue_.reserve(plan_.control_queue_capacity, control_slot_bytes(plan_));
        data_queue_capacity_.store(plan_.frame_queue_capacity, std::memory_order_release);
        control_queue_capacity_.store(plan_.control_queue_capacity, std::memory_order_release);

        SpoolSessionIdentity identity{};
        identity.session_id = plan_.session_id;
        identity.session_uuid = plan_.session_uuid;
        identity.created_unix_nanos = plan_.created_unix_nanos;
        identity.plan_document = plan_.plan_document;
        identity.plan_fingerprint = plan_.plan_fingerprint;

        const SpoolWriterLimits limits{.max_records_per_transaction =
                                           plan_.max_records_per_transaction,
                                       .max_transaction_bytes = plan_.max_transaction_bytes};
        if (writer_.allocate(spool, identity, plan_.durability_policy, limits) !=
            SpoolWriterStatus::ok)
        {
            // `allocate()` did no I/O and left the writer `constructed`, so the
            // object is exactly as it was and may be prepared again.
            rollback();
            return RecorderStatusCode::prepare_failed;
        }
    }
    catch (...)
    {
        // A thrown bad_alloc from the plan storage, the queues, or the writer's
        // own staging allocation is rolled back the same way: nothing was
        // published, so there is nothing to undo on disk.
        rollback();
        return RecorderStatusCode::prepare_failed;
    }

    spool_durability_policy_ = plan_.durability_policy;
    // The worker is not started until the readiness gate passes: a `prepared`
    // recorder holds queues and a built-but-unpublished superblock, not a
    // thread. `worker_mode_` is set here so the worker, when it is started,
    // parks until `start()`.
    worker_mode_.store(WorkerMode::parked, std::memory_order_release);
    worker_finished_.store(false, std::memory_order_release);

    state_.store(RecorderLifecycleState::prepared, std::memory_order_release);
    return RecorderStatusCode::ok;
}

RecorderStatusCode
NativeRecorderCore::pass_readiness_gate(bool critical_edges_attached_and_lossless)
{
    const std::lock_guard<std::mutex> guard(lifecycle_mutex_);
    if (state_.load(std::memory_order_acquire) != RecorderLifecycleState::prepared)
    {
        return RecorderStatusCode::wrong_state;
    }

    const auto fail_gate = [&](RecorderFaultReason reason)
    {
        latch_primary_fault(FaultOrigin::recorder, reason, nullptr,
                            clock_ != nullptr ? clock_->unix_nanos() : 0);
        escalate_to_faulted();
        release_resources();
        // A gate that did not pass holds no committed record, so whatever it
        // did write -- a superblock it committed before a worker could be
        // started, or the partial bytes of a superblock write that failed -- is
        // discarded, not left as a `finalizable` artifact the recorder then
        // denies ever producing (contract section 3.1).
        discard_empty_spool();
        state_.store(RecorderLifecycleState::failed, std::memory_order_release);
        return RecorderStatusCode::not_ready;
    };

    if (!critical_edges_attached_and_lossless)
    {
        // No superblock was committed and no worker started: a gate that did
        // not pass commits nothing. Every prepared resource is released rather
        // than merely refused, because a prepared-but-never-started recorder
        // holding queues and a spool handle is what contract section 3.1
        // forbids.
        return fail_gate(RecorderFaultReason::readiness_gate_failed);
    }

    // Disk I/O runs only on the writer, never the acquisition path. Cancellation
    // capability is reported honestly, not required: a timeout on ordinary
    // files retains the worker and buffers until I/O completes. Durability
    // policy remains a hard readiness requirement.
    if (spool_ != nullptr && !spool_->supports_durability_policy(plan_.durability_policy))
    {
        return fail_gate(RecorderFaultReason::readiness_gate_failed);
    }
    spool_backend_cancellable_.store(spool_->supports_bounded_cancel(), std::memory_order_release);

    // The readiness gate is where the spool superblock becomes durable
    // (contract section 3.1: "readiness gate passes -> commit superblock ->
    // ready"). A gate that cannot commit the superblock has not passed.
    if (writer_.commit_superblock() != SpoolWriterStatus::ok)
    {
        return fail_gate(RecorderFaultReason::spool_writer_failed);
    }
    // A session artifact now exists on disk: a valid-empty spool that a scanner
    // reports `finalizable`. `status()` reads this field rather than equating
    // "session created" with "a record was committed", because those are two
    // different questions and the readiness gate answers the first one.
    session_created_.store(true, std::memory_order_release);
    publish_spool_state();

    try
    {
        worker_running_.store(true, std::memory_order_release);
        worker_ = std::thread([this] { worker_main(); });
    }
    catch (...)
    {
        worker_running_.store(false, std::memory_order_release);
        // The superblock is durable but the worker could not be started, so the
        // recorder cannot run. The recorder is failed; the gate-failure path
        // discards the artifact it just committed so no `finalizable` spool is
        // left behind, and the caller may prepare again against a fresh file.
        return fail_gate(RecorderFaultReason::spool_writer_failed);
    }

    state_.store(RecorderLifecycleState::ready, std::memory_order_release);
    return RecorderStatusCode::ok;
}

RecorderStatusCode NativeRecorderCore::start()
{
    const std::lock_guard<std::mutex> guard(lifecycle_mutex_);
    if (state_.load(std::memory_order_acquire) != RecorderLifecycleState::ready)
    {
        return RecorderStatusCode::wrong_state;
    }
    worker_mode_.store(WorkerMode::running, std::memory_order_release);
    state_.store(RecorderLifecycleState::recording, std::memory_order_release);
    accepting_.store(true, std::memory_order_release);
    return RecorderStatusCode::ok;
}

RecorderStatusCode NativeRecorderCore::stop(std::string_view reason)
{
    return shut_down(RequestedTerminalIntent::normal, reason);
}

RecorderStatusCode NativeRecorderCore::abort(std::string_view reason)
{
    return shut_down(RequestedTerminalIntent::aborted, reason);
}

RecorderStatusCode NativeRecorderCore::finalize()
{
    // Not implemented yet. Saying so is the point: a `not_implemented` here
    // keeps `finalization_required` honest, where a silent success would
    // publish a session that does not exist.
    return RecorderStatusCode::not_implemented;
}

RecorderStatusCode NativeRecorderCore::abandon_finalization()
{
    return RecorderStatusCode::not_implemented;
}

RecorderStatusCode NativeRecorderCore::shut_down(RequestedTerminalIntent intent,
                                                 std::string_view reason)
{
    const std::lock_guard<std::mutex> guard(lifecycle_mutex_);
    const auto state = state_.load(std::memory_order_acquire);

    // `stop()` and `abort()` are a state error where there is nothing to stop.
    // A caller stopping a recorder that never started has a bug, and the
    // contract surfaces it rather than absorbing it into a silent no-op.
    if (state == RecorderLifecycleState::created || state == RecorderLifecycleState::prepared ||
        state == RecorderLifecycleState::ready)
    {
        return RecorderStatusCode::wrong_state;
    }
    // A prior shutdown already reached a terminal lifecycle state. Return the
    // outcome it recorded without restarting the drain.
    if (state == RecorderLifecycleState::stopped || state == RecorderLifecycleState::closed)
    {
        return RecorderStatusCode::ok;
    }
    if (state == RecorderLifecycleState::failed)
    {
        return drain_timed_out_.load(std::memory_order_acquire)
                   ? RecorderStatusCode::drain_timed_out
                   : RecorderStatusCode::writer_failed;
    }

    bool expected = false;
    if (terminal_intent_latched_.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
    {
        requested_terminal_intent_.store(intent, std::memory_order_release);
        if (intent == RequestedTerminalIntent::aborted)
        {
            capture_outcome_.store(CaptureOutcome::aborted, std::memory_order_release);
            if (effective_session_outcome_.load(std::memory_order_acquire) ==
                EffectiveSessionOutcome::normal)
            {
                effective_session_outcome_.store(EffectiveSessionOutcome::aborted,
                                                 std::memory_order_release);
            }
        }
        const auto copied = std::min(reason.size(), terminal_reason_.size());
        std::copy_n(reason.begin(), copied, terminal_reason_.begin());
        terminal_reason_size_ = copied;
    }
    else if (terminal_reason_size_ == 0 &&
             requested_terminal_intent_.load(std::memory_order_acquire) == intent)
    {
        const auto copied = std::min(reason.size(), terminal_reason_.size());
        std::copy_n(reason.begin(), copied, terminal_reason_.begin());
        terminal_reason_size_ = copied;
    }

    // Stop accepting *before* the worker is told to drain, so that "the queues
    // are empty" is a statement the worker can act on. A fault may already have
    // moved `recording -> draining` and started the worker draining; in that
    // case the worker is already draining and we join the drain in progress
    // rather than starting a new one. The terminal intent a fault latched is
    // preserved: the CAS above failed, so a `stop("reason")` after a fault does
    // not rewrite the fault's intent or reason.
    accepting_.store(false, std::memory_order_seq_cst);
    if (state == RecorderLifecycleState::recording)
    {
        state_.store(RecorderLifecycleState::draining, std::memory_order_release);
        worker_mode_.store(WorkerMode::draining, std::memory_order_release);
    }

    const auto poll = std::chrono::nanoseconds(plan_.worker_idle_poll_nanos);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::nanoseconds(plan_.drain_timeout_nanos);
    while (!worker_finished_.load(std::memory_order_acquire))
    {
        if (std::chrono::steady_clock::now() >= deadline)
        {
            break;
        }
        std::this_thread::sleep_for(poll);
    }

    if (!worker_finished_.load(std::memory_order_acquire))
    {
        // Latch a timeout and request cancellation where supported. A disk
        // request may remain in flight: no success, detach or early free is
        // allowed. Once it returns, the worker can seal the committed prefix.
        drain_timed_out_.store(true, std::memory_order_release);
        latch_primary_fault(FaultOrigin::recorder, RecorderFaultReason::drain_timeout, nullptr,
                            clock_ != nullptr ? clock_->unix_nanos() : 0);
        escalate_to_faulted();
        if (spool_ != nullptr)
        {
            spool_->request_cancel();
        }
        if (!spool_backend_cancellable_.load(std::memory_order_acquire))
        {
            // The worker may still own a transaction buffer inside disk I/O.
            // Keep all resources and the spool alive; close() can be retried.
            queue_storage_release_deferred_.store(true, std::memory_order_release);
            state_.store(RecorderLifecycleState::failed, std::memory_order_release);
            return RecorderStatusCode::drain_timed_out;
        }
        const auto second_deadline =
            std::chrono::steady_clock::now() + std::chrono::nanoseconds(plan_.drain_timeout_nanos);
        while (!worker_finished_.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < second_deadline)
        {
            std::this_thread::sleep_for(poll);
        }
        release_resources();
        state_.store(RecorderLifecycleState::failed, std::memory_order_release);
        return RecorderStatusCode::drain_timed_out;
    }

    if (writer_.state() == SpoolWriterState::failed ||
        !spool_ended_cleanly_.load(std::memory_order_acquire))
    {
        // The drain finished but the spool did not close cleanly: the session
        // is incomplete and the spool is retained for the offline finalizer.
        release_resources();
        state_.store(RecorderLifecycleState::failed, std::memory_order_release);
        return RecorderStatusCode::writer_failed;
    }

    state_.store(RecorderLifecycleState::stopped, std::memory_order_release);
    return RecorderStatusCode::ok;
}

RecorderStatusCode NativeRecorderCore::close()
{
    {
        const std::lock_guard<std::mutex> guard(lifecycle_mutex_);
        const auto state = state_.load(std::memory_order_acquire);
        if (state == RecorderLifecycleState::closed || state == RecorderLifecycleState::failed)
        {
            // Idempotent, and the one place a `failed` recorder's resources are
            // reclaimed: a drain that exceeded its bound left the worker
            // running on purpose (see `shut_down`).
            return release_resources() ? RecorderStatusCode::ok
                                       : RecorderStatusCode::drain_timed_out;
        }
        if (state == RecorderLifecycleState::created)
        {
            state_.store(RecorderLifecycleState::closed, std::memory_order_release);
            return RecorderStatusCode::ok;
        }
    }

    const auto state = state_.load(std::memory_order_acquire);
    if (state == RecorderLifecycleState::recording || state == RecorderLifecycleState::draining)
    {
        // Exactly as if a graceful stop had been called first. A shutdown
        // already in progress is waited for, not restarted.
        const auto shutdown = shut_down(RequestedTerminalIntent::normal, "close");
        if (shutdown == RecorderStatusCode::drain_timed_out ||
            shutdown == RecorderStatusCode::writer_failed)
        {
            return shutdown;
        }
    }

    const std::lock_guard<std::mutex> guard(lifecycle_mutex_);
    if (state_.load(std::memory_order_acquire) == RecorderLifecycleState::failed)
    {
        return release_resources() ? RecorderStatusCode::ok : RecorderStatusCode::drain_timed_out;
    }
    if (!release_resources())
    {
        state_.store(RecorderLifecycleState::failed, std::memory_order_release);
        return RecorderStatusCode::drain_timed_out;
    }
    // `prepared`/`ready` -> close holds no committed record, so a spool that
    // carries a committed superblock from the readiness gate is discarded
    // rather than left as a `finalizable` artifact the recorder reports no
    // session for (contract section 3.1). `stopped` -> close holds a committed
    // session-end, so this is a no-op there.
    discard_empty_spool();
    state_.store(RecorderLifecycleState::closed, std::memory_order_release);
    return RecorderStatusCode::ok;
}

streaming::StreamStatus NativeRecorderCore::ready_for_runtime() noexcept
{
    try
    {
        return to_stream_status(pass_readiness_gate(true));
    }
    catch (...)
    {
        raise_recorder_fault(RecorderFaultReason::readiness_gate_failed, 0);
        return streaming::StreamStatus::invalid_state;
    }
}

streaming::StreamStatus NativeRecorderCore::start_observing() noexcept
{
    try
    {
        return to_stream_status(start());
    }
    catch (...)
    {
        raise_recorder_fault(RecorderFaultReason::readiness_gate_failed, 0);
        return streaming::StreamStatus::invalid_state;
    }
}

streaming::StreamStatus NativeRecorderCore::health() const noexcept
{
    const auto current = state_.load(std::memory_order_acquire);
    if (current == RecorderLifecycleState::failed ||
        (fault_published_.load(std::memory_order_acquire) &&
         reserved_fault_.origin == FaultOrigin::recorder))
    {
        return streaming::StreamStatus::consumer_failure;
    }
    return current == RecorderLifecycleState::ready ||
                   current == RecorderLifecycleState::recording ||
                   current == RecorderLifecycleState::draining ||
                   current == RecorderLifecycleState::stopped
               ? streaming::StreamStatus::ok
               : streaming::StreamStatus::invalid_state;
}

streaming::StreamStatus
NativeRecorderCore::observe_accepted(streaming::FrameView frame,
                                     streaming::RuntimeAcceptance acceptance) noexcept
{
    return observe_runtime_frame(frame, acceptance);
}

streaming::StreamStatus
NativeRecorderCore::handle_accepted_discontinuity(const streaming::Discontinuity& discontinuity,
                                                  streaming::RuntimeAcceptance acceptance) noexcept
{
    return observe_runtime_discontinuity(discontinuity, acceptance);
}

void NativeRecorderCore::note_rejected_before_acceptance(
    streaming::RejectedMessage message) noexcept
{
    note_edge_rejection(message.kind == streaming::AcceptedMessageKind::frame
                            ? ProducerIdentityKind::frame
                            : ProducerIdentityKind::discontinuity,
                        message.frame_sequence);
}

void NativeRecorderCore::publish_primary_fault(const streaming::FaultRecord& fault) noexcept
{
    publish_runtime_fault(fault);
}

streaming::StreamStatus
NativeRecorderCore::drain(streaming::RuntimeTerminalNotice terminal) noexcept
{
    static constexpr std::string_view kEndOfStream{"end_of_stream"};
    static constexpr std::string_view kStop{"stop"};
    static constexpr std::string_view kAbort{"abort"};
    static constexpr std::string_view kFault{"fault"};
    try
    {
        switch (terminal.reason)
        {
        case streaming::RuntimeTerminalReason::end_of_stream:
            return to_stream_status(shut_down(RequestedTerminalIntent::normal, kEndOfStream));
        case streaming::RuntimeTerminalReason::stop:
            return to_stream_status(shut_down(RequestedTerminalIntent::normal, kStop));
        case streaming::RuntimeTerminalReason::abort:
            return to_stream_status(shut_down(RequestedTerminalIntent::aborted, kAbort));
        case streaming::RuntimeTerminalReason::fault:
            return to_stream_status(shut_down(RequestedTerminalIntent::fault, kFault));
        }
    }
    catch (...)
    {
        raise_recorder_fault(RecorderFaultReason::spool_writer_failed, terminal.requested_at_ns);
    }
    return streaming::StreamStatus::consumer_failure;
}

void NativeRecorderCore::cancel() noexcept
{
    const auto current = state_.load(std::memory_order_acquire);
    if (current == RecorderLifecycleState::stopped || current == RecorderLifecycleState::closed ||
        current == RecorderLifecycleState::failed)
    {
        return;
    }
    if (current == RecorderLifecycleState::created || current == RecorderLifecycleState::prepared ||
        current == RecorderLifecycleState::ready)
    {
        try
        {
            static_cast<void>(close());
        }
        catch (...)
        {
        }
        return;
    }
    drain_timed_out_.store(true, std::memory_order_release);
    raise_recorder_fault(RecorderFaultReason::drain_timeout, 0);
    if (spool_ != nullptr)
    {
        spool_->request_cancel();
    }
}

streaming::StreamStatus NativeRecorderCore::reset() noexcept
{
    // A native recording session is single-use. A new runtime session needs a
    // newly prepared recorder and session identity.
    return streaming::StreamStatus::invalid_state;
}

bool NativeRecorderCore::release_resources() noexcept
{
    // Exactly one transition into `closed` or `failed` releases, and calling
    // it twice must be free -- `close()` is idempotent in every state.
    if (resources_released_.load(std::memory_order_acquire))
    {
        return true;
    }
    // Stop accepting here too, and sequentially: this is the store a producer's
    // post-registration re-read of `accepting_` is paired against, and the
    // reason the counter read further down cannot miss a producer that is
    // about to take a queue span.
    accepting_.store(false, std::memory_order_seq_cst);
    worker_mode_.store(WorkerMode::stopping, std::memory_order_release);
    if (worker_.joinable())
    {
        const auto poll = std::chrono::nanoseconds(plan_.worker_idle_poll_nanos);
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::nanoseconds(plan_.drain_timeout_nanos);
        while (worker_running_.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline)
        {
            std::this_thread::sleep_for(poll);
        }
        if (worker_running_.load(std::memory_order_acquire) && spool_ != nullptr)
        {
            // Only now, and only if the worker did not finish on its own. A
            // backend is free to latch its cancel, and issuing one on the
            // ordinary path would then break the `truncate` that discards an
            // empty spool a few lines later in `close()`.
            spool_->request_cancel();
            const auto cancel_deadline = std::chrono::steady_clock::now() +
                                         std::chrono::nanoseconds(plan_.drain_timeout_nanos);
            while (worker_running_.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < cancel_deadline)
            {
                std::this_thread::sleep_for(poll);
            }
        }
        if (worker_running_.load(std::memory_order_acquire))
        {
            queue_storage_release_deferred_.store(true, std::memory_order_release);
            return false;
        }
        worker_.join();
    }
    // Cancel is a no-op on a sealed writer and never rewrites a successful
    // outcome, so this is safe on every path into `closed` and `failed`.
    writer_.cancel();
    // Retire first, then look: a producer that leaves after this store sees the
    // flag and frees the storage itself, and a producer that registered before
    // it is visible to the load inside. Sequential consistency is what makes
    // "exactly one of those two" true rather than merely likely.
    queue_storage_retired_.store(true, std::memory_order_seq_cst);
    release_queue_storage_if_quiescent();
    resources_released_.store(true, std::memory_order_release);
    return true;
}

void NativeRecorderCore::release_queue_storage_if_quiescent() noexcept
{
    if (!queue_storage_retired_.load(std::memory_order_seq_cst))
    {
        // The ordinary case: a producer leaving a recorder that is still
        // running. Nothing to reclaim, and nothing allocated or freed on the
        // callback's thread.
        return;
    }
    if (active_data_submissions_.load(std::memory_order_seq_cst) != 0 ||
        active_control_submissions_.load(std::memory_order_seq_cst) != 0)
    {
        // A producer is inside the recorder holding a span into this storage.
        // Freeing it now is a use-after-free and waiting for it without a bound
        // is what contract section 4.7 forbids, so this does neither: the free
        // belongs to whichever producer leaves last. Recorded, because a caller
        // whose callback was still running at shutdown should be able to see
        // that from the status rather than infer it.
        queue_storage_release_deferred_.store(true, std::memory_order_release);
        return;
    }

    bool expected = false;
    if (!queue_storage_released_.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
    {
        return;
    }
    data_queue_.release_storage();
    control_queue_.release_storage();
    data_queue_capacity_.store(0, std::memory_order_release);
    control_queue_capacity_.store(0, std::memory_order_release);
}

void NativeRecorderCore::discard_empty_spool() noexcept
{
    // Contract section 3.1: a readiness-gate failure and a `close()` from
    // `prepared`/`ready` discard a spool that holds no committed record, so a
    // scanner never sees a `finalizable` artifact the recorder says does not
    // exist. A spool that holds a committed record is retained under section 7
    // for the offline finalizer, so this is a no-op on a
    // `recording -> stop -> stopped -> close` run. The readiness gate sets
    // `session_created_` when it commits the superblock; a successful discard
    // clears it, because the artifact it described is gone.
    if (spool_ == nullptr)
    {
        return;
    }
    if (spool_committed_transactions_.load(std::memory_order_acquire) != 0)
    {
        return;
    }
    if (spool_->truncate(0).complete())
    {
        session_created_.store(false, std::memory_order_release);
    }
}

// ---------------------------------------------------------------------------
// the critical callback
// ---------------------------------------------------------------------------

streaming::StreamStatus
NativeRecorderCore::reject_data_after_close(streaming::StreamStatus status) noexcept
{
    // Submitted late (contract section 1.3): never accepted by anything, not a
    // recording loss, and it must not change the verdict of a sealed session.
    rejected_after_close_data_.fetch_add(1, std::memory_order_relaxed);
    return status;
}

streaming::StreamStatus NativeRecorderCore::fail_data_item(std::uint64_t ordinal,
                                                           RecorderFaultReason reason,
                                                           streaming::StreamStatus status) noexcept
{
    failed_between_runtime_and_recorder_.fetch_add(1, std::memory_order_relaxed);
    static_cast<void>(
        data_first_loss_.latch(PositionTag::ordinal, ProducerIdentityKind::none, ordinal, 0));
    accepting_.store(false, std::memory_order_seq_cst);
    latch_primary_fault(FaultOrigin::recorder, reason, nullptr, 0);
    escalate_to_faulted();
    return status;
}

streaming::StreamStatus
NativeRecorderCore::observe_frame(const streaming::FrameView& frame) noexcept
{
    if (!accepting())
    {
        if (state_.load(std::memory_order_acquire) < RecorderLifecycleState::recording)
        {
            // Before `start()` there is no session boundary to be late for.
            // That is a caller bug the contract surfaces rather than counts.
            return streaming::StreamStatus::invalid_state;
        }
        return reject_data_after_close(streaming::StreamStatus::stopped);
    }
    return observe_runtime_frame(
        frame, {.data_message_ordinal = next_data_ordinal_.load(std::memory_order_relaxed),
                .accepted_at_ns = frame.header.host_received_ns});
}

streaming::StreamStatus
NativeRecorderCore::observe_runtime_frame(const streaming::FrameView& frame,
                                          streaming::RuntimeAcceptance acceptance) noexcept
{
    const auto ordinal = acceptance.data_message_ordinal;
    const auto expected = next_data_ordinal_.fetch_add(1, std::memory_order_relaxed);
    runtime_accepted_.fetch_add(1, std::memory_order_relaxed);
    if (ordinal != expected)
    {
        return fail_data_item(ordinal, RecorderFaultReason::plan_violation,
                              streaming::StreamStatus::invalid_frame);
    }
    if (!accepting())
    {
        return fail_data_item(ordinal, RecorderFaultReason::edge_rejection,
                              streaming::StreamStatus::stopped);
    }
    // Register this in-flight submission before touching the queue, so a
    // shutdown that has started (or is about to start) cannot free the span
    // this producer is about to hold. The guard decrements on every exit path
    // below -- a validation fault, a saturated queue, and a normal publish all
    // return through it -- so the counter is zero once the producer has passed
    // its `publish()`, the only point it held a queue span.
    SubmissionGuard data_guard{*this, active_data_submissions_};
    if (!accepting())
    {
        // A shutdown set accepting=false between the fast-path check and the
        // registration. Back out as a late submission -- the same answer a
        // callback after close gets -- rather than publish into a recorder
        // that is draining, and let the guard's destructor drop the counter.
        return fail_data_item(ordinal, RecorderFaultReason::edge_rejection,
                              streaming::StreamStatus::stopped);
    }

    // --- bounded validation, before a single byte is copied ----------------

    if (frame.header.session_id != plan_.native_session_id ||
        frame.header.schema_id != plan_.native_schema_id ||
        frame.blocks.size() != frame.header.signal_block_count ||
        frame.blocks.size() > plan_.max_blocks_per_frame ||
        frame.payload.size() > plan_.max_frame_payload_bytes)
    {
        // The schema id is the one integer compare that stops a frame the plan
        // did not compile against reaching stage 2: a spool that recorded it
        // would carry a schema its superblock plan document does not name, and
        // exact replay/finalization against that plan would be impossible.
        return fail_data_item(ordinal, RecorderFaultReason::plan_violation,
                              streaming::StreamStatus::invalid_frame);
    }

    std::uint32_t recorded_blocks = 0;
    for (const auto& block : frame.blocks)
    {
        if (block.payload_byte_count > frame.payload.size() ||
            block.payload_offset > frame.payload.size() - block.payload_byte_count)
        {
            return fail_data_item(ordinal, RecorderFaultReason::plan_violation,
                                  streaming::StreamStatus::invalid_frame);
        }
        const auto* planned = plan_.find_signal(block.signal_id);
        if (planned == nullptr)
        {
            // Not a violation: a partial-coverage plan deliberately records a
            // subset, and that decision was made by the compiler.
            continue;
        }
        if (block.payload_byte_count > planned->max_block_bytes ||
            block.n_samples > planned->max_block_samples)
        {
            return fail_data_item(ordinal, RecorderFaultReason::plan_violation,
                                  streaming::StreamStatus::invalid_frame);
        }
        ++recorded_blocks;
    }

    // --- copy into storage the recorder owns -------------------------------

    auto slot = data_queue_.try_acquire();
    if (slot.empty())
    {
        // Lossless until fault. There is no drop policy to consult.
        return fail_data_item(ordinal, RecorderFaultReason::data_queue_saturated,
                              streaming::StreamStatus::queue_overflow);
    }

    const auto frame_ordinal = next_frame_ordinal_.fetch_add(1, std::memory_order_relaxed);
    store_u32le(slot, slot_header::kItemKind, kSlotItemFrame);
    store_u32le(slot, slot_header::kRecordCount, recorded_blocks + 1U);
    store_u64le(slot, slot_header::kLogicalOrdinal, ordinal);
    std::size_t cursor = kSlotHeaderBytes;

    store_u16le(slot, cursor + slot_record::kKind, static_cast<std::uint16_t>(RecordKind::frame));
    store_u16le(slot, cursor + slot_record::kReserved, 0);
    store_u32le(slot, cursor + slot_record::kPayloadBytes,
                static_cast<std::uint32_t>(kFramePayloadBytes));
    FramePayloadFields frame_fields{};
    frame_fields.data_message_ordinal = ordinal;
    frame_fields.native_session_id = frame.header.session_id;
    frame_fields.frame_sequence = frame.header.sequence;
    frame_fields.frame_ordinal = frame_ordinal;
    frame_fields.host_received_ns = frame.header.host_received_ns;
    frame_fields.source_tick = frame.header.source_tick;
    frame_fields.valid_until_ns = frame.header.valid_until_ns;
    frame_fields.total_payload_byte_count = frame.payload.size();
    frame_fields.runtime_accepted_host_time_ns = acceptance.accepted_at_ns;
    frame_fields.native_schema_id = frame.header.schema_id;
    frame_fields.source_clock_domain = frame.header.source_clock_domain;
    frame_fields.frame_flags = static_cast<std::uint32_t>(frame.header.flags);
    frame_fields.signal_block_count = frame.header.signal_block_count;
    frame_fields.recorded_signal_block_count = recorded_blocks;
    encode_frame_payload(frame_fields,
                         slot.subspan(cursor + slot_record::kPayload).first<kFramePayloadBytes>());
    cursor += kSlotRecordHeaderBytes + kFramePayloadBytes;

    std::uint32_t block_idx = 0;
    for (const auto& block : frame.blocks)
    {
        const auto* planned = plan_.find_signal(block.signal_id);
        if (planned == nullptr)
        {
            ++block_idx;
            continue;
        }
        const auto payload_bytes = static_cast<std::size_t>(block.payload_byte_count);
        const auto record_bytes = kSignalBlockHeaderPayloadBytes + payload_bytes;
        store_u16le(slot, cursor + slot_record::kKind,
                    static_cast<std::uint16_t>(RecordKind::signal_block));
        store_u16le(slot, cursor + slot_record::kReserved, 0);
        store_u32le(slot, cursor + slot_record::kPayloadBytes,
                    static_cast<std::uint32_t>(record_bytes));

        SignalBlockPayloadFields block_fields{};
        block_fields.signal_block_ordinal =
            next_signal_block_ordinal_.fetch_add(1, std::memory_order_relaxed);
        block_fields.data_message_ordinal = ordinal;
        block_fields.frame_ordinal = frame_ordinal;
        block_fields.sample_idx_start = block.sample_idx_start;
        block_fields.last_sample_idx = block.last_sample_idx;
        block_fields.device_tick_start = block.device_tick_start;
        block_fields.observation_time_start_ns = block.observation_time_start_ns;
        block_fields.payload_offset = block.payload_offset;
        block_fields.payload_byte_count = block.payload_byte_count;
        block_fields.clock_sync_device_tick_reference = block.clock_sync.device_tick_reference;
        block_fields.clock_sync_host_time_reference_ns = block.clock_sync.host_time_reference_ns;
        block_fields.clock_sync_rate_numerator = block.clock_sync.device_tick_rate.numerator;
        block_fields.clock_sync_rate_denominator = block.clock_sync.device_tick_rate.denominator;
        block_fields.clock_sync_uncertainty_ns = block.clock_sync.uncertainty_ns;
        block_fields.block_idx_in_frame = block_idx;
        block_fields.native_signal_id = block.signal_id;
        block_fields.n_samples = block.n_samples;
        block_fields.clock_sync_clock_domain = block.clock_sync.clock_domain;
        block_fields.clock_sync_generation = block.clock_sync.generation;
        block_fields.clock_sync_flags = static_cast<std::uint32_t>(block.clock_sync.flags);
        encode_signal_block_payload(
            block_fields,
            slot.subspan(cursor + slot_record::kPayload).first<kSignalBlockHeaderPayloadBytes>());
        const auto source =
            frame.payload.subspan(static_cast<std::size_t>(block.payload_offset), payload_bytes);
        std::copy(source.begin(), source.end(),
                  slot.begin() + static_cast<std::ptrdiff_t>(cursor + slot_record::kPayload +
                                                             kSignalBlockHeaderPayloadBytes));
        cursor += kSlotRecordHeaderBytes + record_bytes;
        payload_bytes_copied_.fetch_add(payload_bytes, std::memory_order_relaxed);
        ++block_idx;
    }

    data_queue_.publish(cursor);
    recorder_accepted_.fetch_add(1, std::memory_order_relaxed);
    frames_accepted_.fetch_add(1, std::memory_order_relaxed);
    signal_blocks_recorded_.fetch_add(recorded_blocks, std::memory_order_relaxed);
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus
NativeRecorderCore::observe_discontinuity(const streaming::Discontinuity& discontinuity) noexcept
{
    if (!accepting())
    {
        if (state_.load(std::memory_order_acquire) < RecorderLifecycleState::recording)
        {
            return streaming::StreamStatus::invalid_state;
        }
        return reject_data_after_close(streaming::StreamStatus::stopped);
    }
    return observe_runtime_discontinuity(
        discontinuity, {.data_message_ordinal = next_data_ordinal_.load(std::memory_order_relaxed),
                        .accepted_at_ns = 0});
}

streaming::StreamStatus
NativeRecorderCore::observe_runtime_discontinuity(const streaming::Discontinuity& discontinuity,
                                                  streaming::RuntimeAcceptance acceptance) noexcept
{
    const auto ordinal = acceptance.data_message_ordinal;
    const auto expected = next_data_ordinal_.fetch_add(1, std::memory_order_relaxed);
    runtime_accepted_.fetch_add(1, std::memory_order_relaxed);
    if (ordinal != expected)
    {
        return fail_data_item(ordinal, RecorderFaultReason::plan_violation,
                              streaming::StreamStatus::invalid_frame);
    }
    if (!accepting())
    {
        return fail_data_item(ordinal, RecorderFaultReason::edge_rejection,
                              streaming::StreamStatus::stopped);
    }
    // Same in-flight guard as `observe_frame`: the span this producer holds
    // between `try_acquire()` and `publish()` must outlive any shutdown that
    // started concurrently. See `observe_frame` for the double-check rationale.
    SubmissionGuard data_guard{*this, active_data_submissions_};
    if (!accepting())
    {
        return fail_data_item(ordinal, RecorderFaultReason::edge_rejection,
                              streaming::StreamStatus::stopped);
    }

    if (discontinuity.session_id != plan_.native_session_id ||
        discontinuity.signal_gaps.size() > plan_.max_signal_gaps_per_discontinuity)
    {
        return fail_data_item(ordinal, RecorderFaultReason::plan_violation,
                              streaming::StreamStatus::invalid_frame);
    }

    auto slot = data_queue_.try_acquire();
    if (slot.empty())
    {
        return fail_data_item(ordinal, RecorderFaultReason::data_queue_saturated,
                              streaming::StreamStatus::queue_overflow);
    }

    const auto n_gaps = static_cast<std::uint32_t>(discontinuity.signal_gaps.size());
    store_u32le(slot, slot_header::kItemKind, kSlotItemDiscontinuity);
    store_u32le(slot, slot_header::kRecordCount, n_gaps + 1U);
    store_u64le(slot, slot_header::kLogicalOrdinal, ordinal);
    std::size_t cursor = kSlotHeaderBytes;

    store_u16le(slot, cursor + slot_record::kKind,
                static_cast<std::uint16_t>(RecordKind::discontinuity));
    store_u16le(slot, cursor + slot_record::kReserved, 0);
    store_u32le(slot, cursor + slot_record::kPayloadBytes,
                static_cast<std::uint32_t>(kDiscontinuityPayloadBytes));
    DiscontinuityPayloadFields fields{};
    fields.data_message_ordinal = ordinal;
    fields.native_session_id = discontinuity.session_id;
    fields.previous_frame_sequence = discontinuity.previous_frame_sequence;
    fields.actual_frame_sequence = discontinuity.actual_frame_sequence;
    fields.runtime_accepted_host_time_ns = acceptance.accepted_at_ns;
    fields.signal_gap_count = n_gaps;
    fields.reason = static_cast<std::uint32_t>(discontinuity.reason);
    encode_discontinuity_payload(
        fields, slot.subspan(cursor + slot_record::kPayload).first<kDiscontinuityPayloadBytes>());
    cursor += kSlotRecordHeaderBytes + kDiscontinuityPayloadBytes;

    std::uint32_t gap_idx = 0;
    for (const auto& gap : discontinuity.signal_gaps)
    {
        store_u16le(slot, cursor + slot_record::kKind,
                    static_cast<std::uint16_t>(RecordKind::signal_gap));
        store_u16le(slot, cursor + slot_record::kReserved, 0);
        store_u32le(slot, cursor + slot_record::kPayloadBytes,
                    static_cast<std::uint32_t>(kSignalGapPayloadBytes));
        SignalGapPayloadFields gap_fields{};
        gap_fields.signal_gap_ordinal =
            next_signal_gap_ordinal_.fetch_add(1, std::memory_order_relaxed);
        gap_fields.data_message_ordinal = ordinal;
        gap_fields.expected_sample_idx = gap.expected_sample_idx;
        gap_fields.actual_sample_idx = gap.actual_sample_idx;
        gap_fields.missing_samples = gap.missing_samples;
        gap_fields.expected_device_tick = gap.expected_device_tick;
        gap_fields.actual_device_tick = gap.actual_device_tick;
        gap_fields.gap_idx_in_message = gap_idx;
        gap_fields.native_signal_id = gap.signal_id;
        gap_fields.reason = static_cast<std::uint32_t>(gap.reason);
        gap_fields.gap_flags = static_cast<std::uint32_t>(gap.flags);
        encode_signal_gap_payload(
            gap_fields,
            slot.subspan(cursor + slot_record::kPayload).first<kSignalGapPayloadBytes>());
        cursor += kSlotRecordHeaderBytes + kSignalGapPayloadBytes;
        ++gap_idx;
    }

    data_queue_.publish(cursor);
    recorder_accepted_.fetch_add(1, std::memory_order_relaxed);
    discontinuities_accepted_.fetch_add(1, std::memory_order_relaxed);
    signal_gaps_recorded_.fetch_add(n_gaps, std::memory_order_relaxed);
    return streaming::StreamStatus::ok;
}

void NativeRecorderCore::note_edge_rejection(ProducerIdentityKind message_kind,
                                             std::uint64_t frame_sequence) noexcept
{
    rejected_before_runtime_acceptance_.fetch_add(1, std::memory_order_relaxed);
    static_cast<void>(data_first_rejection_.latch(PositionTag::producer_identity, message_kind, 0,
                                                  frame_sequence));
    // For a critical recorder an edge that refused an item is a recording
    // loss, and loss of required data faults the recorder rather than being
    // counted and continued past (contract section 1.3).
    accepting_.store(false, std::memory_order_seq_cst);
    latch_primary_fault(FaultOrigin::recorder, RecorderFaultReason::edge_rejection, nullptr, 0);
    escalate_to_faulted();
}

bool NativeRecorderCore::request_checkpoint() noexcept
{
    if (!accepting())
    {
        return false;
    }
    checkpoint_requested_.store(true, std::memory_order_release);
    return true;
}

bool NativeRecorderCore::submit_control(const ControlSubmission& submission) noexcept
{
    if (!accepting())
    {
        rejected_after_close_control_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    if (!is_control_plane_identity(submission.kind))
    {
        // A kind outside the producer-identity registry is not a control
        // record that the path refused -- it is a call the recorder cannot
        // describe. Its position has no expressible form (the registry is
        // partitioned by plane, and the spool's position field carries a
        // registry value), so counting it as `control_rejected` would either
        // store a kind a finalizer cannot map or break the offered/accepted
        // identity. It is named by the primary fault instead, which is a
        // stronger report than a counter bucket: it is on disk, it ends the
        // session, and it says exactly what went wrong. This check touches no
        // queue, so it runs before the in-flight guard.
        accepting_.store(false, std::memory_order_seq_cst);
        latch_primary_fault(FaultOrigin::recorder, RecorderFaultReason::plan_violation, nullptr, 0);
        escalate_to_faulted();
        return false;
    }

    // Register this in-flight submission before claiming a control slot, so a
    // concurrent shutdown cannot free the span this producer holds between
    // `try_acquire()` and `publish()`. The double-check after registration
    // backs out as a late submission (not a fault) if a shutdown set
    // accepting=false in between; the guard decrements on every exit path,
    // including the saturation fault below.
    SubmissionGuard control_guard{*this, active_control_submissions_};
    if (!accepting())
    {
        rejected_after_close_control_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    control_offered_.fetch_add(1, std::memory_order_relaxed);

    const auto reject = [&](RecorderFaultReason reason)
    {
        control_rejected_.fetch_add(1, std::memory_order_relaxed);
        static_cast<void>(control_first_rejection_.latch(PositionTag::producer_identity,
                                                         submission.kind, 0, submission.identity));
        accepting_.store(false, std::memory_order_seq_cst);
        latch_primary_fault(FaultOrigin::recorder, reason, nullptr, 0);
        escalate_to_faulted();
        return false;
    };

    if (submission.body.size() > plan_.max_control_payload_bytes)
    {
        return reject(RecorderFaultReason::plan_violation);
    }

    auto claim = control_queue_.try_acquire();
    if (!claim)
    {
        return reject(RecorderFaultReason::control_queue_saturated);
    }
    auto slot = claim.slot;

#if defined(NEURALE_RECORDING_TEST_HOOKS)
    if (test_hooks_ != nullptr && test_hooks_->control_claimed != nullptr &&
        test_hooks_->release_control_claim != nullptr)
    {
        test_hooks_->control_claimed->store(true, std::memory_order_release);
        test_hooks_->control_claimed->notify_all();
        test_hooks_->release_control_claim->wait(false, std::memory_order_acquire);
    }
#endif

    // The ordinal is assigned only now, because on this plane stages 1 and 2
    // coincide: a record that was not taken into the recorder's own storage
    // was never accepted, and an item that was never accepted has no ordinal.
    const auto ordinal = next_control_ordinal_.fetch_add(1, std::memory_order_relaxed);
    const auto payload_bytes = kControlHeaderPayloadBytes + submission.body.size();

    store_u32le(slot, slot_header::kItemKind, kSlotItemControl);
    store_u32le(slot, slot_header::kRecordCount, 1);
    store_u64le(slot, slot_header::kLogicalOrdinal, ordinal);
    store_u16le(slot, kSlotHeaderBytes + slot_record::kKind,
                static_cast<std::uint16_t>(RecordKind::control));
    store_u16le(slot, kSlotHeaderBytes + slot_record::kReserved, 0);
    store_u32le(slot, kSlotHeaderBytes + slot_record::kPayloadBytes,
                static_cast<std::uint32_t>(payload_bytes));

    ControlPayloadFields fields{};
    fields.submission_ordinal = ordinal;
    fields.identity = submission.identity;
    fields.time_ns = submission.time_ns;
    fields.control_kind = static_cast<std::uint32_t>(submission.kind);
    fields.clock_domain = submission.clock_domain;
    encode_control_payload(
        fields,
        slot.subspan(kSlotHeaderBytes + slot_record::kPayload).first<kControlHeaderPayloadBytes>());
    std::copy(submission.body.begin(), submission.body.end(),
              slot.begin() + static_cast<std::ptrdiff_t>(kSlotHeaderBytes + slot_record::kPayload +
                                                         kControlHeaderPayloadBytes));

    control_queue_.publish(claim, kSlotHeaderBytes + kSlotRecordHeaderBytes + payload_bytes);
    control_accepted_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void NativeRecorderCore::publish_runtime_fault(const streaming::FaultRecord& fault) noexcept
{
    latch_primary_fault(FaultOrigin::runtime, RecorderFaultReason::none, &fault,
                        fault.detected_at_ns);
    escalate_to_faulted();
    // A runtime fault stops the recorder accepting: section 4.2 step 3 stops
    // new observer-dispatch items, and step 4 drains what was already taken.
    accepting_.store(false, std::memory_order_seq_cst);
}

void NativeRecorderCore::raise_recorder_fault(RecorderFaultReason reason,
                                              std::uint64_t detected_at_ns) noexcept
{
    accepting_.store(false, std::memory_order_seq_cst);
    latch_primary_fault(FaultOrigin::recorder, reason, nullptr, detected_at_ns);
    escalate_to_faulted();
}

void NativeRecorderCore::latch_primary_fault(FaultOrigin origin, RecorderFaultReason reason,
                                             const streaming::FaultRecord* runtime_fault,
                                             std::uint64_t detected_at_ns) noexcept
{
    // The primary fault is preserved. A fault raised while handling a fault is
    // secondary and must not displace it (contract section 4.2), so the CAS
    // decides once and every later arrival is dropped rather than recorded
    // over the top of it.
    bool expected = false;
    if (!fault_claimed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
    {
        return;
    }

    reserved_fault_ = FaultPayloadFields{};
    reserved_fault_.native_session_id = plan_.native_session_id;
    reserved_fault_.detected_at_ns = detected_at_ns;
    reserved_fault_.data_ordinal_at_fault = next_data_ordinal_.load(std::memory_order_relaxed);
    reserved_fault_.control_ordinal_at_fault =
        next_control_ordinal_.load(std::memory_order_relaxed);
    reserved_fault_.origin = origin;
    reserved_fault_.recorder_reason = reason;
    if (runtime_fault != nullptr)
    {
        // A runtime fault raised before any frame header was filled in carries
        // no frame context, and `session_id == 0` is how that absence arrives
        // here -- it is not a claim that the session ID really is zero. The
        // recorder already knows the session identity from its own plan, and
        // every frame, discontinuity, and fault in a recording must share it
        // (finalization rejects a mismatch), so an unavailable context must not
        // overwrite it. A non-zero but wrong ID is still taken verbatim: that
        // is real cross-session corruption and finalization must see it.
        if (runtime_fault->session_id != 0)
        {
            reserved_fault_.native_session_id = runtime_fault->session_id;
        }
        reserved_fault_.runtime_generation = runtime_fault->runtime_generation;
        reserved_fault_.frame_sequence = runtime_fault->frame_sequence;
        reserved_fault_.sample_idx = runtime_fault->sample_idx;
        reserved_fault_.device_tick = runtime_fault->device_tick;
        reserved_fault_.component_id = runtime_fault->component_id;
        reserved_fault_.detail = runtime_fault->detail;
        reserved_fault_.schema_id = runtime_fault->schema_id;
        reserved_fault_.clock_domain = runtime_fault->clock_domain;
        reserved_fault_.signal_id = runtime_fault->signal_id;
        reserved_fault_.fault_code = static_cast<std::uint8_t>(runtime_fault->code);
        reserved_fault_.stream_status = static_cast<std::uint8_t>(runtime_fault->status);
        reserved_fault_.fault_stage = static_cast<std::uint8_t>(runtime_fault->stage);
    }
    encode_fault_payload(reserved_fault_,
                         std::span<std::byte, kFaultPayloadBytes>{reserved_fault_payload_});
    fault_published_.store(true, std::memory_order_release);
}

void NativeRecorderCore::escalate_to_faulted() noexcept
{
    // A fault outranks a request and escalates the effective outcome without
    // rewriting what the caller asked for. Escalation is one-directional.
    effective_session_outcome_.store(EffectiveSessionOutcome::faulted, std::memory_order_release);
    // The capture has not ended yet, so the capture outcome becomes faulted
    // too -- until the session-end record freezes it (contract 3.1, 3.2).
    if (!capture_outcome_known_.load(std::memory_order_acquire))
    {
        capture_outcome_.store(CaptureOutcome::faulted, std::memory_order_release);
    }
    if (!terminal_intent_latched_.load(std::memory_order_acquire))
    {
        bool expected = false;
        if (terminal_intent_latched_.compare_exchange_strong(expected, true,
                                                             std::memory_order_acq_rel))
        {
            requested_terminal_intent_.store(RequestedTerminalIntent::fault,
                                             std::memory_order_release);
        }
    }
    // A fault ends recording: the recorder stops claiming to record and the
    // worker drains what was already accepted (contract section 4.2 step 4).
    // This is the asynchronous shutdown the recorder initiates itself, so a
    // faulted recorder does not sit in `recording` waiting for someone to call
    // `stop()`. The CAS makes it once-only: a fault before recording (the
    // readiness gate, still in `prepared`) or after a drain has already begun
    // changes nothing here, and it never clobbers a later lifecycle state.
    RecorderLifecycleState was_recording = RecorderLifecycleState::recording;
    if (state_.compare_exchange_strong(was_recording, RecorderLifecycleState::draining,
                                       std::memory_order_acq_rel))
    {
        accepting_.store(false, std::memory_order_seq_cst);
        worker_mode_.store(WorkerMode::draining, std::memory_order_release);
    }
}

// ---------------------------------------------------------------------------
// the worker
// ---------------------------------------------------------------------------

void NativeRecorderCore::worker_main() noexcept
{
    worker_running_.store(true, std::memory_order_release);
    const auto poll = std::chrono::nanoseconds(plan_.worker_idle_poll_nanos);
    while (true)
    {
        const auto mode = worker_mode_.load(std::memory_order_acquire);
        if (mode == WorkerMode::stopping)
        {
            break;
        }
        if (mode == WorkerMode::parked)
        {
            std::this_thread::sleep_for(poll);
            continue;
        }
        if (mode == WorkerMode::draining && drain_timed_out_.load(std::memory_order_acquire))
        {
            // The bound has passed. Stop taking new items and seal what is
            // already committed; whatever is still queued is a recording loss
            // and is counted as one at the seal. A drain that times out ends
            // `failed`, not `stopped`, so the state is left for `shut_down`.
            seal_session();
            worker_finished_.store(true, std::memory_order_release);
            break;
        }
        if (drain_once())
        {
            continue;
        }
        if (mode == WorkerMode::draining)
        {
            if (writer_.state() == SpoolWriterState::failed)
            {
                // A failed writer can seal nothing; finish so `shut_down` can
                // report the writer failure and retain the committed prefix.
                seal_session();
                worker_finished_.store(true, std::memory_order_release);
                break;
            }
            const auto pending_data = data_queue_.pending();
            const auto pending_control = control_queue_.pending();
            const auto active_data = active_data_submissions_.load(std::memory_order_acquire);
            const auto active_control = active_control_submissions_.load(std::memory_order_acquire);
            if (pending_data == 0 && pending_control == 0 && active_data == 0 &&
                active_control == 0)
            {
                // The queues are empty and no producer is mid-publish, so the
                // drain completed and the session-end record is committed: the
                // recorder is `stopped` (contract section 3.1). A fault that
                // started this drain autonomously ends here without waiting for
                // a `stop()` -- the CAS is once-only and never clobbers a state
                // `shut_down` may already have moved.
                seal_session();
                if (spool_ended_cleanly_.load(std::memory_order_acquire))
                {
                    RecorderLifecycleState was_draining = RecorderLifecycleState::draining;
                    state_.compare_exchange_strong(was_draining, RecorderLifecycleState::stopped,
                                                   std::memory_order_acq_rel);
                }
                worker_finished_.store(true, std::memory_order_release);
                break;
            }
            // `pending > 0` with `try_peek()` empty is an MPSC cell a producer
            // claimed but has not published yet: that is not an empty queue,
            // and sealing now would lose the in-flight item. `active > 0` is a
            // producer between acquire and publish (an SPSC slot whose claim
            // did not advance `head_`). Either way the drain is not complete;
            // wait for the producer, bounded by the configured drain timeout.
            std::this_thread::sleep_for(poll);
            continue;
        }
        std::this_thread::sleep_for(poll);
    }
    worker_running_.store(false, std::memory_order_release);
}

bool NativeRecorderCore::drain_once() noexcept
{
    if (writer_.state() == SpoolWriterState::failed ||
        writer_.state() == SpoolWriterState::cancelled ||
        writer_.state() == SpoolWriterState::sealed)
    {
        return false;
    }

    const auto now = clock_->unix_nanos();
    if (writer_.begin_transaction(now) != SpoolWriterStatus::ok)
    {
        raise_recorder_fault(RecorderFaultReason::spool_writer_failed, now);
        return false;
    }

    staged_bytes_ = kTransactionHeaderBytes + kTransactionTrailerBytes;
    staged_records_ = 0;
    std::uint64_t staged_data_items = 0;
    std::uint64_t staged_control_items = 0;
    // Tracked per plane, because a position is consistent only when the plane
    // whose counters it names actually lost something: latching a data
    // position for a transaction that carried only control records would make
    // the accounting snapshot one a reader must refuse.
    std::uint64_t first_staged_data_ordinal = 0;
    std::uint64_t first_staged_control_ordinal = 0;
    bool staged_fault = false;

    const auto slot_fits = [&](std::span<const std::byte> slot)
    {
        const auto records = load_u32le(slot, slot_header::kRecordCount);
        std::size_t bytes = 0;
        std::size_t cursor = kSlotHeaderBytes;
        for (std::uint32_t i = 0; i < records; ++i)
        {
            const auto payload_bytes = load_u32le(slot, cursor + slot_record::kPayloadBytes);
            bytes += transaction_cost(payload_bytes);
            cursor += kSlotRecordHeaderBytes + payload_bytes;
        }
        return staged_records_ + records <= plan_.max_records_per_transaction &&
               staged_bytes_ + bytes <= plan_.max_transaction_bytes;
    };

    // The two planes are drained one item at a time, alternately. Taking the
    // data plane to exhaustion first would starve the control plane whenever
    // frames alone fill a transaction -- and a starved control queue is not a
    // slow queue, it is a queue that saturates and faults a recorder that had
    // nothing wrong with it.
    bool data_exhausted = false;
    bool control_exhausted = false;
    bool writer_broke = false;
    const auto take_one = [&]<typename Queue>(Queue& queue, std::uint64_t& items,
                                              std::uint64_t& first_ordinal, bool& exhausted)
    {
        if (exhausted)
        {
            return;
        }
        const auto slot = queue.try_peek();
        if (slot.empty() || (staged_records_ != 0 && !slot_fits(slot)))
        {
            exhausted = true;
            return;
        }
        if (items == 0)
        {
            first_ordinal = load_u64le(slot, slot_header::kLogicalOrdinal);
        }
        if (!stage_slot(slot, now))
        {
            writer_broke = true;
            exhausted = true;
            return;
        }
        queue.release();
        ++items;
    };

    while (!data_exhausted || !control_exhausted)
    {
        take_one(data_queue_, staged_data_items, first_staged_data_ordinal, data_exhausted);
        take_one(control_queue_, staged_control_items, first_staged_control_ordinal,
                 control_exhausted);
        if (writer_broke)
        {
            writer_.discard_transaction();
            raise_recorder_fault(RecorderFaultReason::spool_writer_failed, now);
            return false;
        }
    }

    if (fault_published_.load(std::memory_order_acquire) &&
        !fault_committed_.load(std::memory_order_acquire) &&
        staged_records_ + 1 <= plan_.max_records_per_transaction &&
        staged_bytes_ + transaction_cost(kFaultPayloadBytes) <= plan_.max_transaction_bytes)
    {
        // The primary fault row travels in the ordinary transaction stream but
        // never through the control queue, so it is still writable when that
        // queue is exactly the thing that is full.
        if (writer_.append_record(RecordKind::fault, reserved_fault_payload_, 0, now) ==
            SpoolWriterStatus::ok)
        {
            staged_bytes_ += transaction_cost(kFaultPayloadBytes);
            ++staged_records_;
            staged_fault = true;
        }
    }

    if (staged_records_ == 0)
    {
        writer_.discard_transaction();
        return false;
    }

    if (writer_.commit_transaction() != SpoolWriterStatus::ok)
    {
        // Everything staged is lost: it was recorder accepted and now has no
        // path to stage 3.
        lost_between_recorder_and_spool_.fetch_add(staged_data_items, std::memory_order_relaxed);
        lost_between_control_acceptance_and_spool_.fetch_add(staged_control_items,
                                                             std::memory_order_relaxed);
        if (staged_data_items != 0)
        {
            static_cast<void>(data_first_loss_.latch(
                PositionTag::ordinal, ProducerIdentityKind::none, first_staged_data_ordinal, 0));
        }
        if (staged_control_items != 0)
        {
            static_cast<void>(control_first_loss_.latch(
                PositionTag::ordinal, ProducerIdentityKind::none, first_staged_control_ordinal, 0));
        }
        raise_recorder_fault(RecorderFaultReason::spool_writer_failed, now);
        return false;
    }

    spool_committed_.fetch_add(staged_data_items, std::memory_order_relaxed);
    control_spool_committed_.fetch_add(staged_control_items, std::memory_order_relaxed);
    publish_spool_state();
    if (staged_fault)
    {
        fault_committed_.store(true, std::memory_order_release);
    }
    ++committed_transactions_;

    // `exchange` rather than a load-then-store: a request that arrives while
    // this checkpoint is being written must survive into the next transaction
    // rather than be cleared by the write it did not cause.
    const bool requested = checkpoint_requested_.exchange(false, std::memory_order_acq_rel);
    if (requested || (plan_.checkpoint_interval_transactions != 0 &&
                      committed_transactions_ % plan_.checkpoint_interval_transactions == 0))
    {
        const auto checkpoint_now = clock_->unix_nanos();
        if (writer_.checkpoint(checkpoint_now, checkpoint_now) != SpoolWriterStatus::ok)
        {
            raise_recorder_fault(RecorderFaultReason::spool_writer_failed, checkpoint_now);
            return true;
        }
        publish_spool_state();
    }
    return true;
}

bool NativeRecorderCore::stage_slot(std::span<const std::byte> slot,
                                    std::uint64_t record_unix_nanos) noexcept
{
    const auto records = load_u32le(slot, slot_header::kRecordCount);
    const auto ordinal = load_u64le(slot, slot_header::kLogicalOrdinal);
    std::size_t cursor = kSlotHeaderBytes;
    for (std::uint32_t i = 0; i < records; ++i)
    {
        const auto kind = load_u16le(slot, cursor + slot_record::kKind);
        const auto payload_bytes = load_u32le(slot, cursor + slot_record::kPayloadBytes);
        const auto payload = slot.subspan(cursor + slot_record::kPayload, payload_bytes);
        if (writer_.append_record(static_cast<RecordKind>(kind), payload, ordinal,
                                  record_unix_nanos) != SpoolWriterStatus::ok)
        {
            return false;
        }
        staged_bytes_ += transaction_cost(payload_bytes);
        ++staged_records_;
        cursor += kSlotRecordHeaderBytes + payload_bytes;
    }
    return true;
}

SpoolAccounting NativeRecorderCore::build_accounting() const noexcept
{
    SpoolAccounting accounting{};
    accounting.runtime_accepted = runtime_accepted_.load(std::memory_order_relaxed);
    accounting.recorder_accepted = recorder_accepted_.load(std::memory_order_relaxed);
    accounting.spool_committed = writer_.data_items();
    accounting.rejected_before_runtime_acceptance =
        rejected_before_runtime_acceptance_.load(std::memory_order_relaxed);
    accounting.failed_between_runtime_and_recorder =
        failed_between_runtime_and_recorder_.load(std::memory_order_relaxed);
    // Derived, never counted independently: every accepted item either reached
    // a committed transaction or did not, and the writer's own committed count
    // is the only number a reader can check this against.
    accounting.lost_between_recorder_and_spool =
        accounting.recorder_accepted >= accounting.spool_committed
            ? accounting.recorder_accepted - accounting.spool_committed
            : 0;

    accounting.control_offered = control_offered_.load(std::memory_order_relaxed);
    accounting.control_accepted = control_accepted_.load(std::memory_order_relaxed);
    accounting.control_spool_committed = writer_.control_items();
    accounting.control_rejected = control_rejected_.load(std::memory_order_relaxed);
    accounting.lost_between_control_acceptance_and_spool =
        accounting.control_accepted >= accounting.control_spool_committed
            ? accounting.control_accepted - accounting.control_spool_committed
            : 0;

    accounting.rejected_after_close_data =
        rejected_after_close_data_.load(std::memory_order_relaxed);
    accounting.rejected_after_close_control =
        rejected_after_close_control_.load(std::memory_order_relaxed);

    accounting.control_offered_present = true;
    accounting.producer_acceptance_known = true;

    const auto to_spool_position = [](const RecorderFirstPosition& pos)
    {
        return SpoolAccountingPosition{.tag = pos.tag,
                                       .identity_kind = pos.identity_kind,
                                       .ordinal = pos.ordinal,
                                       .identity_value = pos.identity_value};
    };
    accounting.data_first_loss = to_spool_position(data_first_loss_.read());
    accounting.data_first_rejection = to_spool_position(data_first_rejection_.read());
    accounting.control_first_loss = to_spool_position(control_first_loss_.read());
    accounting.control_first_rejection = to_spool_position(control_first_rejection_.read());
    return accounting;
}

void NativeRecorderCore::account_unwritten_items() noexcept
{
    // Anything still queued when the drain ends never reached stage 3. The
    // first of them is the first loss, unless something earlier already
    // claimed that position.
    const auto pending_data = data_queue_.pending();
    if (pending_data != 0)
    {
        const auto slot = data_queue_.try_peek();
        if (!slot.empty())
        {
            static_cast<void>(
                data_first_loss_.latch(PositionTag::ordinal, ProducerIdentityKind::none,
                                       load_u64le(slot, slot_header::kLogicalOrdinal), 0));
        }
    }
    const auto pending_control = control_queue_.pending();
    if (pending_control != 0)
    {
        const auto slot = control_queue_.try_peek();
        if (!slot.empty())
        {
            static_cast<void>(
                control_first_loss_.latch(PositionTag::ordinal, ProducerIdentityKind::none,
                                          load_u64le(slot, slot_header::kLogicalOrdinal), 0));
        }
    }
}

void NativeRecorderCore::publish_spool_state() noexcept
{
    spool_committed_extent_.store(writer_.committed_extent(), std::memory_order_release);
    spool_durable_extent_.store(writer_.durable_extent(), std::memory_order_release);
    spool_committed_transactions_.store(writer_.committed_transactions(),
                                        std::memory_order_release);
}

void NativeRecorderCore::seal_session() noexcept
{
    account_unwritten_items();
    if (writer_.state() != SpoolWriterState::prepared)
    {
        // No session-end record will exist, so nothing freezes the capture
        // outcome and it stays unknown. Forging one is forbidden (4.5).
        return;
    }

    const auto now = clock_->unix_nanos();
    SpoolSessionEnd session_end{};
    session_end.requested_terminal_intent =
        requested_terminal_intent_.load(std::memory_order_acquire);
    session_end.capture_outcome = capture_outcome_.load(std::memory_order_acquire);
    session_end.primary_fault_committed = fault_committed_.load(std::memory_order_acquire);
    session_end.terminal_reason = std::string_view{terminal_reason_.data(), terminal_reason_size_};
    session_end.end_unix_nanos = now;

    if (writer_.seal(build_accounting(), session_end, now) != SpoolWriterStatus::ok)
    {
        raise_recorder_fault(RecorderFaultReason::spool_writer_failed, now);
        return;
    }
    publish_spool_state();
    // The session-end record is committed, which freezes the capture outcome.
    capture_outcome_known_.store(true, std::memory_order_release);
    spool_ended_cleanly_.store(true, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// status
// ---------------------------------------------------------------------------

RecorderStatus NativeRecorderCore::status() const
{
    RecorderStatus out{};
    out.state = state_.load(std::memory_order_acquire);

    const auto committed_transactions =
        spool_committed_transactions_.load(std::memory_order_acquire);
    out.spool_holds_committed_record = committed_transactions > 0;
    // `session_created` is its own fact -- "was any session artifact produced
    // at all" -- set when the readiness gate commits the superblock and cleared
    // when that artifact is discarded. It is not the same question as "does the
    // spool hold a committed record", so it is not derived from this counter.
    out.session_created = session_created_.load(std::memory_order_acquire);
    out.spool_ended_cleanly = spool_ended_cleanly_.load(std::memory_order_acquire);
    // No finalizer has run, so no NRF session exists to be sealed and there is
    // nothing to judge. Completeness is a property of a *sealed* session
    // (stage 5), which is why the verdict and `complete` are absent rather than
    // false: `false` would say the session was judged and found wanting, and
    // the contract is explicit that "there is no sealed session to judge" is a
    // different answer (section 3.2, the fourth case). `accounting_verified`
    // is false for the same reason -- nothing has checked the accounting -- and
    // the contract pairs exactly this combination with an absent verdict.
    out.sealed = false;
    out.completeness_verdict = std::nullopt;
    out.complete = std::nullopt;
    out.accounting_verified = false;
    // The termination kind is what the artifact *says*, and no artifact says
    // anything yet. Left absent rather than mirrored from
    // `effective_session_outcome`, which is what this recorder concluded.
    out.termination_kind = std::nullopt;
    out.finalization_required = out.spool_holds_committed_record && !out.sealed;
    out.recovery_required = out.spool_holds_committed_record && !out.spool_ended_cleanly;
    out.recoverable = out.recovery_required ? RecoverabilityAnswer::recoverable
                                            : RecoverabilityAnswer::not_applicable;

    out.requested_terminal_intent = requested_terminal_intent_.load(std::memory_order_acquire);
    out.terminal_intent_latched = terminal_intent_latched_.load(std::memory_order_acquire);
    // Absent until the session-end record freezes it. Reading the outcome
    // without the flag would report `normal` for a process that died before
    // anything decided the question.
    out.capture_outcome =
        capture_outcome_known_.load(std::memory_order_acquire)
            ? std::optional<CaptureOutcome>{capture_outcome_.load(std::memory_order_acquire)}
            : std::nullopt;
    out.effective_session_outcome = effective_session_outcome_.load(std::memory_order_acquire);
    out.finalization_status = FinalizationStatus::not_started;
    // Empty, not omitted: a caller asking "what did the finalizer try?" gets
    // the honest "nothing", in the same shape a finalizer will fill.
    out.finalization_attempts.clear();

    out.runtime_accepted = runtime_accepted_.load(std::memory_order_relaxed);
    out.recorder_accepted = recorder_accepted_.load(std::memory_order_relaxed);
    out.spool_committed = spool_committed_.load(std::memory_order_relaxed);
    out.nrf_committed = 0;
    out.rejected_before_runtime_acceptance =
        rejected_before_runtime_acceptance_.load(std::memory_order_relaxed);
    out.failed_between_runtime_and_recorder =
        failed_between_runtime_and_recorder_.load(std::memory_order_relaxed);
    out.lost_between_recorder_and_spool =
        lost_between_recorder_and_spool_.load(std::memory_order_relaxed);
    out.lost_during_finalization = 0;

    out.control_offered = control_offered_.load(std::memory_order_relaxed);
    out.control_accepted = control_accepted_.load(std::memory_order_relaxed);
    out.control_spool_committed = control_spool_committed_.load(std::memory_order_relaxed);
    out.control_nrf_committed = 0;
    out.control_rejected = control_rejected_.load(std::memory_order_relaxed);
    out.lost_between_control_acceptance_and_spool =
        lost_between_control_acceptance_and_spool_.load(std::memory_order_relaxed);
    out.control_lost_during_finalization = 0;

    out.rejected_after_close_data = rejected_after_close_data_.load(std::memory_order_relaxed);
    out.rejected_after_close_control =
        rejected_after_close_control_.load(std::memory_order_relaxed);

    out.data_first_loss = data_first_loss_.read();
    out.data_first_rejection = data_first_rejection_.read();
    out.control_first_loss = control_first_loss_.read();
    out.control_first_rejection = control_first_rejection_.read();

    if (fault_published_.load(std::memory_order_acquire))
    {
        out.primary_fault.present = true;
        out.primary_fault.origin = reserved_fault_.origin;
        out.primary_fault.reason = reserved_fault_.recorder_reason;
        out.primary_fault.stream_status = reserved_fault_.stream_status;
        out.primary_fault.fault_code = reserved_fault_.fault_code;
        out.primary_fault.fault_stage = reserved_fault_.fault_stage;
        out.primary_fault.detected_at_ns = reserved_fault_.detected_at_ns;
        out.primary_fault.frame_sequence = reserved_fault_.frame_sequence;
        out.primary_fault.data_ordinal_at_fault = reserved_fault_.data_ordinal_at_fault;
        out.primary_fault.control_ordinal_at_fault = reserved_fault_.control_ordinal_at_fault;
        out.primary_fault.committed = fault_committed_.load(std::memory_order_acquire);
    }

    out.frames_accepted = frames_accepted_.load(std::memory_order_relaxed);
    out.discontinuities_accepted = discontinuities_accepted_.load(std::memory_order_relaxed);
    out.signal_blocks_recorded = signal_blocks_recorded_.load(std::memory_order_relaxed);
    out.signal_gaps_recorded = signal_gaps_recorded_.load(std::memory_order_relaxed);
    out.payload_bytes_copied = payload_bytes_copied_.load(std::memory_order_relaxed);

    out.data_queue_pending = data_queue_.pending();
    out.data_queue_capacity = data_queue_capacity_.load(std::memory_order_acquire);
    out.data_queue_high_water_mark = data_queue_.high_water_mark();
    out.control_queue_pending = control_queue_.pending();
    out.control_queue_capacity = control_queue_capacity_.load(std::memory_order_acquire);
    out.control_queue_high_water_mark = control_queue_.high_water_mark();
    out.worker_running = worker_running_.load(std::memory_order_acquire);

    out.spool_committed_extent = spool_committed_extent_.load(std::memory_order_acquire);
    out.spool_durable_extent = spool_durable_extent_.load(std::memory_order_acquire);
    out.spool_committed_transactions = committed_transactions;
    out.spool_durability_policy = spool_durability_policy_;

    out.queue_storage_released = queue_storage_released_.load(std::memory_order_acquire);
    out.queue_storage_release_deferred =
        queue_storage_release_deferred_.load(std::memory_order_acquire);
    out.spool_backend_cancellable = spool_backend_cancellable_.load(std::memory_order_acquire);
    return out;
}

} // namespace neurale::recording
