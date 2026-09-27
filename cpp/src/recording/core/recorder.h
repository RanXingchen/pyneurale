/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The native critical recorder core: observer callback, bounded queues,
/// spool worker, lifecycle, and the status surface.
///
/// What runs where, because it is the whole point of this class:
///
/// - **The critical callback** (`observe_accepted`,
///   `handle_accepted_discontinuity`, `submit_control`,
///   `publish_primary_fault`) performs bounded validation, a copy into storage
///   the recorder already owns, and a lock-free enqueue -- nothing else. No
///   filesystem call, no blocking wait, no Python, no GIL, no logging, no
///   allocation, and no clock read. The frame lease belongs to the runtime and
///   is released the instant the callback returns, which is why the item is
///   copied rather than referenced.
/// - **The worker thread** owns the spool: it drains the queues into
///   transactions, syncs per policy, checkpoints, and seals. It is allowed to
///   block, to call the filesystem, and to read a clock; it is not allowed to
///   touch anything the callback owns except through the queues.
///
/// The worker **polls** with a bounded idle sleep instead of waiting on a
/// condition variable, so the callback performs no wake syscall at all. That
/// trades a small, configured amount of idle CPU for a callback whose cost is
/// a memcpy and two atomic stores. `worker_idle_poll_nanos` is the knob.
///
/// Saturation is **lossless-until-fault**, never a drop policy. There is no
/// overflow policy to configure here: a full queue on a critical recorder
/// latches the primary fault and stops accepting (contract sections 1.3, 2,
/// and 4.3). Counting the loss and continuing is the noncritical Python path's
/// behaviour and is deliberately absent.
///
/// This core attaches to streaming's format-independent
/// `NativeCriticalObserver` boundary. The runtime assigns the data-message
/// ordinal and acceptance timestamp before this core copies an item. It still
/// does not finalize: `finalize()` and
/// `abandon_finalization()` return `not_implemented`, so a status after
/// `close()` honestly reports `finalization_required = true`,
/// `sealed = false`, and `nrf_committed = 0` rather than a fabricated success.
/// There is no Python surface here yet, and it will be provisional when it
/// arrives.

#include "bounded_queue.h"
#include "record_payloads.h"
#include "recorder_status.h"
#include "recording_plan.h"
#include "spool_writer.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include <neurale/streaming/critical_observer.h>
#include <neurale/streaming/discontinuity.h>
#include <neurale/streaming/fault.h>
#include <neurale/streaming/frame.h>

namespace neurale::recording
{

/// Wall-clock nanoseconds, injectable so a test can produce a byte-stable
/// spool and so nothing on the critical path ever needs one. Only the worker
/// thread calls it.
class UnixClock
{
  public:
    virtual ~UnixClock() = default;
    [[nodiscard]] virtual std::uint64_t unix_nanos() noexcept = 0;
};

class SystemUnixClock final : public UnixClock
{
  public:
    [[nodiscard]] std::uint64_t unix_nanos() noexcept override;
};

/// A position that exactly one writer may claim, published without a lock.
///
/// Two threads can discover a loss at once -- the callback finding a full
/// queue while the worker finds a failing writer -- and only the first one is
/// the position the contract asks for. The CAS decides the winner; the
/// release-store makes the winner's fields visible to a reader that acquires.
class LatchedPosition
{
  public:
    /// Claim and publish. Returns false when someone already had it.
    bool latch(PositionTag tag, ProducerIdentityKind identity_kind, std::uint64_t ordinal,
               std::uint64_t identity_value) noexcept
    {
        bool expected = false;
        if (!claimed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        {
            return false;
        }
        tag_ = tag;
        identity_kind_ = identity_kind;
        ordinal_ = ordinal;
        identity_value_ = identity_value;
        published_.store(true, std::memory_order_release);
        return true;
    }

    [[nodiscard]] RecorderFirstPosition read() const noexcept
    {
        if (!published_.load(std::memory_order_acquire))
        {
            return {};
        }
        return RecorderFirstPosition{.tag = tag_,
                                     .identity_kind = identity_kind_,
                                     .ordinal = ordinal_,
                                     .identity_value = identity_value_};
    }

    void reset() noexcept
    {
        published_.store(false, std::memory_order_relaxed);
        claimed_.store(false, std::memory_order_relaxed);
        tag_ = PositionTag::absent;
        identity_kind_ = ProducerIdentityKind::none;
        ordinal_ = 0;
        identity_value_ = 0;
    }

  private:
    std::atomic<bool> claimed_{false};
    std::atomic<bool> published_{false};
    PositionTag tag_{PositionTag::absent};
    ProducerIdentityKind identity_kind_{ProducerIdentityKind::none};
    std::uint64_t ordinal_{};
    std::uint64_t identity_value_{};
};

/// What the caller hands over on the control plane. Stages 1 and 2 coincide
/// here by construction: the storage the record is accepted into is already
/// the recorder's, and there is no fictitious second stage (contract 1.1).
struct ControlSubmission
{
    /// The producer-identity registry value for this record's kind. Must be a
    /// control-plane kind (3..11); a data-plane kind is refused.
    ProducerIdentityKind kind{ProducerIdentityKind::none};
    /// What fixes this record's order within its kind. Recoverable from the
    /// record, never inferred from position in a file.
    std::uint64_t identity{};
    /// The clock domain `time_ns` belongs to.
    std::uint32_t clock_domain{};
    std::uint64_t time_ns{};
    /// The record body. Opaque to the recorder and to the container.
    std::span<const std::byte> body{};
};

#if defined(NEURALE_RECORDING_TEST_HOOKS)
/// Test-only synchronization points for deterministic shutdown races.
struct RecorderCoreTestHooks
{
    std::atomic<bool>* control_claimed{};
    std::atomic<bool>* release_control_claim{};
};
#endif

class NativeRecorderCore final : public streaming::NativeCriticalObserver
{
  public:
    NativeRecorderCore() = default;
    ~NativeRecorderCore();

    NativeRecorderCore(const NativeRecorderCore&) = delete;
    NativeRecorderCore& operator=(const NativeRecorderCore&) = delete;
    NativeRecorderCore(NativeRecorderCore&&) = delete;
    NativeRecorderCore& operator=(NativeRecorderCore&&) = delete;

#if defined(NEURALE_RECORDING_TEST_HOOKS)
    void set_test_hooks(RecorderCoreTestHooks* hooks) noexcept
    {
        test_hooks_ = hooks;
    }
#endif

    // --- lifecycle (contract section 3.1) ----------------------------------

    /// Validate the plan, allocate every bounded resource, and build the spool
    /// superblock in memory. This is the only allocation point on any recording
    /// path; it writes no durable byte, so a failure leaves the recorder in
    /// `created` with nothing retained and the object may be prepared again.
    /// The superblock is committed and the worker started at the readiness gate.
    ///
    /// *spool* must be an empty file and must outlive the recorder, and its
    /// backend may perform blocking I/O on the writer thread. Stop/close report
    /// timeout while retaining resources if it cannot cancel. The caller must
    /// retain spool and clock until close succeeds; destruction waits for I/O.
    [[nodiscard]] RecorderStatusCode prepare(const NativeRecordingPlan& plan, SpoolFile& spool,
                                             UnixClock& clock);

    /// The readiness gate. *critical_edges_attached_and_lossless* is what the
    /// runtime knows and the recorder does not; the runtime supplies it from
    /// the edge configuration. A passing gate commits the spool superblock and
    /// starts the worker parked, moving to `ready` (contract section 3.1:
    /// "readiness gate passes -> commit superblock -> ready"). A gate failure
    /// releases **every** prepared resource and moves to `failed`, because a
    /// prepared-but-never-started recorder holding queues and a spool handle is
    /// exactly what section 3.1 forbids -- and a gate that did not pass commits
    /// nothing, so it discards any artifact it may have written and leaves none.
    ///
    /// Cancellation capability is observable, not a readiness requirement.
    /// Unsupported durability policies are still refused, never weakened.
    [[nodiscard]] RecorderStatusCode pass_readiness_gate(bool critical_edges_attached_and_lossless);

    /// Begin accepting on both planes. Legal only in `ready`.
    [[nodiscard]] RecorderStatusCode start();

    /// Graceful stop: latch intent `normal`, stop accepting, drain what was
    /// accepted under the configured bound, seal the spool.
    [[nodiscard]] RecorderStatusCode stop(std::string_view reason);

    /// Explicit abort: latch intent `aborted`. Not a fault -- no fault record
    /// is invented -- and still a full drain, because an abort stops recording
    /// new data rather than destroying data already handed over (section 4.4).
    [[nodiscard]] RecorderStatusCode abort(std::string_view reason);

    /// Legal and idempotent in every state (section 3.1). From `recording` it
    /// performs a graceful stop first. From `stopped` it releases resources
    /// and leaves the spool finalizable: the status says so through
    /// `finalization_required`, because finalization is not implemented yet.
    [[nodiscard]] RecorderStatusCode close();

    /// Not yet implemented. Returns `not_implemented` and changes nothing.
    [[nodiscard]] RecorderStatusCode finalize();
    /// Not yet implemented. Returns `not_implemented` and changes nothing.
    [[nodiscard]] RecorderStatusCode abandon_finalization();

    // --- generic streaming critical-observer lifecycle --------------------

    [[nodiscard]] streaming::StreamStatus ready_for_runtime() noexcept override;
    [[nodiscard]] streaming::StreamStatus start_observing() noexcept override;
    [[nodiscard]] streaming::StreamStatus health() const noexcept override;
    [[nodiscard]] streaming::StreamStatus
    observe_accepted(streaming::FrameView frame,
                     streaming::RuntimeAcceptance acceptance) noexcept override;
    [[nodiscard]] streaming::StreamStatus
    handle_accepted_discontinuity(const streaming::Discontinuity& discontinuity,
                                  streaming::RuntimeAcceptance acceptance) noexcept override;
    void note_rejected_before_acceptance(streaming::RejectedMessage message) noexcept override;
    void publish_primary_fault(const streaming::FaultRecord& fault) noexcept override;
    [[nodiscard]] streaming::StreamStatus
    drain(streaming::RuntimeTerminalNotice terminal) noexcept override;
    void cancel() noexcept override;
    [[nodiscard]] streaming::StreamStatus reset() noexcept override;

    // --- the critical callback ---------------------------------------------
    //
    // Every function in this block is bounded, allocation-free, lock-free, and
    // makes no syscall. They are the ones a critical observer edge calls.

    /// Standalone recorder-core harness entry point. Runtime integration uses
    /// `observe_accepted()` so identity is assigned only by the runtime.
    /// Returns `ok` on acceptance,
    /// `queue_overflow` when the bounded queue is full (which faults the
    /// recorder), `invalid_frame` on a plan violation, and `stopped` when the
    /// recorder is no longer accepting.
    [[nodiscard]] streaming::StreamStatus observe_frame(const streaming::FrameView& frame) noexcept;

    /// Standalone harness counterpart for one discontinuity. Runtime
    /// integration uses `handle_accepted_discontinuity()`.
    /// carries. One discontinuity is **one** item, however many gaps it has.
    [[nodiscard]] streaming::StreamStatus
    observe_discontinuity(const streaming::Discontinuity& discontinuity) noexcept;

    /// Report an item an observer edge refused *before* runtime acceptance.
    /// It never had an ordinal, so its position is the pair
    /// `(message_kind, frame_sequence)`. On a critical recorder this is a
    /// fault, not a statistic.
    void note_edge_rejection(ProducerIdentityKind message_kind,
                             std::uint64_t frame_sequence) noexcept;

    /// Offer one control record. `true` means **exactly** control accepted and
    /// nothing further -- not stage 3, not stage 4, not stage 5. `false` means
    /// it was not accepted, and it has already been reported with a named
    /// reason: as `control_rejected` with a first-rejection position for a
    /// well-formed record the path refused, and as the primary fault for a
    /// `kind` outside the producer-identity registry, whose position has no
    /// expressible form (see the implementation comment).
    [[nodiscard]] bool submit_control(const ControlSubmission& submission) noexcept;

    /// Ask the worker to write a checkpoint after the next transaction it
    /// commits, on top of whatever `checkpoint_interval_transactions` already
    /// schedules. `true` means the request was taken, `false` that the recorder
    /// is not accepting -- a checkpoint request that returned quietly having
    /// written nothing would read like a checkpoint that succeeded.
    ///
    /// It sets a flag and returns; it does not wait for the checkpoint. Waiting
    /// would block the caller on the worker's next transaction, and there is no
    /// bound on when a quiet session commits one.
    [[nodiscard]] bool request_checkpoint() noexcept;

    /// Deliver a primary runtime fault through the reserved, bounded path. It
    /// does not use the ordinary control queue, so it is still writable when
    /// that queue is exactly the thing that is full (contract section 1.1).
    /// A fault raised while a fault is latched is secondary and is dropped
    /// rather than displacing the primary one (section 4.2).
    void publish_runtime_fault(const streaming::FaultRecord& fault) noexcept;

    /// Raise a recorder fault directly. Used by the runtime for an unusable
    /// recorder at arm time, and internally for queue saturation and writer
    /// failure.
    void raise_recorder_fault(RecorderFaultReason reason, std::uint64_t detected_at_ns) noexcept;

    // --- observation --------------------------------------------------------

    [[nodiscard]] RecorderStatus status() const;
    [[nodiscard]] RecorderLifecycleState state() const noexcept
    {
        return state_.load(std::memory_order_acquire);
    }

  private:
    enum class WorkerMode : std::uint8_t
    {
        parked = 0,
        running = 1,
        draining = 2,
        stopping = 3,
    };

    // --- callback helpers ---------------------------------------------------

    /// Sequentially consistent on purpose, and paired with the equally
    /// sequentially consistent `accepting_.store(false)` in `release_resources`.
    /// A producer registers itself and *then* re-reads this flag; a shutdown
    /// clears the flag and *then* reads the registration. With anything weaker
    /// than `seq_cst` both are allowed to miss each other -- the store-buffer
    /// shape -- and a shutdown that misses a producer is a shutdown that frees
    /// storage the producer is writing into. With `seq_cst` exactly one of the
    /// two observes the other, which is the whole safety argument.
    [[nodiscard]] bool accepting() const noexcept
    {
        return accepting_.load(std::memory_order_seq_cst);
    }

    /// One in-flight producer, registered for as long as it may hold a span
    /// into queue storage. The destructor is where deferred reclamation
    /// happens: a producer that is the last one out of a recorder whose storage
    /// was already retired frees that storage itself, because the shutdown
    /// declined to free it while this producer was still inside.
    class SubmissionGuard
    {
      public:
        SubmissionGuard(NativeRecorderCore& core, std::atomic<std::uint64_t>& counter) noexcept
            : core_(core), counter_(counter)
        {
            counter_.fetch_add(1, std::memory_order_seq_cst);
        }
        ~SubmissionGuard() noexcept
        {
            if (counter_.fetch_sub(1, std::memory_order_seq_cst) == 1)
            {
                core_.release_queue_storage_if_quiescent();
            }
        }
        SubmissionGuard(const SubmissionGuard&) = delete;
        SubmissionGuard& operator=(const SubmissionGuard&) = delete;
        SubmissionGuard(SubmissionGuard&&) = delete;
        SubmissionGuard& operator=(SubmissionGuard&&) = delete;

      private:
        NativeRecorderCore& core_;
        std::atomic<std::uint64_t>& counter_;
    };
    [[nodiscard]] streaming::StreamStatus
    reject_data_after_close(streaming::StreamStatus status) noexcept;
    [[nodiscard]] streaming::StreamStatus
    observe_runtime_frame(const streaming::FrameView& frame,
                          streaming::RuntimeAcceptance acceptance) noexcept;
    [[nodiscard]] streaming::StreamStatus
    observe_runtime_discontinuity(const streaming::Discontinuity& discontinuity,
                                  streaming::RuntimeAcceptance acceptance) noexcept;
    [[nodiscard]] streaming::StreamStatus fail_data_item(std::uint64_t ordinal,
                                                         RecorderFaultReason reason,
                                                         streaming::StreamStatus status) noexcept;

    // --- worker -------------------------------------------------------------

    void worker_main() noexcept;
    /// Move one transaction's worth of queued items into the spool. Returns
    /// true when anything was committed.
    [[nodiscard]] bool drain_once() noexcept;
    [[nodiscard]] bool stage_slot(std::span<const std::byte> slot,
                                  std::uint64_t record_unix_nanos) noexcept;
    void seal_session() noexcept;
    void account_unwritten_items() noexcept;
    /// Copy the writer's extents into the atomics `status()` reads. The writer
    /// belongs to the worker thread and keeps its counters in plain members,
    /// so a status call reaching into it from another thread would be a race
    /// however harmless the value looked.
    void publish_spool_state() noexcept;
    [[nodiscard]] SpoolAccounting build_accounting() const noexcept;

    // --- lifecycle helpers --------------------------------------------------

    [[nodiscard]] RecorderStatusCode shut_down(RequestedTerminalIntent intent,
                                               std::string_view reason);
    bool release_resources() noexcept;
    /// Retire the queue storage, and free it if -- and only if -- no producer
    /// is inside the recorder. This is the one function that decides between
    /// the two things a shutdown must never do: free memory a producer still
    /// holds a span into, or wait for that producer without a bound. It does
    /// neither. It marks the storage retired and hands the actual free to
    /// whoever leaves last, which on a compliant callback is a few hundred
    /// nanoseconds later and on a callback that never returns is never -- and
    /// memory a thread is still writing into is memory in use, not a leak.
    ///
    /// When the last one out is a producer, the free happens on the producer's
    /// thread. That is a deallocation on a thread the critical contract keeps
    /// allocation-free, and it is the right trade: it can only happen once, only
    /// after the recorder has stopped, and only in place of a use-after-free.
    /// The steady state it is measured in never reaches here -- the allocation
    /// gate runs entirely inside a recording session.
    void release_queue_storage_if_quiescent() noexcept;
    /// Discard a spool that holds no committed record (contract section 3.1):
    /// a readiness-gate failure and a `close()` from `prepared`/`ready` truncate
    /// the file to nothing so a scanner never sees a `finalizable` artifact the
    /// recorder says does not exist. A spool that holds a committed record is
    /// retained under section 7 for the offline finalizer, so this is a no-op
    /// on a `recording -> stop -> stopped -> close` run.
    void discard_empty_spool() noexcept;
    void latch_primary_fault(FaultOrigin origin, RecorderFaultReason reason,
                             const streaming::FaultRecord* runtime_fault,
                             std::uint64_t detected_at_ns) noexcept;
    void escalate_to_faulted() noexcept;

    // --- fixed configuration ------------------------------------------------

    NativeRecordingPlan plan_{};
    std::vector<PlannedSignalRecording> plan_signals_{};
    std::vector<char> session_id_storage_{};
    std::vector<std::byte> plan_document_storage_{};
    SpoolFile* spool_{};
    UnixClock* clock_{};

    SpoolWriter writer_{};
    BoundedSlotQueue data_queue_{};
    BoundedMpscSlotQueue control_queue_{};

    std::thread worker_{};
    std::atomic<WorkerMode> worker_mode_{WorkerMode::parked};
    std::atomic<bool> worker_running_{false};
    std::atomic<bool> worker_finished_{false};

    std::atomic<RecorderLifecycleState> state_{RecorderLifecycleState::created};
    std::atomic<bool> accepting_{false};
    /// Everything `prepare()` allocated is released by exactly one transition
    /// into `closed` or `failed`; this is what makes that "exactly one".
    std::atomic<bool> resources_released_{false};

    /// In-flight data/control submissions. A producer holds a span into queue
    /// storage between `try_acquire()` and `publish()`; `release_storage()`
    /// must not free that storage first. The callback registers before it
    /// touches the queue and re-checks `accepting_` after registering (a shutdown
    /// that started between the two checks would otherwise leave it untracked).
    /// The worker also waits for these to reach zero before it seals, so a
    /// claimed-but-unpublished item is not mistaken for an empty queue.
    std::atomic<std::uint64_t> active_data_submissions_{0};
    std::atomic<std::uint64_t> active_control_submissions_{0};

    /// The reclamation handshake for the queues' storage. `retired_` says the
    /// shutdown has given the storage up; `released_` is the once-only claim on
    /// actually freeing it, taken by the shutdown when it finds the recorder
    /// quiescent and by the last producer out when it does not. `deferred_`
    /// records that the shutdown had to hand the free over, which is a fact the
    /// status surface reports rather than hides -- an in-flight producer at
    /// shutdown means a callback that outlived the recorder's stop, and an
    /// operator should be able to see that it happened.
    std::atomic<bool> queue_storage_retired_{false};
    std::atomic<bool> queue_storage_released_{false};
    std::atomic<bool> queue_storage_release_deferred_{false};
    /// What the readiness gate checked, kept so `status()` never dereferences
    /// the caller's file to answer a question about it.
    std::atomic<bool> spool_backend_cancellable_{false};

#if defined(NEURALE_RECORDING_TEST_HOOKS)
    RecorderCoreTestHooks* test_hooks_{};
#endif

    /// Serializes stop/abort/close. Never taken by the critical callback.
    mutable std::mutex lifecycle_mutex_{};

    // --- counters -----------------------------------------------------------
    //
    // Written by the callback thread and the worker thread, read by anyone.
    // Relaxed: they are statistics whose final, consistent read happens after
    // the worker has joined, and no decision is made from a mid-flight value.

    std::atomic<std::uint64_t> runtime_accepted_{0};
    std::atomic<std::uint64_t> recorder_accepted_{0};
    std::atomic<std::uint64_t> spool_committed_{0};
    std::atomic<std::uint64_t> rejected_before_runtime_acceptance_{0};
    std::atomic<std::uint64_t> failed_between_runtime_and_recorder_{0};
    std::atomic<std::uint64_t> lost_between_recorder_and_spool_{0};

    std::atomic<std::uint64_t> control_offered_{0};
    std::atomic<std::uint64_t> control_accepted_{0};
    std::atomic<std::uint64_t> control_spool_committed_{0};
    std::atomic<std::uint64_t> control_rejected_{0};
    std::atomic<std::uint64_t> lost_between_control_acceptance_and_spool_{0};

    std::atomic<std::uint64_t> rejected_after_close_data_{0};
    std::atomic<std::uint64_t> rejected_after_close_control_{0};

    std::atomic<std::uint64_t> frames_accepted_{0};
    std::atomic<std::uint64_t> discontinuities_accepted_{0};
    std::atomic<std::uint64_t> signal_blocks_recorded_{0};
    std::atomic<std::uint64_t> signal_gaps_recorded_{0};
    std::atomic<std::uint64_t> payload_bytes_copied_{0};

    /// The two ordinals of contract section 1.1. `next_data_ordinal_` is
    /// monotonic across frames *and* discontinuities; `next_frame_ordinal_`
    /// counts frame rows only. Both are owned by the callback thread.
    std::atomic<std::uint64_t> next_data_ordinal_{0};
    std::atomic<std::uint64_t> next_frame_ordinal_{0};
    std::atomic<std::uint64_t> next_control_ordinal_{0};
    /// The child-row ordinals the finalizer's block and gap ledgers need.
    /// Assigned in the callback, which is the only writer.
    std::atomic<std::uint64_t> next_signal_block_ordinal_{0};
    std::atomic<std::uint64_t> next_signal_gap_ordinal_{0};

    LatchedPosition data_first_loss_{};
    LatchedPosition data_first_rejection_{};
    LatchedPosition control_first_loss_{};
    LatchedPosition control_first_rejection_{};

    // --- the reserved fault path -------------------------------------------
    //
    // Bounded storage that does not depend on free space in the control queue
    // (contract section 1.1). Claimed once; the worker commits it.

    std::atomic<bool> fault_claimed_{false};
    std::atomic<bool> fault_published_{false};
    std::atomic<bool> fault_committed_{false};
    FaultPayloadFields reserved_fault_{};
    std::array<std::byte, kFaultPayloadBytes> reserved_fault_payload_{};

    // --- terminal provenance ------------------------------------------------

    std::atomic<bool> terminal_intent_latched_{false};
    std::atomic<RequestedTerminalIntent> requested_terminal_intent_{
        RequestedTerminalIntent::normal};
    std::atomic<CaptureOutcome> capture_outcome_{CaptureOutcome::normal};
    std::atomic<bool> capture_outcome_known_{false};
    std::atomic<EffectiveSessionOutcome> effective_session_outcome_{
        EffectiveSessionOutcome::normal};
    std::atomic<bool> session_created_{false};
    std::atomic<bool> spool_ended_cleanly_{false};
    std::atomic<bool> drain_timed_out_{false};
    std::array<char, kTerminalReasonBytes> terminal_reason_{};
    std::size_t terminal_reason_size_{};

    // --- worker-local staging bookkeeping ----------------------------------

    std::size_t staged_bytes_{};
    std::size_t staged_records_{};
    std::uint64_t committed_transactions_{};

    /// Set by `request_checkpoint()` on a control thread and cleared by the
    /// worker when it honours the request. A flag rather than a count: several
    /// requests arriving between two transactions ask for the same thing, and
    /// a queue of them would make the worker write a run of identical
    /// checkpoints for no added recovery point.
    std::atomic<bool> checkpoint_requested_{false};

    /// The writer's extents, republished by the worker so a status call never
    /// reaches into the writer across a thread boundary. Per store, as an
    /// extent, never as a stage count (contract section 1.2).
    std::atomic<std::uint64_t> spool_committed_extent_{0};
    std::atomic<std::uint64_t> spool_durable_extent_{0};
    std::atomic<std::uint64_t> spool_committed_transactions_{0};
    DurabilityPolicy spool_durability_policy_{DurabilityPolicy::checkpoint_sync};

    /// Mirrors of the two queues' capacities. `status()` is called from any
    /// thread (a monitoring/control thread, not the critical callback), while
    /// `release_storage()` zeroes the queues' plain `n_slots_` from the
    /// shutdown thread. Reading the queue's capacity from another thread would
    /// race that plain member, so the capacity is mirrored to an atomic set at
    /// `prepare()` and cleared at release, and `status()` reads the mirror.
    std::atomic<std::size_t> data_queue_capacity_{0};
    std::atomic<std::size_t> control_queue_capacity_{0};
};

} // namespace neurale::recording
