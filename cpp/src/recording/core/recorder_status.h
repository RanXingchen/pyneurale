/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The recorder's lifecycle state and its health/status surface.
///
/// Two things in recording code are called "state" and they are not the same
/// question (contract section 3.1): `RecorderLifecycleState` is the *object's*
/// -- which operations are legal now -- and the outcome fields below are the
/// *session's*. The recorder's failure state is `failed`; the word `faulted`
/// belongs to the session and appears only in `CaptureOutcome` and
/// `EffectiveSessionOutcome`. Nothing here lets one be read as the other.
///
/// Every acceptance-ladder stage the native path has is a distinct counter. A
/// single "recorded" number spanning stages is a contract violation, so there
/// is no such field and no accessor that computes one.

#include "record_payloads.h"
#include "spool_layout.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace neurale::recording
{

/// The object's lifecycle (contract section 3.1). `finalizing`,
/// `finalization_failed`, and the transitions into them are not yet
/// implemented; they are present in the enum because the state machine is
/// normative and an implementation that omitted them would have to renumber
/// later.
enum class RecorderLifecycleState : std::uint8_t
{
    created = 0,
    prepared = 1,
    ready = 2,
    recording = 3,
    draining = 4,
    stopped = 5,
    finalizing = 6,
    finalization_failed = 7,
    closed = 8,
    failed = 9,
};

/// The second dimension of contract section 3.2, independent of the session
/// outcome. This build never runs a finalizer, so a status it produces
/// reports `not_started` and never anything else.
enum class FinalizationStatus : std::uint8_t
{
    not_started = 0,
    running = 1,
    failed_retryable = 2,
    succeeded = 3,
    abandoned = 4,
};

/// Whether a *sealed* session's completeness was established, and on what
/// evidence (contract section 5.1). There is no `unknown` enumerator: the
/// absence of a verdict is spelled by the surrounding `std::optional` being
/// empty, and the contract is explicit that the two must not be conflated --
/// "there is no sealed session to judge" is not "there is one, and it carries
/// no evidence".
enum class CompletenessVerdict : std::uint8_t
{
    /// Nothing was lost and the checked accounting supports saying so.
    verified_complete = 0,
    /// Something was lost, or the session did not terminate normally.
    /// Incompleteness has cheaper proofs than completeness, so this does not
    /// require `accounting_verified`.
    verified_incomplete = 1,
    /// A sealed session that carries no accounting to check.
    unverified_legacy = 2,
};

/// What the session actually ended as. Distinct from `CaptureOutcome` because
/// escalation after the capture is over has to be expressible, and distinct
/// from `RequestedTerminalIntent` because a fault outranks a request without
/// rewriting it.
enum class EffectiveSessionOutcome : std::uint8_t
{
    normal = 0,
    aborted = 1,
    faulted = 2,
};

/// The answer to "given that recovery is required, can a committed prefix be
/// established?". `not_applicable` is the required answer wherever
/// `recovery_required` is false: `true` would send an operator to run recovery
/// over an artifact that is merely unfinalized, and `false` reads as "beyond
/// repair", which is the opposite of the truth.
enum class RecoverabilityAnswer : std::uint8_t
{
    not_applicable = 0,
    recoverable = 1,
    unrecoverable = 2,
};

/// Why a call did not do what was asked.
enum class RecorderStatusCode : std::uint8_t
{
    ok = 0,
    /// The call is not legal in the recorder's current state (section 3.1).
    wrong_state = 1,
    /// The plan is unusable. Nothing was allocated and nothing was created.
    invalid_plan = 2,
    /// A bounded resource could not be prepared.
    prepare_failed = 3,
    /// The readiness gate did not pass; every prepared resource was released.
    not_ready = 4,
    /// The spool writer failed, so the drain could not seal the session.
    writer_failed = 5,
    /// The drain did not finish inside the configured bound (section 4.7).
    drain_timed_out = 6,
    /// Deliberately not implemented yet: finalization and the
    /// diagnose/resume/abandon surface are not built; returning this is how
    /// the gap stays visible instead of being papered over with a fake
    /// success.
    not_implemented = 7,
};

/// One finalization attempt, kept as provenance. Contract section 3.2: failed
/// attempts are diagnostics and MUST NOT rewrite what the recording did, so
/// they accumulate here rather than mutating the session's outcome. Empty
/// until a finalizer runs -- an attempt list a caller could read as "we
/// tried" when nothing tried would be the same fabrication as a fake success.
struct RecorderFinalizationAttempt
{
    /// 1 for the first attempt.
    std::uint32_t attempt{};
    /// Where the attempt left `finalization_status`.
    FinalizationStatus outcome{FinalizationStatus::not_started};
    /// Why it stopped, if it failed.
    RecorderStatusCode code{RecorderStatusCode::ok};
    std::uint64_t started_at_ns{};
    std::uint64_t ended_at_ns{};
};

/// There is no code here for a plan violation, a saturated queue, a latched
/// fault, or a late submission. Those happen on the critical callback, which
/// answers in `neurale::streaming::StreamStatus` (or a bool, on the control
/// plane) and reports the reason through the primary fault and the per-handoff
/// counters. A second vocabulary for the same events would be one a caller had
/// to reconcile, and neither half would be authoritative.

/// One first-failed-or-lost position, in the tagged form contract section 5.1
/// requires. A reader must never compare an `ordinal` with a
/// `producer_identity`: they are positions in different sequences.
struct RecorderFirstPosition
{
    PositionTag tag{PositionTag::absent};
    ProducerIdentityKind identity_kind{ProducerIdentityKind::none};
    std::uint64_t ordinal{};
    std::uint64_t identity_value{};

    [[nodiscard]] bool present() const noexcept
    {
        return tag != PositionTag::absent;
    }
};

/// What the recorder latched about the fault that ended the session. A fault
/// raised while handling a fault is secondary and must not displace this
/// (contract section 4.2), so it is written exactly once.
struct RecorderPrimaryFault
{
    bool present{};
    FaultOrigin origin{FaultOrigin::recorder};
    RecorderFaultReason reason{RecorderFaultReason::none};
    /// The streaming `StreamStatus` as a raw value; the core does not depend
    /// on the enum's numbering, only carries it.
    std::uint8_t stream_status{};
    std::uint8_t fault_code{};
    std::uint8_t fault_stage{};
    std::uint64_t detected_at_ns{};
    std::uint64_t frame_sequence{};
    std::uint64_t data_ordinal_at_fault{};
    std::uint64_t control_ordinal_at_fault{};
    /// Whether the fault record itself reached a committed spool transaction.
    /// Escalation happens whether or not it did (contract section 3.2 rule 6);
    /// this field is what lets the finalizer say the row is missing rather
    /// than quietly report a lesser outcome.
    bool committed{};
};

/// The whole surface, taken as one snapshot. Reading it is a plain copy under
/// the recorder's status mutex -- never from the critical callback, which
/// publishes into atomics instead.
struct RecorderStatus
{
    RecorderLifecycleState state{RecorderLifecycleState::created};

    // --- what the artifact is, independently of the object's state ---------

    bool session_created{};
    bool spool_ended_cleanly{};
    bool sealed{};
    /// Contract section 5.1's three-field completeness surface. All three are
    /// derived by whoever is answering the call, never stored and read back, so
    /// an older artifact cannot pin a newer reader's judgement.
    ///
    /// `complete` is deliberately not a `bool`. **Completeness is a property of
    /// a sealed session**, so a recorder that has not been finalized has no
    /// verdict to report -- and `false` there would state the opposite of the
    /// truth: that the session was judged and found wanting. The empty optional
    /// is the fourth case of section 3.2 and is not a defect report.
    std::optional<CompletenessVerdict> completeness_verdict{};
    std::optional<bool> complete{};
    /// Was the persisted accounting *checked*? Never "was nothing lost" -- the
    /// contract separates the two because incompleteness has cheaper proofs
    /// than completeness, and a recovered session is exactly the case where
    /// `complete = false` coexists with unverified accounting.
    bool accounting_verified{};
    bool finalization_required{};
    bool recovery_required{};
    RecoverabilityAnswer recoverable{RecoverabilityAnswer::not_applicable};

    RequestedTerminalIntent requested_terminal_intent{RequestedTerminalIntent::normal};
    bool terminal_intent_latched{};
    /// How *taking the data* ended, or **absent** while no spool session-end
    /// record has frozen it -- which is contract section 3.2's `unknown`, and a
    /// value a caller must be able to read without reconstructing it.
    ///
    /// The container's `CaptureOutcome` has no `unknown` enumerator on purpose:
    /// `unknown` is what a *missing* session-end record means, and a writer may
    /// not state it. That is a rule about what may be written down, not about
    /// what a status may report, so the answer is absent here rather than
    /// spelled with a second boolean the caller has to combine.
    std::optional<CaptureOutcome> capture_outcome{};
    EffectiveSessionOutcome effective_session_outcome{EffectiveSessionOutcome::normal};
    /// The effective session outcome **as written on disk** in the NRF
    /// termination record. Empty while no such record exists, which is every
    /// status this build currently produces. It is a separate field from
    /// `effective_session_outcome` precisely so "what we concluded" and "what
    /// the artifact says" cannot be read as one answer.
    std::optional<EffectiveSessionOutcome> termination_kind{};
    FinalizationStatus finalization_status{FinalizationStatus::not_started};
    /// One entry per finalization attempt, oldest first. Empty until a
    /// finalizer runs.
    std::vector<RecorderFinalizationAttempt> finalization_attempts{};

    // --- the acceptance ladder, one counter per stage, per plane -----------

    std::uint64_t runtime_accepted{};
    std::uint64_t recorder_accepted{};
    std::uint64_t spool_committed{};
    /// Always 0 while no finalizer has run. It is reported rather than
    /// omitted because a native recorder has all five stages and must report
    /// all five.
    std::uint64_t nrf_committed{};

    std::uint64_t rejected_before_runtime_acceptance{};
    std::uint64_t failed_between_runtime_and_recorder{};
    std::uint64_t lost_between_recorder_and_spool{};
    std::uint64_t lost_during_finalization{};

    std::uint64_t control_offered{};
    std::uint64_t control_accepted{};
    std::uint64_t control_spool_committed{};
    std::uint64_t control_nrf_committed{};
    std::uint64_t control_rejected{};
    std::uint64_t lost_between_control_acceptance_and_spool{};
    std::uint64_t control_lost_during_finalization{};

    /// API misuse, not a recording loss. Belongs to no identity and must not
    /// change the session's verdict (section 1.3).
    std::uint64_t rejected_after_close_data{};
    std::uint64_t rejected_after_close_control{};

    RecorderFirstPosition data_first_loss{};
    RecorderFirstPosition data_first_rejection{};
    RecorderFirstPosition control_first_loss{};
    RecorderFirstPosition control_first_rejection{};

    RecorderPrimaryFault primary_fault{};

    // --- diagnostics: named distinctly, never substituted into an identity --

    std::uint64_t frames_accepted{};
    std::uint64_t discontinuities_accepted{};
    std::uint64_t signal_blocks_recorded{};
    std::uint64_t signal_gaps_recorded{};
    std::uint64_t payload_bytes_copied{};

    // --- health ------------------------------------------------------------

    std::uint64_t data_queue_pending{};
    std::uint64_t data_queue_capacity{};
    std::uint64_t data_queue_high_water_mark{};
    std::uint64_t control_queue_pending{};
    std::uint64_t control_queue_capacity{};
    std::uint64_t control_queue_high_water_mark{};
    bool worker_running{};

    /// Per store, as an extent, never as a stage count (contract section 1.2).
    std::uint64_t spool_committed_extent{};
    std::uint64_t spool_durable_extent{};
    std::uint64_t spool_committed_transactions{};
    DurabilityPolicy spool_durability_policy{DurabilityPolicy::checkpoint_sync};
    /// Whether the spool holds at least one committed record -- what decides
    /// whether it is retained or discarded (contract section 7).
    bool spool_holds_committed_record{};

    // --- reclamation --------------------------------------------------------
    //
    // A shutdown must not free storage a producer is still writing into, and
    // must not wait for that producer without a bound (contract section 4.7).
    // It does neither: it retires the storage and lets the last producer out
    // free it. These two fields are what make that visible instead of silent.

    /// Whether the queues' storage has actually been given back.
    bool queue_storage_released{};
    /// Whether reclamation was deferred for a producer or an in-flight disk
    /// request. Retry close after the worker finishes; never free live buffers.
    bool queue_storage_release_deferred{};
    /// Whether the backend guarantees bounded I/O cancellation. Ordinary files
    /// report false without weakening the bounded acquisition handoff.
    bool spool_backend_cancellable{};
};

[[nodiscard]] std::string_view to_string(RecorderLifecycleState state) noexcept;
[[nodiscard]] std::string_view to_string(CompletenessVerdict verdict) noexcept;
[[nodiscard]] std::string_view to_string(FinalizationStatus status) noexcept;
[[nodiscard]] std::string_view to_string(EffectiveSessionOutcome outcome) noexcept;
[[nodiscard]] std::string_view to_string(RecoverabilityAnswer answer) noexcept;
[[nodiscard]] std::string_view to_string(RecorderStatusCode code) noexcept;

} // namespace neurale::recording
