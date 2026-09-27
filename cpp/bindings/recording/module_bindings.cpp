/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// \file
/// The provisional Python surface over the native critical recorder.
///
/// "Provisional" is a contract term here, not a hedge (contract section 2,
/// decision 4): this is documented as experimental and it is not the stable
/// public API. `SessionRecorder` (`docs/development/native_recording_replay.md`
/// section 2) is the sole stable entry point; what survives here is the
/// private boundary that facade delegates to, kept so backend and fault
/// fixtures can exercise the implementation without publishing a second
/// recorder API. Nothing in `neurale.recording`'s stable surface changes
/// because this exists.
///
/// Two rules from contract section 2 shape every function below. Native
/// recording is **explicitly selected** -- nothing here is reachable by
/// accident from the existing recorder. And there is **no silent fallback** in
/// either direction: a plan, prepare, readiness, queue, or writer failure comes
/// back as an explicit status code that the Python facade turns into an
/// exception naming what failed. Falling back to the Python recorder after a
/// native failure is forbidden, and so is the reverse.
///
/// The recorder core itself never sees Python. Everything that crosses this
/// boundary crosses it on a control thread: the plan at `prepare()`, control
/// records at submission, and a status snapshot on demand. The data plane runs
/// entirely inside the runtime and the recorder's worker, takes no GIL, and is
/// not reachable from here at all.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <neurale/streaming/runtime.h>

#include "experiment_attachment.h"
#include "memory_spool_file.h"
#include "record_payloads.h"
#include "recorder.h"
#include "recorder_status.h"
#include "recording_plan.h"
#include "spool_file.h"
#include "spool_layout.h"

namespace py = pybind11;

using namespace neurale::recording;
using neurale::streaming::NativeStreamRunner;
using neurale::streaming::ObserverEdgeConfig;
using neurale::streaming::StreamStatus;

namespace
{

/// Which store the recorder writes its spool into.
///
/// This is an explicit choice with an explicit consequence, which is why it is
/// an enum a caller has to name rather than a default the facade picks.
enum class SpoolBackend : std::uint8_t
{
    /// `MemorySpoolFile`. Bounded, genuinely cancellable, and **not durable**.
    memory = 0,
    /// `PlatformSpoolFile`. Durable according to the selected sync policy, but
    /// refused by the critical-recorder readiness gate because platform file
    /// I/O cannot guarantee bounded cancellation and worker reclamation.
    file = 1,
    /// `BoundedMappedSpoolFile`. Fixed-capacity resident pages, bounded for the
    /// critical writer and process-crash persistent under `buffered`. Linux
    /// accepts tmpfs; Windows accepts a preallocated local file mapping.
    mapped = 2,
};

/// Everything a `NativeRecordingPlan` borrows, owned on this side of the
/// boundary.
///
/// `NativeRecordingPlan` is a view type by design -- it holds spans into the
/// caller's storage and never copies. That is right for the recorder and wrong
/// for a Python caller, whose `bytes` object may be collected the moment the
/// call returns. This struct is the storage that makes the borrow safe: Python
/// fills it, it lives as long as the recorder, and `view()` is the only place
/// the spans are formed.
struct OwnedRecordingPlan
{
    std::string session_id{};
    std::array<std::uint8_t, kSessionUuidBytes> session_uuid{};
    std::uint64_t created_unix_nanos{};
    std::vector<std::byte> plan_document{};
    std::array<std::uint8_t, kPlanFingerprintBytes> plan_fingerprint{};
    std::uint64_t native_session_id{};
    std::uint32_t native_schema_id{};
    std::vector<PlannedSignalRecording> recorded_signals{};

    std::size_t frame_queue_capacity{256};
    std::size_t control_queue_capacity{1024};
    std::uint32_t max_blocks_per_frame{};
    std::uint64_t max_frame_payload_bytes{};
    std::uint32_t max_signal_gaps_per_discontinuity{8};
    std::size_t max_control_payload_bytes{4096};
    std::size_t max_records_per_transaction{256};
    std::size_t max_transaction_bytes{1u << 20};
    std::uint64_t checkpoint_interval_transactions{0};
    std::uint64_t worker_idle_poll_nanos{200000};
    std::uint64_t drain_timeout_nanos{5000000000ULL};
    DurabilityPolicy durability_policy{DurabilityPolicy::buffered};

    [[nodiscard]] NativeRecordingPlan view() const noexcept
    {
        NativeRecordingPlan plan{};
        plan.session_id = session_id;
        plan.session_uuid = session_uuid;
        plan.created_unix_nanos = created_unix_nanos;
        plan.plan_document = plan_document;
        plan.plan_fingerprint = plan_fingerprint;
        plan.native_session_id = native_session_id;
        plan.native_schema_id = native_schema_id;
        plan.recorded_signals = recorded_signals;
        plan.frame_queue_capacity = frame_queue_capacity;
        plan.control_queue_capacity = control_queue_capacity;
        plan.max_blocks_per_frame = max_blocks_per_frame;
        plan.max_frame_payload_bytes = max_frame_payload_bytes;
        plan.max_signal_gaps_per_discontinuity = max_signal_gaps_per_discontinuity;
        plan.max_control_payload_bytes = max_control_payload_bytes;
        plan.max_records_per_transaction = max_records_per_transaction;
        plan.max_transaction_bytes = max_transaction_bytes;
        plan.checkpoint_interval_transactions = checkpoint_interval_transactions;
        plan.worker_idle_poll_nanos = worker_idle_poll_nanos;
        plan.drain_timeout_nanos = drain_timeout_nanos;
        plan.durability_policy = durability_policy;
        return plan;
    }
};

/// Runtime observer used only when an ExperimentSession owns the recorder's
/// terminal ordering. Runtime terminal means the data producer is quiet; it
/// deliberately does not seal the recorder, because the experiment bridge
/// still has its final trace and summary to submit on the control plane.
class ExperimentManagedRecorderObserver final : public neurale::streaming::NativeCriticalObserver
{
  public:
    explicit ExperimentManagedRecorderObserver(NativeRecorderCore& core) noexcept : core_(core) {}

    [[nodiscard]] StreamStatus ready_for_runtime() noexcept override
    {
        return core_.ready_for_runtime();
    }
    [[nodiscard]] StreamStatus start_observing() noexcept override
    {
        return core_.state() == RecorderLifecycleState::recording ? StreamStatus::ok
                                                                  : core_.start_observing();
    }
    [[nodiscard]] StreamStatus health() const noexcept override
    {
        return core_.health();
    }
    [[nodiscard]] StreamStatus
    observe_accepted(neurale::streaming::FrameView frame,
                     neurale::streaming::RuntimeAcceptance acceptance) noexcept override
    {
        return core_.observe_accepted(frame, acceptance);
    }
    [[nodiscard]] StreamStatus handle_accepted_discontinuity(
        const neurale::streaming::Discontinuity& discontinuity,
        neurale::streaming::RuntimeAcceptance acceptance) noexcept override
    {
        return core_.handle_accepted_discontinuity(discontinuity, acceptance);
    }
    void
    note_rejected_before_acceptance(neurale::streaming::RejectedMessage message) noexcept override
    {
        core_.note_rejected_before_acceptance(message);
    }
    void publish_primary_fault(const neurale::streaming::FaultRecord& fault) noexcept override
    {
        core_.publish_primary_fault(fault);
    }
    [[nodiscard]] StreamStatus drain(neurale::streaming::RuntimeTerminalNotice) noexcept override
    {
        return StreamStatus::ok;
    }
    void cancel() noexcept override {}
    [[nodiscard]] StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }

  private:
    NativeRecorderCore& core_;
};

/// Copy a fixed-width identity out of a Python `bytes` object, refusing a wrong
/// length rather than padding it. A short session UUID or plan fingerprint that
/// were silently zero-extended would produce a spool superblock claiming an
/// identity nothing else in the system agrees with.
template <std::size_t N>
void copy_fixed(std::array<std::uint8_t, N>& out, const py::bytes& value, const char* label)
{
    const std::string_view bytes = static_cast<std::string_view>(value);
    if (bytes.size() != N)
    {
        throw std::invalid_argument(std::string{label} + " must be exactly " + std::to_string(N) +
                                    " bytes, got " + std::to_string(bytes.size()));
    }
    for (std::size_t i = 0; i < N; ++i)
    {
        out[i] = static_cast<std::uint8_t>(static_cast<unsigned char>(bytes[i]));
    }
}

/// The recorder, its store, and its clock, kept together for exactly as long as
/// the Python object lives.
///
/// The order of the members is the order they must be destroyed in reverse:
/// the core's destructor releases its worker, and the worker is the thing that
/// touches the spool, so the spool must still exist while that happens.
class PyNativeRecorder
{
  public:
    PyNativeRecorder(OwnedRecordingPlan plan, SpoolBackend backend, std::size_t memory_capacity,
                     std::string path)
        : plan_(std::move(plan)), backend_(backend), memory_capacity_(memory_capacity),
          path_(std::move(path))
    {
        core_ = std::make_unique<NativeRecorderCore>();
        experiment_observer_ = std::make_unique<ExperimentManagedRecorderObserver>(*core_);
    }

    ~PyNativeRecorder()
    {
        // A timed-out disk request can still own core buffers. Destruction is
        // the final blocking ownership barrier, unlike retryable close().
        py::gil_scoped_release release;
        core_.reset();
    }

    /// Allocate every bounded resource, the store included.
    ///
    /// The store is created *here* rather than in the constructor because
    /// `created` is defined as "no plan, no resources, no session on disk"
    /// (contract section 3.1), and a store that exists from construction is a
    /// resource -- for the file and mapped backends, a file on disk -- held by
    /// an object reporting that it holds none. The core already rolls a failed
    /// `prepare()` back to exactly `created` and states that the run leaves no
    /// artifact; creating the store before that promise could be kept is what
    /// made it untrue.
    ///
    /// A failure therefore releases the store again, so a second `prepare()`
    /// starts from the same place the first one did. Releasing is not
    /// unlinking: the path is the caller's, and which paths a failed attempt
    /// may remove is a policy decision that belongs where the path was chosen.
    ///
    /// Only a store *this* call created is released. A store already present
    /// belongs to a `prepare()` that succeeded -- the only way one survives --
    /// and the core is holding a reference to it, so a second `prepare()`
    /// refused as `wrong_state` must leave it exactly where it is.
    [[nodiscard]] RecorderStatusCode prepare()
    {
        // Answered afresh by every call. "This recorder created a spool at some
        // point" and "this attempt created one" are different facts, and only
        // the second licenses a caller to remove the path while cleaning up
        // after an attempt that failed.
        prepare_created_spool_ = false;
        const bool created_here = spool_ == nullptr;
        if (created_here)
        {
            create_store();
            prepare_created_spool_ = spool_created_here_;
        }
        const auto status = core_->prepare(plan_.view(), *spool_, clock_);
        if (status != RecorderStatusCode::ok && created_here)
        {
            release_store();
        }
        return status;
    }

    /// Undo a successful `prepare()` and return the recorder to `created`.
    ///
    /// The caller of `prepare()` is a facade that does more than this call: it
    /// also writes the sidecar that makes the spool interpretable offline. When
    /// that second half fails, one `prepare()` failed as far as anyone outside
    /// can tell, and contract section 3.1 is unconditional about what a failed
    /// `prepare()` leaves behind -- no partial resource, no session artifact,
    /// and an object that may be prepared again. `close()` cannot deliver that:
    /// it is a transition to a *terminal* state, so a recorder closed here could
    /// never be retried and would have to be reported as `failed` instead,
    /// which is the state table's answer for a different thing entirely.
    ///
    /// The core has no un-prepare, and giving it one would add a backwards edge
    /// to a lifecycle whose forward-only shape is what makes it analysable.
    /// Instead the prepared core is closed and *replaced*: a freshly
    /// constructed core is `created` by construction, and it is the same object
    /// identity to Python either way. The store is released after the old core
    /// is destroyed, because the core's worker is what touches it.
    ///
    /// Releasing is not unlinking, here as in `prepare()`. Whether the path may
    /// be removed is the caller's decision and is made where the path was
    /// chosen; what this guarantees is that nothing native still holds it.
    [[nodiscard]] RecorderStatusCode rollback_prepare()
    {
        if (core_->state() != RecorderLifecycleState::prepared)
        {
            return RecorderStatusCode::wrong_state;
        }
        static_cast<void>(core_->close());
        core_ = std::make_unique<NativeRecorderCore>();
        experiment_observer_ = std::make_unique<ExperimentManagedRecorderObserver>(*core_);
        release_store();
        return RecorderStatusCode::ok;
    }

    [[nodiscard]] py::capsule prepare_experiment_attachment()
    {
        if (core_->state() != RecorderLifecycleState::created || spool_ != nullptr)
            throw std::runtime_error(
                "experiment attachment requires a created recorder with no allocated store");
        prepare_created_spool_ = false;
        create_store();
        prepare_created_spool_ = spool_created_here_;
        plan_view_ = plan_.view();
        experiment_attachment_ = {core_.get(), &plan_view_, spool_.get(), &clock_,
                                  experiment_observer_.get()};
        return py::capsule(&experiment_attachment_, kExperimentRecorderAttachmentCapsule);
    }

    void rollback_experiment_attachment() noexcept
    {
        if (core_->state() == RecorderLifecycleState::created)
        {
            release_store();
            experiment_attachment_ = {};
        }
    }

    [[nodiscard]] StreamStatus attach_experiment(NativeStreamRunner& runner,
                                                 ObserverEdgeConfig config)
    {
        if (experiment_attachment_.data_observer == nullptr)
            return StreamStatus::invalid_state;
        return runner.add_input_critical_observer(*experiment_attachment_.data_observer, config);
    }

    /// Register the recorder as the runtime's critical observer.
    ///
    /// Ordering is the caller's responsibility and the facade enforces it: the
    /// edge has to exist before `runner.prepare()`, and the readiness gate runs
    /// inside `runner.arm()`. Attaching later would leave the runtime armed
    /// against a recorder it never gated.
    [[nodiscard]] StreamStatus attach(NativeStreamRunner& runner, ObserverEdgeConfig config)
    {
        return runner.add_critical_observer(*core_, config);
    }

    /// Drive the readiness gate with no runtime attached.
    ///
    /// The gate's argument is what the runtime knows and the recorder does not
    /// -- whether every critical edge is attached and lossless -- so a caller
    /// asserting it without a runtime is asserting something that is not true
    /// of a real session. That is why this is on the private binding and not on
    /// the facade: `NativeSessionRecorder` has exactly one path to `ready`, and
    /// it goes through `runner.arm()`.
    ///
    /// It exists because the recorder core supports being driven standalone --
    /// `observe_frame()` and `observe_discontinuity()` are documented as
    /// exactly that, and the C++ core tests use it -- and because the control
    /// plane cannot otherwise be tested deterministically: with a runtime
    /// driving it, the window between `start()` and the source running out is a
    /// race, and a test that submits into a race is testing the scheduler.
    [[nodiscard]] RecorderStatusCode
    standalone_pass_readiness_gate(bool edges_attached_and_lossless)
    {
        return core_->pass_readiness_gate(edges_attached_and_lossless);
    }

    /// Begin accepting on both planes with no runtime attached. See
    /// `standalone_pass_readiness_gate`.
    [[nodiscard]] RecorderStatusCode start()
    {
        return core_->start();
    }
    [[nodiscard]] RecorderStatusCode stop(const std::string& reason)
    {
        return core_->stop(reason);
    }
    [[nodiscard]] RecorderStatusCode abort(const std::string& reason)
    {
        return core_->abort(reason);
    }
    [[nodiscard]] RecorderStatusCode close()
    {
        const auto status = core_->close();
        if (status == RecorderStatusCode::ok && backend_ == SpoolBackend::file && spool_ != nullptr)
        {
            const auto closed = static_cast<PlatformSpoolFile*>(spool_.get())->close();
            if (closed.status != SpoolIoStatus::ok)
                throw std::runtime_error("could not close the spool file (platform error " +
                                         std::to_string(closed.platform_error) + ")");
        }
        if (status == RecorderStatusCode::ok && backend_ == SpoolBackend::mapped &&
            spool_ != nullptr)
        {
            auto* mapped = static_cast<BoundedMappedSpoolFile*>(spool_.get());
            const auto closed = mapped->close();
            if (closed.status != SpoolIoStatus::ok)
            {
                throw std::runtime_error(
                    "could not release and trim the mapped spool (platform error " +
                    std::to_string(closed.platform_error) + ")");
            }
        }
        return status;
    }
    [[nodiscard]] RecorderStatusCode finalize()
    {
        return core_->finalize();
    }
    [[nodiscard]] RecorderStatusCode abandon_finalization()
    {
        return core_->abandon_finalization();
    }

    [[nodiscard]] bool request_checkpoint()
    {
        return core_->request_checkpoint();
    }

    [[nodiscard]] bool submit_control(ProducerIdentityKind kind, std::uint64_t identity,
                                      std::uint32_t clock_domain, std::uint64_t time_ns,
                                      const py::bytes& body)
    {
        const std::string_view bytes = static_cast<std::string_view>(body);
        const ControlSubmission submission{
            .kind = kind,
            .identity = identity,
            .clock_domain = clock_domain,
            .time_ns = time_ns,
            .body = std::span<const std::byte>{reinterpret_cast<const std::byte*>(bytes.data()),
                                               bytes.size()},
        };
        return core_->submit_control(submission);
    }

    [[nodiscard]] RecorderStatus status() const
    {
        return core_->status();
    }
    [[nodiscard]] RecorderLifecycleState state() const noexcept
    {
        return core_->state();
    }

    /// Whether the store behind this recorder survives a crash. False for the
    /// memory store, and the reason `spool_durable_extent` must never be read
    /// as a survival claim on its own (see `memory_spool_file.h`).
    [[nodiscard]] bool store_provides_crash_durability() const noexcept
    {
        return backend_ != SpoolBackend::memory;
    }

    /// Whether this recorder's own exclusive create is what put the store there.
    ///
    /// The answer comes from the create itself and from nowhere else. Both
    /// stores open with `O_RDWR | O_CREAT | O_EXCL`, so exactly one caller can
    /// win a path, and this flag records whether it was this one -- which is
    /// the only thing that can safely decide, later, whether the path may be
    /// removed.
    ///
    /// A caller cannot derive this by looking first. "The path does not exist"
    /// and "therefore it is mine" are two observations with a scheduling window
    /// between them, and the file that appears inside that window is, in the one
    /// case that produces it, a crashed run's only reconstruction input. The
    /// exclusive create is the atomic operation that answers the question, so
    /// the answer is published from there rather than re-derived.
    ///
    /// It stays true once a create has succeeded, including after the store is
    /// released again: a released store leaves its path behind, and the path is
    /// still this recorder's to remove. It does *not* survive the next attempt
    /// at the path -- see `create_store()`, which answers the question again
    /// rather than carrying the previous answer forward over a different file.
    [[nodiscard]] bool spool_created_by_this_recorder() const noexcept
    {
        return spool_created_here_;
    }

    /// Whether the most recent `prepare()` is what created the store.
    ///
    /// Narrower than `spool_created_by_this_recorder`, and deliberately so: it
    /// is the fact a caller cleaning up after a `prepare()` that raised needs,
    /// because the question there is what *this attempt* created. A `prepare()`
    /// refused as `wrong_state` created nothing -- the store at the path
    /// belongs to the earlier `prepare()` that succeeded and is still holding
    /// it, and a caller that removed it while tidying up after a redundant call
    /// would destroy a live session's spool.
    [[nodiscard]] bool spool_created_by_last_prepare() const noexcept
    {
        return prepare_created_spool_;
    }

    /// Everything written to the store so far, and nothing when there is no
    /// store yet: a recorder that has not been prepared has written nothing,
    /// which is what an empty snapshot says.
    [[nodiscard]] py::bytes spool_snapshot() const
    {
        if (backend_ == SpoolBackend::file && core_->status().worker_running &&
            core_->state() != RecorderLifecycleState::ready)
            throw std::runtime_error("cannot snapshot a spool while its writer is running");
        if (spool_ == nullptr)
        {
            return py::bytes("", 0);
        }
        const auto bytes = spool_->size();
        std::vector<std::byte> buffer(static_cast<std::size_t>(bytes));
        if (bytes != 0)
        {
            const auto result = spool_->read_at(0, buffer);
            buffer.resize(result.transferred);
        }
        return py::bytes(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    }

  private:
    void create_store()
    {
        // A fresh attempt at the path re-answers the ownership question. A
        // `true` left over from an earlier store -- one this recorder created,
        // released, and whose path the caller then removed -- says nothing
        // about whatever is at the path now, and if this attempt loses the
        // exclusive create the file there belongs to whoever won it.
        spool_created_here_ = false;
        if (backend_ == SpoolBackend::memory)
        {
            spool_ = std::make_unique<MemorySpoolFile>(memory_capacity_);
            spool_created_here_ = true;
        }
        else if (backend_ == SpoolBackend::file)
        {
            auto file = std::make_unique<PlatformSpoolFile>();
            const auto result = file->create(path_, memory_capacity_);
            if (result.status != SpoolIoStatus::ok)
            {
                throw std::runtime_error("could not create the spool file at '" + path_ +
                                         "' (platform error " +
                                         std::to_string(result.platform_error) + ")");
            }
            // Set only after the exclusive create returned `ok`: that call is
            // the one that either won the path or found somebody else on it.
            spool_created_here_ = true;
            spool_ = std::move(file);
        }
        else
        {
            auto mapped = std::make_unique<BoundedMappedSpoolFile>();
            const auto result = mapped->create(path_, memory_capacity_);
            if (result.status != SpoolIoStatus::ok)
            {
                throw std::runtime_error("could not create the bounded mapped spool at '" + path_ +
                                         "' (platform error " +
                                         std::to_string(result.platform_error) + ")");
            }
            // `BoundedMappedSpoolFile::create` unlinks the path itself if any
            // step after the exclusive create fails, so `ok` is exactly the case
            // where this recorder both created the path and still holds it.
            spool_created_here_ = true;
            spool_ = std::move(mapped);
        }
    }

    /// Undo `create_store()` after a failed `prepare()`.
    ///
    /// The core rolled itself back and no longer holds the store, so releasing
    /// it here cannot strand a reference. A mapped store is closed first
    /// because its pages are locked; the close status is deliberately not
    /// raised on -- this path is already failing for another reason, and
    /// replacing that reason with a cleanup error would hide it.
    void release_store() noexcept
    {
        if (spool_ != nullptr && backend_ == SpoolBackend::mapped)
        {
            static_cast<BoundedMappedSpoolFile*>(spool_.get())->close();
        }
        spool_.reset();
    }

    OwnedRecordingPlan plan_;
    SpoolBackend backend_;
    std::size_t memory_capacity_;
    std::string path_;
    bool spool_created_here_{false};
    bool prepare_created_spool_{false};
    std::unique_ptr<SpoolFile> spool_{};
    SystemUnixClock clock_{};
    std::unique_ptr<NativeRecorderCore> core_{};
    std::unique_ptr<ExperimentManagedRecorderObserver> experiment_observer_{};
    NativeRecordingPlan plan_view_{};
    ExperimentRecorderAttachment experiment_attachment_{};
};

void bind_enums(py::module_& recording)
{
    py::enum_<RecorderLifecycleState>(recording, "RecorderLifecycleState")
        .value("CREATED", RecorderLifecycleState::created)
        .value("PREPARED", RecorderLifecycleState::prepared)
        .value("READY", RecorderLifecycleState::ready)
        .value("RECORDING", RecorderLifecycleState::recording)
        .value("DRAINING", RecorderLifecycleState::draining)
        .value("STOPPED", RecorderLifecycleState::stopped)
        .value("FINALIZING", RecorderLifecycleState::finalizing)
        .value("FINALIZATION_FAILED", RecorderLifecycleState::finalization_failed)
        .value("CLOSED", RecorderLifecycleState::closed)
        .value("FAILED", RecorderLifecycleState::failed);

    py::enum_<RecorderStatusCode>(recording, "RecorderStatusCode")
        .value("OK", RecorderStatusCode::ok)
        .value("WRONG_STATE", RecorderStatusCode::wrong_state)
        .value("INVALID_PLAN", RecorderStatusCode::invalid_plan)
        .value("PREPARE_FAILED", RecorderStatusCode::prepare_failed)
        .value("NOT_READY", RecorderStatusCode::not_ready)
        .value("WRITER_FAILED", RecorderStatusCode::writer_failed)
        .value("DRAIN_TIMED_OUT", RecorderStatusCode::drain_timed_out)
        .value("NOT_IMPLEMENTED", RecorderStatusCode::not_implemented);

    py::enum_<FinalizationStatus>(recording, "FinalizationStatus")
        .value("NOT_STARTED", FinalizationStatus::not_started)
        .value("RUNNING", FinalizationStatus::running)
        .value("FAILED_RETRYABLE", FinalizationStatus::failed_retryable)
        .value("SUCCEEDED", FinalizationStatus::succeeded)
        .value("ABANDONED", FinalizationStatus::abandoned);

    py::enum_<EffectiveSessionOutcome>(recording, "EffectiveSessionOutcome")
        .value("NORMAL", EffectiveSessionOutcome::normal)
        .value("ABORTED", EffectiveSessionOutcome::aborted)
        .value("FAULTED", EffectiveSessionOutcome::faulted);

    py::enum_<CaptureOutcome>(recording, "CaptureOutcome")
        .value("NORMAL", CaptureOutcome::normal)
        .value("ABORTED", CaptureOutcome::aborted)
        .value("FAULTED", CaptureOutcome::faulted);

    py::enum_<RequestedTerminalIntent>(recording, "RequestedTerminalIntent")
        .value("NORMAL", RequestedTerminalIntent::normal)
        .value("ABORTED", RequestedTerminalIntent::aborted)
        .value("FAULT", RequestedTerminalIntent::fault);

    py::enum_<RecoverabilityAnswer>(recording, "RecoverabilityAnswer")
        .value("NOT_APPLICABLE", RecoverabilityAnswer::not_applicable)
        .value("RECOVERABLE", RecoverabilityAnswer::recoverable)
        .value("UNRECOVERABLE", RecoverabilityAnswer::unrecoverable);

    py::enum_<PositionTag>(recording, "PositionTag")
        .value("ABSENT", PositionTag::absent)
        .value("ORDINAL", PositionTag::ordinal)
        .value("PRODUCER_IDENTITY", PositionTag::producer_identity);

    py::enum_<ProducerIdentityKind>(recording, "ProducerIdentityKind")
        .value("NONE", ProducerIdentityKind::none)
        .value("FRAME", ProducerIdentityKind::frame)
        .value("DISCONTINUITY", ProducerIdentityKind::discontinuity)
        .value("EVENTS", ProducerIdentityKind::events)
        .value("TRIALS", ProducerIdentityKind::trials)
        .value("EXPERIMENT_STATES", ProducerIdentityKind::experiment_states)
        .value("COMMANDS", ProducerIdentityKind::commands)
        .value("TARGETS", ProducerIdentityKind::targets)
        .value("LABELS", ProducerIdentityKind::labels)
        .value("ASSISTANCE", ProducerIdentityKind::assistance)
        .value("FAULTS", ProducerIdentityKind::faults)
        .value("TASK_VARIABLES", ProducerIdentityKind::task_variables);

    py::enum_<CompletenessVerdict>(recording, "CompletenessVerdict")
        .value("VERIFIED_COMPLETE", CompletenessVerdict::verified_complete)
        .value("VERIFIED_INCOMPLETE", CompletenessVerdict::verified_incomplete)
        .value("UNVERIFIED_LEGACY", CompletenessVerdict::unverified_legacy);

    py::enum_<FaultOrigin>(recording, "FaultOrigin")
        .value("RUNTIME", FaultOrigin::runtime)
        .value("RECORDER", FaultOrigin::recorder);

    py::enum_<RecorderFaultReason>(recording, "RecorderFaultReason")
        .value("NONE", RecorderFaultReason::none)
        .value("DATA_QUEUE_SATURATED", RecorderFaultReason::data_queue_saturated)
        .value("CONTROL_QUEUE_SATURATED", RecorderFaultReason::control_queue_saturated)
        .value("PLAN_VIOLATION", RecorderFaultReason::plan_violation)
        .value("SPOOL_WRITER_FAILED", RecorderFaultReason::spool_writer_failed)
        .value("DRAIN_TIMEOUT", RecorderFaultReason::drain_timeout)
        .value("EDGE_REJECTION", RecorderFaultReason::edge_rejection)
        .value("READINESS_GATE_FAILED", RecorderFaultReason::readiness_gate_failed);

    py::enum_<DurabilityPolicy>(recording, "DurabilityPolicy")
        .value("BUFFERED", DurabilityPolicy::buffered)
        .value("CHECKPOINT_SYNC", DurabilityPolicy::checkpoint_sync)
        .value("TRANSACTION_SYNC", DurabilityPolicy::transaction_sync);

    py::enum_<SpoolBackend>(recording, "SpoolBackend")
        .value("MEMORY", SpoolBackend::memory)
        .value("FILE", SpoolBackend::file)
        .value("MAPPED", SpoolBackend::mapped);
}

void bind_status(py::module_& recording)
{
    py::class_<RecorderFirstPosition>(recording, "RecorderFirstPosition", py::is_final())
        .def_readonly("tag", &RecorderFirstPosition::tag)
        .def_readonly("identity_kind", &RecorderFirstPosition::identity_kind)
        .def_readonly("ordinal", &RecorderFirstPosition::ordinal)
        .def_readonly("identity_value", &RecorderFirstPosition::identity_value)
        .def_property_readonly("present", &RecorderFirstPosition::present);

    py::class_<RecorderPrimaryFault>(recording, "RecorderPrimaryFault", py::is_final())
        .def_readonly("present", &RecorderPrimaryFault::present)
        .def_readonly("origin", &RecorderPrimaryFault::origin)
        .def_readonly("reason", &RecorderPrimaryFault::reason)
        .def_readonly("stream_status", &RecorderPrimaryFault::stream_status)
        .def_readonly("fault_code", &RecorderPrimaryFault::fault_code)
        .def_readonly("fault_stage", &RecorderPrimaryFault::fault_stage)
        .def_readonly("detected_at_ns", &RecorderPrimaryFault::detected_at_ns)
        .def_readonly("frame_sequence", &RecorderPrimaryFault::frame_sequence)
        .def_readonly("data_ordinal_at_fault", &RecorderPrimaryFault::data_ordinal_at_fault)
        .def_readonly("control_ordinal_at_fault", &RecorderPrimaryFault::control_ordinal_at_fault)
        .def_readonly("committed", &RecorderPrimaryFault::committed);

    py::class_<RecorderFinalizationAttempt>(recording, "RecorderFinalizationAttempt",
                                            py::is_final())
        .def_readonly("attempt", &RecorderFinalizationAttempt::attempt)
        .def_readonly("outcome", &RecorderFinalizationAttempt::outcome)
        .def_readonly("code", &RecorderFinalizationAttempt::code)
        .def_readonly("started_at_ns", &RecorderFinalizationAttempt::started_at_ns)
        .def_readonly("ended_at_ns", &RecorderFinalizationAttempt::ended_at_ns);

    py::class_<RecorderStatus>(recording, "NativeRecorderStatus", py::is_final())
        .def_readonly("state", &RecorderStatus::state)
        // --- what the artifact is, independently of the object's state ------
        .def_readonly("session_created", &RecorderStatus::session_created)
        .def_readonly("spool_ended_cleanly", &RecorderStatus::spool_ended_cleanly)
        .def_readonly("sealed", &RecorderStatus::sealed)
        // `complete` and `completeness_verdict` are `X | None` on purpose. They
        // are absent while no sealed session exists, and `False` there would
        // report a verdict nobody reached (contract section 3.2).
        .def_readonly("completeness_verdict", &RecorderStatus::completeness_verdict)
        .def_readonly("complete", &RecorderStatus::complete)
        .def_readonly("accounting_verified", &RecorderStatus::accounting_verified)
        .def_readonly("finalization_required", &RecorderStatus::finalization_required)
        .def_readonly("recovery_required", &RecorderStatus::recovery_required)
        .def_readonly("recoverable", &RecorderStatus::recoverable)
        .def_readonly("requested_terminal_intent", &RecorderStatus::requested_terminal_intent)
        .def_readonly("terminal_intent_latched", &RecorderStatus::terminal_intent_latched)
        // `CaptureOutcome | None`. `None` is section 3.2's `unknown`: no
        // session-end record has frozen the answer, and `normal` would be a
        // guess dressed as a fact.
        .def_readonly("capture_outcome", &RecorderStatus::capture_outcome)
        .def_readonly("effective_session_outcome", &RecorderStatus::effective_session_outcome)
        .def_readonly("termination_kind", &RecorderStatus::termination_kind)
        .def_readonly("finalization_status", &RecorderStatus::finalization_status)
        .def_readonly("finalization_attempts", &RecorderStatus::finalization_attempts)
        // --- the acceptance ladder, one counter per stage, per plane --------
        .def_readonly("runtime_accepted", &RecorderStatus::runtime_accepted)
        .def_readonly("recorder_accepted", &RecorderStatus::recorder_accepted)
        .def_readonly("spool_committed", &RecorderStatus::spool_committed)
        .def_readonly("nrf_committed", &RecorderStatus::nrf_committed)
        .def_readonly("rejected_before_runtime_acceptance",
                      &RecorderStatus::rejected_before_runtime_acceptance)
        .def_readonly("failed_between_runtime_and_recorder",
                      &RecorderStatus::failed_between_runtime_and_recorder)
        .def_readonly("lost_between_recorder_and_spool",
                      &RecorderStatus::lost_between_recorder_and_spool)
        .def_readonly("lost_during_finalization", &RecorderStatus::lost_during_finalization)
        .def_readonly("control_offered", &RecorderStatus::control_offered)
        .def_readonly("control_accepted", &RecorderStatus::control_accepted)
        .def_readonly("control_spool_committed", &RecorderStatus::control_spool_committed)
        .def_readonly("control_nrf_committed", &RecorderStatus::control_nrf_committed)
        .def_readonly("control_rejected", &RecorderStatus::control_rejected)
        .def_readonly("lost_between_control_acceptance_and_spool",
                      &RecorderStatus::lost_between_control_acceptance_and_spool)
        .def_readonly("control_lost_during_finalization",
                      &RecorderStatus::control_lost_during_finalization)
        .def_readonly("rejected_after_close_data", &RecorderStatus::rejected_after_close_data)
        .def_readonly("rejected_after_close_control", &RecorderStatus::rejected_after_close_control)
        .def_readonly("data_first_loss", &RecorderStatus::data_first_loss)
        .def_readonly("data_first_rejection", &RecorderStatus::data_first_rejection)
        .def_readonly("control_first_loss", &RecorderStatus::control_first_loss)
        .def_readonly("control_first_rejection", &RecorderStatus::control_first_rejection)
        .def_readonly("primary_fault", &RecorderStatus::primary_fault)
        // --- diagnostics ----------------------------------------------------
        .def_readonly("frames_accepted", &RecorderStatus::frames_accepted)
        .def_readonly("discontinuities_accepted", &RecorderStatus::discontinuities_accepted)
        .def_readonly("signal_blocks_recorded", &RecorderStatus::signal_blocks_recorded)
        .def_readonly("signal_gaps_recorded", &RecorderStatus::signal_gaps_recorded)
        .def_readonly("payload_bytes_copied", &RecorderStatus::payload_bytes_copied)
        // --- health ----------------------------------------------------------
        .def_readonly("data_queue_pending", &RecorderStatus::data_queue_pending)
        .def_readonly("data_queue_capacity", &RecorderStatus::data_queue_capacity)
        .def_readonly("data_queue_high_water_mark", &RecorderStatus::data_queue_high_water_mark)
        .def_readonly("control_queue_pending", &RecorderStatus::control_queue_pending)
        .def_readonly("control_queue_capacity", &RecorderStatus::control_queue_capacity)
        .def_readonly("control_queue_high_water_mark",
                      &RecorderStatus::control_queue_high_water_mark)
        .def_readonly("worker_running", &RecorderStatus::worker_running)
        .def_readonly("spool_committed_extent", &RecorderStatus::spool_committed_extent)
        .def_readonly("spool_durable_extent", &RecorderStatus::spool_durable_extent)
        .def_readonly("spool_committed_transactions", &RecorderStatus::spool_committed_transactions)
        .def_readonly("spool_durability_policy", &RecorderStatus::spool_durability_policy)
        .def_readonly("spool_holds_committed_record", &RecorderStatus::spool_holds_committed_record)
        // --- reclamation ------------------------------------------------------
        .def_readonly("queue_storage_released", &RecorderStatus::queue_storage_released)
        .def_readonly("queue_storage_release_deferred",
                      &RecorderStatus::queue_storage_release_deferred)
        .def_readonly("spool_backend_cancellable", &RecorderStatus::spool_backend_cancellable);
}

void bind_plan(py::module_& recording)
{
    py::class_<PlannedSignalRecording>(recording, "PlannedSignalRecording", py::is_final())
        .def(py::init(
                 [](std::uint32_t signal_id, std::uint64_t max_block_bytes,
                    std::uint64_t max_block_samples)
                 {
                     return PlannedSignalRecording{.signal_id = signal_id,
                                                   .max_block_bytes = max_block_bytes,
                                                   .max_block_samples = max_block_samples};
                 }),
             py::arg("signal_id"), py::arg("max_block_bytes"), py::arg("max_block_samples"))
        .def_readonly("signal_id", &PlannedSignalRecording::signal_id)
        .def_readonly("max_block_bytes", &PlannedSignalRecording::max_block_bytes)
        .def_readonly("max_block_samples", &PlannedSignalRecording::max_block_samples);

    py::class_<OwnedRecordingPlan>(recording, "NativeRecordingPlan", py::is_final())
        .def(py::init(
                 [](const std::string& session_id, const py::bytes& session_uuid,
                    std::uint64_t created_unix_nanos, const py::bytes& plan_document,
                    const py::bytes& plan_fingerprint, std::uint64_t native_session_id,
                    std::uint32_t native_schema_id,
                    const std::vector<PlannedSignalRecording>& recorded_signals,
                    std::size_t frame_queue_capacity, std::size_t control_queue_capacity,
                    std::uint32_t max_blocks_per_frame, std::uint64_t max_frame_payload_bytes,
                    std::uint32_t max_signal_gaps_per_discontinuity,
                    std::size_t max_control_payload_bytes, std::size_t max_records_per_transaction,
                    std::size_t max_transaction_bytes,
                    std::uint64_t checkpoint_interval_transactions,
                    std::uint64_t worker_idle_poll_nanos, std::uint64_t drain_timeout_nanos,
                    DurabilityPolicy durability_policy)
                 {
                     auto plan = std::make_unique<OwnedRecordingPlan>();
                     plan->session_id = session_id;
                     copy_fixed(plan->session_uuid, session_uuid, "session_uuid");
                     plan->created_unix_nanos = created_unix_nanos;
                     const std::string_view document = static_cast<std::string_view>(plan_document);
                     plan->plan_document.resize(document.size());
                     for (std::size_t i = 0; i < document.size(); ++i)
                     {
                         plan->plan_document[i] =
                             static_cast<std::byte>(static_cast<unsigned char>(document[i]));
                     }
                     copy_fixed(plan->plan_fingerprint, plan_fingerprint, "plan_fingerprint");
                     plan->native_session_id = native_session_id;
                     plan->native_schema_id = native_schema_id;
                     plan->recorded_signals = recorded_signals;
                     plan->frame_queue_capacity = frame_queue_capacity;
                     plan->control_queue_capacity = control_queue_capacity;
                     plan->max_blocks_per_frame = max_blocks_per_frame;
                     plan->max_frame_payload_bytes = max_frame_payload_bytes;
                     plan->max_signal_gaps_per_discontinuity = max_signal_gaps_per_discontinuity;
                     plan->max_control_payload_bytes = max_control_payload_bytes;
                     plan->max_records_per_transaction = max_records_per_transaction;
                     plan->max_transaction_bytes = max_transaction_bytes;
                     plan->checkpoint_interval_transactions = checkpoint_interval_transactions;
                     plan->worker_idle_poll_nanos = worker_idle_poll_nanos;
                     plan->drain_timeout_nanos = drain_timeout_nanos;
                     plan->durability_policy = durability_policy;
                     return plan;
                 }),
             py::arg("session_id"), py::arg("session_uuid"), py::arg("created_unix_nanos"),
             py::arg("plan_document"), py::arg("plan_fingerprint"), py::arg("native_session_id"),
             py::arg("native_schema_id"), py::arg("recorded_signals"),
             py::arg("frame_queue_capacity"), py::arg("control_queue_capacity"),
             py::arg("max_blocks_per_frame"), py::arg("max_frame_payload_bytes"),
             py::arg("max_signal_gaps_per_discontinuity"), py::arg("max_control_payload_bytes"),
             py::arg("max_records_per_transaction"), py::arg("max_transaction_bytes"),
             py::arg("checkpoint_interval_transactions"), py::arg("worker_idle_poll_nanos"),
             py::arg("drain_timeout_nanos"), py::arg("durability_policy"))
        .def("validate", [](const OwnedRecordingPlan& plan) { return plan.view().validate(); })
        .def_readonly("session_id", &OwnedRecordingPlan::session_id)
        .def_readonly("native_session_id", &OwnedRecordingPlan::native_session_id)
        .def_readonly("native_schema_id", &OwnedRecordingPlan::native_schema_id)
        .def_readonly("frame_queue_capacity", &OwnedRecordingPlan::frame_queue_capacity)
        .def_readonly("control_queue_capacity", &OwnedRecordingPlan::control_queue_capacity)
        .def_readonly("durability_policy", &OwnedRecordingPlan::durability_policy);

    py::enum_<RecordingPlanStatus>(recording, "RecordingPlanStatus")
        .value("OK", RecordingPlanStatus::ok)
        .value("INVALID_IDENTITY", RecordingPlanStatus::invalid_identity)
        .value("INVALID_BOUND", RecordingPlanStatus::invalid_bound)
        .value("INVALID_SIGNAL_TABLE", RecordingPlanStatus::invalid_signal_table)
        .value("INCONSISTENT_BOUNDS", RecordingPlanStatus::inconsistent_bounds);
}

} // namespace

/// Defined in `replay_image_bindings.cpp`. The replay-image reader is a
/// separate, private surface: it reads a compiled cache, not a recorder.
void bind_replay_image_module(py::module_& recording);

/// Defined in `spool_bindings.cpp`. The spool reader is a separate, private
/// surface: it reads a committed container, and nothing on it can write one.
void bind_spool_module(py::module_& recording);

void bind_recording_module(py::module_& module)
{
    auto recording = module.def_submodule(
        "recording", "Provisional native critical recorder (experimental, not stable API).");

    bind_enums(recording);
    bind_status(recording);
    bind_plan(recording);
    bind_replay_image_module(recording);
    bind_spool_module(recording);

    py::class_<PyNativeRecorder>(recording, "_NativeRecorder", py::is_final())
        .def(py::init(
                 [](const OwnedRecordingPlan& plan, SpoolBackend backend,
                    std::size_t memory_capacity_bytes, const std::string& path)
                 {
                     return std::make_unique<PyNativeRecorder>(plan, backend, memory_capacity_bytes,
                                                               path);
                 }),
             py::arg("plan"), py::arg("backend"), py::arg("memory_capacity_bytes"),
             py::arg("path") = std::string{})
        .def("prepare", &PyNativeRecorder::prepare)
        .def("_prepare_experiment_attachment", &PyNativeRecorder::prepare_experiment_attachment)
        .def("_rollback_experiment_attachment", &PyNativeRecorder::rollback_experiment_attachment)
        .def("_attach_experiment", &PyNativeRecorder::attach_experiment, py::arg("runner"),
             py::arg("config"), py::keep_alive<2, 1>())
        // Bounded by the core's own drain timeout, and the wait must not hold
        // the GIL for the same reason `close()` does not.
        .def("rollback_prepare", &PyNativeRecorder::rollback_prepare,
             py::call_guard<py::gil_scoped_release>())
        // The runner keeps the recorder alive: the edge holds a bare reference
        // to the core, so a recorder collected while the runtime still points
        // at it would be a dangling critical observer.
        .def("attach", &PyNativeRecorder::attach, py::arg("runner"), py::arg("config"),
             py::keep_alive<2, 1>())
        // Standalone driving, for a recorder with no runtime attached. Named
        // so it cannot be mistaken for the ordinary path, and deliberately
        // absent from `NativeSessionRecorder`.
        .def("standalone_pass_readiness_gate", &PyNativeRecorder::standalone_pass_readiness_gate,
             py::arg("edges_attached_and_lossless"))
        .def("standalone_start", &PyNativeRecorder::start)
        // stop/abort/close wait for the worker under the plan's bounded drain
        // timeout. That wait must not hold the GIL: the worker takes no GIL, but
        // a caller blocking on it with the GIL held would stall every other
        // Python thread for the whole drain.
        .def("stop", &PyNativeRecorder::stop, py::arg("reason"),
             py::call_guard<py::gil_scoped_release>())
        .def("abort", &PyNativeRecorder::abort, py::arg("reason"),
             py::call_guard<py::gil_scoped_release>())
        .def("close", &PyNativeRecorder::close, py::call_guard<py::gil_scoped_release>())
        .def("finalize", &PyNativeRecorder::finalize)
        .def("abandon_finalization", &PyNativeRecorder::abandon_finalization)
        .def("submit_control", &PyNativeRecorder::submit_control, py::arg("kind"),
             py::arg("identity"), py::arg("clock_domain"), py::arg("time_ns"), py::arg("body"))
        .def("request_checkpoint", &PyNativeRecorder::request_checkpoint)
        .def("spool_snapshot", &PyNativeRecorder::spool_snapshot)
        .def_property_readonly("status", &PyNativeRecorder::status)
        .def_property_readonly("state", &PyNativeRecorder::state)
        .def_property_readonly("store_provides_crash_durability",
                               &PyNativeRecorder::store_provides_crash_durability)
        .def_property_readonly("spool_created_by_this_recorder",
                               &PyNativeRecorder::spool_created_by_this_recorder)
        .def_property_readonly("spool_created_by_last_prepare",
                               &PyNativeRecorder::spool_created_by_last_prepare);
}
