#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Stable orchestration facade for native critical recording.

The live data plane is entirely native. Python participates only before the
runtime starts, on explicitly submitted control records, and after capture
stops when the committed spool is finalized into canonical NRF.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ._errors import (
    FinalizationError,
    RecorderConfigError,
    RecorderRuntimeShutdownError,
    RecorderStateError,
)
from ._finalizer import (
    FinalizationProgress,
    FinalizationReport,
    FinalizerOptions,
    StagedConversion,
    discard_spool_bundle,
    finalize_spool,
    write_plan_sidecar,
)
from ._native_recorder import NativeRecorderOptions, NativeSessionRecorder
from ._plan import RecordingPlan, compile_recording_plan
from ._spec import RecorderConfig, RecorderState


def _enum_text(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    return str(name).lower() if name is not None else str(value).rsplit(".", 1)[-1].lower()


def _stable_native_options(
    plan: RecordingPlan, output: str | Path, *, native_session_id: int = 1
) -> NativeRecorderOptions:
    """Select continuous native file recording with bounded capture queues."""
    if sys.platform in ("linux", "win32"):
        target = Path(output).absolute()
        spool_path = target.with_name(f"{target.name}.{plan.session.session_id}.spool")
    else:
        raise RecorderConfigError(
            "SessionRecorder is not ready on this platform: the stable critical-recording "
            "path requires Linux or Windows local file storage"
        )
    return NativeRecorderOptions(
        spool="file",
        spool_path=spool_path,
        spool_capacity_bytes=plan.resource_bounds.spool_capacity_bytes,
        edge_capacity=plan.resource_bounds.frame_queue_capacity,
        native_session_id=native_session_id,
    )


def _require_lockable_spool_capacity(capacity_bytes: int) -> None:
    """Refuse a spool larger than this process may lock, before creating one.

    The mapped store locks its whole capacity at ``prepare()``: that is what
    makes the critical writer's append a resident-memory copy with no writeback
    path, and it is a host prerequisite rather than a recorder setting. A host
    whose ``RLIMIT_MEMLOCK`` is below the configured capacity cannot run this
    recorder at all, and the default capacity is 64 MiB while a common default
    limit is a few megabytes -- so the failure is ordinary, not exotic.

    Checking here turns ``mlock`` returning ``ENOMEM`` deep inside store
    creation, reported as "platform error 12", into a message naming the limit,
    the capacity, and the value to raise it to.

    The check is necessary, not sufficient: the limit is per process and covers
    every locked page in it, so a process holding other locked memory can still
    be refused by the kernel after passing this. It is a preflight, and the
    store's own failure remains the authority.
    """
    if sys.platform != "linux":
        return
    import resource

    soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft == resource.RLIM_INFINITY or soft >= capacity_bytes:
        return
    raise RecorderConfigError(
        f"this host allows {soft} bytes of locked memory (RLIMIT_MEMLOCK) and the recorder's "
        f"spool needs to lock {capacity_bytes}. The bounded tmpfs spool locks its whole capacity "
        "before recording starts, so this is a host prerequisite and there is no unlocked "
        f"fallback. Raise the limit (ulimit -l {max(1, capacity_bytes // 1024)} or a "
        "LimitMEMLOCK= in the service unit), or configure a smaller "
        "RecorderLimits.spool_capacity_bytes. Nothing was created."
    )


def _attach_note(error: BaseException, note: str) -> None:
    """Record something that went wrong *while handling* ``error``, on it."""
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(note)
    else:  # pragma: no cover - every supported Python has add_note
        error.__notes__ = [*getattr(error, "__notes__", ()), note]


@dataclass(frozen=True, slots=True)
class FinalizationAttempt:
    """One public, immutable finalization-attempt record."""

    attempt: int
    outcome: str
    category: str | None
    detail: str | None
    started_at_ns: int
    ended_at_ns: int


class RecorderStatus:
    """Immutable public view over capture and finalization status.

    Native capture counters remain authoritative and are forwarded unchanged.
    Only fields owned by offline finalization are overlaid here.
    """

    # Forwarded native fields are declared so the stable public status remains
    # useful to type checkers without importing the native extension here.
    capture_outcome: Any
    control_accepted: int
    control_first_loss: Any
    control_first_rejection: Any
    control_offered: int
    control_queue_capacity: int
    control_queue_high_water_mark: int
    control_queue_pending: int
    control_rejected: int
    data_first_loss: Any
    data_first_rejection: Any
    data_queue_capacity: int
    data_queue_high_water_mark: int
    data_queue_pending: int
    discontinuities_accepted: int
    effective_session_outcome: Any
    failed_between_runtime_and_recorder: int
    frames_accepted: int
    lost_between_control_acceptance_and_spool: int
    lost_between_recorder_and_spool: int
    payload_bytes_copied: int
    primary_fault: Any
    queue_storage_release_deferred: bool
    queue_storage_released: bool
    recorder_accepted: int
    recoverable: bool
    recovery_required: bool
    rejected_after_close_control: int
    rejected_after_close_data: int
    rejected_before_runtime_acceptance: int
    requested_terminal_intent: Any
    runtime_accepted: int
    signal_blocks_recorded: int
    signal_gaps_recorded: int
    spool_backend_cancellable: bool
    spool_committed: int
    spool_committed_extent: int
    spool_committed_transactions: int
    spool_durability_policy: str
    spool_durable_extent: int
    spool_ended_cleanly: bool
    spool_holds_committed_record: bool
    terminal_intent_latched: bool
    worker_running: bool

    __slots__ = (
        "_attempts",
        "_finalization_status",
        "_native",
        "_release_error",
        "_report",
        "_staged",
        "_state",
    )

    def __init__(
        self,
        native: Any,
        *,
        state: RecorderState,
        report: FinalizationReport | None,
        attempts: tuple[FinalizationAttempt, ...],
        finalization_status: str,
        staged: StagedConversion | None = None,
        resource_release_error: str | None = None,
    ) -> None:
        object.__setattr__(self, "_native", native)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_report", report)
        object.__setattr__(self, "_attempts", attempts)
        object.__setattr__(self, "_finalization_status", finalization_status)
        object.__setattr__(self, "_staged", staged)
        object.__setattr__(self, "_release_error", resource_release_error)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("RecorderStatus is immutable")

    @property
    def state(self) -> RecorderState:
        return self._state

    @property
    def finalization_status(self) -> str:
        """How the attempts to convert the spool into a sealed session went.

        One of ``not_started``, ``running``, ``failed_retryable``, ``failed``,
        ``succeeded``, ``abandoned`` -- the second dimension of contract
        section 3.2, independent of the capture outcome.

        ``failed_retryable`` and ``failed`` are different answers to the same
        question a caller has to act on, and they are the finalizer's answers
        rather than this layer's: the progress document recovery reads carries
        exactly this distinction, and only ``failed_retryable`` is resumable.
        ``failed`` asks for diagnosis, repair, or quarantine of the retained
        spool. It is not ``abandoned``: nobody chose to stop.
        """
        return self._finalization_status

    @property
    def finalization_attempts(self) -> tuple[FinalizationAttempt, ...]:
        return self._attempts

    @property
    def session_created(self) -> bool:
        return self._report is not None or bool(self._native.session_created)

    @property
    def sealed(self) -> bool:
        return self._report is not None

    @property
    def finalization_required(self) -> bool:
        return self._report is None and bool(self._native.finalization_required)

    @property
    def resource_release_error(self) -> str | None:
        """Why releasing the recorder's own resources failed, if it did.

        The second of the two facts a finalization produces, and deliberately
        not folded into the first: publishing a session and releasing the
        object that recorded it succeed and fail independently, and a cleanup
        failure after a session is sealed says nothing about the session. When
        this is set the recorder is ``failed`` -- it could not complete its own
        lifecycle -- while ``finalization_status`` still reports what actually
        happened to the recording.
        """
        return self._release_error

    @property
    def cleanup_paths(self) -> tuple[Path, ...]:
        """Residual recovery/work files after successful publication."""
        return () if self._report is None else self._report.cleanup_paths

    @property
    def nrf_committed(self) -> int:
        """Data-plane items committed to NRF.

        Stage 4 is *committed to NRF*, which is not the same as published: a
        conversion that sealed its session and then failed to move it into
        place has committed those rows, and reporting zero for them would
        under-report a commit that happened. The evidence is the finalization's
        own progress document, and it is used only while the staged session it
        describes is still on disk.
        """
        if self._report is not None:
            return self._report.counts.data_items
        return 0 if self._staged is None else self._staged.counts.data_items

    @property
    def control_nrf_committed(self) -> int:
        if self._report is not None:
            return self._report.counts.control_records
        return 0 if self._staged is None else self._staged.counts.control_records

    @property
    def awaiting_finalization(self) -> int:
        """Data-plane items committed to the spool and not yet in NRF.

        These are the items a finalization still has to convert. They are
        pending, not lost, and they exist as bytes in a retained spool -- which
        is exactly why they are counted here rather than left to a caller to
        infer from a difference that would otherwise look like loss.
        """
        if self._report is not None:
            return 0
        return max(0, int(self._native.spool_committed) - self.nrf_committed)

    @property
    def control_awaiting_finalization(self) -> int:
        if self._report is not None:
            return 0
        return max(0, int(self._native.control_spool_committed) - self.control_nrf_committed)

    @property
    def lost_during_finalization(self) -> int:
        """Items the spool committed that finalization did not carry into NRF.

        **Loss is a finalization outcome, not the absence of one.** Until a
        finalization has produced a sealed session there is nothing here to
        report: a recorder that stopped cleanly with a hundred committed items
        and has not been finalized yet has lost none of them, and a retryable
        finalization failure explicitly leaves the capture outcome unchanged
        (contract section 3.2) -- its items are still in the spool, waiting for
        the retry. Both of those cases read zero and count in
        :attr:`awaiting_finalization` instead.
        """
        if self._report is None:
            return 0
        return max(0, int(self._native.spool_committed) - self.nrf_committed)

    @property
    def control_lost_during_finalization(self) -> int:
        if self._report is None:
            return 0
        return max(0, int(self._native.control_spool_committed) - self.control_nrf_committed)

    @property
    def termination_kind(self) -> str | None:
        return None if self._report is None else self._report.termination_kind

    @property
    def completeness_verdict(self) -> str | None:
        if self._report is None:
            return None
        from neurale.io.nrf import NrfReader

        with NrfReader.open(self._report.session_path) as reader:
            return reader.completeness_verdict

    @property
    def complete(self) -> bool | None:
        if self._report is None:
            return None
        from neurale.io.nrf import NrfReader

        with NrfReader.open(self._report.session_path) as reader:
            return reader.complete

    @property
    def accounting_verified(self) -> bool:
        if self._report is None:
            return False
        from neurale.io.nrf import NrfReader

        with NrfReader.open(self._report.session_path) as reader:
            return reader.accounting_verified

    @property
    def legacy_termination_normal(self) -> bool:
        if self._report is None:
            return False
        from neurale.io.nrf import NrfReader

        with NrfReader.open(self._report.session_path) as reader:
            return reader.legacy_termination_normal

    @property
    def effective_session_outcome(self) -> str | None:
        if self._report is not None:
            return self._report.termination_kind
        return _enum_text(self._native.effective_session_outcome)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)


class SessionRecorder:
    """Record one native stream session and finalize it as canonical NRF.

    ``SessionRecorder`` owns orchestration only. Frames and discontinuities
    reach the recorder through a native critical observer edge; there is no
    Python callback, queue, writer thread, or direct live-to-NRF path.
    """

    __slots__ = (
        "_attempts",
        "_automatic_limits",
        "_condition",
        "_finalization_owner",
        "_finalization_status",
        "_finalization_wait_seconds",
        "_impl",
        "_output",
        "_plan",
        "_release_error",
        "_released",
        "_report",
        "_spool_is_ours",
        "_spool_path",
        "_staged",
        "_state_override",
    )

    def __init__(
        self,
        plan: RecordingPlan,
        output: str | Path,
        *,
        options: NativeRecorderOptions | None = None,
    ) -> None:
        options = _stable_native_options(plan, output) if options is None else options
        self._impl = NativeSessionRecorder.from_plan(plan, options=options)
        self._automatic_limits = False
        self._plan = plan
        self._output = Path(output)
        self._spool_path = Path(options.spool_path) if options.spool_path is not None else None
        self._spool_is_ours = False
        self._released = False
        self._report: FinalizationReport | None = None
        self._staged: StagedConversion | None = None
        self._release_error: str | None = None
        self._state_override: RecorderState | None = None
        self._finalization_status = "not_started"
        self._attempts: list[FinalizationAttempt] = []
        self._condition = threading.Condition(threading.RLock())
        self._finalization_owner: int | None = None
        self._finalization_wait_seconds = options.drain_timeout_nanos / 1_000_000_000

    @classmethod
    def create(cls, config: RecorderConfig, source: Any) -> SessionRecorder:
        """Compile *config* against a prepared device or schema."""
        plan = compile_recording_plan(config, source)
        if config.limits is None:
            from ._budget import budget_source

            plan = budget_source(plan)
        native_session_id = int(getattr(source, "_native_session_id", 1))
        recorder = cls(
            plan,
            config.path,
            options=replace(
                _stable_native_options(plan, config.path, native_session_id=native_session_id),
                spool_retention=config.spool_retention,
            ),
        )
        recorder._automatic_limits = config.limits is None
        return recorder

    def _budget_experiment(self, *, duration_seconds: float, control_records: int) -> None:
        """Resolve omitted limits before allocating or attaching any resources."""
        from ._budget import budget_recording

        with self._condition:
            if self.state is not RecorderState.CREATED:
                raise RecorderStateError("recording budgets must be resolved before prepare")
            if not self._automatic_limits:
                return
            plan = budget_recording(
                self._plan,
                duration_seconds=duration_seconds,
                control_records=control_records,
            )
            options = replace(
                self._impl.options, edge_capacity=plan.resource_bounds.frame_queue_capacity
            )
            implementation = NativeSessionRecorder.from_plan(plan, options=options)
            self._impl.close()
            self._impl = implementation
            self._plan = plan

    @property
    def plan(self) -> RecordingPlan:
        return self._plan

    @property
    def session_id(self) -> str:
        """Recorder-generated NRF session identity."""
        return self._plan.session.session_id

    @property
    def spool_path(self) -> Path:
        """Retained native spool used for offline diagnosis and recovery.

        The path is fixed when the recorder is created; the file behind it is
        created by :meth:`prepare`, because a recorder in ``created`` holds no
        resources and has nothing on disk (contract section 3.1).
        """
        if self._spool_path is None:  # private memory-backed test construction only
            raise RecorderStateError("this private recorder backend has no retained spool path")
        return self._spool_path

    def _finalization_source(self) -> Path | bytes:
        return self._impl.spool_snapshot() if self._spool_path is None else self._spool_path

    def _discard_spool_artifacts(self) -> OSError | None:
        """Remove a spool that holds nothing worth keeping, and its sidecar.

        Section 3.1 makes this a rule rather than a courtesy: a recorder that
        closes from ``created``, ``prepared`` or ``ready``, and one whose
        readiness gate refused it, discard a spool holding no committed record.
        The sidecar goes with it rather than being orphaned beside the spool.

        Only a spool this recorder created is removed. The store is opened with
        ``O_EXCL``, so a path that already existed belongs to something else --
        in the one case that can produce it, a crashed run of this very session
        whose spool is the only reconstruction input there is. Deleting that
        would destroy the evidence section 7 requires be retained.

        Ownership ends with the attempt, not with its success: the claim is
        dropped even when the deletion failed. Carrying it forward would delete
        whatever appears at that path next, and retaining it could only
        authorize a retry no caller here is entitled to. A cleanup the
        filesystem refused is recorded as a release failure, what is on disk
        becomes an orphan for explicit operator cleanup, and the next creator's
        ``O_EXCL`` refuses that path rather than adopting it.

        The error is returned as well as recorded because callers differ: a
        terminal close already derives its state from the release diagnostic,
        while a rollback does not -- see :meth:`_fail_on_unfinished_cleanup`.

        The two files go as one bundle and in one order; the reasons belong with
        the pathname that claims them both, in
        :func:`~neurale.recording._finalizer.discard_spool_bundle`.
        """
        if self._spool_path is None or not self._spool_is_ours:
            return None
        error = discard_spool_bundle(self._spool_path)
        self._spool_is_ours = False
        if error is not None:
            self._record_release_failure(
                f"{type(error).__name__}: the spool bundle at {self._spool_path} "
                f"could not be removed and was left on disk: {error}"
            )
        return error

    def _fail_on_unfinished_cleanup(self, error: OSError | None) -> None:
        """Turn a rollback whose cleanup failed into the terminal state it is.

        A failed ``prepare()`` returns the recorder to ``created``, which claims
        three things at once: no partial resource is retained, no session
        artifact is on disk, and the object MAY be prepared again. An artifact
        the cleanup could not remove sits at the exact pathname the next
        ``prepare()`` has to win with ``O_EXCL``, so reporting ``created`` would
        promise a retry guaranteed to fail on ``EEXIST``.

        The rollback that could not finish is therefore terminal ``failed``,
        with :attr:`RecorderStatus.resource_release_error` naming what was left.
        The original ``prepare()`` failure still propagates.

        The native recorder is released here rather than left as the fresh
        ``created`` core a rollback hands back. ``failed`` is where ``stop()``
        and ``abort()`` become idempotent no-ops, but those descend to the core,
        which refuses them as a state error while it believes it is ``created``.
        A public terminal state whose private half disagrees is how a
        contract-legal call turns into an exception.

        Releasing cannot take a second run at the orphan: ownership ended with
        the attempt that failed, so the retention rule inside the release finds
        nothing it is allowed to remove.
        """
        if error is None:
            return
        self._state_override = RecorderState.FAILED
        self._release()

    def _record_release_failure(self, message: str) -> None:
        """Latch a release diagnostic without overwriting an earlier one.

        Releasing the native recorder and discarding the artifacts it owns are
        two different failures that can both happen while closing one recorder,
        and the second is not a correction of the first. An operator who is
        told only about the last one has to guess what else went wrong, so both
        are kept.
        """
        self._release_error = (
            message if self._release_error is None else f"{self._release_error}; {message}"
        )

    def _holds_committed_record(self) -> bool:
        try:
            return bool(self._impl.status.spool_holds_committed_record)
        except Exception:  # pragma: no cover - a status that cannot be read keeps the spool
            return True

    def _release(self, *, keep_spool: bool = False) -> None:
        """Release the native recorder, then apply the spool retention rule.

        Releasing can fail on its own -- unmapping, unlocking, trimming and
        closing the mapped spool are four platform calls, any of which can
        return an error -- and that failure is recorded rather than raised,
        because every caller of this has already decided something the failure
        does not change. What it does change is the recorder's own lifecycle:
        an object that could not release its resources is ``failed``, and
        :attr:`RecorderStatus.resource_release_error` says why.

        Successful cleanup runs at most once. Pending I/O permits close retries.
        The native close is idempotent, but the retention
        rule is not something to re-decide, and the flag is set first so nothing
        reached from here can recurse back into it.

        One terminal close decides one cleanup, and a cleanup the filesystem
        refused is a terminal cleanup failure rather than work left for the next
        call: a second ``close()`` that unlinked what the first could not would
        be a repeated call deciding something new. What could not be removed is
        reported through :attr:`RecorderStatus.resource_release_error` and left
        for explicit cleanup.

        ``keep_spool`` overrides the retention rule for the one caller that
        knows more than it does: a finalization that read the spool and refused
        it non-retryably leaves evidence to diagnose, whatever the spool's own
        committed-record flag says.
        """
        if self._released:
            return
        self._released = True
        try:
            self._impl.close()
        except Exception as error:
            self._record_release_failure(f"{type(error).__name__}: {error}")
            if self._impl.status.worker_running:
                self._released = False
            # Failed close grants no permission to scan or delete the store.
            return
        if not (keep_spool or self._holds_committed_record()):
            self._discard_spool_artifacts()

    def _synchronize_terminal_state(self) -> None:
        """Release a failed recorder only after producers and disk I/O quiesce.

        Status polling must not turn a retained disk request into a blocking
        close. Committed prefixes stay available for offline recovery once the
        writer has finished and its handle has been closed.
        """
        if self._released or self._state_override is not None:
            return
        if _enum_text(self._impl.state) != "failed":
            return
        if self._impl.runtime_is_live() or self._impl.status.worker_running:
            return
        self._release()
        self._state_override = RecorderState.FAILED

    def _staged_conversion_evidence(self) -> StagedConversion | None:
        """What an interrupted finalization is on record as having converted.

        Read from the finalization's own progress document, and only while the
        staged session it describes is still on disk: the counts are a claim
        about rows, and a claim about rows nobody can point at is not evidence.
        An abandoned finalization reports none -- abandoning removes the staged
        session, and no NRF session will ever carry those rows.
        """
        from ._recovery import diagnose_finalization

        try:
            state = diagnose_finalization(self._output)
        except (FinalizationError, OSError, ValueError):
            return None
        if state.progress is None or state.progress.status == "abandoned":
            return None
        return state.progress.staged if state.staged_session_present else None

    @property
    def state(self) -> RecorderState:
        with self._condition:
            if self._state_override is not None:
                return self._state_override
            self._synchronize_terminal_state()
            if self._state_override is not None:
                return self._state_override
            return RecorderState(_enum_text(self._impl.state))

    @property
    def status(self) -> RecorderStatus:
        with self._condition:
            return RecorderStatus(
                self._impl.status,
                state=self.state,
                report=self._report,
                attempts=tuple(self._attempts),
                finalization_status=self._finalization_status,
                staged=self._staged,
                resource_release_error=self._release_error,
            )

    def prepare(self) -> None:
        """Allocate every bounded resource and write the spool's plan sidecar.

        This is where the recorder first exists on disk, and it is one operation
        from outside: allocating the native resources and writing the sidecar
        that makes the resulting spool interpretable are two halves of a
        ``prepare()`` a caller sees once. So the failure is one failure, and
        section 3.1 fixes what it means -- no partial resource retained, no
        session artifact created, the recorder still ``created``, and free to be
        prepared again. Both halves roll back to exactly that, including the
        half that failed after the other had succeeded.

        Only a recorder in ``created`` may be prepared, and that is checked
        here rather than left to the native call to refuse. The difference is
        not the error: it is that a refusal is not a failed attempt, so nothing
        below may treat it as one. A second ``prepare()`` on a recorder that is
        already prepared created nothing, and the spool at the path is the one
        the first call is still holding -- rolling *that* back would destroy a
        live session's only spool because a redundant call was made.
        """
        with self._condition:
            state = self.state
            if state is not RecorderState.CREATED:
                raise RecorderStateError(
                    f"prepare() is legal only on a recorder in {RecorderState.CREATED}, "
                    f"not while it is {state}; nothing was changed"
                )
        options = self._impl.options
        if options.spool == "mapped":
            _require_lockable_spool_capacity(options.spool_capacity_bytes)
        try:
            self._impl.prepare()
        except BaseException:
            self._discard_artifacts_of_this_prepare()
            raise
        self._adopt_spool_ownership()
        if self._spool_path is None:
            return
        try:
            write_plan_sidecar(self._spool_path, self._plan)
        except BaseException:
            self._roll_back_prepare()
            raise

    def _prepare_experiment_attachment(self) -> Any:
        """Prepare storage for a native ExperimentSession-managed lifecycle."""

        with self._condition:
            if self.state is not RecorderState.CREATED:
                raise RecorderStateError(
                    "experiment recording requires a newly created SessionRecorder"
                )
        options = self._impl.options
        if options.spool == "mapped":
            _require_lockable_spool_capacity(options.spool_capacity_bytes)
        try:
            attachment = self._impl._prepare_experiment_attachment()
        except BaseException:
            self._discard_artifacts_of_this_prepare()
            raise
        self._adopt_spool_ownership()
        if self._spool_path is not None:
            try:
                write_plan_sidecar(self._spool_path, self._plan)
            except BaseException:
                self._impl._rollback_experiment_attachment()
                self._discard_artifacts_of_this_prepare()
                raise
        return attachment

    def _attach_experiment(self, runner: Any) -> None:
        self._impl._attach_experiment(runner)

    def _rollback_experiment_attachment(self) -> None:
        self._impl._rollback_experiment_attachment()
        self._fail_on_unfinished_cleanup(self._discard_spool_artifacts())

    @property
    def _experiment_max_control_body_bytes(self) -> int:
        return int(self._impl.options.max_control_payload_bytes)

    def _adopt_spool_ownership(self) -> None:
        """Take the ownership answer from the create that decided it.

        Not from a prior ``exists()`` check. The store is opened with ``O_EXCL``
        and exactly one caller can win a path; between observing that a path is
        free and creating it there is a window, and the file that can appear
        inside it is a crashed run's spool -- the one artifact contract section 7
        requires be retained. A recorder that inferred ownership from the check
        would unlink that file while cleaning up after the very ``prepare()``
        the file caused to fail.
        """
        self._spool_is_ours = bool(self._impl.spool_created_by_this_recorder)

    def _discard_artifacts_of_this_prepare(self) -> None:
        """Remove what *this* attempt created, and nothing else.

        The narrower of the two ownership questions, because it is the one this
        situation asks. "This recorder created a spool at some point" is not a
        licence to unlink whatever is at the path now: the attempt that just
        raised may have lost the exclusive create to another process, and on a
        retry after a rolled-back ``prepare()`` the previous ``true`` would
        otherwise still be sitting there, aimed at a file this recorder has
        never touched.
        """
        if self._spool_path is None or not self._impl.spool_created_by_last_prepare:
            return
        self._spool_is_ours = True
        self._fail_on_unfinished_cleanup(self._discard_spool_artifacts())

    def _roll_back_prepare(self) -> None:
        """Undo a native prepare whose sidecar could not be written.

        The recorder goes back to ``created`` with nothing on disk, which is
        what section 3.1 says a failed ``prepare()`` leaves -- and it may be
        prepared again, which is the part a terminal ``failed`` would take away.

        A rollback that itself fails is a different thing and is reported as
        one: the recorder could not complete its own lifecycle, so it is
        ``failed``, everything possible is released, and
        :attr:`RecorderStatus.resource_release_error` says what went wrong.
        That covers the native undo *and* the artifact cleanup that follows it
        -- a spool this rollback could not unlink is left at the pathname the
        retry would have to win, so see :meth:`_fail_on_unfinished_cleanup`.
        """
        try:
            self._impl.rollback_prepare()
        except Exception as error:
            self._record_release_failure(f"{type(error).__name__}: {error}")
            self._release()
            self._discard_spool_artifacts()
            self._state_override = RecorderState.FAILED
            return
        self._fail_on_unfinished_cleanup(self._discard_spool_artifacts())

    def attach(
        self, runner: Any, *, edge_id: int | None = None, capacity: int | None = None
    ) -> None:
        self._impl.attach(runner, edge_id=edge_id, capacity=capacity)

    def stop(self, reason: str = "stop") -> RecorderStatus:
        self._impl.stop(reason)
        return self.status

    def abort(self, reason: str = "abort") -> RecorderStatus:
        self._impl.abort(reason)
        return self.status

    def _await_finalization_owner(self, what: str) -> bool:
        """Wait for whoever is mutating the finalization to finish.

        Must be called with the condition held. Returns whether the caller may
        proceed straight away. ``False`` means another operation was in flight;
        whether it has since finished is a separate question, and one only the
        caller can decide what to do about -- the ownership flag still says so
        after the wait if it timed out.

        Retrying a finalization and abandoning one both rewrite the same staged
        session and the same progress document, so they are one exclusive
        operation rather than two. Left unserialized they interleave into a
        contradiction that no amount of state checking upstream can prevent:
        abandoning removes the staged session tree, and a retry that checked
        the state before that decision was taken is a finalizer converting into
        a directory that is being deleted underneath it.
        """
        if self._finalization_owner is None:
            return True
        if self._finalization_owner == threading.get_ident():
            raise RecorderStateError(f"{what}() cannot recursively wait for itself")
        self._condition.wait_for(
            lambda: self._finalization_owner is None,
            timeout=self._finalization_wait_seconds,
        )
        return False

    def finalize(self) -> RecorderStatus:
        with self._condition:
            if self._report is not None:
                return self.status
            if not self._await_finalization_owner("finalize"):
                # Another thread's finalization or abandonment is the answer:
                # a second conversion of the same spool is not a retry of it.
                return self.status
            state = self.state
            if self._finalization_status == "failed":
                raise RecorderStateError(
                    "the finalizer refused this spool non-retryably, so finalize() will read "
                    "the same bytes to the same conclusion; this needs diagnosis, repair, or "
                    "quarantine of the retained spool rather than another attempt"
                )
            if state not in (RecorderState.STOPPED, RecorderState.FINALIZATION_FAILED):
                raise RecorderStateError(
                    "finalize() is legal only after capture stopped, "
                    f"not while the recorder is {state}"
                )
            if (
                self._impl.options.spool in ("file", "mapped")
                and _enum_text(self._impl.state) != "closed"
            ):
                # Release the native writer's handle before offline conversion
                # and Windows cleanup. A mapped test store additionally needs
                # flush/unmap for coherent reads. This is not NRF finalization.
                self._state_override = RecorderState.STOPPED
                try:
                    self._impl.close()
                except Exception as error:
                    self._record_release_failure(f"{type(error).__name__}: {error}")
                    self._state_override = RecorderState.FAILED
                    raise RecorderStateError(
                        "could not close the spool before finalization"
                    ) from error
            started = time.time_ns()
            self._state_override = RecorderState.FINALIZING
            self._finalization_status = "running"
            self._finalization_owner = threading.get_ident()
        try:
            report = finalize_spool(
                self._finalization_source(),
                self._output,
                options=FinalizerOptions(
                    recording_plan=self._plan, spool_retention=self._impl.options.spool_retention
                ),
            )
        except BaseException as error:
            self._latch_finalization_failure(error, started=started)
            raise
        return self._latch_finalization_success(report, started=started)

    @property
    def finalization_progress(self) -> FinalizationProgress | None:
        """Read offline finalization progress; do not poll from a realtime thread."""
        if self._report is not None:
            return self._report.progress
        path = self._output.with_name(f"{self._output.name}.finalizing") / "progress.json"
        try:
            return FinalizationProgress.load(path)
        except FileNotFoundError:
            return None

    def _latch_finalization_failure(self, error: BaseException, *, started: int) -> None:
        """Latch a failed attempt as what it was: retryable, or not.

        ``FinalizationError.retryable`` is the finalizer's own answer to
        "would running this again over the same bytes help", and the layers
        below already act on it -- the finalizer stamps the progress document
        ``failed`` rather than ``failed_retryable``, and recovery reads only
        ``running`` and ``failed_retryable`` as resumable. A public recorder
        that reported ``failed_retryable`` regardless would tell the caller to
        try again about a fault the progress document tells recovery cannot be
        retried, which is one artifact giving two answers.

        So a non-retryable failure is terminal here too: the recorder is
        ``failed``, ``finalization_status`` is ``failed``, and
        :meth:`finalize` will not run again. It is not ``abandoned`` -- nobody
        decided to stop, the finalizer reported that continuing is pointless --
        and the spool is retained unconditionally, because a source the
        finalizer read and refused is exactly what diagnosis, repair, or
        quarantine needs (contract section 7).
        """
        ended = time.time_ns()
        retryable = not isinstance(error, FinalizationError) or error.retryable
        with self._condition:
            try:
                if isinstance(error, FinalizationError):
                    self._attempts.append(
                        FinalizationAttempt(
                            attempt=len(self._attempts) + 1,
                            outcome="failed_retryable" if retryable else "failed",
                            category=error.category,
                            detail=str(error),
                            started_at_ns=started,
                            ended_at_ns=ended,
                        )
                    )
                self._staged = self._staged_conversion_evidence()
                if retryable:
                    self._finalization_status = "failed_retryable"
                    self._state_override = RecorderState.FINALIZATION_FAILED
                else:
                    self._finalization_status = "failed"
                    self._release(keep_spool=True)
                    self._state_override = RecorderState.FAILED
            finally:
                # Ownership and the wakeup are released in a `finally` on every
                # path out of a finalization, successful or not. A waiter blocked
                # on a finalizer that died holding the ownership flag would never
                # be woken and would never see the state move -- the recorder
                # would read `finalizing` forever with nothing running.
                self._finalization_owner = None
                self._condition.notify_all()

    def _latch_finalization_success(
        self, report: FinalizationReport, *, started: int
    ) -> RecorderStatus:
        """Record a published session, then release the recorder.

        The order is the point. Publication is irreversible -- the session is
        on disk and sealed -- so the finalization outcome is latched *before*
        anything that can still fail runs. Releasing the recorder's own
        resources is one of those things, and when it fails the recorder is
        ``failed`` while the finalization stays ``succeeded``: two facts, two
        fields, neither one able to overwrite the other.
        """
        ended = time.time_ns()
        with self._condition:
            try:
                self._attempts.append(
                    FinalizationAttempt(
                        attempt=len(self._attempts) + 1,
                        outcome="succeeded",
                        category=None,
                        detail=None,
                        started_at_ns=started,
                        ended_at_ns=ended,
                    )
                )
                self._report = report
                self._finalization_status = "succeeded"
                self._staged = None
                self._release()
                self._state_override = (
                    RecorderState.CLOSED if self._release_error is None else RecorderState.FAILED
                )
            finally:
                self._finalization_owner = None
                self._condition.notify_all()
            return self.status

    def abandon_finalization(self, reason: str) -> RecorderStatus:
        """Stop retrying, terminally, and record that decision.

        Abandoning is written to disk before anything here changes, and the
        public outcome is latched before the recorder is released -- for the
        same reason :meth:`finalize` does: once the progress document says
        ``abandoned`` there is no going back to ``failed_retryable``, and a
        failure to release resources afterwards must not leave the object
        claiming a retry is still possible.

        It takes the same exclusive ownership :meth:`finalize` does, and holds
        it across the offline call. Both operations rewrite the staged session
        and the progress document, so checking the state and then acting on it
        without holding anything is a check about a past that a concurrent
        retry has already left: this removes the staged session tree, and the
        finalizer it would be removed from underneath is converting into it.
        Two concurrent abandonments are the same problem in a smaller form --
        the second would overwrite the first's recorded reason.
        """
        with self._condition:
            if not self._await_finalization_owner("abandon_finalization"):
                # Waiting is not enough on its own: unlike a finalization, an
                # abandonment in flight publishes no state a second caller
                # could see, so proceeding after the wait expired would run two
                # of them over one staged session -- which is the interleaving
                # this is here to prevent.
                if self._finalization_owner is not None:
                    raise RecorderStateError(
                        "abandon_finalization() cannot proceed while a finalization operation "
                        "is already running over the same staged session"
                    )
            state = self.state
            if state is not RecorderState.FINALIZATION_FAILED:
                raise RecorderStateError(
                    "abandon_finalization() requires a retryable finalization failure, "
                    f"and the recorder is {state}"
                )
            self._finalization_owner = threading.get_ident()
        from ._recovery import abandon_finalization

        try:
            abandon_finalization(self._output, reason=reason)
        except BaseException:
            # Nothing was decided, so nothing is latched -- but the ownership
            # is this call's and has to be handed back, or every later
            # finalization operation waits on a decision that never comes.
            with self._condition:
                self._finalization_owner = None
                self._condition.notify_all()
            raise
        with self._condition:
            try:
                self._finalization_status = "abandoned"
                self._state_override = RecorderState.FAILED
                self._staged = None
                self._release()
            finally:
                self._finalization_owner = None
                self._condition.notify_all()
            return self.status

    def close(self) -> RecorderStatus:
        with self._condition:
            # Waits on the finalization operation rather than on the
            # `finalizing` state: abandoning mutates the same staged session
            # and the same progress document without ever passing through that
            # state, and closing on top of it would release the recorder while
            # a terminal decision about it is still being committed.
            if not self._await_finalization_owner("close"):
                return self.status
            state = self.state
        if state in (RecorderState.CLOSED, RecorderState.FAILED):
            self._release()
            return self.status
        if state is RecorderState.FINALIZATION_FAILED:
            return self.status
        if state in (RecorderState.RECORDING, RecorderState.DRAINING):
            self.stop("close")
            state = self.state
        if state is RecorderState.STOPPED:
            return self.finalize()
        # `created`, `prepared`, `ready`: nothing was finalized here, so the
        # spool is kept only if it holds a committed record -- and it does not
        # on any of these paths unless the readiness gate already committed the
        # superblock over real data. `_release()` applies that rule.
        self._release()
        self._state_override = (
            RecorderState.CLOSED if self._release_error is None else RecorderState.FAILED
        )
        return self.status

    def runtime_is_live(self) -> bool:
        return self._impl.runtime_is_live()

    def record_event(
        self,
        name: str,
        *,
        time_ns: int | None = None,
        value: float | None = None,
        text: str | None = None,
    ) -> bool:
        return self._impl.record_event(name, time_ns=time_ns, value=value, text=text)

    def record_state(
        self,
        state: str,
        *,
        time_ns: int | None = None,
        value: float | None = None,
        text: str | None = None,
    ) -> bool:
        return self._impl.record_state(state, time_ns=time_ns, value=value, text=text)

    def record_command(
        self,
        command: str,
        *,
        time_ns: int | None = None,
        value: float | None = None,
        text: str | None = None,
    ) -> bool:
        return self._impl.record_command(command, time_ns=time_ns, value=value, text=text)

    def record_task_variable(
        self,
        name: str,
        *,
        time_ns: int | None = None,
        value: float | None = None,
        text: str | None = None,
    ) -> bool:
        return self._impl.record_task_variable(name, time_ns=time_ns, value=value, text=text)

    def record_label(
        self,
        label: str,
        *,
        time_ns: int | None = None,
        value: float | None = None,
        text: str | None = None,
    ) -> bool:
        return self._impl.record_label(label, time_ns=time_ns, value=value, text=text)

    def record_target(
        self,
        name: str,
        *,
        time_ns: int | None = None,
        value: float | None = None,
        text: str | None = None,
    ) -> bool:
        return self._impl.record_target(name, time_ns=time_ns, value=value, text=text)

    def record_assistance(
        self,
        name: str,
        *,
        time_ns: int | None = None,
        value: float | None = None,
        text: str | None = None,
    ) -> bool:
        return self._impl.record_assistance(name, time_ns=time_ns, value=value, text=text)

    def record_trial(
        self,
        *,
        start_ns: int,
        stop_ns: int,
        label: str | None = None,
        outcome: str | None = None,
    ) -> bool:
        return self._impl.record_trial(
            start_ns=start_ns, stop_ns=stop_ns, label=label, outcome=outcome
        )

    def record_fault(
        self,
        code: str,
        *,
        time_ns: int | None = None,
        stage: str = "",
        frame_sequence: int | None = None,
        signal_id: int | None = None,
        text: str | None = None,
    ) -> bool:
        return self._impl.record_fault(
            code,
            time_ns=time_ns,
            stage=stage,
            frame_sequence=frame_sequence,
            signal_id=signal_id,
            text=text,
        )

    def checkpoint(self) -> None:
        self._impl.checkpoint()

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        """Leave the block, and on the error path leave the caller's error alone.

        On a clean exit the recorder stops and finalizes, and a finalization
        failure propagates -- it is the result of the block, and swallowing it
        would report a recording that is not in NRF form as one that is.

        On the error path the caller's exception is the diagnosis and every
        cleanup failure is a footnote to it, so all of them are attached as
        notes rather than raised. That applies to both phases: aborting capture
        *and* the finalization ``close()`` performs. A ``FinalizationError``
        raised out of here would replace the application error that caused the
        block to unwind with a description of what the recorder was doing about
        it -- and the finalization failure is recoverable from the retained
        spool, while an exception that has been thrown away is not.
        """
        if exc_type is None:
            self.stop("context-exit")
            self.close()
            return False
        try:
            self.abort(f"context-exit: {exc_type.__name__}")
        except RecorderRuntimeShutdownError as shutdown_error:
            # The runtime is still live, so closing would be refused for the
            # same reason: there is nothing further to attempt here.
            _attach_note(exc, f"the native recorder was left open: {shutdown_error}")
            return False
        except Exception as abort_error:
            _attach_note(exc, f"aborting the recorder failed: {abort_error!r}")
            return False
        try:
            self.close()
        except Exception as close_error:
            _attach_note(exc, f"finalizing the recording failed: {close_error!r}")
        return False
