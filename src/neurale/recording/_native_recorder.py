#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Private control facade over the native critical recorder.

:class:`~neurale.recording.SessionRecorder` is the sole stable public entry
point and delegates capture here with a fixed internal spool choice. The extra
options on this boundary stay private so backend and fault fixtures can drive
the implementation without publishing a second recorder API or a backend
selector. There is no fallback path.

The status returned here is the native capture status, including every
acceptance-ladder stage per plane; the public facade forwards those counters and
adds the offline finalization state and result.

Who drives what
---------------

The recorder does not run itself. Once attached, the **runtime** drives the
lifecycle transitions, and this facade drives the runtime rather than reaching
past it::

    recorder.prepare()          created   -> prepared
    recorder.attach(runner)     registers the critical observer edge
    runner.prepare()
    runner.arm()                prepared  -> ready      (readiness gate)
    runner.start()              ready     -> recording  (start_observing)

    recorder.stop()             runner.stop(); runner.join(); then drain
    recorder.abort()            runner.abort(); runner.join(); then drain
    recorder.close()            stopped   -> closed

There is deliberately no public ``start()``: only the runtime knows that every
critical edge is attached and lossless, so a recorder started behind its back
could never have been gated.

:meth:`stop <NativeSessionRecorder.stop>` and :meth:`abort
<NativeSessionRecorder.abort>` quiesce the runtime first. Telling the recorder
to stop accepting while the runtime is still producing does not end the session
-- it turns the next frame into a critical-edge failure, faulting a session the
caller asked to end cleanly. :meth:`close <NativeSessionRecorder.close>` refuses
outright while the runtime is live: the same hazard with none of the intent.

Quiescing is a gate, not a courtesy call. ``stop()``, ``abort()`` and ``join()``
may each answer ``deadline_exceeded``, meaning the bound elapsed and the workers
may still be there, so a runtime that does not reach a terminal state raises
:class:`~neurale.recording.RecorderRuntimeShutdownError` and **no recorder call
is made at all**. Proceeding would charge a runtime shutdown timeout to the
recording as a critical-edge fault.

Canonical NRF conversion is not implemented here: the native core stops at a
committed spool, and ``finalize`` / ``abandon_finalization`` are owned by the
public facade's offline finalizer and recovery API. A private status after
``close`` therefore reports ``finalization_required = True``, ``sealed =
False``, ``nrf_committed = 0``, and -- completeness being a property of a
*sealed* session -- ``completeness_verdict`` and ``complete`` of ``None`` rather
than ``False``.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from neurale.io.nrf._canonical import canonical_json_bytes
from neurale.io.nrf._plan_document import _DEFAULT_SPOOL_CAPACITY_BYTES

from ._errors import (
    RecorderConfigError,
    RecorderError,
    RecorderRuntimeShutdownError,
    RecorderStateError,
)
from ._plan import RecordingPlan, compile_recording_plan
from ._spec import RecorderConfig

#: Clock domain the control plane's ``time_ns`` belongs to when the caller does
#: not name one.
#:
#: It is the **host monotonic** clock -- :func:`time.monotonic_ns` -- and not the
#: wall clock, because that is the clock the NRF session registers for control
#: records: ``session.clock`` is declared ``clock_type="host_monotonic"`` by
#: :mod:`neurale.recording._registry`, and the stable
#: :class:`~neurale.recording.SessionRecorder` timestamps every control row with
#: :func:`time.monotonic_ns`. Writing Unix wall-clock nanoseconds into a field a
#: reader will interpret as host-monotonic is a timestamp-domain error, not a
#: naming difference, so the two paths use one clock.
#:
#: It must not collide with a signal's clock domain, and
#: :class:`NativeRecorderOptions` refuses a plan where it does.
HOST_MONOTONIC_CLOCK_DOMAIN = 0

#: Private stores: the stable facade uses continuous file recording. Memory and
#: mapped stores remain explicit native test paths, never silent fallbacks.
SPOOL_BACKENDS = frozenset({"memory", "file", "mapped"})

#: Durability policies, spelled as the native enum spells them.
DURABILITY_POLICIES = frozenset({"buffered", "checkpoint_sync", "transaction_sync"})

#: What happens to the spool once a finalization has been validated.
#:
#: ``"retain"`` is the default and is the conservative one: the spool is the
#: only copy of a session until an NRF artifact has been produced *and*
#: validated, so nothing here ever deletes it.
SPOOL_RETENTION_POLICIES = frozenset({"retain", "delete_after_validated_finalization"})

#: Control-plane record kinds, and the producer-identity registry name each one
#: is recorded under. The data-plane kinds (``frame``, ``discontinuity``) are not
#: submittable here and the native core refuses them.
CONTROL_KINDS: tuple[str, ...] = (
    "events",
    "trials",
    "experiment_states",
    "commands",
    "targets",
    "labels",
    "assistance",
    "faults",
    "task_variables",
)


def _epoch_nanos(created_at: str) -> int:
    """Return NRF's ``created_at`` spelling as Unix nanoseconds.

    The spool superblock stores an integer, and the plan stores the NRF text.
    Converting here rather than storing both keeps one source of truth.

    The arithmetic is integer throughout. Going via a float seconds value would
    lose the low nanosecond digits of a timestamp the superblock stores exactly,
    and two components disagreeing about a session's creation time in the last
    three digits is the kind of difference that only surfaces once something
    compares them.
    """
    moment: datetime | None = None
    # Both spellings occur in this repository: `utc_now()` writes the fractional
    # form, and hand-written session identities routinely leave the fraction off.
    # Accepting only one of them would reject a `created_at` the rest of the
    # stack is perfectly happy with.
    for spelling in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            moment = datetime.strptime(created_at, spelling).replace(tzinfo=UTC)
            break
        except (ValueError, TypeError):
            continue
    if moment is None:
        raise RecorderConfigError(
            f"created_at {created_at!r} is not NRF's UTC spelling (YYYY-MM-DDTHH:MM:SS[.ffffff]Z)"
        )
    delta = moment - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def _session_uuid_bytes(session_id: str) -> bytes:
    """Return the 16-byte session UUID the spool superblock stores.

    A session id that is not a UUID is refused rather than hashed into one. The
    superblock's UUID field is an identity other components match on, and
    inventing a value here would produce a spool claiming an identity nothing
    else in the system agrees with. This is a native spool constraint rather
    than an NRF session-id constraint.
    """
    try:
        return uuid.UUID(session_id).bytes
    except (ValueError, AttributeError, TypeError) as error:
        raise RecorderConfigError(
            f"the native recorder needs a UUID session_id for the spool superblock, "
            f"got {session_id!r}"
        ) from error


@dataclass(frozen=True, slots=True)
class NativeRecorderOptions:
    """Private native-core configuration and validation controls.

    The stable facade fixes these choices. Tests use this object to exercise
    backend capabilities and injected limits without publishing them as public
    configuration. Everything a *plan* decides remains taken from the plan.
    """

    #: Private backend. The stable facade uses "file": a native worker appends
    #: continuously without reserving disk extent in RAM. Ordinary file I/O is
    #: not bounded-cancellable; timeout retains resources until I/O completes.
    #: "memory" and "mapped" remain fixed-capacity test backends; neither is a
    #: fallback. Mapped storage still requires whole-capacity page locking.
    spool: str = "memory"
    #: Bytes reserved by memory/mapped stores; a disk ceiling for file stores.
    #: File capacity is never preallocated. Appending past it faults recording.
    spool_capacity_bytes: int = _DEFAULT_SPOOL_CAPACITY_BYTES
    #: Where the ``"file"`` or ``"mapped"`` backend creates its spool. Required
    #: for those backends, ignored by ``"memory"``.
    spool_path: str | os.PathLike[str] | None = None

    #: ``SessionId`` the runtime stamps into its frame headers. A frame from
    #: another session is a plan violation, not data to record, so this has to
    #: match the source and there is no safe default to guess.
    native_session_id: int = 1

    #: The critical observer edge. Its drop policy is not configurable: a
    #: critical edge is lossless-until-fault by definition (contract section 2).
    edge_id: int = 1
    edge_capacity: int = 256
    edge_drop_history_capacity: int = 8

    #: Bounds the plan does not carry. ``None`` takes the value the compiled
    #: plan implies -- one block per declared signal, and the sum of their
    #: maximum block sizes.
    max_blocks_per_frame: int | None = None
    max_frame_payload_bytes: int | None = None
    max_signal_gaps_per_discontinuity: int = 8
    #: Ceiling on one control record's whole encoded body. Owned by the native
    #: core, which treats a body over it as a plan violation and faults the
    #: session -- an oversized record is not something a lossless-until-fault
    #: recorder may drop and carry on from.
    max_control_payload_bytes: int = 4096
    #: Ceiling on any **single string** inside a control record, in UTF-8 bytes.
    #:
    #: A separate contract from the payload bound, and not derivable from it:
    #: without it one caller-supplied string may fill an entire record, which is
    #: how an unbounded free-text field reaches a container that promised
    #: bounded records. It is checked in this facade, before the record is
    #: offered, and an over-long string raises rather than counting as a
    #: recording loss -- see :meth:`NativeSessionRecorder.submit_control`.
    max_control_string_bytes: int = 1024
    #: The clock domain stamped on control records. See
    #: :data:`HOST_MONOTONIC_CLOCK_DOMAIN`.
    control_clock_domain: int = HOST_MONOTONIC_CLOCK_DOMAIN

    #: Spool transaction shape.
    max_records_per_transaction: int = 256
    max_transaction_bytes: int = 1 << 20

    #: How long the worker sleeps when both queues are empty. It polls rather
    #: than waiting on a condition variable so the critical callback performs no
    #: wake syscall at all.
    worker_idle_poll_nanos: int = 200_000
    #: The bounded shutdown timeout of contract section 4.7. Exceeding it is a
    #: reportable outcome, not a longer wait.
    drain_timeout_nanos: int = 5_000_000_000

    #: Durability policy for the spool (contract section 6). ``"buffered"`` is
    #: the only policy consistent with the memory store, and selecting anything
    #: else alongside it is refused rather than quietly downgraded: a writer
    #: that synced a store which survives nothing would publish a durable extent
    #: that is not durable.
    durability_policy: str = "buffered"

    #: What becomes of the spool once a finalization has been validated. See
    #: :data:`SPOOL_RETENTION_POLICIES`.
    #:
    #: Independently of the policy, the ``"memory"`` store retains nothing past
    #: the process. Retention is observable for the accepted mapped store.
    spool_retention: str = "delete_after_validated_finalization"

    def __post_init__(self) -> None:
        if self.spool not in SPOOL_BACKENDS:
            raise RecorderConfigError(
                f"spool must be one of {sorted(SPOOL_BACKENDS)}, got {self.spool!r}"
            )
        if self.durability_policy not in DURABILITY_POLICIES:
            raise RecorderConfigError(
                f"durability_policy must be one of {sorted(DURABILITY_POLICIES)}, "
                f"got {self.durability_policy!r}"
            )
        if self.spool_retention not in SPOOL_RETENTION_POLICIES:
            raise RecorderConfigError(
                f"spool_retention must be one of {sorted(SPOOL_RETENTION_POLICIES)}, "
                f"got {self.spool_retention!r}"
            )
        if self.spool == "memory" and self.durability_policy != "buffered":
            raise RecorderConfigError(
                "the memory spool provides no crash durability, so it may only be used with "
                f"durability_policy='buffered'; got {self.durability_policy!r}. Selecting a "
                "syncing policy would publish a durable extent the store cannot honour."
            )
        if self.spool == "mapped" and self.durability_policy != "buffered":
            raise RecorderConfigError(
                "the mapped spool performs its required superblock sync before readiness but "
                "provides no runtime transaction/checkpoint sync, so it may only be used with "
                f"durability_policy='buffered'; got {self.durability_policy!r}"
            )
        if self.spool in {"file", "mapped"} and self.spool_path is None:
            raise RecorderConfigError(f"spool={self.spool!r} needs a spool_path to create")
        if self.spool == "mapped" and sys.platform not in {"linux", "win32"}:
            raise RecorderConfigError(
                "the bounded mapped spool is available only on Linux tmpfs and Windows local "
                "storage; "
                f"platform {sys.platform!r} has no approved bounded process-crash backend"
            )
        if self.spool_capacity_bytes <= 0:
            raise RecorderConfigError(
                f"spool_capacity_bytes must be positive, got {self.spool_capacity_bytes}"
            )
        if self.max_control_string_bytes <= 0:
            raise RecorderConfigError(
                f"max_control_string_bytes must be positive, got {self.max_control_string_bytes}"
            )
        if self.max_control_string_bytes > self.max_control_payload_bytes:
            raise RecorderConfigError(
                f"max_control_string_bytes ({self.max_control_string_bytes}) exceeds "
                f"max_control_payload_bytes ({self.max_control_payload_bytes}), so a string "
                "inside the bound could still overflow the record that carries it"
            )


class NativeSessionRecorder:
    """Record a live native streaming session through the native critical path.

    Create it with :meth:`create`, attach it to a runner before the runner is
    prepared, and let the runtime drive the lifecycle.
    """

    __slots__ = (
        "_attached_edge_id",
        "_control_identities",
        "_native",
        "_options",
        "_plan",
        "_recorder",
        "_runner",
    )

    def __init__(self, plan: RecordingPlan, options: NativeRecorderOptions) -> None:
        # Import here, not at module scope: `import neurale.recording` must not
        # load the native extension. Importing `neurale.recording` remains a
        # declaration-only operation; constructing a recorder is the explicit
        # native initialization boundary.
        from . import _native

        self._native = _native
        self._plan = plan
        self._options = options
        self._attached_edge_id: int | None = None
        self._runner: Any = None
        self._control_identities: dict[str, int] = dict.fromkeys(CONTROL_KINDS, 0)
        self._recorder = _native._NativeRecorder(
            plan=self._build_native_plan(plan, options),
            backend={
                "memory": _native.SpoolBackend.MEMORY,
                "file": _native.SpoolBackend.FILE,
                "mapped": _native.SpoolBackend.MAPPED,
            }[options.spool],
            memory_capacity_bytes=options.spool_capacity_bytes,
            path="" if options.spool_path is None else os.fspath(options.spool_path),
        )

    # --- construction -------------------------------------------------------

    @classmethod
    def create(
        cls,
        config: RecorderConfig,
        source: Any,
        *,
        options: NativeRecorderOptions | None = None,
    ) -> NativeSessionRecorder:
        """Compile *config* against a prepared device or schema.

        This private constructor and the stable facade share the one
        :func:`compile_recording_plan` owner.
        """
        return cls(compile_recording_plan(config, source), options or NativeRecorderOptions())

    @classmethod
    def from_plan(
        cls, plan: RecordingPlan, *, options: NativeRecorderOptions | None = None
    ) -> NativeSessionRecorder:
        """Build a recorder for an already-compiled *plan*."""
        return cls(plan, options or NativeRecorderOptions())

    def _build_native_plan(self, plan: RecordingPlan, options: NativeRecorderOptions) -> Any:
        """Mirror the compiled plan into the struct the native core prepares against.

        The canonical document travels through as opaque bytes and the native
        side never parses it. That is deliberate (see ``recording_plan.h``): a
        second implementation that read the plan would be a second place the
        plan could be interpreted differently.
        """
        native = self._native
        by_id = {signal.id: signal for signal in plan.native_schema.signals}

        # A control record's clock domain has to be distinguishable from a
        # signal's, because both end up as a `clock_domain` number that the
        # finalizer maps to one declared clock. Two kinds of time sharing a
        # number is not a collision anything downstream can detect, so it is
        # refused where it is still cheap to see.
        colliding = sorted(
            signal.clock_domain
            for signal in plan.native_schema.signals
            if signal.clock_domain == options.control_clock_domain
        )
        if colliding:
            raise RecorderConfigError(
                f"control_clock_domain {options.control_clock_domain} is already declared by a "
                f"signal in this plan; control records carry host-monotonic time and a signal "
                "carries device time, and one number cannot name both"
            )

        recorded = []
        for signal_id in plan.recorded_signal_ids:
            signal = by_id.get(signal_id)
            if signal is None:
                raise RecorderConfigError(
                    f"the plan records signal {signal_id}, which its native schema does not declare"
                )
            recorded.append(
                native.PlannedSignalRecording(
                    signal_id=signal_id,
                    max_block_bytes=signal.max_block_bytes,
                    max_block_samples=signal.max_block_samples,
                )
            )

        bounds = plan.resource_bounds
        return native.NativeRecordingPlan(
            session_id=plan.session.session_id,
            session_uuid=_session_uuid_bytes(plan.session.session_id),
            created_unix_nanos=_epoch_nanos(plan.session.created_at),
            plan_document=canonical_json_bytes(plan.document()),
            plan_fingerprint=bytes.fromhex(plan.fingerprint),
            native_session_id=options.native_session_id,
            native_schema_id=plan.native_schema_id,
            recorded_signals=recorded,
            frame_queue_capacity=bounds.frame_queue_capacity,
            control_queue_capacity=bounds.control_queue_capacity,
            max_blocks_per_frame=(
                plan.max_block_count
                if options.max_blocks_per_frame is None
                else options.max_blocks_per_frame
            ),
            max_frame_payload_bytes=(
                plan.max_frame_bytes
                if options.max_frame_payload_bytes is None
                else options.max_frame_payload_bytes
            ),
            max_signal_gaps_per_discontinuity=options.max_signal_gaps_per_discontinuity,
            max_control_payload_bytes=options.max_control_payload_bytes,
            max_records_per_transaction=options.max_records_per_transaction,
            max_transaction_bytes=options.max_transaction_bytes,
            checkpoint_interval_transactions=bounds.checkpoint_interval,
            worker_idle_poll_nanos=options.worker_idle_poll_nanos,
            drain_timeout_nanos=options.drain_timeout_nanos,
            durability_policy=getattr(native.DurabilityPolicy, options.durability_policy.upper()),
        )

    # --- observation --------------------------------------------------------

    @property
    def plan(self) -> RecordingPlan:
        """The compiled plan this recorder prepared against."""
        return self._plan

    @property
    def options(self) -> NativeRecorderOptions:
        return self._options

    @property
    def state(self) -> Any:
        """The recorder object's lifecycle position (contract section 3.1).

        This is the *object's* state, not the session's outcome. A recorder in
        a terminal state says nothing by itself about whether the session it
        produced is complete -- read :attr:`status` for that.
        """
        return self._recorder.state

    @property
    def status(self) -> Any:
        """A snapshot of the native recorder's whole status surface.

        This is the native status type, not
        :class:`~neurale.recording.RecorderStatus`: the native path has all
        five acceptance-ladder stages and reports each one separately per
        plane, which the Python path's four-stage status cannot express.

        Three of its fields are ``X | None`` and mean something the Python
        status has no room for. ``completeness_verdict`` and ``complete`` are
        ``None`` while no sealed session exists, because completeness is a
        property of a sealed session and ``False`` would report a verdict
        nobody reached; ``termination_kind`` is ``None`` because it is what the
        *artifact* says, and no artifact says anything until finalization writes one.
        """
        return self._recorder.status

    @property
    def store_provides_crash_durability(self) -> bool:
        """Whether the selected spool store survives a crash.

        ``False`` for the memory store, and the reason
        ``status.spool_durable_extent`` must not be read as a survival claim on
        its own there.
        """
        return self._recorder.store_provides_crash_durability

    @property
    def spool_created_by_this_recorder(self) -> bool:
        """Whether this recorder's own exclusive create put the spool there.

        Answered by the ``O_EXCL`` create itself rather than by looking at the
        path first, and it is the only safe basis for deciding later that the
        path may be removed: between "it does not exist" and "so it is mine"
        there is a scheduling window, and what can appear inside it is a crashed
        run's spool -- the one file contract section 7 requires be retained.

        It stays true after the store is released, because a released store
        leaves its path behind and that path is still this recorder's -- but
        not across a further :meth:`prepare`, which answers the question again
        rather than carrying an answer about one file forward onto another.
        """
        return self._recorder.spool_created_by_this_recorder

    @property
    def spool_created_by_last_prepare(self) -> bool:
        """Whether the most recent :meth:`prepare` is what created the spool.

        The narrower of the two ownership facts, and the one a caller cleaning
        up after a ``prepare()`` that raised has to use: the question there is
        what *this attempt* created. A ``prepare()`` refused because the
        recorder was already prepared created nothing, and the spool at the
        path belongs to the earlier call that succeeded and still holds it.
        """
        return self._recorder.spool_created_by_last_prepare

    def spool_snapshot(self) -> bytes:
        """Everything written to the spool so far.

        For diagnosis and for the tests that check the container the recorder
        actually produced. Stage 3 is a private, rebuildable checkpoint on the
        way to stage 4 and is not storage a caller is meant to read as a
        session (contract section 1).
        """
        return self._recorder.spool_snapshot()

    # --- lifecycle ----------------------------------------------------------

    def prepare(self) -> None:
        """Validate the plan and allocate every bounded resource.

        This is the only allocation point on any recording path. It writes no
        durable byte, so a failure leaves the recorder in ``created`` with
        nothing retained.
        """
        self._check(self._recorder.prepare(), "prepare")

    def rollback_prepare(self) -> None:
        """Undo a successful :meth:`prepare`, back to ``created``.

        For a caller whose own ``prepare()`` is wider than this one -- the
        stable facade also writes the spool's plan sidecar -- and whose second
        half failed. Contract section 3.1 gives one answer for a failed
        ``prepare()`` regardless of how far it got: no partial resource, no
        session artifact, and an object that may be prepared again.
        :meth:`close` cannot produce that, because it is a transition into a
        terminal state.

        Only the native resources are released. The spool's *path* is not
        removed here: that decision belongs to whoever chose the path, and it
        turns on whether this recorder created it.
        """
        self._check(self._recorder.rollback_prepare(), "rollback_prepare")

    def _prepare_experiment_attachment(self) -> Any:
        """Allocate the store while leaving core prepare to ExperimentSession."""

        return self._recorder._prepare_experiment_attachment()

    def _rollback_experiment_attachment(self) -> None:
        self._recorder._rollback_experiment_attachment()

    def _attach_experiment(
        self,
        runner: Any,
        *,
        edge_id: int | None = None,
        capacity: int | None = None,
    ) -> None:
        native_streaming = self._streaming()
        if self._attached_edge_id is not None:
            raise RecorderStateError("this recorder is already attached to a runtime")
        if runner.state != native_streaming.RuntimeState.CREATED:
            raise RecorderStateError(
                "experiment recording must attach before the runner is prepared"
            )
        config = native_streaming.ObserverEdgeConfig()
        config.id = self._options.edge_id if edge_id is None else edge_id
        config.capacity = self._options.edge_capacity if capacity is None else capacity
        config.drop_history_capacity = self._options.edge_drop_history_capacity
        config.drop_policy = native_streaming.ObserverDropPolicy.FAULT
        config.critical_recorder = True
        target = getattr(runner, "_runner", runner)
        status = self._recorder._attach_experiment(target, config)
        if status != native_streaming.StreamStatus.OK:
            raise RecorderError(f"the runtime refused the experiment recorder edge: {status}")
        self._attached_edge_id = config.id
        self._runner = runner

    def attach(
        self,
        runner: Any,
        *,
        edge_id: int | None = None,
        capacity: int | None = None,
    ) -> None:
        """Register this recorder as *runner*'s critical observer.

        Ordering is part of the contract, not a convenience: the edge has to
        exist before the runner is prepared, because the readiness gate runs
        inside ``runner.arm()`` and the runtime must not arm while a critical
        recorder is not ``ready`` (section 3.1). Attaching after the runner is
        prepared would leave the runtime armed against a recorder it never
        gated, so it is refused here rather than discovered later.

        The runner is kept, not just its edge id. Ending a session is a
        two-object operation in a fixed order -- runtime first, recorder second
        -- and a recorder that had forgotten which runtime it was observing
        could not perform it. :meth:`stop`, :meth:`abort` and :meth:`close`
        all use it.
        """
        native_streaming = self._streaming()
        if self._attached_edge_id is not None:
            raise RecorderStateError(
                f"this recorder is already attached on edge {self._attached_edge_id}; "
                "a session is single-use and a recorder observes exactly one runtime"
            )
        if self.state != self._native.RecorderLifecycleState.PREPARED:
            raise RecorderStateError(
                f"attach() needs a prepared recorder, not one in {self.state}; call prepare() first"
            )
        if runner.state != native_streaming.RuntimeState.CREATED:
            raise RecorderStateError(
                f"attach() must happen before the runner is prepared, but the runner is in "
                f"{runner.state}; the readiness gate runs inside arm() and cannot gate an edge "
                "that did not exist yet"
            )

        options = self._options
        config = native_streaming.ObserverEdgeConfig()
        config.id = options.edge_id if edge_id is None else edge_id
        config.capacity = options.edge_capacity if capacity is None else capacity
        config.drop_history_capacity = options.edge_drop_history_capacity
        # Not configurable: a critical recorder edge is lossless until fault.
        # Any other policy would let the edge drop a required item and count it
        # as a statistic, which is the noncritical path's behaviour.
        config.drop_policy = native_streaming.ObserverDropPolicy.FAULT
        config.critical_recorder = True

        target = getattr(runner, "_runner", runner)
        status = self._recorder.attach(target, config)
        if status != native_streaming.StreamStatus.OK:
            raise RecorderError(f"the runtime refused the critical recorder edge: {status}")
        self._attached_edge_id = config.id
        self._runner = runner

    def runtime_is_live(self) -> bool:
        """Whether an attached runtime is still running or still stopping.

        A recorder with no runtime attached is never live: it is being driven
        by something else, and this facade has nothing to coordinate.
        """
        if self._runner is None:
            return False
        states = self._streaming().RuntimeState
        return self._runner.state in (states.RUNNING, states.STOPPING)

    def _quiesce_runtime(self, *, graceful: bool) -> None:
        """End the attached runtime and wait for it, before touching the recorder.

        This is the ordering contract section 3.1 fixes, and it is not
        cosmetic. The runtime's terminal notice is what drives the recorder's
        own ``drain()``, so after ``join()`` the recorder has usually already
        reached ``stopped`` and the caller's ``stop()``/``abort()`` is the
        idempotent second call the contract describes.

        Doing it the other way round does not end the session, it damages it:
        a recorder that stopped accepting while its producer is still running
        turns the next frame into a critical-edge failure, and a caller who
        asked for a graceful stop gets a faulted session.

        A terminal intent the runtime already latched is not rewritten by the
        recorder call that follows -- that is the core's rule, and it is what
        makes ``abort()`` after a runtime fault report the fault rather than the
        abort.

        This is a **gate**, not a courtesy call: it returns only once the
        runtime is provably quiesced, and raises
        :class:`~neurale.recording.RecorderRuntimeShutdownError` otherwise, so
        the recorder stage that follows is never entered on an unfinished one.
        """
        if not self.runtime_is_live():
            return
        verb = "stop" if graceful else "abort"
        shutdown = self._runner.stop() if graceful else self._runner.abort()
        # Bounded by the runtime's own shutdown, not by a timeout invented here:
        # adding a second deadline would create two answers to "did it stop?".
        joined = self._runner.join()
        self._require_quiesced(verb, shutdown, joined)

    def _require_quiesced(self, verb: str, shutdown: Any, joined: Any) -> None:
        """Return only if the runtime is genuinely finished producing.

        Deliberately *not* ``shutdown == OK``. A runtime that already went
        terminal through its own fault path answers with that original fault,
        and refusing to end the session because the runtime ended badly would
        make a faulted runtime impossible to record the ending of -- exactly the
        case the spool exists for. What has to hold is narrower and is only
        about whether anything can still produce:

        * neither call reported ``deadline_exceeded`` -- the one status that
          means "the bound elapsed and the workers may still be there";
        * ``join()`` returned, so the worker threads are collected;
        * the runtime is in a terminal state (``stopped`` or ``failed``).

        When it does not hold, no recorder call is made at all. The recorder is
        left to the runtime's own shutdown and fault protocol, which still owns
        it, rather than being stopped underneath a producer that may still be
        running -- which would charge a runtime shutdown timeout to the
        recording as a critical-edge fault.
        """
        streaming = self._streaming()
        deadline = streaming.StreamStatus.DEADLINE_EXCEEDED
        states = streaming.RuntimeState
        state = self._runner.state
        timed_out = [
            f"{name}() returned {status}"
            for name, status in ((verb, shutdown), ("join", joined))
            if status == deadline
        ]
        if not timed_out and state in (states.STOPPED, states.FAILED):
            return
        if not timed_out:
            timed_out = [f"{verb}() returned {shutdown} and join() returned {joined}"]
        raise RecorderRuntimeShutdownError(
            f"the attached runtime did not quiesce: {'; '.join(timed_out)}, leaving it in "
            f"{state}. The recorder was left untouched -- stopping it now would let a worker "
            f"that is still running meet a recorder that no longer accepts, turning this "
            f"runtime shutdown timeout into a critical-edge fault charged to the recording. "
            f"Resolve the runtime, then end the session."
        )

    def stop(self, reason: str = "stop") -> Any:
        """Graceful stop: quiesce the runtime, latch intent ``normal``, drain, seal.

        Stops and joins the attached runner first (see :meth:`_quiesce_runtime`),
        then stops the recorder. Legal from ``recording`` onward and idempotent
        once the terminal intent is latched (section 3.1): a call arriving after
        the shutdown already ran returns that shutdown's status and changes
        nothing.

        Raises :class:`~neurale.recording.RecorderRuntimeShutdownError`, without
        touching the recorder at all, if the runtime does not quiesce.
        """
        self._quiesce_runtime(graceful=True)
        self._check(self._recorder.stop(reason), "stop")
        return self.status

    def abort(self, reason: str = "abort") -> Any:
        """Explicit abort: abort the runtime, then latch intent ``aborted``.

        Not a fault -- no fault record is invented -- and still a full drain,
        because an abort stops recording new data rather than destroying data
        already handed over (section 4.4). The runtime is aborted and joined
        first, so the producer is genuinely stopped rather than left to
        discover a recorder that is no longer accepting -- and if the abort does
        not quiesce it, this raises
        :class:`~neurale.recording.RecorderRuntimeShutdownError` and leaves the
        recorder alone rather than aborting it under a live producer.
        """
        self._quiesce_runtime(graceful=False)
        self._check(self._recorder.abort(reason), "abort")
        return self.status

    def close(self) -> Any:
        """Release every recorder-owned resource. Idempotent.

        Refused while an attached runtime is still live, and that refusal is
        the point. ``close()`` releases the storage the critical edge is
        writing into, so running it under a live producer is the one ordering
        the contract's shutdown rules exist to prevent -- and unlike
        :meth:`stop` and :meth:`abort` there is no intent here to act on.
        Ending a session says *how* it ended, and ``close()`` must not pick
        ``normal`` or ``aborted`` on the caller's behalf: that choice is latched
        into the session's terminal intent and cannot be revised. Call
        :meth:`stop` or :meth:`abort` first, or use the context manager, which
        does exactly that.

        ``closed`` says nothing about the artifact: read ``session_created``,
        ``sealed``, and ``finalization_required`` on the returned status for
        that. Finalization is a separate offline step, so a spool holding
        committed records is left finalizable and the status says so.
        """
        if self.runtime_is_live():
            raise RecorderStateError(
                f"close() would release the recorder's storage while the attached runtime is "
                f"{self._runner.state}; stop() or abort() the recorder first so the session "
                "records how it ended, rather than closing out from under a running producer"
            )
        # Idempotence lives in the native core, and this deliberately keeps no
        # cached copy of the first answer to replay. The contract's wording is
        # exact about why: a repeated call returns the same status and re-decides
        # nothing, but work already running may have finished in between, and
        # that is the background making progress rather than this call deciding
        # something new. A cache here would hide the difference.
        self._check(self._recorder.close(), "close")
        return self.status

    def finalize(self) -> Any:
        """Convert the spool into a canonical NRF session. **Not implemented here.**

        Owned by the public facade's offline finalizer; see the module docstring.
        """
        self._check(self._recorder.finalize(), "finalize")
        return self.status

    def abandon_finalization(self) -> Any:
        """Stop retrying a failed finalization. **Not implemented here.**

        Owned by the public facade's recovery API; see the module docstring.
        """
        self._check(self._recorder.abandon_finalization(), "abandon_finalization")
        return self.status

    # --- the control plane --------------------------------------------------
    #
    # These mirror `SessionRecorder`'s typed methods argument for argument, so the
    # two surfaces can be driven by the same calls. Typing them here also puts
    # field validation before the enqueue rather than at finalization, where a
    # malformed body would surface
    # as a conversion failure over a session that has already been recorded.
    #
    # What travels in the body is the record minus two things the container
    # already carries: the timestamp (the control record header's `time_ns`) and
    # the string record id. The spool specification is explicit that the string
    # id (`event-00000001`) is a materialization-time identity and *not* the
    # accounting identity -- the `uint64` identity is -- so writing it into the
    # body would put a second, disagreeing identity into the container.

    def record_event(
        self,
        name: str,
        *,
        time_ns: int | None = None,
        value: float | None = None,
        text: str | None = None,
    ) -> bool:
        """Record one timestamped experiment event."""
        return self._record_named("events", name, time_ns, value, text)

    def record_state(
        self,
        state: str,
        *,
        time_ns: int | None = None,
        value: float | None = None,
        text: str | None = None,
    ) -> bool:
        """Record one experiment state transition."""
        return self._record_named("experiment_states", state, time_ns, value, text)

    def record_command(
        self,
        command: str,
        *,
        time_ns: int | None = None,
        value: float | None = None,
        text: str | None = None,
    ) -> bool:
        """Record one issued command."""
        return self._record_named("commands", command, time_ns, value, text)

    def record_task_variable(
        self,
        name: str,
        *,
        time_ns: int | None = None,
        value: float | None = None,
        text: str | None = None,
    ) -> bool:
        """Record one task variable observation.

        The kind the spool container went to version 1.1 for: before
        ``task_variables = 11`` entered the producer-identity registry there was
        no number to record this under, and no way to name it as the first
        control record a plane refused.
        """
        return self._record_named("task_variables", name, time_ns, value, text)

    def record_label(
        self,
        label: str,
        *,
        time_ns: int | None = None,
        value: float | None = None,
        text: str | None = None,
    ) -> bool:
        """Record one timestamped label."""
        return self._record_named("labels", label, time_ns, value, text)

    def record_target(
        self,
        name: str,
        *,
        time_ns: int | None = None,
        value: float | None = None,
        text: str | None = None,
    ) -> bool:
        """Record one target. A multi-dimensional target goes in *text* as JSON."""
        return self._record_named("targets", name, time_ns, value, text)

    def record_assistance(
        self,
        name: str,
        *,
        time_ns: int | None = None,
        value: float | None = None,
        text: str | None = None,
    ) -> bool:
        """Record one assistance value."""
        return self._record_named("assistance", name, time_ns, value, text)

    def record_trial(
        self,
        *,
        start_ns: int,
        stop_ns: int,
        label: str | None = None,
        outcome: str | None = None,
    ) -> bool:
        """Record one completed trial.

        The record's ``time_ns`` is the trial's ``start_ns``. A trial spans an
        interval and the header carries one instant, so the interval stays whole
        in the body and the header names the instant the trial is ordered by.
        """
        return self.submit_control(
            "trials",
            {
                "start_ns": int(start_ns),
                "stop_ns": int(stop_ns),
                "label": label,
                "outcome": outcome,
            },
            time_ns=int(start_ns),
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
        """Offer one fault row.

        A recorded fault row is data, not a verdict: a session with fault rows
        can still end ``normal`` and complete (contract section 3.2). The
        *primary* fault that ends a session is latched by the recorder or the
        runtime through a reserved path, not submitted here.
        """
        return self.submit_control(
            "faults",
            {
                "code": str(code),
                "stage": str(stage),
                "frame_sequence": None if frame_sequence is None else int(frame_sequence),
                "signal_id": None if signal_id is None else int(signal_id),
                "text": text,
            },
            time_ns=time_ns,
        )

    def checkpoint(self) -> None:
        """Ask the spool worker for a checkpoint at its next transaction.

        Only an open session can be checkpointed. Accepting the request in any
        other state would return quietly having written nothing, which reads
        like a checkpoint that succeeded.

        It does not wait for the checkpoint to be written. There is no bound on
        when a quiet session commits its next transaction, and a wait without a
        bound is exactly what a recorder shutdown may not contain.
        """
        if not self._recorder.request_checkpoint():
            raise RecorderStateError(
                f"the recorder is {self.state}; a checkpoint needs a session that is still "
                "accepting"
            )

    def _record_named(
        self,
        kind: str,
        name: str,
        time_ns: int | None,
        value: float | None,
        text: str | None,
    ) -> bool:
        """The uniform ``name``/``value``/``text`` body seven kinds share.

        The same shape the Python recorder writes, minus its string record id.
        One shape a reader can consume without a per-kind special case is worth
        more than seven bespoke ones, and keeping the two engines on the same
        one is what makes the parity fixtures compare like with like.
        """
        return self.submit_control(
            kind,
            {
                "name": str(name),
                "value": None if value is None else float(value),
                "text": text,
            },
            time_ns=time_ns,
        )

    def submit_control(
        self,
        kind: str,
        body: Mapping[str, Any],
        *,
        identity: int | None = None,
        time_ns: int | None = None,
        clock_domain: int | None = None,
    ) -> bool:
        """Offer one control record and report whether it was **control accepted**.

        The typed methods above are the surface to use; this is what they are
        built from, and it is public because a parity harness needs to submit a
        kind by name.

        ``True`` means exactly control accepted and nothing further -- not spool
        committed, not NRF committed, not finalized (contract section 1.1). On
        this plane acceptance stages 1 and 2 coincide by construction, because
        the storage the record is accepted into is already the recorder's.

        ``False`` means the record was **not** accepted. It has already been
        counted with a named reason on the native side and appears in
        ``status.control_rejected`` with a first-rejection position; it is never
        silently discarded.

        A string longer than ``max_control_string_bytes`` **raises** rather than
        returning ``False``, and the difference is deliberate. ``False`` is the
        answer for a record the recording path refused, which is an event in the
        session's accounting. A string over the bound never entered the path:
        nothing was offered, nothing was lost, and the session's verdict is
        untouched -- it is caller error, in the same category as a late
        submission (section 1.3), and reporting it as a recording outcome would
        put API misuse into the numbers that describe the recording.

        The whole-body bound is *not* checked here. It belongs to the native
        core, which treats an oversized body as a plan violation and faults the
        session, and duplicating the check would hide that behaviour behind a
        Python exception that never reaches the recorder.

        *identity* is what fixes this record's order within its kind, and it is
        stored on the record so ordering is recoverable from the record rather
        than inferred from position in a file. Left out, a per-kind counter
        supplies it.
        """
        if kind not in CONTROL_KINDS:
            raise RecorderConfigError(
                f"{kind!r} is not a control-plane kind; expected one of {list(CONTROL_KINDS)}"
            )
        self._check_string_bounds(kind, body)
        if identity is None:
            identity = self._control_identities[kind]
            self._control_identities[kind] = identity + 1
        return self._recorder.submit_control(
            kind=getattr(self._native.ProducerIdentityKind, kind.upper()),
            identity=identity,
            clock_domain=(
                self._options.control_clock_domain if clock_domain is None else clock_domain
            ),
            # Host monotonic, matching the clock the session declares for control
            # records. See `HOST_MONOTONIC_CLOCK_DOMAIN`.
            time_ns=time.monotonic_ns() if time_ns is None else int(time_ns),
            body=canonical_json_bytes(dict(body)),
        )

    def _check_string_bounds(self, kind: str, body: Mapping[str, Any]) -> None:
        """Refuse any single string over the bound, naming where it was found.

        Every string in the body, at any depth: values nested in mappings and
        sequences, and mapping *keys* too. A top-level-only check is not a bound
        at all, because ``{"metadata": {"note": <huge>}}`` and a huge key both
        route the same unbounded free text into the same record while satisfying
        it -- and the bound exists precisely so that one caller-supplied string
        cannot fill a record the container promised would be bounded. The
        whole-body maximum would still catch some of these later, in the native
        core, as a *fault*; the point of this check is that caller error is
        refused before anything is offered, not converted into a recording loss.

        Measured in UTF-8 bytes, because that is what the container stores; a
        character count would let a string of astral-plane characters be four
        times the bound it passed.

        The walk is iterative and remembers the containers it has entered, so a
        deeply nested or self-referential body raises the ordinary bound or
        encoding error rather than exhausting the stack here.
        """
        limit = self._options.max_control_string_bytes
        seen: set[int] = set()
        pending: list[tuple[str, Any]] = [("", body)]
        while pending:
            path, value = pending.pop()
            if isinstance(value, str):
                self._check_one_string(kind, path or "record", value, limit)
            elif isinstance(value, Mapping):
                if id(value) in seen:
                    continue
                seen.add(id(value))
                for key, item in value.items():
                    child = f"{path}.{key}" if path else str(key)
                    if isinstance(key, str):
                        self._check_one_string(kind, f"{child} (key)", key, limit)
                    pending.append((child, item))
            elif isinstance(value, (list, tuple)):
                if id(value) in seen:
                    continue
                seen.add(id(value))
                for i, item in enumerate(value):
                    pending.append((f"{path}[{i}]", item))

    def _check_one_string(self, kind: str, where: str, value: str, limit: int) -> None:
        size = len(value.encode("utf-8"))
        if size > limit:
            raise RecorderConfigError(
                f"the {where!r} string of this {kind} record is {size} UTF-8 bytes, over the "
                f"max_control_string_bytes bound of {limit}. Nothing was offered to the "
                "recorder, so the session's accounting is unchanged."
            )

    # --- context manager ----------------------------------------------------

    def __enter__(self) -> NativeSessionRecorder:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        """Leave normally with a graceful stop; leave on an exception with an abort.

        What is normative is the outcome, not the number of calls (section 3.1):
        an exception never ends a session as ``normal``. Both paths quiesce an
        attached runtime before touching the recorder, and both then close,
        which is idempotent.

        The two paths differ in what a runtime that will not quiesce does. On
        the normal path it raises: leaving the block was supposed to end the
        session, and it did not. On the exception path the caller's exception is
        the more important one, so the shutdown failure is attached to it as a
        note and the original propagates -- and ``close()`` is skipped, because
        closing would release the storage a runtime that never stopped may still
        be writing into.
        """
        if exc_type is None:
            self.stop("context-exit")
            self.close()
            return False

        try:
            self.abort(f"context-exit: {exc_type.__name__}")
        except RecorderRuntimeShutdownError as shutdown_error:
            note = f"the native recorder was left open: {shutdown_error}"
            add_note = getattr(exc, "add_note", None)
            if add_note is not None:
                add_note(note)
            else:  # pragma: no cover - every supported Python has add_note
                exc.__notes__ = [*getattr(exc, "__notes__", ()), note]
            return False
        self.close()
        return False

    # --- internals ----------------------------------------------------------

    def _streaming(self) -> Any:
        """The streaming package, imported lazily and through its public surface.

        Public rather than ``neurale.streaming._native``: the package resolves
        these names through its own lazy ``__getattr__``, so going around it
        would load the extension by a path the import contract does not
        describe, and would bind this facade to another package's private
        module for names it exports anyway.
        """
        import neurale.streaming as streaming

        return streaming

    def _check(self, code: Any, operation: str) -> None:
        """Turn a native status code into an exception, or return.

        Every failure surfaces. There is no path here that absorbs a native
        failure and continues on the Python engine instead (contract section 2,
        decision 3).
        """
        native = self._native
        if code == native.RecorderStatusCode.OK:
            return
        if code == native.RecorderStatusCode.WRONG_STATE:
            raise RecorderStateError(
                f"{operation}() is not legal while the recorder is {self.state}"
            )
        if code == native.RecorderStatusCode.INVALID_PLAN:
            raise RecorderConfigError(
                f"{operation}() refused the recording plan: the native core reported "
                f"{code}. Nothing was allocated and no session was created."
            )
        if code == native.RecorderStatusCode.NOT_READY:
            raise RecorderError(
                f"{operation}() did not pass the readiness gate ({code}), and every prepared "
                "resource was released. Check the lossless edge and durability policy; "
                "there is no fallback."
            )
        if code == native.RecorderStatusCode.NOT_IMPLEMENTED:
            raise NotImplementedError(
                f"{operation}() is not implemented on the native core: finalization and "
                "the diagnose/resume/abandon surface are owned by the offline finalizer "
                "and recovery API. The status reports finalization_required rather than "
                "a fabricated success."
            )
        raise RecorderError(f"{operation}() failed on the native recorder: {code}")
