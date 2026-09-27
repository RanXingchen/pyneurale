# Native recording and replay lifecycle contract

## Status and scope

This document is the single normative definition of the recording and replay
lifecycle. It fixes the vocabulary, the session states, the ordering of every
shutdown path, which records must survive each outcome, when an artifact may be
deleted, and how replay relates to the recording it came from. The
implementation exposes one public native-backed ``SessionRecorder``.
Normative requirements use MUST, MUST NOT, SHOULD, and MAY. Current release
support remains owned by the repository-root ``SUPPORT.md`` matrix.

This document specifies the lifecycle **and the externally observable
recording/replay semantics**. It may define required logical records, metadata
fields, and their meanings -- ledgers, the accounting summary, replay
configuration, omission metadata, ranges, runtime sequence, discontinuity
transformation -- and it leaves the **physical** encoding of those records
(binary layout, JSON shape, Zarr structure, compression, checksum placement) to
the owning implementation and format specifications. It does not cover device
protocols, experiment state machines, actuator safety policy, or archive
packaging.

Being the single source has a consequence in both directions: a rule that is
observable through a public API belongs **here**, not in a task description, and
a rule about bytes on disk belongs in the owning format specification, not here.

Related contracts: {doc}`native_streaming` owns the generic runtime, and the
[NRF v1 specification](https://github.com/RanXingchen/pyneurale/blob/main/specifications/nrf/v1/README.md)
owns canonical session storage. Neither is restated here.

## 1. The acceptance ladder

An item of recording data passes through a fixed sequence of stages. Each stage
is a different guarantee, and the terms MUST NOT be used interchangeably in
code, documentation, status fields, log text, or test names.

| Stage | Term | Established when | Guarantee |
| --- | --- | --- | --- |
| 1 | **producer accepted** | the recording path has taken the item over from its producer -- **runtime accepted** on the data plane, **control accepted** on the control plane; see section 1.1 | the producer is released from the item; nothing has been committed to any store |
| 2 | **recorder accepted** | the item has been copied into storage the recorder owns and bounds | the item is the recorder's responsibility and its loss is now a recording loss |
| 3 | **spool committed** | the item lies inside a spool transaction whose commit marker is written and checksum-valid | eligible for promotion, for as long as that transaction remains readable and checksum-valid; a torn or uncommitted tail is never eligible |
| 4 | **NRF committed** | an NRF transaction commit record covers the item and the committed extent of its target includes it | the transaction is *logically* visible in the journal and the target extent, and a reader returns the item once every referenced object validates |
| 5 | **finalized** | the NRF session carries a termination record | the session is sealed and its outcome is stated on disk |

Rules:

- A recorder MUST NOT report an item as recorded before stage 4.
- Stage 2 MUST NOT be described as durable, committed, or recorded. An
  acknowledged enqueue is an acknowledged enqueue.
- Stage 3 is a private, rebuildable checkpoint on the way to stage 4. It MUST
  NOT be presented to users as storage.
- Stage 5 is not stage 4 for every item. A session can be finalized while
  reporting that data was lost; see section 3.2.
- Durability is **not** a stage. It is an attribute of a committed extent, and
  it is defined in section 1.2. No stage of this ladder promises survival across
  a crash on its own: survival is bounded by the reported durable extent and the
  policy in force, and recovery MAY roll back a logically committed but
  non-durable tail. A stage that claimed to "survive crash recovery"
  unconditionally would make durability a stage again.
- A recorder status MUST expose every stage its own path has as a distinct
  counter. A single "recorded" number spanning stages is a contract violation.
  An offline writer or a readable session that was recorded without native
  spool capture has no stage 3 to report; the current public live recorder has
  all five and MUST report all five.

### 1.1 The two producer planes

Stage 1 has two forms, because two different kinds of producer hand data to a
recorder. They MUST be named separately and counted separately.

**Data plane.** The producer is the runtime. Stage 1 is **runtime accepted**:
the runtime has validated the item and placed it on the observer dispatch edge.
That is a statement about the runtime and says nothing about recording. Stage 2
is separate and later: the recorder copies the item out of the edge queue into
storage it owns.

The **item** on this plane is exactly one accepted data `StreamMessage`, and
that is the unit every data-plane ladder counter counts:

| `StreamMessageKind` | Recording data? | Counted in the ladder |
| --- | --- | --- |
| `frame` | yes | yes, one item |
| `discontinuity` | yes | yes, one item |
| `end_of_stream`, `shutdown`, `abort` | no -- lifecycle signals | no |

A `SignalBlock` is a member of a frame, not a separately accepted item: the
runtime never hands a block to an edge on its own, so a frame carrying three
blocks is **one** runtime-accepted item and MUST NOT become three. Frames,
blocks, observations, rows, and bytes MAY all be reported as diagnostic
counters, and they are useful ones, but they MUST NOT appear in the conservation
identities of section 5.1 -- mixing units there produces equations that cannot
hold and cannot be checked (a three-block frame would read
`runtime_accepted = 1` against `nrf_committed = 3`).

Because an item is one message, its commit is **all-or-nothing**:

- A **frame** counts toward `nrf_committed` only once *everything the recording
  plan requires for it* is committed: every signal block the plan records, its
  `native-frames-v1` row, each corresponding `native-signal-blocks-v1` row, and
  the payload and index data they reference.
- A **discontinuity** counts only once its `native-discontinuities-v1` row and
  every `native-signal-gaps-v1` row that row references are committed.

An item whose parts committed but whose ledger row did not is **not** partially
committed -- it is not committed, and it MUST be counted as lost at the handoff
where it stopped. There is no fractional item anywhere in this contract.

**Every data item MUST carry a global data-message ordinal**, assigned at
runtime acceptance, monotonic across frames *and* discontinuities, and carried
into the native frame ledger, the native discontinuity ledger, the spool record,
and the accounting summary. A frame sequence cannot do this job: a discontinuity
whose `actual_frame_sequence` is 42 sits immediately before the frame whose
header sequence is 42, so "the first item lost was 42" does not say *which* of
the two was lost, or whether both failed at different handoffs. This is the same
problem the control plane solves with a submission ordinal (below), and it takes
the same solution. An implementation that will not carry an ordinal MUST record
the pair `(message_kind, frame_sequence)` instead; a bare sequence number is not
an item identity on this plane.

The data-message ordinal is not the only identity the ledgers carry. The frame
ledger additionally carries a **`frame_ordinal`**: a contiguous, zero-based,
gap-free count over **frame rows only** -- it skips discontinuities, which the
data-message ordinal counts. The two are not interchangeable and a ledger needs
both: the ordinal is the item identity shared with the discontinuity ledger and
the accounting summary, while `frame_ordinal` is the frame-row identity the block
ledger's rows point back to. The block ledger carries each block's original
`payload_offset` (its byte position in the frame payload, from the native
`SignalBlockHeader`, not the per-stream array `row_offset`) and the frame ledger
carries `total_payload_byte_count`, so exact replay verifies the recorded blocks
cover the whole frame with no trailing or overlapping bytes -- against the
original recording, not against a reconstruction. On-disk shape, including the
nullable zero-child anchor rule for a parent row with no children, is fixed by
the extension README; this contract fixes the meaning.

**Control plane** -- events, experiment states, commands, targets, labels,
assistance, trials, and fault rows. The producer is the experiment or the
application, calling the recorder directly; nothing on this path passes through
the runtime or an observer edge. Stage 1 is **control accepted**: the record is
inside the recorder's own bounded control storage.

On the control plane stages 1 and 2 coincide by construction -- the storage the
record is accepted into is already the recorder's. An implementation MUST say
so plainly and MUST NOT report a fictitious second stage for control records.

Rules for the control plane:

- A control API returning `True` means **exactly** control accepted, and
  nothing further. It is not a statement about stage 3, 4, or 5, and MUST NOT
  be documented, logged, or tested as one.
- A control API returning `False` means the record was **not** accepted. It
  MUST be counted with a named reason (section 1.3) and MUST NOT be silently
  discarded.
- Every control record MUST carry, at acceptance: the session identity, the
  clock domain its `time_ns` belongs to, and an identity that fixes its order
  within its kind. Ordering MUST be recoverable from the record, never inferred
  from position in a file.
- **Naming one control record across kinds.** A per-kind identity cannot name
  "the first control record that was lost", because two kinds' identities are
  not comparable. An implementation MUST therefore either record a **global
  control submission ordinal** -- assigned at acceptance, monotonic across every
  kind, and stored on the record -- or report that first-lost position as the
  pair `(kind, identity)`. A bare number that silently means "within some kind"
  is forbidden: it reads like a plane-wide position and is not one. Section 5.1
  states which form the summary carries.
- The primary fault record and the terminal reason MUST NOT depend on free
  space in the ordinary control queue. They MUST have reserved, bounded storage
  that is still writable when the control queue is exactly the thing that is
  full (section 4.3).

Completeness is decided by *both* planes: a session is complete only if neither
plane lost or rejected anything (section 3.2). Neither plane's counters may be
folded into the other's.

### 1.2 Durability is an attribute of a committed extent

Durability is not a point in the ladder. Each store that commits has its own
committed extent and its own durability level for that extent, and neither
implies anything about the other:

```text
capture path:
  producer accepted -> recorder accepted -> spool committed
                                                └── spool durability level

finalization path:
  spool committed   -> NRF committed -> finalized
                            └── NRF durability level
```

An offline writer or a readable session recorded without native spool capture
has no spool, so its ladder is:

```text
  producer accepted -> recorder accepted -> NRF committed -> finalized
                                                 └── NRF durability level
```

A status, benchmark record, or recovery report that reports durability MUST
report it per store, as an extent, not as a stage count:

```text
spool_committed_count   /  spool_durable_extent
nrf_committed_count     /  nrf_durable_extent
```

The durability *levels* themselves are defined in section 6. Nothing here
implies that either store is synchronized by default.

### 1.3 Losing, rejecting, and submitting late

Three distinct things happen to an item that does not reach stage 4. They MUST
be counted separately and MUST NOT be merged into one "lost" number.

**Rejected.** The recording path was open and refused the item before it became
that path's responsibility. Rejections do **not** all happen at the same place,
and each one MUST be counted at the handoff where it happened:

| Refusal | Where it sits in the ladder | Counted as |
| --- | --- | --- |
| an over-capacity observer edge | the item never became **runtime accepted for that edge** | `rejected_before_runtime_acceptance`, data plane |
| a full recorder frame queue | already runtime accepted, never **recorder accepted** | `failed_between_runtime_and_recorder`, data plane |
| a full control queue | never **control accepted** | `control_rejected`, control plane |

Every row is a recording loss: a producer offered data to a path that had
promised to take it. What differs is *which stage's number the item is missing
from*, and that is why there is no single `accepted = committed + rejected +
lost` identity for a session (section 5.1 states the per-handoff identities that
do hold). A rejection MUST NOT be subtracted from a stage the item never
reached.

**Lost after acceptance.** The item reached stage 2 or 3 and no longer has a
path to stage 4 -- an abort that discarded accepted items, a writer failure, an
uncommitted spool tail. This is a recording loss.

**Submitted late.** The item arrived after the recorder stopped accepting. It
was never accepted by anything, and the session boundary it belongs after has
already passed. This is **not** a recording loss; it is API misuse by the
caller. It MUST be counted under a separate rejected-after-close counter, and
it MUST NOT change the verdict of a session that was already sealed
(section 3.2, and gap 9).

Both kinds of recording loss MUST be counted at the stage where they happened,
MUST name that stage, and MUST be reflected on disk before the session is
finalized. Silent loss is the one failure this contract does not tolerate
anywhere.

For a **critical recorder**, loss of required data is not a countable event to
be recorded and continued past: it MUST fault the recorder and abort the
runtime (section 4.3). Counting and continuing is the noncritical recorder's
behavior, not the critical one's.

## 2. The recording path

The repository has exactly one public live-recording path:

| Concern | Current contract |
| --- | --- |
| Public API | `neurale.recording.SessionRecorder` |
| Attachment | native observer on an edge with `critical_recorder = true` |
| Loss policy | lossless until fault; a lossy policy is rejected before prepare |
| On queue/spool failure | recorder faults and the runtime aborts |
| Crash behavior | committed native spool prefix and bounded recovery point |
| Critical data plane | no Python callback, GIL, NRF/Zarr, or filesystem work |
| Finalization | noncritical offline conversion through the canonical NRF writer |

There is no alternate public recorder backend, stable public
`NativeSessionRecorder`, backend selector, or fallback. Sessions without the
current accounting summary remain readable under the compatibility verdict in
section 5.1, but they do not provide a selectable live-capture path. A native
load, plan, prepare, readiness, queue, spool, or writer failure surfaces as an
explicit failure. The generic `PythonObserverBridge` remains available for
non-recording observer use and MUST NOT be accepted as a critical recorder: it
runs Python on its worker and takes the GIL, which the critical contract
forbids.

## 3. Recorder lifecycle states and session outcomes

Two different things are called "state" in recording code and they are not the
same question. Section 3.1 is the **recorder object's** lifecycle: which
operations are legal, in which order, and what each transition must have
achieved. Section 3.2 is the **session's** outcome: what the artifact on disk
turned out to be. A recorder in a terminal state says nothing by itself about
whether the session it produced is complete.

### 3.1 Recorder lifecycle states

This state machine is normative for the native recorder. It is fixed here so
the implementation follows it rather than reinventing it.

**The word `faulted` belongs to the session, not to the recorder.** The
recorder's failure state is called **`failed`** and means only "this recorder's
lifecycle could not complete normally". The two are independent, and the normal
result of a runtime fault whose session was saved successfully is exactly:

```text
recorder lifecycle = closed
session outcome    = faulted
```

A recorder MUST NOT reuse the word `faulted` for its own state, and a status
API MUST NOT let one be read as the other.

| State | Meaning |
| --- | --- |
| `created` | constructed; no plan, no resources, no session on disk |
| `prepared` | the plan is validated and every bounded resource is allocated and fixed; nothing has been accepted |
| `ready` | prepared **and** the readiness gate passed; the runtime may now be armed |
| `recording` | accepting records |
| `draining` | not accepting; delivering what was already accepted, under a bound |
| `stopped` | the worker finished and the spool session-end is committed; not yet finalized |
| `finalizing` | converting the spool into a canonical NRF session; read-only over the spool |
| `finalization_failed` | **not terminal**: finalization stopped without producing a sealed session, and MAY be retried |
| `closed` | terminal: every recorder-owned resource is released and the object accepts no further work |
| `failed` | terminal capture failure; resources still referenced by disk I/O remain owned until close can reclaim them (section 4.7); any committed spool prefix is retained |

**`closed` says nothing about the artifact.** It is a statement about the
*object*: its resources are gone and it will do no more work. Whether a session
was ever created, sealed, complete, or finalized is reported by separate fields
that a status MUST expose independently:

| Field | Question it answers |
| --- | --- |
| `session_created` | was any session artifact produced at all? |
| `sealed` | does the NRF session carry a termination record? |
| `complete` | section 3.2's completeness verdict |
| `requested_terminal_intent` | what the first exit from `recording` latched: `normal`, `aborted`, or `fault` (section 3.2) |
| `termination_kind` | the `effective_session_outcome` as written on disk: `normal`, `aborted`, `faulted`, or none |
| `capture_outcome` | how *taking the data* ended, frozen when the spool commits its session-end record: `normal`, `aborted`, `faulted`, or `unknown` when no session-end record exists to freeze it (section 3.2) |
| `effective_session_outcome` | what the session actually ended as: `normal`, `aborted`, or `faulted` (section 3.2) |
| `finalization_status` | `not_started`, `running`, `failed_retryable` (with a category), `failed` (with a category), `succeeded`, or `abandoned` -- the second dimension of section 3.2, independent of the session outcome |

Both of these are legal and mean different things, which is exactly why the
state alone cannot carry them:

```text
closed + session_created = false            (nothing was ever recorded)
closed + session_created = true, sealed = true, complete = true
```

A caller MUST NOT infer the existence of a session, or its verdict, from
`closed`. An implementation that overloads `closed` to mean "sealed session on
disk" reintroduces the ambiguity this table exists to remove.

Transitions:

| Current | Operation or event | Next | Required effect |
| --- | --- | --- | --- |
| `created` | `prepare()` succeeds | `prepared` | plan fingerprint, queues, pools, and spool resources fixed; no allocation on the data path after this point |
| `created` | `prepare()` fails | `created` | no partial resource is retained and no session is created on disk; the object MAY be prepared again |
| `created` | `prepare()` fails **and its own rollback cannot produce that effect** | `failed` | the undo failed, or an artifact this attempt created could not be removed; the row above's effect is unavailable, so its promise is withdrawn rather than reported falsely. The recorder could not complete its own lifecycle, `resource_release_error` names what was left on disk, and reporting `created` here would offer a retry that the leftover -- sitting at the pathname the next `prepare()` must win with `O_EXCL` -- guarantees will fail |
| `prepared` | readiness gate passes | `ready` | every critical edge is attached and lossless; the spool superblock is committed |
| `prepared` | readiness gate fails | `failed` | **every prepared resource is released**; a spool holding no committed record is discarded, so this path normally leaves **no session artifact at all**; the runtime MUST NOT be armed or started |
| `ready` | `start()` | `recording` | the recorder accepts records from both planes |
| `recording` | graceful `stop()` | `draining` | latches terminal intent `normal`; stops accepting; drains what was accepted |
| `recording` | `abort()` | `draining` | latches terminal intent `aborted`; stops accepting; drains what was accepted |
| `recording` | recorder or runtime fault | `draining` | latches the primary fault, inhibits safety, terminal intent `fault` |
| `draining` | `stop()` or `abort()` again | `draining` | **the latched intent is not replaced**; the call waits for the shutdown already running and returns its status |
| `draining` | primary fault while draining | `draining` | the capture has not ended yet, so both `capture_outcome` and `effective_session_outcome` become faulted (section 3.2); `requested_terminal_intent` is unchanged |
| `draining` | drain completes within the bound | `stopped` | the spool session-end record is committed, which **freezes** `capture_outcome` (section 3.2); a latched fault makes the *session* faulted, not this recorder |
| `draining` | drain times out or the writer fails | `failed` | the session is incomplete; in-flight I/O retains its resources until close can reclaim them; the spool is available for recovery only after the writer has quiesced |
| `stopped` | `finalize()` | `finalizing` | the spool is read, never mutated |
| `finalizing` | finalization succeeds | `closed` | the NRF session is validated and the termination record is written |
| `finalizing` | the finalizer stops without sealing, and another attempt over the same bytes could succeed | `finalization_failed` | already-committed NRF data is not rewritten; the spool is retained; `finalization_status` becomes `failed_retryable` with its category, and the session outcome is **unchanged** (section 3.2). Only reachable once the finalizer has **actually stopped** |
| `finalizing` | the finalizer stops without sealing, and **no** attempt over the same bytes can succeed | `failed` | `finalization_status` becomes `failed` with its category. Already-committed NRF data is not rewritten and the session outcome is **unchanged** (section 3.2), exactly as above; what differs is that this recorder MUST NOT retry, so the state is terminal. The spool MUST be retained -- the finalizer read it and refused it, which makes it the input to diagnosis, repair, or quarantine (section 7). This is **not** `abandoned`: no operator decided anything |
| `finalizing` | a caller's wait expires | `finalizing` | the state does not move: the finalizer is still running, and the call returns with a timed-out indication |
| `finalization_failed` | `finalize()` retry | `finalizing` | resumes from the recorded finalizer progress; MUST NOT append a logical record twice; a retry that succeeds seals a `normal`, complete session if the capture dimension was clean (section 3.2) |
| `finalization_failed` | `close()` | `finalization_failed` | idempotent; releases nothing that a retry still needs |
| `finalization_failed` | `abandon_finalization()` | `failed` | the explicit decision to stop retrying; `finalization_status` becomes `abandoned`, the spool is retained, and the session stays unsealed |
| `stopped`, `finalizing`, `finalization_failed` | `stop()` or `abort()` | unchanged | the terminal intent was latched on the way out of `recording`; the call returns that shutdown's status and changes nothing |
| `closed`, `failed` | `stop()`, `abort()`, `close()` | unchanged | idempotent; the same status is returned |

`close()` is defined in every state, because a caller who is finished with a
recorder must not have to know which state it is in:

| Current | `close()` | Effect |
| --- | --- | --- |
| `created` | `closed` | nothing to release; `session_created` is false |
| `prepared`, `ready` | `closed`, or `failed` if something it owns could not be released | releases every prepared resource; discards the spool only if it holds no committed record, and otherwise retains it under section 7; no NRF session is produced |
| `recording` | `closed`, `finalization_failed`, or `failed` | performs a graceful `stop()` and then finalizes, exactly as if both had been called; the outcome is the *combination* of the two, so a clean stop still reaches a finalization outcome |
| `draining` | `closed`, `finalization_failed`, or `failed` | waits for the drain under the configured bound, then finalizes; likewise both stages decide the outcome |
| `stopped` | `closed`, `finalization_failed`, or `failed` | finalizes; `close()` never leaves a finalizable spool unfinalized silently |
| `finalizing` | `closed`, or **still `finalizing`** if the wait expires | waits for the running finalizer under a bound; MUST NOT cancel it silently, MUST NOT report `closed` while it runs, and MUST NOT report it failed merely because the caller stopped waiting |
| `finalization_failed` | `finalization_failed` | idempotent; giving up is `abandon_finalization()`, not a second `close()` |
| `closed`, `failed` | unchanged | idempotent |

`close()` from `recording`, `draining`, or `stopped` runs the capture shutdown
and then the finalization, so its outcome is decided by both stages and a
status MUST let a reader tell them apart -- they ask for different things:

| Why `close()` reported `failed` | What it says | Where it is reported |
| --- | --- | --- |
| the capture shutdown failed -- the stop failed, or the drain exceeded its bound | the *session* is incomplete; the spool is retained for the offline finalizer | the capture dimension: `capture_outcome`, `effective_session_outcome`, `primary_fault` |
| the finalizer refused the spool non-retryably | the capture dimension is untouched and MAY be clean; the spool is retained, and this recorder MUST NOT retry it | `finalization_status` = `failed` with its category |
| a resource or artifact this recorder owns could not be released | the object could not complete its own lifecycle; nothing about the session's verdict changed | `resource_release_error` |

A recorder that could not remove an artifact it owns MUST report that as a
release failure and MUST NOT carry the removal forward as work for a later
call: a terminal state is the statement that no further work is coming, and
`close()` is idempotent, so a second `close()` unlinking what the first one
could not would be a repeated call deciding something new. What was left behind
is an orphan for explicit cleanup; the next creator's `O_EXCL` refuses that
pathname rather than adopting it.

Rules this table is not allowed to leave open:

- **Legality.** `prepare()` is legal only in `created` and `start()` only in
  `ready`. `stop()` and `abort()` are a **state error** in `created`,
  `prepared`, and `ready` -- there is nothing to stop -- and are legal in every
  state from `recording` onward: the first one out of `recording` latches the
  terminal intent, and one arriving in `draining`, `stopped`, `finalizing`,
  `finalization_failed`, `closed`, or `failed` returns the status of the
  shutdown already latched without changing it. `finalize()` is legal only in
  `stopped`, `finalization_failed`, or as the resumption of an interrupted
  `finalizing`; `abandon_finalization()` only in `finalization_failed`;
  `close()` is legal in every state, per the table above. Every other
  combination MUST be rejected with a state error and MUST NOT be silently
  ignored.
- **Idempotence, stated exactly.** Once a terminal intent has been latched,
  repeated `stop()` or `abort()` calls are idempotent throughout **all**
  subsequent lifecycle states: each returns the same status, does not move the
  state, and does not alter a sealed session. `close()` is legal and idempotent
  in **every** state. Nothing a caller can reach by repeating a call may change
  what the recorder decided the first time. (Work already running may of course
  finish in between -- a second `close()` can observe a finalizer that has since
  completed. That is the background making progress, not the repeated call
  deciding something new.)

  Idempotence is deliberately *not* claimed for a first `stop()` or `abort()`
  before there is anything to stop: in `created`, `prepared`, and `ready` those
  are state errors, not silent no-ops, because a caller stopping a recorder that
  never started has a bug the contract should surface rather than absorb.
  `prepare()` and `start()` are not idempotent either; a second call is a state
  error.
- **Giving up is its own operation.** Because `close()` is idempotent, it can
  never be the thing that abandons a retryable finalization. A caller that wants
  to stop retrying calls `abandon_finalization()`, which is explicit, appears in
  the status, and is the only path from `finalization_failed` to `failed`. An
  API that made the second `close()` mean "give up" would make the first one a
  lie.
- **Waiting is not cancelling.** A bounded wait that expires MUST NOT change the
  recorder's state, and MUST NOT mark work failed that is still running. If a
  finalizer is still executing when a caller's wait expires, the state stays
  `finalizing` and the call returns a timed-out indication. `finalization_failed`
  is reachable only once the finalizer has actually stopped -- otherwise a
  retry could start a second finalizer over the same target, and no one would
  own the spool. A retry MUST be rejected while a finalizer is still running.
- **No restart.** A recorder that has left `recording` MUST NOT return to it. A
  session is single-use: recording again requires a new recorder and a new
  session identity. Reusing a session identity is forbidden.
- **Resource ownership.** Everything `prepare()` allocated -- queues, pools,
  spool handles, threads -- is owned by the recorder and MUST be released by
  exactly one transition into `closed` or `failed`. No path may leave a
  prepared-but-never-started recorder holding them, which is why the readiness
  failure above releases rather than merely refusing.
- **The first exit from `recording` latches the terminal intent.** Whichever of
  `stop()`, `abort()`, or a fault gets there first fixes the reason the session
  will end with. A later `stop()` or `abort()` arriving while the recorder is
  `draining` MUST NOT replace it: those calls only wait for the shutdown already
  in progress and return its status, and two callers racing MUST receive the
  same answer. This is not a theoretical concern -- one thread stopping while
  another aborts is an ordinary shutdown race, and without this rule the end
  reason on disk depends on scheduling.

  A **primary runtime or recorder fault** is the one thing that may still change
  the outcome after the intent is latched, because a fault is a fact about the
  session rather than a request from a caller. It escalates
  `effective_session_outcome` to faulted -- at any point before the session is
  sealed, including during finalization, where `capture_outcome` is already
  frozen and stays as it was -- without rewriting `requested_terminal_intent`,
  and no ordinary second API call may escalate anything. Section 3.2 states the precedence in full, and it is the only place
  that does.
- **Context manager.** Leaving the context normally performs a graceful stop and
  the close that follows it. Leaving it because of an exception aborts with the
  exception as the reason. The public facade expresses this as `abort()` followed
  by an idempotent `close()`. What is normative is the outcome -- an exception
  never ends a session as `normal` -- not the number of calls.
- **`ready` versus the runtime's `arm()`.** `ready` is the recorder's own
  precondition, and it is what the runtime's readiness gate consults. The
  runtime MUST NOT arm while a critical recorder is not `ready`. `arm()` remains
  a safety transition of the runtime and does not introduce a runtime state.
- **A fault does not skip the drain.** The fault path goes through `draining`,
  under the bounded shutdown timeout. A recorder fault is not permission to
  discard what was already accepted (section 4.2 step 4). A session that drains
  and finalizes successfully after a fault ends with the recorder `closed` and
  the session `faulted`.
- **Where each store closes.** The spool session-end belongs to
  `draining` → `stopped`. The NRF termination record belongs to
  `finalizing` → `closed`. They are different records in different stores and
  MUST NOT be conflated.
- **Finalization failure moves the recorder, not the data, and not the
  verdict.** It leaves every already-committed NRF transaction -- and the
  session's capture outcome -- exactly as they were (section 3.2). The partial
  NRF target MUST NOT be read as a finished session. Where it puts the
  *recorder* depends on one question, and only one: whether another attempt
  over the same bytes could succeed.
  - **A retryable failure** puts the recorder in `finalization_failed` and
    moves `finalization_status` to `failed_retryable`. The clean source spool
    remains **finalizable**: the finalizer MAY retry or resume after validating
    its recorded progress.
  - **A non-retryable failure** -- a spool whose committed prefix cannot be
    accounted for, an accounting identity a second pass reads the same way --
    puts the recorder in `failed` and moves `finalization_status` to `failed`.
    A retry reads the same bytes to the same conclusion, so this recorder MUST
    NOT offer one; the spool MUST be retained and handed to diagnosis, repair,
    or quarantine (section 7). Reporting such a failure as `failed_retryable`
    is forbidden: it asks a caller to retry what cannot succeed, and it
    contradicts the finalizer progress document, which records the same
    distinction and is what offline recovery reads to decide resumability.
  - `failed` and `abandoned` are different facts and MUST NOT be conflated.
    `failed` is the finalizer's technical conclusion that retrying is
    pointless; `abandoned` is an operator's explicit decision to stop retrying
    something that could still have been retried.
  Recovery is required only when the spool or the NRF committed prefix itself
  needs reconstruction (section 3.2). A finalization that failed because the
  target filled up, because publication failed, because a codec or write error
  hit, or because the finalizer was deliberately stopped, says nothing about the
  spool's health -- reporting it as corruption would send an operator hunting
  for damage that is not there. Those are the retryable cases, and they are the
  common ones; a non-retryable failure is the narrow case where the spool
  itself is what the finalizer objected to.
- **Who may finalize a retained spool.** A recorder in `finalization_failed`
  MAY retry its own finalization, because it still owns the spool and the
  finalizer progress. A recorder in `failed` MAY NOT: it failed during capture,
  abandoned its retries, or was refused non-retryably by the finalizer, its
  resources are gone, and any spool it left behind is handed to **offline**
  diagnosis, recovery, or finalization under the identity and progress rules of
  section 7. Being retained is not the same as being resumable, and only the
  recorded progress decides which: `running` and `failed_retryable` MAY be
  resumed; `failed` MUST NOT be, because the finalizer already read those bytes
  and refused them, so it goes to diagnosis, repair, or quarantine; `abandoned`
  MUST NOT be either, because an operator decided to stop, and only a new
  explicit decision restarts it. No recorder ever re-opens a spool it has
  released.
- **`failed` does not imply a spool.** A recorder that failed its readiness gate
  never wrote a committed record, so that path normally leaves no artifact at
  all -- nothing to retain, nothing to recover, and nothing for a finalizer to
  resume. Retention applies to a spool holding at least one committed record, or
  one being kept as evidence; it is not a property of the `failed` state itself.
- **Workers finished, finalizer still running.** The recorder MUST report
  `finalizing`. It MUST NOT report `closed`, MUST NOT report a termination
  kind, and MUST NOT report the session complete, because none of those are
  true yet.

The public `RecorderState` names every state in the table above. Recorder
failure is always ``failed``; ``faulted`` remains a session outcome and MUST
NOT be used for recorder lifecycle state.

### 3.2 Session outcomes

Six terms describe a recorded session. They are independent questions, not
points on one scale, and every implementation MUST be able to answer each one
separately.

**Cleanly terminated.** A termination record exists in the NRF session. Says
only that something sealed the session deliberately; says nothing about
completeness. `RecorderStatus.sealed` is this property.

Not being cleanly terminated does **not** by itself mean recovery is needed. A
spool that ended cleanly and is simply waiting to be converted is in a normal
lifecycle position, not a damaged one, and conflating the two would send an
operator hunting for corruption that does not exist. Four questions, answered
separately:

| Question | Meaning |
| --- | --- |
| `spool_ended_cleanly` | the spool carries its session-end record |
| `nrf_sealed` | the NRF session carries a termination record |
| `finalization_required` | there is a spool whose content is not yet in a sealed NRF session |
| `recovery_required` | reading the artifact needs a recovery pass before it can be treated as finished |

The combinations that matter:

```text
clean spool, no NRF yet     → finalization_required, recovery NOT required
                              (a recorder in `stopped`: ordinary, expected)
clean spool, partial NRF    → finalization_required, recovery NOT required
                              (`finalization_failed`: resume or retry it)
spool without session-end   → finalization_required AND recovery_required
NRF with a termination      → neither required
```

`recoverable` / `unrecoverable` below apply **only** to an artifact for which
`recovery_required` is true. Normal finalization is not recovery, MUST NOT be
reported as recovery, and MUST NOT emit a recovery report.

`finalization_required` is the artifact's question -- is there still a spool to
convert? `finalization_status` (below) is the recorder's answer about the
attempts -- has one run, is one running, did one fail retryably, did someone
give up? They are not the same field and MUST NOT be collapsed.

**Complete.** Every item that was producer accepted on either plane reached
stage 4, nothing required was rejected or lost at any stage, the session is
cleanly terminated, and its termination kind is `normal`.

`NrfReader.complete` is the *intended* reading of this property, but it is not
yet proof of it. What the reader evaluates today is exactly "a termination
record exists and its kind is `normal`", which is **necessary but not
sufficient**: the session carries no persisted acceptance or loss summary, so
nothing in the artifact establishes that each handoff's numbers agree
(section 5.1), that no queue rejected anything, or that no observer dropped
anything. The sufficiency
currently rests entirely on the *writer* -- `SessionRecorder` refuses to write a
`normal` termination once it has counted any loss -- so a session written by a
generic `NrfWriter`, or assembled by any other tool, can read `complete` without
having any of the evidence above. Closing that is open work (gap 11); until then
no code, doc, or test may treat `NrfReader.complete` as a verified completeness
proof for a session it did not itself write.

A session MUST NOT be marked complete after any of:

- required-data loss at any stage, including a frame sequence gap that no
  discontinuity explains;
- rejection of a required record at stage 1 on either plane;
- recorder queue saturation, data or control;
- an unrecovered spool-writer failure, or any writer failure that cost required
  data -- **not** an NRF finalization attempt that failed and later succeeded;
- runtime fault or recorder fault;
- explicit abort;
- abnormal process termination.

This rule has no exceptions and no configuration. A policy that continues
recording after loss changes what is recorded afterwards; it MUST NOT change
this verdict.

**Capture and finalization are two dimensions, and only one of them can end a
session.** A recording that captured everything and then failed to *convert* has
lost nothing; a full disk on the target is a fact about the target. Treating
that as a permanent verdict would make `finalization_failed` -- deliberately a
non-terminal, retryable state (section 3.1) -- a state from which nothing worth
having could ever be produced.

| Dimension | Values | What moves it |
| --- | --- | --- |
| Capture outcome (`capture_outcome`) | `normal`, `aborted`, `faulted`, `unknown` | the runtime, the recorder, and the spool: what happened while data was being taken. Frozen at the spool's session-end record; `unknown` if the process died before there was one |
| Finalization status (`finalization_status`) | `not_started`, `running`, `failed_retryable`, `failed`, `succeeded`, `abandoned` | attempts to convert a committed spool into a sealed NRF session |

`effective_session_outcome` is neither of these two: it starts equal to
`capture_outcome` and is the only one of the three that a post-capture discovery
can escalate (the fault-precedence rules below).

Rules:

- A runtime fault, a recorder queue failure, or a **spool** writer failure is a
  capture fault. It ends the session, and both `capture_outcome` and
  `effective_session_outcome` are `faulted`.
- A failed **NRF finalization attempt** moves `finalization_status` to
  `failed_retryable`, or to `failed` when no attempt over the same bytes could
  succeed, and nothing else. It is **not** a primary recorder fault, it does not
  escalate `effective_session_outcome` under the precedence rules below, and it
  does not by itself make the session incomplete. That is true of both values:
  `failed` says a retry cannot help, not that the capture was damaged.
- A retry that succeeds over a clean spool -- with every accounting identity,
  ledger reference, and payload extent verifying (section 5.1) -- produces a
  `normal`, complete session. The failed attempts are kept as provenance and
  diagnostics, and MUST NOT rewrite what the recording did.
- Only two things make a finalization failure fatal to completeness: a failure
  that lost or could not recover required data, and an explicit
  `abandon_finalization()`. Both are decisions or losses, both are stated on
  disk, and neither is a bare attempt count. They act differently: proven loss
  escalates `effective_session_outcome` to `faulted` and leaves `capture_outcome`
  as it was, so a session can be sealed as `faulted` while stating that the
  capture itself ran cleanly; `abandon_finalization()` moves neither outcome and
  leaves the session unsealed, because nothing was published to be judged.
- A session that has not been finalized at all is not complete either -- not
  because it is damaged, but because completeness is a property of a sealed
  session (stage 5). `finalization_required` is the field that says so
  (section 3.2's four questions), and it is not a defect report.

**Incomplete.** Cleanly terminated but not complete. The session states on disk
why: an `aborted` termination kind, a reason, and, where one exists, the fault
it points at. An incomplete session is a valid NRF session and is read with the
ordinary reader.

**Faulted.** The terminal outcome was *caused* by a fault: the end reason is a
runtime fault or a recorder fault, or the termination record references a
primary session fault. Faulted implies incomplete. Incomplete does not imply
faulted: an operator abort and a `CONTINUE`-policy loss are both incomplete
without a fault.

The **presence of a fault record does not make a session faulted**, and the two
MUST NOT be conflated:

- A *fault record* is data. A device dropout, an experiment error, a harvested
  native `FaultRecord` from a runtime that then recovered -- these are rows in
  the session, recorded because they happened, and a session full of them can
  still be complete. Recording one is an observation, not a verdict.
- A *session fault* is a terminal cause. It ends the session, is latched as the
  primary fault, and is referenced by the termination record.

A caller says "a fault ended this session" by aborting with that reason, not by
writing a fault row. An implementation MUST NOT derive `faulted` from the
existence, count, or severity of fault records.

**Fault precedence.** A fault can arrive *after* a caller has already latched
`aborted`, and it can arrive *after the capture itself is over* -- a finalizer
reading a committed spool can discover that required data is unrecoverable. What
was asked for, how taking the data ended, and how the session ended are three
different facts. They MUST be reported as separate fields, and no section of
this contract may define the outcome on its own:

| Field | Values | Meaning |
| --- | --- | --- |
| `requested_terminal_intent` | `normal`, `aborted`, `fault` | what the first exit from `recording` latched (section 3.1) |
| `capture_outcome` | `normal`, `aborted`, `faulted`, `unknown` | how *taking the data* ended. Frozen when the spool commits its session-end record, and never moved afterwards. `unknown` when there is no such record to freeze it -- see below |
| `effective_session_outcome` | `normal`, `aborted`, `faulted` | what the session actually ended as, and what the NRF termination kind states |
| `finalization_status` | `not_started`, `running`, `failed_retryable`, `failed`, `succeeded`, `abandoned` | how the attempts to convert a committed spool went (the second dimension above). `failed_retryable` and `failed` differ only in whether another attempt could succeed; `failed` and `abandoned` differ in who decided (section 3.2) |

The precedence rules, frozen:

1. The first `stop()` or `abort()` latches `requested_terminal_intent`.
2. No later ordinary API call changes it. Repeating `stop()` or `abort()` is
   idempotent (section 3.1).
3. A **primary runtime or recorder fault raised at any point before the session
   is sealed** -- during recording, draining, or finalization -- escalates
   `effective_session_outcome` to `faulted`. A fault is a fact about the
   session; it outranks a request. A **retryable NRF finalization failure is not
   such a fault**: it moves `finalization_status` and nothing else (see the two
   dimensions above). What does escalate during finalization is a fault in the
   capture dimension -- a spool that turns out to be unreadable, required data
   that cannot be recovered -- not an attempt that can be run again.
4. **`capture_outcome` and `effective_session_outcome` are separate fields
   because escalation after the capture is over has to be expressible.**
   `capture_outcome` starts as the outcome the capture path reached and is
   **frozen** at the spool's session-end record. `effective_session_outcome`
   starts equal to `capture_outcome` and escalates to `faulted` when
   finalization or recovery proves required data lost or the source
   unrecoverable. `termination_kind` on disk always equals
   `effective_session_outcome`. So the case the contract must be able to
   state -- a clean capture whose artifact turned out to be short -- is:

   ```text
   requested_terminal_intent = normal
   capture_outcome           = normal      (the run itself ended cleanly)
   effective_session_outcome = faulted     (proven incomplete before sealing)
   finalization_status       = succeeded   (a session was sealed, honestly)
   termination_kind          = faulted
   ```

   An implementation MUST NOT collapse these into one field. Reporting only
   `faulted` erases that the recording ran correctly; reporting only `normal`
   publishes a session that is known to be short.

   **`capture_outcome` is `unknown` when nothing froze it.** It is frozen by the
   spool's session-end record, and a process crash is defined by that record's
   absence (section 4.5): the next process cannot tell a crash just before a
   normal stop from a crash during an abort, after a runtime fault, or after a
   recorder fault whose row never committed. All four leave the same committed
   prefix. Recovery MUST therefore report `capture_outcome = "unknown"`
   (`capture_outcome_known = false`) and MUST NOT pick one of the three values
   as if the evidence existed. What recovery MAY still determine is the
   *effective* outcome, from what is on disk:

   ```text
   a committed primary fault row exists  → effective_session_outcome = faulted
   no fault evidence, no session-end     → effective_session_outcome = aborted,
                                           termination_origin = "recovery",
                                           source_session_end_present = false
   ```

   Neither case may ever be `normal` (section 4.5). The rule this encodes:
   **recovery may derive an effective outcome, but MUST NOT present it as a
   known capture outcome.** An artifact that reported `capture_outcome:
   "aborted"` for a crashed recorder would be stating that a caller asked to
   abort, which nothing on disk says.

   Both are on disk, and
   `capture_outcome` travels in the termination record's `extensions` alongside
   `requested_terminal_intent` (NRF v1's termination kinds are a closed enum,
   section 4.5).
5. Escalation does **not** rewrite the request or the capture. Both
   `requested_terminal_intent` and `capture_outcome` stay as latched and MUST be
   preserved as provenance in the session, so "aborted by the operator, then the
   writer failed" is not flattened into either half.
6. Escalation happens **whether or not the primary fault row itself reached a
   committed extent**. The outcome describes what ended the session, not what
   the recorder managed to persist about it. When the fault row could not be
   committed, the termination record MUST still declare kind `faulted` and MUST
   say that its primary fault row is missing -- through the same `extensions`
   mechanism section 4.5 uses for recovery provenance, because NRF v1's
   termination kinds are a closed enum. A session that quietly reported
   `aborted` because it failed to write the evidence would be hiding the failure
   twice.

Escalation is one-directional: `normal` may become `aborted` or `faulted`, and
`aborted` may become `faulted`. Nothing ever moves back down.

**Recoverable / unrecoverable.** These two are the answer to a *second*
question, and they are meaningful only once the first one has been asked:

- **Recovery required** -- must a recovery pass run before this artifact can be
  read as a finished recording? True when a spool has no session-end record, or
  when an NRF session's committed prefix has to be re-established. It is **not**
  true merely because finalization has not run yet: a clean spool awaiting
  conversion needs finalization, not recovery (see the table above). Nor is it
  true because finalization *failed* over a clean spool -- that target is
  unfinished, not damaged, and the answer is to resume or retry finalization. A
  cleanly terminated session needs neither, however incomplete or faulted it is.
- **Recoverable** -- *given that recovery is required*, can a deterministic
  committed prefix be established from what is on disk? For an NRF session this
  means the manifest and journal are readable and a prefix of committed
  transactions verifies. For a spool it means the superblock validates and at
  least the empty prefix is well defined -- a spool with a valid superblock and
  a wholly torn first transaction is recoverable to zero records, which is a
  legitimate answer.
- **Unrecoverable** -- no committed prefix can be established: the superblock or
  manifest is unreadable, identity does not match, or the format major version
  is unknown. An unrecoverable artifact MUST be preserved as evidence and
  reported as a failure. It MUST NOT be deleted, truncated, or repaired into
  looking usable.

A status API MUST report **not applicable** wherever `recovery_required` is
false, rather than `true` or `false`. Answering `true` would invite callers to
run recovery over artifacts that are merely unfinalized or already finished;
answering `false` would read as "this is beyond repair", which is the opposite
of the truth. This is the one place where the six terms are not fully
independent, and saying so here is cheaper than leaving each caller to guess.

### 3.3 Relationships between outcomes

```text
cleanly terminated ──┬── complete            (normal termination, nothing lost)
     recovery: n/a   └── incomplete ──┬── faulted
                                      └── not faulted  (abort, tolerated loss)

clean spool, not yet finalized ──┬── no NRF target yet  (the ordinary `stopped`)
     finalization required        └── partial NRF target (`finalization_failed`:
     recovery: n/a                                        resume or retry)

spool or session without a clean end ──┬── recoverable   → finalize as abnormal
     recovery: required                └── unrecoverable → preserve, report
```

Reading the diagram: the left column is one question (how did this end?), the
branches are others (was anything lost? did a fault end it?), and
recoverable/unrecoverable only ever apply to the bottom row. The middle row is
the one this contract most wants kept distinct: it is a healthy artifact with
work still to do, not a damaged one.

## 4. Lifecycle outcomes and ordering

Five ways a recording session ends. Each has a required ordering. "Recorder"
below means the critical recorder; noncritical observers are covered in
section 4.6.

### 4.1 Graceful stop

Requested by the operator, or reached because the source signalled end of
stream.

1. Stop new source production; the runtime stops accepting new items.
2. Let the already-accepted items flow through the fixed topology to completion.
3. Close each observer edge and **drain** everything already accepted by it.
4. Flush the recorder: close the open spool transaction, sync per policy, write
   the spool session-end record.
5. Stop and join every runtime thread.
6. Finalize to canonical NRF and write a `normal` termination when, and only
   when, nothing was lost.
7. Validate the NRF output before applying any spool retention policy.

Every producer-accepted item MUST reach stage 4 on this path. A graceful stop
that loses an item is a fault, not a graceful stop.

### 4.2 Runtime fault

A source, processor, consumer, actuator, watchdog, or safety failure.

1. **Inhibit actuator and safety-sensitive output immediately.** This precedes
   everything else, always.
2. Stop new source, processor, and actuator production.
3. Stop accepting new observer-dispatch items.
4. **Drain the items already accepted by the critical recorder edge.** They are
   the recorder's responsibility and were paid for; abandoning them destroys
   evidence about the fault that is often the only evidence there is.
5. Deliver the primary fault record and the abnormal terminal reason to the
   recorder through a bounded, allocation-free path.
6. Ask the recorder to commit, drain, and finish within a configured bounded
   shutdown timeout.
7. If finish times out or fails: cancel it, mark the recording incomplete, and
   release every runtime resource anyway.

The **primary fault MUST be preserved**. A fault raised while handling a fault
is secondary and MUST NOT displace it. If the recorder or the event sink itself
fails while receiving a fault, that failure MUST NOT be republished into the
same path; recursive fault publication is forbidden.

### 4.3 Recorder fault

Queue saturation, spool writer failure, plan violation, or an unusable
recorder at arm time.

- A recorder fault MUST enter the ordinary runtime fault path of
  section 4.2, including the safety inhibit.
- The runtime MUST abort or inhibit; a critical recorder that cannot record is
  not a degraded mode to continue in.
- The **first** failed or lost sequence number and a cumulative count MUST be
  latched in bounded status storage, and MUST remain readable even when the
  normal control queue is the thing that is full.
- The session MUST remain incomplete, and any committed prefix MUST remain
  recoverable and MUST NOT be discarded. It MUST NOT be marked complete.
- The one recorder fault with no session to keep is a failure at the readiness
  gate, before a valid superblock exists. The honest result there is **no
  session artifact at all**, reported as such. It MUST NOT be reported as a
  recoverable session, because there is nothing to recover, and a caller who
  goes looking for one has been misled.

### 4.4 Explicit abort

Requested by the caller. Not a fault: no fault record is invented.

1. Inhibit safety-sensitive output.
2. Stop production and stop accepting.
3. Drain what the critical recorder already accepted, exactly as in
   section 4.2. An abort is a decision to stop recording new
   data, not a licence to destroy data already handed over.
4. Write the spool session-end record with an abnormal reason.
5. Finalize to NRF with an `aborted` termination kind.

The session is incomplete, and it is not faulted **only if no primary runtime or
recorder fault occurs during the drain or the finalization**. A fault arriving
after the abort was latched escalates the effective outcome to `faulted` under
the precedence rules of section 3.2, while `requested_terminal_intent` stays
`aborted`; step 5 then writes the `faulted` kind. This section does not define
the outcome on its own.

### 4.5 Process crash

No ordering is available; the contract is about what the next process finds.

- Recovery MUST start from the last valid committed spool transaction.
- A record inside an incomplete or checksum-invalid transaction MUST NOT be
  promoted, ever, by any code path.
- A committed prefix without a spool session-end record MUST be finalized as
  abnormal and incomplete. It MUST NOT be finalized as normal.
- The distinction that makes this implementable: the **spool** session-end
  record MUST NOT be forged, and the **NRF** termination record MUST be
  written. They are different records in different stores.

  - *Forbidden:* inventing a spool session-end record, or treating the spool as
    though it had ended cleanly. The spool's committed prefix is the evidence
    and it is left exactly as found.
  - *Required:* the finalizer writes an NRF termination record, because an NRF
    session with no termination is unreadable as a finished recording and
    would strand the recovered data. That record MUST declare its own origin.

  A termination record written this way MUST convey three facts, and MUST do so
  **within NRF v1 as it is specified**. The `termination_kind` enum is closed --
  `normal`, `aborted`, `faulted` -- and NRF v1 requires readers to reject unknown
  enum values, so recovery MUST NOT invent a new kind. It uses the existing kind
  that is true (`faulted` when a committed fault caused the end, `aborted`
  otherwise, never `normal`) -- that kind is the *effective* outcome recovery
  derived, and the accompanying `capture_outcome` stays `unknown` because no
  session-end record froze it (section 3.2) -- and carries the rest under the
  journal record's
  `extensions` member, which the specification defines as an optional minor
  addition that older readers may ignore:

  ```json
  {
    "kind": "termination",
    "termination_kind": "faulted",
    "reason": "recovered from an interrupted session",
    "extensions": {
      "neurale.recovery": {
        "termination_origin": "recovery",
        "source_session_end_present": false,
        "recovery_reason": "process_crash",
        "capture_outcome": "unknown"
      }
    }
  }
  ```

  `termination_origin` distinguishes a termination the recorder wrote from one
  recovery reconstructed, and it MUST be present on both -- `"recorder"` in the
  ordinary case. Putting it in `extensions` keeps a recovered session readable
  by every existing NRF v1 reader; putting a new enum value in
  `termination_kind` would make those readers reject the session outright.
  The recovery implementation fixes the exact spelling of the extension keys.
  Introducing a new termination kind is a specification change with its own
  version policy, and this contract does not make one.
- Ordinary diagnosis MUST be read-only. Truncating an invalid tail MUST be an
  explicit, separately requested, auditable operation that emits a report.
- Finalization MUST be resumable and idempotent: interrupting it and running it
  again MUST NOT append the same logical record twice, and MUST NOT rewrite
  already committed valid NRF data.

### 4.6 Noncritical observers

Noncritical observers keep their existing best-effort semantics. They MAY be
cancelled immediately on abort, MAY drop under their configured policy, and
MUST NOT be forced to take the critical recorder's drain guarantee. The
critical-recorder requirements in this section apply only to an edge configured
as one.

### 4.7 Ownership at shutdown

- Every item accepted by the critical recorder edge MUST be either delivered or
  explicitly accounted for as lost before the recorder finishes.
- Successful storage close releases every frame, lease, queue, worker and file
  handle. Acquisition shutdown and storage reclamation are separate: a disk
  operation may outlive the configured drain timeout. Such a timeout MUST be
  reported as a fault, never as successful finalization.
- While disk I/O is outstanding, the recorder MUST retain the buffers, worker,
  spool and clock it references. No detached worker or early free is allowed.
  `queue_storage_release_deferred` and `worker_running` expose this condition.
  Explicit close is retryable after I/O returns. Destruction is the final
  ownership barrier and may wait for I/O; it is not a bounded shutdown API.
- A spool with an active writer MUST NOT be finalized, repaired or deleted.
- The runtime MUST NOT wait indefinitely for a stalled recorder. The bound is
  configured, and exceeding it is itself a reportable outcome.

## 5. Which records must survive

"Survive" means: present at stage 4 in the finalized NRF session, given that
the artifact is recoverable at all.

| Record | Graceful stop | Runtime fault | Recorder fault | Explicit abort | Process crash |
| --- | --- | --- | --- | --- | --- |
| Session identity, plan fingerprint, schemas, descriptors, clocks | required | required | required | required | required (in the spool superblock) |
| Frames accepted by the recorder, each with every block its plan records (section 1.1) | all | all accepted before the fault | committed prefix | all accepted before the abort | committed prefix |
| Discontinuities, each with its signal gaps (section 5.2) | all | all accepted before the fault | committed prefix | all accepted before the abort | committed prefix |
| Control records | all | all accepted before the fault | committed prefix | all accepted before the abort | committed prefix |
| Primary fault record | n/a | required | required | n/a | required if it was committed |
| First lost/failed sequence and loss counts | n/a | required | required | required if anything was discarded | required if committed |
| Terminal reason (`effective_session_outcome`, section 3.2) | `normal` | `faulted`, whether or not the primary fault row committed; a missing row is declared in `extensions` | same | `aborted`, or `faulted` if a fault arrived during the drain or finalization | `faulted` or `aborted`, never `normal`, with `termination_origin: "recovery"` in `extensions` |
| Session accounting summary (section 5.1) | required | required | required | required | required, rebuilt by the finalizer and marked `recovery_rebuilt` with `producer_acceptance_known: false` |
| Spool session end | required | required if the recorder finished within its timeout | required if the recorder finished within its timeout | required | absent, and MUST NOT be forged (section 4.5) |
| NRF termination record | required | required | required | required | required, written by the finalizer, declaring `source_session_end_present: false` |

The finalizer MUST NOT invent a `normal` termination for a session that has no
valid clean end, and MUST NOT write any NRF termination whose origin is
`recovery` without saying so in the record itself.

### 5.1 The session accounting summary

A session that does not carry its own accounting cannot be checked by a reader;
its completeness verdict rests entirely on the process that wrote it, which is
gap 11 today. This contract therefore requires the evidence to be **persisted in
the session**, and names who does it.

Every **summary-capable** session -- every session finalized by the native
 path, and every session written by any writer that implements this contract --
 MUST carry an accounting summary covering both planes. Readable sessions that
 contain no accounting summary use the `unverified_legacy` compatibility rule
 below instead.

The summary is stated **per handoff**, never as one total: the ladder's stages
are separated by handoffs that fail independently, and an item missing at one
stage was never present at the next.

Every counter below counts **items** in the sense of section 1.1 -- one accepted
data `StreamMessage` on the data plane, one control record on the control
plane. Frame, block, observation, row, and byte counts MAY accompany the
summary as diagnostics; they MUST be named distinctly and MUST NOT be
substituted into any identity.

```text
data plane
  runtime_accepted
  recorder_accepted
  spool_committed
  nrf_committed
  rejected_before_runtime_acceptance          never accepted by anything
  failed_between_runtime_and_recorder
  lost_between_recorder_and_spool
  lost_during_finalization

control plane
  control_offered                             optional; see the identities below
  control_accepted
  control_spool_committed
  control_nrf_committed
  control_rejected                            never accepted by anything
  lost_between_control_acceptance_and_spool
  control_lost_during_finalization

first failed or lost position      a tagged value, per plane; see below
rejected after close               per plane; API misuse, not a recording loss
                                   (section 1.3): it belongs to no identity below
                                   and MUST NOT change the session's verdict
termination origin                 "recorder" | "recovery"
accounting origin                  "recorder" | "recovery_rebuilt"
producer_acceptance_known          bool
```

**The persisted summary records facts, never verdicts.** `accounting_verified`
is deliberately **not** a stored field: it is a *judgement about* the stored
record, and a writer cannot make it, because layer 2 below checks the writer's
numbers against the artifact. Storing it would create two values that can
disagree -- an artifact asserting `accounting_verified = true` while the reader
in front of it derives `false` -- with no rule saying which one a caller sees,
and the schema and the reader could each implement a defensible but different
authority. The record therefore carries only `accounting_origin`,
`producer_acceptance_known`, the counters, and the identity/position fields
above; `accounting_verified` is **always derived by the current reader** from
both verification layers, on every read, and is never read back from the
artifact. A writer-side attestation MAY be added later as a distinctly named
field (`writer_accounting_attested`) if some producer needs to record what it
believed; it would be provenance, and MUST NOT feed the derived value.

A session produced by an offline or compatible non-spool writer reports no
stage-3 numbers and MUST NOT synthesize them. Its chains run from acceptance
directly to NRF committed on both planes, so the two spool-adjacent identities
below collapse into one per plane. Omitting an inapplicable stage is legal;
inventing one is not.

**The first failed or lost position is a tagged union, not one number.** The
ordinals of section 1.1 are assigned *at acceptance*, so an item refused before
that point never had one -- and those are exactly the items
`rejected_before_runtime_acceptance` and `control_rejected` count. A schema that
carried only the ordinal form could not name the very first rejection a session
suffered. Both forms are therefore required, and each carries its tag:

| Plane | When the item was accepted | When it was refused before acceptance |
| --- | --- | --- |
| data | `ordinal`: the global data-message ordinal | `producer_identity`: the pair `(message_kind, frame_sequence)` |
| control | `ordinal`: the global control submission ordinal | `producer_identity`: the pair `(kind, identity)` |

A reader MUST NOT compare an `ordinal` value with a `producer_identity` value,
or order one against the other: they are positions in different sequences. An
implementation MAY report one of each per plane -- the first loss after
acceptance and the first pre-acceptance rejection are different events, and
collapsing them into one field loses whichever happened second.

Rules:

- The summary MUST be written in the **same transaction that seals the
  session**, so a session cannot be sealed with accounting that disagrees with
  it or with none at all.
- **Verification has two layers, and only the second one stops trusting the
  writer.** A reader MUST perform both.

  *Layer 1 -- internal identities.* The summary must be arithmetically
  consistent with itself, per handoff, adjacent stages only:

  ```text
  data     runtime_accepted        = recorder_accepted       + failed_between_runtime_and_recorder
           recorder_accepted       = spool_committed         + lost_between_recorder_and_spool
           spool_committed         = nrf_committed           + lost_during_finalization

  control  control_accepted        = control_spool_committed + lost_between_control_acceptance_and_spool
           control_spool_committed = control_nrf_committed   + control_lost_during_finalization

  when control_offered is recorded
           control_offered         = control_accepted        + control_rejected
  ```

  *Layer 2 -- consistency with the artifact.* Layer 1 alone proves nothing about
  the recording: a writer that claimed
  `runtime_accepted = recorder_accepted = spool_committed = nrf_committed = 100`
  with every loss term zero satisfies every identity while the session holds 99
  frames. The reader MUST therefore also check the committed counters against
  what is actually in the session:

  ```text
  data nrf_committed        ↔ the committed extent of the frame and
                              discontinuity ledgers, and every frame's
                              references to its blocks, payload, and index
                              resolving within a committed extent
  control_nrf_committed     ↔ the committed extent of the control record sets
  the accounting record     ↔ committed in the same sealing transaction as the
                              termination record, in the journal
  ```

  A counter that exceeds what the artifact contains MUST be reported as a
  failed verification, never reconciled by trusting the number.
- **What `accounting_verified` actually means, and what it does not.** The
  field was called `completeness_verified` in an earlier draft and is renamed
  here on purpose: it answers "was the persisted accounting checked?", not "is
  the completeness verdict settled?", and one name for both questions made a
  recovery-rebuilt session -- incomplete beyond doubt, with no accounting worth
  checking -- impossible to express. Both layers together establish exactly
  three things:

  ```text
  the persisted accounting is internally consistent;
  its committed counters match the committed artifact;
  the accounting and the termination were sealed in one transaction.
  ```

  That is what `accounting_verified: true` MUST be documented as -- as a result
  the reader computed just now from the two layers, not as something the
  artifact told it. It is **not**
  an independent proof that everything the producer handed over was counted. A
  recorder that dropped one accepted frame and wrote every counter one lower --
  acceptance and committed alike -- satisfies layer 1 and layer 2 both, and no
  reader can see the difference, because the only record of that acceptance was
  in the process that lost it. Producer-side acceptance is attested by the
  writer; the artifact can only be checked against itself.

  So a reader MUST NOT describe this as auditing the writer or as zero trust,
  and MUST NOT let `accounting_verified` be read as "nothing was ever lost".
  What layer 2 removes is the *specific* failure of a session claiming more than
  it contains -- which is the failure gap 11 is about, and the one a generic
  `NrfWriter` product hits. What it does not remove is a recorder miscounting
  its own inputs; that is caught by parity testing, not by a reader.
- **A rejection MUST NOT be added to a stage the item never reached.**
  `control_rejected` and `rejected_before_runtime_acceptance` count items that
  were never accepted at all, so they appear in no identity above except the
  optional `control_offered` one. Writing
  `accepted = committed + rejected + lost` for a plane is a contract violation:
  it puts items inside a count they were refused entry to, and no reader can
  satisfy it.
- **Completeness needs the rejections checked separately.** The identities alone
  cannot decide the verdict. A session is complete only if, in addition to every
  identity holding, **every** loss term and **every** recording-loss rejection
  term is zero -- including `rejected_before_runtime_acceptance` and
  `control_rejected`, which appear in no identity and would otherwise go
  unchecked -- and the session is cleanly terminated with kind `normal`.
  `rejected after close` is the one refusal counter that is *not* part of this:
  it is caller misuse of a session that had already ended (section 1.3). A finalized session with a
  nonzero `lost_during_finalization` is incomplete by definition.
- **A rebuilt summary cannot claim acceptance.** Recovery MUST rebuild what it
  can from the recovered committed prefix and MUST mark the result
  `accounting_origin: "recovery_rebuilt"`, `producer_acceptance_known: false`,
  `accounting_verified: false`; the session then reads `verified_incomplete`
  with `complete = false` -- a verdict its abnormal termination and missing
  session-end establish on their own, without any accounting to check.
  Acceptance counts that cannot be derived from the surviving prefix MUST be
  reported as unknown, never filled in from what survived. What recovery MAY
  state is what it found:

  ```text
  known_spool_committed
  known_nrf_committed
  known_surviving_prefix
  ```

  This limit is structural, not an implementation shortcut: an item the producer
  handed over and the worker had not yet committed when the process died leaves
  nothing on disk to count, so a rebuilt summary that printed a precise
  `runtime_accepted` would be inventing it. Because an abnormally recovered
  session can never be complete anyway, `complete = false` stays a reliable
  verdict -- what MUST NOT be claimed is that the original acceptance accounting
  was recovered.
- **Sessions without an accounting summary, and the one public completeness
  surface.** Such a session has no accounting to check and MUST NOT be presented as verified
  complete. Naming *two* acceptable shapes for that -- a flag alongside a
  boolean `complete`, or a third verdict -- is what would let two readers ship
  different answers to the same question. Frozen: there is **one** verdict, the
  boolean is derived from it and never the reverse, and whether the accounting
  was checked is a **separate** field:

  ```python
  CompletenessVerdict = Literal[
      "verified_complete",
      "verified_incomplete",
      "unverified_legacy",
  ]

  completeness_verdict: CompletenessVerdict | None
  complete: bool | None
  accounting_verified: bool
  ```

  All three are **derived by the reader that is answering the call**, from the
  artifact plus the two verification layers of section 5.1. None of them is
  stored in the accounting summary and read back (see the persisted-fields rule
  above), so an older artifact never pins a newer reader's judgement.

  ```text
  verified_complete    → complete = True    accounting_verified = True
  verified_incomplete  → complete = False   accounting_verified = True or False
  unverified_legacy    → complete = None    accounting_verified = False
  None (not yet determinable)
                       → complete = None    accounting_verified = False
  ```

  Why the verdict and `accounting_verified` are two fields: **incompleteness has
  cheaper proofs than completeness.** A crash-recovered session with no
  session-end record, or any session whose termination kind is `aborted` or
  `faulted`, is incomplete on that evidence alone -- there is nothing to
  reconcile and, after a crash, no trustworthy accounting to reconcile with
  (the rebuilt-summary rule above). Binding the verdict to
  `accounting_verified: true` would leave `complete = false` with an unverified
  accounting unrepresentable, which is precisely the state every recovered
  session is in. Completeness has no such shortcut: `verified_complete` requires
  `accounting_verified: true`, because it is the claim that nothing was lost and
  only the checked accounting can support it.

  `completeness_verdict = None` is the fourth case, and it is not a defect
  report: **completeness is a property of a sealed session** (stage 5), so it
  does not exist yet for a recorder in `created`, `prepared`, `ready`,
  `recording`, `draining`, `stopped`, `finalizing`, or `finalization_failed`,
  nor for a session whose finalization was abandoned -- there, `finalization_status`
  (`abandoned`) is what says why, and no verdict will ever appear. `None` MUST
  NOT be reported as `unverified_legacy`: one means "there is no sealed session
  to judge", the other "there is one, and it carries no evidence".

  The compatibility question this settles, stated as the case that occurs:
  **a session without an accounting summary whose termination kind is `normal`
  reads `unverified_legacy` with `complete is None`.** It MUST NOT
  read `complete is True`, and it MUST NOT raise -- the session is perfectly
  readable, it is the completeness claim that is unavailable. What the artifact
  does say is exposed separately as `legacy_termination_normal`: a fact about
  the termination record, not a completeness verdict. Calling that fact
  `complete` would return the API to a boolean plus a flag every caller must
  remember to combine, which is the shape this rule exists to remove, and it
  would assert exactly what section 3.2 says the artifact cannot support. Such
  a session terminated `aborted` or `faulted` is **not** `unverified_legacy`:
  its own termination record proves it incomplete, so it reads
  `verified_incomplete` with `accounting_verified: false`.

  Two consequences to note: `complete` is `bool | None`, so every caller that
  assumes a bool needs the third case; and because `None` is falsy,
  `if not reader.complete` keeps refusing such sessions unchanged, while
  `reader.complete is False` is what distinguishes "verified incomplete" from
  "unverifiable". Sessions without accounting summaries MUST stay readable.

The obligations above are met by the current system: the
`native-session-accounting-v1` schema and its ledger define the summary; the
recorder maintains the per-handoff item counters in bounded prepared storage and
writes the summary bound to the termination transaction, cross-checking it
against what was actually written before publishing; diagnosis and recovery
verify the summary and rebuild it during recovery, and `NrfReader.complete`
implements the reader-side verification, checking both layers -- the internal
identities and the summary against the committed ledgers, record sets, and
payload extents; and the public status API exposes one `CompletenessVerdict`,
`complete` typed `bool | None` and derived from it, `legacy_termination_normal`
kept separate, while keeping pre-summary sessions readable as `unverified_legacy`.

This document is the single source for these contracts; a rule restated
elsewhere is a rule with two versions.

### 5.2 Discontinuities need a ledger of their own

A discontinuity is one data-plane item (section 1.1), so `nrf_committed` on that
plane can only be checked against the artifact if the artifact contains one
record per discontinuity *message*. The standard per-stream NRF discontinuity
records do not satisfy that, and the mismatch is not a naming detail:

- One native `Discontinuity` fans out into **one record per affected stream**,
  so the count on disk is a stream count, not an item count.
- When `signal_gaps` is empty, the frame-level gap is written against **every**
  recorded stream, so the fan-out factor is not even derivable from the gaps.
- Gaps for signals the recording plan did not select are dropped entirely.
- The per-stream record keeps `expected_sample_index`, `actual_sample_index`,
  and a derived missing-tick count, and does **not** keep each gap's
  `expected_device_tick`, `actual_device_tick`, or `SignalGapFlags`.

From that projection one cannot recover how many messages there were, how many
gaps each carried, in what order, which were frame-level and which per-signal,
or what the dropped signals' gaps said. Exact replay of a discontinuity is
therefore impossible from the per-stream records, and the explicit
discontinuity requirement would degrade into re-inferring gaps from sample
indices.

The native path MUST therefore write two additional ledgers, alongside the frame
and block ledgers, preserving at least:

```text
native-discontinuities-v1
  data-message ordinal            the identity from section 1.1
  session ID
  previous frame sequence
  actual frame sequence
  reason                          the message-level GapReason
  first signal-gap ordinal        null when signal-gap count is zero (a
                                  frame-level gap); required otherwise, with
                                  the gap range [first, first + count)
  signal-gap count                zero is meaningful: a frame-level gap
  runtime_accepted_host_time_ns   host time at runtime acceptance, a uint64
                                  matching the native HostTimeNs, stamped
                                  where the ordinal is assigned; see below

native-signal-gaps-v1
  global gap ordinal
  owning discontinuity's data-message ordinal, and the index within it
                                  the same identity section 1.1 gives the
                                  message; there is no second "discontinuity
                                  ordinal" numbering to keep in step
  signal ID                       including signals the plan does not record
  expected and actual sample index
  missing samples
  expected and actual device tick
  reason                          the per-gap GapReason
  flags                           SignalGapFlags, not a derived approximation
```

Rules:

- The standard per-stream discontinuity records remain, unchanged, for readers
  that consume streams. The ledgers are additive, and neither is derived from
  the other at read time.
- A discontinuity's `nrf_committed` counts messages, never fanned-out records.
- Exact frame replay MUST reject a session whose discontinuity ledgers are
  missing, exactly as it does for the frame ledgers (section 8.1).
- Gaps for unselected signals MUST be preserved in the gap ledger even though no
  stream records them, because "which signals were affected" is part of what the
  discontinuity said.
- **A discontinuity needs a time of its own, and today has none.** The native
  `Discontinuity` carries no timestamp
  (`cpp/include/neurale/streaming/discontinuity.h`), and the per-stream record's
  `time_ns` is *placement* time. The native recorder persists the
  `runtime_accepted_host_time_ns` assigned at acceptance. Without a time on the message,
  recorded pacing (section 8.7) has nothing to schedule a replayed discontinuity
  against. The ledger therefore carries `runtime_accepted_host_time_ns`, stamped
  at **runtime acceptance** -- the same point that assigns the data-message
  ordinal (section 1.1), so one action produces both halves of the item's
  identity -- and carried unchanged through the spool into the ledger. It is
  recording-envelope provenance: it says when the recording path took the
  message over, not when the underlying gap occurred, and MUST NOT be described
  as the latter. The alternative -- deriving a pacing time from the target
  frame's host time -- was rejected as the *stored* answer: it is recoverable at
  replay time anyway, and storing a derived value would leave no way to tell it
  from a measured one.

The two schemas are defined by the native path; the recorder stamps
`runtime_accepted_host_time_ns` where it assigns the ordinal and writes the
ledgers; diagnosis and recovery verify them and rebuild their accounting during
recovery; and exact replay consumes them.

## 6. Durability policies

Durability is parameterized and MUST be stated exactly. No implementation, doc,
docstring, or status field may promise more than the selected policy delivers.
It is an attribute of a committed extent, not a stage of the ladder
(section 1.2): the spool and the NRF session each have their own policy, their
own committed extent, and their own durable extent within it.

Three policies are defined for the native spool (the format fixes their names;
these are their semantics):

- **buffered** -- after the required superblock/identity sync performed before
  readiness, runtime records are written without another explicit sync.
  Runtime records survive a process crash to the extent the operating system's
  page cache does. Guarantees **nothing** against power loss or kernel panic.
- **checkpoint sync** -- an explicit sync at checkpoint boundaries. Everything
  before the last successful checkpoint survives power loss; everything after
  it has the `buffered` guarantee.
- **transaction sync** -- an explicit sync as part of every transaction commit.
  A transaction reported as committed survives power loss. This is the only
  policy under which "spool committed" implies "durable".

Rules:

- A successful enqueue MUST NOT be described as durable under any policy.
- The policy in force MUST be recorded in session status and in benchmark
  output, because a throughput number without its durability policy is not a
  measurement.
- Tests MUST NOT assert a guarantee stronger than the policy under test.
- The default critical recorder uses a continuously appended file beside its
  NRF target on Linux and Windows. Only capture queues and transaction staging
  are preallocated; total disk extent is not reserved or locked in RAM.
  `buffered` remains the default, with the same process-crash/page-cache
  boundary and no power-loss guarantee. Cancellation capability is reported,
  not required at readiness; storage shutdown follows section 4.7.
- The private bounded mapped test store supports Linux tmpfs and Windows local
  storage under `buffered`. Linux rejects ordinary disk-backed paths because
  background writeback can make an ordinary mapping fault through the
  filesystem again after readiness. Windows preallocates the complete local
  file, expands the process working-set bound, maps it, and requires
  `VirtualLock`; the Windows contract then guarantees subsequent access to the
  locked range incurs no page fault. Both platforms writable-prefault and lock
  every page before prepare, then append only by copying into resident pages.
  Superblock commit performs a real mapping and file sync before readiness
  (`msync`/`fsync` on Linux, `FlushViewOfFile`/`FlushFileBuffers` on Windows);
  no sync occurs on the running buffered transaction path. Windows remote paths
  and any allocation, mapping, locking, or sync mismatch are readiness refusals,
  never fallback. A clean close flushes and trims the reservation. Process death
  leaves the named spool for ordinary diagnosis and repair; this is not a
  power-loss guarantee, and Linux tmpfs does not persist across reboot.
- NRF finalization keeps the NRF durability rules; the spool policy does not
  extend to it and vice versa.
- A durable extent MUST be reported as an extent -- how far synchronization has
  been acknowledged -- and never as a count of items that "reached a durable
  stage". The two stores' extents MUST be reported separately.

Honest statement of what NRF v1 gives today, as implemented: journal records
are flushed and `fsync`ed individually, and cache files are written through
`fsync` plus atomic rename, so a committed transaction is **atomic and
verifiable**. Chunk payloads are staged and renamed but are not themselves
`fsync`ed before the commit record, so a power loss can leave a committed
transaction whose object is missing or short. That case is *detectable* --
the prepare record carries each object's SHA-256 and byte length -- and
`diagnose_session` plus `recover` are what turn it back into a committed
prefix. NRF commit therefore means "atomic and detectably verifiable", not
"the bytes are on the platter". Any stronger claim about NRF durability
requires an implementation change, not a documentation change.

## 7. Artifact retention

Two artifacts have lifetimes: the private spool and the canonical NRF session.

**The spool MAY be deleted** only when all of:

1. finalization produced an NRF session;
2. that session was validated by `NrfReader` and diagnosis after finalization;
3. the configured publication step succeeded;
4. the configured retention policy asks for deletion.

**The spool MUST be retained** when finalization failed, validation failed,
publication failed, the finalizer was interrupted, or the policy asks for
retention. The default is `delete_after_validated_finalization`: remove the
spool and plan sidecar only after all four conditions hold. Explicit `retain`
keeps them for debugging. Cleanup failure is reported separately from successful
publication. The canonical output is one compressed `.nrf` file under the
[package envelope](https://github.com/RanXingchen/pyneurale/blob/main/specifications/nrf/package-v1.md); conversion may
temporarily require spool, a metadata workspace and package space together.
The stopped-spool converter writes immutable payload objects directly into the
temporary package. Generic offline writes and explicit repair may still use a
full uncompressed workspace. The final cross-check performs the full payload
hash pass once per attempt, including resumed publication attempts.

One narrow exception, and it is not really an exception: **an empty spool** --
superblock only, no committed record -- MAY be discarded when a recorder is
closed from `prepared` or `ready` (section 3.1). There is nothing to
reconstruct, so no reconstruction input is being deleted. A spool holding even
one committed record is never empty in this sense, and the retention rules above
apply to it in full.

**The spool MUST be quarantined**, not deleted, when it is unrecoverable, or
when an explicit repair truncated an invalid tail. Quarantine preserves the
evidence and emits a report naming what was moved and why.

**A spool MAY be resumed** when its superblock validates, its session ID and
plan fingerprint match the finalization target, and the recorded finalizer
progress identifies the last consumed transaction. Any mismatch of session ID,
plan fingerprint, output format version, or committed extents MUST reject the
resume rather than reconcile it.

**An NRF session MUST NOT be deleted by recording, finalization, or replay
code.** Deleting a recorded session is a user action.

**Ordinary NRF reading and replay MUST NOT mutate the source session.** Opening
a session, reading it, diagnosing it, building a replay image from it, and
replaying it are all read-only. Mutation happens only through `recover` or an
equally explicit repair call, and only when the caller asked for it.

## 8. Replay semantics

**Three** modes, with deliberately different promises. The mode MUST be selected
explicitly, MUST be visible in the replay result, and the mode a result reports
MUST be the mode that was requested -- no request is ever answered by a
different one.

| Mode | Requires | Promises |
| --- | --- | --- |
| `exact_frames` | the native ledgers **and** a plan that covered every block of the frames it reconstructs | the original frame and block topology |
| `recorded_projection` | the native ledgers, over a session recorded under a partial plan | emitted blocks exactly as recorded, with their native provenance; a projected message set and a replay-run frame sequence, both reported |
| `stream_frames` | a compatible committed per-stream block index | a documented synthesis, not the original |

A partial-plan session MUST NOT be replayed as `exact_frames`. The request fails
with a clear error naming `recorded_projection` as the applicable mode; it is
never silently answered with a projection, exactly as a missing ledger is never
silently answered with a synthesis. This is decided here rather than left to
the implementer, because the alternative -- letting the implementer choose
between failing, succeeding, and quietly changing the result's mode -- is three
different public APIs wearing one name.

### 8.1 Exact frame replay

Reconstructs the original native frame and signal-block topology: the same
frames, carrying the same blocks, in the same order, with the same sample
indices, device ticks, observation times, and clock-sync snapshots.

The `RecordingPlan` is the one authoritative prepared statement the recorder and
finalizer share, and it freezes the complete native schema -- every signal,
feature set, and unit, with every field `StreamSchema::equivalent()` compares --
so the original `StreamSchema` is rebuilt from the plan, not guessed. Its
fingerprint covers that schema *content*, not just a numeric schema id, so two
schemas that share an id but differ in dtype, layout, channel count, rate, clock
domain, or block limits are different plans; storage sizing and session identity
are in the plan but not the fingerprint. On-disk plan shape is fixed by the
native-replay extension README.

- Requires the native frame, block, discontinuity, and signal-gap ledgers
  (sections 5.2 and 1.1). Discontinuities are part of the topology being
  reconstructed, not context around it.
- A session without those ledgers MUST reject exact replay with a clear error.
  It MUST NOT silently degrade to synthesized replay.
- **It requires full block coverage.** A `RecordingPlan` MAY record a subset of
  the streams a frame carries; that is a legitimate configuration, and it is
  also the one case where "the same frames, carrying the same blocks" cannot be
  delivered -- the unrecorded blocks are not on disk in any form. This contract
  reserves the term **exact frame replay** for a plan that covers every block of
  the frames it reconstructs. A session recorded under a partial plan MUST
  declare that in its ledger metadata, and its replay is the separate
  `recorded_projection` mode below -- never `exact_frames` with a caveat.
- **Replay-time stream selection is subject to the same rule, and the rule is
  the conservative one.** `exact_frames` requires `ReplayConfig.selected_streams`
  to be **empty (meaning every recorded stream) or exactly equal to the recorded
  stream set**. Any proper subset MUST be rejected with a clear error naming
  `recorded_projection` -- including a subset that happens to cover every block
  inside the requested range. A full-coverage session does not stay
  `exact_frames` merely because its plan was complete; requesting two of its
  four streams is a projection, and answering it as `exact_frames` would let the
  one mode that promises the original topology return something else.

  The alternative was considered and rejected: coverage could be evaluated
  against the requested `message_range` (section 8.4), so a subset that omits
  nothing *within that range* would still be exact. It is implementable, but it
  makes the legality of a mode depend on the range, so the same session and the
  same `selected_streams` would be `exact_frames` for one range and an error for
  the next, and a caller could not tell which without inspecting the ledgers.
  A whole-session rule needs no range-dependent coverage proof and no
  explanation. If range-scoped exactness is ever wanted, it is a new mode with
  its own name, not a relaxation of this one.

### 8.2 Recorded-projection replay

The same reconstruction as section 8.1, over a session whose `RecordingPlan`
recorded only some of the blocks its frames carried. What it promises is
narrower than "the recorded part, exactly", and the wording matters because
parity tests are written against it:

**Recorded block payloads and their native provenance are preserved exactly for
every emitted item. The emitted message set, the contents of an emitted
discontinuity, and the runtime frame sequence are an explicit projection.**

Emitted blocks are byte-for-byte what was recorded, with their sample indices,
device ticks, observation times, clock-sync snapshots, and original data-message
ordinals unchanged. What is *not* preserved, and is stated here rather than
discovered in the rules below: frames that project to no block are omitted,
irrelevant discontinuities are omitted, an emitted discontinuity drops the gaps
of unprojected signals, and frame sequence numbers are the replay run's own. The
mode is also not *silent* about any of that -- it reports coverage and every
omitted item -- so neither "exact" nor "silent" is an accurate one-word summary,
and neither may be used as one.

- Requires the same native ledgers as `exact_frames`. It is not a fallback for
  a session that lacks them.
- The result MUST report which streams the plan covered, so "absent" is
  distinguishable from "lost".
- It MUST NOT be described, compared, or tested as reproducing the original
  frame. A projection that claimed frame fidelity would be the exact confusion
  section 8.1 exists to prevent.
- **Requesting `recorded_projection` for a full-coverage session is legal**, and
  full coverage promises exactly one thing: with the complete recorded stream
  set selected, **no frame and no signal block is omitted because of
  `RecordingPlan` coverage**. It does not promise a complete message set.
  Discontinuities may still be omitted by the requested-range, signal-relevance,
  and boundary-representability rules below -- a range that begins at a
  discontinuity, or one whose target frame lies outside the range, omits it
  under full coverage exactly as under a partial plan. And it is still this
  mode, so the emitted frame sequence is the replay-run numbering below rather
  than the recorded one. Parity with `exact_frames` therefore holds for the
  emitted frames, their order, their block payloads, and their provenance --
  **not** necessarily for the complete discontinuity set, and not for header
  sequence numbers. The mode reported is the one requested. The reverse --
  `exact_frames` over a partial-plan session, or over a stream subset
  (section 8.1) -- is an error.

**What a projection emits when a frame projects to nothing.** A partial plan, a
stream selection, or both can leave an original frame with **zero** blocks to
emit -- frame 11 carried only signal B, and B is not being replayed. This is not
an edge case to be discovered late; it is the ordinary consequence of the
mode existing, and the native contract already forbids the obvious answer:

- `StreamSchema` rejects a schema with no signals
  (`cpp/src/streaming/schema.cpp`).
- `FrameValidator::validate` returns `block_count_mismatch` for
  `frame.blocks.empty()` (`cpp/src/streaming/frame_validation.cpp`).

So an empty frame cannot be emitted, and the remaining choices -- drop it, forge
a block, or invent a topology-marker message -- differ in what they claim, not
just in how they are coded. Frozen:

- A projected frame carrying **no** emitted block MUST NOT be emitted. The
  streaming contract is not widened to admit empty frames for replay's benefit,
  and no placeholder or synthetic block is ever manufactured to keep a frame
  alive. A replay-only marker message was considered and rejected: it would
  enlarge the generic streaming contract that every native consumer compiles
  against, to describe something only one replay mode can produce.
- Every item that *is* emitted keeps its **original data-message ordinal**
  (section 1.1). Ordinals are not renumbered to close the holes -- the gaps in
  the ordinal sequence are the honest record of what was projected away.
- **The replay result MUST list what it dropped, in one field that can hold
  both kinds of item.** A caller must be able to tell "frame 11 projected to
  nothing" from "frame 11 never existed"; the ledgers hold both answers, and a
  projection that stayed silent would force the caller back to the artifact to
  find out. An earlier draft called the field `omitted_frame_ordinals`, which a
  discontinuity cannot go into: a discontinuity has no frame ordinal, and
  leaving the schema to choose between stuffing one in, adding a second
  unfrozen field, or inventing a union produces implementations that disagree.
  Frozen as a list of tagged entries:

  ```text
  omitted_data_messages
    data_message_ordinal      uint64        the section 1.1 identity; the only
                                            field present on every entry, and
                                            the only one that orders the two
                                            kinds against each other
    kind                      frame | discontinuity
    original_frame_sequence   uint64|null   the omitted frame's own sequence;
                                            null for a discontinuity
    omission_reason           no_projected_blocks
                            | no_projected_signals
                            | no_preceding_emitted_frame
                            | target_frame_not_emitted
  ```

  The data-message ordinal is required on every entry because it is the one
  identity both kinds share (section 1.1); `original_frame_sequence` is
  provenance for readers that think in frames, and MUST NOT be used to order
  entries.
- **`omission_reason` is single-valued, so its precedence is frozen.** One
  discontinuity can satisfy several reasons at once -- no preceding emitted
  frame, *and* an unemitted target, *and* no projected signal. Leaving the
  choice open would make the field implementation-defined and break the
  deterministic replay-image bytes the replay builder depends on. For a
  **discontinuity**,
  the first matching reason wins, in this order:

  1. `no_preceding_emitted_frame`
  2. `target_frame_not_emitted`
  3. `no_projected_signals`

  A **frame** has only one reason, `no_projected_blocks`. Two rules keep
  eligibility and reporting from contradicting each other:

  - **Eligibility** is decided by the relevance and boundary tests frozen later
    in this section. Failing any one of them omits the message.
  - **The reason is chosen afterwards**, over *all* conditions that hold, using
    the precedence above -- not by whichever test happened to fail first. So an
    irrelevant discontinuity is reported as `no_projected_signals` **only when
    neither `no_preceding_emitted_frame` nor `target_frame_not_emitted`
    applies**. A signal-irrelevant discontinuity that is also the first item of
    the requested range is reported as `no_preceding_emitted_frame`.

  The precedence is a reporting rule only: it selects which reason is recorded,
  and never changes whether the item is omitted.
- **The list is scoped to the request, not to the session.**
  `omitted_data_messages` contains only source data messages whose original
  data-message ordinal lies **inside the requested `message_range`** and which
  the `recorded_projection` rules excluded. A message outside the range was
  never requested, so it is not an omission and MUST NOT appear -- "not
  requested" and "requested and dropped" are different answers, and conflating
  them would make the metadata of a small range grow with the size of the
  session. For an **empty** range, both the emitted items and
  `omitted_data_messages` are empty. A range with no explicit bounds covers the
  whole session, so the two readings coincide only there. This scoping is part
  of the replay-image fingerprint: a sub-range's metadata is a function of the
  sub-range alone, which is what makes sub-range results cacheable and
  independently reproducible.
- `recorded_projection` therefore does **not** claim to preserve the original
  data-message topology. It preserves *order* and *identity* over the items it
  emits. Section 8.2's promise is stated that way deliberately, and no
  documentation, test, or metric may restate it as topology preservation.
- **Discontinuity relevance is decided by an explicit rule, not per
  implementation** -- and relevance is only the *first* of two independent
  tests. A discontinuity is **signal-relevant** when it is frame-level
  (signal-gap count zero -- it describes the stream as a whole, and no signal
  selection makes it irrelevant), or when at least one of its signal gaps names
  a projected signal. An irrelevant one is omitted -- with which *reason*
  decided later, by the precedence below, since a discontinuity that is
  irrelevant can also be unrepresentable. **Frame-level discontinuities are always
  signal-relevant, but remain subject to boundary representability** -- the
  second test, frozen later in this section, which can still omit them. An
  emitted discontinuity carries only the gaps of projected signals; the gaps it
  does not carry stay in `native-signal-gaps-v1` (section 5.2), where the full
  message remains recoverable.

**Frame sequence in a projection is a replay-run number, not the original.**
Omitting a frame leaves a hole in the original sequence, and the generic
continuity tracker reads a hole as a real gap: `ContinuityChecker` compares
`frame.header.sequence` against its `expected_frame_sequence` and, on every
frame that jumps ahead, emits a `SignalGap` with reason `frame_sequence_gap`
(`cpp/src/streaming/continuity.cpp`). Replaying frames 10 and 12 under their
original sequences would therefore hand the processor a discontinuity that never
happened -- resetting filters, windows, and decoder state, and changing the very
results the projection was replayed to compute. A deliberate projection would be
delivered as a runtime data loss. Frozen:

- `exact_frames` emits the **original** `FrameHeader.sequence`. Nothing is
  omitted there, so no hole exists and none is invented.
- `recorded_projection` emits a **contiguous replay-run frame sequence**, and
  keeps the original as provenance (`original_frame_sequence`) next to the
  unchanged original data-message ordinal. The ordinals stay original because
  they identify recorded items; the frame sequence is renumbered because it is
  what the consuming runtime interprets as continuity.
- **The numbering is fixed, not merely contiguous, and it is local to one
  request.** The first frame emitted by a replay run, image, or range has
  `FrameHeader.sequence` **0**, and each subsequent emitted frame is exactly one
  higher. "Contiguous" alone would have left the base to the implementer, and
  The replay builder requires deterministic replay-image bytes and indexes: two
  builders that
  started at 0 and 1 would produce different images, different
  sequence-addressed fault injection, and different re-recorded output from the
  same input.
- **What that determinism does and does not promise.** The mapping is a pure
  function of the source session, the mode, the stream selection, and the range;
  a reset restores exactly the same mapping, and re-requesting the identical
  configuration reproduces it byte for byte. Nothing about *when* the run
  happened enters it. But **a different `message_range` is a different replay
  run**, and the same original frame may legitimately carry a different
  replay-run sequence in each: replaying items A, B, C numbers B as 1, while
  requesting only B and C numbers B as 0. Both are correct; per-request local
  numbering and a range-invariant sequence cannot both hold, and this contract
  keeps the local one because it is what makes a single run's bytes
  reproducible. Anything that must stay stable across ranges MUST use
  `original_frame_sequence` or the data-message ordinal (section 1.1), which are
  the cross-range identities; no test, index, or metric may key on the
  replay-run sequence across differing ranges.
- The generic continuity tracker MUST NOT be taught about projection omissions.
  That would push recording/replay semantics into the streaming core, against
  the dependency rule this contract works under, and would weaken gap detection
  for live runs to serve one replay mode.
- This is a further reason the mode does not claim the original topology: its
  frame numbering is honestly its own, and only `omitted_data_messages` plus
  the provenance fields tie it back.

**A projected discontinuity is emitted only when it has a preceding emitted
frame and its target frame is emitted.** The heading is stated that way because
the rule does *not* require the frame the recorded `previous_frame_sequence`
names to be emitted: that field is re-derived below from the last frame this run
actually emitted, so an omitted original predecessor is not by itself a reason
to omit. `Discontinuity` carries `previous_frame_sequence` and
`actual_frame_sequence` as required values, not optional ones, and consumers --
runtime, observer accounting, downstream processors -- act on
`actual_frame_sequence`. Under a projection there are three ordinary situations
where no emitted frame can fill them, and they MUST NOT be left unspecified:

| Situation | What is missing |
| --- | --- |
| the run or range begins at a discontinuity | no frame has been emitted yet, so `previous_frame_sequence` has nothing to name |
| the frame the discontinuity precedes projects to zero blocks | the frame `actual_frame_sequence` names is not emitted |
| the discontinuity is inside the range but the frame it precedes is not | same, for a range boundary rather than a projection hole |

Frozen, as one rule covering all three -- and it is the **second** of the two
tests a discontinuity must pass. The full decision, in order:

```text
1. Signal relevance          (frozen earlier in this section)

   frame-level discontinuity   relevant = true
   per-signal discontinuity    relevant = at least one gap names a
                                          projected signal

2. Boundary representability  (frozen here)

   emitted only if all of:
     - relevant
     - a frame has already been emitted in this run
     - the frame named by the original actual_frame_sequence is
       itself emitted in this run
```

Failing **either** test omits the message into `omitted_data_messages`. The two
tests are independent: relevance is about *which signals* were projected,
representability about *which frames* this run emits, and passing one says
nothing about the other. Both tests are evaluated for every omitted message, and
only then is the single `omission_reason` chosen by the precedence frozen above
-- the tests decide *whether*, the precedence decides *what is reported*.

- A discontinuity is emitted **only if** it is signal-relevant, at least one
  frame has already been emitted in this run, *and* the frame it precedes -- the
  one its original `actual_frame_sequence` names -- is itself emitted in this
  run. Otherwise it is **omitted** and recorded in `omitted_data_messages` with
  whichever of `no_preceding_emitted_frame`, `target_frame_not_emitted`, and
  `no_projected_signals` comes first in the frozen precedence among the
  conditions that actually hold.
- For an emitted one, `previous_frame_sequence` is the replay-run sequence of
  the last frame emitted before it, and `actual_frame_sequence` is the
  replay-run sequence of the frame it precedes. Intervening frames that were
  omitted appear in `omitted_data_messages`; they do not silently shift what
  these two fields mean.
- Both original values are preserved as provenance beside them. They are
  provenance *only*: no consumer may read them as runtime continuity, because
  they name a numbering this run does not use.
- **A dangling discontinuity is never emitted**, and no synthetic boundary frame
  is invented to give one a target. Either would hand the consumer a message
  about frames it will never see.
- Test 2 applies to **frame-level** discontinuities (signal-gap count zero)
  exactly as to per-signal ones. Being frame-level settles test 1 permanently
  -- such a message is always signal-relevant -- and settles nothing about
  test 2: it still describes the boundary before one specific frame, and if that
  frame is not emitted, the boundary does not exist in this run. And a
  discontinuity emitted before any frame would
  reach a consumer whose continuity tracker has no expectation to compare it
  against -- it carries no information there, and would only misreport the
  start of a run as a gap.
- `exact_frames` needs none of this. It renumbers nothing and omits nothing, so
  it emits the recorded values verbatim -- including, at a range boundary, a
  reference to a frame outside the requested range, which is the recorded truth
  rather than a constructed one.

### 8.3 Synthesized stream replay

Builds frames from the standard per-stream committed block index. Legitimate and
useful, but it is a reconstruction, not the original.

- **It requires a compatible committed per-stream block index**, and MUST NOT be
  described as available for every NRF session. The index is optional --
  `StreamProvenanceConfig.block_index` can be turned off, and a session assembled by a
  generic `NrfWriter` need not have one at all. A session without a compatible
  index MUST reject `stream_frames` with a clear error, and MUST NOT synthesize
  frames from anything else instead. *Compatible* means the index carries, per
  committed block, the sample extent needed to locate its rows --
  `sample_index_start`, `sample_count`, `row_offset` -- **`host_received_ns`**,
  which the ordering rule below requires, and **`frame_sequence`**, which the
  discontinuity mapping below requires. A per-stream NRF discontinuity record
  names the frame after the gap by `actual_frame_sequence`, and the block index
  carries no `segment_id` or `target_discontinuity_id` to find that block by, so
  without `frame_sequence` the builder cannot tell which block a discontinuity
  precedes -- and a frame-level gap's `actual_sample_index` is null, so there is
  nothing to fall back on. Both are minimum columns, not optional provenance.
  Compatibility block indexes already carry `frame_sequence`, so requiring it
  for every `stream_frames` run avoids a conditional rule that admits the
  column only where a discontinuity is present. The remaining compatibility
  columns (`last_sample_index`, `device_tick_start`) are provenance: their
  absence lowers reported fidelity without making the index incompatible.
- It MUST NOT be described as reproducing the original frames, and its results
  MUST NOT be compared against an exact-replay expectation as if they were.
- What it can carry is bounded by what that index stores; section 8.6 states the
  difference from the ledger-based modes, which is larger than it looks.

Everything below is **public replay behavior**, not an implementation detail:
two implementations that chose differently would deliver different frames from
the same session under the same request. It is frozen here, in full, and no
task description restates it.

**Frame construction.** One committed block becomes exactly one synthesized
frame, carrying that one stream's one signal. Blocks are never merged -- not
within a stream, and not across streams -- and never split.

- **The run declares one union `StreamSchema` covering every selected stream,
  and every synthesized frame carries that one `schema_id`.** An earlier draft
  had each frame declare only its own signal, which the existing runtime cannot
  consume: `NativeStreamRunner` is constructed with one `StreamSchema` for its
  whole lifetime (`cpp/include/neurale/streaming/runtime.h`), and
  `FrameValidator::validate` returns `schema_changed` for any frame whose
  `schema_id` differs from it (`cpp/src/streaming/frame_validation.cpp`). A run
  interleaving two streams would have been rejected on its second frame. No
  extension of the generic frame contract is needed for the union form: the
  validator already accepts a frame whose blocks are a **subset** of the
  schema's signals -- it rejects only an empty block list or more blocks than
  the schema has signals -- so a one-block frame under a many-signal schema is
  already legal.
- Each NRF stream is one recorded signal with its own companion block index
  (`<stream_id>.blocks`), so a synthesized frame carries exactly one block; the
  union schema says which signals the run may carry, not which ones each frame
  does.
- Grouping by the index's recorded `frame_sequence` was considered and
  **rejected**: it would re-create the original frame topology, assembling
  multi-signal frames out of streams that were committed independently and may
  not have all committed, against the one-frame-per-block rule this mode is
  defined by. `frame_sequence` is a required minimum column above, but only for
  discontinuity placement -- it is not used to reconstruct frame topology, and
  the recorded value is carried as provenance (section 8.6).
- Cross-stream merging is rejected for a further reason: two streams need not
  share a rate, a clock domain, or block boundaries, so merging requires an
  alignment policy that no recorded fact supplies.

**Synthesized schema and signal IDs.** `stream_frames` is defined not to
require the native ledgers, so a compatible session may carry only the NRF
string stable IDs (`stream_id`, `clock_id`, `feature_set_id`, `unit_ids`) and
no native numeric IDs. But a synthesized frame and its union `StreamSchema` must
still carry native IDs, and -- for a feature stream -- the `SignalSchema` and
`FeatureSetDescriptor` must cross-reference by those IDs
(`SignalSchema.feature_set_id` and `.clock_domain`, `FeatureSetDescriptor.id`,
`.unit_ids`, and `.source_stream_id`; `cpp/include/neurale/streaming/schema.h`).
A `SignalId` alone is not enough: native validation rejects a feature signal
whose `feature_set_id` is 0, a `feature_set_id` with no matching
`FeatureSetDescriptor`, or a descriptor that names a `UnitId` with no
`UnitDescriptor` (`cpp/src/streaming/schema.cpp`). The whole synthesized
registry is therefore frozen here, not left to the implementer:

```text
- each native ID namespace is filled independently from its NRF descriptor's
  string IDs. SchemaId is fixed at 1 for the one union schema; every other
  namespace sorts its IDs by UTF-8 byte order and numbers them from 1 (0
  reserved for unavailable / absent):
    SchemaId        fixed 1 for the one union schema
    SignalId        selected stream_id
                    plus source_stream_id referenced by a selected
                    feature stream (provenance-only, see below)
    ClockDomainId   clock_id named by a selected stream
    FeatureSetId    feature_set_id named by a selected feature stream
    UnitId          unit_id named by a selected feature stream's units
- a referenced-but-unselected source receives a deterministic provenance-only
  SignalId so that FeatureSetDescriptor.source_stream_id can name a nonzero
  mapped id even when the source is not in the emitted signal array; it does
  not add a SignalSchema and is not emitted. Lexical order applies over the
  full namespace, so the selected set does not start at 1 when an unselected
  referenced source sorts before it -- e.g. with selected feature
  "z-feature" and unselected source "a-source", UTF-8 byte order assigns
  a-source = 1 and z-feature = 2;
- every cross-reference uses the mapped native ID: SignalSchema.clock_domain,
  SignalSchema.feature_set_id, FeatureSetDescriptor.id,
  FeatureSetDescriptor.unit_ids, and FeatureSetDescriptor.source_stream_id all
  point at the assigned IDs above, never at the strings;
- ChannelSetId, CalibrationId, and ReferenceId are not promised by this mode:
  synthesized signals carry 0 for all three, and the achieved-fidelity metadata
  reports them unavailable, the same way it reports clock-sync unavailable
  (section 8.6). Deterministic mapping for them is a later task, not a
  replay-builder choice;
- the full string -> native ID mapping for every namespace is recorded in the
  replay metadata;
- a reset or rebuild of the same (session, selected streams, stream_ranges)
  reproduces the identical mapping.
```

Byte-order sorting makes every namespace independent of insertion order, so
two runs over the same selected set produce the same IDs and the same schema; a
`stream_id` that appeared twice would be a duplicate key in `stream_ranges`,
which section 8.4 already rejects. Where the source NRF already carried a
`native_id`, it is provenance only -- the synthesized IDs are re-derived from
the strings, so a recorder that assigned different `native_id` values cannot
change the replay image.

**Global ordering: a deterministic k-way merge, not a sort.** Ordering is
defined as a merge over per-stream cursors, and MUST NOT be stated as a
comparator over all blocks at once. Those are not the same rule, and the
comparator form has no solution: with `A0` at host 100, `A1` at host 10, and
`B0` at host 50, "committed order within a stream always wins" demands
`A0 < A1`, while "ascending host time across streams" demands
`A1 < B0 < A0` -- a cycle, so no total order exists. Frozen:

1. every stream keeps its **committed block-ordinal order**, and holds a cursor
   at its first not-yet-emitted block;
2. at each step, only the streams' current **head** items are compared, and the
   smallest tuple `(host_received_ns, stream_id, block_ordinal)` is emitted --
   `stream_id` compared byte-wise;
3. only that stream's cursor advances; the run ends when every cursor is
   exhausted;
4. a discontinuity occupies the same head position as the block it precedes and
   is emitted first (see interleaving below).

This never reorders a stream against its own commit order, is total by
construction, and depends on nothing but the recorded values -- so it is
reproducible without any assumption that recorded host times are monotonic
within a stream. `host_received_ns` is a minimum index column precisely because
step 2 cannot run without it.

**Runtime frame sequence.** Synthesized frames carry **one run-wide, zero-based,
contiguous** `FrameHeader.sequence`: the first frame emitted by the run is 0 and
each subsequent frame is one higher, across all selected streams in emission
order. Per-stream numbering was drafted first and is wrong for this runtime:
`ContinuityChecker` keeps a single `previous_frame_sequence_` for the frame
stream it validates (`cpp/src/streaming/continuity.cpp`), so a second stream
restarting at 0 is reported as `duplicate_frame_sequence`, and further
interleaving as `frame_sequence_regressed`. The "per-stream continuity tracker"
that numbering assumed does not exist. The recorded `frame_sequence`, the
`stream_id`, and the `block_ordinal` are provenance and carry the per-stream
identity instead; synthesized discontinuities reference the same run-wide
numbering (below). Everything else is as in section 8.2: the numbering is local
to the request, and a different `stream_ranges` request is a different run.

**Reset determinism.** The emitted item sequence, the run-wide numbering, and
the reported metadata are a pure function of (session, mode, selected streams,
`stream_ranges`). A reset reproduces the run exactly (section 8.10).

**Selected-stream interaction.** `stream_frames` accepts **any** non-empty subset of the
recorded streams -- one stream is enough, and the empty set is rejected
(section 8.4). This is deliberately *not* section 8.1's
conservative rule: that rule exists because an exact frame must carry all of its
blocks, and a synthesized frame makes no such claim -- per-stream reconstruction
is the mode's purpose. A selected stream without a compatible index is an error
(above), never a silently skipped stream. Which streams a run replays is decided
by **one** surface, frozen in section 8.4: the keys of `stream_ranges`, with
`selected_streams` absent or exactly equal to them.

**Discontinuity source and transformation.** Synthesized replay does **not** read
`native-discontinuities-v1` -- it is defined not to require the native ledgers.
Its source is the standard per-stream NRF discontinuity/segment records.

- Each per-stream record becomes one discontinuity message. Records from
  different streams are **never merged** back into one message: the evidence
  that they came from a single recorded message lives in
  `native-discontinuities-v1` (section 5.2), which this mode does not read, so
  merging would be a guess. One recorded message that fanned out over three
  streams therefore reappears as three messages here, and the result MUST NOT
  be described or counted as the recorded discontinuity set.
- **A synthesized discontinuity is a global runtime barrier, not a
  stream-scoped reset.** Generic streaming hands every discontinuity to the
  whole processor chain: `NativeStreamRunner::handle_discontinuity` forwards
  it to the `LinearProcessorChain`, which calls `handle_discontinuity` on
  every stage (`cpp/src/streaming/runtime.cpp`,
  `cpp/src/streaming/linear_processor_chain.cpp`). Today's processors do not
  inspect `signal_gaps`; they reset their entire state on any discontinuity
  (`SosFilterAdapter::reset`, `ResamplerAdapter::reset`,
  `KalmanDecoderAdapter::clear_stream_state`), so a discontinuity that
  originated from one stream resets state that may have been tracking another.
  `signal_gaps` records which data the discontinuity affects; it provides
  **no** selective processor reset. True per-stream state reset would require
  extending the processor contract, which is out of scope for this mode.
- Frame references are the **run-wide** replay-run numbers above, and the
  boundary-representability rule of section 8.2 applies unchanged: a
  discontinuity is emitted only if a frame has already been emitted in this run
  and the block it precedes is itself emitted. Otherwise it is omitted and
  reported. Because the discontinuity is a global barrier,
  `previous_frame_sequence` is the run-wide sequence of the **immediately
  preceding emitted frame** regardless of which stream that frame belonged to,
  and `actual_frame_sequence` is the run-wide sequence of the target frame the
  discontinuity precedes. `signal_gaps` carries only the gaps of the
  originating synthesized stream. So for `A0(seq 0) -> B0(seq 1) -> [A gap] ->
  A1(seq 2)`, the A discontinuity reports `previous_frame_sequence = 1`,
  `actual_frame_sequence = 2`, `signal_gaps = [A]`: the barrier sits between
  the globally emitted frames, not between A's frames.
- Its pacing timestamp is the per-stream record's own `time_ns`, which is
  **record placement time, not underlying gap time**. Current native sessions
  carry the host time assigned at runtime acceptance; compatibility sessions
  may carry the host time of the last frame before the gap. The native
  `Discontinuity` itself carries no event timestamp (see section
  8.7 and gap 13). The value MUST be labelled as placement time wherever it is
  reported.

**Frame/discontinuity interleaving.** A stream's discontinuity is emitted
immediately before the first frame carrying the block that starts the segment
after the gap, and it takes that frame's position in the global order, sorting
before it. Two discontinuities at one position keep their committed record
order.

**Achieved-fidelity metadata.** Because the mode reconstructs, the result MUST
say what it reconstructed *from*. Per selected stream:

```text
synthesized_fidelity
  block_index_columns_present   list of column names actually present
  clock_sync_available          bool   false for a compatibility block index
                                         without clock-sync fields (section 8.6)
  descriptor_metadata_available   bool   false: ChannelSetId,
                                         CalibrationId, ReferenceId are
                                         synthesized 0 (section 8.3)
  frame_construction            "one_frame_per_block"          v1 fixed
  ordering_key                  "original_host_received_ns"    v1 fixed
  frames_emitted                uint64
  blocks_emitted                uint64
  discontinuities_emitted       uint64
```

Omissions are reported too, but **not** through `omitted_data_messages`: that
field is keyed by the data-message ordinal of section 1.1, and this mode has no
ledgers and therefore no such ordinals. It reports `omitted_stream_messages`,
each entry naming `stream_id`, `kind` (`frame`\|`discontinuity`), the
`block_ordinal` or committed discontinuity-record position it came from, and a
reason drawn from `no_preceding_emitted_frame` and `target_block_not_emitted`.
The two fields are never populated by the same run, and neither may be
substituted for the other. Neither carries an injected omission: a fault-injected
skip is reported by section 8.11's `injected_effects`, which is a record of what
the *run* did, not of what the projection could not represent.

### 8.4 Range selection

`ReplayConfig` can ask for part of a session. **The unit of a range is not the
same in every mode**, and a single `committed range` field cannot express the
difference: the ledger-based modes are sequences of data messages, while
synthesized replay is per-stream sample data with no message identity at all.
Two fields, and a mode accepts exactly one of them:

| Field | Unit | Valid in |
| --- | --- | --- |
| `message_range` | `[first_data_message_ordinal, end_data_message_ordinal)`, half-open, in the global data-message ordinal of section 1.1 | `exact_frames`, `recorded_projection` |
| `stream_ranges` | per stream, a tagged `StreamReplayRange` -- see below | `stream_frames` |

Rules:

- Passing `stream_ranges` to a ledger-based mode, or `message_range` to
  `stream_frames`, MUST fail with a clear error. It is not translated: a
  message ordinal cannot be derived from a per-stream sample position without
  the ledgers that the synthesized mode is defined not to require.
- A `message_range` boundary MUST fall **between** data messages. A request that
  starts inside a frame, part-way through a frame's blocks, or inside a
  discontinuity is illegal and MUST fail. Blocks of one frame are never split
  across the boundary, and a discontinuity is atomic (section 1.1).
- Boundaries are **never silently widened** to the nearest legal position. A
  replay that quietly expanded a range to the enclosing frame would return data
  the caller did not ask for and report it under the requested range.
- An empty range is legal and yields no items; a range beyond the committed
  extent is an error, not an empty result, because those are different facts.
- Under `recorded_projection` the range is expressed in **original** ordinals
  (section 8.2 keeps them), so a range is stable regardless of which streams the
  projection emits. Frames that project to nothing inside the range are reported
  in `omitted_data_messages` exactly as elsewhere.

`stream_ranges` is **tagged**, and its first version deliberately admits one
unit. "A sample, observation, or block range" is not implementable: it does not
say which unit a given range is in, whether two streams may use different ones,
or what a range starting mid-block would even produce.

```python
StreamReplayRange(
    unit="block_ordinal",   # the only unit v1 accepts
    start=...,              # inclusive
    stop=...,               # exclusive
)
```

- `unit` is required and explicit. v1 accepts **`block_ordinal`** only.
  `sample_index` and `observation_index` are reserved names and MUST be
  rejected with an error saying so, not silently reinterpreted, until a task
  defines what slicing means. All ranges in one request MUST share a unit.
- A `block_ordinal` is the position of a committed block within that stream's
  committed block index, counted from zero in committed order. The range is
  half-open, so `stop` is exclusive and `start == stop` is a legal empty range.
  `start > stop`, or a `stop` beyond the committed extent, is an error.
- **A range names whole blocks and never splits one.** This is why v1 stops
  here: with no partial block, a synthesized frame carries a committed block
  exactly as stored, and its sample count, sample indices, starting device tick,
  and original host time are the stored index columns unchanged rather than
  values a builder had to recompute for a fragment. Section 8.3's frozen
  ordering and frame-construction rules still decide how blocks become frames,
  and the synthesized frame sequence is a run-wide replay-run number as it is
  there.
- A stream absent from `stream_ranges` is **not replayed**. Absence is not
  shorthand for its full extent -- one spelling per intent.

**One selection surface per mode**, so no caller maintains the same stream set
twice:

| Mode | What selects the streams | Empty means |
| --- | --- | --- |
| `exact_frames`, `recorded_projection` | `selected_streams` | every mode-eligible stream: the whole recorded set for `exact_frames` (section 8.1 admits nothing between), every stream the plan covered for `recorded_projection` |
| `stream_frames` | the **keys of `stream_ranges`** | rejected; replay nothing by giving one stream an empty range |

- Under `stream_frames`, `selected_streams` MUST be absent or exactly equal
  to the key set of `stream_ranges`. A disagreement is an error, never a silent
  precedence rule, and the empty-means-everything reading of the ledger-based
  modes does **not** carry over: `stream_ranges` MUST name at least one stream.
  The run declares one union `StreamSchema` over the selected streams (section
  8.3), and `StreamSchema` rejects a schema with no signals
  (`cpp/src/streaming/schema.cpp`), so a run with zero selected streams has no
  legal schema and cannot build a `NativeReplaySource`. An empty replay is
  expressed by giving one stream an empty range -- e.g.
  `stream_ranges = {"stream-a": [5, 5)}` -- which selects one stream, yields no
  blocks, and reaches EOF immediately, rather than by a `stream_ranges` with no
  keys.
- Under the ledger-based modes, `stream_ranges` is already rejected (above), so
  `selected_streams` is the only surface there.

### 8.5 Corrupt and incomplete sessions

- The default MUST reject a corrupt or incomplete session.
- An explicit allow-incomplete mode MAY replay the valid committed prefix. It
  MUST expose an abnormal terminal result rather than ending as if the data ran
  out naturally.

### 8.6 Timestamp provenance

The single most confusable part of replay. Three distinct quantities:

| Quantity | Origin | Rule |
| --- | --- | --- |
| Original recorded host time | the recording run | preserved as **provenance**; MUST NOT be overwritten, and MUST NOT be reinterpreted as current-run timing |
| Current replay-run host time | generated by the replay runtime | what watchdogs, deadlines, and dwell limits use |
| Recorded timeline | derived from recorded inter-message deltas | what recorded pacing follows |

Rules:

- `host_received_ns` on a replayed frame is **current-run** time. The original
  value is carried separately.
- A recorded `valid_until_ns` is stale by construction and MUST be cleared or
  recomputed by default. Preserving an original deadline is allowed only as an
  explicit test or fault-injection behavior, never as a default.
- Recorded pacing MUST use the recorded timeline frozen in section 8.7. Mixing
  clock domains without an explicit conversion is forbidden.
- Sample indices, device ticks, and observation times are recorded data and are
  replayed unchanged in every mode.
- **Clock-sync fidelity is not the same in every mode**, and MUST NOT be stated
  as if it were:

  | Mode | What is preserved |
  | --- | --- |
  | `exact_frames` and `recorded_projection`, from the native ledgers | the complete `ClockSyncSnapshot`: `device_tick_reference`, `host_time_reference_ns`, `device_tick_rate`, `uncertainty_ns`, `clock_domain`, `generation`, `flags` |
  | `stream_frames`, from a per-stream block index | only what that index stores. The compatibility index shape is `frame_sequence`, `sample_index_start`, `last_sample_index`, `sample_count`, `row_offset`, `device_tick_start`, `host_received_ns` -- sample positions, the device tick each block starts at, and the original host arrival time as provenance |

  A block index of that shape carries **no clock-sync snapshot at all**. A
  synthesized replay MUST NOT invent one, MUST NOT reuse the session-level clock
  registration as if it were the per-block snapshot, and MUST report the
  snapshot as unavailable rather than approximate. Any timing derived from the
  declared rate and the stored device tick is a derivation and MUST be labelled
  as one.

### 8.7 Pacing

A replay source has exactly **three** pacing modes, and they are mutually
exclusive -- one field, one value, no combinations:

| Mode | When the next item is emitted |
| --- | --- |
| `as_fast_as_possible` | as soon as the consumer takes the previous one; no waiting of any kind |
| `recorded` | at its position on the recorded timeline, scaled by `speed_factor` |
| `step` | only when the caller grants a permit (section 8.8) |

- **The recorded timeline is built from recorded host time, in one domain**, and
  from nothing else. Device ticks are per-domain and observation times are
  per-signal; mixing domains here is the error section 8.6 exists to prevent.
  Which host time, per item kind, is frozen -- a discontinuity has no arrival
  time of its own today, so this cannot be left as "the item's timestamp":

  | Item | Timeline value |
  | --- | --- |
  | frame, any mode | its original `host_received_ns` provenance (section 8.6) |
  | discontinuity, `exact_frames` and `recorded_projection` | `runtime_accepted_host_time_ns` from `native-discontinuities-v1` (section 5.2) -- recording-envelope provenance, not the instant the gap occurred |
  | discontinuity, `stream_frames` | the per-stream record's `time_ns`, which is **record placement** time: today's recorder places it at the last frame before the gap |

  Every one of these MUST be reported under the name that says what it is. A
  replay that presented placement time as arrival time would make its own
  pacing metadata unfalsifiable.
- Intervals are taken between **consecutive emitted items** of this run. An item
  omitted by projection, or lying outside the requested range, leaves its time
  in place -- the run still spans the original elapsed time -- and no time is
  waited *before* the first emitted item, whose emission starts the run.
- `speed_factor` applies to `recorded` only and MUST be rejected -- not ignored
  -- for the other two modes. It is a finite double in `(0, 1000]`; zero,
  negative, NaN, and infinity are errors. The scheduled interval is the recorded
  delta divided by `speed_factor`.
- **Deadlines are absolute, so drift does not accumulate.** Each item's target
  is computed from the run's start instant plus its cumulative scaled offset,
  never by adding one sleep to the last actual wake-up.
- **A late deadline emits immediately and nothing is skipped or compressed.** If
  the consumer or the scheduler falls behind, an already-past deadline fires at
  once -- which is catch-up, bounded below by `as_fast_as_possible` and never
  faster. A pacing implementation MUST NOT drop, reorder, or coalesce items to
  recover time, and MUST report lateness (`late_item_count`, and the maximum
  lateness in nanoseconds) rather than hide it.
- A **negative** recorded delta (non-monotonic recorded host time) is clamped to
  zero -- the item is emitted immediately, never waited for backwards -- and the
  occurrence is counted and reported.
- Pacing is source-side timing only. It MUST NOT change item content, ordering,
  replay-run sequence numbers, or any recorded timestamp.

### 8.8 Step control

`step` replaces pacing with an explicit permit model.

- `advance(n)` **adds** `n` permits to a running total; permits accumulate across
  calls and are not a level to set. `n` MUST be at least 1. The counter is
  **`uint64` and saturates at `UINT64_MAX`** -- a frozen bound, not an
  implementation choice, because saturation is observable: `advance(k)` at the
  bound grants nothing and MUST be reported as saturated rather than silently
  accepted or wrapped. No replay run can contain `UINT64_MAX` items, so
  saturation is a misuse signal, and a configurable `max_step_permits` was
  rejected as a second way to spell the same limit.
- The source emits at most one **item** per permit and consumes exactly one
  permit per emitted item -- a discontinuity costs a permit exactly as a frame
  does, because both are items in the sense of section 1.1. Any other accounting
  makes `advance(1)` mean different things at different positions.
- With zero permits the source produces nothing: a blocking read waits, a
  non-blocking read reports "no item available", and neither spins.
- Permits are cleared by reset (section 8.10) and by nothing else. They are not
  consumed by an item the source declines to emit -- an omitted item costs no
  permit, so `advance(k)` always yields `k` observed items unless the run ends
  first.
- `speed_factor` is rejected in `step` mode; step timing is entirely
  caller-driven.

### 8.9 Cancellation and end of data

- **Cancellation wakes a waiting source promptly.** A pacing sleep or a step wait
  MUST be interruptible: cancellation latency is bounded by a documented wake
  latency, **not** by the remaining pacing interval or by the arrival of the next
  permit. A replay that could only be cancelled after a 30-second recorded gap
  would be unusable in a test harness, which is where this mode lives.
- After cancellation the source emits nothing further, enters a terminal
  `cancelled` state, and further cancellation calls are no-ops.
- **End of data is reported exactly once as a distinct terminal result**
  (`end_of_data`) -- not an error, not an empty successful read that a caller
  must guess about. The terminal state is **sticky**: every subsequent read
  returns the same terminal result immediately, never blocks, and never re-emits
  an item.
- Terminal states do not downgrade. Cancelling after `end_of_data` leaves
  `end_of_data`; the abnormal terminal result of section 8.5 (a truncated
  session replayed under allow-incomplete) is distinct from `end_of_data` and
  MUST NOT be reported as it.
- Only reset clears a terminal state.

### 8.10 Reset

- Reset returns the source to the start of the **same** run: the same items in
  the same order, the same replay-run sequence mapping (sections 8.2 and 8.3),
  and the same omission and fidelity metadata. Two runs separated by a reset are
  indistinguishable.
- Reset clears: the sticky terminal state (section 8.9), all accumulated step
  permits, the pacing timeline origin -- which is re-anchored on the first item
  emitted after the reset -- and the fired/unfired state of every injected
  fault, so each fault **re-arms** and fires again at the same position.
- Reset does **not** change: the configuration, the stream selection, the range,
  the set of injected fault definitions, or anything in the source artifact.
- Reset requires a quiesced source -- between reads, at a terminal state, or
  after cancellation. A reset concurrent with an in-flight read MUST be rejected
  with an error rather than raced.

### 8.11 Fault injection

Replay fault injection MUST be deterministic and bounded, and MUST modify only
the replay run. It MUST NOT mutate the source NRF session or the replay image.

- **A fault is positioned by a recorded identity, never by a run-local one.**
  The identity is a tagged union: `stream_frames` lets one block have more than
  one discontinuity ahead of it (section 8.3 -- "two discontinuities at one
  position keep their committed record order"), so the pair `(stream_id,
  block_ordinal)` alone could name the frame or any of several discontinuities
  and the run could not tell which. Frozen:

  ```text
  Ledger-based modes:
    kind                  frame | discontinuity
    data_message_ordinal  uint64   (section 1.1)

  stream_frames:
    StreamFrameFaultTarget
      kind          = frame
      stream_id
      block_ordinal
    StreamDiscontinuityFaultTarget
      kind          = discontinuity
      stream_id
      discontinuity_record_ordinal
  ```

  `data_message_ordinal` already identifies one item, but the ledger-based form
  keeps `kind` so configuration validation can reject a `sequence_gap` aimed at
  a discontinuity (below). The replay-run sequence MUST NOT be used -- it is
  request-local (section 8.2) -- and neither may wall-clock time, which is not
  reproducible.
- Each fault fires **at most once per run**, immediately before the identified
  item would be emitted, and re-arms on reset. A fault whose position lies
  outside the requested range never fires and MUST be **reported as unfired**
  rather than silently dropped.
- The declared effects are these four, and nothing else without a change to this
  contract:

  | Effect | What the consumer observes |
  | --- | --- |
  | `stall` | emission of that item is delayed by a configured, finite duration, then the run continues normally; the stall MUST remain interruptible by cancellation (section 8.9) |
  | `read_failure` | the read for that item fails with a distinct error and the run enters a terminal faulted state; v1 defines no recoverable form |
  | `sequence_gap` | the identified **frame** is not emitted, so the consumer sees a genuine jump in the replay-run frame sequence |
  | `abnormal_end` | the run ends at that position with the abnormal terminal result of section 8.5, never with `end_of_data` |

- **`sequence_gap` targets a frame and only a frame.** Its identity is a
  `data_message_ordinal` of kind `frame` in the ledger-based modes, or a
  `StreamFrameFaultTarget` in `stream_frames` -- never a
  `StreamDiscontinuityFaultTarget`. Aiming it at a discontinuity MUST fail
  **at configuration validation**, before the run starts, because dropping a
  discontinuity produces no frame-sequence hole at all -- the effect would not
  happen, and a fault that silently does nothing is worse than a rejected
  configuration. The other three effects accept either kind of item.
- **A `sequence_gap` target MUST have an emitted frame on both sides of it.**
  The effect's promise is that the consumer *sees a jump*: the frame before the
  target is emitted and the frame after it is emitted, so the run-wide sequence
  visibly skips. A target that is the run's first emitted frame, or its last,
  has no frame on one side and therefore no observable jump, so it MUST fail at
  configuration validation. Skipping a boundary frame without that guarantee is
  a different effect -- drop the frame, emit nothing -- and would need its own
  name and definition, not `sequence_gap`.
- **Injected effects are reported in their own record, not in the omission
  metadata.** `omitted_data_messages` and `omitted_stream_messages` describe
  what a *projection* could not represent; an injected skip is something the run
  did on purpose. Mixing them would require a reason value
  (`injected_sequence_gap`) that belongs to neither frozen schema, and would let
  a caller read an injected gap as a projection artifact. Frozen:

  ```text
  injected_effects
    target_identity   tagged union frozen above
    effect            stall | read_failure | sequence_gap | abnormal_end
    fired             bool   false when the position lay outside the range
    emitted           bool | null
                      unfired -> null
                      stall -> true
                      read_failure -> false
                      sequence_gap -> false
                      abnormal_end -> false
  ```

- **What "the only way" means, stated narrowly.** Injection is the only way a
  **renumbered** run -- `recorded_projection` or `stream_frames` -- acquires a
  *new* sequence hole; sections 8.2 and 8.3 renumber precisely so that
  projection never produces one. `exact_frames` is different: it emits the
  recorded sequence unchanged, so any hole the recording already contained
  survives into the replay, described by the recorded discontinuity that
  explains it. That is faithful replay, not injection, and the two MUST NOT be
  reported under one name.
- Bounded means declared up front and finite: the number of faults and the total
  injected delay of a run are finite and known before the run starts. No fault
  may stall indefinitely or depend on external input.
- No effect modifies payload bytes. Corrupting a payload while claiming recorded
  provenance would make every provenance field in this section untrustworthy.

### 8.12 Replay session identity

Every native `FrameHeader` carries a `SessionId`, and so does every
`Discontinuity` (`cpp/include/neurale/streaming/frame.h`,
`cpp/include/neurale/streaming/discontinuity.h`; `SessionId` is a non-zero
`uint64` -- `cpp/include/neurale/streaming/clock.h`). Section 8 freezes the
schema, signal, clock, feature, and unit IDs, the run-wide frame sequence, and
the current-run host time; the replay-run `SessionId` is also emitted output,
so it is frozen here rather than left to the implementer:

- `NativeReplaySource` obtains one non-zero **`replay_run_session_id`** when it
  is prepared or constructed. Every frame and every discontinuity the source
  emits in its lifetime carries that one ID -- the run's identity, not the
  recording's.
- **Reset keeps the same `replay_run_session_id`** (section 8.10): a reset is
  the same run replayed again, indistinguishable from before the reset,
  including in `SessionId`.
- A **new** replay source or run obtains a **new** `replay_run_session_id` by
  default, so two concurrent replays of one session do not collide on
  `SessionId`, and re-recording or composing a replay's output does not
  impersonate the source session.
- The **source NRF session ID is carried separately as `source_session_id`
  provenance**, never as the emitted `SessionId`. Using the recording's
  `SessionId` as the replay-run ID would make a replay indistinguishable from
  the original recording on the wire.
- The runtime-allocated `replay_run_session_id` is **not** part of any
  source-derived deterministic replay-image content or fingerprint (section
  8.3): it is a runtime-run attribute, like current-run host time (section
  8.6), not a fact about the source. A caller MAY supply
  `replay_run_session_id` explicitly for a fully reproducible test run, but
  the default MUST NOT be the source session ID.

## Runtime validation invariants

The native characterization tests pin the shutdown, loss, and dependency
properties required by the contract above.

### Native runtime

`cpp/tests/streaming_recording_lifecycle_characterization_test.cpp` pins:

- **Graceful end drains an existing backlog and flushes.** The test holds the
  observer inside its first `observe()` until the source has ended, every frame
  has been enqueued on the edge, the critical path has finished, and the runtime
  has reached its terminal state -- so a real backlog of accepted-but-undelivered
  messages exists, and the test asserts it does (`delivered == 0`,
  `enqueued == 8`) before releasing the observer. The dispatch loop closes each
  edge queue; the edge worker then drains every remaining message, delivers it,
  and calls `flush()` once. Nothing accepted by the edge is dropped.
- **Abort drops instead of draining.** `abort()` marks every edge inactive,
  closes its queue, and calls `cancel()`. The edge worker leaves its loop
  immediately, and every message still queued is counted through
  `record_drop` and destroyed. This is true for a `critical_recorder` edge
  exactly as it is for a noncritical one.
- **Abort suppresses flush.** `flush()` is not called on any edge when the
  runtime is aborting.
- **A critical edge escalates; a noncritical edge does not.** An observer
  failure or dispatch overrun on an edge with `critical_recorder = true`
  records a fault, inhibits safety, and aborts the runtime. The same failure on
  a noncritical edge deactivates only that edge.
- **Terminal state is not authoritative.** The terminal `RuntimeState` is
  computed once by the actuator loop as `failed || primary_fault().has_value()`.
  A fault recorded by an observer edge worker after that point does not change
  it, and the `request_abort(true)` that follows the fault stores `stopping`
  over it. A faulted session can therefore read `stopped` *or* `stopping` after
  `join()` has returned and every worker has finished. `primary_fault()` and
  the status returned by `join()` are the authoritative signals; both are
  computed after all workers are joined. An explicit `abort()` records no fault
  and reliably ends in `stopped`.
- **`arm()` does not change `RuntimeState`.** It releases the safety inhibit
  from the `prepared` state; there is no distinct armed state.

### Recording boundary

The generic streaming layer owns `NativeCriticalObserver`, readiness and
health checks, the lossless-only registration rule, runtime acceptance
metadata, reserved primary-fault delivery, terminal notice, bounded drain and
cancellation. Abort and runtime fault drain the shared observer-dispatch items
already accepted for the critical path; noncritical observer behavior is
unchanged. `RuntimeState` becomes terminal only after workers join. The
streaming target contains no recording-format or concrete-recorder dependency;
`neurale_recording` implements the generic boundary in the allowed direction.
