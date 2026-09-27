# Private native spool, version 1

Status: normative binary specification for native spool version 1.0. The
reference tools and the binary vectors in this directory are part of the
specification.

## 0. What this is, and what it is not

The spool is the **private, append-only staging store** the native recorder
writes while a session is being captured. It is not a distribution format, not
a public API, and not an archive:

- **Private.** No public Python, C++, or CLI surface reads or writes it. It has
  no stability guarantee to anyone outside this repository, and no third party
  is invited to depend on its bytes. It is specified here because *we* need one
  agreed description for a writer, a finalizer, and a recovery tool that are
  built by different tasks and must agree byte for byte -- not because it is
  published.
- **Rebuildable, in one direction only.** Everything a reader needs is promoted
  into the canonical NRF session by finalization; nothing downstream keeps
  pointing at the spool, so once finalization has been validated and published
  the spool is a build input that may be discarded (retention rules:
  `docs/development/native_recording_replay.md` section 7). The reverse is
  **not** true: a spool cannot be reconstructed from an NRF session. Until
  finalization has succeeded and been validated, the spool is the only
  reconstruction input there is, and deleting it is the failure mode section 7
  exists to prevent.
- **Not the source of truth for a finished recording.** The canonical NRF
  session is. A question about what a *recording* contains is answered from
  NRF; the spool answers only what was captured and not yet promoted.

Normative source for everything this format serves:
`docs/development/native_recording_replay.md` -- lifecycle (section 3),
what must survive (section 5), accounting (section 5.1), durability policies
(section 6), retention (section 7), and process crash (section 4.5). Where this
document appears to say something different from that contract, the contract
wins and this document has a bug.

This specification defines the **container**: framing, checksums, visibility,
scanning, tails, repair, versioning, and the three record payloads the
lifecycle contract itself requires (checkpoint, accounting snapshot, session
end). It does **not** define the interior of a frame, signal-block,
discontinuity, signal-gap, control, or fault payload; those belong to the
recorder core (M6-05), and a container that also defined them would give those
fields two specifications. They are carried here as opaque, length-framed,
checksummed byte strings.

The keywords MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT,
and MAY are normative.

## 1. Conventions

- A spool is **one file**. There is no directory, no sidecar, no index, and no
  second copy of anything. The conventional extension is `.nspool`; the name is
  the recorder's business and this specification does not constrain it.
- Every integer field is **fixed-width little-endian**, unsigned unless stated.
  There are no varints, no length-prefixed strings outside the fixed fields
  named below, no alignment padding that a reader has to infer, and no
  structure whose size depends on the platform. Nothing here is a memory image
  of a C++ struct: the layout is defined by the tables in this document, and an
  implementation that writes `sizeof(T)` bytes of anything is not conforming.
- All offsets and lengths are **byte counts from the start of the file**, and
  every structure begins at a multiple of the alignment (8).
- Text fields are UTF-8 in a fixed-size field, padded with `0x00`. A value that
  does not fit MUST be rejected by the writer; truncating an identifier is
  never allowed.
- Reserved bytes MUST be written as zero and MUST be ignored on read, except
  where a rule below says otherwise.
- `crc32c` means CRC-32C as defined in section 3.

## 2. File layout

```text
offset 0                        fixed superblock            256 bytes
offset 256                      plan document               plan_bytes
                                zero padding                to the next multiple of 8
offset first_transaction_offset transaction 1
                                transaction 2
                                ...
                                (possibly an invalid tail)
```

A spool is written strictly by appending. A byte that has been written and
covered by a commit trailer MUST NOT be rewritten, and no code path may seek
backwards to amend a committed transaction. There is no free list, no
in-place update, and no compaction.

### 2.1 Superblock

Written once, before the spool is reported ready, and never modified.

| Offset | Size | Field | Rule |
| --- | --- | --- | --- |
| 0 | 8 | `magic` | ASCII `NRLSPOOL` |
| 8 | 2 | `version_major` | 1 |
| 10 | 2 | `version_minor` | 1 |
| 12 | 4 | `superblock_bytes` | 256 |
| 16 | 1 | `durability_policy` | 0 buffered, 1 checkpoint sync, 2 transaction sync |
| 17 | 1 | `checksum_algorithm` | 1 (CRC-32C); no other value is defined |
| 18 | 1 | `plan_encoding` | 1 (recording-plan document, RFC 8785 canonical JSON) |
| 19 | 1 | reserved | 0 |
| 20 | 4 | `alignment` | 8 |
| 24 | 4 | `plan_bytes` | length of the plan document |
| 28 | 4 | `plan_crc32c` | CRC-32C of the plan document |
| 32 | 32 | `plan_fingerprint` | SHA-256 of the plan document |
| 64 | 16 | `session_uuid` | binary session identity |
| 80 | 8 | `created_unix_nanos` | wall-clock nanoseconds since the Unix epoch |
| 88 | 8 | `first_transaction_offset` | `256 + pad8(plan_bytes)` |
| 96 | 128 | `session_id` | UTF-8, NUL-padded; the NRF session id |
| 224 | 28 | reserved | zero |
| 252 | 4 | `superblock_crc32c` | CRC-32C of bytes `[0, 252)` |

**The plan document is stored, not referenced.** Section 5 of the lifecycle
contract requires session identity, the plan fingerprint, and the schemas,
descriptors, and clocks to survive a process crash *in the spool*; a spool that
pointed at a file owned by somebody else would survive nothing on its own.

**The container never parses it.** The plan document is an opaque byte string.
The container checks its length, its CRC-32C, and its SHA-256 against
`plan_fingerprint` -- which it can do without interpreting a single byte,
because the fingerprint of the native replay recording plan *is* the SHA-256 of
that canonical document. Interpreting it is the finalizer's job. That is why
`plan_encoding` exists: a later encoding may be registered without any change
to the container.

The plan document is followed by `pad8(plan_bytes) - plan_bytes` bytes of
`0x00`, which MUST be zero and MUST be checked. **No checksum reaches them**:
the plan CRC stops at `plan_bytes` and the superblock CRC stops at byte 252, so
the zero rule is the only thing covering that gap, exactly as it is the only
thing covering record padding.

A reader MUST refuse the whole spool when the magic is wrong, the superblock
checksum fails, `version_major` is not implemented, any fixed field above holds
a value this version does not define, the file is shorter than the superblock
region, the plan padding is not zero, or the plan document does not match its
checksum and fingerprint. These are the only conditions under which nothing at
all is readable.

### 2.2 Transaction

```text
transaction header    48 bytes
record 1              32-byte header + payload + zero padding to 8
...
record N
commit trailer        48 bytes
```

Transaction header:

| Offset | Size | Field | Rule |
| --- | --- | --- | --- |
| 0 | 8 | `magic` | ASCII `NNSTXBEG` |
| 8 | 8 | `transaction_id` | 1 for the first transaction, then exactly `previous + 1` |
| 16 | 8 | `previous_transaction_offset` | offset of the previous transaction header; 0 for the first |
| 24 | 4 | `record_count` | at least 1 |
| 28 | 4 | `header_bytes` | 48 |
| 32 | 8 | `begin_unix_nanos` | wall-clock nanoseconds |
| 40 | 4 | reserved | 0 |
| 44 | 4 | `header_crc32c` | CRC-32C of bytes `[0, 44)` of the header |

Commit trailer:

| Offset | Size | Field | Rule |
| --- | --- | --- | --- |
| 0 | 8 | `magic` | ASCII `NNSTXEND` |
| 8 | 8 | `transaction_id` | equal to the header's |
| 16 | 8 | `body_bytes` | bytes from the start of the transaction header to the start of this trailer |
| 24 | 4 | `record_count` | equal to the header's |
| 28 | 4 | `data_items` | data-plane items in this transaction (section 4.3) |
| 32 | 4 | `control_items` | control-plane items in this transaction |
| 36 | 4 | `body_crc32c` | CRC-32C over `[transaction header start, trailer start)` |
| 40 | 4 | `flags` | 0 |
| 44 | 4 | `trailer_crc32c` | CRC-32C of bytes `[0, 44)` of the trailer |

**The trailer is what makes the records visible**, which is why it is written
last and carries the checksum over everything before it. A writer interrupted
anywhere inside a transaction leaves bytes no reader will promote, and it does
so without the writer having to know it was interrupted.

The back link exists for repair tooling, which needs to walk backwards from a
damaged region without re-scanning from the superblock, and for a reader that
wants to check the chain rather than trust the sequence of ids alone.

### 2.3 Record

| Offset | Size | Field | Rule |
| --- | --- | --- | --- |
| 0 | 2 | `record_kind` | section 4.1 |
| 2 | 2 | `record_flags` | 0 in version 1 |
| 4 | 4 | `payload_bytes` | payload length, before padding |
| 8 | 8 | `logical_ordinal` | the owning logical item ordinal, or 0 when the record has no owning item (see below) |
| 16 | 8 | `record_unix_nanos` | wall-clock nanoseconds, or 0 |
| 24 | 4 | `payload_crc32c` | CRC-32C of the payload, excluding padding |
| 28 | 4 | `header_crc32c` | CRC-32C of bytes `[0, 28)` of the header |

The payload is followed by `pad8(payload_bytes) - payload_bytes` bytes of
`0x00`. Padding MUST be zero: it carries no information, so a nonzero padding
byte is unexplained, and a reader MUST treat it as corruption rather than
ignore it.

`record_flags` is reserved for a future minor version and MUST be 0; a reader
MUST report a nonzero value (`SPOOL-040`) rather than ignore it.

`logical_ordinal` is the **owning** item ordinal, not the record's own: a
`signal_block` is part of the `frame` that owns it, and a `signal_gap` is part
of its `discontinuity`, so each carries its owner's data-message ordinal rather
than zero. This is what lets a reader attribute a block to a frame (and a gap
to a discontinuity) without parsing the opaque payloads, which is the
container's whole job here. The owner is the nearest preceding owner of the
right kind in the **same transaction**, and a block or gap with no owner in its
transaction is a violation. The rules, frozen:

| Record | `logical_ordinal` |
| --- | --- |
| `frame` | its own data-message ordinal |
| `signal_block` | its owning frame's data-message ordinal |
| `discontinuity` | its own data-message ordinal |
| `signal_gap` | its owning discontinuity's data-message ordinal |
| `control` | its own global control submission ordinal |
| `fault`, `accounting_snapshot`, `session_end`, `checkpoint` | 0 (no owning item) |

Timestamps are integer nanoseconds. There is no floating-point field anywhere
in this format, on any path.

## 3. Checksums

The only checksum algorithm in version 1 is **CRC-32C** (Castagnoli): normal
polynomial `0x1EDC6F41`, reflected form `0x82F63B78`, reflected input and
output, initial value `0xFFFFFFFF`, final XOR `0xFFFFFFFF`. This is the
parameterization computed by the `crc32c` instructions on x86-64 and AArch64,
which is the point: the writer runs on the recorder's worker thread.

Known answers, for an implementation to check itself before it checks a spool:

| Input | CRC-32C |
| --- | --- |
| empty | `0x00000000` |
| `123456789` | `0xE3069283` |
| 32 bytes of `0x00` | `0x8A9136AA` |
| 32 bytes of `0xFF` | `0x62A8AB43` |

CRC-32C detects accidental corruption -- a torn write, a bit flip, a truncated
transfer. It is **not** a tamper check and MUST NOT be described as one. The
plan document additionally carries a SHA-256, because that value has to match a
fingerprint computed elsewhere, not because the spool is defending against an
adversary.

Every checksum covers a **contiguous, explicitly named** byte range, and no
checksum covers another checksum's field. That is what lets a reader say which
structure failed rather than only that something did.

## 4. Records

### 4.1 Record kind registry

| Value | Kind | Payload defined by | Item |
| --- | --- | --- | --- |
| 1 | `frame` | M6-05 (opaque here) | one data-plane item |
| 2 | `signal_block` | M6-05 (opaque here) | none: part of its frame |
| 3 | `discontinuity` | M6-05 (opaque here) | one data-plane item |
| 4 | `signal_gap` | M6-05 (opaque here) | none: part of its discontinuity |
| 5 | `control` | M6-05 (opaque here) | one control-plane item |
| 6 | `fault` | M6-05 (opaque here) | none |
| 7 | `accounting_snapshot` | section 4.4 | none |
| 8 | `session_end` | section 4.5 | none |
| 9 | `checkpoint` | section 4.6 | none |

Values 10-4095 are unassigned. Values 4096 and above are reserved and MUST NOT
be written. The "Item" column says what a committed record contributes to the
item counts of section 4.3; the `logical_ordinal` each kind carries is a
separate question, settled in section 2.3.

### 4.2 Unknown record kinds

A reader that meets a committed record whose kind it does not know MUST report
the spool as **not finalizable** and MUST NOT promote its committed prefix.

This is deliberately stricter than "skip what you do not understand". The
writer committed that record, so it is part of what the session captured;
promoting a prefix that stepped over it would produce an NRF session missing a
record nobody counted, which is exactly the silent loss the lifecycle contract
does not tolerate anywhere. Skipping is safe for a *cache*; a spool is the only
copy.

### 4.3 Item counting

The lifecycle contract counts **items**: one accepted data message on the data
plane, one control record on the control plane. In this container:

- data-plane items are `frame` and `discontinuity` records;
- control-plane items are `control` records;
- everything else contributes nothing. A signal block belongs to the frame that
  owns it and a signal gap to its discontinuity, so counting either again would
  double-count one item.

Each commit trailer states its own transaction's counts. A reader derives the
committed totals by summing over the committed prefix, and MUST NOT take them
from any single record. A trailer whose counts disagree with the records the
transaction actually carries invalidates that transaction.

### 4.4 `accounting_snapshot` payload (208 bytes)

Thirteen `u64` counters, then eight flag bytes, then four 24-byte positions.

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 8 | `runtime_accepted` |
| 8 | 8 | `recorder_accepted` |
| 16 | 8 | `spool_committed` |
| 24 | 8 | `rejected_before_runtime_acceptance` |
| 32 | 8 | `failed_between_runtime_and_recorder` |
| 40 | 8 | `lost_between_recorder_and_spool` |
| 48 | 8 | `control_offered` |
| 56 | 8 | `control_accepted` |
| 64 | 8 | `control_spool_committed` |
| 72 | 8 | `control_rejected` |
| 80 | 8 | `lost_between_control_acceptance_and_spool` |
| 88 | 8 | `rejected_after_close_data` |
| 96 | 8 | `rejected_after_close_control` |
| 104 | 1 | `control_offered_present` (0 or 1) |
| 105 | 1 | `producer_acceptance_known` (0 or 1) |
| 106 | 1 | `accounting_origin` (0 = recorder; the only legal value in a spool) |
| 107 | 5 | reserved, zero |
| 112 | 24 | `data_first_loss` |
| 136 | 24 | `data_first_rejection` |
| 160 | 24 | `control_first_loss` |
| 184 | 24 | `control_first_rejection` |

Position (24 bytes), the tagged union of section 5.1 of the contract:

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 1 | `tag`: 0 absent, 1 ordinal, 2 producer identity |
| 1 | 3 | reserved, zero |
| 4 | 4 | `identity_kind` (a registered producer kind; 0 unless `tag` is 2) |
| 8 | 8 | `ordinal` (0 unless `tag` is 1) |
| 16 | 8 | `identity_value` (the rejected item's frame sequence or control identity; 0 unless `tag` is 2) |

The two flag bytes and the four positions are not free-form: a reader MUST
reject an accounting snapshot whose `control_offered_present` or
`producer_acceptance_known` is neither 0 nor 1, whose `accounting_origin` is
not the recorder, whose position `tag` is not 0, 1, or 2, or whose inactive
union member (the fields the `tag` does not select) is not zero. The contract
also fixes the **tag form** each position must use when it is present: a loss
after acceptance names the accepted item's `ordinal`, and a rejection before
acceptance names its `producer identity` -- recording a loss as a producer
identity would read like a pre-acceptance event, and a rejection as an ordinal
would name an ordinal the item never had. A position is **present** exactly when
at least one of its related counters is nonzero -- the first loss or rejection is
latched in bounded storage (contract section 4.3), so a present position over
all-zero counters, or an absent position over any nonzero one, is an inconsistency a
reader must report. The four positions name:

| Position | Required tag | Present when any listed counter is nonzero |
| --- | --- | --- |
| `data_first_loss` | `ordinal` | `failed_between_runtime_and_recorder` + `lost_between_recorder_and_spool` |
| `data_first_rejection` | `producer identity` | `rejected_before_runtime_acceptance` |
| `control_first_loss` | `ordinal` | `lost_between_control_acceptance_and_spool` |
| `control_first_rejection` | `producer identity` | `control_rejected` |

`data_first_loss` covers both post-acceptance loss handoffs because an item
that failed between runtime and recorder was already accepted and already has a
data-message ordinal; binding the position to `lost_between_recorder_and_spool`
alone would miss that loss. When both data-plane loss counters are nonzero the
position carries the data-message ordinal of the earliest lost item across the
two handoffs -- a writer decision the container carries but cannot re-derive.

**The producer identity is a fixed numeric registry, not a free-form kind.**
When `tag` is `producer identity`, `identity_kind` MUST be one of the values
below; the number is the only bridge between the spool and the NRF kind string
the finalizer writes, so two writers may not pick different digits for the same
kind. The registry is partitioned by plane: a data rejection names a
`StreamMessageKind` (singular), and a control rejection names a control
record-set kind (the plural NRF `record_schema.kind`), matching the two NRF
accounting fields. Lifecycle signals (`end_of_stream`, `shutdown`, `abort`)
are not recorded and never reach a first-rejection position, so only the two
recorded data message kinds appear.

| Value | Canonical name | Plane | Names |
| --- | --- | --- | --- |
| 1 | `frame` | data | a rejected `frame` |
| 2 | `discontinuity` | data | a rejected `discontinuity` |
| 3 | `events` | control | a rejected event record |
| 4 | `trials` | control | a rejected trial record |
| 5 | `experiment_states` | control | a rejected experiment-state record |
| 6 | `commands` | control | a rejected command record |
| 7 | `targets` | control | a rejected target record |
| 8 | `labels` | control | a rejected label record |
| 9 | `assistance` | control | a rejected assistance record |
| 10 | `faults` | control | a rejected fault row |
| 11 | `task_variables` | control | a rejected task-variable record (added in 1.1) |

A reader MUST report a position whose `identity_kind` is not in the registry,
or whose kind belongs to the other plane. Adding a kind is a minor-version
change (section 9).

**`identity_value` is a `uint64` producer identity, and its NRF form is fixed.**
The native control API MUST assign a `uint64` identity to every control record
before it is submitted; an arbitrary string is not a valid accounting identity
here, and the recorder's string record id (for example `event-00000001`) is a
separate, materialization-time identity, not the one the accounting position
carries. The finalizer converts the stored value without loss:

- a data rejection's `identity_value` is the rejected item's `frame_sequence`,
  copied unchanged into the NRF `data_first_rejected_frame_sequence` (`uint64`);
- a control rejection's `identity_value` is the control record's `uint64`
  identity, rendered as **unsigned decimal ASCII with no leading zeros** (`0`
  renders as `"0"`) into the NRF `control_first_rejected_identity` (`utf8`).

The kind renders through the registry above: `identity_kind` becomes the
canonical name in `data_first_rejected_message_kind` or
`control_first_rejected_kind`. With both fixed, a finalizer reconstructs the
M6-02 accounting row from the spool alone.

Three properties of this layout are decisions, not omissions:

- **The finalization-side counters are absent.** `nrf_committed`,
  `lost_during_finalization`, and their control-plane twins do not appear,
  because the spool is written before finalization runs. A spool that carried
  them would be carrying numbers nothing had measured. The finalizer (M6-08)
  supplies them when it writes the NRF accounting summary.
- **There is no `accounting_verified` field, and there MUST NOT be one.**
  Section 5.1 is explicit: it is a judgement a reader makes about a record, not
  a fact a writer holds. Storing it would create two values that can disagree.
- **An ordinal and a producer identity are different sequences.** A reader MUST
  NOT compare or order one against the other. Both first-position slots per
  plane exist because the first loss after acceptance and the first refusal
  before acceptance are different events.

Verification, mirroring the two layers of section 5.1 but over the capture side
only, is defined in section 7.

### 4.5 `session_end` payload (80 bytes)

The session-end record is what *freezes* the capture outcome (contract section
3.2), and it is the only place the terminal provenance survives before
finalization. The lifecycle contract requires `requested_terminal_intent`,
`capture_outcome`, the terminal reason, and an indication of whether the
primary fault row committed to be preserved as **separate** fields -- a caller
may abort and a recorder fault may then escalate the capture to `faulted` while
the request stays `aborted`, and a faulted session may end without a committed
fault row at all. None of these may be recovered from any of the others, and
without them the finalizer (M6-08) cannot write the NRF termination record the
contract requires.

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 1 | `requested_terminal_intent`: 0 normal, 1 aborted, 2 fault |
| 1 | 1 | `capture_outcome`: 0 normal, 1 aborted, 2 faulted |
| 2 | 1 | `primary_fault_committed`: 0 or 1 |
| 3 | 5 | reserved, zero |
| 8 | 64 | `terminal_reason`: UTF-8, NUL-padded; the bounded reason the finalizer copies into the NRF termination record |
| 72 | 8 | `end_unix_nanos` |

`requested_terminal_intent` is what the first exit from `recording` latched
(contract section 3.2): `normal` for a graceful stop, `aborted` for an explicit
abort, `fault` for a runtime or recorder fault that ended the capture.
`capture_outcome` is how *taking the data* ended, frozen at this record and
never moved afterwards. `primary_fault_committed` is `1` when the primary fault
row reached a committed extent and `0` when it did not -- so the finalizer can
declare the row missing in the NRF termination's `extensions`, exactly as
contract section 3.2 requires for a faulted session whose fault row could not
be persisted. A reader MUST report a value outside the defined enum for
`requested_terminal_intent` or `capture_outcome`, or a `primary_fault_committed`
that is neither 0 nor 1, rather than guess what a writer meant.

There is no `unknown` outcome, on purpose. `unknown` is what a **missing**
session-end record means (contract section 3.2), and giving it a value would
let a writer state the one thing only an absence can say. A recovery tool MUST
NOT synthesize this record under any circumstance (contract section 4.5): the
spool's committed prefix is the evidence, and it is left exactly as found.

A `session_end` record MUST be preceded, in the same transaction, by the
`accounting_snapshot` it seals -- the spool's form of section 5.1's rule that
the summary is written in the transaction that seals the session. There is at
most one `accounting_snapshot` and at most one `session_end` in a spool, and an
`accounting_snapshot` with no `session_end` in its transaction is a violation:
the summary seals exactly one session, and a snapshot that sealed nothing is
unaccountable. No record may be committed after the transaction containing
`session_end`.

### 4.6 `checkpoint` payload (24 bytes)

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 8 | `durable_extent_bytes` |
| 8 | 8 | `synced_unix_nanos` |
| 16 | 1 | `durability_policy`, equal to the superblock's |
| 17 | 7 | reserved, zero |

`durable_extent_bytes` is an **extent** -- how far synchronization was
acknowledged -- never a count of records that reached a durable stage
(contract section 1.2). It MUST NOT exceed the offset at which the transaction
carrying the checkpoint begins: the writer cannot have synced bytes it had not
written. A reader MUST report a checkpoint whose `durability_policy` is not the
superblock's: the policy is stored twice so that each record is self-describing,
and a disagreement is a writer bug, not a choice the reader may silently pick a
side of.

## 5. Durability

The three policies, named exactly as section 6 of the lifecycle contract names
them, with the value stored in the superblock:

| Value | Policy | What is promised |
| --- | --- | --- |
| 0 | `buffered` | records are written without an explicit sync; survives a process crash to the extent the page cache does; **nothing** against power loss or kernel panic |
| 1 | `checkpoint_sync` | an explicit sync at checkpoint boundaries; everything up to the last committed checkpoint survives power loss, everything after it has the `buffered` guarantee |
| 2 | `transaction_sync` | an explicit sync as part of every commit; a transaction reported as committed survives power loss. The only policy under which "spool committed" implies "durable" |

Rules:

- The superblock region MUST be synced before the spool is reported ready,
  under **every** policy. A spool whose own identity did not survive the crash
  could not be matched to a finalization target at all.
- The **durable extent** is a byte offset, derived by the reader, and it is the
  only durability number this format reports:

  ```text
  transaction_sync   the end of the last committed transaction
  checkpoint_sync    the largest extent claimed by a committed checkpoint,
                     clamped to the committed prefix, floored at the
                     superblock region
  buffered           the superblock region, and no record
  ```

- No status, report, benchmark record, or docstring may promise more than the
  policy in force delivers, and a successful enqueue is never durable under any
  policy. The policy is stored in the superblock precisely so that no reader
  has to infer a guarantee from a sync it cannot see.
- The committed extent and the durable extent are different numbers and MUST be
  reported separately. `valid-complete`, `valid-checkpoint-sync`, and
  `valid-buffered` are the same session bytes under the three policies, and
  report three different durable extents over the same committed prefix.

## 6. Reading

A read of a spool is **read-only**: it never writes, never truncates, and never
repairs (contract section 4.5). The scan is:

1. Validate the superblock region (section 2.1). On failure the spool is
   `rejected` and nothing is readable.
2. Set `offset` to `first_transaction_offset`, `expected_id` to 1, and
   `previous_offset` to 0.
3. While bytes remain, validate the transaction at `offset`:
   1. enough bytes for a header; the begin magic; the header checksum;
   2. `transaction_id == expected_id`; `previous_transaction_offset == previous_offset`;
      `header_bytes == 48`; `record_count >= 1`;
   3. each record in turn: enough bytes; header checksum; payload checksum;
      zero padding;
   4. enough bytes for a trailer; the end magic; the trailer checksum;
      `transaction_id`, `body_bytes`, and `record_count` matching the header
      and the actual body; the body checksum; the item counts matching the
      records.
4. Any failure in step 3 ends the committed prefix **at the start of that
   transaction**. The scan stops. Nothing after that byte is visible to any
   caller, ever.
5. On success the transaction is committed: its records are visible, its item
   counts are added to the totals, `offset` advances past the trailer,
   `expected_id` increments by exactly one, and `previous_offset` becomes the
   transaction's own offset.

A record is **visible** if and only if it lies inside a transaction that
completed step 3. This is stage 3 of the acceptance ladder ("spool committed")
and nothing weaker: a record inside an incomplete or checksum-invalid
transaction MUST NOT be promoted, ever, by any code path.

### 6.1 Tail handling

The bytes after the committed prefix are the **tail**. Reported status:

| Status | Meaning |
| --- | --- |
| `ok` | the committed prefix is the whole file |
| `torn_tail` | the tail is explained by truncation -- the file ends inside a header, a payload, or before a commit trailer |
| `corrupt_tail` | the tail is present and complete-looking but does not verify |
| `rejected` | the superblock region did not validate; there is no prefix |

Both tail statuses leave the same committed prefix and the same promotion rule.
They are distinguished because they call for different human responses: a torn
tail at end-of-file is the ordinary result of a crash mid-write, while bytes
that are present and wrong are evidence of corruption and call for quarantine
(section 8).

The distinction is structural, not a guess. A torn tail is **truncation**: the
file ends inside a transaction header, a record header, a payload, or before a
commit trailer -- there are not enough bytes to form the next structure. A tail
that has the bytes for a complete transaction header but whose begin magic is
not `NNSTXBEG`, or whose `header_bytes` is not 48, is **not** truncation: the
structure is present and complete-looking but does not verify, so it is a
`corrupt_tail`. A reader reports the two with different codes so an operator
can tell an ordinary crash from corruption.

An unfinished spool -- one whose writer is still appending -- looks exactly
like a torn tail to a concurrent reader, which is correct and intended: a
reader may not promote a transaction whose trailer has not landed, whatever the
reason it has not landed.

### 6.2 Finalizability

A spool is **finalizable** when its superblock validates and no rule violation
of section 4 or 7 was found, every required read completed, and the bounded
diagnostic set did not overflow. A read failure is an incomplete observation,
not evidence of a corrupt tail, and MUST block finalization. A torn or corrupt
tail does **not** make a spool
unfinalizable: recovery starts from the last valid committed transaction, and
that is the ordinary crash path. What blocks finalization is a committed prefix
a reader cannot account for -- an unknown record kind, a session end that seals
no accounting, accounting that seals no session end, more than one accounting
or session end, records after the session end, a record whose flags or
logical_ordinal are not what this version defines, or accounting that does not
verify (either layer, including the position tag form, the position
identity-kind registry, and position/counter consistency of section 4.4).

A spool with a valid superblock and no committed transaction is **readable and
empty**: the empty prefix is well defined. It holds no committed record and is
the one case section 7 of the contract allows to be discarded.

## 7. Accounting verification

A reader that finds an `accounting_snapshot` in the committed prefix MUST check
both layers, adapted to the capture side:

*Layer 1 -- internal identities.*

```text
runtime_accepted  = recorder_accepted        + failed_between_runtime_and_recorder
recorder_accepted = spool_committed          + lost_between_recorder_and_spool
control_accepted  = control_spool_committed  + lost_between_control_acceptance_and_spool

when control_offered_present:
control_offered   = control_accepted         + control_rejected
```

These are mathematical integer identities over encoded unsigned 64-bit
counters. Implementations MUST detect an addend larger than the left side (or
use an equivalent checked/subtractive comparison); modulo-2^64 wraparound MUST
NOT make an invalid identity valid.

`rejected_before_runtime_acceptance`, `control_rejected`, and the two
rejected-after-close counters appear in no identity except the optional
`control_offered` one. An item that was refused was never inside the stage it
was refused entry to, and adding it to one is a contract violation.

*Layer 2 -- consistency with the container.*

```text
spool_committed         == data-plane items in the committed prefix
control_spool_committed == control-plane items in the committed prefix
```

A counter that exceeds what the container holds MUST be reported as a failed
verification and MUST NOT be reconciled by trusting the number. Layer 1 alone
cannot catch this: `accounting-exceeds-prefix` satisfies every identity while
claiming ninety-nine data items over a prefix holding two.

Layer 2 is checked **at the prefix the snapshot seals**, with the running totals
at that point, not after later transactions have inflated them: a transaction
committed after the session end must not mask a summary that claimed more than
the sealing prefix held.

An `accounting_snapshot` MUST be committed in a transaction that carries no
data-plane and no control-plane items, so that its counters name a settled
prefix rather than a moment inside one. There is at most one
`accounting_snapshot` in a spool and at most one `session_end`, and an
`accounting_snapshot` MUST be sealed by a `session_end` in its own transaction;
a snapshot with nothing to seal is unaccountable, and a second snapshot or a
second session end would give a session two capture outcomes. The structural
rules of section 4.4 -- the flag bytes are 0 or 1, the position tags are 0, 1,
or 2, the inactive union members are zero, each position carries the tag form
the contract fixes for it (a loss its ordinal, a rejection its producer
identity), a producer-identity position's `identity_kind` is a registered kind
for its plane, and a position is present exactly when its related counters sum
to nonzero -- are part of verification: a snapshot that breaks them is not a
snapshot a finalizer may trust.

What these two layers establish is exactly what section 5.1 says and no more:
that the persisted counters are internally consistent, that the committed ones
match the container, and that the accounting and the session end were sealed
together. They do not establish that everything a producer handed over was
counted, and no reader-side check can: a recorder that dropped one accepted
item and wrote every counter one lower satisfies both layers. That failure is
caught by parity testing, not by a reader.

## 8. Repair and quarantine

Ordinary reading and diagnosis MUST NOT mutate a spool. Removing an invalid
tail is a **separate, explicitly requested operation** that:

1. verifies the committed prefix in full before touching anything;
2. preserves the removed bytes -- quarantine, never delete -- so the evidence
   of what went wrong survives the repair;
3. emits a report naming the file, the committed prefix end, the removed byte
   range, the transaction ids involved, the diagnostic codes, and the reason;
4. truncates only at the committed prefix end, and never at any other offset.

A spool that is unrecoverable, or one whose tail an explicit repair truncated,
MUST be quarantined rather than deleted (contract section 7). Retention is the
default; the only spool that may be discarded without ceremony is an empty one.

## 9. Versioning

- `version_major` identifies the container layout. A reader MUST refuse a major
  it does not implement, whole, without reading any transaction.
- `version_minor` may only **add** record kinds and reserved-field meanings. It
  MUST NOT change any layout defined here. A reader MUST accept an unknown
  minor and read the container normally -- and then refuse to finalize if it
  meets a record kind it does not know (section 4.2), which is the case an
  unknown minor actually produces.
- A change to any table in sections 2, 3, or 4 is a **major** version change.
  There is no negotiated feature flag and no optional field.

### 9.1 Released minors

| Minor | What it adds |
| --- | --- |
| 0 | the container as first specified |
| 1 | `task_variables = 11` in the producer-identity registry (section 4.4) |

1.1 illustrates the rule rather than bending it: the registry is an enumeration
of names, not a layout table, so adding a value changes no offset and no size,
and the `identity_kind` field it is stored in was already a `uint32`. A 1.0
reader reads a 1.1 container normally, and meets the new value only if a
task-variable record happened to be the first one a plane refused -- where it
reports an unregistered kind, which is the correct answer for a reader that
does not know the name.

## 10. Test vectors

`vectors/` holds the normative binary vectors and `vectors/index.json`
describes them. The vectors are language-neutral: an implementation in any
language reads the `.spool` files and the expectations in the index, and needs
no Python for it.

Each index entry carries the file name, its length, its SHA-256, one sentence
saying why the vector exists, and the `expect` object -- the scan verdict a
conforming reader MUST produce:

| Field | Meaning |
| --- | --- |
| `status` | section 6.1 |
| `committed_prefix_end` | first byte after the last committed transaction |
| `committed_transactions`, `last_transaction_id` | the committed prefix |
| `data_items`, `control_items` | summed over the committed prefix |
| `session_end_present`, `capture_outcome` | section 4.5 |
| `requested_terminal_intent`, `primary_fault_committed`, `terminal_reason` | section 4.5 terminal provenance |
| `durability_policy`, `durable_extent_bytes` | section 5 |
| `finalizable` | section 6.2 |
| `codes` | the diagnostics, in the order they were found |

The vectors cover: a prepared spool with no records; a spool still open; a
clean end; the same session under all three durability policies; a future minor
version; a legally incomplete session with losses and first positions; a
committed fault record; an abort escalated to faulted by a drain fault; four
truncation points; five corruption sites; a
duplicated and a skipped transaction id; a broken back link; count
disagreements; an unknown record kind; an unknown major version; a
`logical_ordinal` that does not match its owning item; a record or trailer
flags field that is not zero; a `session_end` with an undefined enum; a
checkpoint whose policy is not the superblock's; an accounting flag, position tag, position/counter consistency, position
tag-form, or position identity-kind violation; a duplicated accounting snapshot
or session end; an accounting snapshot with no session end; an accepted-item
loss with no first-loss ordinal latched; and two corrupt tails
-- a complete header with the wrong magic and one with the wrong
`header_bytes` -- so that the line `every rule violation of sections 4 and 7`
is covered by a vector rather than asserted.

`tools/generate_vectors.py --check` verifies that the committed vectors and
index reproduce **byte for byte** from the generator. Every input is a constant
in that file -- no clock, no random source, no filesystem state -- so a
difference means the format or the reference changed, which is the only thing
the gate is there to catch.

## 11. Diagnostic codes

Wording may change; a code may not. Codes marked "ends the prefix" stop the
scan where they are found.

| Code | Ends the prefix | Meaning |
| --- | --- | --- |
| `SPOOL-001` | rejects | file magic is not `NRLSPOOL` |
| `SPOOL-002` | rejects | container major version not implemented |
| `SPOOL-003` | rejects | superblock checksum mismatch |
| `SPOOL-004` | rejects | superblock region truncated |
| `SPOOL-005` | rejects | plan document does not match its checksum and fingerprint |
| `SPOOL-006` | rejects | a fixed superblock field holds an undefined value |
| `SPOOL-010` | yes | torn tail: the file ends inside a transaction |
| `SPOOL-011` | yes | transaction header checksum mismatch |
| `SPOOL-012` | yes | commit trailer checksum mismatch, or a trailer that does not describe its transaction |
| `SPOOL-013` | yes | transaction body checksum mismatch |
| `SPOOL-014` | yes | transaction id is not the successor of its predecessor (duplicate or gap) |
| `SPOOL-015` | yes | back link does not name the previous transaction |
| `SPOOL-016` | yes | header and trailer disagree on the record count |
| `SPOOL-017` | yes | record header checksum mismatch |
| `SPOOL-018` | yes | record payload checksum mismatch |
| `SPOOL-019` | yes, or rejects | padding is not zero: in a record it ends the prefix, in the superblock region it rejects the spool |
| `SPOOL-020` | yes | a transaction commits no record |
| `SPOOL-021` | yes | trailer item counts disagree with the records |
| `SPOOL-030` | no | committed record of an unknown kind |
| `SPOOL-031` | no | a transaction was committed after the session end |
| `SPOOL-032` | no | a session end that seals no accounting snapshot |
| `SPOOL-033` | no | an accounting snapshot sharing its transaction with an item |
| `SPOOL-034` | no | an accounting identity does not hold |
| `SPOOL-035` | no | accounting counters exceed the committed prefix |
| `SPOOL-036` | no | a checkpoint claims bytes beyond its own transaction |
| `SPOOL-037` | no | spool accounting claims an origin other than the recorder |
| `SPOOL-038` | no | a container-owned record payload is not its fixed size |
| `SPOOL-039` | no | a record's `logical_ordinal` does not match its owning item, or a record with no owning item carries a nonzero ordinal |
| `SPOOL-040` | no | a record or trailer `flags` field is not the version-1 zero |
| `SPOOL-041` | no | a `session_end` enum or boolean field (`requested_terminal_intent`, `capture_outcome`, `primary_fault_committed`) holds an undefined value |
| `SPOOL-042` | no | a checkpoint's `durability_policy` is not the superblock's |
| `SPOOL-043` | no | an accounting flag byte is neither 0 nor 1 |
| `SPOOL-044` | no | an accounting position tag is not 0, 1, or 2, or an inactive union member is not zero |
| `SPOOL-045` | no | an accounting position is present while its counter is zero, or absent while its counter is nonzero |
| `SPOOL-046` | no | more than one accounting snapshot was committed |
| `SPOOL-047` | no | more than one session_end record was committed |
| `SPOOL-048` | no | an accounting snapshot was committed without a session_end in its transaction |
| `SPOOL-049` | yes | the tail is a complete header that does not begin a valid transaction (wrong begin magic, or `header_bytes` is not 48) -- corruption, not truncation |
| `SPOOL-050` | no | an accounting position is present with a tag form its event does not use (a loss carrying a producer identity, or a rejection carrying an ordinal) |
| `SPOOL-051` | no | an accounting position's `identity_kind` is not a registered producer kind, or is a kind from the other plane |

## 12. Conformance

A conforming **writer** appends only, writes every field as specified, writes
the commit trailer last, syncs according to the declared policy, never rewrites
a committed byte, and never forges a session end.

A conforming **reader** validates the superblock region before anything else,
promotes only committed transactions, stops at the first framing failure,
reports the durable extent its policy supports and no more, verifies the
container-owned payloads in full -- the `flags` fields, the `session_end`
enums, the checkpoint policy, the accounting flag bytes and positions, the
position tag form, the position identity-kind registry, and position/counter
consistency, the accounting multiplicity and pairing, and both accounting
layers -- refuses to finalize on any unknown record kind or any conformance
finding, and mutates nothing.

A conforming implementation reproduces the verdict in `vectors/index.json` for
every vector, and reproduces the vectors themselves byte for byte if it can
write.

## 13. What this specification does not define

- The interior of a frame, signal-block, discontinuity, signal-gap, control, or
  fault payload (M6-05).
- How a committed prefix becomes a canonical NRF session, and the accounting
  the finalizer adds there (M6-08).
- Diagnosis, dry-run, explicit repair, resume, and abandon operations as public
  behavior, or the recovery-rebuilt accounting in NRF (M6-09).
- Any replay semantics whatsoever (contract section 8; M6-10 and later).
