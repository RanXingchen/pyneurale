# Neurale Recording Format (NRF) v1

Status: normative logical specification for NRF version 1.0. The schemas and
test vectors in this directory are part of the specification.

Transport update: PyNeurale now requires the
[single-file package envelope](../package-v1.md). It supersedes this document's
directory-container and optional `.nrf.zip` transport wording, not its logical
object layout, schemas, journal visibility or checksum rules. The layout below
is an entry namespace inside one `.nrf` file; append operations occur only in
private finalization workspaces. Public directory input is not supported.

This document is independent of any implementation and does not track the state
of one. A session is written against this specification, not against the code
that happened to produce it, and an implementation is conforming or not
regardless of who wrote it. PyNeurale's `neurale.io.nrf` is one implementation;
its conformance is established by checking it against the artifacts here, never
by amending them.

## 1. Scope and terminology

NRF is a language-neutral recording contract for typed neural and behavioral
streams, derived features, events, trials, experiment records, discontinuities,
drops, faults, and session termination. A live recording is a directory named
`<session>.nrf/`. A closed directory may later be packaged as
`<session>.nrf.zip`; a zip is never an append target.

The keywords MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, and
MAY are normative. Intervals are half-open `[start, end)`. Array extents count
logical records or observations, never bytes. All registry order is descriptive;
identity is by stable ID.

NRF persists values, never Python pickle data or Python implementation class
names. Large or appendable typed data MUST use Zarr v3 arrays. JSON is used only
for manifests, descriptors, journal/checkpoint records, reports, and bounded
metadata.

## 2. Required directory layout

```text
session.nrf/
  manifest.json
  zarr.json
  streams/<stream-id>/data/zarr.json
  streams/<stream-id>/data/c/...
  streams/<stream-id>/timestamps/zarr.json       # explicit timing only
  streams/<stream-id>/timestamps/c/...
  streams/<stream-id>/discontinuities/columns/<field>/zarr.json  # per-stream record set
  streams/<stream-id>/discontinuities/columns/<field>/c/...
  streams/<stream-id>/discontinuities/validity/<field>/...      # nullable fields
  feature_sets/<feature-set-id>.json
  metadata/transactions/<transaction-id>/<metadata-object-id>.json
  records/<record-kind>/<schema-id>/...          # Zarr v3 column arrays
  journal/transactions.jsonl
  journal/checkpoints/<checkpoint-id>.json
  journal/head.json
  indexes/<index-id>/...                         # derived, rebuildable
  checksums/objects.jsonl                        # optional checksum index
  recovery/<recovery-id>.json                    # recovery reports
  .staging/<transaction-id>/...                  # never reader-visible
  .staging/tails/<target-id>/...                  # uncommitted writer tails
```

`manifest.json`, `journal/transactions.jsonl`, checkpoint files, and `head.json`
are REQUIRED. `streams/`, `feature_sets/`, and `records/` are required when the
manifest declares corresponding objects. `indexes/` and `checksums/objects.jsonl`
are caches and MUST be rebuildable from committed data and the journal. A reader
MUST ignore `.staging/` and any object beyond a committed extent.

Each Zarr array MUST be a conforming Zarr format 3 array. NRF does not define a
second chunk encoding. The root `zarr.json` is a Zarr v3 group.

Every NRF-internal path in the manifest, journal, checkpoint, head cache, and
recovery report is one canonical relative POSIX path. `/` is the only
separator. Paths have no leading or trailing `/`, empty segment, `.` or `..`
segment, colon, Windows drive prefix, backslash, whitespace, or control
character. Each segment contains only lower-case ASCII letters, digits, `.`,
`_`, and `-`, does not end in `.`, and has no Windows-reserved basename
(`con`, `prn`, `aux`, `nul`, `com1` through `com9`, or `lpt1` through `lpt9`),
including when that basename has an extension.
Writers MUST emit exactly this canonical spelling before performing ownership,
checksum, or create-if-absent checks; readers MUST reject rather than normalize
a non-canonical spelling. Portable ownership comparison uses the
case-insensitive, trailing-space/dot-normalized segment form even though a
conforming path is already lower-case and has neither suffix.

`manifest.json` and `zarr.json` are reserved files. The top-level `journal/`,
`.staging/`, `metadata/`, `recovery/`, `checksums/`, `indexes/`, and
`feature_sets/` trees are reserved control namespaces and cannot contain
manifest-owned payload arrays. Stream payloads have fixed paths
`streams/<stream-id>/data` and, for explicit timing,
`streams/<stream-id>/timestamps`. Non-discontinuity record sets have fixed
paths `records/<record-kind>/<schema-id>`; per-stream discontinuities use
`streams/<stream-id>/discontinuities`. Their physical column and validity
arrays remain descendants of those record-set paths.

## 3. Stable identity and references

`session_id` is a lower-case canonical UUID string. Clock, stream, schema,
feature-set, unit, channel, electrode, event-schema, trial-schema, record-schema,
index, checkpoint, event, and trial IDs are non-empty stable ASCII identifiers
matching `^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$`. They are immutable and MUST NOT
be reused for a different object within a session.

Transaction IDs are `tx-` followed by a 16-digit, zero-padded decimal sequence;
checkpoint IDs are `checkpoint-` followed by the same representation. Journal
`sequence` starts at 1 and increases by exactly one per record. References MUST
resolve within the manifest or, for event/trial instances, within their declared
record set. Duplicate IDs are invalid even when the duplicated objects are equal.

Native numeric `SessionId`, `SchemaId`, `SignalId`, `ClockDomainId`,
`FeatureSetId`, and channel-set identifiers map to stable NRF strings through
explicit descriptor fields such as `native_id`; their numeric values are not
globally stable identities.

When a current typed object has no string identity, the future writer must
materialize one before its first append. Event IDs are either caller-supplied or
derived from the record-schema ID and the monotonic event-record sequence;
trial IDs use the same rule while retaining the current integer `Trial.trial_id`
as a typed source field. The chosen ID is then persisted and reused. It must not
be regenerated from mutable labels, timestamps, or Python object identity.

## 4. Manifest and compatibility

`manifest.json` MUST validate against `manifest.schema.json`. Required content
includes:

- format name and `{major, minor}` version;
- session ID, UTC creation time, writer name and version;
- checksum and canonical-JSON algorithms;
- clock, unit, channel, electrode, schema, stream, feature-set, codec, and record
  registries;
- event and trial schemas represented by record descriptors;
- dtype, byte order, axes, timing, clock, units, source links, segmentation, and
  committed extent for each stream;
- authoritative checkpoint/transaction sequence and a cached committed-extent
  map.

Version 1 readers MUST reject any major version other than 1. A higher minor
version is readable only when all unknown required features are absent. Unknown
members under `extensions`, and unknown members whose JSON Schema explicitly
allows them, are optional minor additions: a read/modify/write tool MUST preserve
them byte-for-value semantically, while a read-only tool MAY ignore them. Unknown
enum values, unknown required fields, or unknown fields outside such extension
points MUST be rejected. Enum values are stable lower-case strings.

Manifest validation has two mandatory, ordered layers. First,
`manifest.json` MUST validate against `manifest.schema.json`. Second, it MUST
pass the version 1.0 rules in
[`semantic-validation.md`](semantic-validation.md). JSON Schema is
authoritative for locally expressible structure and conditional fields; the
semantic-validation specification is authoritative for registry references,
rank/shape relationships, extent agreement, and other cross-object rules.
Passing either layer alone is insufficient.

The manifest's `commit` object is a cache. Journal commit records and a valid
checkpoint are authoritative. The manifest cache MUST NOT lead the journal. If
it lags, readers reconstruct current committed extents by replay; if it leads,
validation fails.

## 5. Clocks, schemas, and stream payloads

A clock descriptor states its type, optional exact rational rate, epoch,
synchronization domain, offset, and drift. Rates use positive integer
`numerator` and `denominator`; binary floating-point rates are not stored as the
normative value. Cross-clock alignment requires an explicit synchronization
relation or snapshots. Absence of one forbids implicit alignment.

A stream descriptor is fixed before its first committed append (the
session-wide registry freeze is in section 8). It defines:

- `kind`: `neural`, `behavioral`, `feature`, or `spike`;
- payload `dtype`: `int16`, `int32`, `int64`, `uint8`, `uint32`, `uint64`,
  `float32`, `float64`, `bool`, or `utf8` where permitted by the schema;
- `endianness`: `little`, `big`, or `not_applicable` (`bool`, `uint8`, `utf8`);
- axes and exact axis order;
- Zarr v3 array path, shape, chunk shape, and codec reference;
- clock, timing, units, channels, source streams, and segment semantics.

Dense sampled neural and behavioral payloads use `(sample, channel)`. Feature
payloads use `(observation, feature)` and reference exactly one immutable
feature-set descriptor. Spike waveform payloads, when present, use
`(spike, sample, channel)` and companion per-spike arrays. No implicit dtype,
layout, endianness, codec, provider, or algorithm conversion is allowed.

Timing is either:

- `regular`: an exact rational rate, a clock, the time/index of the first
  observation in each segment, and implicit subsequent positions; or
- `explicit`: a one-dimensional timestamp Zarr array with one timestamp per
  leading-axis item, using the declared clock and `int64` nanoseconds.

Feature streams with regular timing use window-center timestamps, including
fractional centers expressed exactly in clock nanoseconds where representable.
Their feature descriptor records ordered names, ordered units, source stream,
algorithm name/version, window length, shift, and `window_center` reference.

Every stream has a segment policy. Segment IDs are stable within the stream.
A stream's segment policy references a `discontinuities`-kind record schema (see
section 6); that schema's path is the per-stream discontinuity record-set location
and the committed-extent key for its chunks. Discontinuity records contain
discontinuity ID, stream, old/new segment IDs, expected and actual sample
indices, optional missing count/device ticks, reason, and clock time. The
discontinuity record set is a typed record set like any other: its fields, dtype,
endianness, column layout, chunk length, codecs, nullable validity arrays,
clock, and committed extent are all declared by the referenced record schema, so
a reader can decode every chunk. No regular grid, feature window, waveform, event
interval, or trial interval may be inferred across a discontinuity. Missing
samples are not represented by invented payload values.

## 6. Typed record sets

Bounded metadata may be JSON, but appendable record columns use Zarr v3 arrays.
Each `record_schema` fixes its fields, dtypes, byte order, nullability, exact
column-array paths, codecs, leading-axis chunk length, trailing value shape,
units, clock, and references.
Its `primary_key` MUST name a non-nullable scalar field with dtype `utf8`,
`int64`, or `uint64`. Floating-point, boolean, shaped, and nullable primary-key
fields are invalid descriptors. The M4-04 writer/reader validation layer is
responsible for rejecting missing or duplicate primary-key values in committed
record data.
`clock_id` is mandatory for every NRF v1 record schema and MUST resolve in the
non-empty clock registry. NRF v1 has no implicit timeless record obtained by
omitting `clock_id`; a future timeless record model would require an explicit
timing mode.
Every column has leading `record` axis and the same committed extent. A nullable
field has a required `uint8` validity array at `validity_path` using the
manifest's byte codec; a missing value is defined by validity zero, never a
dtype-specific sentinel. UTF-8 columns use the declared Zarr v3 variable-length
UTF-8 codec. The record-set `path` is its committed-extent key and atomically
covers all of its columns.
Required logical kinds are:

- `events`: stable event ID, `[onset, onset + duration)`, label/code/source,
  optional value, sample index, confidence, clock, stream, segment, and attrs;
- `trials`: stable trial ID, `[start, stop)`, label, target ID/vector, outcome,
  block, clock, segment references, and attrs;
- `experiment_state`: time, clock, previous state, new state, reason, trial ID;
- `commands`: time, clock, command type, ordered typed value vector, source,
  destination, trial ID, and status;
- `task_variables`, `labels`, `targets`, and `assistance`: explicitly named,
  typed observations with time/clock/trial/source links and units;
- `drops`: observer/edge, policy, first and last frame sequence, count, time,
  affected stream/segment, and reason;
- `faults`: the stable string encodings of native fault code/status/stage plus
  component, detail, session, runtime generation, frame/schema/clock/stream,
  sample/device tick, and detection time;
- `discontinuities`: per-stream segment-transition records with stable
  discontinuity ID, owning stream, previous/next segment IDs, expected and
  actual sample indices, optional missing sample count and device ticks,
  reason, and clock time. One `discontinuities` record schema is referenced by
  each stream's segment policy; its path is `streams/<stream-id>/discontinuities`;
- `native_frames`, `native_signal_blocks`, `native_discontinuities`,
  `native_signal_gaps`, `session_accounting`: the native replay ledgers and the
  session accounting summary. These are **minor additions**, introduced at
  version 1.1 and defined by
  [`extensions/native-replay-v1/`](extensions/native-replay-v1/README.md). A
  session that declares any of them MUST declare minor version 1; a session that
  declares none of them is unaffected and stays at minor version 0. They are
  ordinary typed record sets and obey every rule in this section;
- `session_termination`: exactly one normal, aborted, or faulted terminal row
  with `termination_id`, `time_ns`, `termination_kind`, non-empty `reason`,
  `last_transaction_id`, and nullable `fault_id`. Its fixed path is
  `records/session_termination/termination-v1`, its extent is zero while the
  session is open and exactly one after termination, and its target is sealed
  by the transaction that appends the row.

Targets, external/internal commands, assistance values, task labels, block
identity, and state transitions are distinct typed record sets; they MUST NOT be
collapsed into an all-double row. Normal end-of-stream is a
`session_termination` record with `termination_kind: normal`. A normal row has
`fault_id: null`; a faulted row MUST reference a committed fault, while an
aborted row MAY reference one. Abort/shutdown faults are recorded as faults plus
an abnormal termination. Every termination kind uses the same typed-row plus
journal-reference protocol below; observer drops are records, not
discontinuities, unless they also create a known stream gap.

## 7. Checksums and canonical JSON

The only NRF v1 checksum algorithm is SHA-256. Digest text is 64 lower-case
hexadecimal characters.

- A committed immutable Zarr metadata snapshot or chunk object checksum covers
  the exact final object bytes after compression/codec processing and before
  filesystem storage. The active `zarr.json` files are rebuildable caches and
  are not committed objects.
- A journal/checkpoint/report `record_checksum` covers RFC 8785 (JCS) canonical
  UTF-8 JSON of that object with `record_checksum` omitted.
- Each JSONL journal line is the same canonical JSON object including its
  checksum, followed by one LF byte. Partial lines are invalid and invisible.

JCS input MUST satisfy the I-JSON constraints required by RFC 8785. Writers
MUST reject duplicate property names, non-finite numbers, lone Unicode
surrogates, and integer values outside the interoperable exact range
`[-9007199254740991, 9007199254740991]`. Larger signed or unsigned integer
payloads belong in typed Zarr arrays; when such a value is required in
checksummed JSON, its schema MUST define a decimal string rather than a JSON
number. Negative zero canonicalizes as `0`.

Implementations MUST use RFC 8785 number serialization, string escaping,
recursive property sorting by UTF-16 code units, and final UTF-8 encoding.
Ordinary key-sorted JSON serialization is not conforming. The checked-in
canonical vectors cover the RFC primitive example, supplementary-plane and
Unicode property names, control/string escaping, negative zero, finite
small/large floats, exact integer boundaries, and nested property ordering.
All NRF v1 JSON `date-time` values use UTC with a mandatory trailing `Z`, a
year in `0001..9999`, and seconds in `00..59`; year zero and leap-second
timestamps are not supported.

## 8. Append transaction and visibility

There is one logical writer per live session. All registries are frozen before
the first committed transaction: the writer registers every clock, unit,
channel, electrode, schema, stream, feature set, codec, and record schema,
prepares and freezes the session schema, then begins appending. After the
first commit, registries are immutable for the life of the session — no new
clock, unit, channel, electrode, schema, stream, feature set, codec, or record
schema may be registered. The journal carries no registry content; registry
identity and descriptors live only in the manifest. Because the manifest
written before the first transaction already contains every descriptor needed
to decode any committed extent, a stale manifest cache can never lack a
descriptor for committed data, and recovery never has to reconstruct a registry
from the journal. An append that touches one or more arrays follows this exact
order:

1. Allocate the next transaction ID and stage every new immutable Zarr chunk
   and versioned metadata snapshot under `.staging/<transaction-id>/` using a
   final relative path that has never existed.
2. Flush staged bytes and calculate SHA-256 over every exact object. Calculate
   each array's `before` and `after` leading-axis extent.
3. Append and flush one checksum-valid `prepare` journal record. It names all
   staged objects, binds each object to exactly one logical extent target and
   physical array, records every chunk coordinate and record-column role, lists
   the exact object dependencies of every extent transition, and names the
   previous committed transaction. Preparation is not reader-visible data.
4. Atomically rename staged objects to their new final paths on the same
   filesystem. Object promotion MUST use create-if-absent semantics and MUST
   fail rather than replace any existing object. Append and flush the matching
   `commit` record. The complete, checksum-valid commit line is the single
   logical visibility point. A reader MUST cap every array at the extents in
   committed transactions even if caches or orphan objects expose more data.
5. Rebuild or atomically replace the active `zarr.json`, `manifest.json`, and
   `journal/head.json` caches, each by a single rename, so each cache's metadata
   version, last transaction, journal sequence, committed extents, and sealed
   targets equal the new commit. The renames are independent; a crash between
   them leaves the caches at different journal sequences, which is recoverable
   because each is non-authoritative. Failure here leaves lagging caches; it
   does not undo the commit.
6. Periodically write a checksum-valid checkpoint to a temporary file and rename
   it into `journal/checkpoints/`, append a `checkpoint` journal record, then
   atomically replace `head.json` and the manifest checkpoint cache, each by a
   single rename.

Prepare and commit records share a transaction ID. A transaction cannot commit
without exactly one earlier prepare, cannot commit twice, and commits in
transaction-number order. Extents never decrease. `before` must equal the last
committed extent, and every array referenced by a transaction must have
`after >= before`; a transition that advances the extent (`after > before`) must
have a positive `after - before` and be backed by staged objects, while a
seal-only transition may have `after == before` (see section 8.3). Every object
MUST be assigned to exactly one extent transition. An object cannot support
multiple targets, an unassigned object is invalid, and object paths are unique
across transactions. Within one prepare, both final `path` and `staged_path`
values are unique. Every object uses the exact staging path
`.staging/<transaction-id>/<final-path>`; no alternate name, shared staged
object, or staged/final path equality is allowed. This makes each promotion a
one-to-one create-if-absent rename.

### 8.1 Session termination transaction

Termination is not represented independently in the typed record store and the
journal. The journal `termination` record is the terminal reference to the sole
committed typed `session_termination` row.

For every termination kind, the writer MUST:

1. allocate the final transaction ID;
2. append exactly one `session_termination` row whose
   `last_transaction_id` equals that final transaction ID;
3. seal the `session_termination` record set in the same transaction;
4. commit the final transaction; and
5. immediately append the journal `termination` record.

The journal record contains `termination_record_id`, `termination_kind`,
`time_ns`, `reason`, nullable `fault_id`, and `last_transaction_id`. A reader
MUST resolve `termination_record_id` to the sole typed row and require all six
values to agree. The final transaction's prepare record MUST contain the
`session_termination` extent transition from 0 to 1 with
`seals_target: true`; this proves that the referenced row belongs to the final
committed transaction. No checkpoint or other journal record may appear
between that commit and the journal termination record, and no record may
follow journal termination.

For `normal` termination, the final transaction also seals every other
appendable target in the frozen manifest, including targets whose extent
remains zero. Consequently, replay accepts normal termination only when the
sealed-target set equals the full frozen appendable-target set. `faulted` and
`aborted` termination MAY leave unrelated targets open, but MUST still append
and seal exactly one typed termination row. A journal termination without that
committed typed row is invalid. `faulted` MUST name exactly one fault row
committed before the final transaction. `aborted` MAY use `fault_id: null`; if
non-null, it has the same committed-before-final resolution requirement.
Finding the ID in an uncommitted physical tail or in the final termination
transaction does not satisfy this rule. Replay obtains this association from
decoded committed fault rows together with their owning committed transaction,
not from uncommitted objects or manifest extent alone.

### 8.2 Extent and physical-object ownership

The frozen manifest defines the physical arrays required by each logical
target. A stream data or explicit-timestamp target owns one physical Zarr
array. A record-set target owns every field `array_path` in its referenced
record schema and, for each nullable field, its `validity_path`. A prepare
extent transition repeats this frozen association in `required_array_paths`;
for a record set it also names `record_schema_id`. The repeated values MUST
equal the manifest contract.

Every prepared object declares `target_path`, `array_path`, and `logical_role`.
A chunk object also declares its full `chunk_coordinate`. Record-column and
validity chunks additionally declare `record_schema_id` and `column_name`.
Every transition lists its dependencies in `object_paths`; each listed object
MUST name that transition's target and one of its required arrays.
Final paths are role-specific: chunk roles use
`<array-path>/c/<chunk-coordinate>`, metadata snapshots use
`metadata/transactions/<transaction-id>/<metadata-object-id>.json`, and index
objects use `indexes/<index-id>/<index-object-path>`. An index object cannot
occupy any other cache or control namespace. NRF v1 has no generic JSON
transaction-object kind; JSON control files and descriptor caches follow their
separate update rules and are not prepared as payload objects.

For every advancing transition, chunk coordinates for every required physical
array MUST exactly cover `[before, after)` on the leading axis using the
declared `chunk_length`, and MUST cover every trailing chunk coordinate implied
by the frozen array shape and chunk grid. A record set therefore advances only
when every required column and every required validity array advances together.
Each advancing physical array also has exactly one transaction-unique
`metadata_snapshot` object associated with it. A missing, extra, unrelated,
multiply assigned, or incorrectly coordinated object makes the prepare
invalid. The only transition allowed to have no dependent object is the
non-advancing seal-only transition defined below.

### 8.3 Chunk immutability and partial tails

A committed Zarr chunk is immutable for the lifetime of the session. A prepare
record's object `disposition` is always `create`; NRF v1 has no replace,
supersede, or in-place chunk operation. Consequently, a transaction can never
rewrite the partially filled final chunk of a committed extent.

For every stream data array, timestamp array, discontinuity array, and record
set, the prepare extent transition records `chunk_length` and `seals_target`.
While a target is open:

- `before` MUST be aligned to `chunk_length`;
- an unsealed `after` MUST also be aligned to `chunk_length`; and
- every promoted chunk coordinate MUST be new and strictly beyond all committed
  chunk coordinates for that target.

Append calls smaller than a chunk remain in the writer-owned tail under
`.staging/tails/` or equivalent bounded writer storage. They do not create a
prepare record, do not advance committed extent, and are invisible to readers.
Multiple append calls may fill that tail. Once full, the writer encodes one new
Zarr v3 chunk, commits it, and clears the tail. A crash may lose or quarantine
this uncommitted tail but cannot alter committed bytes.

Normal finalization may commit one final short chunk by setting
`seals_target: true`. Its `before` remains chunk-aligned, its `after` is the
exact logical extent, and the chunk uses its normal new coordinate with the
unwritten suffix represented only by the Zarr fill value. A sealed target can
never be appended again. The final transaction seals every still-open
appendable target and appends the normal typed `session_termination` row before
its commit; the journal termination reference follows that commit. A faulted
or crashed session may retain an uncommitted tail; recovery does not promote it
implicitly.

A target that stays at extent 0 for the whole session (for example a record
set registered for an experiment that never produced that record kind) is still
open and must also be sealed at finalization. Such a target is sealed by a
seal-only extent transition with `before == after == 0` and
`seals_target: true`. The prepare record for a seal-only finalization carries no
staged objects: it changes no physical shape, so it writes no chunk and no
metadata snapshot. An objectless prepare is valid only when every one of its
extent transitions is such a seal-only transition; a transition that advances
the extent (`after > before`) must be backed by staged objects. Sealing the
empty target adds its path to `sealed_targets` so a reader and recovery treat it
as non-appendable exactly like a short-chunk seal.

This rule handles a committed extent of 900 with chunk length 1024 by making
that state possible only after the target has been sealed, in which case a
further append is rejected. Before finalization, the corresponding open target
would have committed extent 0 and retain those 900 items in its uncommitted
tail. Appending another 100 items to that open target changes only the tail; it
cannot rewrite a committed chunk.

### 8.4 Metadata copy-on-write

Mutable active Zarr metadata is a cache, not the authority for visibility.
Every transaction that changes physical shape writes a checksum-covered,
transaction-unique metadata snapshot below
`metadata/transactions/<transaction-id>/<metadata-object-id>.json`. The
snapshot names its target array path and complete Zarr v3 metadata. Metadata
snapshots are outside Zarr array nodes so they do not create illegal child
nodes in the Zarr hierarchy. The commit selects the snapshot.
Only after commit may the writer materialize it as the active `zarr.json`.
Recovery selects the last committed snapshot and reconstructs active metadata,
so a crash before commit cannot make an uncommitted shape authoritative and a
crash after commit cannot hide committed data merely because cache replacement
was incomplete.

Index updates may be part of a transaction, but their final paths are confined
to `indexes/<index-id>/...` and indexes never determine data visibility.
Checksum indexes are likewise caches but are not represented by the `index`
logical role in NRF v1. Zarr fill values beyond the committed extent are not
observations.

### 8.5 head.json and standalone descriptor caches

`journal/head.json` and `feature_sets/<feature-set-id>.json` are rebuildable
caches, not authoritative state. They carry no `record_checksum`: like
`manifest.json` and the active `zarr.json` files, their integrity is
established by consistency with journal replay, not by an intrinsic checksum.

`head.json` is the fast pointer to the current committed session state. It
MUST validate against `head.schema.json`. Its `journal_sequence`,
`last_transaction_id`, `last_checkpoint_id`, `committed_extents`, and
`sealed_targets` are the same fields as the manifest `commit` cache, but
`head.json` and `manifest.json` are independent files: the update protocol
(steps 5 and 6) replaces each by a single atomic rename, and a crash between
the two renames may leave them at different journal sequences. That is
acceptable, because neither cache is authoritative — journal commit records and
the last valid checkpoint are. Each cache is validated against journal replay
independently: a cache MUST NOT lead replayed journal state (leading is a
validation failure); either cache MAY lag, in which case a reader or recovery
reconstructs current state by replay. The two caches therefore need not equal
each other at a crash boundary; each only needs to not lead replay. A reader
MUST NOT use `head.json` or manifest cached extents that lead replayed journal
state, and MUST NOT use a `last_checkpoint_id` that does not reference a
checksum-valid checkpoint journal record.

Each `feature_sets/<feature-set-id>.json` file is a convenience copy of one
entry from the manifest `feature_sets` registry. The manifest registry is
authoritative for feature-set identity and content. A standalone descriptor
file MUST be byte-for-value equal to the manifest `feature_sets` entry with
the same `id`; if it differs, the manifest wins and the standalone file is a
stale cache that a reader MUST rebuild or ignore. Standalone descriptor files
are not transaction objects: they do not appear in prepare/commit records, are
not staged under `.staging/`, and are not listed in a checkpoint's
`committed_objects`. They are materialized from the manifest registry after
commit, like the active `zarr.json` caches, and are required when the manifest
declares feature sets.

## 9. Checkpoints, recovery, and reader rules

A checkpoint contains its ID, journal sequence, last committed transaction,
all committed stream/record extents, sealed targets, index build positions, and
every committed object's checksum plus its logical target, physical array,
role, chunk coordinate, and record-column association. This preserves the
extent/object ownership graph when recovery begins at a checkpoint rather than
at journal sequence 1. A checkpoint is valid only when its checksum is valid
and its state equals replay through its journal sequence.

Recovery is read-only with respect to committed logical data:

1. select the newest checksum-valid checkpoint referenced by a valid checkpoint
   journal record, or the empty session state;
2. replay complete checksum-valid records after it in sequence;
3. validate prepare/commit order, references, extents, and every committed
   object's checksum;
4. ignore incomplete lines, prepares without commits, writer tails, staged
   objects, physical data beyond committed extents, and newly created orphan
   final objects;
5. restore active `zarr.json` caches from the last committed metadata snapshots
   and rebuild derived indexes from committed extents; and
6. emit a `recovery-report.schema.json` report listing retained commits,
   ignored records, missing/corrupt objects, quarantined paths, and rebuilt
   indexes.

A recovery tool MAY move uncommitted/orphan objects to a quarantine directory
only when explicitly requested. It MUST NOT truncate, rewrite, repair, or
silently modify committed bytes. A missing or checksum-invalid committed object
is corruption and makes the affected committed range unreadable; it is not an
uncommitted tail.

Readers expose only transactions at or before the last valid commit and only
leading-axis items below committed extents. They MUST verify required metadata
and references before returning typed values. Readers MUST NOT use manifest
cached extents that lead replayed journal state.

## 10. Mapping PyNeurale data

Current `SignalArray`, `FeatureMatrix`, `EventSeries`, `TrialTable`, `Recording`,
`Clock`, channel/electrode descriptors, native `StreamSchema`,
`FeatureSetDescriptor`, `StreamMessage`, discontinuity, observer-drop, and fault
contracts map to the logical objects above. The mapping preserves axis order,
dtype, sample/observation timing, clocks, units, channels, source links, segment
boundaries, and immutable schema identity.

## 11. Normative artifacts

- `manifest.schema.json`: manifest and descriptor definitions.
- `journal-record.schema.json`: prepare, commit, checkpoint, and termination
  journal records.
- `checkpoint.schema.json`: checkpoint snapshots.
- `recovery-report.schema.json`: explicit recovery results.
- `head.schema.json`: `journal/head.json` committed-state pointer cache.
- `test-vectors.json`: canonical language-neutral examples and expected
  SHA-256 values.
- `extensions/native-replay-v1/`: the version 1.1 native replay ledger
  extension -- its own README, machine-readable schema, and fixtures. An
  implementation that does not write native replay ledgers needs none of it.

`tools/` holds the executable reference implementations of this specification:
`semantic_validation.py` (the version 1.0 reference validator), `jcs.py` (RFC
8785 canonicalization), and `generate_test_vectors.py`, which regenerates
`test-vectors.json`. They are informative rather than normative — where a tool
and this document differ, this document governs — and they are not part of any
installed package.

Where prose and schema differ, the stricter requirement applies. A future minor
revision must resolve any ambiguity without changing v1 committed meaning.
