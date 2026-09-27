# NRF v1 extension: native replay ledgers (`neurale.native_replay`, version 1)

Status: normative minor extension to NRF version 1. It adds five record kinds
and one manifest/journal extension namespace. It changes no v1 committed
meaning: a session that does not use it is unaffected, and a v1.0 session stays
valid exactly as written.

**Semantics live elsewhere.** What the ledgers are *for* -- the acceptance
ladder, the data-message ordinal, exact replay, projection, coverage,
completeness accounting, session outcomes -- is fixed by
`docs/development/native_recording_replay.md` in the PyNeurale repository, which
is the single normative source for recording and replay behaviour. This document
adds only what that contract cannot express: on-disk shape. Where the two touch,
the contract governs meaning and this document governs encoding.

## 1. Version and compatibility

A session that declares any record schema of a kind defined here, or the
manifest extension of section 4, MUST declare `version: {"major": 1, "minor": 1}`
in its manifest. A session that uses none of them MUST keep
`{"major": 1, "minor": 0}`.

- A reader that understands only v1.0 MUST reject a v1.1 session that carries
  these kinds: they are an unknown *required* feature, not an ignorable
  addition, because the record sets they name hold committed data.
- A v1.1 reader MUST read a v1.0 session unchanged. Nothing here is required of
  a session that does not use it, and no v1.0 field changes meaning.
- The extension is identified in `extensions` by the exact namespace key
  `neurale.native_replay`. Unknown members inside that object follow the v1 rule
  for extension members: a read/modify/write tool preserves them, a read-only
  tool may ignore them.

## 2. Record kinds

Five kinds are added to `record_schema.kind`:

| Kind | Schema ID | Record-set path | Primary key | Chunk length |
| --- | --- | --- | --- | --- |
| `native_frames` | `native-frames-v1` | `records/native_frames/native-frames-v1` | `data_message_ordinal` | 1024 |
| `native_signal_blocks` | `native-signal-blocks-v1` | `records/native_signal_blocks/native-signal-blocks-v1` | `signal_block_ordinal` | 1024 |
| `native_discontinuities` | `native-discontinuities-v1` | `records/native_discontinuities/native-discontinuities-v1` | `data_message_ordinal` | 64 |
| `native_signal_gaps` | `native-signal-gaps-v1` | `records/native_signal_gaps/native-signal-gaps-v1` | `signal_gap_ordinal` | 64 |
| `session_accounting` | `session-accounting-v1` | `records/session_accounting/session-accounting-v1` | `accounting_id` | 1 |

Each is an ordinary NRF v1 typed record set: the core rules for record schemas,
column and validity array paths, chunk alignment, extent ownership, sealing, and
transaction membership apply without exception. The chunk lengths above are the
values these fixtures use and the defaults an implementation SHOULD use; a
session MAY declare another positive chunk length, and nothing in this extension
depends on the value.

A session MUST declare exactly one record schema of each of the five kinds, or
none of them. Two schemas of one kind is not a richer session -- it is a session
whose ledger a reader cannot identify -- and four kinds without the fifth is not
a partial capability but a session whose replay capability cannot be evaluated.

The tables in section 3 are the normative field lists. Field **order** is
normative: it is the order the columns appear in the record schema's `fields`
array, so two conforming writers produce the same descriptor. `dtype`,
`nullable`, and `reference` are normative. `array_path`, `validity_path`,
`endianness`, and `codec_ids` are not restated here: they follow the core v1
rules from the record-set path, the dtype, and the session's codec registry.

## 3. Ledger fields

### 3.1 `native_frames`

One row per recorded frame. `recorded_signal_block_count` may be lower than
`signal_block_count` under a partial plan; that difference is what makes a
projection visible per frame rather than only per session.

| # | Field | dtype | Nullable | Reference |
| --- | --- | --- | --- | --- |
| 1 | `data_message_ordinal` | uint64 | no | — |
| 2 | `native_session_id` | uint64 | no | — |
| 3 | `frame_sequence` | uint64 | no | — |
| 4 | `frame_ordinal` | uint64 | no | — |
| 5 | `host_received_ns` | uint64 | no | — |
| 6 | `source_tick` | uint64 | yes | — |
| 7 | `valid_until_ns` | uint64 | yes | — |
| 8 | `native_schema_id` | uint32 | no | — |
| 9 | `source_clock_domain` | uint32 | no | — |
| 10 | `frame_flags` | uint32 | no | — |
| 11 | `signal_block_count` | uint32 | no | — |
| 12 | `recorded_signal_block_count` | uint32 | no | — |
| 13 | `total_payload_byte_count` | uint64 | no | — |
| 14 | `first_signal_block_ordinal` | uint64 | yes | — |

`frame_ordinal` is the contiguous frame-row identity: zero-based, gap-free, and
counting only frame rows. It is not `data_message_ordinal`, which counts every
data item -- frames and discontinuities alike -- and is the item identity of the
contract's section 1.1. Both are needed, and they are not interchangeable.
`total_payload_byte_count` is the byte length of the original frame payload, so
a reader can verify the recorded blocks cover the whole frame with no trailing
or overlapping bytes; the check is against the original recording, not against a
reconstruction. `source_tick` and `valid_until_ns` are nullable because the
native frame header carries them only when its flags say so; a null is "the
frame did not carry one", never a zero standing in for it. `host_received_ns`
and `valid_until_ns` are `uint64`, matching the native `HostTimeNs` type rather
than narrowing it. `first_signal_block_ordinal` follows the zero-child anchor
rule of section 3.6: it is null when `recorded_signal_block_count` is zero, and
required otherwise, with the child range `[first, first + count)`.

### 3.2 `native_signal_blocks`

One row per recorded signal block, in frame order. A block the recording plan
did not record has no row -- its absence is described by the frame row's
`recorded_signal_block_count` and by the coverage metadata of section 4.

| # | Field | dtype | Nullable | Reference |
| --- | --- | --- | --- | --- |
| 1 | `signal_block_ordinal` | uint64 | no | — |
| 2 | `data_message_ordinal` | uint64 | no | — |
| 3 | `frame_ordinal` | uint64 | no | — |
| 4 | `block_index_in_frame` | uint32 | no | — |
| 5 | `native_signal_id` | uint32 | no | — |
| 6 | `stream_id` | utf8 | no | stream |
| 7 | `sample_idx_start` | uint64 | no | — |
| 8 | `last_sample_idx` | uint64 | no | — |
| 9 | `n_samples` | uint32 | no | — |
| 10 | `device_tick_start` | uint64 | no | — |
| 11 | `observation_time_start_ns` | uint64 | no | — |
| 12 | `row_offset` | uint64 | no | — |
| 13 | `payload_byte_count` | uint64 | no | — |
| 14 | `payload_offset` | uint64 | no | — |
| 15 | `clock_sync_device_tick_reference` | uint64 | no | — |
| 16 | `clock_sync_host_time_reference_ns` | uint64 | no | — |
| 17 | `clock_sync_rate_numerator` | uint64 | no | — |
| 18 | `clock_sync_rate_denominator` | uint64 | no | — |
| 19 | `clock_sync_uncertainty_ns` | uint64 | no | — |
| 20 | `clock_sync_clock_domain` | uint32 | no | — |
| 21 | `clock_sync_generation` | uint32 | no | — |
| 22 | `clock_sync_flags` | uint32 | no | — |

`frame_ordinal` is the owning frame's `frame_ordinal` from `native_frames`, so a
reader groups blocks by their original frame without re-deriving it.
`payload_offset` is the block's byte offset in the **original** frame payload,
taken from the native `SignalBlockHeader`. It is not `row_offset`, which is
where the block's samples land in the per-stream NRF data array: the two are
different spaces, and the original offset is what lets a reader verify
overlapping payload ranges against the recording rather than against a
reconstruction. `observation_time_start_ns` and the clock-sync host-time and
uncertainty columns are `uint64`, matching the native types. Fields 15 to 22 are
the complete clock-sync snapshot the block carried, stored per block rather than
per session. A reader MUST NOT substitute the session's clock registration for a
missing one; under this extension it is never missing.

### 3.3 `native_discontinuities`

One row per discontinuity **message**, not per affected stream.

| # | Field | dtype | Nullable | Reference |
| --- | --- | --- | --- | --- |
| 1 | `data_message_ordinal` | uint64 | no | — |
| 2 | `native_session_id` | uint64 | no | — |
| 3 | `previous_frame_sequence` | uint64 | no | — |
| 4 | `actual_frame_sequence` | uint64 | no | — |
| 5 | `reason` | utf8 | no | — |
| 6 | `first_signal_gap_ordinal` | uint64 | yes | — |
| 7 | `signal_gap_count` | uint32 | no | — |
| 8 | `runtime_accepted_host_time_ns` | uint64 | no | — |

`first_signal_gap_ordinal` follows the zero-child anchor rule of section 3.6: it
is null when `signal_gap_count` is zero (a frame-level discontinuity), and
required otherwise, with the gap range `[first, first + count)`.
`signal_gap_count` is zero for a frame-level discontinuity, and that zero is
meaningful: it is not the absence of information. `runtime_accepted_host_time_ns`
is `uint64`, matching the native `HostTimeNs` type, and is recording-envelope
provenance -- when the recording path took the message over -- not the instant
the underlying gap occurred.

### 3.4 `native_signal_gaps`

One row per signal gap, addressed by its owning message's data-message ordinal
and its index within that message. Gaps for signals the plan did not record are
still written.

| # | Field | dtype | Nullable | Reference |
| --- | --- | --- | --- | --- |
| 1 | `signal_gap_ordinal` | uint64 | no | — |
| 2 | `data_message_ordinal` | uint64 | no | — |
| 3 | `gap_index_in_message` | uint32 | no | — |
| 4 | `native_signal_id` | uint32 | no | — |
| 5 | `expected_sample_index` | uint64 | no | — |
| 6 | `actual_sample_index` | uint64 | no | — |
| 7 | `missing_samples` | uint64 | yes | — |
| 8 | `expected_device_tick` | uint64 | no | — |
| 9 | `actual_device_tick` | uint64 | no | — |
| 10 | `reason` | utf8 | no | — |
| 11 | `gap_flags` | uint32 | no | — |

### 3.5 `session_accounting`

Exactly one row, committed in the transaction that seals the session.

| # | Field | dtype | Nullable | Reference |
| --- | --- | --- | --- | --- |
| 1 | `accounting_id` | utf8 | no | — |
| 2 | `accounting_origin` | utf8 | no | — |
| 3 | `termination_origin` | utf8 | no | — |
| 4 | `producer_acceptance_known` | bool | no | — |
| 5 | `runtime_accepted` | uint64 | no | — |
| 6 | `recorder_accepted` | uint64 | no | — |
| 7 | `spool_committed` | uint64 | yes | — |
| 8 | `nrf_committed` | uint64 | no | — |
| 9 | `rejected_before_runtime_acceptance` | uint64 | no | — |
| 10 | `failed_between_runtime_and_recorder` | uint64 | no | — |
| 11 | `lost_between_recorder_and_spool` | uint64 | yes | — |
| 12 | `lost_during_finalization` | uint64 | no | — |
| 13 | `control_offered` | uint64 | yes | — |
| 14 | `control_accepted` | uint64 | no | — |
| 15 | `control_spool_committed` | uint64 | yes | — |
| 16 | `control_nrf_committed` | uint64 | no | — |
| 17 | `control_rejected` | uint64 | no | — |
| 18 | `lost_between_control_acceptance_and_spool` | uint64 | yes | — |
| 19 | `control_lost_during_finalization` | uint64 | no | — |
| 20 | `data_first_lost_ordinal` | uint64 | yes | — |
| 21 | `data_first_rejected_message_kind` | utf8 | yes | — |
| 22 | `data_first_rejected_frame_sequence` | uint64 | yes | — |
| 23 | `control_first_lost_ordinal` | uint64 | yes | — |
| 24 | `control_first_rejected_kind` | utf8 | yes | — |
| 25 | `control_first_rejected_identity` | utf8 | yes | — |
| 26 | `data_rejected_after_close` | uint64 | no | — |
| 27 | `control_rejected_after_close` | uint64 | no | — |

Three properties of this row are normative and easy to get wrong:

- **The four spool-adjacent counters are nullable** (7, 11, 15, 18). A recording
  path with no spool has no stage-3 numbers and MUST write null rather than
  synthesize a value. Dropping a stage is legal; inventing one is not.
- **First-failed position is a tagged pair per plane, not one number.** Fields 20
  and 23 name items that were accepted and then lost, in ordinal space; fields 21
  to 22 and 24 to 25 name items refused *before* acceptance, which never received
  an ordinal. The two forms MUST NOT be compared or ordered against each other.
- **There is no `accounting_verified` field, and adding one is a violation.**
  Whether the accounting checks out is a judgement about this row, derived by the
  reader that is answering the call; a stored copy would let an artifact assert a
  verification result the reader in front of it disagrees with. A writer that
  wants to record what it believed MUST use a distinctly named field
  (`writer_accounting_attested`) under `extensions`, and it MUST NOT feed the
  derived value.

### 3.6 Zero-child ordinal anchor

Two parent ledgers carry a `first_*_ordinal` anchor into their child ledger:
`native_frames.first_signal_block_ordinal` into `native_signal_blocks`, and
`native_discontinuities.first_signal_gap_ordinal` into `native_signal_gaps`. A
parent row whose child count is zero has no first child -- a partial-plan frame
that records no block, or a frame-level discontinuity with no per-signal gaps --
and the anchor is **nullable** to say so exactly:

- `first_*_ordinal` is `null` when the corresponding count is zero.
- `first_*_ordinal` is required (non-null) when the count is greater than zero,
  and the child range is `[first, first + count)`.

A zero count is never represented by a standing-in ordinal. A reader MUST treat
a non-null anchor with a zero count, or a null anchor with a non-zero count, as a
validation failure: the two fields name one interval, and they must agree.

## 4. Manifest extension

The manifest's top-level `extensions` object carries the key
`neurale.native_replay` whenever the ledgers are declared. It states what a
replay of this session may claim, and it is the only place that answers "was
this plan complete?" without decoding a ledger.

```json
{
  "extension_version": 1,
  "plan_fingerprint": "<64 lower-case hex characters>",
  "native_schema": {
    "schema_id": 7,
    "signals": [
      {
        "id": 1, "clock_domain": 1, "channel_count": 4,
        "nominal_block_samples": 8, "max_block_samples": 16,
        "sample_rate": { "numerator": 1000, "denominator": 1 },
        "dtype": "INT16", "layout": "SAMPLE_MAJOR",
        "device_tick_tracking": "SAMPLE_COUNTER", "kind": "SAMPLED",
        "physical_unit": "VOLTS", "channel_set_id": 0, "calibration_id": 0,
        "reference_id": 0, "feature_set_id": 0,
        "observation_timing": "NOT_APPLICABLE",
        "fixed_block_bytes": 0, "max_block_bytes": 128
      },
      {
        "id": 2, "clock_domain": 1, "channel_count": 2,
        "nominal_block_samples": 2, "max_block_samples": 4,
        "sample_rate": { "numerator": 100, "denominator": 1 },
        "dtype": "FLOAT32", "layout": "SAMPLE_MAJOR",
        "device_tick_tracking": "UNAVAILABLE", "kind": "SAMPLED",
        "physical_unit": "DIMENSIONLESS", "channel_set_id": 0, "calibration_id": 0,
        "reference_id": 0, "feature_set_id": 0,
        "observation_timing": "NOT_APPLICABLE",
        "fixed_block_bytes": 0, "max_block_bytes": 32
      }
    ],
    "feature_sets": [],
    "units": []
  },
  "coverage": "full",
  "planned_signal_ids": [1, 2],
  "recorded_signal_ids": [1, 2],
  "streams": [
    {
      "native_signal_id": 1, "stream_id": "neural", "kind": "neural",
      "timing": "explicit", "block_index": true,
      "name": "neural",
      "unit": ["V", "V", "V", "V"],
      "channel_names": ["neural-0", "neural-1", "neural-2", "neural-3"],
      "feature_names": [],
      "segment_start_index": 0, "segment_start_time_ns": 0,
      "algorithm_name": "", "algorithm_version": "",
      "window_length_ns": 0, "shift_ns": 0,
      "source_stream_ids": []
    },
    {
      "native_signal_id": 2, "stream_id": "emg", "kind": "behavioral",
      "timing": "explicit", "block_index": true,
      "name": "emg",
      "unit": ["1", "1"],
      "channel_names": ["emg-0", "emg-1"],
      "feature_names": [],
      "segment_start_index": 0, "segment_start_time_ns": 0,
      "algorithm_name": "", "algorithm_version": "",
      "window_length_ns": 0, "shift_ns": 0,
      "source_stream_ids": []
    }
  ],
  "ledgers": {
    "native_frames": "native-frames-v1",
    "native_signal_blocks": "native-signal-blocks-v1",
    "native_discontinuities": "native-discontinuities-v1",
    "native_signal_gaps": "native-signal-gaps-v1",
    "session_accounting": "session-accounting-v1"
  },
  "replay_capabilities": {
    "exact_frames": { "available": true, "reason": null },
    "recorded_projection": { "available": true, "reason": null },
    "stream_frames": { "available": true, "reason": null }
  },
  "session": {
    "session_id": "018f4f30-6f9d-736-8c61-3f96cf5c5a40",
    "created_at": "2026-08-01T00:00:00Z",
    "writer_name": "pyneurale",
    "writer_version": "1"
  },
  "resource_bounds": {
    "frame_queue_capacity": 256,
    "control_queue_capacity": 1024,
    "control_chunk_length": 16,
    "checkpoint_interval": 32,
    "overflow_policy": "fault",
    "streams": [
      { "stream_id": "neural", "capacity": 4096, "chunk_length": 1024, "block_index_chunk_length": 256 },
      { "stream_id": "emg", "capacity": 1024, "chunk_length": 1024, "block_index_chunk_length": 256 }
    ]
  },
  "session_metadata": {},
  "metadata": {}
}
```

Rules:

- `native_schema` is the complete prepared native schema the session was
  recorded against: `schema_id`, every `signal` carrying the fields
  `StreamSchema::equivalent()` compares (`id`, `clock_domain`, `channel_count`,
  `nominal_block_samples`, `max_block_samples`, `sample_rate`, `dtype`,
  `layout`, `device_tick_tracking`, `kind`, `physical_unit`, `channel_set_id`,
  `calibration_id`, `reference_id`, `feature_set_id`, `observation_timing`,
  `fixed_block_bytes`, `max_block_bytes`), every `feature_set`, and every
  `unit`. Signals, feature sets, and units are kept in the order the prepared schema declared them: `StreamSchema::equivalent()` compares by position, so two schemas that carry the same entries in a different order are not equivalent, and the order is part of the fingerprinted document. It is carried in
  full so exact replay can rebuild the original `StreamSchema` and so the
  fingerprint of section 5 identifies schema *content*, not just a numeric id.
  `planned_signal_ids` MUST equal the `native_schema` signal ids as a set. Enum values
  are carried by their stable member name (for example `INT16`,
  `SAMPLE_MAJOR`, `SAMPLED`), not by an integer a particular build assigned.
- `planned_signal_ids` is every signal the native schema declared, ascending.
  `recorded_signal_ids` is the subset the plan records, ascending, and MUST be a
  subset of `planned_signal_ids`. `streams` names one entry per recorded signal,
  in `recorded_signal_ids` order, and carries the full per-stream mapping the
  plan document fingerprints: `native_signal_id`, `stream_id`, `kind`,
  `timing`, `block_index`, the resolved `name`, the per-column `unit` and
  `channel_names` (empty for a `feature` stream), the `feature_names`,
  `algorithm_name`, `algorithm_version`, `window_length_ns`, and `shift_ns`
  (emptied or zeroed for a non-`feature` stream), the `source_stream_ids`, and
  the `segment_start_index` and `segment_start_time_ns` that anchor a `regular`
  stream's time axis (zero for an `explicit` stream, which stores the
  timestamps that arrived instead). Every field is resolved to the value the
  manifest will carry, so two declarations that spell the same recording
  differently fingerprint identically, and two that record different things
  fingerprint differently. `source_stream_ids` follows the feature-set
  descriptor's numeric `source_stream_id` mapped through the recording
  declarations (`SignalId` -> `stream_id`), not its `source_stream` text, with no
  declaration-order fallback; explicit `source_stream_ids` must each resolve to a
  stream this plan records -- on a stream of any kind, not only a `feature` one,
  since a dangling link cannot be written into a legal manifest whichever stream
  it appears on. A reader rebuilds the plan from `native_schema` and
  `streams` together and arrives at the same fingerprint; a member that changes
  what is recorded but is absent here would make the stored fingerprint
  unverifiable.
- `coverage` is `full` when `recorded_signal_ids` equals `planned_signal_ids`,
  and `partial` otherwise. It is a property of the plan against the schema, not
  of any replay request.
- `replay_capabilities.exact_frames.available` is true **exactly when** coverage
  is `full`. A partial-plan session is not "exact replay with a caveat"; the
  applicable mode is `recorded_projection`, and the reason string MUST say so.
- `replay_capabilities.recorded_projection.available` is true whenever the five
  ledgers are declared.
- `replay_capabilities.stream_frames.available` is true exactly when at least one
  recorded stream declares `block_index: true`.
- `reason` is null when `available` is true, and a non-empty string when it is
  false. A capability that is unavailable without a stated reason is invalid.
- `session` is the recording identity (session id, creation time, writer name
  and version) and `resource_bounds` is the storage sizing (frame and control
  queue capacities, optional spool byte ceiling, control chunk length, checkpoint interval, overflow policy,
  and per-stream capacity and chunk bounds). `session_metadata` and `metadata`
  are the free-form session and manifest metadata the recorder declared, frozen
  as prepared metadata. All four are carried so the recorder and finalizer share
  one prepared plan and the finalizer never re-reads the original declaration,
  but they are **not** part of the fingerprinted document: they change how a
  session is stored or describe it, not what it records.
- `resource_bounds.spool_capacity_bytes` is an optional positive I-JSON integer.
  Omission preserves the legacy 64 MiB default. A continuous-file writer with no
  configured disk limit emits the representable ceiling `9007199254740991`.
  This value is not a RAM reservation. Explicit smaller limits remain disk
  ceilings; crossing one faults capture rather than silently dropping records.
- The extension is a second statement of facts the core manifest already
  carries, so each restated fact MUST equal the core manifest's: the session id
  and creation time, the set of recorded data streams, each stream's
  `native_signal_id`, kind, timing mode, name, per-column units, channel names
  (or feature names and feature descriptor for a `feature` stream), and the
  session and manifest metadata. A session that declared two different truths
  would leave a finalizer, reader, or replay builder unable to pick the
  authoritative one, so the validator rejects the contradiction rather than
  choosing a side.
- `native_schema` MUST be a schema a native `StreamSchema` could carry: legal
  enum members, positive channel counts and block-sample limits with
  `nominal <= max`, a positive sample-rate denominator, `max_block_bytes`
  consistent with the dtype, channel count, and block limits, unique
  signal/feature-set/unit ids, and a feature registry whose descriptors resolve,
  whose feature count matches its signal's channel count, and whose observation
  rate matches its shift. A fingerprint only proves the extension was not
  modified; it does not prove the extension describes a constructible schema.
- `plan_fingerprint` is section 5's value. A reader MUST recompute it from the
  fingerprinted members and compare; it MUST NOT be trusted as stored, and it is
  not a content checksum of the session -- it identifies the plan the session
  was recorded under.

## 5. Plan fingerprint

The fingerprint is the lower-case hexadecimal SHA-256 of the RFC 8785 (JCS)
canonical UTF-8 JSON of the **plan document**: the object of section 4 with
`plan_fingerprint`, `replay_capabilities`, `session`, `resource_bounds`,
`session_metadata`, and `metadata` omitted -- the first is the value being
computed, the second is derived from the rest, and the last four are storage,
identity, and description rather than what is recorded. Every other member is
included: `extension_version`, `native_schema` (with its signals, feature sets,
and units exactly as section 4 spells them, in prepared order), `coverage`,
`planned_signal_ids`, `recorded_signal_ids`, `streams` (with their resolved
metadata), and `ledgers`.

It is deterministic and portable: two compilers that produced the same plan
produce the same fingerprint, on any platform, in any process, whatever order
the caller declared the streams in. It changes when, and only when, something
that changes what gets recorded changes -- a stream added or removed, a stream
identifier renamed, a signal's mapping altered, the block index turned off, a
stream's unit, channel names, feature names, source links, or regular-timing
origin changed, or the native schema **content** changed (dtype, layout, channel
count, rate, clock domain, block limits, feature sets, units, or the order the
schema declared them in -- not just the numeric schema id).
It does not change with capacities, chunk lengths, queue sizes, session
identity, creation time, writer version, or free-form metadata: those alter how
a session is stored or describe it, not what it records, and a fingerprint that
moved with them could not be used to ask "was this the same recording plan?". A reader rebuilds the document from the
extension, recomputes the fingerprint, and MUST reject the extension if the
recomputed value does not equal the stored `plan_fingerprint`.

## 6. Journal extension

The journal `termination` record MAY carry `extensions` with the same namespace
key. It binds the sealed session to the plan and states the outcome fields the
lifecycle contract fixes and NRF's closed `termination_kind` enum cannot carry:

```json
{
  "extension_version": 1,
  "plan_fingerprint": "<64 lower-case hex characters>",
  "termination_origin": "recorder",
  "requested_terminal_intent": "normal",
  "capture_outcome": "normal",
  "effective_session_outcome": "normal",
  "finalization_status": "succeeded",
  "primary_fault_row_committed": true
}
```

The enumerated values are those of the lifecycle contract and are restated here
only so a schema can check them.

`termination_origin` is required, and it is required because the lifecycle
contract requires it on **both** kinds of termination -- `"recorder"` in the
ordinary case -- so that a reader can tell a termination the recorder wrote
from one a recovery reconstructed by reading the termination record itself,
signed by that record's checksum. Absence is not the ordinary case restated:
a reader that inferred `"recorder"` from silence would be deriving provenance
from a missing field rather than reading it, and could not distinguish an
ordinary session from one whose provenance was never written.

Its only permitted value here is `"recorder"`, and that is a constraint rather
than a redundancy. This extension is written exactly when a recorder's
session-end record froze the outcome dimensions, so the **namespace is itself a
provenance claim** and the value inside it must agree: `"recovery"` under this
key would describe a session whose recorder both did and did not reach the end.
Writing the value anyway is what lets a reader check that agreement instead of
trusting whichever of the two it happened to read.

Recovery provenance -- the keys under `neurale.recovery` -- is a different
namespace, owned by the recovery task, and is deliberately not defined here,
beyond the constraint this section is the other half of: a termination record
carries exactly one of the two, never both and never neither. This namespace
when a session-end record froze the outcome dimensions, `neurale.recovery` when
none did; each declares `termination_origin`, and each admits only the origin
its namespace means. That is what makes "read the origin off the termination
record" both total and checkable.

## 7. Artifacts

- `native-replay.schema.json`: machine-readable form of sections 2 to 6.
- `fixtures/`: one canonical record-schema descriptor per ledger, two manifest
  extension objects (full and partial coverage), and one journal termination
  extension object.
- `tools/generate_fixtures.py`: regenerates the schema and the fixtures from the
  tables in this document. This document is the source; the tool is mechanical
  and its output is byte-stable.

The fixtures use one writer's codec registry (`bytes-le`, `utf8-vlen`) because a
record-schema descriptor cannot be written without codec IDs. Those IDs are
session-scoped registry entries, not values this extension fixes.
