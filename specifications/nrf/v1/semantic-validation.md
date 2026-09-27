# NRF v1 semantic validation

Status: normative
Semantic validation version: 1.0

## 1. Authority and validation order

An NRF v1 manifest is valid only when it passes both layers, in this order:

1. Draft 2020-12 validation against `manifest.schema.json`;
2. semantic validation against this document.

The JSON Schema is authoritative for JSON structure, required members, scalar
domains, conditional stream layouts, timing fields, feature linkage presence,
and nullable-field validity-path presence. This document is authoritative for
relationships that JSON Schema cannot express without external registry or
array state. The prose in `README.md` defines the logical format and MUST remain
consistent with both validation layers.

`tools/semantic_validation.py`, alongside this document, is the version 1.0
executable reference
validator. It is not a reader or writer and does not make Python class names
part of the format.

Journal records, checkpoints, and recovery reports likewise first validate
against their Draft 2020-12 schemas. Transaction replay then applies the
cross-record and manifest-dependent rules in section 3. JSON Schema is
authoritative for each record's local structure; replay validation is
authoritative for ownership, coverage, ordering, and committed-state
transitions that require multiple records or the frozen manifest.

## 2. Semantic rules

Implementations MUST report which layer rejected an input. They MAY use
different diagnostic wording, but MUST enforce these rules:

| Rule | Requirement |
|---|---|
| NRF-SEM-001 | IDs are unique within each registry. |
| NRF-SEM-002 | Every clock, schema, stream, source-stream, channel, electrode, unit, feature-set, codec, and record reference resolves in the corresponding registry. |
| NRF-SEM-003 | A Zarr array's `shape` and `chunk_shape` ranks equal the declared payload rank; dimensions are non-negative and chunk dimensions are positive. |
| NRF-SEM-004 | Stream `unit_ids` and non-empty `channel_ids` match the payload width. Feature names and units match the feature width. |
| NRF-SEM-005 | Stream and explicit-timestamp committed extents agree with the commit map and physical leading-axis bounds. Explicit timestamps cover every committed item. |
| NRF-SEM-006 | Endianness is `not_applicable` for `bool`, `uint8`, and `utf8`, and is explicit for numeric multi-byte dtypes. Referenced codecs are supported and registered. |
| NRF-SEM-007 | Feature-set names are non-empty and unique; its source stream and unit references resolve; window length and shift are positive. |
| NRF-SEM-008 | The clock registry is non-empty. Every record schema explicitly declares `clock_id`; record field names and column paths are unique; the primary key names a non-nullable scalar field whose dtype is `utf8`, `int64`, or `uint64`; record clock, unit, and codec references resolve; all mandatory record kinds exist. Timeless records are not represented by omitting `clock_id`. Descriptor validation establishes that the field can carry stable identity; writers and readers validate primary-key values and uniqueness when record data is written or read. |
| NRF-SEM-009 | Manifest, stream, record, and timestamp committed extents agree. Open targets end on chunk boundaries; sealed targets exist in the committed-extent map. |
| NRF-SEM-010 | Schema-to-stream, electrode-to-channel, synchronization-clock, and commit-cache relationships are internally consistent. |
| NRF-SEM-011 | Each stream segment policy references a dedicated `discontinuities`-kind record schema whose path is exactly `streams/<stream-id>/discontinuities`; no discontinuity schema may be shared by multiple streams. Every committed-extent key is owned by exactly one stream data array, stream timestamp array, or record schema path (so no committed extent is an unowned, non-decodable path). |
| NRF-SEM-012 | All registries (clocks, units, channels, electrodes, schemas, streams, feature sets, codecs, record schemas) are registered before the first committed transaction and are immutable for the life of the session; no registry entry may be added after the first commit (section 8). The journal carries no registry content, so the manifest registries must already contain every descriptor needed to decode every committed extent; the observable consequence (extent ownership) is enforced by NRF-SEM-011. |
| NRF-SEM-013 | Every integer in the manifest lies within the I-JSON exact range `[-9007199254740991, 9007199254740991]` and no number is non-finite. Named integer fields enforce this structurally via the shared `$defs/safeInteger`, `nonNegativeSafeInteger`, and `positiveSafeInteger` definitions; the semantic validator recursively rechecks the whole manifest, including free-form `metadata`, `extensions`, and Zarr `fill_value` that JSON Schema cannot bound. |
| NRF-SEM-014 | A `date-time` field is a strict RFC 3339 UTC value matching `^\d{4}-(0[1-9]\|1[0-2])-(0[1-9]\|[12]\d\|3[01])T([01]\d\|2[0-3]):[0-5]\d:[0-5]\d(\.\d+)?Z$`, with a mandatory trailing `Z`, no offset, and seconds restricted to `00` through `59`. NRF v1 does not support leap-second timestamps. The schema pattern enforces basic component ranges without relying on an optional format backend; the `date-time` format annotation and semantic validator enforce calendar validity such as month-specific day counts. |
| NRF-SEM-015 | Record schema paths are globally unique. Every stream data array, explicit timestamp array, record column array, and nullable validity array has one globally unique, non-overlapping physical path. A physical array cannot be an ancestor or descendant of another physical array. A record field path is exactly `<record-schema-path>/columns/<field-name>` and its validity path, when nullable, is exactly `<record-schema-path>/validity/<field-name>`. |
| NRF-SEM-016 | Every NRF path is already a portable canonical relative path: `/` is the only separator; segments are non-empty, lower-case ASCII, neither `.` nor `..`, contain only letters, digits, `.`, `_`, and `-`, do not end in `.`, and do not have a Windows-reserved basename (including before an extension). Paths have no leading/trailing slash, colon, drive, whitespace, or control character. Manifest-owned physical arrays are confined to `streams/` and `records/`; control files and the `journal/`, `.staging/`, `metadata/`, `recovery/`, `checksums/`, `indexes/`, and `feature_sets/` namespaces are reserved. Stream and record-set paths follow the fixed layouts in section 2 of `README.md`. Validators reject rather than normalize paths and apply a portable collision key before ownership checks. |
| NRF-SEM-017 | The manifest defines exactly one `session_termination` schema at `records/session_termination/termination-v1`, with `termination_id` as primary key and exactly the typed fields, dtypes, nullability, and fault/transaction references defined in section 8.1. Its committed extent is zero while open or one when terminated, and the target is sealed if and only if that extent is one. |
| NRF-SEM-018 | When a session declares any `neurale.native_replay` ledger record kind or the `neurale.native_replay` manifest extension, it declares each of the five native replay ledger kinds exactly once (each with its fixed schema id and `records/<kind>/<id>` path), the manifest extension object, and NRF minor version 1. Each native replay ledger's descriptor must match its schema id exactly: the fixed primary key, the field count, and each field's name, dtype, endianness, nullability, array path, validity path, and reference (codec ids, clock id, and chunk length are session-scoped and are left to the core record-schema rules). The extension's `native_schema` must be a schema a native `StreamSchema` could carry: legal enum members, positive channel counts and block-sample limits, a positive sample-rate denominator, `max_block_bytes` consistent with the dtype and block limits, unique signal/feature-set/unit ids, and a feature registry whose descriptors resolve, whose feature count matches its signal's channel count, and whose observation rate matches its shift. A feature stream's source linkage follows the descriptor's numeric `source_stream_id` mapped through the recording declarations, not its `source_stream` text, with no declaration-order fallback; explicit `source_stream_ids` must each resolve to a recorded stream. The extension's native schema (signals, feature sets, and units in the order the prepared schema declared them), planned and recorded signal ids, stream mappings (carrying the resolved per-stream metadata that reaches the manifest), coverage, replay capabilities, and ledger set are mutually consistent; every extension stream id resolves to a manifest stream; the extension's session identity, recorded stream ids, stream kind/timing/name/units/channel names/feature names/feature descriptors, `session_metadata` (equal to every core `session` member except `id` and `created_at`), and manifest metadata each equal the core manifest's; and the stored `plan_fingerprint` equals the SHA-256 of the RFC 8785 canonical JSON of the plan document (the extension object with `plan_fingerprint`, `replay_capabilities`, `session`, `resource_bounds`, `session_metadata`, and `metadata` omitted). A session that declares none of these is unaffected. |

Rules already enforced structurally by `manifest.schema.json` MUST NOT be
weakened by semantic validation. A semantic validator MAY defensively recheck
them, but passing semantic validation never compensates for failing JSON Schema.

## 3. Transaction replay rules

| Rule | Requirement |
|---|---|
| NRF-TX-001 | Each prepare object belongs to exactly one extent transition through `object_paths`; unassigned objects and objects named by multiple transitions are invalid. |
| NRF-TX-002 | Each object names the same `target_path` as its owning transition and an `array_path` present in that transition's `required_array_paths`. |
| NRF-TX-003 | A transition's target kind, record schema, and required physical arrays equal the frozen manifest contract. A record-set contract contains every field array and every nullable field's validity array. |
| NRF-TX-004 | Chunk `path`, `chunk_coordinate`, rank, logical role, record schema, and column association agree with the physical array and frozen manifest. |
| NRF-TX-005 | For each required physical array, the transaction's chunk coordinates exactly cover `[before, after)` and every trailing chunk coordinate implied by its shape and chunk grid. |
| NRF-TX-006 | Every advancing physical array has exactly one transaction-unique metadata snapshot. The only transition that may have no objects is a non-advancing seal-only transition. |
| NRF-TX-007 | Transaction IDs, journal sequences, prepare/commit references, previous-commit links, extents, seals, object paths, and commit visibility obey sections 8 and 9 of `README.md`. Within one prepare, final paths and staged paths are independently unique, and each staged path is exactly `.staging/<transaction-id>/<final-path>`. |
| NRF-TX-008 | A checkpoint preserves the full committed-object ownership records and equals replay state through its declared journal sequence. |
| NRF-TX-009 | A metadata snapshot path is transaction-unique under `metadata/transactions/<transaction-id>/` and neither equals, contains, nor is contained by a manifest-owned physical array path. |
| NRF-TX-010 | Every journal, checkpoint, head-cache, and recovery path uses the same portable canonical relative representation as NRF-SEM-016 before uniqueness, ownership, checksum attribution, or create-if-absent checks are applied. |
| NRF-TX-011 | A journal termination immediately follows the commit of its `last_transaction_id`, references the sole typed termination row, and matches that row's kind, time, reason, fault, and final transaction. The final transaction appends and seals the termination target from extent 0 to 1. Normal termination additionally requires every frozen appendable target to be sealed and has no fault. Faulted termination resolves `fault_id` to exactly one fault row committed before the final transaction. Aborted termination may omit `fault_id`, but a supplied ID has the same resolution rule. Uncommitted-tail and final-transaction fault rows do not qualify. |
| NRF-TX-012 | Every transaction and checkpoint object has a role-specific final path. Chunk paths equal `<array-path>/c/<coordinate>`, metadata snapshots are transaction-unique descendants of `metadata/transactions/<transaction-id>/`, and index objects are strict descendants of `indexes/<index-id>/`. NRF v1 defines no generic JSON transaction object. |

The checked-in language-neutral vectors and their replay oracle are the
executable conformance cases for these rules. They are specification tooling,
not an NRF reader or writer.

## 4. Versioning

Semantic validation version 1.x applies only to NRF format major version 1.
Additive clarifications that do not change accepted manifests increment the
minor semantic-validation version. Any change that makes a previously valid
manifest invalid, or vice versa, requires a new NRF format major version unless
the manifest explicitly opts into a versioned optional feature allowed by the
NRF compatibility rules.
