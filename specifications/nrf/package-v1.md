# NRF single-file envelope, version 1

New NRF artifacts are immutable ZIP64-capable files, not directories. The ZIP
comment is exactly the ASCII bytes `PyNeurale NRF package 1`. Internal NRF v1
manifest, journal, object names, object lengths and SHA-256 semantics remain
unchanged. SHA-256 covers the uncompressed object bytes. Every entry uses
DEFLATE (writer default level 1), independently of other entries. Directory
entries, duplicate names, encrypted members, symlinks, absolute paths, backslashes,
drive prefixes and noncanonical path components are rejected.

Readers address entries through the central directory. A signal range decompresses
only its covering chunks; no session extraction is performed. Journal parsing is
streamed, although replay state and the ZIP index still consume metadata memory
proportional to their entry counts. This is not a constant-total-memory guarantee.
An open reader refuses a package whose file identity, size or modification time
changes; reopen it after an explicit repair or replacement.

`NrfWriter` owns a private working directory until finalization. A terminated
session is streamed into a sibling temporary archive, reopened and checksum
validated, then published without replacing an existing destination. The public
destination does not expose in-progress transactions. Failed writes retain their
private work; successful publication removes it where permissions allow.

`SessionRecorder` continues recording to a native file spool, outside compression.
After capture stops, its converter writes immutable chunks and metadata snapshots
directly into a temporary compressed package, without intermediate payload files.
Mutable manifest, journal and array metadata still use a private workspace.
The envelope is finished and structurally checked, then the finalizer performs
one full committed-object checksum pass together with semantic and count checks.
That pass also runs when resuming a sealed staged package; prior verification is
never assumed to survive an interruption. Generic `NrfWriter` and explicit repair
keep their workspace-based publication path and payload verification.
The default retention policy is `delete_after_validated_finalization`; explicit
`RecorderConfig(spool_retention="retain")` retains recovery input for debugging.
The spool and plan sidecar are deleted only after cross-check and publication.
Cleanup failure does not undo publication: inspect `status.cleanup_paths` and
the finalization report's `spool_retained` field. Original recordings are never
automatically converted or deleted.

`SessionRecorder.finalization_progress` exposes phase and packing byte counters
for non-realtime polling. Packing counts are uncompressed input bytes, not an
estimate of compressed size or verification progress. A run may simultaneously
need disk space for the spool, private metadata workspace and archive during
spool conversion. Generic writes and explicit package repair can still require
a full uncompressed workspace. Compression now occurs during `converting`;
`packing` finishes the envelope, and `cross_checking` verifies its payloads.

This is a breaking storage change: public directory input is not supported.
No dtype conversion, resampling or lossy compression is introduced. Buffered
capture does not gain a power-loss durability guarantee through packaging.

Explicit `recover()` is separate from ordinary reading. Its dry run performs
no extraction or writes. Cache repair uses a private workspace and replaces
the original package only after validating the rebuilt package. Committed data
corruption is reported, not repaired or overwritten. The repair report lives
inside the resulting package; `RecoveryResult.report_path` names that package.
`recovery/latest` holds the canonical entry name of the most recently written
report, so custom recovery IDs and identical timestamps do not reorder history.
For an unrepairable package, the report is returned in memory and `report_path`
is `None`; the damaged original is not rewritten just to attach a report.
