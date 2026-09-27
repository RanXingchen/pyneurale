# Trial slicing and alignment contract

The data contract for trial splitting, epoch extraction, and neural/behavior
alignment is published through typed `neurale.data` APIs backed by the existing
data model.

Trial splitting, epoch extraction, and alignment are typed contracts over
`Recording`, `TrialTable`, `EventSeries`, `SignalArray`, and `Clock`; they
expose no bare-array signatures.

## Time and Intervals

Trial and epoch intervals are half-open: `[start, end)`. Samples exactly at
`start` are included. Samples exactly at `end` are excluded. Trial and epoch
intervals must have positive length; zero-length and negative intervals are
invalid.

Events keep the existing data-model distinction: `Event(duration=0)` is a valid
instantaneous event, while `duration > 0` is a half-open event interval
`[onset, onset + duration)`.

`Trial.start` and `Trial.stop` are absolute seconds in the recording reference
clock. A trial-relative epoch uses `[trial.start + tmin, trial.start + tmax)`;
if `tmax` is omitted, `trial.stop` is the stop boundary. `tmax` must be greater
than `tmin`, and epoch stops beyond `trial.stop` are invalid. Negative `tmin`
is allowed for pre-trial baseline windows; those baseline samples are still
subject to normal signal coverage and discontinuity checks.

## Trial Splitting

Trial splitting consumes a `Recording` with a `TrialTable`. The trial table
must be sorted by `Trial.start` and trials must not overlap. Adjacent trials are
allowed when the earlier trial's `stop` equals the later trial's `start`.
If callers provide `signal_names`, it must be an iterable of unique non-empty
strings. Passing a bare string is invalid, duplicate names are invalid, and
unknown names raise `ValidationError` before any trial is sliced.

The result type is a typed epoch container with these fields:

| Field | Contract |
| --- | --- |
| `trial` | The source `Trial` value. |
| `start`, `stop` | The actual epoch interval in recording-reference seconds. Strict extraction is the only mode, so this equals the source trial interval. |
| `signals` | Mapping from signal name to the signal's own `SignalArray` epoch. |
| `events` | `EventSeries` filtered to the actual epoch interval, or `None` if the recording has no events. |
| `recording_metadata` | Copy of recording-level metadata. |
| `trial_attrs` | Copy of trial-level attrs. |

Each signal is sliced independently on its own clock and sampling grid. Trial
start/stop values are already in recording-reference seconds, so converting a
trial boundary to signal-local time uses `source_clock=None`; an explicit
recording reference `Clock` is used only to validate synchronization identity,
not to apply another affine transform to the trial values. Trial splitting must
not merge recording, trial, and signal metadata into a single attrs dictionary
implicitly.

Trial splitting may use `reference_clock=None` only when every selected signal,
the recording `EventSeries`, and the discontinuity series also have
`clock=None` and are therefore already expressed in the recording-reference
coordinate. If any selected input carries a `Clock`, callers must provide an
explicit `reference_clock`; otherwise trial splitting raises `ValidationError`
before any trial is sliced. Every clocked input must belong to the same
synchronization domain as `reference_clock` (a cross-domain signal, event, or
discontinuity raises `ValidationError`), so a `Clock` from another session can
never be silently treated as the current recording's reference.

## Out-of-Range Behavior

Strict extraction is the only mode. An epoch whose requested interval extends
outside the signal coverage raises `ValidationError`; there is no partial
clipping option. For trial splitting, every selected signal and the event
table are sliced with the trial's own `[trial.start, trial.stop)` interval.

Signal coverage is `[signal.time[0], signal.time[-1] + 1 / sampling_rate)`.
This is the only range used for epoch acceptance; it does not infer coverage
from nominal trial duration or from unrelated signals.

## Clock Conversion

All clock conversion uses the affine reference mapping:

```text
reference_time = clock.offset + local_time * (1 + clock.drift)
```

Clock operations are explicit:

- `to_reference_time(value, source_clock)` maps source-local time to recording
  reference time. If `source_clock` is `None`, the input is already reference
  time.
- `from_reference_time(value, target_clock)` maps recording reference time to
  target-local time. If `target_clock` is `None`, the output remains reference
  time.
- `convert_time(value, source_clock, target_clock)` is exactly
  `from_reference_time(to_reference_time(value, source_clock), target_clock)`.

Clock-to-clock conversion is legal only when both clocks belong to the same
synchronization domain. Missing or different domains raise `ValidationError`.
Clock dataclass value equality is not a synchronization proof; two equal-valued
`Clock` instances still require explicit domain compatibility when both are
present.
Synchronization identity is represented with the structured
`Clock.synchronization_domain` field. `Clock.attrs` remains
application-specific metadata and is not used for clock compatibility. The
`Clock` model requires `1 + drift > 0`; invalid drift is rejected at
construction.

Trial times are recording-reference seconds. When trial boundaries are
converted to signal-local or event-local time, the source clock is `None`
because the values are already in reference coordinates. The APIs may still
accept or carry an explicit recording reference `Clock`, but that clock defines
reference identity and synchronization-domain compatibility; it must not be
used as the numeric source clock for `Trial.start` or `Trial.stop`. `None` never
means "unknown clock"; at source positions it means "already reference time",
and at target positions it means "leave the result in reference time".

## Discontinuities

Epoch extraction rejects any discontinuity that overlaps the requested
half-open interval. Instantaneous discontinuities are barriers at their onset:
an epoch ending at the barrier is valid, while an epoch starting before or at
the barrier and extending after it is invalid. Future APIs may expose
segment-aware splitting, but default extraction must not cross a gap.

Sustained discontinuities use normal half-open overlap:
`gap.start < epoch.stop and gap.stop > epoch.start`. Therefore a sustained gap
ending exactly at `epoch.start`, or starting exactly at `epoch.stop`, does not
overlap the epoch. Instantaneous gaps reject epochs where
`epoch.start <= gap.onset < epoch.stop`.

## Missing Events

Event filtering uses event onset and the same `[start, end)` rule. Required
event labels are strict by default: if a required label is absent from the
filtered event set, extraction raises `ValidationError`. Optional behavior must
be explicit, for example `allow_missing=True`, and must preserve the selected
events without fabricating placeholders. `required_labels` must be an iterable
of non-empty strings; passing a bare string is invalid.

A `Recording` with no `events` is valid when no event labels are required. If
trial splitting receives non-empty `required_event_labels` and
`Recording.events is None`, it raises `ValidationError` before slicing trials
because every requested label is absent.

## Resampled Signals

Epoch extraction is time-based, not original-sample-count based. A resampled
signal is sliced using its own `SignalArray.time` and `sampling_rate`, and the
output sample count follows that signal's current grid. The extraction layer
must not silently resample, interpolate, or convert dtype/layout.

Epoch extraction requires a regular `SignalArray.time` axis consistent with
`1 / sampling_rate`; irregular time axes are unsupported.

## Neural/Behavior Alignment

The default neural/behavior alignment mode is nearest-neighbor point matching.
It does not resample or interpolate either input.

Inputs are two named `SignalArray` objects, usually one neural signal and one
behavior signal. Each sample time is mapped into recording-reference seconds
using its signal clock. Both clocks must have the same synchronization domain
when both are present. If exactly one clock is missing, the missing side is
already in recording-reference coordinates. If both clocks are missing, both
signals are already in the same recording-reference coordinates.

Alignment establishes one reference synchronization domain before any
discontinuity or nearest-neighbor checks. If either signal has a `Clock`, that
clock defines the alignment domain after neural/behavior compatibility has
been checked. If both signal clocks are missing, the alignment domain has no structured
`Clock`.

The two signals must have non-empty common coverage after each coverage
interval is mapped into reference time. If the intersection is empty or only
touches at a boundary, alignment raises `ValidationError` with
`"neural and behavior signals have no common time coverage."`. It must not
return an all-unmatched result, and the result must not depend on whether an
unrelated discontinuity series was supplied.

The caller must choose the target grid: `"neural"` or `"behavior"`. The target
signal's time vector defines the output row count and `reference_time`. The
other signal is matched to this grid by nearest reference-time sample.

Nearest matching is a target-ordered monotonic greedy contract, not a global
minimum-cost assignment. Target rows are processed in increasing target-grid
order after timestamps are mapped into recording-reference seconds. Crossing
matches are not allowed.

Nearest matching rules:

- Both neural and behavior signals must contain at least one sample. Empty
  inputs raise `ValidationError`.
- Nearest alignment requires strictly increasing observation timestamps. A
  `SignalArray` only guarantees monotonic non-decreasing time, so repeated
  timestamps are rejected with `ValidationError` before matching, rather than
  letting the matcher skip the duplicate and break the earlier-source tiebreak.
- A match is accepted when absolute error is `<=` tolerance within the defined floating-point comparison tolerance.
- `tolerance` must be non-negative and finite.
- Matching is one-to-one; a source sample cannot be reused.
- Matched neural indices and matched behavior indices must both be strictly
  increasing. A later target row must not match an earlier source row than any
  previous matched target row.
- Distance ties choose the earlier source sample.
- If the nearest eligible source sample has already been used or would cross a
  previous match, the next nearest unused non-crossing source sample may match
  if it is still within tolerance.
- Unmatched target rows remain present with source index `-1` and
  `matched=False`.

The result type is a typed alignment-index container with:

| Field | Contract |
| --- | --- |
| `mode` | Currently `"nearest"`. |
| `target` | `"neural"` or `"behavior"`. |
| `tolerance` | Inclusive tolerance in reference seconds. |
| `reference_time` | Target grid in recording-reference seconds. |
| `neural_indices` | Row indices into the neural input. If neural is the target, these are always valid target-grid indices. Otherwise unmatched rows contain `-1`. |
| `behavior_indices` | Row indices into the behavior input. If behavior is the target, these are always valid target-grid indices. Otherwise unmatched rows contain `-1`. |
| `matched` | Boolean mask of rows where both neural and behavior indices are valid. |

The alignment uses one-to-one monotonic matching so aligned rows have
unambiguous temporal order and no duplicated source observations.

Because source samples cannot be reused, this nearest mode is intended
primarily for signals that have already been binned, summarized, or resampled
onto comparable rates before alignment. Directly aligning a high-rate
continuous target grid to a much lower-rate source grid is expected to be
sparse. For example, with a 1000 Hz neural target and a 100 Hz behavior source,
at most about 10% of neural target rows can match before tolerance is even
considered; the remaining target rows stay present with source index `-1` and
`matched=False`. Separate, explicit APIs for interpolation, resampling, binning,
or many-to-one alignment are not provided and would require their own contracts
if needed.

For discontinuity gating, the reference-time alignment range is only the common
signal coverage:

```text
intersection(
  neural [time[0], time[-1] + 1 / sampling_rate) mapped to reference time,
  behavior [time[0], time[-1] + 1 / sampling_rate) mapped to reference time
)
```

Any discontinuity overlapping that common coverage in any input stream rejects
the whole alignment. Target-grid rows outside common coverage may still be
present in the result as unmatched rows (`matched=False`), and discontinuities
that only overlap those target-only or source-only tails do not reject the
alignment.

Discontinuity clock compatibility is validated before overlap. If a
discontinuity series has a `Clock`, it must belong to the same synchronization
domain as the alignment domain. If a discontinuity series has no `Clock`, its
timestamps are already in the alignment reference coordinate. If both signal
clocks are missing, a clocked discontinuity series is rejected because the
alignment domain has no explicit reference `Clock` against which to verify it.

After compatibility is established, discontinuity timestamps are mapped with
`to_reference_time(event.onset, discontinuity.clock)`. Segment-aware alignment
is not provided by the current API.

## Metadata Preservation

Epoch outputs preserve the selected signal's channel table, units, clock,
name, sampling rate, and attrs. Recording-, trial-, or event-level metadata are
not merged into `SignalArray.attrs` by default because doing so would create an
implicit policy about provenance shape. Higher-level containers may attach
trial metadata explicitly in a separate result object.

Filtered `EventSeries` outputs preserve the input event clock and collection
attrs. Individual events are immutable and retained by value.

## Shape and Time-Axis Reconstruction

Signal payloads remain sample-major with shape `(n_samples, n_channels)`.
Epoch time vectors are copied from the selected signal samples in that signal's
clock. They are not rebased to trial-relative zero by default. The returned
`SignalArray.t0` is the first selected sample time, which may be greater than
the requested start if the start boundary falls between samples.

Downstream feature or decoder code that needs trial-relative time must derive
it explicitly from `epoch.time - trial.start` after any clock conversion.
