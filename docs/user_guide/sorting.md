# Offline spike sorting

The internal native pipeline also owns a prepare-time fixed-capacity online
spike block. One output block may contain multiple spikes; overflow is explicit
(``fault`` or ``drop_newest``), and cross-frame waveform tails are cleared by a
discontinuity or reset. ``spike_block_to_waveform_batch`` is the allocating
control-plane conversion into the public immutable batch and must remain
outside realtime processor callbacks.

`neurale.sorting.detect_threshold_spikes` produces one
`SpikeWaveformBatch` per continuous segment. `project_waveform_features`
consumes one such batch and returns a `FeatureMatrix` whose rows remain in the
same spike order.

```python
from neurale.models import PCA
from neurale.sorting import WaveformProjector, project_waveform_features

projector = WaveformProjector(PCA(3))

features = project_waveform_features(
    waveforms,
    projector,
    measurements=("amplitude", "width", "energy"),
)
```

The waveform is flattened from `(sample, channel)` in C order only at the
model boundary. PCA and LPP/OPP are the existing estimators from
`neurale.models`; `WaveformProjector` owns one such estimator and its waveform
schema. Sorting neither reimplements model mathematics nor exposes a
backend/provider option. Passing `fit=False` transforms with a wrapper
previously fitted by `project_waveform_features(..., fit=True)`, so the stored
`device_` is not changed by the ambient runtime context. The wrapper records
sample geometry, alignment, sampling rate, ordered channel metadata, and C
flatten order, and rejects an incompatible later batch before entering the
model. A bare estimator is not accepted. If `projector.estimator.fit(...)` is
called independently, the wrapper detects the changed fitted state, invalidates
its old waveform schema, and requires another waveform fit. LPP/OPP coordinates
use arbitrary units, while PCA coordinates retain the shared source amplitude
unit.

Projection columns come first. Measurement columns follow in the requested
order:

- `amplitude` is the batch's signed, centered detection amplitude;
- `width` is the time from the aligned dominant extremum on the peak channel
  to the earliest following opposite extremum;
- `energy` is `sum(waveform**2) / sampling_rate` over all waveform samples and
  channels.

Width is measured in seconds and energy in `<amplitude-unit>^2*s`. Projection,
amplitude, and energy reject channel tables with mixed amplitude units because
one `FeatureMatrix` column cannot represent observation-dependent units.
Width-only extraction remains valid for such batches.

Width is computed per detected spike using the batch's known alignment and
polarity and is returned in seconds, not milliseconds.

The output `time` contains the aligned spike times. Since these observations
may be irregular, both `sampling_rate` and `shift` are `None`; the source sample
rate is retained as `attrs["source_sampling_rate"]`, and `window_size` records
the extracted waveform duration. Source linkage is explicit in `attrs`:
batch-row indices, absolute source sample indices, peak channels, electrode
groups, segment, clock, channel metadata, alignment sample counts, waveform
axis order, batch polarity, and per-spike polarities when present are retained.

An empty batch can produce measurement-only empty features. Fitting PCA or LPP
to an empty batch follows the estimator's existing validation and fails rather
than inventing projection state.

## Minimal offline workflow

`neurale.sorting.run_offline_sorting` exposes the small, explicit path from one
continuous `SignalArray` through detection, waveform feature projection,
Valley Seeking, tiny-cluster curation, and final `SpikeTrain` conversion. It
returns `OfflineSortingResult`, retaining the `SpikeWaveformBatch`,
`FeatureMatrix`, `ValleySeekingResult`, `CurationResult`, and `SpikeTrain` for
diagnostics. It is a function composition, not a mutable sorter object.

```python
from neurale.sorting import DetectionConfig, run_offline_sorting

result = run_offline_sorting(
    signal,
    DetectionConfig(
        alignment_search_radius=4,
        pre_samples=16,
        post_samples=31,
    ),
    radius=2.0,
    min_cluster_size=20,
    noise_label=-1,
)
```

The workflow requires an explicit Valley Seeking radius, curation threshold,
and noise label. By default each detected event starts with its observation
index as a unique nominal seed; callers may instead supply exact event-aligned
`int64` `initial_labels`. The native public `valley_seeking` wrapper accepts
only typed `FeatureMatrix` input with exact C-contiguous `float64` data and does
not estimate a radius, convert dtype/layout, or select another provider.

The convenience workflow is deliberately single-segment. For discontinuous or
multi-segment data, call `detect_threshold_spikes` with discontinuities and
compose projection, clustering, curation, and conversion independently for
each returned batch. `fit_projection` explicitly selects whether an optional
`WaveformProjector` is fitted or uses its existing fitted state.

`waveform_batch_to_spike_train` orders units by ascending numeric label,
excludes only the explicitly supplied noise label, and never renumbers labels.
Exact numeric labels, event/sample indices, per-event peak channels, source
clock, channel table, stream, and segment remain in `SpikeTrain.attrs`.
Waveforms are omitted unless `retain_waveforms=True`; only then does the final
`SpikeTrain.sampling_rate` describe retained waveform payloads.

## Foundational sorting metrics

`neurale.sorting.metrics.matching_events` compares ordered absolute sample
indices, not floating-point seconds. It performs a one-to-one linear match with
an inclusive integer tolerance. The earliest unmatched source is paired with
the earliest eligible target, which fixes duplicate and tie behavior.
`sample_errors` is signed as `target_sample - source_sample`; all returned
matched and unmatched indices address positions in the corresponding input.

Optional source and target segment arrays must both be one-dimensional
`int64`. Each segment ID occupies one contiguous run, and samples are
nondecreasing inside a run. Segment run order may differ between inputs, but
events with unequal segment IDs are never matched. Inputs that violate their
declared order are rejected rather than sorted.

`neurale.sorting.metrics.isi_metrics` computes each cluster independently in
each continuous segment and returns immutable `ClusterIsiMetrics` objects in
ascending numeric label order. Labels are arbitrary signed, non-contiguous
`int64`; a label is noise only when `noise_label` is explicitly supplied.
Within-segment ISIs are concatenated in input order, so no interval crosses a
segment boundary and repeated sample indices produce zero ISI.

The refractory sample threshold is
`ceil(refractory_interval * sampling_rate)`, matching threshold detection. An ISI
is a violation only when it is strictly less than that threshold; equality is
allowed. A cluster with no ISIs reports zero violations and fraction `0.0`,
with `None` for minimum, maximum, mean, and median summaries.

## Explicit sorting curation

`neurale.sorting.curation` contains label-preserving, offline curation
operations. Every input label vector is one-dimensional `int64`; labels may be
signed and non-contiguous. Results never mutate or reorder the input. Instead,
`updated_labels[i]` remains linked to input event `i`, and each operation emits
an immutable `CurationOperationRecord` containing source and target labels,
changed-event counts, original event indices, and frozen parameters. No
operation silently renumbers labels or proposes merge/split labels.

`remove_tiny_clusters` sends every non-noise cluster with
`count < min_cluster_size` to the explicitly supplied noise label. Equality is
retained. `merge_clusters` requires non-empty, unique, existing source labels;
the caller supplies the target, which may already exist but cannot also be a
source.

`split_cluster` accepts a mapping from explicit child labels to original event
indices. At least two non-empty children must cover the source cluster exactly
once. Foreign, overlapping, incomplete, source-equal, and already-owned child
labels are rejected, so a split cannot conceal a merge.

`reject_waveform_outliers` processes each non-noise cluster independently. Let
`M` be its elementwise median waveform and
`d_i = ||flatten(waveform_i - M)||_2`. It computes
`d_med = median(d_i)` and the unscaled centered
`MAD = median(abs(d_i - d_med))`, then rejects only
`d_i > d_med + threshold_multiplier * MAD`. With zero MAD the threshold is
exactly `d_med`; equal scores are retained. Clusters below
`minimum_cluster_size` receive scores but `NaN` thresholds and are not
rejected. Noise events receive `NaN` scores and thresholds. Returned sample
indices preserve linkage to the source `SpikeWaveformBatch`.

The curation operations are immutable and emit audit records; labels are never
mutated in place.
