# Offline typed neural tuning

## Bin neural observations by behavior

```python
import numpy as np
from neurale.data import FeatureMatrix, SignalArray
from neurale.features import binned_tuning

time = np.arange(6) / 10
neural = FeatureMatrix(
    data=np.array([[1.], [2.], [3.], [4.], [5.], [6.]]),
    fs=10, time=time, feature_names=("unit-1",),
    source_signal="neural", window_size=None, unit="Hz",
)
behavior = SignalArray.from_array(
    np.array([0.2, 0.2, 0.2, 0.8, 0.8, 0.8]),
    fs=10, time=time, channel_names=("position",),
    channel_types="behavior", units="m", name="position",
)
result = binned_tuning(
    neural, behavior, behavior_dim="position",
    bin_edges=np.array([0.0, 0.5, 1.0]), alignment_tol=0.0,
)
print(result.values, result.occupancy)
```

`values` has `(bin, feature_or_unit)` order; `occupancy` counts matched
observations per bin. Choose `alignment_tol` for the timestamp precision of
the real neural and behavioral sources.

`neurale.features.tuning` provides deterministic offline tuning analyses for a
typed `FeatureMatrix` or `SpikeTrain` and a selected behavioral dimension from
a `SignalArray`. Decoder fitting, prediction, cross-validation, experiment
state, and online processors are outside this module.

## Alignment and axes

Neural feature observations use `(observation, feature)` order. Spike trains
are converted to `(observation, unit)` firing rates using caller-provided
`spike_observation_edges`. Every spike interval is `[start, stop)` and its
observation time is the interval center. These intervals must have equal
width; the edges, interval convention, and center time reference are retained
in result metadata. Behavior uses
`(observation, behavioral_dimension)` order.

`binned_tuning()` delegates nearest one-to-one clock alignment to
`align_neural_behavior_nearest()`. When a `TrialTable` is provided, it first
delegates trial extraction to `split_recording_trials()` and aligns each trial
independently, so a match cannot cross a trial boundary. The result preserves
the neural and behavioral clocks and complete selected `Trial` values.

Trial slicing is currently supported only for point-observation
`FeatureMatrix` input where `window_size` is `None`. A windowed `FeatureMatrix`
is explicitly rejected because `FeatureMatrix` does not yet carry a typed
`start`/`center`/`end` timestamp reference and shared slicing cannot prove that
the complete window lies inside a trial. A `timestamp_reference` value in
application attrs is not treated as this missing typed contract. `SpikeTrain`
firing-rate observations also represent intervals rather than point samples,
so combining `SpikeTrain` with `TrialTable` is rejected until support-aware
observation slicing is available. These restrictions prevent out-of-trial
samples or spikes from entering a trial through an overlapping observation.

`TuningResult.values` has `(bin, feature_or_unit)` order,
`bin_coordinates` has `(bin, behavioral_dimension)` order, and `occupancy` has
`(bin,)` order. The first implementation supports exactly one selected
behavioral dimension per result. Feature names and source identity are retained;
`SpikeTrain.channels`, when present, is preserved as `feature_channels`.
Without normalization, `feature_units` contains the source units. With
`normalization="zscore"`, `feature_units` is `"z-score"` and the original units
are retained in `attrs["source_feature_units"]`.
For `FeatureMatrix`, result metadata also records `source_sampling_rate`,
`source_window_size`, and `source_shift` so the source observation geometry is
not lost. Public results require and validate this geometry. Spike-derived
results similarly require immutable observation edges, `[start, end)` interval
semantics, center timestamps, `Hz` source units, and an empty trial selection.

`FeatureMatrix` and `SpikeTrain` do not currently carry a `Clock`; callers use
the explicit `neural_clock` argument when their timestamps are not already in
recording-reference coordinates.

## Binning contract

- `bin_edges` must be finite and strictly increasing.
- Every bin, including the final bin, is half-open `[low, high)`.
- Out-of-range observations do not contribute.
- Occupancy is the number of complete aligned observations in each bin.
- Bins below `minimum_sample_count` retain their occupancy and return NaN.
- `empty_bin_behavior="nan"` and `smoothing="none"` are the only supported
  first-version policies.
- `normalization="none"` leaves neural values unchanged.
- `normalization="zscore"` standardizes each feature/unit over complete,
  aligned, in-range observations before bin averaging. Constant inputs become
  zero after normalization and are recorded in result metadata. Z-scored
  tuning values are dimensionless standard scores rather than values in the
  source physical units. The normalization center, scale, and constant-feature
  indices are required result provenance. When no complete observation falls
  inside the requested bin range, center and scale are `None` and the
  constant-feature tuple is empty.
- NaNs are rejected by default. `nan_policy="omit"` uses complete-case rows, so
  occupancy is shared across every neural feature/unit. Infinite values are
  always rejected.

## Directional tuning

`directional_tuning()` requires radians, wraps angles into `[0, 2*pi)`, and
uses uniform circular bins. It fits

```text
response = cos_coefficient * cos(angle)
         + sin_coefficient * sin(angle)
         + intercept
```

to finite bin means. `cosine_coefficients` stores those three coefficients in
that order. Preferred direction is `atan2(sin_coefficient, cos_coefficient)` in
`[0, 2*pi)`. `modulation_depth` is the cosine amplitude, not the peak-to-trough
range. Each finite bin mean has equal weight in the fit; occupancy is reported
but does not weight the cosine regression. Fewer than three usable bins, a
rank-deficient fit, or a constant profile produces NaN preferred direction
rather than fabricating a value. Modulation depth and all three cosine
coefficients use the same value units as `tuning.feature_units`; consequently,
they are expressed as z-scores when normalization is enabled.
`DirectionalTuningResult` requires uniform bins spanning `[0, 2*pi)` and
records `tuning_kind="directional"`, the circular period and interval, and
uniform weighting across valid bins. These fields are validated when a public
result is reconstructed.

Typed direction calculation belongs to `neurale.features.kinematics`.
