# Offline oscillation detection

## Detect episodes

```python
import numpy as np
from neurale.data import SignalArray
from neurale.features import detect_oscillations

fs = 100
time = np.arange(600) / fs
values = np.sin(2 * np.pi * 30 * time)
values[100:300] += 3 * np.sin(2 * np.pi * 10 * time[100:300])
signal = SignalArray.from_array(
    values, fs=fs, channel_names=("C3",), channel_types="eeg",
    units="uV", name="neural",
)
events = detect_oscillations(
    signal, bands={"alpha": (8, 12)}, reference_band=(28, 32),
    threshold=4, window_size=1.0, min_duration=1.0,
)
print([(event.onset, event.stop) for event in events])
```

`events` is an `EventSeries`; use `return_score=True` to inspect window scores
and the validity mask alongside detected intervals. Choose bands and thresholds
for the actual sampling rate and analysis question.

`neurale.features.detect_oscillations` detects oscillatory episodes in a
`SignalArray`, or in one named signal selected from a `Recording`. The current
implementation is offline Python/SciPy code; it has no native or online
processor. The detector is stateless: every call analyzes the supplied signal
independently and retains no history between calls.

The only supported detection method is `windowed_power_ratio`. The input is
split into complete, non-overlapping Hann windows. The window duration is
quantized to whole source samples with Python `round` (round-half-to-even),
and must yield at least 3 samples; a duration quantizing to fewer (notably the
2-sample symmetric Hann window, which is identically zero) is rejected rather
than silently producing a degenerate, all-zero score. The quantized duration
`window_length / sampling_rate` -- not the requested value -- is what every
window covers and what is stored as the result `window_size`. For every target
band and channel, the score is

```text
mean PSD over target [low, high) / mean PSD over reference [low, high)
```

Both bands use an exclusive upper frequency bound. Each window is
mean-detrended before spectral estimation, so a constant (DC) signal has zero
power in every band and scores zero -- a DC offset cannot become a detection.
A window is active when its score is greater than or equal to the positive
`threshold`. A window whose reference-band power is negligible but whose
target band carries genuine power scores high; that is the intended
ratio-detector behavior. No positive reference-power floor is imposed.
Per-window mean detrending removes DC offsets but does not clamp the
denominator. Exact zero denominator power has an explicit finite rule: `0 / 0`
scores `0`, while positive target power over zero reference power scores the
largest finite `float64` value. Positive target and reference powers use
ordinary division; a finite positive ratio exceeding float64 range saturates
to the same largest finite value. Non-finite band powers and NaN or infinite
signal values are rejected rather than omitted.

Active adjacent windows form candidates with half-open absolute-time bounds
`[start, end)`. `min_duration` is applied to each candidate before qualified
candidates separated by at most `merge_gap` are merged. Both `threshold` (the
per-window ratio comparison) and `min_duration` (candidate duration) operate
on the quantized window geometry, not the requested `window_size`. A
discontinuity makes every intersecting window invalid and is a hard merge
barrier. Only the selected continuous signal is analyzed; the detector does
not concatenate signals, trials, or recording segments, and ordinary recording
events are not implicitly interpreted as gaps. Only `n_samples //
window_length` complete windows contribute -- a trailing partial window is
ignored, and a signal too short to contain one complete window raises
`ValidationError` rather than returning a silent empty result.

The default result is an `EventSeries`. It retains the selected signal clock,
and collection metadata records the source sampling rate, full channel table,
bands, threshold, window size, duration and merge rules. It also records the
fixed analysis configuration: Hann window, per-window mean detrending,
non-overlapping shift equal to `window_size`, window-start timestamps, and PSD
density scaling. Every event records
its channel, target and reference bands, inclusive threshold, peak score, and
detection method. Event onsets and durations use the source signal's absolute
time coordinate.

Pass `return_score=True` to receive an `OscillationDetectionResult`. Its
`score` is an immutable `OscillationScore` with layout
`(window, band, channel)`, absolute window-start times, source channel table,
clock, source sampling rate, and a validity mask for discontinuity barriers.
Direct `OscillationScore` construction enforces the same sample-quantized
window geometry, usable FFT bins, and fixed analysis configuration as the
detector. `OscillationDetectionResult` validates that its event intervals and
metadata can be reconstructed from its score plus the recorded duration and
merge configuration; it is not a loose tuple for unrelated objects.

Episode filtering and zero-phase extraction are separate signal operations;
the detector does not apply them implicitly.
