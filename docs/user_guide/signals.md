# Neural signal workflows

## Construct a signal

`SignalArray.from_array` creates channel metadata and accepts one- or
two-dimensional arrays:

```python
import numpy as np
from neurale.data import SignalArray

signal = SignalArray.from_array(
    np.zeros((1000, 2)),
    sampling_rate=1000.0,
    channel_names=["C3", "C4"],
    channel_types="eeg",
    units="uV",
    name="eeg",
)
```

Explicit `SignalArray.unit` metadata must match the units in its channel table.
Conflicts raise `ValidationError`; constructors never silently rewrite channel
metadata.

## Data ownership and invariants

`SignalArray` borrows the supplied data buffer by default so in-place signal
processing and zero-copy acquisition workflows remain possible. Pass
`copy_data=True` to `SignalArray.from_array` when the signal should own an
independent data buffer.

Object fields, timestamps, per-channel units, and attribute metadata are
immutable after construction. Explicit timestamps are copied and read-only.
The borrowed data values remain writable, but callers must not change its
shape. Use `with_data()` or `replace()` for controlled updates and call
`validate()` at persistence or API boundaries when external code retains the
data buffer.

## Select and process

Time selection uses half-open intervals. `select_time` accepts absolute clock
times, while `crop` accepts seconds relative to the first sample:

```python
selected = signal.select_time(0.5, 1.0)
cropped = signal.crop(0.5, 1.0)

from neurale.signal import butter, resample, sos_filter

filtered = sos_filter(
    signal,
    butter(4, [8, 30], band="bandpass", sampling_rate=1000),
)
resampled = resample(filtered, target_rate=250)
```

`SignalArray` only owns data and metadata. Signal processing operations live in
`neurale.signal` and accept data objects as inputs.

`duration` is the elapsed time from the first sample timestamp to the last
(`signal.time[-1] - signal.time[0]`). When explicit timestamps are omitted,
`SignalArray` constructs them from `t0` and `sampling_rate`.

## Deterministic signal simulation

`neurale.signal.simulation.SignalGenerator` defines sample values by absolute
sample position and channel. It does not create frames, pace delivery, model a
device clock, or inject transport faults. The native path currently writes only
sample-major `float64` data into caller-owned prepared buffers:

```python
import numpy as np
from neurale.signal.simulation import SignalGenerator

generator = SignalGenerator.tones(
    channel_count=2,
    sample_rate=4000.0,
    frequencies=[100.0, 150.0],
    amplitudes=[1.0, 0.25],
)
output = np.empty((256, 2), dtype=np.float64)
generator.generate_into(output, start=4096)
```

One-dimensional `frequencies`, `amplitudes`, and `phases` describe tones
shared across channels. Two-dimensional parameters use `(tone, channel)`
order. `sine` accepts scalar or per-channel values. The named constructors also
provide zeros, per-channel constants, counter-based uniform noise, and finite or
repeating supplied samples. A seed and absolute `(sample, channel)` counter
fully determine every noise value; no mutable RNG or reset operation exists.

For every generator, requesting `[N, A + B)` produces exactly the same values
as concatenating requests `[N, N + A)` and `[N + A, N + A + B)`. The allocating
`generate` convenience is intended for offline use; `generate_into` is the
prepared caller-buffer boundary used by simulated acquisition.

## Intent-driven neural signal surrogate

`NeuralSignalGenerator` produces continuous, sample-major voltage with hidden
spikes. Its native model combines an intent-tuned spiking population, biphasic
extracellular spike waveforms, a damped stochastic narrow-band LFP component,
colored spatially correlated background noise, slow rate nonstationarity, and
controlled tuning drift. The LFP component reproduces selected temporal and
spectral statistics; it is not a biophysical tissue or volume-conduction model.
Spike spatial spread treats channel indices as a linear order and omits
connections below `1e-4` of a unit's primary-channel amplitude.

The algorithm depends only on `neurale.signal` and consumes a generic
two-dimensional intent. Experiment integrations remain responsible for mapping
task state, such as target and cursor positions, into that intent:

```python
from neurale.signal.simulation import (
    NeuralSignalConfig,
    NeuralSignalGenerator,
)

generator = NeuralSignalGenerator(
    NeuralSignalConfig(n_channels=32, fs=30_000.0, seed=17)
)
block = generator.generate_with_truth(
    count=3_000,
    intent=(0.5, 0.2),
    drift_progress=0.25,
)
```

`block.signal` has shape `(sample, channel)` in volts. `block.spikes` has shape
`(sample, hidden_unit)` and is intended for simulator validation and controlled
studies. Normal signal consumers can use `generate` or a prepared
`generate_into` output buffer without requesting truth. Generation is
reproducible after `reset`; the generator advances its absolute `sample_index`
after each call. Changing intent or drift at block boundaries gives a
deterministic, piecewise-constant control trajectory.

`NeuralDriftConfig.rotation_degrees` is a population-wide endpoint rotation.
For heterogeneous drift, set `per_unit_rotation_std_degrees` and optionally
`per_unit_rotation_limit_degrees`. Each hidden unit then receives a
seed-derived, zero-mean truncated-normal endpoint offset; the same configuration
and seed reproduce the same offsets. `generator.drift_fingerprint` is a compact
identity for the prepared endpoint rotations and can be stored with simulation
provenance. The fingerprint documents the realized drift but does not replace
the full configuration and seed. A finite limit that is extremely narrow
relative to the requested standard deviation is rejected rather than producing
clipped values with artificial point masses at the bounds.

## Common reference

Integer ADC arrays are promoted to `float64` before common-average or
common-median referencing. This prevents truncation and integer overflow.
In-place common referencing therefore requires floating-point input.

## Time-frequency coordinates

Spectrogram functions return window-center times by default. Use
`time_reference="start"` or `"end"` when the application requires another
convention:

```python
from neurale.signal import stft_spectrogram

power, frequencies, times = stft_spectrogram(
    signal,
    window_size=0.5,
    shift=0.05,
    time_reference="center",
    smoothing_width_hz=2.0,
)
```

`smoothing_width` is measured in frequency bins. `smoothing_width_hz` is
measured in hertz, and the two options are mutually exclusive.

## Robust TTL detection

TTL event extraction supports hysteresis and stable-duration checks:

```python
from neurale.data import EventSeries

events = EventSeries.from_ttl(
    ttl,
    sampling_rate=1000,
    low_threshold=0.2,
    high_threshold=0.8,
    debounce=0.002,
)
```

Transitions are timestamped at the first sample of the accepted stable state.
