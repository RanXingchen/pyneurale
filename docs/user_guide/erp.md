# Offline ERP

## Extract and average event-locked epochs

```python
import numpy as np
from neurale.data import Event, EventSeries, SignalArray
from neurale.features import erp_epochs

signal = SignalArray.from_array(
    np.zeros((600, 2)), fs=100, channel_names=("C3", "C4"),
    channel_types="eeg", units="uV", name="neural",
)
events = EventSeries((Event(2.0, label="cue"), Event(4.0, label="cue")))
epochs = erp_epochs(
    signal, events, tmin=-0.2, tmax=0.5,
    label="cue", baseline=(-0.2, 0.0),
)
waveform = epochs.average()
print(epochs.data.shape, waveform.data.shape)
```

Use real event onsets and signal values for analysis. Both epochs must fit
entirely within the signal; the baseline interval is relative to each onset.

`neurale.features.erp` extracts typed event- or trial-locked epochs through the
shared slicing implementation. `ERPEpochs.data` is ordered as
`(epoch, sample, channel)`; `ERPWaveform.data` is `(sample, channel)`. The
one-dimensional `time` coordinate is relative to the event onset or trial
start on the selected signal's sampling grid.

An explicit baseline is a half-open relative interval `[start, end)`. Baseline
means are calculated independently for every epoch and channel. NaN values are
not omitted: baseline correction and averaging use ordinary NumPy propagation.
ERP extraction is strict-only. An epoch extending outside signal coverage raises
`ValidationError`; multiple epochs still need equal lengths and identical
relative grids. ERP performs no clipping, padding, interpolation, resampling, or
silent epoch removal.

## Time coordinates and clocks

`ERPEpochs` and `ERPWaveform` carry several time-related fields that are easy to
confuse:

`time`
:   The anchor-relative sample axis shared across epochs (or samples, for a
    waveform). It is `epoch.time - anchor_local` on the selected signal's
    sampling grid, so `time[0]` need not be zero when `tmin` is negative.

`source_clock`
:   The original signal clock, kept as source metadata only. Its offset does not
    apply to the relative `time` axis.

`anchor_clock`
:   The clock that interprets the raw onsets in `selections`. It is the
    event-series clock for `Event` anchors, and `None` for `Trial` anchors,
    whose bounds are already recording-reference seconds.

`anchor_reference_time`
:   Each anchor converted to recording-reference seconds -- for an event,
    `to_reference_time(event.onset, anchor_clock)`; for a trial, `trial.start`.
    Stored alongside `selections` and `anchor_clock` so the full coordinate of
    every anchor is preserved losslessly without the original collection.

`source_clock` and `anchor_clock`, when both present, must share one
synchronization domain; a result mixing clocks from different sessions is
rejected on construction.

Filtering, resampling, and normalization are separate signal operations; ERP
extraction does not apply them implicitly.
