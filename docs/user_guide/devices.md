# Simulated acquisition device

`neurale.devices.simulation.SimulatedNeuralDevice` turns a prepared
`SignalGenerator` into the existing native streaming source. The generator owns
sample values; the device owns frame boundaries, schema metadata, device ticks,
clock-sync snapshots, pacing, reset, cancellation, and finite end-of-stream.

```python
from neurale.devices.simulation import (
    DeviceClockRestart,
    KnownSampleLoss,
    SimulationFaultPlan,
    SimulatedNeuralDevice,
    TransientWouldBlock,
)
from neurale.signal.simulation import SignalGenerator

generator = SignalGenerator.tones(
    channel_count=2,
    sample_rate=4000.0,
    frequencies=[100.0, 150.0],
)
device = SimulatedNeuralDevice(
    generator,
    samples_per_frame=4,
    channel_names=["electrode_0", "electrode_1"],
    channel_impedances_ohm=[8_000, 12_000],
    physical_unit="volts",
    faults=SimulationFaultPlan(
        events=(
            TransientWouldBlock(frame_ordinal=4),
            KnownSampleLoss(frame_ordinal=8, n_samples=4),
            DeviceClockRestart(frame_ordinal=12, device_tick_origin=0),
        )
    ),
)

# Pass device.schema and device.source to neurale.streaming.StreamRunner.
```

The resolved channel names are frozen in both `device.channel_names` and the
sampled signal's `device.schema.signals[0].channel_names`, so downstream stages
can inherit them without another argument.
Optional `channel_impedances_ohm` measurements are likewise frozen in payload
column order. Leave the argument as `None` when impedance was not measured;
`BadChannelRemovalStage` can use the metadata without receiving another array.

Payload layout is fixed to `(n_samples, n_channels)`. `sample_dtype` selects
the payload element type: the default `"float64"` stores generated values
unchanged, and `"int16"` rounds each value to the nearest integer and saturates
it to the 16-bit range, which models a device shipping raw ADC counts. Configure
the generator in count units for that case and use an honest `physical_unit`
rather than labelling uncalibrated counts as volts. `samples_per_frame` fixes
both the normal frame size and prepared frame capacity. A finite source may emit
one shorter final frame. Each new simulated source receives its native schema,
signal, clock-domain, channel-set, and nonzero session identifiers internally;
`reset()` retains the session ID.

The default `paced=True` models delivery against the native monotonic clock.
The first valid source read establishes the acquisition epoch; time spent
constructing the device or preparing and arming a runner does not create a
startup catch-up burst. After `reset()`, the next valid read establishes a new
epoch in the same way.
Use `paced=False` only when wall-clock delivery is not part of the scenario,
such as deterministic offline tests. `cancel()` wakes a paced read promptly.
`reset()` restores the configured starting sample index, sequence, and device
tick. Host receive timestamps are assigned by `StreamRunner` after
the native source returns; the device does not fabricate them.

Acquisition events are applied immediately before the named next data-frame
ordinal. A `would_block` response or a discontinuity does not consume that
ordinal. Known loss advances the absolute sample position and sample-counter
tick, so the next payload still contains the generator values for its true
absolute positions. Tick jumps, device restarts, and known loss are delivered
as the existing typed discontinuity contract; the runtime does not infer a
second copy. Configure the runner's discontinuity pool for the event schedule
and queue topology.

Advanced timing behavior belongs to the immutable `SimulationTimingConfig`.
Its `initial_sample_idx` selects the first absolute generator position and
`initial_device_tick` selects the first sample-counter value. Device ticks are
sample counters when enabled. `clock_offset_ns` shifts only the clock-sync
mapping, integer `clock_drift_ppm` changes the effective device clock rate
(positive is faster), and `clock_sync_uncertainty_ns` records fixed simulated
uncertainty. Tests that need deterministic pacing can place a forward-only
`ManualHostClock` in that config and advance it from their control thread. It
controls host pacing deadlines and `ClockSyncSnapshot.host_time_reference_ns`;
it does not alter `DeviceTick`.
`DeviceClockRestart` operates on that separate simulated device-tick domain.
`Disconnect` and `SourceFault` use the existing terminal `source_failure` path;
the simulator does not reconnect.

`cancel()` is idempotent and remains recoverable through `reset()`. `close()` is
also idempotent, but terminal: it cancels any blocked read and a later
`reset()` raises `StreamStateError` rather than silently reopening the source.
Close the runner before closing the device when both are in use.

`reset()` needs the source to be idle. A reset issued while a read is still in
flight raises `StreamStateError` instead of racing the acquisition state, so
cancel and then join the runner before resetting. `cancel()` and `close()` are
callable at any time, including from another thread while a read is blocked.

The simulator is a concrete device rather than a generic device framework. It
does not generate sine/noise values itself, inject vendor packets, write
recordings, or call Python for each frame. All simulator-specific classes are
owned by `neurale.devices.simulation`; `neurale.devices` does not re-export them
as generic device contracts.

For feedback simulations, `IntentDrivenNeuralDevice` is a separate opt-in
device that owns a native `NeuralSignalGenerator` configured from an immutable
`NeuralSignalConfig`. It can consume manual controls or bind
once to a native in-memory intent capability such as
`CenterOutSession.intent_source`; the two modes are mutually exclusive. Bound
generation reads the latest intent without a Python callback on the acquisition
thread. An optional `NeuralDriftSchedule` maps the source's context ordinal to
drift progress without changing the experiment-session API.

```python
from neurale.devices.simulation import IntentDrivenNeuralDevice, NeuralDriftSchedule
from neurale.signal.simulation import NeuralSignalConfig

device = IntentDrivenNeuralDevice(
    NeuralSignalConfig(),
    samples_per_frame=30,
    drift_schedule=NeuralDriftSchedule(start_ordinal=32, end_ordinal=63),
)
device.bind_intent_source(center_out_session.intent_source)
```

`pop_applied_control()` returns bounded acquisition-side evidence containing the
exact intent, drift progress, sample range, and generated frame sequence used
for one frame. Drain it while long sessions run and require
`dropped_applied_control_count == 0` before treating that evidence as complete.
NRF recording remains the durable session record and replay input; it is not the
live feedback transport.

No physical/vendor device adapter, SDK distribution, or vendor extra is shipped
in this release. The [provider framework](device_providers.md) supports independently
installed adapters and SDKs. The existing
``NativeFrameSource``/``NativeActuator`` boundary remains the integration point
for device-specific implementations; their absence does not add a fallback or
weaken the simulated-device contract. See the
[support information](https://github.com/RanXingchen/pyneurale/blob/main/SUPPORT.md)
for the supported versus unavailable distinction.
