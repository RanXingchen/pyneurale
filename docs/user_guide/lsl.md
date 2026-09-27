# LSL acquisition

Install `pip install "pyneurale[lsl]"`. The built-in `lsl` provider receives one
continuous numerical LSL stream through a spawned Python acquisition process.
It supports int16, int32, float32 and float64 at a positive nominal sample rate.
No hardware SDK or PyNeurale compilation is needed when installing a wheel.
Some platforms require a separate precompiled liblsl; set `PYLSL_LIB` to its
library path if pylsl cannot find it. Importing devices or listing providers
does not load liblsl; `devices.list()` reports registration, not readiness.

## Discover and connect

```shell
python -m neurale.devices.lsl
python -m neurale.devices.lsl --source-id YOUR_SOURCE_ID
```

The second command prints full XML and original channel labels/units. Equivalent
Python APIs are `neurale.devices.lsl.discover(timeout=1.0)` and
`lsl.inspect(source_id="...", timeout=1.0)`. Discovery does not subscribe to samples.

```python
from neurale import devices

def main():
    with devices.open("lsl", source_id="YOUR_SOURCE_ID") as device:
        # Build and prepare/arm a StreamRunner with device.schema/device.source.
        # Then call device.start() and run the runner.
        # Stop and join the runner before leaving this context.
        print(device.schema)

if __name__ == "__main__":
    main()
```

See `examples/lsl_stream.py` for a complete native runner and a simulated outlet:
run `python examples/lsl_stream.py --publish --seconds 60` in one terminal and
`python examples/lsl_stream.py` in another. Use the same receiver with an eego
EEG source ID after enabling EEG LSL output in the manufacturer's software.
TTL channels embedded in that EEG stream remain ordinary channels. Independent
event streams, string markers and hardware configuration are not supported.
Physical eego compatibility and end-to-end latency require hardware validation.

Selectors `source_id`, `stream_name`, `stream_type` and `uid` are combined with
AND; supply at least one. Zero matches time out; multiple matches are errors.
A UID identifies a particular outlet instance and may change after restart.

| Option | Default | Meaning |
|---|---:|---|
| `max_samples` | 256 | Upper bound per read, not a target to accumulate |
| `buffer_seconds` | 2 | liblsl buffer length, integer seconds |
| `resolve_timeout` | 1.0 | Discovery interval and full-description timeout |
| `data_timeout` | 5.0 | Maximum silence after acquisition starts |
| `max_lag` | 0.5 | Maximum age of the oldest sample in a block, seconds |

The existing `timeout` option controls the provider process; increase it if a
longer discovery interval is configured. `queue_capacity` counts whole frames.
Neither sample rate nor gain is configured by this provider. Samples preserve
their dtype, values and channel order. Missing labels become `ch1`, `ch2`, etc.
Only directly representable, uniform units become a schema physical unit;
microvolts and mixed-unit streams are marked unspecified and are never silently
scaled. Inspect the original XML before configuring a unit-sensitive processor.

## Data and time signals

Every frame contains two sample-major blocks with identical sample indices:

| Signal ID | Channels | dtype |
|---|---|---|
| 1: data | Original channels | Original supported numerical dtype |
| 2: timestamps | `lsl_source_seconds`, `host_monotonic_seconds` | float64 |

Both timestamp columns use seconds; their schema unit is unspecified because
the current physical-unit enum has no seconds member. Original LSL doubles are
preserved exactly. The second column is the estimated sample time in the
default native runtime's monotonic clock. Float64 seconds retain LSL precision,
not integer nanosecond precision at arbitrarily large clock values.

Configure processing for signal ID 1; signal ID 2 is auxiliary timing data, not
an EEG channel. Identity forwarding and recording preserve both blocks. A
processor changing sample count or meaning must define its own output timing;
it must not attach the original timestamps to unrelated output samples.

Clock correction uses liblsl's extended estimate plus a bracketed measurement
between `lsl.local_clock()` and the native runtime clock. It polls every
second and faults when the last actual remote clock measurement is older than
five seconds. Reading a cached estimate does not renew its freshness. Each block
has a `ClockSyncSnapshot` containing the estimated uncertainty (LSL uncertainty
plus the local measurement bracket); it is not a bound on hardware timestamp
error or unmodelled drift. Device ticks are nanoseconds relative to the first
LSL sample, **not** hardware sample counters. The generation is 1 within a
session; reset establishes a new mapping/session timeline. Host receive time
retains its existing runtime meaning and is distinct from sample acquisition time.
Custom runtime clocks are not supported by this provider's time mapping.

LSL smoothing and timestamp postprocessing are disabled. Clock reset,
non-finite timestamps or backwards source/mapped time are faults, not silently
corrected. Synchronization does not resample data or correct the manufacturer's
timestamp placement. Nominal rate is metadata, not proof of exact sample timing.

## Failure and shutdown

Reads use a reusable NumPy buffer and return available samples without waiting
to fill a block. Empty reads wait up to 1 ms on a cancellation event. Startup
flushes pre-acquisition data once; there is no runtime flushing or automatic
reconnect. Disconnection, prolonged silence, excessive lag, queue overflow and
acquisition exceptions fail the source. Stop/join and close propagate errors
after cleanup. Explicit runner reset reconnects and verifies the description.

Data and timestamps are committed atomically through `Ingress.publish_frame`;
a rejected pair publishes neither block. The native consumer path allocates no
memory for this operation; the Python producer and liblsl have no hard deadline
guarantee. Timestamps alone cannot establish exact missing sample counts, and
liblsl/upstream buffering is not covered by a zero-loss guarantee.

The full description from `inspect()` can be saved alongside experiment metadata.
When recording, assign signal 2 the unit `s` and its two descriptive channel names
in the recording configuration. Both signals use existing NRF numerical streams;
no file-format extension is required.
