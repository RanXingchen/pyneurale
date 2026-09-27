# External acquisition providers

Install a device provider alongside an existing PyNeurale wheel. Providers do
not rebuild PyNeurale or link its private C++ libraries. Native providers use
the wheel's C ABI; Python-only SDKs run in a separate acquisition process.
Both publish into a bounded native queue consumed by `NativeFrameSource`.
No physical vendor adapter or hardware support claim is included with these
interfaces. The three example distributions generate deterministic test data.

## Use a provider

```python
from neurale import devices, streaming

def main():
    print(devices.list())  # Installed names; does not probe hardware or load SDKs.
    with devices.open("example.pull") as device:
        runner = streaming.StreamRunner(
            schema=device.schema,
            config=streaming.RealtimeConfig(),
            source=device.source,
            processor=streaming.IdentityNativeProcessor(),
            actuator=streaming.CountingNativeConsumer(),
            safety_controller=streaming.RecordingNativeSafetyController(),
        )
        try:
            assert runner.prepare() == streaming.StreamStatus.OK
            assert runner.arm() == streaming.StreamStatus.OK
            device.start()
            assert runner.run() == streaming.StreamStatus.OK
        finally:
            runner.stop()

if __name__ == "__main__":
    main()  # Required for spawn-based Python SDK providers.
```

`open()` connects, describes signals and allocates the queue. `start()` starts
acquisition; the context manager only closes resources. Stop and join the
runner before closing the device. To repeat, call `runner.reset()`, prepare and
arm again, then `device.start()`. Reset reconnects the provider; a changed
description fails rather than changing the running schema. No automatic
reconnect is performed.

`queue_capacity` (default 256 blocks) and `timeout` (default 5 seconds for Python
process control) are framework options to `devices.open`; other keywords go to
the provider. `device.stats` contains submitted/consumed queue-item counts and
the native ingress error code. Counts include discontinuities.

Queue exhaustion, excessive callback contention, and oversized native blocks
are explicit faults. There is no implicit dropping, queue growth, dtype
conversion, resampling, or Python fallback. Capacity and pipeline throughput
must match the acquisition rate. Shutdown requests immediately stop native
reads; SDK cancellation and joining complete on the control plane. A Python
process exceeding its shutdown timeout is terminated, the ingress is marked
failed and `close()` raises `StreamTimeoutError` after cleanup. Final statistics
remain readable after close. Python SDK `cancel()`/`close()` exceptions and
abnormal process exits raise `StreamExecutionError` after cleanup and mark the
ingress failed. Reset returns a source failure instead of reconnecting when
shutdown fails. A native SDK must itself provide a safe cancellation
and close contract.

## Write a provider

Register an entry-point factory returning either `NativeProvider(library_path)`
or `PythonProvider("your_module:YourDevice")` in the `neurale.devices` group.
Keep the entry-point module free of vendor imports; the Python factory module
is imported only by the child process. Importing or listing devices loads no SDK.
Direct construction with `devices.Device(provider, config=...)` is also available.

The independently installable examples are in `examples/device_providers/`:

- `pull`: C++ SDK-read style, using `PullWorker`.
- `callback`: SDK-callback style, using `CallbackIngress`.
- `python`: a Python SDK-style worker, managed by the process host.

Each example emits the same four frames: a two-channel signal at 1000 Hz
(including a short block and known loss), an auxiliary signal at 500 Hz, and
an integer event. A final device-restart message also exercises ordered events.

Native SDK headers are available from
`neurale.devices.provider.get_include()`. `provider.h` is C-compatible;
`helpers.hpp` requires C++17. The example CMake projects discover these files
through the installed Python interpreter. They link only platform thread
support and their own vendor dependencies, never PyNeurale's C++ libraries.
For a local build with PyNeurale and scikit-build-core already installed:

```shell
python -m pip install --no-build-isolation examples/device_providers/pull
python -m pip install --no-build-isolation examples/device_providers/callback
python -m pip install --no-build-isolation examples/device_providers/python
```

Distribute native plugins as platform wheels. Bundle dependencies only where
their licences allow it; otherwise require the vendor SDK installation and
report load failures. The native plugin must match the host architecture.

## Data and timing contract

`Signal` describes actual negotiated data, not requested settings. Describe all
signals before acquisition; sampled signals have a positive rational rate,
events have zero rate and exactly one event per block. The supported scalar
types are int16, int32, float32 and float64. Payload layout is explicitly
sample-major or channel-major. Uncalibrated ADC counts use an unspecified unit.

Python `ingress.publish(signal_index, array, metadata={...})` requires an exact
native-endian dtype and contiguous two-dimensional array. Native `publish`
accepts a borrowed pointer and byte count. Both copy the complete block before
returning, so SDK callback storage can immediately be reused. Concurrent calls
are supported; providers must preserve ordering within each signal.

Optional metadata keys are `sample_index` and `device_tick`, both referring to
the first sample. Without a sample index, the bridge assigns per-signal indices.
A signal declaring `sample_counter=True` must supply a tick on every data block.
Hardware packet decoding, counter unwrapping, and restart detection belong to
the provider. Do not infer exact sample loss from an ambiguous packet number.

`ingress.gap(index, metadata={"missing_samples": count})` submits known loss.
Omit the count for unknown loss; `restart=True` reports a device restart.
An optional `sample_index` specifies the next actual position. These are
independent ordered items and need no following data block. Only known loss
advances automatically assigned indices. `finish()` reports normal EOF after
all publishers have finished; `fail()` reports a source fault.

Measured clock mappings optionally supply `tick_reference`, `host_reference_ns`,
`tick_rate_numerator`, `tick_rate_denominator`, `uncertainty_ns`,
`clock_generation`, and `synchronized`. Host references must use the same
monotonic clock domain as the runtime; a wall-clock timestamp is not suitable.
Mappings are not synthesized. Host receive time remains assigned by the runtime
after a successful source read, so queue and SDK dwell are not included in that
timestamp. Measure end-to-end latency separately when it matters.

Python runs only in the producer process and in explicit control-plane reset;
the acquisition consumer and downstream realtime path invoke no Python. This
isolates the GIL, but does not guarantee deadlines inside a Python SDK or driver.

## Validation

Run `tools/validate_device_providers.py` in an environment containing an installed
PyNeurale wheel and the three separately built example packages. It exercises
public realtime runners, repeated sessions, checks exact values and ordered
events through a research sink, and prints timings for each path. The timings
characterize examples, not physical devices. Native CTest additionally covers
zero steady-state heap allocation, concurrent publishing, and overflow faults.

## Atomic frames and the runtime clock

Python producers may call `ingress.publish_frame([(index, array, metadata), ...])`
to publish several distinct signals atomically in one frame. Each array obeys
`publish`'s dtype/layout contract; all validation and copies complete before the
single queue item is made visible. Gaps still use `gap()`. Statistics count queue
items, not signal blocks. Existing native C ABI v1 plugins remain compatible.

`neurale.devices.provider.host_time_ns()` reads the default native runtime clock
for measured clock-domain mappings. Do not substitute an unrelated Python clock.
The optional built-in [LSL provider](lsl.md) uses both facilities to preserve
paired data and per-sample timestamps.
