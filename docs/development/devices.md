# Devices integration architecture

## Status and scope

This document is the normative devices ownership and hardware-integration
contract. It freezes boundaries for current and future device adapters. The
checked-out repository provides ``neurale.signal.simulation``, concrete simulated
devices and the external acquisition-provider framework. The generic native
runtime described in {doc}`native_streaming` is current; vendor-specific device
names in this document remain target owners, not implemented adapters.

The simulated devices and external-provider framework are implemented.
Physical/vendor adapters, SDK distributions, and device extras remain unavailable;
the standalone example providers do not establish hardware support.
See {doc}`../user_guide/device_providers` for the provider API and SDK.

``Device`` is a control-plane resource owner, not a required vendor superclass.
``GenericDeviceSource`` bridges the versioned provider C ABI and bounded ingress
to the existing native source contract. Python SDK providers use a spawned
process and a native shared-memory ingress. Neither path changes runtime semantics.

## Unique ownership

The allowed code-dependency direction is one-way toward lower-level owners:

```text
neurale.devices.simulation
    -> neurale.signal.simulation

neurale.devices
    -> neurale.streaming
```

``neurale.signal.simulation`` depends on neither ``neurale.devices`` nor
``neurale.streaming`` and must remain usable on its own. The synthetic data
flow runs the opposite way and is not a code dependency:

```text
neurale.signal.simulation
    -> simulated device (neurale.devices.simulation)
    -> NativeFrameSource (neurale.streaming)
```

More precisely:

- ``neurale.streaming`` owns ``NativeFrameSource``, ``NativeActuator``,
  ``Frame``/logical signal blocks, ``StreamSchema``, clock mapping,
  continuity/discontinuity, and generic source/actuator fault, cancellation,
  reset, and end-of-stream semantics.
- ``neurale.signal.simulation`` owns algorithmic sampled-signal generation. Its
  stateless native ``float64`` generators cover zeros, per-channel constants,
  sine/sums of known tones, counter-based seeded uniform noise, and finite or
  repeating supplied sample matrices. Its stateful neural surrogate combines
  intent-tuned hidden spikes, colored background noise, stochastic LFP
  dynamics, nonstationarity, and controlled drift. Both write sample-major
  values into caller-owned storage. This package does not own device pacing,
  frame/block formation, discovery, transport, experiment semantics, or faults.
- ``neurale.devices`` owns concrete acquisition, peripheral, actuator, and
  stimulator adapters. Its ``neurale.devices.simulation`` submodule owns
  simulated hardware behavior -- pacing, frame/block formation, bounded
  buffering, lifecycle, injected loss/faults, reset, cancellation, and end of
  stream -- and may compose a signal generator but does not reimplement its
  signal equations.
- A concrete adapter may implement or expose the existing
  ``NativeFrameSource`` or ``NativeActuator`` contract directly. Those types
  remain the acquisition and actuation runtime boundaries; devices introduces
  no parallel device-stream interface.

The current ``SimulatedNeuralDevice`` is the first concrete acquisition
adapter. It owns one prepared native ``SimulatedNeuralSource``, exposes it via
``source``, and composes a caller-supplied ``SignalGenerator``. Construction
fixes one sample-major sampled signal, its payload dtype, channel count and
control-plane channel names, rational sample rate, frame size, physical unit,
and optional finite extent. Native schema, signal, clock-domain, channel-set,
calibration, reference, and session identifiers are source-owned rather than
public constructor inputs. The frame size is used until a finite final short
frame; changing it does not change values at any absolute sample index. Pacing uses the
generic native clock and ``cancel()`` wakes a blocked read. Construction and
``reset()`` leave the pacing epoch unset; the first valid acquisition ``read()``
establishes it, so control-plane prepare/arm delays never cause a catch-up
burst. ``reset()`` restores the configured initial sample index, sequence zero,
and device tick and requires the next valid read to establish a new pacing
epoch. The source allocates no memory during ``read()`` and uses only
runtime-owned frame storage.

``sample_dtype`` selects the payload element type, which is a schema fact the
runtime cannot renegotiate later. ``float64`` is the default and generates
straight into the payload. ``int16`` models a device that ships raw ADC counts:
generation targets a scratch buffer sized once at construction, and the
acquisition path then rounds each value to the nearest integer and saturates it
to the 16-bit range. That conversion is a declared cost on the acquisition path
-- the same determination item 6 below requires of a vendor adapter -- and it
allocates nothing, because the scratch is not sized per read. A non-finite
generated value is reported as a source failure rather than coerced to zero.
Only ``float64`` and ``int16`` are accepted; the remaining schema dtypes are
rejected at construction rather than silently approximated.

The source writes device tick and ``ClockSyncSnapshot`` fields when configured.
It deliberately leaves ``host_received_ns`` and ``source_received`` unset;
the generic acquisition runtime stamps both immediately after a successful
read, as required by the shared source contract. This simulator introduces no
packet format, discovery API, generic Device base, recording dependency, or
vendor fault/loss framework.

Nominal pacing is the retained pacing requirement. The deterministic
simulated-acquisition contract is complete. ``paced=False`` emits as fast as
the runtime accepts frames, except that an explicitly scheduled bounded stall
is still honored. ``paced=True`` uses absolute deadlines from the first
successful data read; it never accumulates relative sleeps. A public
``ManualHostClock`` supplies the same native host-clock interface for tests and
can only move forward. It drives pacing deadlines and host clock-sync
references, not the simulated hardware ``DeviceTick``. Normal tests therefore
do not need long wall-clock waits.

Advanced position and device-clock fields are grouped in the immutable public
``SimulationTimingConfig``. ``device_ticks=True`` has exactly one meaning: the device tick is a sample
counter, so and only so the schema declares ``DeviceTickTracking::sample_counter``.
``initial_device_tick`` fixes its origin. Integer ``clock_drift_ppm`` changes the
effective device/sample-clock rate by ``1 + ppm / 1_000_000``; positive drift is
faster. Signed ``clock_offset_ns`` changes only the tick-to-host mapping in the
snapshot, not the runtime host-receive timestamp. Uncertainty is a fixed bound in
the current simulator. A tick jump or clock restart emits an explicit barrier and
increments the snapshot generation; a restart never reuses the prior mapping.

The immutable public ``SimulationFaultPlan`` contains the typed acquisition
events ``KnownSampleLoss``, ``TransientWouldBlock``,
``BoundedStall``, ``Disconnect``, ``SourceFault``, ``DeviceTickJump``, and
``DeviceClockRestart`` form one immutable schedule, limited to 4096 entries.
Every event is positioned immediately before an exact *next data-frame ordinal*.
Returning ``would_block`` or ``discontinuity`` does not advance that ordinal;
same-ordinal events retain caller order. A known loss consumes the configured
absolute acquisition/sample-counter interval, omits those generated values, and
publishes the existing ``SignalGap`` before the following frame. The next frame
sequence remains the next PyNeurale sequence: sample index, device tick, frame
sequence, and host time are deliberately distinct. Unknown-size loss is not a
current event type and is never converted into an invented count. The schedule is
fixed at construction and replayed identically on ``reset()``: it is a
control-plane input, not a general fault-injection framework, and it adds no
runtime hook, no injection API, and no way to perturb a source the caller did
not configure.

``Disconnect`` and ``SourceFault`` both terminate through the existing
``source_failure`` path; this simulator defines no reconnect/backoff policy.
Cancellation wakes pacing and stall waits, and ``reset()`` restores the initial
sample/tick position, sync generation, schedule cursor, sequence, cancellation,
and unset pacing epoch. Explicit gaps clear the continuity checker anchor before
the next frame, so the runtime does not infer a duplicate; stateful processors
and the terminal consumer receive the same ordered discontinuity. Runtime
``discontinuity_capacity`` must be provisioned for the configured event delivery
and queue topology; exhausting it remains the normal bounded fault path.

The small public lifecycle is frozen without introducing a common device base.
``cancel()`` is concurrent, idempotent, and recoverable by ``reset()``;
``close()`` is concurrent and idempotent but terminal. A closed simulated
source continues to report ``stopped`` and rejects ``reset()`` with the
existing invalid stream-state contract.

``reset()`` requires a quiescent source, and enforces it rather than assuming
it. ``read()``, ``read_message()``, and ``reset()`` share one compare-and-exchange
gate, so a reset issued while a read is still in flight is refused with
``invalid_state`` instead of rewriting the sample offset, device tick, pacing
epoch, and schedule cursor that the acquisition thread is using. ``cancel()``
and ``close()`` stay outside that gate, because they are what makes a blocked
read return; the recovery sequence is cancel, wait for the read to return, then
reset. This is the same gate and the same reason as the native replay source's,
and the runner path never contends for it -- ``NativeStreamRunner::reset()``
joins its workers first. These operations execute on the native source. Python
is only the configuration and control facade, while ``StreamRunner.run()``
releases the GIL and invokes no Python component for the native source path.
Simulator-specific configuration and fault controls are public only from
``neurale.devices.simulation``. The ``neurale.devices`` domain package exports
none of them, so names such as ``Disconnect`` and ``DeviceClockRestart`` cannot
be mistaken for generic contracts shared by future physical adapters.

The native device test uses both clocks: manual-clock assertions freeze exact
deadlines and a host-monotonic sanity case emits three one-sample frames at
100 Hz. The latter expects the two acquisition intervals (20 ms nominal) to
take at least 15 ms and less than one second. This deliberately loose scheduler
check is not a strict-realtime or universal pacing-accuracy claim.

A concrete device or session object is permitted when it must own an SDK
connection, configuration, vendor resources, vendor callbacks or workers, or
hardware-specific health and diagnostics. Its object model stays private or
concrete-device-specific. A shared base becomes eligible only after multiple
real implementations demonstrate the same stable contract; devices does not
require one.

Device adapters do not own signal processing, feature extraction, kinematics,
decoding, recording, experiments, or visualization. Those domains consume or
produce typed streams through their existing owners.

## Streaming boundary

``StreamSchema`` is fixed before startup. It declares each signal's dtype,
sample/channel layout, rational rate, channel count, maximum block size, clock
domain, unit and other stable references. A logical signal block is one
``SignalBlockHeader`` plus its bounded slice of a frame payload; it is a member
of a frame, not an independently accepted runtime message. A source fills the
caller-owned ``MutableFrame`` storage supplied by the runtime and must not
retain its frame, spans, or payload pointers.

``DeviceTick`` is the source device-domain counter or tick when one is
available. When a signal declares ``sample_counter`` tracking, continuity follows
the existing unsigned-counter wrap and device-restart semantics; the type itself
does not promise strict monotonicity across a clock restart. ``ClockSyncSnapshot`` records a measured mapping from a device-tick
reference to the runtime host clock, including exact tick rate, uncertainty,
clock domain, generation, and synchronization flags. An adapter must not invent
device ticks or a synchronized snapshot when the hardware cannot support them.
The runtime stamps ``host_received_ns`` after ``read`` returns; a vendor
callback timestamp is distinct source provenance and must not be relabelled as
that runtime timestamp.

Continuity is checked per signal from frame sequence, absolute sample indices,
and device ticks where declared. Known recoverable continuity breaks -- missing
samples, a source gap, a device restart, or a recoverable bounded-handoff loss --
are represented as an ordered ``Discontinuity`` with one or more ``SignalGap``
records. A bounded-handoff overflow that cannot be recovered safely may instead
enter the normal source-fault path under the concrete adapter's frozen policy;
the runtime's own ingress overflow is a fault, and no generic contract requires
every overflow to recover as a discontinuity. No adapter may silently drop data
or conceal a break by restarting counters or joining samples across segments.
Exact missing-sample counts are reported only when supported by sample-index or
device-tick evidence; an unknown loss must not be fabricated as an exact count.

The existing lifecycle meanings remain unchanged; the normative source and
runner contract lives in {doc}`native_streaming` and is not redefined here:

- ``would_block`` is transient and non-faulting;
- ``discontinuity`` is an ordered source message, not a fault;
- ``end_of_stream`` terminates the current source session normally;
- ``stopped`` is non-faulting when produced in response to a requested
  shutdown;
- any other unexpected source or actuator status enters the generic fixed
  fault and safety-stop path;
- ``cancel()`` is concurrent and idempotent, and must make a blocked operation
  return promptly;
- ``reset()`` participates in the existing runner reset contract: the runner
  returns to ``Created`` and must be ``prepare()``-ed again before another run,
  and it must not conceal a failed reconnect or renegotiation.

An actuator receives only a callback-scoped ``FrameView`` inside an
``ActuatorCommand``. ``NativeActuator::submit`` rejects an expired command
before the concrete write is entered; a concrete stimulator must not bypass
that boundary. Its ``flush``, ``reset``, and ``cancel`` operations use the same
bounded, explicit-status lifecycle rather than a vendor exception or an
unbounded shutdown wait.

The generic watchdog observes bounded runtime heartbeats and stage deadlines.
It does not make an unbounded vendor call realtime-safe and cannot cancel a
vendor operation that the concrete adapter cannot interrupt.

## Vendor integration boundary

Physical discovery, optional SDK loading, connection, negotiation, schema
construction, configuration, resource allocation, and vendor-worker startup
happen on the control plane before the realtime runtime starts. Vendor
protocols, SDK and driver types, transport details, callbacks, polling, TCP or
UDP, shared memory, USB, and BLE/GATT remain private to the concrete adapter.
They do not enter generic streaming headers or public core-wheel dependencies.
Where a vendor protocol exposes packet or sequence-number concepts, the provider
interprets them privately and submits a known/unknown loss or restart. The bridge
maps these ordered items to ``SignalGap``/``Discontinuity``. Raw packet formats
do not enter generic streaming contracts.

Vendor callback buffers are ephemeral unless the SDK documentation explicitly
guarantees a longer lifetime. When a callback or polling path cannot directly
satisfy the native realtime contract, the required shape is:

```text
vendor callback / polling / SDK-owned worker
    -> bounded preallocated ownership handoff
    -> NativeFrameSource
```

The callback copies or transfers only into storage whose capacity and ownership
were fixed during prepare; queue-full behavior is explicit and results in a
gap or fault under the concrete adapter's frozen policy. It cannot retain a
runtime frame lease, grow a queue, allocate a fallback buffer, or expose the
vendor buffer after callback return.

Vendor SDKs are optional dependencies. Importing ``neurale`` or an unrelated
domain must not import, load, initialize, discover, or probe them. Requesting a
concrete adapter with a missing SDK fails explicitly through the public
exception hierarchy; it never installs a hidden simulated or CPU fallback.

``neurale.devices`` is the logical domain owner. A concrete vendor adapter may
ship in an optional device-specific distribution (for example
``neurale-device-blackrock``) while directly implementing PyNeurale's native
streaming contracts; distribution does not create a second device contract.
The ``neurale.devices`` entry-point group discovers installed provider definitions
without loading them. Only an explicit open loads the selected provider.

## Realtime evidence and recording provenance

The realtime prohibitions and frame-ownership rules in
{doc}`native_streaming` apply unchanged. A strict-realtime statement includes a
vendor callback, worker, driver, or SDK only when that exact path and
environment have separately executed the required allocation, blocking,
latency, cancellation, fault, and leak validation. Evidence for the bounded
handoff and downstream native runtime alone does not validate the vendor side.

The native simulated-BCI integration test owns the simulated device's
steady-state allocation evidence. It arms the allocation counter
after the chain has warmed up and requires a zero count for two prepared
topologies: ``SimulatedNeuralSource -> SOS -> LMP -> consumer`` and
``SimulatedNeuralSource -> LMP -> Kalman decoder -> consumer``. Those two
chains are the whole claim -- the resampler, Hilbert, and multitaper bandpower
chains in the same file are run for value correctness and frame-size invariance
only, with allocation tracking off, and the per-adapter allocation gates in
{doc}`validation_matrix` remain their owners.

Measured cost for the same layer is the devices characterization described in
{doc}`benchmarks`: `neurale_devices_simulation_benchmark` for generation cost,
source read latency, pacing error and jitter, cancellation, scheduled loss, and
the simulated device -> native sink integration, and
`benchmarks/devices/benchmark_simulation.py` for the simulated device ->
`SessionRecorder` integration. Those are measurements of the *simulated*
device. They characterize the shared framing, clock, continuity, and runtime
path a concrete adapter also uses, and they say nothing about any vendor SDK:
a physical device's numbers are its own item 23 above.

Recording is not a device responsibility. At runtime acceptance,
``neurale.streaming`` assigns the global data-message ordinal and acceptance
host time across frames and discontinuities. Recording preserves that
identity together with frame/block schema, absolute sample indices, device
ticks, complete clock-sync snapshots where available, gaps, and original
source provenance. Device adapters must therefore provide stable session
schemas and truthful timing fields, but they neither assign recording
ordinals nor write NRF or spool data. See {doc}`native_recording_replay`.

## Integration evidence

The native integration test composes the three owners without moving their
boundaries. It exercises the simulator directly through
``NativeStreamRunner`` and a native consumer,
then through three representative internal pipelines: ``SOS -> LMP``,
``FIR -> Resampler -> Hilbert``, and ``IIR -> Bandpower``. It also composes
``LMP -> native Kalman``.
Different legal device frame sizes must produce the same flattened feature
observations and observation indices. A known sample loss reaches both a
stateful chain and the terminal consumer as one discontinuity; reset repeats
the decoded run; an injected source fault leaves no frame or discontinuity
lease outstanding. The test enables the shared allocation tracker only after
four native source frames and disables it from the terminal flush, so its zero
count covers the prepared steady-state source, pipeline, decoder, and consumer
path rather than control-plane construction or teardown.

The public device/recording integration suite uses the shipping continuous-file
spool and composes ``SimulatedNeuralDevice ->
StreamRunner -> SessionRecorder -> canonical NRF -> native exact replay``. It
checks generated payloads against the same absolute-position generator,
per-block sample index/device tick provenance, the recorded gap, verified
accounting, replayed clock-sync fields, and terminal resource counts. It is
paired with a paced source-fault case whose accepted prefix is finalized as
verified-incomplete, retains the committed fault record, and replays to the
native ``abnormal_end`` terminal. Whole-spool locking is no longer a prerequisite.
Tests of the private mapped backend still require tmpfs/memlock. Windows coverage uses
the public persistent path; a private memory store is not acceptance evidence
for this chain.

These are integration facts for the exact simulated chains and executed build
environment. They do not validate a physical NSP, a vendor SDK callback or
worker, or the latency/throughput of algorithms already owned by signal,
features and decoders, and recording. The existing focused signal,
device-fault, adapter-reference, strict-chain, recorder, and replay suites
remain the owners of their detailed contracts.

## Concrete acquisition adapter requirements

This section is the frozen completion standard for every future physical
acquisition endpoint in the Devices domain -- NSP, IMU, eye tracker, EMG
amplifier, force sensor, or anything else that produces sampled or event data.
It applies per device, not per vendor and not per release.

Each numbered item must have a **recorded answer with its source**. A source is
either an authoritative vendor document (named, with version) or an executed
measurement in this repository. "The SDK probably does X", an inference from
example code, and a value observed once on one unit are not answers. When the
honest answer is "the vendor does not document this", that is a recorded answer
too, and it forces the conservative branch wherever a rule below has one.

### Device identity and prerequisites

1. **Manufacturer, model, and family.** The exact hardware the adapter claims,
   not a product line. Two models in one family are two determinations unless
   the vendor documents them as identical for every item here.
2. **Supported firmware.** The versions validated, and the behavior when the
   device reports one outside that set.
3. **Authoritative SDK/API version.** The exact SDK, driver, or protocol
   revision the adapter is written against, and its compatibility policy.
4. **Supported platforms.** OS, architecture, and any kernel driver, udev rule,
   privilege, or real-time configuration the path requires.
5. **Hardware configuration required before streaming.** Everything that must be
   set on the device or in vendor software before acquisition can start
   (sampling rate, channel banks, filters, references, triggers), and whether
   the adapter sets it or requires it preset.

### Stream mapping

6. **`StreamSchema` mapping.** Every field the schema freezes before startup,
   because the runtime cannot renegotiate any of them once it is running:

   - **signal identity and kind** -- one `SignalId` per signal, and its
     `SignalKind`;
   - **channel count, channel identity, and stable output order** -- the count
     alone is not a mapping. A device that delivers vendor channels 17, 3, 9,
     11 into payload columns 0-3 must have that correspondence frozen and
     recorded, together with the hardware channel identifier, electrode or
     channel name, and physical connector/bank where the vendor exposes them.
     `SignalSchema.channel_names` freezes one stable label per payload column
     when the adapter can provide them, while `channel_set_id` identifies the
     richer channel-set mapping. Adapters that leave names empty must still own
     and publish that mapping through their device metadata;
   - **measured impedance** -- when the device reports a finite non-negative
     impedance magnitude for every sampled channel, store it in
     `SignalSchema.channel_impedances_ohm` in payload-column order. An empty
     vector means impedance was not measured; adapters must not invent values;
   - **sample rate** as an exact rational, not a rounded float;
   - **dtype**, and where a vendor integer count is converted, the exact
     conversion;
   - **`SignalLayout`** -- the vendor's own memory layout, the PyNeurale layout
     (`sample_major` or `channel_major`), whether a conversion is needed, and if
     so whether that conversion runs on the realtime path and whether it copies
     or allocates. A transpose on the acquisition thread is a cost that has to
     be declared, not discovered;
   - **`nominal_block_samples` and `max_block_samples`** -- usually fixed by the
     vendor's packet or callback block size. Without them, frame-pool capacity,
     payload buffer size, and bounded-handoff capacity cannot be derived at all;
   - **units and calibration** -- the vendor's documented scaling, never a
     guessed factor. An uncalibrated raw count is declared as one rather than
     relabelled as volts;
   - **reference and channel metadata**, including `reference_id` semantics;
   - **auxiliary or event streams that the device actually exposes**. A stream
     the hardware does not produce is not declared.

### Ingress and execution

7. **Ingress mechanism.** The concrete transport: direct SDK read, vendor
   callback, polling loop, TCP/UDP socket, shared memory, USB or controller API,
   or another named mechanism. "The SDK" is not a mechanism.
8. **SDK-buffer ownership and lifetime.** Who owns each buffer the vendor hands
   over and how long it stays valid. **Vendor callback buffers are ephemeral by
   default**; only explicit authoritative SDK documentation makes them anything
   else. An ephemeral buffer is copied inside the callback or not used.
9. **Direct versus isolated acquisition execution.** Which of the two permitted
   data paths below this device uses, with the evidence that justifies the
   choice.
10. **Bounded handoff requirements.** For the isolated path: the queue's fixed
    capacity, its preallocated storage, who fills and who drains it, and the
    frozen queue-full policy (gap or fault -- never a grown queue, a fallback
    allocation, or a silent drop).

### Timing and continuity

11. **Hardware clock.** Whether one exists, its nominal rate, its stability, and
    whether the adapter can read it at all.
12. **Hardware sample counter.** Whether the device provides one, its width, and
    its wrap and restart behavior.
13. **Vendor sequence number.** Only when the protocol actually carries one.
    A sequence number synthesized by the adapter is not a vendor sequence number
    and must not be recorded as one.
14. **PyNeurale sample/frame sequence mapping.** The exact rule taking vendor
    counters to absolute sample indices, `DeviceTick`, and frame sequence,
    including what happens at wrap and at device restart.
15. **Host-clock synchronization.** How the device-tick reference maps to host
    time, the measured uncertainty, and when a `ClockSyncSnapshot` may be marked
    synchronized. An adapter that cannot measure the mapping does not emit one.
16. **Loss and gap semantics supported by actual evidence.** Which losses the
    device makes detectable, and how each becomes a `SignalGap`/`Discontinuity`.
    Exact missing-sample counts are reported only where sample-index or
    device-tick evidence supports them; an undetectable loss is reported as an
    unknown-extent break, never as a fabricated count and never concealed.

### Lifecycle and integration

17. **Initialization, start, stop, cancel, and cleanup.** The concrete call
    sequence for each, with its bound. `cancel()` must make a blocked operation
    return promptly and must be safe to call concurrently and repeatedly.
18. **Disconnect and recovery behavior.** What the adapter does when the link
    drops mid-session, and what `reset()` guarantees. A failed reconnect or
    renegotiation surfaces as a status; it is never concealed by a successful
    reset.
19. **`NativeFrameSource` integration.** The adapter implements or exposes the
    existing contract, allocates nothing in `read`, retains no frame, span, or
    payload pointer past return, and uses the existing status vocabulary.
20. **NRF and replay compatibility.** A session recorded from this device
    finalizes with verified accounting and replays, carrying its schema, sample
    indices, device ticks, clock-sync snapshots, and gaps intact.

### Evidence

21. **Fixtures, mocks, or vendor test facilities.** What allows the adapter's
    logic to be tested without hardware -- a vendor simulator, a recorded
    protocol capture, or a fake SDK boundary. `neurale.devices.simulation` is
    not a substitute: it simulates a device, not this vendor's SDK.
22. **Hardware-in-the-loop tests.** The executed tests against the physical
    device, and where their results live. HIL is not optional for a support
    claim; see the closing rule.
23. **Device-specific performance evidence.** Executed latency, throughput,
    allocation, cancellation, and fault measurements for this exact path,
    environment, and configuration.

### Permitted data paths

```text
A. verified direct path

vendor SDK/API
    -> concrete Devices-domain source adapter
    -> NativeFrameSource
```

```text
B. isolated vendor ingress

vendor callback / polling / SDK-owned worker
    -> bounded preallocated ownership handoff
    -> concrete Devices-domain source adapter
    -> NativeFrameSource
```

**B is the default.** Path A is available only when vendor allocation, locking,
callback threading, blocking, and latency are all *proven* bounded for this SDK
version on this platform. Anything unproven -- including anything the vendor
simply does not document -- selects B. Choosing A is a claim requiring the
evidence of items 8, 9, and 23; it is not a starting assumption.

### Distribution

Vendor-specific adapters belong to the Devices domain architecturally, wherever
they ship. An adapter needing a vendor SDK, platform-specific native libraries,
or hardware CI should normally ship as an optional device-specific distribution
(for example `neurale-device-blackrock`). Such a package can use the provider SDK
or implement PyNeurale's streaming contracts directly. It does not need, and must not
introduce, a generic `Device` superclass.

External providers use the wheel-shipped C ABI and helpers. Discovery is a
distribution mechanism; ``GenericDeviceSource`` remains an implementation of
the existing native source contract, with streaming retaining semantic ownership.

### What this standard does not impose

The list above fixes what must be **determined**, not how the work is broken
down. Different vendors have genuinely different shapes -- a TCP protocol
device, a callback-driven USB device, and a BLE peripheral do not decompose into
the same implementation tasks, and prescribing one decomposition for all of them
would be invention rather than planning.

One rule admits no exception: **do not claim physical-device support without
executed hardware-in-the-loop evidence.** An adapter that compiles, passes unit
tests against a fake SDK, and has never moved a byte from the real device is
unvalidated work, and must be described as such in every document, changelog,
and release note that mentions it.

## Acceptance criteria

Device work must preserve all of the following:

- every generic runtime contract has only ``neurale.streaming`` as owner;
- signal generation and hardware simulation have distinct owners;
- acquisition reaches the runtime through ``NativeFrameSource``;
- no common Device hierarchy, transport framework, public
  ``neurale.platform``, or global device manager exists;
- vendor dependencies remain optional and outside unrelated imports;
- devices has no generic acquisition/actuation base class and no
  display-thread ownership; rendering, window, and input adaptation belong to
  the experiment presentation domain, not devices.
