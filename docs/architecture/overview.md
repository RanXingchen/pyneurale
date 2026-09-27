# Architecture overview

PyNeurale is a layered Python, C++, CUDA, and pybind11 library for typed neural
data, offline algorithms, native streaming, acquisition, durable recording and
replay, and deterministic experiments. Public modules are split
by semantic ownership rather than collected under a shared ``core`` or
``utils`` package.

## Module ownership

| Module | Responsibility |
| --- | --- |
| ``neurale.data`` | Typed neural data, metadata, and time values. |
| ``neurale.runtime`` | Device policy, initialization, diagnostics, and process-level services. |
| ``neurale.signal`` | Signal generation and transformations. |
| ``neurale.features`` | Typed feature extraction and preprocessing. |
| ``neurale.models`` | Statistical estimators and numerical models. |
| ``neurale.decoding`` | Typed decoder lifecycle, classical continuous and classification decoders, sequence decoding, persistence, and the optional Torch dependency boundary. |
| ``neurale.io`` | MAT/CSV interchange and offline NRF v1 reading, writing, validation, diagnosis, and recovery. |
| ``neurale.streaming`` | Native schemas, buffer ownership, processor contracts, runner lifecycle, realtime configuration, observers, faults, and safety interfaces. |
| ``neurale.pipeline`` | Versioned typed plans that compile fixed built-in processing stages into one native processor chain and attach it to ``StreamRunner``. |
| ``neurale.recording`` | Native critical capture into a bounded spool, offline NRF finalization and recovery, and native replay. |
| ``neurale.sorting`` | Spike-sorting algorithms and typed offline workflows. |
| ``neurale.devices`` | Simulated acquisition and external-provider discovery, lifecycle, and ingress; an optional built-in LSL provider. |
| ``neurale.experiments`` | Task semantics, sessions, and provenance for Center-Out, WebGrid, Speech, and SSVEP. |
| ``neurale.experiments.presentation`` | Optional presentation and input for Center-Out, WebGrid, Speech, and SSVEP. |

``neurale.exceptions`` is the dependency-light public exception hierarchy.
``neurale._validation``, ``neurale._native_loader``, compiled modules whose
names begin with an underscore, and internal CMake targets are implementation
details rather than public APIs.

## Dependency direction

Dependencies point from higher-level orchestration toward narrower semantic
owners and infrastructure:

```text
experiments.presentation -> experiments
recording                -> streaming
pipeline                 -> streaming
devices                  -> streaming
decoding                 -> models

domain modules           -> data values
explicit execution       -> runtime policy and services
```

The diagram is an ownership summary, not a claim that every module directly
imports every lower box. These rules are authoritative:

- ``neurale.streaming`` does not depend on signal, features, models, sorting,
  decoding, recording formats, Zarr, or Python callbacks.
- Generic fixed linear composition belongs to streaming. Algorithm-to-stream
  adapters belong to the private ``neurale_pipeline`` C++ target, which links
  streaming, signal, features, models, and sorting in that direction.
- ``neurale.recording`` integrates streaming with NRF persistence. Streaming
  does not know about spools, NRF, replay images, or filesystem layouts, and
  ``neurale.io.nrf`` does not know about runtime scheduling.
- ``neurale.decoding`` owns decoder lifecycle and depends on
  ``neurale.models``; models do not depend on decoders.
- Device providers publish through a bounded native ingress consumed by a
  ``NativeFrameSource``. Simulated acquisition can use signal generation, but
  external providers do not add a second streaming data plane.
- The shared native experiment value target links no streaming or recording
  target. Private headless/session integration sits above experiments,
  streaming, and recording.
- Presentation depends on experiment values. Experiment state machines do not
  depend on GLFW, OpenGL, FreeType, HarfBuzz, windows, or input devices.

Catch-all public ``core``, ``common``, ``math``, ``metrics``, ``platform``, or
``utils`` packages are intentionally absent. Reusable code belongs to the
narrowest module that owns its semantics.

External device distributions can use the wheel-shipped C provider ABI or a
Python SDK in a separate acquisition process; neither requires rebuilding the
core wheel. The optional built-in LSL provider uses the same boundary. These
interfaces do not imply support for a specific vendor device. See the
{doc}`provider guide <../user_guide/device_providers>` and
{doc}`LSL guide <../user_guide/lsl>`.

## Python, native, and CUDA boundaries

The main ``neurale._native`` extension binds the runtime, streaming, recording,
signal, device, feature, model, sorting, experiment, and private realtime
pipeline-construction namespaces. Bindings translate and validate values; they
do not reimplement algorithms.

CPU algorithms use native kernels where the public module requires them.
Supported CUDA operations live in the optional ``neurale._native_cuda``
extension and are selected through the runtime device contract:

- the requested device is ``"auto"``, ``"cpu"``, or ``"cuda"``;
- ``"auto"`` resolves to CPU and does not probe for CUDA;
- ``"cuda"`` is a hard requirement and never silently falls back to CPU;
- a fitted estimator exposes its resolved persistent device through
  read-only ``device_`` where applicable;
- no boundary silently transfers data, converts dtype, or substitutes a
  Python implementation.

Experiment sessions and optional presentation bindings share ``neurale._native``
with the domain bindings. Presentation remains a build-time capability exposed
through ``neurale.experiments.presentation``; loading bindings does not initialize
graphics or create windows.

## Streaming and realtime invariants

The streaming data plane uses fixed-resource native structures prepared before
start. Its core invariants are:

- frames and payloads are borrowed from bounded pools and queues;
- steady-state callbacks do not allocate, block, log, call Python, or perform
  filesystem work;
- processor input and output borrows are callback-scoped and invalid after
  publication or callback completion;
- ``FrameEmitter`` owns output acquisition and publication;
- ``LinearProcessorChain`` is same-thread, ordered, depth-first, and bounded;
  it does not provide dynamic mutation, cycles, arbitrary graph topology, or
  per-stage worker threads;
- processor input/output schemas, fan-out limits, workspace, and pool bounds
  are declared and validated during preparation;
- flush proceeds source-to-sink, while discontinuity handling clears
  intermediate state before the control message reaches the terminal;
- faults and shutdown use bounded paths, and realtime claims require measured
  platform-specific evidence.

The streaming core owns generic transport and lifecycle contracts. The private
``neurale_pipeline`` target adapts reviewed algorithms to that core; it has no
public headers, install surface, or CMake alias. ``neurale.pipeline`` exposes
immutable Python specifications and a compiler for the built-in stages while
keeping adapter classes private. See the {doc}`pipeline guide
<../user_guide/pipelines>` for available stages.

## Stream, feature, and spike schemas

Realtime ``StreamSchema`` values contain bounded structural data. Feature
streams use observation-major payloads with shape
``(n_observations, n_features)`` and reference a stable
``FeatureSetDescriptor``. Immutable session registries own feature names,
units, source identity, algorithm identity, window length and shift, and
timestamp reference. Frames carry identifiers rather than strings or mutable
metadata, and no feature window crosses a discontinuity.

Offline spike processing uses ``neurale.data.SpikeWaveformBatch`` with waveform
shape ``(n_spikes, n_samples, n_channels)``. It preserves absolute source
indices, times, channel metadata, clock, segment, alignment, and polarity.
Native online detection uses a separate fixed-capacity sparse block with
explicit ``fault`` or ``drop_newest`` overflow policy; converting that block to
the offline batch is a control-plane allocation.

## Recording and persistence

The recording subsystem owns durable stream persistence. NRF v1 is the logical
session format, with a losslessly compressed single-file envelope, versioned manifest, Zarr v3 arrays,
transaction journal, checksums, committed extents, and explicit termination.
The normative format, schemas, and language-neutral vectors live in the
[NRF v1 specification](https://github.com/RanXingchen/pyneurale/blob/main/specifications/nrf/v1/README.md).

``neurale.recording.SessionRecorder`` is the only public live recorder. It
attaches as a critical native observer, captures accepted data into a bounded
checksummed spool, and performs NRF/Zarr finalization off the realtime path.
There is no backend selector, in-memory substitute, or Python callback fallback
for critical capture. Unsupported hosts and unavailable resources fail
explicitly.

Readers expose only committed NRF extents. Recovery diagnoses or repairs an
artifact explicitly and never silently changes committed data. Exact,
recorded-projection, and synthesized replay preserve their distinct schema,
sequence, discontinuity, timing, and omission semantics. The full lifecycle is
defined in the {doc}`recording and replay contract
<../development/native_recording_replay>`.

## Experiment and presentation boundaries

Experiment state machines own task truth: schedules, states, geometry,
containment, correctness, assistance inputs, and semantic presentation
requests. Presentation owns windows, rendering, monitor selection, pointer and
selection input, and software evidence that a request was submitted or
presented. It does not perform task transitions or redefine correctness.

The presentation time contract keeps three values distinct:

```text
semantic/request time
        != renderer submit or swap-return time
        != physical display onset
```

Swap return is software evidence only. PyNeurale does not report it as pixel or
photon onset. Physical onset requires external measurement such as a
photodiode/display trigger and enters recording as separate provenance.
The {doc}`SSVEP guide <../user_guide/ssvep>` covers the session and display
interfaces.

## Import and release boundaries

Imports expose declarations; initialization and hardware discovery occur only
at explicit execution boundaries.

- ``import neurale`` exports only ``__version__``.
- Importing runtime, models, decoding, streaming, experiments, or I/O does not
  initialize CUDA, load Torch, create a window, discover hardware, or configure
  process state.
- Optional CUDA, Torch, presentation, NRF, MAT, and hardware dependencies are
  resolved lazily by their owning modules.
- Missing optional capabilities fail through the public exception hierarchy
  when requested.

The repository-root
[support matrix](https://github.com/RanXingchen/pyneurale/blob/main/SUPPORT.md)
is the normative source for supported, tested, experimental, and unavailable
artifact profiles. The current package does not provide physical/vendor device
adapters, a scientific visualization package, user-defined realtime Python
nodes, or an installed C++ SDK. The device-provider C ABI is the external
plugin boundary; the C++ pipeline adapters remain private.
