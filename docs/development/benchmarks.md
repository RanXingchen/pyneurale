# Benchmark conventions and inventory

PyNeurale benchmarks are evidence-gathering tools, not correctness tests or
portable performance promises. Run them against an installed build, use a
Release native build for comparisons, and retain the emitted metadata with the
measurements. This normalization does not establish CI regression thresholds.
This page is the canonical inventory of current benchmark entry points.

The Release physical-chain target also runs in a separate scheduled/manual
workflow for builtin and MKL builds. That workflow archives JSON Lines and
validates their structure, but is not an ordinary pull-request correctness gate
and applies no hard latency or throughput threshold.

## Shared conventions

Python benchmarks use `benchmarks/_harness.py` and native streaming
uses an equivalent JSON Lines schema. The common conventions are:

- perform explicit warm-up outside the measured samples and report `warmups`;
- report the measured `repeats` or `repetitions` and sample count;
- retain the established minimum or median result fields and additionally
  report minimum, median, p95, p99, and maximum latency;
- report an operation-appropriate median throughput when a meaningful item
  count exists;
- record Python/native versions, compiler, build type, CPU identity, thread
  provider/count, MKL/FFT providers, and compiled CUDA information;
- record the operation dtype and a compact sample-major shape;
- report allocation counts and the tracking backend where the native streaming
  harness can measure them. General Python benchmarks mark allocations
  `not_measured`; the 100k-spike benchmark separately labels sampled process
  RSS and its derived aggregate temporary-resident delta, never treating that
  delta as a malloc-event trace;
- write flat CSV or JSON Lines records. CSV remains the default for existing
  matrix scripts so the FIR/FFT analysis scripts continue to consume them.

CUDA-aware KNN and KDE runs explicitly probe CUDA metadata. CPU-only benchmark
scripts record compiled CUDA capability but do not probe devices.

## Existing inventory

| Area | Entry point | Preserved scenarios | Output |
| --- | --- | --- | --- |
| Native streaming | CMake target `neurale_streaming_benchmark` | frame pool, SPSC single/burst/cross-CPU/near-full, identity pipelines, blocked observer, emergency inhibit | JSON Lines; latency tails, throughput, affinity and steady-state allocation counts |
| Pipeline multitaper bandpower | CMake target `neurale_pipeline_multitaper_bandpower_benchmark` | prepared 256-sample, four-channel adaptive bandpower adapter | JSON Lines; latency distribution, FFT provider/build metadata, shape, weighting, and allocation count |
| Physical native pipeline chains | CMake target `neurale_pipeline_strict_realtime_benchmark` | 4 kHz, 256-channel, 1 ms float64 frames for 60 simulated seconds; LMP [1, 5] Hz and Hilbert [70, 200] Hz SOS/IIR/FIR pre-bandpasses; unfiltered [70, 200) Hz bandpower; with and without 1 kHz resampling | JSON Lines; median/p95/p99 source-to-native-consumer latency, throughput, realtime factor, complete filter/feature geometry, and resolved/build provider metadata; allocations not measured |
| Native Valley Seeking | CMake target `neurale_sorting_valley_seeking_benchmark` | deterministic clustered observation-major float64 features with controlled label corruption | JSON Lines; whole-kernel median/p95/p99, observation throughput, realized neighbor pairs, bounded workspace bytes, convergence, shape, radius, and build metadata |
| Online spike-detector adapter | CMake target `neurale_pipeline_spike_detector_adapter_benchmark` | per-frame `process()` at 0, 1, and full spikes-per-frame (default 64-capacity, 61x32 float64 waveforms); the adapter clears the full fixed-capacity spike payload on every frame, so the empty/sparse regime isolates that capacity-proportional cost from spike-dependent append work | JSON Lines; per-regime per-frame median/p95/p99 latency, frame throughput, crossings per frame, published frames, spike-block payload bytes, geometry, and build metadata |
| Kalman decoder adapter | CMake target `neurale_pipeline_kalman_decoder_adapter_benchmark` | prepared feature frames with 1, 8, and 32 observations per frame | JSON Lines; per-frame and per-observation p50/p95/p99 latency, geometry, build/runtime metadata, and allocation tracking explicitly marked `not_measured` |
| Simulation and simulated device | CMake target `neurale_devices_simulation_benchmark` | deterministic generation cost for all five generator kinds; unpaced simulated-source read; paced acquisition timing error and inter-frame jitter; scheduled sample-loss handling; cancellation of a blocked paced read; simulated device -> native sink integration; runtime fault stop | JSON Lines; p50/p95/p99 latency, throughput and realtime factor, steady-state allocation counts, frame-pool and ingress high-water usage, process CPU and peak RSS, and the full generator/geometry/pacing/clock/fault-schedule configuration of every row |
| Simulated device to recorder | `benchmarks/devices/benchmark_simulation.py` | the simulated device -> `SessionRecorder` integration, continuous and with a scheduled sample-loss schedule; also aggregates the native half into the same artifact | JSON Lines; paced record and finalize durations, realtime factor, ingress high-water mark, recorded sample and discontinuity extents, session bytes and size ratio, scenario CPU and peak RSS |
| Experiment semantics and headless control | CMake target `neurale_experiments_benchmark` | linear velocity assistance; Center-Out machine, guidance, and four-observation decoded-frame -> headless cursor; WebGrid 8x8/16x16 hit tests, direct selection, seeded schedule, metric-v1 recomputation, and headless pointer/selection; Speech phase transition and 128-trial schedule preparation; representative semantic replay for all three paradigms | JSON Lines; per-operation p50/p95/p99, throughput, exact command/config/build/host, bounded trace capacity/drops, replay verdict/completeness, and allocation scope/backend |
| Offline provenance queries | `benchmarks/experiments/benchmark_provenance.py` | complete representative Center-Out command, WebGrid selection/metric, and Speech trial/presentation/acquisition evidence chains | JSON Lines; p50/p95/p99 query latency, query throughput, evidence count, complete verdict/gap count, exact command/build/host; allocations explicitly not measured and execution class `offline_noncritical` |
| 100k-spike workflow | `benchmarks/sorting/benchmark_workflow.py` | 100,000 spikes, 61 waveform samples, 32 channels; separate detection, waveform extraction, fitted PCA/LPP transform, Valley Seeking, tiny-cluster curation, and waveform outlier rejection stages (the latter with an optional single-large-cluster probe) | JSON Lines; per-stage median/p95/p99, spike and scalar-sample throughput, sampled peak RSS, derived temporary resident delta, retained ndarray backing bytes, dtype, window/channel geometry, per-stage provider, and build metadata |
| Native record/replay | `benchmarks/recording/benchmark_record_replay.py` plus CMake target `neurale_recording_replay_benchmark` | native critical recording; spool finalization and recoverable torn-tail repair; exact-frame, recorded-projection, and stream-frame replay under as-fast, recorded, and step pacing | JSON Lines; callback/enqueue and spool latency tails, queue high-water mark, sync/checkpoint/drain cost, throughput, CPU/RSS/I/O, size ratio, codecs/checksums, replay timing, exact commands, build/host/storage, and durability metadata |
| FIR | `benchmarks/signal/benchmark_fir.py` | native kernels and auto dispatch across samples, channels, taps and thread counts | CSV matrix plus cold/warm distribution |
| FFT | `benchmarks/signal/benchmark_fft.py` | builtin/MKL/auto kernels across lengths, channels and threads | CSV matrix plus cold/warm distribution |
| IIR/SOS | `benchmarks/signal/benchmark_iir.py` | public, native kernels and auto dispatch across direct-form orders and SOS sections | CSV on stdout or `--output` |
| DPSS | `benchmarks/signal/benchmark_dpss.py` | uncached generation and cached copies across lengths | CSV |
| Stateful processors | `benchmarks/signal/benchmark_fir_realtime.py`, `benchmarks/signal/benchmark_iir_realtime.py` | public FIR, IIR and SOS one-block processing | CSV on stdout |
| KNN | `benchmarks/models/benchmark_knn.py` | standard/large cases, CPU/CUDA selection and include-self behavior | CSV |
| KDE | `benchmarks/models/benchmark_kde.py` | existing CPU/CUDA sample/query/feature cases | CSV |

`benchmarks/signal/analyze_fir.py` and
`benchmarks/signal/analyze_fft.py` remain
optional, manually invoked dispatch-analysis tools. Their existing command-line
limits are not CI policy.

## 100k-spike characterization

The default sorting characterization is run manually against an installed
Release build:

```bash
python benchmarks/sorting/benchmark_workflow.py \
  --output build/sorting-100k.jsonl
```

Its synthetic source contains exactly `100000 x 61 x 32` float64 waveform
windows (`pre_samples=30`, `post_samples=30`) at 30 kHz. Detection and waveform
extraction use separate native entry points, so neither duration is inferred by
subtracting two fused timings. PCA and LPP rows measure full 100,000-waveform
transform through a projector fitted on the recorded `calibration_spikes`;
calibration fitting is deliberately outside the timed interval. The clustering
row uses the current native Valley Seeking implementation over all 100,000
projected observations. It is not silently sampled or replaced by another
algorithm; its current pairwise neighborhood construction may make this stage
long-running.

Every stage runs in a fresh subprocess. `peak_resident_memory_bytes` is sampled
from the process working set/RSS at the recorded interval.
`derived_temporary_resident_delta_bytes` is the sampled aggregate resident peak:

```text
peak RSS - max(stage baseline RSS, stage final RSS)
```

It is a sampled resident delta, not a malloc-event trace: it must not be
described as the largest single native allocation. Accordingly
`allocation_event_tracking` is `"not_measured"` and
`largest_single_allocation_bytes` is `null` until native allocator
instrumentation is connected. `output_memory_bytes` counts unique retained
ndarray backing buffers reachable from the stage result. These definitions
ensure the artifact reports memory evidence without presenting Python tracing
as complete native allocation coverage. A failed stage still produces a
machine-readable failure row and makes the command exit nonzero.

## Record/replay characterization

Build the manual native helper and run the Python coordinator against the same
Release build. On Linux or macOS, use this explicit single-config Unix
Makefiles build:

```bash
PYNEURALE_VERSION=$(python -c 'import importlib.metadata; print(importlib.metadata.version("pyneurale"))')
PYNEURALE_BASE_VERSION=$(python -c 'import importlib.metadata; from packaging.version import Version; print(Version(importlib.metadata.version("pyneurale")).base_version)')
PYBIND11_DIR=$(python -m pybind11 --cmakedir)
cmake -S . -B build/record-replay-benchmark \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEURALE_BUILD_BENCHMARKS=ON \
  -DNEURALE_ENABLE_MKL=AUTO \
  -DNEURALE_ENABLE_CUDA=AUTO \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DSKBUILD_PROJECT_VERSION="$PYNEURALE_BASE_VERSION" \
  -DSKBUILD_PROJECT_VERSION_FULL="$PYNEURALE_VERSION"
cmake --build build/record-replay-benchmark \
  --target neurale_recording_replay_benchmark
python benchmarks/recording/benchmark_record_replay.py \
  --native-executable build/record-replay-benchmark/cpp/benchmarks/neurale_recording_replay_benchmark \
  --output build/record-replay.jsonl \
  --work-dir build/record-replay-work
```

On Windows, the repository's preferred Visual Studio multi-config generator
uses `--config Release` and the configuration subdirectory:

```powershell
$PyNeuraleVersion = python -c 'import importlib.metadata; print(importlib.metadata.version("pyneurale"))'
$PyNeuraleBaseVersion = python -c 'import importlib.metadata; from packaging.version import Version; print(Version(importlib.metadata.version("pyneurale")).base_version)'
$Pybind11Dir = python -m pybind11 --cmakedir
cmake -S . -B build/record-replay-benchmark `
  -G "Visual Studio 18 2026" `
  -DNEURALE_BUILD_BENCHMARKS=ON `
  -DNEURALE_ENABLE_MKL=AUTO `
  -DNEURALE_ENABLE_CUDA=AUTO `
  -Dpybind11_DIR="$Pybind11Dir" `
  -DSKBUILD_PROJECT_VERSION="$PyNeuraleBaseVersion" `
  -DSKBUILD_PROJECT_VERSION_FULL="$PyNeuraleVersion"
cmake --build build/record-replay-benchmark --config Release `
  --target neurale_recording_replay_benchmark
python benchmarks/recording/benchmark_record_replay.py `
  --native-executable build/record-replay-benchmark/cpp/benchmarks/Release/neurale_recording_replay_benchmark.exe `
  --output build/record-replay.jsonl `
  --work-dir build/record-replay-work
```

The work directory must be empty; the coordinator never recursively deletes an
existing result tree. Do not mix the single-config executable path with the
multi-config build command: `--config Release` does not set
`CMAKE_BUILD_TYPE` for Ninja or Unix Makefiles. Ninja may replace Unix
Makefiles when available, with the same `CMAKE_BUILD_TYPE` and executable path.
The example's provider flags must match the installed extension; replace
`AUTO` with its explicit configuration when automatic discovery differs.
Supplying the installed full/base versions prevents a standalone CMake
configure from using its `0.0.0` fallback, which the same-build probe
deliberately rejects.
Defaults are a 1-second native-recording warm-up, one discarded replay of
each measured mode/pacing range, and three isolated repetitions over 60 seconds
of deterministic sample-major input: 4 kHz x 256-channel `int16`
neural frames uploaded every 1 ms, a 100 Hz two-dimensional `float32` cursor,
44.1 kHz single-channel `int16` audio in exact 441-sample/10 ms blocks, and a
100 Hz x 256-feature `float64` bandpower stream with a 100 ms window and 10 ms
shift. The native source groups cursor, audio, and bandpower with every tenth
neural frame. A frozen counter-based generator uses `--seed`, and the coordinator
requires the one-second neural, cursor, audio, and bandpower template SHA-256
values to be byte-identical before measurement. It also rejects a native helper
whose version, ABI, compiler, build type, math/FFT/threading providers, CUDA
build flag, or thread count differs from the installed Python extension. The
recording source is paced from a monotonic clock at one frame per millisecond;
only the explicitly named as-fast replay scenarios run without pacing.

The benchmark's default native critical store is the bounded in-memory `buffered` backend.
Its spool snapshot is materialized after the measured critical interval and is
reported as such. `mapped` remains the explicitly supported Linux-tmpfs-only
backend; it is not a portable disk backend. Finalization and recovery report
NRF Zarr v3 codecs, SHA-256 object checksums, CRC32C spool checksums, output
size, and process I/O. Replay covers every supported image and pacing mode;
all replay timings use the same explicitly reported two-second range by
default, configurable with `--replay-range-seconds`. Keeping the range common
prevents image-build cost or item count from confounding comparisons among
as-fast, recorded, and step pacing; recording, finalization, and recovery still
cover the complete 60-second session. Every replay-image mode is built into a
fresh output once per requested repetition. Aggregate rows report elapsed-time
distributions and the median of every throughput field present in all raw rows.
Native spool timing buffers use a prepare-time upper bound derived from frames
and checkpoint interval; dropped timing counts are reported and make the native
recording stage fail rather than silently truncating latency evidence.

This is scheduled/manual characterization, not an ordinary PR correctness
gate. It sets no threshold and supports claims only for the recorded command,
build, host, OS, storage, queue, payload, codec, checksum, and durability
configuration.

## Simulation and device characterization

Simulation characterization covers deterministic signal generation and a
simulated acquisition device. It has two halves and one artifact. The native
half owns everything that must be timed without a Python callback or the GIL on
it; the Python half owns the simulated device -> `SessionRecorder` integration,
because `SessionRecorder` is a Python control-plane object. It deliberately
does not re-measure signal filters, features and decoders, or the record/replay
workflow.

Build the native half in a Release configuration matching the installed
extension, then run the coordinator:

```bash
PYBIND11_DIR=$(python -m pybind11 --cmakedir)
cmake -S . -B build/device-benchmark \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEURALE_BUILD_BENCHMARKS=ON \
  -DNEURALE_ENABLE_MKL=AUTO \
  -DNEURALE_ENABLE_CUDA=AUTO \
  -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build build/device-benchmark --target neurale_devices_simulation_benchmark

python benchmarks/devices/benchmark_simulation.py \
  --native-executable build/device-benchmark/cpp/benchmarks/neurale_devices_simulation_benchmark \
  --output build/device-simulation.jsonl \
  --work-dir build/device-simulation-work
```

The native target also runs standalone and takes the same geometry flags:

```bash
build/device-benchmark/cpp/benchmarks/neurale_devices_simulation_benchmark \
  --channels 256 --fs 30000 --frame-samples 30 \
  --warmup-frames 512 --pipeline-frames 20000 \
  --generation-iterations 20000 --read-iterations 20000 \
  --paced-seconds 2 --cancel-repetitions 32 \
  --loss-period-frames 64 --loss-samples 4
```

An unknown flag is rejected rather than ignored. The default geometry is an
NSP-class amplifier: 256 channels of sample-major `int16` at 30 kHz in
30-sample (1 ms) frames -- 15,360 payload bytes per frame, 15.36 MB/s -- from a
two-tone-per-channel generator, with device ticks enabled, a 250 ns clock
offset, 100 ppm drift, and 50 ns clock-sync uncertainty. The payload holds raw
ADC counts, so the generator's amplitudes are in counts and the schema declares
`dimensionless` rather than volts: nothing here supplies a calibration, and
labelling uncalibrated integers as volts would be a false claim. Every row
repeats its own generator kind, channel count, rate, dtype, frame size, pacing
mode, clock parameters, fault schedule, and the exact command, so no row
depends on this page to be read.

What each case means:

- `signal_generation` measures `SignalGenerator::generate` for all five
  generator kinds at the frame geometry, with throughput in channel-samples per
  second. This is the cost the simulated source pays per frame, isolated from
  framing. Its rows report `dtype: "float64"` and the matching payload size on
  purpose: the generator is dtype-agnostic and always writes float64 into a
  caller buffer, and the device converts afterwards.
- `simulated_source_read` is one unpaced `read()` into runtime-owned frame
  storage. Its `realtime_factor` row expresses the same run as a multiple of
  the acquisition timeline it simulates. At an integer dtype this read is
  strictly more than the generation cost above, because it also runs the
  quantization pass that turns generated values into saturating 16-bit counts;
  the difference between the two rows is that pass.
- `simulated_source_paced` reports `pacing_absolute_error`, the deviation of
  each completed read from its intended deadline, and `inter_frame_jitter`, the
  deviation of each observed interval from the intended one. Both take the
  intended deadline from the frame's own `ClockSyncSnapshot`
  (`host_time_reference_ns - clock_offset_ns` is exactly the deadline the source
  paced to), not from a timeline the benchmark reconstructs. That matters twice:
  pacing follows the *drifted* device tick rate, so a nominal-rate timeline
  would report the configured drift as error; and the acquisition epoch is
  latched at the top of the first read, before that frame is generated, so
  anchoring to a host timestamp taken after the first read returns biases every
  deadline late by one frame's whole generation cost -- tens of microseconds at
  the default geometry, the same order as the metric. Frame 0 is not a measured
  sample: its deadline is the epoch latched inside that very read, so its error
  is by construction that read's own cost and can never be early. Read the error
  as "how late the frame was available against when it was due": the deadline is
  the acquisition instant and the frame is generated after the wait, so the
  distribution's floor is one frame's generation cost plus wake-up latency, not
  scheduler jitter alone. `inter_frame_jitter` is the interval-to-interval
  figure and does not carry that floor. The reported
  `throughput_per_second` is derived from the observed deadlines, so it shows
  the drifted rate the source actually paced to.
- `simulated_source_discontinuity` separates data reads under a loss schedule
  from `gap_publication`, the extra read that carries a gap and no frame. The
  handling cost is that whole call, not a slower data read.
- `simulated_source_cancel` measures from `cancel()` to a blocked paced `read()`
  returning `stopped`, using a one-frame-per-second acquisition so the interval
  is cancellation and never a read that was about to return anyway.
- `simulated_device_to_native_sink` is the first small integration:
  simulated device -> runtime -> passthrough processor -> native consumer,
  reporting acquisition-to-consumer latency, throughput, ingress high-water
  mark, and steady-state allocations armed after the warm-up frames.
  `frame_pool_capacity` is the runtime's own `required_buffer_count()`, which
  reserves one buffer per ingress and critical-edge queue slot as well as the
  stage-owned leases; `frame_pool_payload_bytes` is that count times the buffer
  size and is payload only -- the pool additionally allocates block-header
  storage and its slot registry, and `peak_resident_memory_bytes` remains the
  process-level figure.
- `simulated_device_fault_stop` measures from the acquisition read that
  reported `source_failure` to the runner having stopped with a latched primary
  fault.

Allocation reporting is per row. Cases that create a measurement thread
(`simulated_source_cancel`) or run one unrepeatable event
(`simulated_device_fault_stop`) report `allocation_tracking: "not_measured"`
rather than a zero they did not earn.

The `SessionRecorder` half is **paced** at the acquisition rate. That is what
makes it an integration measurement: a recorder fed by an unpaced simulated
source is not slow, it is simply overrun, and the question is whether the
recording chain keeps up with a device and at what cost. It reports
`record_session` (the acquisition and critical-recording interval) separately
from `finalize_session` (the control-plane interval that turns the spool into a
canonical NRF session, after acquisition has ended). `process_cpu_seconds` is
scenario-wide -- all runs, record plus finalize plus read-back -- and can exceed
the scenario's wall time because finalization is threaded.

The recorder's bounded spool uses disk space, not locked RAM. Its default
capacity includes frame payloads and bookkeeping margin; an explicit
`--spool-capacity-bytes` limit must hold the complete session. The two halves
accept independent geometry: `--record-channels`, `--record-fs`,
`--record-frame-samples`, and `--record-frames` override the native geometry
for the recorder half only. A session that does not fit fails the run; the
script reports capacity, frame count, and payload size rather than treating
`StreamStatus.STOPPED` as a measurement.

## Experiment characterization

The native experiment benchmark is a registered `benchmark;performance` CTest
target and is also discovered automatically by `tools/run_benchmarks.ps1`.
Build and run it directly when only experiment evidence is required:

```powershell
cmake --build build/benchmarks --config Release `
  --target neurale_experiments_benchmark
build/benchmarks/cpp/benchmarks/Release/neurale_experiments_benchmark.exe `
  --warmups 256 --repetitions 4096 --metric-records 4096 `
  --replay-inputs 1024 --speech-trials 128 `
  > build/experiments.jsonl

.venv/Scripts/python.exe benchmarks/experiments/benchmark_provenance.py `
  --warmups 128 --repeats 4096 `
  --output build/experiment-provenance.jsonl
```

The public-value and steady-state controller rows require zero allocations seen
by the selected tracker. Windows rows state `operator_new`, which is not proof
about C or third-party allocators. Linux GNU/Clang builds apply the repository's
existing `malloc`/`calloc`/`realloc` link wrappers to this target. The Speech
large-schedule row is intentionally `allocation_scope=prepare`: its arrays and
bounded trace queue are allocated before scheduling begins.

WebGrid's largest case is the current fixed-capacity maximum, 16x16 (256 cells),
and its metric row recomputes metric v1 over 4096 raw selections. Speech's
schedule-preparation case uses the current maximum 128 trials. The headless
Speech scheduler has no real paced mode, so this benchmark reports neither
scheduler jitter nor display latency. Orthogonal impedance is omitted because
the experiment benchmark makes no realtime claim for it; only the
linear-assistance path is measured.

The three replay rows execute representative semantic input timelines. Their
expected-output streams are deliberately absent, so the required result is
`incomplete` with `stream_absent`, not `match`; the benchmark measures
regeneration without converting partial evidence into a correctness claim.
The provenance script measures complete offline evidence joins and records
`allocation_metric=not_measured`. Neither entry point defines a hard CI
threshold or a portable realtime budget.

## Experiment-presentation characterization

Build presentation explicitly, then run the registered performance benchmark:

```powershell
cmake -S . -B build/presentation `
  -DNEURALE_BUILD_CPP_TESTS=ON `
  -DNEURALE_BUILD_BENCHMARKS=ON `
  -DNEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON
cmake --build build/presentation --config Release `
  --target neurale_experiment_presentation_benchmark
build/presentation/cpp/benchmarks/Release/neurale_experiment_presentation_benchmark.exe `
  --font C:/Windows/Fonts/arial.ttf `
  --cjk-font C:/Windows/Fonts/msyh.ttc `
  --warmups 64 --repetitions 512 --long-session-frames 100000 `
  > build/experiment-presentation.jsonl
```

The repository runner can discover the same conditional CTest entry with
`tools/run_benchmarks.ps1 -ExperimentPresentation ON`; leave the option at its
default `OFF` for a headless benchmark build. Font paths may be supplied through
the executable arguments shown above or the documented presentation test-font
environment variables.

The executable requires fonts covering its prepared ASCII and Chinese catalog;
it never substitutes fonts. Its JSONL records the exact command and
display/GPU/build/font configuration. Operation rows report p50/p95/p99,
allocation count, CPU and RSS. Speech preparation reports shaping and glyph
cache construction separately, while timed text rows exercise only the prepared
cache. Swap-return is software evidence, not physical display onset.

### Windows presentation characterization (2026-08-23)

The 2026-08-23 Windows characterization used Release/MSVC on an Intel Core
Ultra 9 285K and NVIDIA GeForce RTX 5090, with the driver OpenGL string
`3.3.0 NVIDIA 610.47`, a hidden 800x600 window and swap interval 0. GLFW
reported 32 Hz for the selected monitor; that metadata is recorded as reported
and is not an independently measured physical refresh rate. The exact command
was:

```powershell
build\presentation\cpp\benchmarks\Release\neurale_experiment_presentation_benchmark.exe `
  --font C:\Windows\Fonts\arial.ttf `
  --cjk-font C:\Windows\Fonts\msyh.ttc `
  --warmups 64 --repetitions 512 --long-session-frames 100000
```

The measured software times in microseconds were:

| Scenario | p50 | p95 | p99 |
|---|---:|---:|---:|
| Center-Out snapshot update and render | 241.45 | 383.66 | 569.48 |
| WebGrid 12x12 update and render | 448.80 | 505.01 | 592.50 |
| WebGrid pointer mapping and input adaptation | 0.00 | 0.10 | 0.10 |
| Speech prepared ASCII text | 187.55 | 304.25 | 418.00 |
| Speech prepared Chinese text | 185.15 | 301.50 | 493.91 |
| Speech prepared mixed text | 203.10 | 323.25 | 458.35 |
| Speech prepared representative prompt | 215.85 | 388.13 | 656.78 |
| Speech request to swap-return software evidence | 190.75 | 306.14 | 460.78 |

Preparing the four-text catalog took 1.227 ms in total, including 0.143 ms of
HarfBuzz shaping and 0.195 ms of glyph-cache construction. The 100,000-presentation
prepared-cache run took 18.468 s, recorded zero tracked C++ allocations, zero
unexpected glyph-cache misses, current RSS from 73,269,248 to 76,472,320 bytes,
and a 95,563,776-byte process peak RSS. These allocation counts cover the benchmark's
C++ allocation tracker, not every opaque allocation a graphics driver may make.
The raw JSON Lines artifact for this run is not tracked in the repository;
these figures are a retained summary, not a current-build performance claim.

## Structural smoke runs

For a quick check that representative benchmark entry points produce valid
output:

```bash
python benchmarks/signal/benchmark_fir.py --samples 32 --channels 1 --taps 5 \
  --threads 1 --max-repeats 3 --output build/fir-smoke.csv
python benchmarks/signal/benchmark_fft.py --lengths 32 --channels 1 \
  --threads 1 --max-repeats 3 --output build/fft-smoke.csv
python benchmarks/signal/benchmark_iir.py --warmups 0 --repeats 1 \
  --output build/iir-smoke.csv
```

These commands validate result production only. They are not statistically
meaningful performance runs.
