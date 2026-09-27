# Native streaming benchmarks

Native streaming benchmarks are a separate build target. They do not add
measurement callbacks, sample histories, logging, or formatting to the
production runtime. The benchmark executable preallocates its sample storage,
runs a warm-up phase, measures into fixed slots, and computes percentiles only
after the realtime interval has ended.

## Build and run

On a Unix single-config build, enable the targets explicitly:

```bash
cmake -S . -B build/benchmark \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEURALE_BUILD_CPP_TESTS=ON \
  -DNEURALE_BUILD_BENCHMARKS=ON \
  -DNEURALE_ENABLE_MKL=OFF \
  -DNEURALE_ENABLE_CUDA=OFF \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/benchmark --config Release \
  --target neurale_streaming_benchmark \
           neurale_pipeline_multitaper_bandpower_benchmark \
           neurale_pipeline_strict_realtime_benchmark \
           neurale_pipeline_kalman_decoder_adapter_benchmark
```

CTest invokes the registered benchmark commands, including the physical-chain
benchmark's full default 1-second warm-up, 60-second measurement, and three
runs. Performance benchmarks are not ordinary PR-CI gates and do not assert
absolute timing:

```bash
ctest --test-dir build/benchmark -C Release \
  -L benchmark --output-on-failure
```

For a longer machine-readable run, redirect the JSON Lines output:

```bash
build/benchmark/cpp/benchmarks/neurale_streaming_benchmark \
  --iterations 1000000 \
  --pipeline-iterations 10000 \
  --producer-cpu 2 --consumer-cpu 3 \
  --deadline-ns 1000000 > native-streaming.jsonl
```

The representative prepared multitaper-bandpower adapter case uses the same
JSON Lines and allocation conventions:

```bash
build/benchmark/cpp/benchmarks/neurale_pipeline_multitaper_bandpower_benchmark \
  --warmups 100 --repetitions 1000 > bandpower.jsonl

build/benchmark/cpp/benchmarks/neurale_pipeline_strict_realtime_benchmark \
  --warmup-seconds 1 --duration-seconds 60 --runs 3 \
  > physical-native-chains.jsonl

build/benchmark/cpp/benchmarks/neurale_pipeline_kalman_decoder_adapter_benchmark \
  --features 128 --selected 96 --state 3 --max-observations 32 \
  > kalman-decoder.jsonl
```

The Kalman decoder adapter benchmark reports one record per block size (1, 8,
and the schema maximum), each with `min`, `median`, `p95`, `p99`, `max`, the
median cost per decoded observation, and the environment that produced them:
version, compiler, build type, CPU math backend, threading backend and thread
count, host CPU model, and logical core count. It measures latency only and
says so -- every record carries `"allocation_tracking":"not_measured"`, because
the zero-allocation evidence for this adapter comes from
`neurale_pipeline_kalman_decoder_adapter_test`, which counts `operator new`
across a steady-state `process`/`handle_discontinuity`/`flush`/`reset` cycle.

The per-observation cost is dominated by the factorization of the
observation-dimension innovation covariance (cubic in the number of selected
features), with the state-dimension covariance propagation a smaller term when
the observation dimension is much larger than the state dimension: on one
developer host (Release, MKL build, Intel Core Ultra 9 285K) 96 selected
features driving 3 state dimensions cost about 60 us per decoded observation,
and 24 selected features about 3.5 us. Block size changes the per-frame number
and leaves the per-observation number alone, which is the expected shape: the
recursion is per observation and nothing is batched across them. The
Kalman adapter benchmark does not accept `--deadline-ns`; it measures
per-frame latency only.

For the generic streaming benchmark, `--deadline-ns` is optional. A zero value reports no deadline misses. Absolute
deadline acceptance belongs on a dedicated machine with the intended kernel,
BIOS, CPU isolation, affinity, scheduler, and device configuration. Ordinary
pull-request CI runs correctness and allocation contracts, not performance.
The separate scheduled/manual `Physical-chain performance benchmarks` workflow
collects builtin and MKL characterization artifacts without a hard threshold;
hosted-runner timings are descriptive rather than platform acceptance.

## Coverage

The executable reports:

- `FramePool` acquire, zero-copy read-only publication, and release for several
  payload sizes and one or four synchronous readers;
- SPSC single-message latency and throughput, burst behavior, an attempted
  producer/consumer split across separately configurable CPUs, and a queue
  held one slot below capacity;
- synthetic source to identity processor to actuator pipelines for several
  channel and block shapes;
- the same critical pipeline while a noncritical observer remains blocked;
- actuator-failure to `SafetyController::inhibit()` invocation latency.

Pipeline timestamps have the following meanings:

- `acquisition_interval`: interval between successful source reads;
- `ingress_dwell`: `processor start - FrameHeader.host_received_ns`;
- `process_execution`: duration of `NativeFrameProcessor::process()` including
  publication to the critical edge;
- `source_to_output`: critical-edge enqueue time minus source receive time;
- `source_to_actuator`: actuator write entry minus source receive time;
- `emergency_inhibit_latency`: actuator failure detection to safety inhibit
  entry.

Each timing metric contains `min`, `median`, `p99`, `p99_9`, `p99_99`, `max`,
and `deadline_misses`. SPSC cases also report messages per second. CPU pinning
is diagnostic: `affinity_requested` and `affinity_applied` distinguish a real
cross-CPU run from an unsupported or restricted environment.

Every JSON Lines record also contains the benchmark schema version, warm-up
and repetition counts, dtype/shape information, native version, compiler,
build type, host CPU, thread provider/count, CPU math and FFT providers, and
compiled CUDA metadata. CUDA hardware is not probed by this CPU streaming
benchmark.

## Physical native-chain performance profile

`neurale_pipeline_strict_realtime_benchmark` is a pure C++ performance
executable. Its critical path contains a synthetic `NativeFrameSource`, a
fixed `LinearProcessorChain`, and a `NativeFrameConsumer`; it neither links the
Python extension nor installs a Python callback. The default input is float64,
sample-major, 4 kHz, 256 channels, four samples per 1 ms frame, and 60,000
frames per run. The benchmark runs three times unless overridden.

Every chain starts with a common average reference across all 256 channels,
followed by three second-order SOS notches centered at 100, 150, and 200 Hz
with a +/-1 Hz half-width. The reference stage is unconditional rather than a
case axis, so the case count is unchanged at 14 -- but latencies from these
records are not comparable with JSONL produced before the stage existed. The
record's `schema_version` is 3 and its `reference` field names the stage.

The reference stage here is always the mean. What the median costs is not
answerable from this benchmark and is measured separately by
`neurale_pipeline_spatial_reference_adapter_benchmark`, which sweeps
mean against median over 2/16/64/128/256 channels and both whole-set and
every-other-channel reference selections, and reports per-frame min, median,
p95, p99 and max. Read its p99 rather than its max: the benchmark runs on an
ordinary thread, and both statistics show multi-microsecond max outliers that
are scheduler preemption, not algorithm jitter.

LMP uses a [1, 5] Hz pre-bandpass and Hilbert uses [70, 200] Hz; each is
measured with second-order Butterworth SOS,
second-order Butterworth direct-form IIR, and order-24/25-tap FIR variants.
Bandpower has no pre-bandpass and integrates [70, 200) Hz. Each of those seven
paths is measured directly and with 4 kHz to 1 kHz resampling, for 14
non-duplicated scenarios.

Feature windows are 100 ms with a 10 ms shift. Bandpower uses NW=4, seven
tapers, adaptive weighting, and the next power-of-two FFT length: 512 without
resampling and 128 after resampling. The resampler uses an explicitly reported
order-64, 450 Hz low-pass prototype. JSON Lines records include all design and
effective orders, coefficient/section counts, resolved feature FFT provider,
build-level provider metadata, median (p50), p95/p99 latency, throughput, and realtime
factor. Allocation is explicitly `not_measured` in this performance target.

## Allocation boundary

Allocation acceptance belongs to
`neurale_pipeline_strict_realtime_contract_test`, not to a benchmark. Its
thread creation, pool construction, and warm-up occur outside the tracked
interval. Any tracked allocation during the steady-state interval fails the
allocation test.

The allocation backend is included in every record:

- Windows tracks global `operator new` variants used by the C++ runtime. This
  is partial evidence only;
- GNU/Clang Linux additionally wraps `malloc`, `calloc`, and `realloc` at link
  time.

The GNU/Clang Linux `allocation-full` CTest label is the repository's complete
application-level allocation acceptance path for direct C allocation calls.
The current hook does not claim visibility into every allocation made wholly
inside a separately loaded provider library. In particular, Windows
`operator new` results are not evidence about MKL-internal allocation. Complete
provider-internal allocation claims require provider-supported memory statistics
or allocation hooks; the repository does not currently make that claim.

The observer-isolation verification compares source-to-actuator median latency.
It permits normal scheduler noise and bounded fan-out overhead, but fails if a
blocked observer exceeds three times the baseline plus 50 microseconds. No
absolute latency threshold is imposed by CI.
