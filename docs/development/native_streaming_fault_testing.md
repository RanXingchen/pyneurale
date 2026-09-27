# Native streaming fault and soak testing

The native streaming test suite separates deterministic fault injection from
long-running stability checks. Fault components are native C++ implementations;
they do not invoke Python or allocate from the runtime processing path.

## Deterministic fault matrix

`neurale_streaming_fault_injection_test` injects source stalls, malformed
headers, sample gaps, session changes, unreleased processor output leases,
ingress overruns, processor deadline and fatal failures, actuator timeout and
write failures, blocked observers, delayed watchdog execution, and shutdown
timeouts.

The test verifies that the first critical fault remains the primary fault,
secondary history never exceeds its configured capacity, safety becomes
inhibited, actuator submission stops after a critical fault, observer stalls do
not stop the actuator, and all recoverable frame and discontinuity leases return
to their pools.

Run only the deterministic matrix with:

```bash
ctest --test-dir build/native -C Release \
  -L fault-injection --output-on-failure
```

## Soak modes

`neurale_streaming_soak_test` repeatedly prepares, starts, stops, and resets one
runtime instance. It also drives the SPSC ring and one-slot `FramePool` through
many wraps, verifies every message and slot generation, checks the explicit
frame-sequence overflow contract, and samples resident memory after warm-up.

The environment controls are:

- `NEURALE_STREAMING_SOAK_MODE=ci|nightly|soak`
- `NEURALE_STREAMING_SOAK_CYCLES`
- `NEURALE_STREAMING_SOAK_RING_ITERATIONS`
- `NEURALE_STREAMING_SOAK_POOL_ITERATIONS`
- `NEURALE_STREAMING_SOAK_SECONDS`

`ci` is the default short mode. `nightly` raises all iteration counts. `soak`
runs for at least one hour by default; the seconds override can extend or
shorten it for a local dedicated-machine run. The executable emits one JSON
summary suitable for collection by CI or a soak harness.

```bash
NEURALE_STREAMING_SOAK_MODE=nightly \
ctest --test-dir build/native -C Release -L soak --output-on-failure
```

Resident-set sampling is a coarse sustained-growth guard, not a leak detector.
Sanitizer runs provide the ownership and undefined-behavior acceptance path.

## Sanitizer builds

Use separate build directories for ThreadSanitizer and memory sanitizers.
Sanitizer options require `NEURALE_BUILD_CPP_TESTS=ON` and consistently
instrument the native libraries, internal pipeline adapters, bindings, and
native test executables. These commands disable MKL and CUDA so sanitizer
results are not conflated with provider validation.

GCC or Clang memory sanitizer build:

```bash
cmake -S . -B build/asan \
  -DNEURALE_BUILD_CPP_TESTS=ON \
  -DNEURALE_ENABLE_MKL=OFF \
  -DNEURALE_ENABLE_CUDA=OFF \
  -DNEURALE_ENABLE_ASAN=ON \
  -DNEURALE_ENABLE_UBSAN=ON \
  -DNEURALE_ENABLE_LSAN=ON
cmake --build build/asan --config RelWithDebInfo
ctest --test-dir build/asan -C RelWithDebInfo \
  -R '^(neurale_streaming_|neurale_pipeline_)' --output-on-failure
```

GCC or Clang race detector build:

```bash
cmake -S . -B build/tsan \
  -DNEURALE_BUILD_CPP_TESTS=ON \
  -DNEURALE_ENABLE_MKL=OFF \
  -DNEURALE_ENABLE_CUDA=OFF \
  -DNEURALE_ENABLE_TSAN=ON
cmake --build build/tsan --config RelWithDebInfo
ctest --test-dir build/tsan -C RelWithDebInfo \
  -R '^(neurale_streaming_|neurale_pipeline_)' --output-on-failure
```

MSVC supports the AddressSanitizer option only:

```powershell
cmake -S . -B build/asan-msvc `
  -DNEURALE_BUILD_CPP_TESTS=ON `
  -DNEURALE_ENABLE_MKL=OFF `
  -DNEURALE_ENABLE_CUDA=OFF `
  -DNEURALE_ENABLE_ASAN=ON
cmake --build build/asan-msvc --config RelWithDebInfo
ctest --test-dir build/asan-msvc -C RelWithDebInfo `
  -R '^(neurale_streaming_|neurale_pipeline_)' --output-on-failure
```

Run absolute realtime and long-duration acceptance on the intended deployment
machine. CI sanitizer runs establish memory, ownership, and race safety but do
not establish hard realtime guarantees.
