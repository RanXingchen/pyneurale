# Runtime configuration

`neurale.runtime` stores execution preferences and reports the capabilities of
the current installation. Domain modules select implementations internally from
the active runtime policy, input constraints, and available kernels.

The requested/resolved-device and package-wide lazy-import boundaries are
summarized in the {doc}`architecture overview <../architecture/overview>`.

Importing the module does not load the native extension, initialize CUDA,
change thread settings, configure logging, or seed random generators.

## Configure the process

```python
from neurale.runtime import RuntimeConfig, initialize_runtime

info = initialize_runtime(
    RuntimeConfig(
        device="auto",
        num_threads=4,
        random_seed=42,
        deterministic=True,
        log_level="INFO",
    )
)
```

`device` is the requested execution device and accepts only `auto`, `cpu`, or
`cuda`. In this release, `auto` resolves deterministically to `cpu` without
probing CUDA. An explicit `cpu` request also avoids CUDA discovery. An explicit
`cuda` request is a hard requirement: initialization or an operation that
requires resolution raises `DeviceUnavailableError` when CUDA is not compiled,
the runtime/device is unavailable, the ordinal is invalid, or the operation has
no CUDA implementation. There is no silent CPU fallback.

The requested value remains in `RuntimeInfo.config.device`; explicit
initialization reports the actual `cpu` or `cuda` result in
`RuntimeInfo.resolved_device`. Operation-specific dispatch remains inside the
owning domain module. The CPU math library, currently MKL for native
builds, is a build property reported by diagnostics.

CUDA support is implemented by the C++/CUDA native extension. PyNeurale does
not use CuPy to detect or execute CUDA operations.

CUDA is currently an experimental source-build/hardware-validation profile,
not a published release wheel or runtime bundle. Builtin CPU and oneMKL are the
supported CPU profiles. The exact tested platforms and toolchains are frozen
by the repository-root
[release/support matrix](https://github.com/RanXingchen/pyneurale/blob/main/SUPPORT.md).

`random_seed` seeds Python, NumPy, and PyTorch when PyTorch is installed.
`deterministic=True` additionally enables PyTorch deterministic algorithms and
configures its cuBLAS workspace before PyTorch is imported. It may reduce
performance and still cannot guarantee identical results across releases or
platforms. `PYTHONHASHSEED` must be set before launching Python and is not
modified by runtime initialization.

## Override one task

```python
from neurale.runtime import runtime_context

with runtime_context(device="cpu", num_threads=1):
    # Operations in this context observe the local preferences.
    ...
```

Context-local configuration is isolated with Python context variables and is
safe to use in nested and asynchronous tasks. CPU math thread limits are still
process-wide and must be changed with `thread_limit` only when that global
effect is acceptable.

## Inspect native and CUDA capabilities

```python
from neurale.runtime import show_runtime_info

print(show_runtime_info())
print(show_runtime_info(probe_cuda=True))
```

The first call loads the native extension only to read build metadata. The
second call may initialize the CUDA runtime and enumerate devices.

The runtime-diagnostics subset of the native extension contract uses ABI
version 1 and must provide:

```text
build_info() -> mapping
cpu_info() -> mapping                    # optional
threading_info() -> mapping              # optional
cuda.info() -> mapping                   # optional CUDA build
```

`build_info()` reports at least `abi_version`; native builds should additionally
report the compiler, build type, MKL/FFT backend, OpenMP status, CUDA compile
status, and CUDA toolkit version.

The same native extension also contains the current signal, features, models,
and streaming namespaces. Runtime imports do not load those namespaces eagerly
through Python. CUDA queries occur only through `cuda.info()` and related
explicit calls; `build_info()` reports compile metadata without initializing
the CUDA runtime.

## Environment variables

The process-level defaults can be set before importing `neurale.runtime`:

```text
NEURALE_DEVICE
NEURALE_CUDA_DEVICE
NEURALE_ALLOW_CUDA
NEURALE_NUM_THREADS
NEURALE_RANDOM_SEED
NEURALE_DETERMINISTIC
NEURALE_LOG_LEVEL
NEURALE_STRICT
```

Explicit Python configuration takes precedence over these defaults.

`NEURALE_REQUIRE_CUDA` is not a supported variable. Use
`NEURALE_DEVICE=cuda`. `NEURALE_ALLOW_CUDA=false` may prohibit CUDA globally,
but configuration rejects combining that prohibition with an explicit
`device="cuda"` request rather than silently changing devices.

## Logging integration

Before explicit configuration, the `neurale` logger has no package-owned
handlers and propagates records to application logging. Calling
`configure_logging()` installs package-owned handlers and disables propagation
to avoid duplicate records.

File logging appends by default. Set `file_mode="w"` for replacement, or use
`max_bytes` and `backup_count` for rotating files:

```python
from neurale.runtime import configure_logging

configure_logging(
    log_file="logs/neurale.log",
    max_bytes=10_000_000,
    backup_count=3,
)
```

Runtime and domain dispatch decisions are emitted at `DEBUG`; recoverable
native availability issues are emitted at `INFO`. Per-frame signal processing
does not log.
