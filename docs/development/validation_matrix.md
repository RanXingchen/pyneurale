# Development validation

This page lists durable repository validation entry points. Run commands from
the repository root with the root ``.venv`` activated unless the command is
explicitly validating a clean wheel installation.

The repository-root
[support information](https://github.com/RanXingchen/pyneurale/blob/main/SUPPORT.md)
summarizes supported platforms and capabilities. This page provides the
corresponding validation commands and evidence.

## Run CI locally

On Windows with PowerShell 7.2 or newer, use the local runner to execute the
workflow jobs against isolated working-tree snapshots. Linux jobs run in WSL
Ubuntu 24.04 on its ext4 filesystem; the original worktree and staging area
are not changed.

```powershell
.\tools\run_local_ci.ps1 -List
.\tools\run_local_ci.ps1 -Check
.\tools\run_local_ci.ps1
```

The default run includes every mapped job and matrix entry and can take hours.
Use `-Only 'pr-smoke.yml:*'` or another pattern from `-List` for a smaller
run. The runner needs the Python versions, compilers, system libraries, and
display tools required by each job already installed; it does not change
system packages. Results and logs are written under `temp/ci/`. PASS, FAIL,
and BLOCKED are distinct, and either FAIL or BLOCKED gives a nonzero exit code.
It mirrors local build and test steps, not GitHub-hosted setup, artifact upload,
or runner identity; the workflows remain the source of truth.
The controlled-history benchmark remains BLOCKED without its administered
self-hosted runner; CUDA requires a visible GPU and `nvcc` in WSL.

## Build profiles

Python builds use scikit-build-core and CMake. C++20, Python development
headers, and pybind11's CMake package are required.

| Profile | CMake definitions |
| --- | --- |
| Core CPU | ``NEURALE_ENABLE_MKL=OFF``, ``NEURALE_ENABLE_CUDA=OFF``, ``NEURALE_ENABLE_EXPERIMENT_PRESENTATION=OFF`` |
| oneMKL CPU | ``NEURALE_ENABLE_MKL=ON``, ``NEURALE_ENABLE_CUDA=OFF``, ``NEURALE_ENABLE_EXPERIMENT_PRESENTATION=OFF`` |
| CUDA | ``NEURALE_ENABLE_MKL=OFF``, ``NEURALE_ENABLE_CUDA=ON``, ``NEURALE_ENABLE_EXPERIMENT_PRESENTATION=OFF`` |
| Presentation | ``NEURALE_ENABLE_MKL=OFF``, ``NEURALE_ENABLE_CUDA=OFF``, ``NEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON`` |

``ON`` is a hard requirement: configuration fails if the requested toolkit or
dependency is unavailable. ``AUTO`` is valid for MKL and CUDA during local
development, but release artifacts use explicit values. Presentation accepts
only ``ON`` or ``OFF``.

Create a core editable install with tests:

```bash
python -m pip install -e ".[test]" \
  --config-settings=cmake.define.NEURALE_ENABLE_MKL=OFF \
  --config-settings=cmake.define.NEURALE_ENABLE_CUDA=OFF \
  --config-settings=cmake.define.NEURALE_ENABLE_EXPERIMENT_PRESENTATION=OFF
```

For a direct native build:

```bash
python -m pip install pybind11
cmake -S . -B build/native \
  -DNEURALE_BUILD_CPP_TESTS=ON \
  -DNEURALE_ENABLE_MKL=OFF \
  -DNEURALE_ENABLE_CUDA=OFF \
  -DNEURALE_ENABLE_EXPERIMENT_PRESENTATION=OFF \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/native --config Release
```

Use a fresh build directory when changing feature profiles. CMake cache values
set with ``-D`` remain authoritative over same-named environment variables.

## Python tests

The default suite validates an installed native extension before collection
and excludes tests marked ``slow``, ``gpu``, ``interactive``, ``hardware``, and
``python_only`` according to ``pytest.ini``:

```bash
python -m pytest
```

Use focused directories before the full suite:

```bash
python -m pytest tests/unit/runtime
python -m pytest tests/unit/signal tests/unit/features tests/unit/models
python -m pytest tests/unit/streaming tests/unit/recording
python -m pytest tests/unit/experiments
python -m pytest tests/specification
```

The source-only import boundary is a separate mode. It intentionally disables
the repository defaults and selects only tests marked ``python_only``:

```bash
python -m pytest -o addopts="" -m python_only
```

Do not interpret a source-only pass as validation of native bindings, ABI,
CUDA, presentation, recording, or streaming behavior.

## Native tests

Run the complete registered CTest suite for the configured build:

```bash
ctest --test-dir build/native -C Release --no-tests=error --output-on-failure
```

Labels provide durable focused selections:

```bash
ctest --test-dir build/native -C Release \
  -L strict-realtime --no-tests=error --output-on-failure
ctest --test-dir build/native -C Release \
  -L allocation-full --no-tests=error --output-on-failure
```

The strict-realtime label validates deterministic contract properties. It is
not a portable latency promise. Allocation tests establish only the scope and
tracking backend reported by each test.

## Optional presentation

Presentation must be configured explicitly in a separate build directory.
The native tests run against that build:

```bash
cmake -S . -B build/presentation \
  -DNEURALE_BUILD_CPP_TESTS=ON \
  -DNEURALE_ENABLE_MKL=OFF \
  -DNEURALE_ENABLE_CUDA=OFF \
  -DNEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/presentation --config Release
ctest --test-dir build/presentation -C Release \
  -R '^neurale_experiment_presentation_(dependency|surface_contract|center_out_contract|ssvep_contract|webgrid_contract|speech_contract|recording)$' \
  --no-tests=error --output-on-failure
```

Install a presentation-enabled extension before running its Python tests;
the direct CMake build above does not replace the installed extension:

```bash
python -m pip install -e ".[test]" \
  --config-settings=cmake.define.NEURALE_ENABLE_MKL=OFF \
  --config-settings=cmake.define.NEURALE_ENABLE_CUDA=OFF \
  --config-settings=cmake.define.NEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON
python -m pytest tests/unit/experiments/test_presentation_import.py
```

Window tests require a usable desktop/OpenGL context and explicit fonts. On a
headless Linux runner, use Xvfb:

```bash
xvfb-run -a env \
  NEURALE_PRESENTATION_TEST_FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
  NEURALE_PRESENTATION_TEST_CJK_FONT=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc \
  ctest --test-dir build/presentation -C Release \
  -R '^neurale_experiment_presentation_(surface|center_out|ssvep|webgrid|speech|recording)_window$' \
  --no-tests=error --output-on-failure
```

Code 77 means the test could not obtain the required window-system capability;
it is a skip, not executed presentation coverage. Ordinary
``neurale.experiments`` imports must remain usable without the presentation
extension or its native dependencies.

## Sanitizers and fault tests

GNU and Clang support combined address, undefined-behavior, and leak sanitizer
builds. ThreadSanitizer is configured separately because it is incompatible
with that combination:

```bash
cmake -S . -B build/sanitizers \
  -DNEURALE_BUILD_CPP_TESTS=ON \
  -DNEURALE_ENABLE_MKL=OFF \
  -DNEURALE_ENABLE_CUDA=OFF \
  -DNEURALE_ENABLE_ASAN=ON \
  -DNEURALE_ENABLE_UBSAN=ON \
  -DNEURALE_ENABLE_LSAN=ON
cmake --build build/sanitizers --config RelWithDebInfo
ctest --test-dir build/sanitizers -C RelWithDebInfo \
  --no-tests=error --output-on-failure
```

Use {doc}`native_streaming_fault_testing` for deterministic native fault and
soak commands. Sanitizer and soak results apply only to the configured compiler,
platform, and test selection.

## Wheel and release validation

Release evidence is produced from the wheel that will be distributed, not from
an editable install. A core wheel uses:

```bash
CMAKE_ARGS="-DNEURALE_ENABLE_MKL=OFF -DNEURALE_ENABLE_CUDA=OFF -DNEURALE_ENABLE_EXPERIMENT_PRESENTATION=OFF" \
python -m pip wheel . --no-deps --wheel-dir dist
python tools/artifacts/validate_wheel_profile.py --profile core --wheel-dir dist
auditwheel show dist/*.whl > build/core-audit.txt 2>&1
python tools/artifacts/validate_native_binary.py --profile core --wheel-dir dist \
  --audit-report build/core-audit.txt
python tools/artifacts/validate_clean_wheel.py --profile core --wheel-dir dist
```

Use ``--profile presentation`` or ``--profile cuda`` only for a wheel built
with the matching explicit CMake definitions. The integrated distribution uses
``--profile release`` with all three native capabilities ON. Profile validation checks archive
contents and declared dependencies; native binary validation checks extension
exports, direct dependencies, and the complete-wheel audit report; clean-wheel
validation installs the artifact in an isolated environment and exercises only
declared runtime dependencies.

Install the selected wheel with its test extra before running the Python suite:

```bash
python -c 'from pathlib import Path; import subprocess, sys; wheels = list(Path("dist").glob("pyneurale-*.whl")); assert len(wheels) == 1, wheels; subprocess.check_call([sys.executable, "-m", "pip", "install", f"{wheels[0]}[test]"])'
python -m pytest
```

Linux wheel jobs also inspect the artifact with ``auditwheel show`` and Windows
jobs with ``delvewheel show``. Development-profile jobs pass the captured report
to ``tools/artifacts/validate_native_binary.py --audit-report`` with the same profile and
wheel. The integrated release build inspects and repairs the wheel before
its final archive, binary, and clean-install acceptance. See the
{doc}`release procedure <release>`.

CUDA validation requires a hardware runner with a compatible driver and at
least one visible device. A build on a host without executed GPU tests is not
CUDA behavior evidence. Presentation release validation likewise requires the
window tests above on the supported desktop profile.

## Documentation and formatting

Build the complete Sphinx and Doxygen site with warnings treated as errors:

```bash
python -m pip install -e ".[docs]"
python tools/build_docs.py --clean
```

Run the repository formatting hooks exactly as CI does:

```bash
pre-commit run --all-files --show-diff-on-failure
```

## Benchmarks

Benchmarks are manual evidence, not correctness tests or universal regression
thresholds. Use Release builds, preserve emitted metadata, and report the exact
command and host. The current entry points and interpretation rules are listed
in {doc}`benchmarks` and {doc}`native_streaming_benchmarks`.

Do not claim realtime, latency, throughput, allocation, or hardware behavior
from an unexecuted profile or from a benchmark that does not measure that
property.
