# PyNeurale Release and Support Matrix

This document is the normative release and support matrix for PyNeurale. It defines the environments and optional capabilities included in the release contract.

---

## Core Release Matrix

| Area | Status | Frozen Matrix | Evidence and Limits |
|-|-|-|-|
| Python | **Supported** | CPython 3.11 and 3.12 | Both versions are installed from built CPU wheels and run through the primary Python test suite. |
| 3.10 and earlier, PyPy | **Unavailable** | None | The minimum Python version supported by NRF is 3.11. |
| Linux | **Supported** | Ubuntu 24.04, x86-64, glibc, GCC 13.3, C++20 | Linux has the complete installed-package/native CPU gates, sanitizers, strict-chain tests, and full allocation wrapping. The current WSL validation host is Ubuntu 24.04 x86-64 with GCC 13.3; this does not claim support for other distributions. |
| Windows | **Supported** | Windows 11 x86-64, MSVC 19.51, C++20 | Native, filesystem, recording/replay, and MSVC ASan validation have been executed on the Windows family. |

GitHub runner labels such as **ubuntu-24.04** and **windows-latest** are validation entry points, not additional platform promises. Each release record must retain the resolved runner image and compiler version. When a latest image changes, the fixed Supported rows above remain unchanged.

---

## Native Toolchain and CPU Baseline

Build-system dependency requirements:

| Requirement | Accepted Range |
|-|-|
| CMake | `>=3.24,<4.4` |
| pybind11 | `>=2.13` |

The release-supported native toolchain is:

| Area | Status | Version |
|-|-|-|
| C++ | **Supported** | C++20. |
| MSVC | **Supported** | MSVC 19.51 x64. |
| GCC | **Supported** | GCC 13.3 x86-64. |
| CPU ISA | **Supported** | Compiler-default x86-64 baseline. PyNeurale adds no AVX, SSE, `-march`, `-mcpu`, or MSVC `/arch` requirement. |

Provider dispatch may use instructions selected by a retained provider such as oneMKL. Such provider dispatch does not change the PyNeurale CPU build baseline and must be recorded separately in benchmark output.

---

## MKL and CUDA

| Capability | Status | Frozen Policy |
|-|-|-|
| Builtin CPU provider | **Supported** | MKL OFF and CUDA OFF is the portable release baseline. |
| oneMKL | **Supported** | MKL ON; LP64 interface, static MKL libraries, Intel threading. |
| CUDA kernels | **Experimental** | Some algorithms, including KNN, KDE, and LPP, support CUDA computation. |
| CUDA release distribution | **Unavailable** | No CUDA wheel/runtime bundle is published. Users must not infer support from CUDA AUTO detection. |

For CUDA source builds, CUDA ON requires both the Toolkit and a CUDA compiler; OFF excludes CUDA; AUTO is only opportunistic build detection. The runtime never probes CUDA merely on import. An explicitly requested CUDA device is a hard requirement; if the compiled extension or device is unavailable, the public device-unavailable error is raised, with no fallback to CPU.

---

## Experiment Presentation

| Platform | Status | Frozen Policy and Evidence |
|-|-|-|
| Windows 11 x86-64 desktop OpenGL | **Supported** | Optional presentation ON; OpenGL 3.3 Core-compatible path. The recorded Windows characterization used MSVC 19.51, NVIDIA RTX 5090/driver 610.47, GLFW 3.4, FreeType 2.13.3, and HarfBuzz 14.3.1. |
| Ubuntu 24.04 x86-64 X11/Xvfb with Mesa | **Supported** | System-dependency and bundled GLFW/FreeType/HarfBuzz builds, required-font preparation, hidden-window presenter tests, and OpenGL 3.3-compatible behavior form the Linux gate. |
| Native Wayland, macOS/Cocoa, other window systems or GPUs | **Unavailable** | No executed release evidence. GLFW or OpenGL discovery alone does not promote a platform. |
| Physical display onset or photodiode timing | **Unavailable** | M9 records software submit/swap-return evidence only. |

Presentation remains optional and defaults to OFF. Headless **neurale** and **neurale.experiments** imports do not load graphics dependencies. Bundled dependency versions are GLFW 3.4, FreeType 2.13.3, and HarfBuzz 14.3.1; OpenGL is always provided by the system. Required Speech fonts and glyph coverage are experiment resources and are not part of the core distribution.

---

## Installable Artifacts and Extras

Native capabilities are fixed when a wheel is built; installing a Python dependency extra cannot add a compiled extension to an existing wheel. PyNeurale therefore does not publish inert `cuda`, `presentation`, `torch`, `visualization`, or vendor-device extras.

| Artifact profile | Status | Required wheel contents and policy |
|-|-|-|
| Core | **Supported** | Contains `neurale._native` only, including the fixed built-in native processing stages compiled through `neurale.pipeline`. Its build sets MKL OFF, CUDA OFF, and presentation OFF explicitly so AUTO detection cannot change artifact contents. Runtime Python dependencies are exactly NumPy and SciPy. |
| Experiment presentation | **Supported** | Intentionally compiled into `neurale._native` in selected Windows 11 and Ubuntu 24.04 desktop wheels. GLFW, FreeType, and HarfBuzz are private static native dependencies; OpenGL remains a system dependency. This is a wheel profile, not a `[presentation]` extra. |
| CUDA | **Experimental** | Source/hardware-validation profile containing `neurale._native_cuda` alongside `neurale._native`. No CUDA release wheel, CUDA runtime bundle, or `[cuda]` extra is published. |
| Torch | **Unavailable** | No Torch artifact or `[torch]` extra is shipped. Runtime seed integration may use a separately installed Torch package but does not make it a PyNeurale capability distribution. |
| Device-specific | **Unavailable** | No physical-device artifact, vendor SDK dependency, `device-*` extra, or vendor extra is shipped. The simulated device remains part of core. |
| Visualization / visualization-live | **Unavailable** | M10 has not passed, so neither artifact nor extra exists. |

The selected-desktop presentation policy is deliberate: for a given distribution version, Python tag, ABI tag, and platform tag, the release index publishes one `pyneurale` wheel rather than indistinguishable core and presentation variants. Headless builds remain available from source with MKL, CUDA, and presentation set explicitly to OFF. `tools/validate_wheel_profile.py` verifies native-module membership, exact core runtime dependencies, and the absence of aspirational capability extras for every built profile.

The current dependency extras are functional Python dependency sets only:

| Extra | Owner |
|-|-|
| `nrf` | Canonical NRF JSON/Zarr/checksum dependencies. |
| `test` | Installed-package test dependencies, including `nrf`. |
| `docs` | Documentation build dependencies. |
| `dev` | Test, documentation, and repository lint tooling. |

Every release-profile wheel is accepted from a temporary virtual environment,
not from the repository interpreter. The acceptance command is:

```text
python tools/validate_clean_wheel.py --profile <core|presentation|cuda> <wheel>
```

The selected Windows and Ubuntu presentation release profiles run this clean
acceptance for every supported Python tag, CPython 3.11 and 3.12. The separate
Linux system-dependency presentation job is a build/runtime compatibility smoke
and therefore remains on one supported Python version.

It installs the wheel without extras so pip resolves only its declared runtime
dependencies, runs `pip check`, and executes the installed package with isolated
Python path handling from outside the checkout. The generated JSON records the
installed package path, metadata/native versions, ABI/build information, a CPU
operation, one compiled `neurale.pipeline` stage, CUDA behavior, presentation
behavior, and absent optional Python dependencies. Linux artifacts additionally retain `auditwheel show` output and
Windows artifacts retain `delvewheel show` output; profile checks reject native
dependencies owned by CUDA, presentation, MKL, or Qt when they occur in the
wrong artifact. A presentation load is tested without opening
a display, and a simulated missing native graphics dependency must retain its
public error context.

The executed Windows audit treats the supported MSVC redistributable and
OpenMP runtime (`msvcp140*`, `vcruntime140*`, and `vcomp140`) as platform
prerequisites. The presentation profile additionally uses Windows system
`opengl32`, GDI, shell, and user libraries; GLFW, FreeType, and HarfBuzz remain
statically linked and must not appear as external DLL dependencies.

---

## Native ABI, Headers, and Binary Boundary

PyNeurale wheels publish Python extension modules, not a separately installed
C++ SDK or shared-library ABI. `cpp/include/neurale` is the documented native
source surface used inside repository builds. Every header in that tree is
compiled as the first include in its own C++20 translation unit by the
`neurale_public_header_self_containment` target on the supported GCC and MSVC
validation paths; passing that check does not promise ABI compatibility for an
external C++ consumer.

The wheel module boundary is deliberately narrower:

| Profile | Native modules | Permitted exported entry points |
|-|-|-|
| Core | `_native` | `PyInit__native` |
| Experiment presentation | `_native` | Only `PyInit__native` |
| CUDA validation | `_native`, `_native_cuda` | One matching `PyInit_*` per module |

All extension targets explicitly use hidden C++ visibility, disable CMake's
automatic Windows symbol export, and use a Linux linker version script to hide
symbols contributed by static archives. The native ABI version, package version,
compiler, build type, provider, and CUDA compile state are checked through
`_native.build_info()` against installed wheel metadata. ABI version 1 refers
to this native-extension diagnostics contract; it is not a promise that C++
class layouts in the source header tree are stable across releases.

Release validation inspects the module itself with `dumpbin /EXPORTS` and
`dumpbin /DEPENDENTS` on Windows, or `nm -D` and `readelf -d` on Linux.
`tools/validate_native_binary.py` requires the exact initializer export and an
allowlisted set of platform/runtime dependencies. This catches an accidental
vendor SDK, CUDA runtime, MKL runtime, or graphics/font/window dependency in a
core artifact. Presentation permits only its documented system window/OpenGL
dependencies; GLFW, FreeType, and HarfBuzz remain private static dependencies.
With `--audit-report`, the same validator also checks complementary
`delvewheel show`/`auditwheel show` output, which covers the complete wheel
rather than only each extension's direct dependency table.

The configure-time dependency assertions freeze the current direction:
streaming links only the platform thread target; the core device target
links streaming, signal and the platform library loader; algorithm composition remains in the
internal pipeline; and presentation is reachable only from its optional module
and private recording seam.

---

## Optional Domains

| Capability | Status | Frozen Policy |
|-|-|-|
| Torch-backed decoder/training or Torch extra | **Unavailable** | Not implemented. |
| Simulated neural acquisition device | **Supported** | The simulated device is the supported hardware substitute for deterministic headless development. |
| External acquisition provider framework | **Implemented; example validation** | Native C ABI plugins and isolated Python SDK processes use the existing streaming boundary. Wheel-shipped SDK and standalone examples do not establish vendor hardware support. |
| Physical/vendor device distributions and SDKs | **Unavailable** | No concrete vendor adapter, SDK extra, hardware-in-the-loop gate, or vendor support claim exists. |
| Visualization package | **Unavailable** | M10 has not passed; no package or optional dependency extra is shipped. |

---

## Recording and Replay

| Capability | Status | Frozen Policy |
|-|-|-|
| Canonical NRF v1 offline I/O, diagnosis, recovery, finalization, and replay | **Supported** | Available on the supported Windows and Linux families with the `nrf` dependencies installed. Ordinary reads expose committed data only and do not repair or mutate a session. |
| Native critical `SessionRecorder` continuous-file capture on Ubuntu 24.04 x86-64 | **Experimental** | Replaces the former locked `/dev/shm` store with bounded queues and a growing disk spool. The revised file-backend lifecycle requires Linux validation before restoring the former backend's support claim. |
| Native critical `SessionRecorder` continuous-file capture on Windows 11 x86-64 | **Experimental** | A native worker appends beside the NRF target without whole-session RAM reservation. Disk-I/O timeout retains live resources until safe cleanup; remote Windows paths and silent memory fallback are refused. Default `buffered` durability supports process-crash recovery within the OS page-cache boundary, not power loss or kernel panic. Full release/crash-gate and sustained-throughput coverage remain required. |
| Old M4 NRF session reads | **Supported** | Read compatibility only; this does not restore the removed Python live-recording engine. |

---

## CI Evidence Layers

| Layer | Workflow | Trigger and scope |
|-|-|-|
| Fast PR smoke | `.github/workflows/pr-smoke.yml` | Pull requests: source imports on Python 3.11/3.12 plus builtin-CPU wheel smoke on Ubuntu 24.04 and Windows. |
| Full correctness | `.github/workflows/test.yml` | `main` or manual: clean-venv core/presentation wheel acceptance and shared-library audits, installed wheels, builtin/MKL native tests, headless experiments, Windows presentation build/load contracts, Linux presentation/Xvfb window smoke, strict contracts, and allocation gates. |
| Sanitizers | `.github/workflows/sanitizers.yml` | `main` or manual: Linux ASan/UBSan/LSan and Windows MSVC ASan. |
| CUDA hardware | `.github/workflows/hardware.yml` | Manual labelled self-hosted CUDA runner: clean-venv CUDA wheel acceptance, shared-library audit, required runtime probe, and single-device KNN/KDE/LPP tests; capability absence or selected-test skips fail the job. |
| Performance characterization | `.github/workflows/benchmarks.yml` | Scheduled/manual builtin and MKL physical-chain JSONL artifacts on GitHub-hosted runners; structure smoke and release characterization only. |
| Controlled benchmark history | `.github/workflows/benchmarks.yml` | Manual collection on the administered `pyneurale-benchmark` self-hosted runner class; exact environment and eligibility reports are archived, but no timing gate is active yet. |

No visualization CI is registered because M10 is unavailable. A required provider, display, font, sanitizer, or hardware capability must fail its owning job when absent; it must not be converted into a passing skip.

The benchmark regression contract is defined by
`benchmarks/regression_policy.md` and its machine-readable
`benchmarks/regression_policy.json`. Deterministic zero-allocation requirements
remain correctness gates. Timing, throughput, RSS, storage, acquisition, and
presentation results remain characterization or release evidence until at least
five independent runs on one controlled environment satisfy the configured
reproducibility bound and an explicit reviewed baseline and threshold are added.
Developer machines and GitHub-hosted runner measurements cannot establish a
universal hard gate.

---

## Release Evidence Rule

Before publication, the release record must contain the exact commands and resolved versions for every Supported row it exercises. Skipped hardware, window, font, sanitizer, allocation, or provider tests are not passing evidence.
