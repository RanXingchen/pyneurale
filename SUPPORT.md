# Support

PyNeurale supports CPython 3.11 and 3.12.

## Operating Systems

- Windows 11, x86-64.
- Linux, x86-64, with glibc 2.39 or newer (Ubuntu 24.04 is the tested baseline).

The release distribution consists of one source archive and four wheels, one
for each Python and operating-system combination above. The wheels include
oneMKL, CUDA kernels, and native experiment presentation. CPU use does not
require a CUDA device.

## Capabilities

| Capability | Support |
| --- | --- |
| CPU processing and oneMKL | Supported. |
| CUDA kernels | Experimental; currently available for selected algorithms, including KNN, KDE, and LPP. Requires a compatible NVIDIA GPU and driver. |
| Experiment presentation | Supported on Windows desktop and Linux X11 with OpenGL 3.3 or newer. |
| NRF recording and replay | Supported with the `nrf` extra. |
| Native continuous recording | Experimental on Windows and Linux. Buffered durability does not guarantee recovery after power loss. |
| External acquisition | Device-provider APIs are included; LSL ingestion uses the `lsl` extra, while vendor adapters require separate plugins. |

macOS and native Wayland presentation are not currently supported. The wheel
does not install a separate C++ SDK or promise a stable C++ ABI.

## Source Builds

Source builds require C++20, CMake `>=3.24,<4.4`, and pybind11 2.13 or newer.
The tested compilers are MSVC 19.51 on Windows and GCC 13.3 on Linux. MKL,
CUDA, and presentation can each be selected at build time; a CPU-only build
does not require their toolkits.

For validation commands and release packaging, see
[development validation](docs/development/validation_matrix.md) and
[release](docs/development/release.md).
