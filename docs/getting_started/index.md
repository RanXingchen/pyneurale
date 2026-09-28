# Getting started

## Install from source

Use CPython 3.11 or 3.12 in a virtual environment. A source build requires a
C++20 compiler and CMake. See the repository-root
[support information](https://github.com/RanXingchen/pyneurale/blob/main/SUPPORT.md)
for supported platforms and capabilities.

Clone the repository and create a virtual environment in its root:

```console
git clone https://github.com/RanXingchen/pyneurale.git
cd pyneurale
python -m venv .venv
```

Activate `.venv` (`source .venv/bin/activate` on Linux or
`.venv\Scripts\Activate.ps1` in PowerShell), then build the core profile. The
explicit settings prevent local MKL, CUDA, or graphics discovery from changing
the build:

```console
python -m pip install . --config-settings=cmake.define.NEURALE_ENABLE_MKL=OFF --config-settings=cmake.define.NEURALE_ENABLE_CUDA=OFF --config-settings=cmake.define.NEURALE_ENABLE_EXPERIMENT_PRESENTATION=OFF
```

To enable graphical experiment windows, change `NEURALE_ENABLE_EXPERIMENT_PRESENTATION=OFF`
to `ON` in that command; this requires a supported desktop OpenGL setup.

To read or write NRF v1 sessions, use `".[nrf]"` instead of `.` in the same
command; keep all three CMake settings. Choose the extra before installing so
the source is built once. Native capabilities such as CUDA and presentation
are build profiles, not extras that can change an installed wheel. For an
editable development install, replace `pip install .` with
`pip install -e ".[dev]"` in the same command and keep the CMake settings.
The `dev` extra includes `nrf`, tests, and documentation tools; see the
{doc}`development validation guide <../development/validation_matrix>` for
validation commands.

## Verify the installation

```python
from neurale.signal.simulation import SignalGenerator

samples = SignalGenerator.tones(2, 1000.0, [10.0, 20.0]).generate(0, 8)
print(samples.shape)
```

The output is `(8, 2)`. This runs native signal generation; printing
`neurale.__version__` alone would not test the native extension.

Continue with the {doc}`signal guide <../user_guide/signals>`,
{doc}`native pipelines <../user_guide/pipelines>`,
{doc}`Center-Out demo <../user_guide/center_out>`, or
{doc}`SSVEP experiments <../user_guide/ssvep>`.
