# Development

```{toctree}
:maxdepth: 1

native_streaming
adding_a_pipeline_adapter
devices
native_recording_replay
experiments
experiment_presentation
native_streaming_benchmarks
benchmarks
native_streaming_fault_testing
threshold_spike_detection
trial_alignment_contract
valley_seeking_reference
validation_matrix
release
```

For supported platforms and capabilities, see the repository-root
[support information](https://github.com/RanXingchen/pyneurale/blob/main/SUPPORT.md).

## Build locally

Install the Python dependencies and Doxygen, then run the shared build entry
point from the repository root:

```bash
python -m pip install -e ".[docs]"
python tools/build_docs.py --clean
```

Editable installation builds `neurale._native` with CMake and pybind11. Use an
explicit minimal profile for release-equivalent core work:

```bash
python -m pip install -e . \
  --config-settings=cmake.define.NEURALE_ENABLE_MKL=OFF \
  --config-settings=cmake.define.NEURALE_ENABLE_CUDA=OFF \
  --config-settings=cmake.define.NEURALE_ENABLE_EXPERIMENT_PRESENTATION=OFF
```

MKL and CUDA source discovery can be controlled during a build with `AUTO`,
`ON`, or `OFF`:

```bash
python -m pip install -e . \
  --config-settings=cmake.define.NEURALE_ENABLE_MKL=ON \
  --config-settings=cmake.define.NEURALE_ENABLE_CUDA=OFF \
  --config-settings=cmake.define.NEURALE_ENABLE_EXPERIMENT_PRESENTATION=OFF
```

`ON` requires the corresponding toolkit, `AUTO` enables it when found, and
`OFF` builds the runtime extension without that capability.

Scientific visualization is not shipped; no visualization dependency is needed
to build core or experiment presentation.

On Windows, `winget install DimitriVanHeesch.Doxygen` installs Doxygen. On
Ubuntu, use `sudo apt-get install doxygen graphviz`.

The HTML output is written to `docs/_build/html`. The build treats every
Sphinx and Doxygen warning as an error.

## Write Python API documentation

Document public modules, classes, methods, functions, properties, parameters,
return values, and exceptions with NumPy-style docstrings where the generated
API reference needs them. Source comments are not rendered by Autodoc. Keep
implementation-only explanations in concise source comments rather than
expanding the public reference with private details.

## Python module boundaries

Follow the {doc}`architecture overview <../architecture/overview>` for module
ownership and dependency direction. The {doc}`pipeline adapter guide
<adding_a_pipeline_adapter>` covers the private native integration boundary.

## Write C++ API documentation

Place public headers under `cpp/include` and use Doxygen comments:

```cpp
/// Compute a decoded output value.
/// @param input Input feature value.
/// @return Decoded output value.
double decode(double input);
```

Headers under `cpp/include` are scanned recursively. New declarations appear
in the C++ reference on the next build.

Only headers under `cpp/include` are scanned for the C++ reference; this does
not imply an installed C++ SDK or ABI guarantee. Declarations there require
Doxygen comments. Implementations under `cpp/src` and Python bindings under
`cpp/bindings` should use ordinary source comments.

## Continuous integration

The documentation workflow builds the complete site on matching pull requests,
matching pushes to `main`, and manual dispatch. Read the Docs can build the
same source using `.readthedocs.yaml`.
