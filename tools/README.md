# Developer Tools

Run these commands from the repository root.

| Task | Entry point |
| --- | --- |
| Local CI | `tools/run_local_ci.ps1` |
| C++ tests | `tools/run_cpp_tests.ps1` or `.sh` |
| Benchmarks | `tools/run_benchmarks.ps1` or `.sh` |
| Documentation | `tools/build_docs.py` |
| Formatting | `tools/check_format.ps1` |
| Release build | `tools/build_release.ps1` or `.sh` |

`tools/artifacts/` contains the wheel, sdist, installed-package, and device-provider
validators used by CI and release builds. Invoke a validator directly when
checking a specific artifact; see [development validation](../docs/development/validation_matrix.md)
for examples.
