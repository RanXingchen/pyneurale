# Release

Each release contains four wheels (CPython 3.11/3.12 on Windows and Linux
x86-64) and one sdist. The wheels include oneMKL, CUDA, and native experiment
presentation. Linux wheels use the `manylinux_2_39_x86_64` tag.

## Prepare

1. Check the repository CI results and review the redistribution terms for
   the native libraries bundled in the wheels. `NOTICE.md` records notices;
   it does not grant redistribution rights.
2. Commit the release source and create a new `vX.Y.Z` tag. The build script
   requires that exact tag at `HEAD` and a clean tracked worktree.
3. Use the configured Windows x86-64 release machine with CPython 3.11/3.12,
   MSVC, Intel oneAPI, CUDA Toolkit, PowerShell, and WSL Ubuntu 24.04. The
   current WSL script expects pyenv Python 3.11.16, system Python 3.12,
   GCC 13.3, oneAPI under `/opt/intel/oneapi`, CUDA under `/usr/local/cuda`,
   and the X11/Xvfb development and font packages used by the presentation
   smoke test. A CUDA device must be visible on both Windows and WSL.

## Build and Verify

From the repository root on the tagged commit:

```powershell
tools/build_release.ps1 -Version X.Y.Z
```

The script builds and repairs all four wheels, builds the sdist, rebuilds a
wheel from the sdist, and checks the artifacts through isolated installs,
native-dependency inspection, GPU tests, a Linux presentation smoke test,
and `twine check`. It also confirms that the output contains exactly four
wheels and one sdist. It does not upload anything.

The five distribution files are written to `dist/release/X.Y.Z/`. Test reports
and `SHA256.json` are under `temp/release-build/X.Y.Z/evidence/`. Review both
directories before publishing, alongside the regular CI and hardware workflow
results.

## Publish

Configure the `pyneurale-release` Windows self-hosted runner, a protected
GitHub `pypi` environment, and a PyPI Trusted Publisher for
`RanXingchen/pyneurale` and `release.yml` before the first upload.

Run the manual `release.yml` workflow with the tagged version:

- `publish=false` builds and stores the artifacts on GitHub without uploading
  to PyPI.
- `publish=true` starts a new build from the tag, then waits for `pypi`
  environment approval before uploading its five artifacts. It does **not**
  reuse files from an earlier `publish=false` run.

Inspect that run's artifacts and hashes before approving publication. Do not
move an existing release tag to retry an upload; use a new version instead.

Setup references: [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/),
[GitHub environment protection](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments).
