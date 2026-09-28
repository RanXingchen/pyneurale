# PyPI release checklist

For maintainers publishing PyNeurale from a local Windows workstation. Each
release contains four wheels (CPython 3.11/3.12 on Windows and Linux x86-64)
and one sdist. The wheels include oneMKL, CUDA, and experiment presentation.

## Prepare

1. Commit the release source and confirm that commit passed CI. Review
   `LICENSES_bundled.txt` and `NOTICE.md` against the native libraries included
   in the wheels.
2. Check that the workstation has CPython 3.11/3.12, MSVC, oneAPI, CUDA, WSL
   Ubuntu 24.04, and a CUDA device visible from Windows and WSL. Check
   `tools/build_release.sh` for its exact interpreter and toolkit paths.
3. Tag `HEAD` as `vX.Y.Z`. The build requires that exact tag and no uncommitted
   tracked changes.

## Build and review

From the repository root:

```powershell
.\tools\build_release.ps1 -Version X.Y.Z
```

The script validates the four wheels and sdist without uploading them. Review
the five files in `dist/release/X.Y.Z/` and the reports and `SHA256.json` in
`temp/release-build/X.Y.Z/evidence/`. Do not publish if a required check failed
or an expected artifact is missing.

## Upload

Create a PyPI API token (project-scoped if the project already exists). Twine
prompts for it interactively; do not put the token in the command line. Run:

```powershell
$version = 'X.Y.Z'
$artifacts = @(Get-ChildItem "dist/release/$version" -File | Select-Object -ExpandProperty FullName)
if ($artifacts.Count -ne 5) { throw 'Expected four wheels and one sdist' }
& "temp/release-build/$version/windows-py3.12/venv/Scripts/python.exe" -m twine upload --repository pypi $artifacts
```

Confirm all five files appear on PyPI, then push the tag and create the GitHub
release. If the upload is interrupted, inspect PyPI and upload only the missing
files from the same validated build; do not move the tag or rebuild files under
the published version.
