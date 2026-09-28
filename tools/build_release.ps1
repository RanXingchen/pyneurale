# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9]+\.[0-9]+\.[0-9]+$')]
    [string]$Version
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$tag = "v$Version"
$currentTag = git -C $repo describe --tags --exact-match HEAD 2>$null
if ($LASTEXITCODE -ne 0 -or $currentTag -ne $tag) {
    throw "HEAD must be exactly tagged $tag"
}
git -C $repo diff --quiet
if ($LASTEXITCODE -ne 0) { throw 'Tracked files have uncommitted changes' }
git -C $repo diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw 'Staged changes are not part of the tag' }

$output = Join-Path $repo "dist/release/$Version"
if ((Test-Path $output) -and (Get-ChildItem $output -File)) {
    throw "Release output is not empty: $output"
}
New-Item -ItemType Directory -Force $output | Out-Null
$evidence = Join-Path $repo "temp/release-build/$Version/evidence"
New-Item -ItemType Directory -Force $evidence | Out-Null

function Run([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE"
    }
}

$env:CMAKE_ARGS = '-DNEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON -DNEURALE_ENABLE_MKL=ON -DNEURALE_ENABLE_CUDA=ON -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER'
foreach ($minor in @('11', '12')) {
    $work = Join-Path $repo "temp/release-build/$Version/windows-py3.$minor"
    Run 'py' @("-3.$minor", '-m', 'venv', (Join-Path $work 'venv'))
    $python = Join-Path $work 'venv/Scripts/python.exe'
    $delvewheel = Join-Path $work 'venv/Scripts/delvewheel.exe'
    Run $python @('-m', 'pip', 'install', '--upgrade', 'pip', 'pybind11', 'delvewheel', 'packaging', 'build', 'twine')
    $raw = Join-Path $work 'raw'
    New-Item -ItemType Directory -Force $raw | Out-Null
    Run $python @('-m', 'pip', 'wheel', $repo, '--no-deps', '--wheel-dir', $raw, "--config-settings=build-dir=$(Join-Path $work 'build')")
    $wheels = @(Get-ChildItem $raw -Filter '*.whl')
    if ($wheels.Count -ne 1) { throw "Expected one raw Windows wheel for Python 3.$minor" }
    Run $delvewheel @('show', $wheels[0].FullName)
    Run $delvewheel @('repair', '--ignore-existing', '--no-mangle-all', '--wheel-dir', $output, $wheels[0].FullName)
    $repaired = Join-Path $output "pyneurale-$Version-cp3$minor-cp3$minor-win_amd64.whl"
    if (-not (Test-Path $repaired)) { throw 'Repaired Windows wheel has an unexpected name/version' }
    Run $python @((Join-Path $repo 'tools/artifacts/validate_wheel_profile.py'), '--profile', 'release', $repaired)
    Run $python @((Join-Path $repo 'tools/artifacts/validate_native_binary.py'), '--profile', 'release', '--wheel', $repaired, '--report', (Join-Path $evidence "windows-py3.$minor-native.json"))
    Run $python @((Join-Path $repo 'tools/artifacts/validate_clean_wheel.py'), '--profile', 'release', '--require-cuda-device', $repaired, '--report', (Join-Path $evidence "windows-py3.$minor-clean.json"))
    Run $python @('-m', 'pip', 'install', "${repaired}[test]")
    $gpuReport = Join-Path $evidence "windows-py3.$minor-gpu.xml"
    Run $python @(
        '-m', 'pytest', '-o', 'addopts=--require-native -ra', '-m', 'gpu',
        '--deselect=tests/unit/models/test_neighbors.py::test_knn_cuda_restores_previous_device',
        "--junitxml=$gpuReport", '-q',
        (Join-Path $repo 'tests/integration/runtime/test_native_extension.py'),
        (Join-Path $repo 'tests/unit/models/test_neighbors.py'),
        (Join-Path $repo 'tests/unit/models/test_density.py'),
        (Join-Path $repo 'tests/unit/models/test_manifold.py')
    )
    Run $python @((Join-Path $repo 'tools/artifacts/validate_required_pytest.py'), $gpuReport, '--minimum', '31')
}

$wslRepo = (wsl -e wslpath -a $repo).Trim()
$wslOutput = (wsl -e wslpath -a $output).Trim()
$wslEvidence = (wsl -e wslpath -a $evidence).Trim()
Run 'wsl' @('-e', 'bash', "$wslRepo/tools/build_release.sh", $wslRepo, $wslOutput, $Version, $wslEvidence)

$python = Join-Path $repo "temp/release-build/$Version/windows-py3.12/venv/Scripts/python.exe"
Run $python @('-m', 'build', '--sdist', '--outdir', $output, $repo)
$sdist = Join-Path $output "pyneurale-$Version.tar.gz"
if (-not (Test-Path $sdist)) { throw 'Source distribution has an unexpected name/version' }
$sdistWheelDir = Join-Path $repo "temp/release-build/$Version/sdist-wheel"
New-Item -ItemType Directory -Force $sdistWheelDir | Out-Null
Run $python @('-m', 'pip', 'wheel', $sdist, '--no-deps', '--wheel-dir', $sdistWheelDir, "--config-settings=build-dir=$(Join-Path $repo "temp/release-build/$Version/sdist-build")")
$sdistWheels = @(Get-ChildItem $sdistWheelDir -Filter '*.whl')
if ($sdistWheels.Count -ne 1) { throw 'Expected one wheel built from sdist' }
Run $python @((Join-Path $repo 'tools/artifacts/validate_wheel_profile.py'), '--profile', 'release', $sdistWheels[0].FullName)
$artifacts = @(Get-ChildItem $output -File | ForEach-Object FullName)
Run $python (@('-m', 'twine', 'check') + $artifacts)
Run $python @((Join-Path $repo 'tools/artifacts/validate_release_dist.py'), $output, '--version', $Version)
$hashes = $artifacts | ForEach-Object { Get-FileHash -Algorithm SHA256 $_ }
$hashes | Format-Table Path, Hash
$hashes | Select-Object @{Name = 'File'; Expression = { [System.IO.Path]::GetFileName($_.Path) }}, Hash |
    ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $evidence 'SHA256.json')
Write-Host "Validated release artifacts: $output"
