# SPDX-FileCopyrightText: 2026 PyNeurale contributors
# SPDX-License-Identifier: MIT
#Requires -Version 7.2

[CmdletBinding()]
param(
    [switch]$List,
    [switch]$Check,
    [string[]]$Only = @(),
    [string]$WslDistro = "Ubuntu-24.04",
    [switch]$StopOnFailure
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
$BlockedCode = 86

function New-Job([string]$Workflow, [string]$Name, [string]$Platform, [string]$Python = '3.12', [string]$Variant = '') {
    $suffix = if ($Variant) { "-$Variant" } else { '' }
    [pscustomobject]@{
        Workflow = $Workflow
        Name = $Name
        Platform = $Platform
        Python = $Python
        Variant = $Variant
        Key = "${Workflow}:${Name}:${Platform}-py${Python}${suffix}"
    }
}

$Jobs = @(
    New-Job 'format.yml' 'format' 'linux'
    New-Job 'docs.yml' 'check' 'linux'
    New-Job 'pr-smoke.yml' 'source-imports' 'linux' '3.11'
    New-Job 'pr-smoke.yml' 'source-imports' 'linux' '3.12'
    New-Job 'pr-smoke.yml' 'cpu-wheel' 'linux'
    New-Job 'pr-smoke.yml' 'cpu-wheel' 'windows'
    New-Job 'test.yml' 'release-matrix-contract' 'linux'
    New-Job 'test.yml' 'python-only' 'linux'
    New-Job 'test.yml' 'built-install' 'linux' '3.11'
    New-Job 'test.yml' 'built-install' 'linux' '3.12'
    New-Job 'test.yml' 'sdist-install' 'linux'
    New-Job 'test.yml' 'windows-install' 'windows' '3.11'
    New-Job 'test.yml' 'windows-install' 'windows' '3.12'
    New-Job 'test.yml' 'experiment-presentation-windows' 'windows' '3.11'
    New-Job 'test.yml' 'experiment-presentation-windows' 'windows' '3.12'
    New-Job 'test.yml' 'native-cpu' 'linux'
    New-Job 'test.yml' 'experiment-presentation' 'linux'
    New-Job 'test.yml' 'experiment-presentation-bundled' 'linux' '3.11'
    New-Job 'test.yml' 'experiment-presentation-bundled' 'linux' '3.12'
    New-Job 'test.yml' 'strict-realtime-and-allocation-gate' 'linux'
    New-Job 'test.yml' 'mkl-native-and-allocation-gate' 'linux'
    New-Job 'sanitizers.yml' 'linux-asan-ubsan-lsan' 'linux'
    New-Job 'sanitizers.yml' 'windows-msvc-asan' 'windows'
    New-Job 'benchmarks.yml' 'physical-chain' 'linux' '3.12' 'builtin'
    New-Job 'benchmarks.yml' 'physical-chain' 'linux' '3.12' 'mkl'
    New-Job 'benchmarks.yml' 'controlled-history' 'self-hosted' '3.12' 'builtin'
    New-Job 'benchmarks.yml' 'controlled-history' 'self-hosted' '3.12' 'mkl'
    New-Job 'hardware.yml' 'cuda-runtime-and-models' 'linux'
)

function Get-ShortKey([string]$Value) {
    $bytes = [System.Security.Cryptography.SHA1]::HashData([System.Text.Encoding]::UTF8.GetBytes($Value))
    return [Convert]::ToHexString($bytes).Substring(0, 8).ToLowerInvariant()
}

function Expand-Template([string]$Template, [hashtable]$Values) {
    foreach ($name in $Values.Keys) {
        $Template = $Template.Replace("__${name}__", [string]$Values[$name])
    }
    return $Template
}

function Assert-WorkflowCoverage {
    foreach ($workflow in @($Jobs.Workflow | Sort-Object -Unique)) {
        $path = Join-Path $Root ".github/workflows/$workflow"
        $text = [System.IO.File]::ReadAllText($path)
        $jobsBlock = [regex]::Match($text, '(?m)^jobs:\s*$')
        if (-not $jobsBlock.Success) { throw "No jobs block in $workflow" }
        $actual = @([regex]::Matches($text.Substring($jobsBlock.Index + $jobsBlock.Length), '(?m)^  ([a-z][a-z0-9-]*):\s*$') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
        $mapped = @($Jobs | Where-Object Workflow -EQ $workflow | ForEach-Object Name | Sort-Object -Unique)
        $missing = @($actual | Where-Object { $_ -notin $mapped })
        $removed = @($mapped | Where-Object { $_ -notin $actual })
        if ($missing.Count -or $removed.Count) {
            throw "Local CI map drifted from ${workflow}: unmapped=$($missing -join ','); removed=$($removed -join ',')"
        }
    }
}

function Get-RunBlock([string]$Workflow, [string]$StepName) {
    $lines = [System.IO.File]::ReadAllLines((Join-Path $Root ".github/workflows/$Workflow"))
    $marker = "      - name: $StepName"
    $index = [Array]::IndexOf($lines, $marker)
    if ($index -lt 0) { throw "Missing step '$StepName' in $Workflow" }
    $start = -1
    for ($i = $index + 1; $i -lt $lines.Length; $i++) {
        if ($lines[$i].StartsWith('      - ')) { break }
        if ($lines[$i] -eq '        run: |') { $start = $i; break }
    }
    if ($start -lt 0) { throw "Expected literal run block in '$StepName'" }
    $block = [System.Collections.Generic.List[string]]::new()
    for ($i = $start + 1; $i -lt $lines.Length; $i++) {
        if ($lines[$i] -and -not $lines[$i].StartsWith('          ')) { break }
        $block.Add($(if ($lines[$i]) { $lines[$i].Substring(10) } else { '' }))
    }
    return ($block -join "`n")
}

function Invoke-LoggedProcess(
    [string]$File,
    [string[]]$Arguments,
    [string]$WorkingDirectory,
    [string]$InputText = '',
    [string]$LogPath = ''
) {
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $File
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.StandardInputEncoding = [System.Text.UTF8Encoding]::new($false)
    foreach ($argument in $Arguments) { [void]$startInfo.ArgumentList.Add($argument) }
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    try {
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if ($InputText) { $process.StandardInput.Write($InputText) }
        $process.StandardInput.Close()
        $process.WaitForExit()
        $output = $stdout.GetAwaiter().GetResult() + $stderr.GetAwaiter().GetResult()
        if ($LogPath) { [System.IO.File]::WriteAllText($LogPath, $output) }
        return [pscustomobject]@{ ExitCode = $process.ExitCode; Output = $output }
    }
    finally { $process.Dispose() }
}

function Quote-Bash([string]$Value) {
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function New-Snapshot([string]$Directory) {
    $source = Join-Path $Directory 'source'
    [void][System.IO.Directory]::CreateDirectory($source)
    $tracked = @(git -C $Root ls-files --cached --others --exclude-standard -z)
    if ($LASTEXITCODE -ne 0) { throw 'git ls-files failed' }
    foreach ($relative in ($tracked -split [char]0)) {
        if (-not $relative) { continue }
        $original = Join-Path $Root $relative
        if (-not (Test-Path -LiteralPath $original -PathType Leaf)) { continue }
        $destination = Join-Path $source $relative
        [void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $destination))
        [System.IO.File]::Copy($original, $destination, $true)
    }
    Copy-Item -LiteralPath (Join-Path $Root '.git') -Destination (Join-Path $source '.git') -Recurse
    $added = Invoke-LoggedProcess 'git' @('add', '-A') $source
    if ($added.ExitCode) { throw $added.Output }
    return $source
}

function New-WslSnapshot([string]$Source, [string]$RunId) {
    $path = Invoke-LoggedProcess 'wsl' @('-d', $WslDistro, '--', 'wslpath', '-u', $Source.Replace('\', '/')) $Root
    if ($path.ExitCode) { throw $path.Output }
    $from = $path.Output.Trim()
    $destination = "`$HOME/.cache/pyneurale-local-ci/$RunId/source"
    $command = "set -euo pipefail; mkdir -p $destination; rsync -a $(Quote-Bash $from)/ $destination/; cd $destination; git config core.filemode false; pwd"
    $result = Invoke-LoggedProcess 'wsl' @('-d', $WslDistro, '--', 'bash', '-lc', $command) $Root
    if ($result.ExitCode) { throw $result.Output }
    return @($result.Output.Trim() -split "`n")[-1].Trim()
}

$LinuxPreamble = @'
set -euo pipefail
require() { command -v "$1" >/dev/null || { echo "BLOCKED: missing $1"; exit 86; }; }
require cmake
require g++
require git
require rsync
export PIP_DISABLE_PIP_VERSION_CHECK=1
venv() {
  local version="$1" name="$2"
  local executable="python${version}" pyenv_executable prefix
  if ! command -v "$executable" >/dev/null; then
    pyenv_executable="${PYENV_ROOT:-$HOME/.pyenv}/bin/pyenv"
    if [[ -x "$pyenv_executable" ]] && prefix=$("$pyenv_executable" prefix "$version" 2>/dev/null); then
      executable="$prefix/bin/python${version}"
    fi
  fi
  if ! command -v "$executable" >/dev/null; then
    echo "BLOCKED: missing python${version} (also checked pyenv)"
    exit 86
  fi
  "$executable" -c "import sys; assert sys.version_info[:2] == tuple(map(int, '$version'.split('.')))"
  local environment="$PWD/../venvs/${name}"
  "$executable" -m venv "$environment"
  source "$environment/bin/activate"
  python -m pip install --upgrade pip
}
wheel() {
  local directory="$1"
  mapfile -t wheels < <(find "$directory" -maxdepth 1 -name '*.whl' -type f)
  [[ "${#wheels[@]}" -eq 1 ]] || { echo "Expected one wheel in $directory"; exit 1; }
  printf '%s\n' "${wheels[0]}"
}
oneapi() {
  if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    set +u
    source /opt/intel/oneapi/setvars.sh --force >/dev/null
    set -u
  fi
}
'@

# Job scripts are generated below; each runs in its own source snapshot.

function Get-LinuxJobScript($Job) {
    $key = Get-ShortKey $Job.Key
    $setup = "venv $($Job.Python) $key`n"
    switch ($Job.Name) {
        'release-matrix-contract' {
            return $LinuxPreamble + "`n" + $setup + (Get-RunBlock 'test.yml' 'Verify packaging and unavailable optional domains')
        }
        { $_ -in 'python-only', 'source-imports' } {
            return $LinuxPreamble + "`n" + $setup + @'
python -m pip install numpy scipy pytest
mapfile -t suite < <(python -c "import sys; sys.path.insert(0, 'tests'); import conftest; print('\n'.join('tests/' + x for x in sorted(conftest._PYTHON_ONLY_FILES)))")
PYTHONPATH="$PWD/src" python -m pytest -o addopts="" -m python_only "${suite[@]}"
'@
        }
        'format' {
            return $LinuxPreamble + "`n" + $setup + "python -m pip install pre-commit==4.6.0`npre-commit run --all-files --show-diff-on-failure`n"
        }
        'check' {
            return $LinuxPreamble + "`n" + @'
require doxygen
require dot
oneapi
'@ + "`n" + $setup + @'
python -m pip install -e '.[docs]'
python -c "import neurale.experiments.presentation; import neurale._native as n; assert not hasattr(n.experiments, 'presentation')"
python tools/build_docs.py --clean
'@
        }
        { $_ -in 'built-install', 'cpu-wheel', 'sdist-install', 'experiment-presentation', 'experiment-presentation-bundled' } {
            return Get-LinuxWheelScript $Job $key $setup
        }
        { $_ -in 'native-cpu', 'strict-realtime-and-allocation-gate', 'mkl-native-and-allocation-gate', 'linux-asan-ubsan-lsan' } {
            return Get-LinuxNativeScript $Job $key $setup
        }
        'physical-chain' { return Get-LinuxBenchmarkScript $Job $key $setup }
        'cuda-runtime-and-models' { return Get-LinuxCudaScript $Job $key $setup }
        default { throw "No Linux runner for $($Job.Key)" }
    }
}

function Get-LinuxWheelScript($Job, [string]$Key, [string]$Setup) {
    $name = $Job.Name
    $dist = "d/$Key"
    $build = "build/lc-$Key"
    if ($name -eq 'sdist-install') {
        $body = @'
export CMAKE_ARGS='-DNEURALE_ENABLE_MKL=OFF -DNEURALE_ENABLE_CUDA=OFF'
python -m pip install build auditwheel
mkdir -p __DIST__
python -m build --sdist --outdir __DIST__
mapfile -t archives < <(find __DIST__ -maxdepth 1 -name '*.tar.gz' -type f)
[[ "${#archives[@]}" -eq 1 ]] || exit 1
python -m pip wheel "${archives[0]}" --no-deps --wheel-dir __DIST__ --config-settings=build-dir=__BUILD__
artifact=$(wheel __DIST__)
auditwheel show "$artifact" > __DIST__/audit.txt 2>&1
python tools/artifacts/validate_native_binary.py --profile core --wheel-dir __DIST__ --audit-report __DIST__/audit.txt
python tools/artifacts/validate_clean_wheel.py --profile core --wheel-dir __DIST__
python -m pip install "${artifact}[test]"
python -m pytest tests/unit/io/nrf/test_schema_layer.py
'@
        return $LinuxPreamble + "`n" + $Setup + (Expand-Template $body @{ DIST = $dist; BUILD = $build })
    }

    $presentation = $name -in 'experiment-presentation', 'experiment-presentation-bundled'
    $profile = if ($presentation) { 'presentation' } else { 'core' }
    $flags = '-DNEURALE_ENABLE_MKL=OFF -DNEURALE_ENABLE_CUDA=OFF '
    $flags += if ($presentation) { '-DNEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON' } else { '-DNEURALE_ENABLE_EXPERIMENT_PRESENTATION=OFF' }
    if ($name -eq 'experiment-presentation-bundled') { $flags += ' -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER' }
    $preflight = ''
    if ($presentation) {
        $preflight = @'
require xvfb-run
[[ -f /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf ]] || { echo 'BLOCKED: DejaVu font missing'; exit 86; }
[[ -f /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc ]] || { echo 'BLOCKED: Noto CJK font missing'; exit 86; }
[[ -f /usr/include/GL/gl.h ]] || { echo 'BLOCKED: OpenGL development headers missing'; exit 86; }
'@ + "`n"
        if ($name -eq 'experiment-presentation') {
            $preflight += @'
require pkg-config
pkg-config --exists glfw3 freetype2 harfbuzz || { echo 'BLOCKED: system GLFW/FreeType/HarfBuzz development packages missing'; exit 86; }
'@ + "`n"
        } else {
            $preflight += @'
require pkg-config
pkg-config --exists x11 xrandr xinerama xcursor xi xext || { echo 'BLOCKED: bundled GLFW X11 development packages missing'; exit 86; }
'@ + "`n"
        }
    }
    $body = @'
export CMAKE_ARGS='__FLAGS__'
python -m pip install pybind11 auditwheel
mkdir -p __DIST__
python -m pip wheel . --no-deps --wheel-dir __DIST__ --config-settings=build-dir=__BUILD__/wheel
artifact=$(wheel __DIST__)
__BINARY_VALIDATION__
python tools/artifacts/validate_clean_wheel.py --profile __PROFILE__ --wheel-dir __DIST__
python -m pip install "${artifact}[test]"
'@
    $binaryValidation = if ($name -eq 'experiment-presentation') { '' } else {
        'auditwheel show "$artifact" > __DIST__/audit.txt 2>&1' + "`n" +
        'python tools/artifacts/validate_native_binary.py --profile __PROFILE__ --wheel-dir __DIST__ --audit-report __DIST__/audit.txt'
    }
    $values = @{ FLAGS = $flags; DIST = $dist; BUILD = $build; PROFILE = $profile }
    $values.BINARY_VALIDATION = Expand-Template $binaryValidation $values
    $script = $LinuxPreamble + "`n" + $preflight + $Setup + (Expand-Template $body $values) + "`n"
    if ($name -eq 'cpu-wheel') {
        $body = @'
python -c "import neurale._native as n; assert not n.build_info()['cuda_compiled']; assert n.build_info()['cpu_math_backend']=='none'; assert not hasattr(n.experiments, 'presentation')"
python -m pytest tests/integration/runtime/test_native_extension.py tests/unit/experiments/test_experiments_import.py tests/unit/experiments/test_presentation_import.py
python -m pytest -o "addopts=--require-native -ra" tests/integration/experiments/test_center_out_closed_loop.py::test_headless_center_out
'@
        return $script + (Expand-Template $body @{ DIST = $dist })
    }
    if ($name -eq 'built-install') {
        $body = @'
python -m pip install scikit-build-core setuptools 'pylsl>=1.17,<2'
python -m pip install --no-build-isolation examples/device_providers/pull examples/device_providers/callback examples/device_providers/python
python tools/artifacts/validate_device_providers.py
python -m pytest tests/unit/devices
python -c "import neurale.experiments; import neurale._native as n; assert not hasattr(n.experiments, 'presentation')"
python -m pytest -o "addopts=--require-native -ra" tests/integration/experiments/test_center_out_closed_loop.py::test_headless_center_out
cmake -S . -B __BUILD__/record-replay -DNEURALE_BUILD_BENCHMARKS=ON -DNEURALE_ENABLE_MKL=OFF -DNEURALE_ENABLE_CUDA=OFF -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build __BUILD__/record-replay --config Release --target neurale_recording_replay_benchmark --parallel 4
python -m pytest
mapfile -t suite < <(python -c "import sys; sys.path.insert(0, 'tests'); import conftest; print('\n'.join('tests/' + x for x in sorted(conftest._PYTHON_ONLY_INSTALLED_FILES)))")
python -m pytest -o addopts="-ra" -m python_only "${suite[@]}"
'@
        return $script + (Expand-Template $body @{ DIST = $dist; BUILD = $build })
    }
    $script += @'
python -c "from neurale.experiments.presentation import dependency_versions; print(dependency_versions())"
python -m pytest tests/unit/experiments/test_presentation_import.py
'@ + "`n"
    if ($name -eq 'experiment-presentation') {
        $script += "xvfb-run -a python -m pytest -o 'addopts=--require-native -ra' tests/integration/experiments/test_center_out_closed_loop.py::test_center_out_presentation`n"
    }
    $body = @'
cmake -S . -B __BUILD__/native -DNEURALE_BUILD_CPP_TESTS=ON -DNEURALE_BUILD_BENCHMARKS=OFF -DNEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON -DNEURALE_ENABLE_MKL=OFF -DNEURALE_ENABLE_CUDA=OFF -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build __BUILD__/native --config Release --target neurale_experiment_presentation_dependency_test neurale_experiment_presentation_surface_test neurale_experiment_presentation_center_out_test neurale_experiment_presentation_webgrid_test neurale_experiment_presentation_speech_test neurale_experiment_presentation_ssvep_test neurale_experiment_presentation_recording_test --parallel 4
ctest --test-dir __BUILD__/native -C Release -R '^neurale_experiment_presentation_(dependency|surface_contract|center_out_contract|webgrid_contract|speech_contract|ssvep_contract|recording)$' --no-tests=error --output-on-failure
xvfb-run -a env NEURALE_PRESENTATION_TEST_FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf NEURALE_PRESENTATION_TEST_CJK_FONT=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc NEURALE_PRESENTATION_TEST_FULLSCREEN=1 NEURALE_PRESENTATION_REQUIRE_DISPLAY=1 ctest --test-dir __BUILD__/native -C Release -R '^neurale_experiment_presentation_(surface|center_out|webgrid|speech|ssvep|recording)_window$' --no-tests=error --output-on-failure
'@
    return $script + (Expand-Template $body @{ BUILD = $build })
}

function Get-LinuxNativeScript($Job, [string]$Key, [string]$Setup) {
    $build = "build/lc-$Key"
    $mkl = if ($Job.Name -eq 'mkl-native-and-allocation-gate') { 'ON' } else { 'OFF' }
    $preflight = if ($mkl -eq 'ON') {
        "[[ -f /opt/intel/oneapi/mkl/latest/lib/cmake/mkl/MKLConfig.cmake ]] || { echo 'BLOCKED: oneMKL development package missing'; exit 86; }`noneapi`n"
    } else { '' }
    $sanitizers = if ($Job.Name -eq 'linux-asan-ubsan-lsan') { ' -DNEURALE_ENABLE_ASAN=ON -DNEURALE_ENABLE_UBSAN=ON -DNEURALE_ENABLE_LSAN=ON' } else { '' }
    $configuration = if ($Job.Name -eq 'linux-asan-ubsan-lsan') { 'RelWithDebInfo' } else { 'Release' }
    $body = @'
python -m pip install pybind11
cmake -S . -B __BUILD__ -DNEURALE_BUILD_CPP_TESTS=ON -DNEURALE_BUILD_BENCHMARKS=OFF -DNEURALE_ENABLE_MKL=__MKL__ -DNEURALE_ENABLE_CUDA=OFF__SANITIZERS__ -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
'@
    $script = $LinuxPreamble + "`n" + $preflight + $Setup + (Expand-Template $body @{ BUILD = $build; MKL = $mkl; SANITIZERS = $sanitizers }) + "`n"
    if ($Job.Name -in 'native-cpu', 'linux-asan-ubsan-lsan') {
        $script += "cmake --build $build --config $configuration --target neurale_public_header_self_containment --parallel 4`n"
    }
    $script += "cmake --build $build --config $configuration --parallel 4`n"
    switch ($Job.Name) {
        'native-cpu' { $script += "ctest --test-dir $build -C Release -LE 'strict-realtime|allocation-full|allocation-operator-new' --no-tests=error --output-on-failure`n" }
        'strict-realtime-and-allocation-gate' { $script += "ctest --test-dir $build -C Release -L 'strict-realtime|allocation-full' --no-tests=error --output-on-failure`n" }
        'mkl-native-and-allocation-gate' {
            $script += "ctest --test-dir $build -C Release -LE 'strict-realtime|allocation-full|allocation-operator-new' --no-tests=error --output-on-failure`n"
            $script += "ctest --test-dir $build -C Release -L 'strict-realtime|allocation-full' --no-tests=error --output-on-failure`n"
        }
        default { $script += "ctest --test-dir $build -C RelWithDebInfo -R '^(neurale_streaming_|neurale_pipeline_|neurale_recording_|neurale_experiments_|neurale_execution_)' -LE 'allocation-full|allocation-operator-new' --no-tests=error --output-on-failure`n" }
    }
    return $script
}

function Get-LinuxBenchmarkScript($Job, [string]$Key, [string]$Setup) {
    $mkl = $Job.Variant -eq 'mkl'
    $build = "build/lc-$Key"
    $output = "d/$Key"
    $backend = if ($mkl) { 'mkl' } else { 'none' }
    $enableMkl = if ($mkl) { 'ON' } else { 'OFF' }
    $softwareMkl = if ($mkl) { 'local' } else { 'disabled' }
    $preflight = if ($mkl) {
        "[[ -f /opt/intel/oneapi/mkl/latest/lib/cmake/mkl/MKLConfig.cmake ]] || { echo 'BLOCKED: oneMKL development package missing'; exit 86; }`noneapi`n"
    } else { '' }
    $body = @'
python -m pip install pybind11
cmake -S . -B __BUILD__ -DCMAKE_BUILD_TYPE=Release -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$PWD/__BUILD__/bin" -DNEURALE_BUILD_CPP_TESTS=OFF -DNEURALE_BUILD_BENCHMARKS=ON -DNEURALE_ENABLE_MKL=__MKL__ -DNEURALE_ENABLE_CUDA=OFF -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build __BUILD__ --config Release --target neurale_pipeline_strict_realtime_benchmark --parallel 4
mkdir -p __OUTPUT__
__BUILD__/bin/neurale_pipeline_strict_realtime_benchmark --warmup-seconds 1 --duration-seconds 60 --runs 3 > __OUTPUT__/results.jsonl
python tools/benchmark_policy.py capture --results __OUTPUT__/results.jsonl --output __OUTPUT__/environment.json --runner-class local-wsl-characterization --runner-label local-wsl --runner-label linux --runner-label x64 --runner-name WSL --runner-image Ubuntu-24.04-WSL --evidence-run-id __KEY__ --git-sha "$(git rev-parse HEAD)" --exact-command 'neurale_pipeline_strict_realtime_benchmark --warmup-seconds 1 --duration-seconds 60 --runs 3' --provider __VARIANT__ --software-component "cmake=$(cmake --version | head -n 1)" --software-component "pybind11=$(python -c 'import pybind11; print(pybind11.__version__)')" --software-component "mkl=__SOFTWARE_MKL__"
export EXPECTED_BUILD_FFT_BACKEND=__BACKEND__
export RESULT_PATH=__OUTPUT__/results.jsonl
'@
    $values = @{ BUILD = $build; OUTPUT = $output; MKL = $enableMkl; KEY = $Key; VARIANT = $Job.Variant; SOFTWARE_MKL = $softwareMkl; BACKEND = $backend }
    $validation = Get-RunBlock 'benchmarks.yml' 'Validate machine-readable result structure'
    return $LinuxPreamble + "`n" + $preflight + $Setup + (Expand-Template $body $values) + "`n" + $validation
}

function Get-LinuxCudaScript($Job, [string]$Key, [string]$Setup) {
    $dist = "d/$Key"
    $build = "build/lc-$Key"
    $runtime = Get-RunBlock 'hardware.yml' 'Require compiled and available CUDA runtime'
    $rejectSkips = Get-RunBlock 'hardware.yml' 'Reject an all-skipped CUDA result'
    $body = @'
if [[ -x /usr/local/cuda/bin/nvcc ]]; then export PATH="/usr/local/cuda/bin:$PATH"; fi
require nvidia-smi
require nvcc
nvidia-smi
nvcc --version
'@ + "`n" + $Setup + @'
export CMAKE_ARGS='-DNEURALE_ENABLE_MKL=OFF -DNEURALE_ENABLE_CUDA=ON -DNEURALE_ENABLE_EXPERIMENT_PRESENTATION=OFF'
python -m pip install pybind11 auditwheel
mkdir -p __DIST__ build/cuda-results
python -m pip wheel . --no-deps --wheel-dir __DIST__ --config-settings=build-dir=__BUILD__/wheel
artifact=$(wheel __DIST__)
auditwheel show "$artifact" > build/cuda-results/wheel-audit.txt 2>&1
python tools/artifacts/validate_native_binary.py --profile cuda --wheel-dir __DIST__ --audit-report build/cuda-results/wheel-audit.txt
python tools/artifacts/validate_clean_wheel.py --profile cuda --wheel-dir __DIST__
python -m pip install "${artifact}[test]"
'@ + "`n" + $runtime + "`n" + @'
python -m pytest -o addopts='--require-native -ra' -m gpu --deselect=tests/unit/models/test_neighbors.py::test_knn_cuda_restores_previous_device --junitxml=build/cuda-results/pytest.xml tests/integration/runtime/test_native_extension.py tests/unit/models/test_neighbors.py tests/unit/models/test_density.py tests/unit/models/test_manifold.py
'@ + "`n" + $rejectSkips
    return $LinuxPreamble + "`n" + (Expand-Template $body @{ DIST = $dist; BUILD = $build })
}

$WindowsPreamble = @'
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
function Run([string]$File, [string[]]$Arguments) {
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$File failed with exit code $LASTEXITCODE" }
}
'@

function Get-WindowsJobScript($Job) {
    $key = Get-ShortKey $Job.Key
    $envDir = "v/$key"
    $setup = @'
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { Write-Host 'BLOCKED: Python launcher unavailable'; exit 86 }
& py '-__PYTHON__' --version *> $null
if ($LASTEXITCODE -ne 0) { Write-Host 'BLOCKED: Python __PYTHON__ unavailable'; exit 86 }
Run 'py' @('-__PYTHON__', '-m', 'venv', '__ENV__')
$python = (Resolve-Path '__ENV__/Scripts/python.exe').Path
$env:PATH = "$((Resolve-Path '__ENV__/Scripts').Path);$env:PATH"
Run $python @('-m', 'pip', 'install', '--upgrade', 'pip')
'@
    $setup = $WindowsPreamble + "`n" + (Expand-Template $setup @{ PYTHON = $Job.Python; ENV = $envDir }) + "`n"
    if ($Job.Name -in 'windows-install', 'cpu-wheel', 'experiment-presentation-windows') {
        return Get-WindowsWheelScript $Job $key $setup
    }
    if ($Job.Name -eq 'windows-msvc-asan') {
        $build = "b/$key"
        $body = @'
Run $python @('-m', 'pip', 'install', 'pybind11')
$pybind = (& $python -m pybind11 --cmakedir).Trim()
Run 'cmake' @('-S', '.', '-B', '__BUILD__', '-DNEURALE_BUILD_CPP_TESTS=ON', '-DNEURALE_ENABLE_MKL=OFF', '-DNEURALE_ENABLE_CUDA=OFF', '-DNEURALE_ENABLE_ASAN=ON', "-Dpybind11_DIR=$pybind")
Run 'cmake' @('--build', '__BUILD__', '--config', 'RelWithDebInfo', '--target', 'neurale_public_header_self_containment', '--parallel', '4')
Run 'cmake' @('--build', '__BUILD__', '--config', 'RelWithDebInfo', '--parallel', '4')
Run 'ctest' @('--test-dir', '__BUILD__', '-C', 'RelWithDebInfo', '-R', '^(neurale_streaming_|neurale_pipeline_|neurale_recording_|neurale_experiments_|neurale_execution_)', '-LE', 'allocation-full|allocation-operator-new', '--no-tests=error', '--output-on-failure')
'@
        return $setup + (Expand-Template $body @{ BUILD = $build })
    }
    throw "No Windows runner for $($Job.Key)"
}

function Get-WindowsWheelScript($Job, [string]$Key, [string]$Setup) {
    $presentation = $Job.Name -eq 'experiment-presentation-windows'
    $profile = if ($presentation) { 'presentation' } else { 'core' }
    $enablePresentation = if ($presentation) { 'ON' } else { 'OFF' }
    $envDir = "v/$Key"
    $dist = "d/$Key"
    $build = "b/$Key"
    $body = @'
Run $python @('-m', 'pip', 'install', 'pybind11', 'delvewheel')
$env:CMAKE_ARGS = '-DNEURALE_ENABLE_MKL=OFF -DNEURALE_ENABLE_CUDA=OFF -DNEURALE_ENABLE_EXPERIMENT_PRESENTATION=__PRESENTATION__'
New-Item -ItemType Directory -Force '__DIST__' | Out-Null
Run $python @('-m', 'pip', 'wheel', '.', '--no-deps', '--wheel-dir', '__DIST__', '--config-settings=build-dir=__BUILD__/wheel')
$wheels = @(Get-ChildItem '__DIST__' -File -Filter '*.whl')
if ($wheels.Count -ne 1) { throw 'Expected exactly one wheel' }
$wheel = $wheels[0].FullName
& (Join-Path (Resolve-Path '__ENV__') 'Scripts/delvewheel.exe') show $wheel 2>&1 | Tee-Object -FilePath '__DIST__/audit.txt'
if ($LASTEXITCODE -ne 0) { throw 'delvewheel show failed' }
Run $python @('tools/artifacts/validate_native_binary.py', '--profile', '__PROFILE__', '--wheel-dir', '__DIST__', '--audit-report', '__DIST__/audit.txt')
Run $python @('tools/artifacts/validate_clean_wheel.py', '--profile', '__PROFILE__', '--wheel-dir', '__DIST__')
Run $python @('-m', 'pip', 'install', "${wheel}[test]")
'@
    $script = $Setup + (Expand-Template $body @{ PRESENTATION = $enablePresentation; DIST = $dist; BUILD = $build; ENV = $envDir; PROFILE = $profile }) + "`n"
    if ($Job.Name -eq 'cpu-wheel') {
        $body = @'
Run $python @('-c', "import neurale._native as n; assert not n.build_info()['cuda_compiled']; assert n.build_info()['cpu_math_backend']=='none'; assert not hasattr(n.experiments, 'presentation')")
Run $python @('-m', 'pytest', 'tests/integration/runtime/test_native_extension.py', 'tests/unit/experiments/test_experiments_import.py', 'tests/unit/experiments/test_presentation_import.py')
Run $python @('-m', 'pytest', '-o', 'addopts=--require-native -ra', 'tests/integration/experiments/test_center_out_closed_loop.py::test_headless_center_out')
'@
        return $script + (Expand-Template $body @{ DIST = $dist })
    }
    if ($Job.Name -eq 'windows-install') {
        return $script + @'
Run $python @('-m', 'pip', 'install', 'scikit-build-core', 'setuptools', 'wheel', 'pylsl>=1.17,<2')
Run $python @('-m', 'pip', 'install', '--no-build-isolation', 'examples/device_providers/pull', 'examples/device_providers/callback', 'examples/device_providers/python')
Run $python @('tools/artifacts/validate_device_providers.py')
Run $python @('-m', 'pytest', 'tests/unit/devices')
Run $python @('-m', 'pytest', 'tests/unit/io', 'tests/unit/recording', 'tests/specification')
'@
    }
    $reject = Get-RunBlock 'test.yml' 'Reject skipped Windows presentation contracts'
    $body = @'
Run $python @('-c', "from neurale.experiments.presentation import dependency_versions; from neurale import _native; assert hasattr(_native.experiments, 'presentation'); print(dependency_versions())")
Run $python @('-m', 'pytest', '-o', 'addopts=--require-native -ra', '--deselect=tests/unit/experiments/test_presentation_import.py::test_webgrid_presenter_accepts_misselection_feedback', '--junitxml=__DIST__/presentation-windows-pytest.xml', 'tests/unit/experiments/test_presentation_import.py')
Move-Item '__DIST__/presentation-windows-pytest.xml' 'presentation-windows-pytest.xml'
'@ + "`n" + $reject
    return $script + (Expand-Template $body @{ DIST = $dist })
}

function Assert-GeneratedScripts {
    if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) { throw '-Check requires WSL' }
    foreach ($job in $Jobs) {
        if ($job.Platform -eq 'linux') {
            $script = Get-LinuxJobScript $job
            if ($script -cmatch '__[A-Z_]+__') { throw "$($job.Key): unresolved template placeholder: $($Matches[0])" }
            $result = Invoke-LoggedProcess 'wsl' @('-d', $WslDistro, '--', 'bash', '-n') $Root $script
            if ($result.ExitCode) { throw "$($job.Key): $($result.Output)" }
        }
        elseif ($job.Platform -eq 'windows') {
            $script = Get-WindowsJobScript $job
            $tokens = $null
            $errors = $null
            [void][System.Management.Automation.Language.Parser]::ParseInput($script, [ref]$tokens, [ref]$errors)
            if ($errors.Count) { throw "$($job.Key): $($errors.Message -join '; ')" }
        }
    }
    Write-Host "Checked workflow coverage and $($Jobs.Count) job mappings"
}

function Invoke-OneJob($Job, [string]$Source, [string]$LinuxSource, [string]$Results) {
    $slug = [regex]::Replace($Job.Key.ToLowerInvariant(), '[^a-z0-9]+', '-').Trim('-')
    $log = Join-Path $Results "$slug.log"
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    if ($Job.Platform -eq 'self-hosted') {
        $detail = 'Requires the administered self-hosted controlled-history runner'
        [System.IO.File]::WriteAllText($log, "BLOCKED: $detail`n")
        return [pscustomobject]@{ job = $Job.Key; status = 'BLOCKED'; seconds = 0; log = $log; detail = $detail }
    }
    if ($Job.Platform -eq 'linux' -and -not $LinuxSource) {
        $detail = 'WSL snapshot unavailable'
        [System.IO.File]::WriteAllText($log, "BLOCKED: $detail`n")
        return [pscustomobject]@{ job = $Job.Key; status = 'BLOCKED'; seconds = 0; log = $log; detail = $detail }
    }
    $key = Get-ShortKey $Job.Key
    try {
        if ($Job.Platform -eq 'linux') {
            $jobSource = "$LinuxSource-jobs/$key"
            $copyCommand = "set -e; mkdir -p $(Quote-Bash "$LinuxSource-jobs"); cp -a $(Quote-Bash $LinuxSource) $(Quote-Bash $jobSource)"
            $copied = Invoke-LoggedProcess 'wsl' @('-d', $WslDistro, '--', 'bash', '-lc', $copyCommand) $Root
            if ($copied.ExitCode) { throw "Could not isolate WSL job: $($copied.Output)" }
            $script = "set -euo pipefail`ncd $(Quote-Bash $jobSource)`n" + (Get-LinuxJobScript $Job)
            $execution = Invoke-LoggedProcess 'wsl' @('-d', $WslDistro, '--', 'bash', '-s') $Root $script $log
        }
        else {
            $jobSource = Join-Path (Join-Path (Split-Path -Parent $Results) 'w') $key
            [void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $jobSource))
            Copy-Item -LiteralPath $Source -Destination $jobSource -Recurse
            $script = Get-WindowsJobScript $Job
            $execution = Invoke-LoggedProcess 'pwsh' @('-NoProfile', '-NonInteractive', '-Command', $script) $jobSource '' $log
        }
        $status = if ($execution.ExitCode -eq 0) { 'PASS' } elseif ($execution.ExitCode -eq $BlockedCode) { 'BLOCKED' } else { 'FAIL' }
        $output = $execution.Output
    }
    catch {
        $status = 'FAIL'
        $output = $_.ToString()
        [System.IO.File]::WriteAllText($log, $output + "`n")
    }
    $timer.Stop()
    $detail = if ($status -eq 'PASS') { '' } else { (@($output -split "`r?`n" | Where-Object { $_ } | Select-Object -Last 8) -join "`n") }
    return [pscustomobject]@{
        job = $Job.Key
        status = $status
        seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 2)
        log = $log
        detail = $detail
    }
}

Assert-WorkflowCoverage
if ($List) {
    $Jobs.Key
    exit 0
}
if ($Check) {
    Assert-GeneratedScripts
    exit 0
}
if (-not $IsWindows) { throw 'Run this entry point from Windows; Linux jobs use WSL Ubuntu' }

$selected = @($Jobs | Where-Object {
    $job = $_
    $Only.Count -eq 0 -or @($Only | Where-Object { $job.Key -like $_ }).Count -gt 0
})
if (-not $selected.Count) { throw '-Only did not match any job; see -List' }

$runId = [DateTime]::UtcNow.ToString('yyMMddHHmmss') + "-$PID"
$directory = Join-Path $Root "temp/ci/$runId"
$results = Join-Path $directory 'results'
[void][System.IO.Directory]::CreateDirectory($results)
$source = New-Snapshot $directory
$linuxSource = ''
if (@($selected | Where-Object Platform -EQ 'linux').Count -and (Get-Command wsl -ErrorAction SilentlyContinue)) {
    try { $linuxSource = New-WslSnapshot $source $runId }
    catch { Write-Warning "WSL snapshot unavailable: $_" }
}

Write-Host "Snapshot: $source"
Write-Host "Logs: $results"
$outcomes = [System.Collections.Generic.List[object]]::new()
for ($i = 0; $i -lt $selected.Count; $i++) {
    $job = $selected[$i]
    Write-Host "[$($i + 1)/$($selected.Count)] RUN $($job.Key)"
    $outcome = Invoke-OneJob $job $source $linuxSource $results
    $outcomes.Add($outcome)
    Write-Host "  $($outcome.status) ($($outcome.seconds)s) $($outcome.log)"
    if ($outcome.detail) { Write-Host "  $($outcome.detail.Replace("`n", "`n  "))" }
    [System.IO.File]::WriteAllText((Join-Path $results 'summary.json'), (ConvertTo-Json -InputObject @($outcomes.ToArray()) -Depth 5) + "`n")
    if ($StopOnFailure -and $outcome.status -eq 'FAIL') { break }
}
$passed = @($outcomes | Where-Object status -EQ 'PASS').Count
$failed = @($outcomes | Where-Object status -EQ 'FAIL').Count
$blocked = @($outcomes | Where-Object status -EQ 'BLOCKED').Count
Write-Host "Result: $passed PASS, $failed FAIL, $blocked BLOCKED"
Write-Host "Summary: $(Join-Path $results 'summary.json')"
if ($failed) { exit 1 }
if ($blocked) { exit 2 }
exit 0
