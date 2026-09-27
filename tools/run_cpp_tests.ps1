[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Config = "Release",

    [string]$BuildDir = "build/cpp-tests",

    [string]$TestFilter = "",

    # ctest 并行度。非正数表示自动取 min(4, 处理器数)。
    [int]$CTestJobs = 0,

    # 单个测试用例的执行超时（秒）。非正数表示不设全局超时。
    [int]$CTestTimeout = 600,

    # 默认包含 soak 和 fault-injection；仅在传入此开关时跳过。
    [switch]$SkipLongTests,

    [switch]$Clean,

    # AUTO：检测到可用后启用；ON：必须可用；OFF：禁用。
    [ValidateSet("AUTO", "ON", "OFF")]
    [string]$Cuda = "AUTO",

    [ValidateSet("AUTO", "ON", "OFF")]
    [string]$Mkl = "AUTO",

    [ValidateSet("ON", "OFF")]
    [string]$ExperimentPresentation = "OFF",

    # CMake 生成器。AUTO：按操作系统探测（Windows 优先 VS 2026，否则 Ninja；
    # Unix 优先 Ninja，否则 Unix Makefiles）。也可显式指定，如 "Ninja"。
    [string]$Generator = "AUTO"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$MinimumCMakeVersion = [Version]"3.24"

Import-Module (Join-Path $PSScriptRoot "cmake-runner-common.psm1")

function Invoke-Checked {
    param(
        [Parameter(Mandatory)][string]$Description,
        [Parameter(Mandatory)][scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Description" -ForegroundColor Cyan
    & $Command

    if ($LASTEXITCODE -ne 0) {
        throw "$Description 失败，退出码：$LASTEXITCODE"
    }
}

Assert-Command "cmake"
Assert-Command "ctest"
Assert-Command "python"

# 检测操作系统（兼容 Windows PowerShell 5.1 与 pwsh 7+）。
$OnWindows = if ($PSVersionTable.PSEdition -eq "Desktop") { $true } else { $IsWindows }

$CMakeVersion = Get-CMakeVersion
$CMakeHelp = Get-CMakeGeneratorHelp
$ResolvedGenerator = Resolve-Generator -Requested $Generator -CMakeHelp $CMakeHelp -OnWindows $OnWindows

if ($CMakeVersion -lt $MinimumCMakeVersion) {
    throw "当前 CMake 版本 $CMakeVersion 低于最低要求 $MinimumCMakeVersion。"
}

$ScriptDir = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ScriptDir)) {
    $ScriptDir = (Get-Location).Path
}

# 脚本必须位于 pyneurale/tools。
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$RootCMakeLists = Join-Path $ProjectRoot "CMakeLists.txt"

if (-not (Test-Path $RootCMakeLists -PathType Leaf)) {
    throw (
        "未在项目根目录找到 CMakeLists.txt：$RootCMakeLists。`n" +
        "请将本脚本放在 pyneurale/tools 目录下。"
    )
}

if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    $ResolvedBuildDir = [System.IO.Path]::GetFullPath($BuildDir)
}
else {
    $ResolvedBuildDir = [System.IO.Path]::GetFullPath(
        (Join-Path $ProjectRoot $BuildDir)
    )
}

Push-Location $ProjectRoot
try {
    if ($Clean -and (Test-Path $ResolvedBuildDir)) {
        Assert-SafeBuildDirectoryRemoval `
            -Directory $ResolvedBuildDir `
            -ProjectRoot $ProjectRoot `
            -OnWindows $OnWindows
        if ($PSCmdlet.ShouldProcess($ResolvedBuildDir, "删除旧构建目录（-Clean）")) {
            Write-Host "删除旧构建目录：$ResolvedBuildDir" -ForegroundColor Yellow
            Remove-Item -Recurse -Force $ResolvedBuildDir
        }
    }

    # 若构建目录由别的生成器创建，自动清理，避免 CMake generator mismatch。
    $CachedGenerator = Get-CachedGenerator -Directory $ResolvedBuildDir
    if ($CachedGenerator -and $CachedGenerator -ne $ResolvedGenerator) {
        Assert-SafeBuildDirectoryRemoval `
            -Directory $ResolvedBuildDir `
            -ProjectRoot $ProjectRoot `
            -OnWindows $OnWindows
        if ($PSCmdlet.ShouldProcess($ResolvedBuildDir, "删除生成器不匹配的构建目录")) {
            Write-Host (
                "构建目录使用旧生成器 '$CachedGenerator'；" +
                "将自动清理并切换到 '$ResolvedGenerator'。"
            ) -ForegroundColor Yellow
            Remove-Item -Recurse -Force $ResolvedBuildDir
        }
    }

    # 多配置生成器（VS / Ninja Multi-Config / Xcode）在 build/ctest 用 --config/-C；
    # 单配置生成器（Ninja / Unix Makefiles）改在 configure 传 CMAKE_BUILD_TYPE。
    $IsMultiConfig = $ResolvedGenerator -match 'Visual Studio|Ninja Multi-Config|Xcode'
    $IsVsGenerator = $ResolvedGenerator -like "Visual Studio *"

    # 探测 pybind11 的 CMake 目录；未安装则直接报错退出，不自动安装。
    $RawPybind11Dir = & python -m pybind11 --cmakedir 2>$null
    $Pybind11Dir = if ($RawPybind11Dir) { "$RawPybind11Dir".Trim() } else { "" }

    if ([string]::IsNullOrWhiteSpace($Pybind11Dir)) {
        throw "未找到 pybind11，请先安装（例如：python -m pip install pybind11）后重试。"
    }

    # 并行度：显式传入优先；否则默认 min(4, 处理器数)。
    if ($CTestJobs -gt 0) {
        $ResolvedCTestJobs = $CTestJobs
    }
    else {
        $ProcessorCount = [Math]::Max(1, [Environment]::ProcessorCount)
        $ResolvedCTestJobs = [Math]::Min(4, $ProcessorCount)
    }

    Write-Host ""
    Write-Host "PyNeurale C++ Test Runner" -ForegroundColor Green
    Write-Host "  项目根目录：$ProjectRoot"
    Write-Host "  构建目录：  $ResolvedBuildDir"
    Write-Host "  CMake：     $CMakeVersion"
    Write-Host "  生成器：    $ResolvedGenerator"
    Write-Host "  平台：      $(if ($OnWindows) { 'Windows' } else { 'Unix' })"
    Write-Host "  构建配置：  $Config"
    Write-Host "  CUDA：      $Cuda（默认 AUTO）"
    Write-Host "  MKL：       $Mkl（默认 AUTO）"
    Write-Host "  ExperimentPresentation：       $ExperimentPresentation（默认 OFF）"
    Write-Host "  并行度：    $ResolvedCTestJobs"
    Write-Host "  超时：      $(if ($CTestTimeout -gt 0) { "${CTestTimeout}s" } else { '不设' })"
    Write-Host "  长测试：    $(if ($SkipLongTests) { '跳过' } else { '包含（默认）' })"

    $ConfigureArgs = @(
        "-S", $ProjectRoot,
        "-B", $ResolvedBuildDir,
        "-G", $ResolvedGenerator,
        "-DNEURALE_BUILD_CPP_TESTS=ON",
        "-DNEURALE_ENABLE_CUDA=$Cuda",
        "-DNEURALE_ENABLE_MKL=$Mkl",
        "-DNEURALE_ENABLE_EXPERIMENT_PRESENTATION=$ExperimentPresentation",
        "-Dpybind11_DIR=$Pybind11Dir"
    )

    if ($IsVsGenerator) {
        $ConfigureArgs += @("-A", "x64")
    }

    if (-not $IsMultiConfig) {
        $ConfigureArgs += @("-DCMAKE_BUILD_TYPE=$Config")
    }

    if ($PSCmdlet.ShouldProcess("C++ 测试流程", "配置/编译/运行 C++ 测试")) {
        Invoke-Checked "配置 CMake" {
            cmake @ConfigureArgs
        }

        $BuildArgs = @("--build", $ResolvedBuildDir, "--parallel")
        if ($IsMultiConfig) {
            $BuildArgs += @("--config", $Config)
        }

        Invoke-Checked "编译 C++ 测试（$Config）" {
            cmake @BuildArgs
        }

        $CTestArgs = @(
            "--test-dir", $ResolvedBuildDir,
            "--no-tests=error",
            "--output-on-failure",
            "--parallel", $ResolvedCTestJobs
        )

        if ($IsMultiConfig) {
            $CTestArgs += @("-C", $Config)
        }

        if ($CTestTimeout -gt 0) {
            $CTestArgs += @("--timeout", $CTestTimeout)
        }

        if (-not [string]::IsNullOrWhiteSpace($TestFilter)) {
            $CTestArgs += @("-R", $TestFilter)
        }

        if ($SkipLongTests) {
            $CTestArgs += @("-LE", "soak|fault-injection")
        }

        Invoke-Checked "运行 C++ 测试" {
            ctest @CTestArgs
        }

        Write-Host ""
        Write-Host "全部指定测试已通过。" -ForegroundColor Green
    }
}
finally {
    Pop-Location
}
