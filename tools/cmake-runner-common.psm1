# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT
#
# Shared CMake runner helpers used by tools/run_benchmarks.ps1 and
# tools/run_cpp_tests.ps1: command presence check, CMake version parse,
# generator help/listing, generator resolution, cached-generator probe.
# Import-Module this from both scripts so the two runners stay in sync.

$VsGenerator = "Visual Studio 18 2026"

function Assert-Command {
    param([Parameter(Mandatory)][string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "未找到命令 '$Name'，请先安装并确保它已加入 PATH。"
    }
}

function Get-CMakeVersion {
    $Output = @(& cmake --version 2>&1)
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -ne 0) {
        throw (
            "无法读取 CMake 版本，退出码：$ExitCode。`n" +
            ($Output -join [Environment]::NewLine)
        )
    }

    $VersionLine = [string]$Output[0]
    if ($VersionLine -notmatch 'cmake version\s+(\d+)\.(\d+)(?:\.(\d+))?') {
        throw "无法解析 CMake 版本：$VersionLine"
    }

    $Patch = if ($Matches[3]) { [int]$Matches[3] } else { 0 }
    return [Version]::new(
        [int]$Matches[1],
        [int]$Matches[2],
        $Patch
    )
}

function Get-CMakeGeneratorHelp {
    $CMakeHelp = (& cmake --help) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw "无法读取 CMake 生成器列表（cmake --help 失败，退出码 $LASTEXITCODE）。"
    }
    return $CMakeHelp
}

function Test-GeneratorAvailable {
    param(
        [Parameter(Mandatory)][string]$GeneratorName,
        [Parameter(Mandatory)][string]$Help
    )

    # 按行精确匹配生成器名（后跟 =），避免 'Ninja' 子串命中 'Ninja Multi-Config'
    # 或 'CodeBlocks - Ninja' 之类的额外生成器。cmake --help 的生成器行形如
    # '* Unix Makefiles  = ...' 或 '  Ninja  = ...'。
    $Pattern = '(?m)^\s*\*?\s*' + [regex]::Escape($GeneratorName) + '\s*='
    return $Help -match $Pattern
}

function Resolve-Generator {
    param(
        [Parameter(Mandatory)][string]$Requested,
        [Parameter(Mandatory)][string]$CMakeHelp,
        [Parameter(Mandatory)][bool]$OnWindows
    )

    if ($Requested -ne "AUTO") {
        if (-not (Test-GeneratorAvailable -GeneratorName $Requested -Help $CMakeHelp)) {
            throw "当前 CMake 未提供生成器 '$Requested'。可用生成器见 'cmake --help'。"
        }
        return $Requested
    }

    $NinjaOnPath = [bool](Get-Command ninja -ErrorAction SilentlyContinue)

    if ($OnWindows) {
        if (Test-GeneratorAvailable -GeneratorName $VsGenerator -Help $CMakeHelp) {
            return $VsGenerator
        }
        if ($NinjaOnPath -and (Test-GeneratorAvailable -GeneratorName "Ninja" -Help $CMakeHelp)) {
            return "Ninja"
        }
        throw "未找到可用的 CMake 生成器：既无 '$VsGenerator'，也未检测到 Ninja。请安装支持 VS 2026 的 CMake，或安装 ninja。"
    }

    if ($NinjaOnPath -and (Test-GeneratorAvailable -GeneratorName "Ninja" -Help $CMakeHelp)) {
        return "Ninja"
    }
    if (Test-GeneratorAvailable -GeneratorName "Unix Makefiles" -Help $CMakeHelp) {
        return "Unix Makefiles"
    }
    throw "未找到可用的 CMake 生成器：既未检测到 Ninja，也无 'Unix Makefiles'。请安装 ninja 或 make。"
}

function Get-CachedGenerator {
    param([Parameter(Mandatory)][string]$Directory)

    $Cache = Join-Path $Directory "CMakeCache.txt"
    if (-not (Test-Path $Cache -PathType Leaf)) {
        return $null
    }

    $Line = Get-Content $Cache |
        Where-Object { $_ -like "CMAKE_GENERATOR:INTERNAL=*" } |
        Select-Object -First 1

    if (-not $Line) {
        return $null
    }

    return ($Line -split "=", 2)[1]
}

function ConvertTo-ComparablePath {
    param([Parameter(Mandatory)][string]$Path)

    $FullPath = [System.IO.Path]::GetFullPath($Path)
    $PathRoot = [System.IO.Path]::GetPathRoot($FullPath)
    if ($FullPath.Length -eq $PathRoot.Length) {
        return $FullPath
    }

    $TrimChars = [char[]]@(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    return $FullPath.TrimEnd($TrimChars)
}

function Test-PathEqual {
    param(
        [Parameter(Mandatory)][string]$Left,
        [Parameter(Mandatory)][string]$Right,
        [Parameter(Mandatory)][bool]$OnWindows
    )

    $Comparison = if ($OnWindows) {
        [System.StringComparison]::OrdinalIgnoreCase
    }
    else {
        [System.StringComparison]::Ordinal
    }
    return [string]::Equals(
        (ConvertTo-ComparablePath $Left),
        (ConvertTo-ComparablePath $Right),
        $Comparison
    )
}

function Test-PathWithin {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Directory,
        [Parameter(Mandatory)][bool]$OnWindows
    )

    $Comparison = if ($OnWindows) {
        [System.StringComparison]::OrdinalIgnoreCase
    }
    else {
        [System.StringComparison]::Ordinal
    }
    $Path = ConvertTo-ComparablePath $Path
    $Directory = ConvertTo-ComparablePath $Directory
    $Separator = [System.IO.Path]::DirectorySeparatorChar
    return $Path.StartsWith("$Directory$Separator", $Comparison)
}

function Assert-SafeBuildDirectoryRemoval {
    param(
        [Parameter(Mandatory)][string]$Directory,
        [Parameter(Mandatory)][string]$ProjectRoot,
        [Parameter(Mandatory)][bool]$OnWindows
    )

    $Directory = ConvertTo-ComparablePath $Directory
    $ProjectRoot = ConvertTo-ComparablePath $ProjectRoot
    $FileSystemRoot = [System.IO.Path]::GetPathRoot($Directory)
    $UserHome = [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)

    if (
        (Test-PathEqual $Directory $FileSystemRoot $OnWindows) -or
        (Test-PathEqual $Directory $ProjectRoot $OnWindows) -or
        (-not [string]::IsNullOrWhiteSpace($UserHome) -and
            (Test-PathEqual $Directory $UserHome $OnWindows)) -or
        (Test-PathWithin $ProjectRoot $Directory $OnWindows)
    ) {
        throw "拒绝删除不安全的构建目录：$Directory"
    }

    $ProjectBuildRoot = Join-Path $ProjectRoot "build"
    if (
        (Test-PathEqual $Directory $ProjectBuildRoot $OnWindows) -or
        (Test-PathWithin $Directory $ProjectBuildRoot $OnWindows)
    ) {
        return
    }

    $CachePath = Join-Path $Directory "CMakeCache.txt"
    $CMakeFilesPath = Join-Path $Directory "CMakeFiles"
    $HomeLine = if (Test-Path $CachePath -PathType Leaf) {
        Get-Content -LiteralPath $CachePath |
            Where-Object { $_ -like "CMAKE_HOME_DIRECTORY:INTERNAL=*" } |
            Select-Object -First 1
    }
    $CachedProjectRoot = if ($HomeLine) { ($HomeLine -split "=", 2)[1] } else { "" }
    if (
        -not (Test-Path $CMakeFilesPath -PathType Container) -or
        [string]::IsNullOrWhiteSpace($CachedProjectRoot) -or
        -not (Test-PathEqual $CachedProjectRoot $ProjectRoot $OnWindows)
    ) {
        throw (
            "拒绝删除未经确认的外部目录：$Directory。" +
            "外部构建目录必须包含属于当前项目的 CMakeCache.txt 和 CMakeFiles。"
        )
    }
}

Export-ModuleMember -Function `
    Assert-Command,
    Get-CMakeVersion,
    Get-CMakeGeneratorHelp,
    Test-GeneratorAvailable,
    Resolve-Generator,
    Get-CachedGenerator,
    Test-PathEqual,
    Test-PathWithin,
    Assert-SafeBuildDirectoryRemoval `
    -Variable VsGenerator
