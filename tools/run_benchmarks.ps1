[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Config = "Release",

    [string]$BuildDir = "build/benchmarks",

    [string]$OutputDir = "",

    [ValidateSet("AUTO", "ON", "OFF")]
    [string]$Cuda = "AUTO",

    [ValidateSet("AUTO", "ON", "OFF")]
    [string]$Mkl = "AUTO",

    [ValidateSet("ON", "OFF")]
    [string]$ExperimentPresentation = "OFF",

    [switch]$IncludeLargeKnn,

    [switch]$SkipNative,

    [switch]$SkipPython,

    [switch]$Clean,

    # CMake 生成器。AUTO：按操作系统探测（Windows 优先 VS 2026，否则 Ninja；
    # Unix 优先 Ninja，否则 Unix Makefiles）。也可显式指定，如 "Ninja"。
    [string]$Generator = "AUTO",

    # native benchmark 单次执行的超时（秒）。超时则终止进程并抛错；非正数表示不设。
    [int]$Timeout = 600
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptVersion = "1.3"
$MinimumCMakeVersion = [Version]"3.24"

Import-Module (Join-Path $PSScriptRoot "cmake-runner-common.psm1")

function Write-Utf8NoBom {
    [CmdletBinding(DefaultParameterSetName = 'Lines')]
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory, ParameterSetName = 'Lines')]$Lines,
        [Parameter(Mandatory, ParameterSetName = 'Text')][string]$Text
    )

    # 统一用 .NET 无 BOM UTF8 写入，避免 Set-Content -Encoding UTF8 在
    # Windows PowerShell 5.1 下写入 BOM 破坏 JSONL/CSV/JSON 下游解析。
    $Encoding = [System.Text.UTF8Encoding]::new($false)
    if ($PSCmdlet.ParameterSetName -eq 'Text') {
        [System.IO.File]::WriteAllText($Path, $Text, $Encoding)
    }
    else {
        [System.IO.File]::WriteAllLines($Path, [string[]]$Lines, $Encoding)
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory)][string]$Description,
        [Parameter(Mandatory)][scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Description" -ForegroundColor Cyan

    # 流到控制台 + 临时文件（流式，不把长构建输出全量驻留内存）。
    # try/catch 捕获 native 命令自身在 EAP=Stop 下抛的终止错误
    # （pwsh 7.4+ $PSNativeCommandUseErrorActionPreference=$true 的非零退出、
    #  PS 5.1 的 NativeCommandError），否则会绕过下面的失败日志落盘与路径上报。
    $TmpLog = [System.IO.Path]::GetTempFileName()
    $NativeError = $null
    try {
        try {
            & $Command 2>&1 | Tee-Object -FilePath $TmpLog
        }
        catch {
            $NativeError = $_
        }
        $ExitCode = $LASTEXITCODE

        if ($NativeError -or $ExitCode -ne 0) {
            $SafeName = $Description -replace '[^A-Za-z0-9_.-]', '_'
            $LogPath = Join-Path $ResolvedOutputDir "failed_$SafeName.log"
            Move-Item $TmpLog $LogPath -Force
            $Reason = if ($NativeError) {
                "$($NativeError.Exception.Message)"
            }
            else {
                "退出码：$ExitCode"
            }
            throw "$Description 失败（$Reason）；输出：$LogPath"
        }
    }
    finally {
        Remove-Item $TmpLog -ErrorAction SilentlyContinue
    }
}

function Invoke-ToUtf8File {
    param(
        [Parameter(Mandatory)][string]$Description,
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Description" -ForegroundColor Cyan

    # stdout 落到 $Path（结果文件），stderr 单独到临时文件用于失败诊断（不混进结果）。
    # 失败时把 stdout+stderr 写到 failed_*.log（不写残缺的 $Path，避免下游 glob 误当有效结果）。
    $StderrFile = [System.IO.Path]::GetTempFileName()
    $NativeError = $null
    try {
        try {
            $Lines = @(& $Command 2>$StderrFile)
        }
        catch {
            $NativeError = $_
        }
        $ExitCode = $LASTEXITCODE

        if ($NativeError -or $ExitCode -ne 0) {
            $StderrText = if (Test-Path $StderrFile -PathType Leaf) {
                Get-Content $StderrFile -Raw
            }
            else {
                ""
            }
            $SafeName = $Description -replace '[^A-Za-z0-9_.-]', '_'
            $LogPath = Join-Path $ResolvedOutputDir "failed_$SafeName.log"
            $AllLines = @($Lines) +
                @($StderrText -split "`r?`n" | Where-Object { $_ })
            Write-Utf8NoBom -Path $LogPath -Lines $AllLines
            $Reason = if ($NativeError) {
                "$($NativeError.Exception.Message)"
            }
            else {
                "退出码：$ExitCode"
            }
            throw "$Description 失败（$Reason）；输出：$LogPath"
        }

        Write-Utf8NoBom -Path $Path -Lines $Lines
    }
    finally {
        Remove-Item $StderrFile -ErrorAction SilentlyContinue
    }
}

function Initialize-CMakeFileApiQuery {
    param([Parameter(Mandatory)][string]$Directory)

    $QueryDir = Join-Path `
        $Directory `
        ".cmake\api\v1\query\client-run-benchmarks"
    New-Item -ItemType Directory -Force -Path $QueryDir | Out-Null

    $QueryFile = Join-Path $QueryDir "codemodel-v2"
    if (-not (Test-Path $QueryFile -PathType Leaf)) {
        [System.IO.File]::WriteAllText($QueryFile, "")
    }
}

function Get-CMakeBenchmarkTargets {
    param(
        [Parameter(Mandatory)][string]$Directory,
        [Parameter(Mandatory)][string]$Configuration
    )

    $ReplyDir = Join-Path $Directory ".cmake\api\v1\reply"
    $idxFile = Get-ChildItem `
        -Path $ReplyDir `
        -Filter "index-*.json" `
        -File `
        -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1

    if (-not $idxFile) {
        throw (
            "CMake File API 没有生成 codemodel 回复。" +
            "请删除构建目录后重新运行脚本。"
        )
    }

    $idx = Get-Content $idxFile.FullName -Raw | ConvertFrom-Json
    $CodeModelObject = @($idx.objects) |
        Where-Object { $_.kind -eq "codemodel" } |
        Select-Object -First 1

    if (-not $CodeModelObject) {
        throw "CMake File API 回复中没有 codemodel 对象。"
    }

    $CodeModelPath = Join-Path $ReplyDir $CodeModelObject.jsonFile
    $CodeModel = Get-Content $CodeModelPath -Raw | ConvertFrom-Json
    $Configurations = @($CodeModel.configurations)
    $Selected = $Configurations |
        Where-Object { $_.name -eq $Configuration } |
        Select-Object -First 1

    # 单配置生成器的 codemodel 可能只有一个配置且名称为空。
    if (-not $Selected -and $Configurations.Count -eq 1) {
        $Selected = $Configurations[0]
    }
    if (-not $Selected) {
        $Names = @($Configurations | ForEach-Object { $_.name }) -join ", "
        throw (
            "CMake codemodel 中找不到配置 '$Configuration'。" +
            "可用配置：$Names"
        )
    }

    $BenchmarkDirectoryIdxs = [System.Collections.Generic.HashSet[int]]::new()
    for ($i = 0; $i -lt @($Selected.directories).Count; $i++) {
        $SourcePath = [string]$Selected.directories[$i].source
        $Normalized = ($SourcePath -replace '\\', '/').Trim('/')
        if ($Normalized -eq "cpp/benchmarks") {
            [void]$BenchmarkDirectoryIdxs.Add($i)
        }
    }

    if ($BenchmarkDirectoryIdxs.Count -eq 0) {
        throw "CMake codemodel 中没有找到 cpp/benchmarks 目录。"
    }

    $Targets = [System.Collections.Generic.List[string]]::new()
    foreach ($TargetReference in @($Selected.targets)) {
        if (-not $BenchmarkDirectoryIdxs.Contains(
            [int]$TargetReference.directoryIndex
        )) {
            continue
        }

        $TargetPath = Join-Path $ReplyDir $TargetReference.jsonFile
        $Target = Get-Content $TargetPath -Raw | ConvertFrom-Json
        if ($Target.type -eq "EXECUTABLE") {
            $Targets.Add([string]$Target.name)
        }
    }

    $Result = @($Targets | Sort-Object -Unique)
    if ($Result.Count -eq 0) {
        throw "cpp/benchmarks 中没有发现可执行 benchmark target。"
    }
    return $Result
}

function Get-CTestBenchmarkTests {
    param(
        [Parameter(Mandatory)][string]$Directory,
        [Parameter(Mandatory)][string]$Configuration,
        [switch]$MultiConfig
    )

    $ListArgs = @(
        "--test-dir", $Directory,
        "-N",
        "-L", "benchmark",
        "--show-only=json-v1"
    )
    if ($MultiConfig) {
        $ListArgs += @("-C", $Configuration)
    }

    # stdout 只承载 JSON，stderr 单独到临时文件（避免 ctest 告警行混进 ConvertFrom-Json）。
    $StderrFile = [System.IO.Path]::GetTempFileName()
    $NativeError = $null
    try {
        try {
            $Output = @(& ctest @ListArgs 2>$StderrFile)
        }
        catch {
            $NativeError = $_
        }
        $ExitCode = $LASTEXITCODE
        $StderrText = if (Test-Path $StderrFile -PathType Leaf) {
            Get-Content $StderrFile -Raw
        }
        else {
            ""
        }

        if ($NativeError -or $ExitCode -ne 0) {
            $Reason = if ($NativeError) {
                "$($NativeError.Exception.Message)"
            }
            else {
                "退出码：$ExitCode"
            }
            throw (
                "无法枚举 native benchmark CTest（$Reason）。`n" +
                "stdout: " + ($Output -join [Environment]::NewLine) + "`n" +
                "stderr: $StderrText"
            )
        }

        try {
            $Info = ($Output -join [Environment]::NewLine) | ConvertFrom-Json
        }
        catch {
            throw (
                "无法解析 CTest benchmark 列表：`n" +
                "stdout: " + ($Output -join [Environment]::NewLine) + "`n" +
                "stderr: $StderrText"
            )
        }
    }
    finally {
        Remove-Item $StderrFile -ErrorAction SilentlyContinue
    }

    $Tests = @($Info.tests)
    if ($Tests.Count -eq 0) {
        throw (
            "没有发现带 benchmark 标签的 CTest。" +
            "新增 native benchmark 时必须使用 add_test 注册并设置 LABELS benchmark。"
        )
    }

    foreach ($Test in $Tests) {
        if (-not $Test.PSObject.Properties["command"]) {
            throw (
                "CTest '$($Test.name)' 没有可执行命令。" +
                "通常表示对应 benchmark target 尚未编译。"
            )
        }
    }
    return $Tests
}

function Get-CTestPropertyValue {
    param(
        [Parameter(Mandatory)]$Test,
        [Parameter(Mandatory)][string]$Name
    )

    if (-not $Test.PSObject.Properties["properties"]) {
        return $null
    }

    $Property = @($Test.properties) |
        Where-Object { $_.name -eq $Name } |
        Select-Object -First 1
    if (-not $Property) {
        return $null
    }
    return $Property.value
}

function Invoke-CTestBenchmarkToUtf8File {
    param(
        [Parameter(Mandatory)]$Test,
        [Parameter(Mandatory)][string]$Path,
        [int]$TimeoutSeconds = 0
    )

    $CommandParts = @($Test.command)
    if ($CommandParts.Count -eq 0) {
        throw "CTest '$($Test.name)' 没有 command。"
    }

    $Executable = [string]$CommandParts[0]
    $Arguments = @()
    if ($CommandParts.Count -gt 1) {
        $Arguments = @($CommandParts[1..($CommandParts.Count - 1)])
    }

    $WorkingDirectory = Get-CTestPropertyValue `
        -Test $Test `
        -Name "WORKING_DIRECTORY"
    if ([string]::IsNullOrWhiteSpace([string]$WorkingDirectory)) {
        $WorkingDirectory = Split-Path -Parent $Executable
    }

    $SavedEnvironment = @{}
    $RememberEnvironment = {
        param([string]$VariableName)
        if (-not $SavedEnvironment.ContainsKey($VariableName)) {
            $SavedEnvironment[$VariableName] = `
                [Environment]::GetEnvironmentVariable(
                    $VariableName,
                    "Process"
                )
        }
    }

    $EnvironmentValues = Get-CTestPropertyValue `
        -Test $Test `
        -Name "ENVIRONMENT"
    foreach ($Entry in @($EnvironmentValues)) {
        if ([string]$Entry -match '^([^=]+)=(.*)$') {
            $Name = $Matches[1]
            & $RememberEnvironment $Name
            [Environment]::SetEnvironmentVariable(
                $Name,
                $Matches[2],
                "Process"
            )
        }
    }

    # 属性未设置时 Get-CTestPropertyValue 返回 $null；@($null) 是单元素数组，
    # 会让下面的 -notmatch 命中空 entry 而误抛（Linux/MKL-OFF 等无该属性的场景）。
    # 在此过滤掉空 entry，未设置时得到空数组、foreach 不迭代。
    $EnvironmentModifications = @(
        Get-CTestPropertyValue -Test $Test -Name "ENVIRONMENT_MODIFICATION" |
            Where-Object { $_ }
    )
    foreach ($Entry in @($EnvironmentModifications)) {
        if ([string]$Entry -notmatch '^([^=]+)=([^:]+):(.*)$') {
            throw (
                "无法解析 CTest ENVIRONMENT_MODIFICATION：$Entry"
            )
        }

        $Name = $Matches[1]
        $Operation = $Matches[2]
        $Value = $Matches[3]
        & $RememberEnvironment $Name

        $Current = [Environment]::GetEnvironmentVariable(
            $Name,
            "Process"
        )
        if ($null -eq $Current) {
            $Current = ""
        }

        switch ($Operation) {
            "reset" {
                $NewValue = $SavedEnvironment[$Name]
            }
            "set" {
                $NewValue = $Value
            }
            "unset" {
                $NewValue = $null
            }
            "string_append" {
                $NewValue = "$Current$Value"
            }
            "string_prepend" {
                $NewValue = "$Value$Current"
            }
            "path_list_append" {
                $Separator = [System.IO.Path]::PathSeparator
                $NewValue = if ([string]::IsNullOrEmpty($Current)) {
                    $Value
                }
                else {
                    "$Current$Separator$Value"
                }
            }
            "path_list_prepend" {
                $Separator = [System.IO.Path]::PathSeparator
                $NewValue = if ([string]::IsNullOrEmpty($Current)) {
                    $Value
                }
                else {
                    "$Value$Separator$Current"
                }
            }
            "cmake_list_append" {
                $NewValue = if ([string]::IsNullOrEmpty($Current)) {
                    $Value
                }
                else {
                    "$Current;$Value"
                }
            }
            "cmake_list_prepend" {
                $NewValue = if ([string]::IsNullOrEmpty($Current)) {
                    $Value
                }
                else {
                    "$Value;$Current"
                }
            }
            default {
                throw (
                    "不支持的 CTest 环境修改操作：$Operation"
                )
            }
        }

        [Environment]::SetEnvironmentVariable(
            $Name,
            $NewValue,
            "Process"
        )
    }

    Write-Host ""
    Write-Host (
        "==> 保存 native benchmark：$($Test.name)"
    ) -ForegroundColor Cyan

    # 用 Start-Process 启动 benchmark（继承当前进程已设置的环境变量、用
    # -WorkingDirectory 设定工作目录），stdout/stderr 重定向到临时文件，
    # 再按超时 WaitForExit；超时则 Kill 进程，避免挂死的 benchmark 无限阻塞。
    $StdoutFile = [System.IO.Path]::GetTempFileName()
    $StderrFile = [System.IO.Path]::GetTempFileName()
    $ExitCode = 0
    $TimedOut = $false
    $StderrText = ""
    try {
        $StartArgs = @{
            FilePath               = $Executable
            WorkingDirectory       = [string]$WorkingDirectory
            NoNewWindow            = $true
            PassThru               = $true
            Wait                   = $false
            RedirectStandardOutput = $StdoutFile
            RedirectStandardError  = $StderrFile
        }
        if ($Arguments.Count -gt 0) {
            $StartArgs.ArgumentList = $Arguments
        }
        $Process = Start-Process @StartArgs

        if ($TimeoutSeconds -gt 0) {
            $Exited = $Process.WaitForExit($TimeoutSeconds * 1000)
            if ($Exited) {
                $Process.WaitForExit()
            }
        }
        else {
            $Process.WaitForExit()
            $Exited = $true
        }

        if (-not $Exited) {
            if (-not $Process.HasExited) {
                $Process.Kill()
            }
            $Process.WaitForExit()
            $TimedOut = $true
        }
        else {
            $ExitCode = $Process.ExitCode
        }
        # 两条路径都读 stderr：超时分支恰恰最需要挂死前的 stderr 诊断。
        if (Test-Path $StderrFile -PathType Leaf) {
            $StderrText = Get-Content $StderrFile -Raw
        }

        # 落盘 stdout（含超时被杀时的部分输出）。
        $Lines = @()
        if (Test-Path $StdoutFile -PathType Leaf) {
            $Lines = @(Get-Content $StdoutFile)
        }
        Write-Utf8NoBom -Path $Path -Lines $Lines
    }
    finally {
        Remove-Item $StdoutFile, $StderrFile -ErrorAction SilentlyContinue
        foreach ($Name in $SavedEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable(
                $Name,
                $SavedEnvironment[$Name],
                "Process"
            )
        }
    }

    if ($TimedOut) {
        $StderrSuffix = if ([string]::IsNullOrWhiteSpace($StderrText)) {
            ""
        }
        else {
            "；stderr：$StderrText"
        }
        throw (
            "native benchmark '$($Test.name)' 超时" +
            "（>${TimeoutSeconds}s），已终止；部分输出：$Path$StderrSuffix"
        )
    }
    if ($ExitCode -ne 0) {
        $StderrSuffix = if ([string]::IsNullOrWhiteSpace($StderrText)) {
            ""
        }
        else {
            "；stderr：$StderrText"
        }
        throw (
            "native benchmark '$($Test.name)' 失败，" +
            "退出码：$ExitCode；输出：$Path$StderrSuffix"
        )
    }
}

function Find-IntelOpenMpRuntime {
    # 跨平台定位 Intel OMP 运行时：Windows 找 libiomp5md.dll，Unix 找 libiomp5.so。
    if ($OnWindows) {
        $RuntimeName = "libiomp5md.dll"
        $SubDirs = @(
            "compiler\latest\bin",
            "compiler\latest\windows\redist\intel64_win\compiler",
            "compiler\latest\redist\intel64_win\compiler"
        )
    }
    else {
        $RuntimeName = "libiomp5.so"
        $SubDirs = @(
            "compiler/latest/lib",
            "compiler/latest/lib/intel64_lin",
            "compiler/latest/linux/compiler/lib/intel64_lin"
        )
    }

    $Roots = [System.Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($env:ONEAPI_ROOT)) {
        $Roots.Add($env:ONEAPI_ROOT)
    }
    if ($OnWindows) {
        $ProgramFilesX86 = [Environment]::GetFolderPath(
            [Environment+SpecialFolder]::ProgramFilesX86
        )
        if (-not [string]::IsNullOrWhiteSpace($ProgramFilesX86)) {
            $Roots.Add((Join-Path $ProgramFilesX86 "Intel\oneAPI"))
        }
    }
    else {
        $Roots.Add("/opt/intel/oneapi")
    }

    # 1) 显式候选路径
    foreach ($Root in ($Roots | Select-Object -Unique)) {
        foreach ($Sub in $SubDirs) {
            $Candidate = Join-Path $Root (Join-Path $Sub $RuntimeName)
            if (Test-Path $Candidate -PathType Leaf) {
                return (Get-Item $Candidate)
            }
        }
    }

    # 2) Unix：用 ldconfig 缓存定位
    if (-not $OnWindows) {
        $LdPattern = '\b' + [regex]::Escape($RuntimeName) + '\b.*=>\s*(\S+)'
        foreach ($Line in @(ldconfig -p 2>$null)) {
            if ([string]$Line -match $LdPattern) {
                $SoPath = $Matches[1]
                if (Test-Path $SoPath -PathType Leaf) {
                    return (Get-Item $SoPath)
                }
            }
        }
    }

    # 3) Windows 兜底：限定深度的递归搜索（避免无界 -Recurse 扫整棵 oneAPI 树）。
    if ($OnWindows) {
        foreach ($Root in ($Roots | Select-Object -Unique)) {
            $CompilerRoot = Join-Path $Root "compiler"
            if (-not (Test-Path $CompilerRoot -PathType Container)) {
                continue
            }

            $Match = Get-ChildItem `
                -Path $CompilerRoot `
                -Filter $RuntimeName `
                -File `
                -Recurse `
                -Depth 5 `
                -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTimeUtc -Descending |
                Select-Object -First 1

            if ($Match) {
                return $Match
            }
        }
    }

    return $null
}

function Test-PythonPackage {
    param([Parameter(Mandatory)][string]$StderrPath)

    $Probe = @(
        "import json",
        "import neurale._native as n",
        "b=n.build_info()",
        "c=getattr(n,'cuda',None)",
        "available=bool(c and c.info().get('available',False))",
        "print(json.dumps({'build_type':b.get('build_type','unknown'),'cuda_available':available}))"
    ) -join ";"

    # 分流捕获：stdout 只承载探测 JSON，stderr 单独落到文件。
    # 既避免导入时的 warning/banner 污染 JSON 解析，也把 native 模块
    # 导入时的 stderr 作为诊断 artifact 留在结果目录里。
    $Stdout = @(& python -c $Probe 2>$StderrPath)
    $ExitCode = $LASTEXITCODE
    $Stderr = if (Test-Path $StderrPath -PathType Leaf) {
        Get-Content $StderrPath -Raw
    }
    else {
        ""
    }

    if ($ExitCode -ne 0) {
        throw (
            "无法导入已安装的 PyNeurale native 包。`n" +
            "请先在项目根目录执行 Release editable 安装，例如：`n" +
            "python -m pip install -e . " +
            "--config-settings=cmake.build-type=Release`n`n" +
            "stdout: " + ($Stdout -join [Environment]::NewLine) + "`n" +
            "stderr ($StderrPath): $Stderr"
        )
    }

    # 从 stdout 里挑出 JSON 行，而不是假定最后一行就是 JSON
    # （native 模块可能在导入时往 stdout 打 banner）。
    $JsonLine = @($Stdout) |
        Where-Object {
            -not [string]::IsNullOrWhiteSpace([string]$_) -and
            ([string]$_).TrimStart().StartsWith('{')
        } |
        Select-Object -Last 1
    if (-not $JsonLine) {
        throw (
            "PyNeurale 环境探测没有输出 JSON 行。`n" +
            "stdout: " + ($Stdout -join [Environment]::NewLine) + "`n" +
            "stderr ($StderrPath): $Stderr"
        )
    }

    try {
        $Info = $JsonLine | ConvertFrom-Json
    }
    catch {
        throw (
            "无法解析 PyNeurale 环境探测 JSON：$JsonLine`n" +
            "stdout: " + ($Stdout -join [Environment]::NewLine) + "`n" +
            "stderr ($StderrPath): $Stderr"
        )
    }

    return @{
        BuildType = [string]$Info.build_type
        CudaAvailable = [bool]$Info.cuda_available
    }
}

function Get-PythonPackageVersion {
    $Probe = @(
        "import importlib.metadata as metadata",
        "import json",
        "from packaging.version import Version",
        "version=metadata.version('pyneurale')",
        "print(json.dumps({'full':version,'base':Version(version).base_version}))"
    ) -join ";"
    $Output = @(& python -c $Probe 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw (
            "无法读取已安装的 PyNeurale 版本。请先安装当前项目。`n" +
            ($Output -join [Environment]::NewLine)
        )
    }

    try {
        $Info = ($Output -join [Environment]::NewLine) | ConvertFrom-Json
    }
    catch {
        throw "无法解析已安装的 PyNeurale 版本：$($Output -join ' ')"
    }
    return @{
        FullVersion = [string]$Info.full
        BaseVersion = [string]$Info.base
    }
}

if ($SkipNative -and $SkipPython) {
    throw "不能同时指定 -SkipNative 和 -SkipPython；至少需要运行一类 benchmark。"
}

Assert-Command "cmake"
Assert-Command "ctest"
Assert-Command "python"

# 检测操作系统（兼容 Windows PowerShell 5.1 与 pwsh 7+）。
$OnWindows = if ($PSVersionTable.PSEdition -eq "Desktop") { $true } else { $IsWindows }

$CMakeVersion = Get-CMakeVersion
$CMakeHelp = Get-CMakeGeneratorHelp
$ResolvedGenerator = Resolve-Generator `
    -Requested $Generator `
    -CMakeHelp $CMakeHelp `
    -OnWindows $OnWindows

if ($CMakeVersion -lt $MinimumCMakeVersion) {
    throw "当前 CMake 版本 $CMakeVersion 低于最低要求 $MinimumCMakeVersion。"
}

# 多配置生成器（VS / Ninja Multi-Config / Xcode）在 build/ctest 用 --config/-C；
# 单配置生成器（Ninja / Unix Makefiles）改在 configure 传 CMAKE_BUILD_TYPE。
$IsMultiConfig = $ResolvedGenerator -match 'Visual Studio|Ninja Multi-Config|Xcode'
$IsVsGenerator = $ResolvedGenerator -like "Visual Studio *"

$ScriptDir = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ScriptDir)) {
    $ScriptDir = (Get-Location).Path
}

# 本脚本应放在 pyneurale\tools。
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$RootCMakeLists = Join-Path $ProjectRoot "CMakeLists.txt"

if (-not (Test-Path $RootCMakeLists -PathType Leaf)) {
    throw (
        "未在预期的项目根目录找到 CMakeLists.txt：$RootCMakeLists。`n" +
        "请将脚本放在 pyneurale\tools 目录下。"
    )
}

Push-Location $ProjectRoot
try {

if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    $ResolvedBuildDir = [System.IO.Path]::GetFullPath($BuildDir)
}
else {
    $ResolvedBuildDir = [System.IO.Path]::GetFullPath(
        (Join-Path $ProjectRoot $BuildDir)
    )
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $ResolvedOutputDir = Join-Path `
        $ProjectRoot `
        "build\benchmark-results\$Timestamp"
}
elseif ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $ResolvedOutputDir = [System.IO.Path]::GetFullPath($OutputDir)
}
else {
    $ResolvedOutputDir = [System.IO.Path]::GetFullPath(
        (Join-Path $ProjectRoot $OutputDir)
    )
}

if (
    (Test-PathEqual $ResolvedOutputDir $ResolvedBuildDir $OnWindows) -or
    (Test-PathWithin $ResolvedOutputDir $ResolvedBuildDir $OnWindows)
) {
    throw "输出目录不能等于或位于构建目录内：$ResolvedOutputDir"
}

New-Item -ItemType Directory -Force -Path $ResolvedOutputDir | Out-Null

if ($Clean -and (Test-Path $ResolvedBuildDir)) {
    Assert-SafeBuildDirectoryRemoval `
        -Directory $ResolvedBuildDir `
        -ProjectRoot $ProjectRoot `
        -OnWindows $OnWindows
    if ($PSCmdlet.ShouldProcess($ResolvedBuildDir, "删除旧构建目录（-Clean）")) {
        Write-Host "删除旧 benchmark 构建目录：$ResolvedBuildDir" `
            -ForegroundColor Yellow
        Remove-Item -Recurse -Force $ResolvedBuildDir
    }
}

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

Write-Host ""
Write-Host "PyNeurale Benchmark Runner $ScriptVersion" -ForegroundColor Green
Write-Host "  项目根目录： $ProjectRoot"
Write-Host "  输出目录：   $ResolvedOutputDir"
Write-Host "  CMake：      $CMakeVersion"
Write-Host "  生成器：     $ResolvedGenerator"
Write-Host "  平台：       $(if ($OnWindows) { 'Windows' } else { 'Unix' })"
Write-Host "  配置：       $Config"
Write-Host "  CUDA 配置：  $Cuda"
Write-Host "  MKL 配置：   $Mkl"

# -WhatIf 时跳过下面的实际执行——含两个有副作用的探测：
#   Find-IntelOpenMpRuntime 会修改 $env:PATH，Test-PythonPackage 会 import native 模块。
# 守卫放在探测之前，确保 -WhatIf 是干净 dry-run。-Confirm 时在此确认一次。
if (-not $PSCmdlet.ShouldProcess(
        "PyNeurale benchmark",
        "配置/编译/运行 native + Python benchmark 并收集结果")) {
    Write-Host ""
    Write-Host "（-WhatIf：跳过实际执行）" -ForegroundColor Yellow
    return
}

$ProjectVersionInfo = Get-PythonPackageVersion
Write-Host "  PyNeurale：  $($ProjectVersionInfo.FullVersion)"

$IompRuntime = $null
if ($Mkl -ne "OFF") {
    $IompRuntime = Find-IntelOpenMpRuntime
    if ($IompRuntime) {
        $IompDir = $IompRuntime.Directory.FullName
        $PathSep = [System.IO.Path]::PathSeparator
        if (@($env:PATH -split $PathSep) -notcontains $IompDir) {
            $env:PATH = "$IompDir$PathSep$env:PATH"
        }
    }
    elseif ($Mkl -eq "ON") {
        $RuntimeName = if ($OnWindows) { "libiomp5md.dll" } else { "libiomp5.so" }
        throw (
            "MKL 被强制启用，但没有找到 $RuntimeName。`n" +
            "请安装 Intel oneAPI Compiler Runtime，或使用 -Mkl OFF。"
        )
    }
}

$PythonInfo = $null
if (-not $SkipPython) {
    $PythonProbeStderr = Join-Path $ResolvedOutputDir "python_probe_stderr.log"
    $PythonInfo = Test-PythonPackage -StderrPath $PythonProbeStderr
    if ($PythonInfo.BuildType -ne "Release") {
        Write-Warning (
            "当前已安装 PyNeurale native build_type=" +
            "$($PythonInfo.BuildType)，性能比较建议使用 Release。"
        )
    }
}

if ($PythonInfo) {
    Write-Host "  Python build：$($PythonInfo.BuildType)"
    Write-Host "  CUDA 可用：  $($PythonInfo.CudaAvailable)"
}
if ($IompRuntime) {
    Write-Host "  Intel OMP：  $($IompRuntime.FullName)"
}

$NativeBenchmarkTargets = @()
$NativeBenchmarkTests = @()

if (-not $SkipNative) {
    $Pybind11Dir = (& python -m pybind11 --cmakedir).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($Pybind11Dir)) {
        throw (
            "无法确定 pybind11 CMake 目录。请执行：" +
            " python -m pip install pybind11"
        )
    }

    $ConfigureArgs = @(
        "-S", $ProjectRoot,
        "-B", $ResolvedBuildDir,
        "-G", $ResolvedGenerator,
        "-DNEURALE_BUILD_CPP_TESTS=ON",
        "-DNEURALE_BUILD_BENCHMARKS=ON",
        "-DNEURALE_ENABLE_CUDA=$Cuda",
        "-DNEURALE_ENABLE_MKL=$Mkl",
        "-DNEURALE_ENABLE_EXPERIMENT_PRESENTATION=$ExperimentPresentation",
        "-Dpybind11_DIR=$Pybind11Dir",
        "-DSKBUILD_PROJECT_VERSION=$($ProjectVersionInfo.BaseVersion)",
        "-DSKBUILD_PROJECT_VERSION_FULL=$($ProjectVersionInfo.FullVersion)"
    )

    if ($IsVsGenerator) {
        $ConfigureArgs += @("-A", "x64")
    }

    if (-not $IsMultiConfig) {
        $ConfigureArgs += @("-DCMAKE_BUILD_TYPE=$Config")
    }

    Initialize-CMakeFileApiQuery -Directory $ResolvedBuildDir

    Invoke-Checked "配置 native benchmark" {
        cmake @ConfigureArgs
    }

    $NativeBenchmarkTargets = @(
        Get-CMakeBenchmarkTargets `
            -Directory $ResolvedBuildDir `
            -Configuration $Config
    )
    Write-Host (
        "  Native targets：" + ($NativeBenchmarkTargets -join ", ")
    )

    Invoke-Checked "编译全部 native benchmark target" {
        $BuildArgs = @(
            "--build", $ResolvedBuildDir,
            "--parallel",
            "--target"
        ) + $NativeBenchmarkTargets
        if ($IsMultiConfig) {
            $BuildArgs += @("--config", $Config)
        }
        cmake @BuildArgs
    }

    $NativeBenchmarkTests = @(
        Get-CTestBenchmarkTests `
            -Directory $ResolvedBuildDir `
            -Configuration $Config `
            -MultiConfig:$IsMultiConfig
    )
    Write-Host (
        "  Native verify tests：" +
        (@($NativeBenchmarkTests | ForEach-Object { $_.name }) -join ", ")
    )

    foreach ($Test in $NativeBenchmarkTests) {
        $CommandParts = @($Test.command)
        $ExecutableName = [System.IO.Path]::GetFileNameWithoutExtension(
            [string]$CommandParts[0]
        )
        $SafeName = $ExecutableName -replace '[^A-Za-z0-9_.-]', '_'
        $NativeOutput = Join-Path `
            $ResolvedOutputDir `
            "native_$SafeName.jsonl"

        Invoke-CTestBenchmarkToUtf8File `
            -Test $Test `
            -Path $NativeOutput `
            -TimeoutSeconds $Timeout
    }
}

if (-not $SkipPython) {
    Invoke-Checked "Import benchmark" {
        python benchmarks/benchmark_import.py `
            --output (Join-Path $ResolvedOutputDir "import.jsonl")
    }

    Invoke-Checked "Model inference benchmark" {
        python benchmarks/models/benchmark_inference.py `
            --output (Join-Path $ResolvedOutputDir "models.jsonl")
    }

    Invoke-Checked "Resample benchmark" {
        python benchmarks/signal/benchmark_resample.py `
            --output (Join-Path $ResolvedOutputDir "signal.jsonl")
    }

    Invoke-Checked "LMP benchmark" {
        python benchmarks/features/benchmark_lmp.py `
            --output (Join-Path $ResolvedOutputDir "features.jsonl")
    }

    Invoke-Checked "FIR benchmark" {
        python benchmarks/signal/benchmark_fir.py `
            --output (Join-Path $ResolvedOutputDir "fir.csv")
    }

    Invoke-Checked "FFT benchmark" {
        python benchmarks/signal/benchmark_fft.py `
            --output (Join-Path $ResolvedOutputDir "fft.csv")
    }

    Invoke-Checked "IIR/SOS benchmark" {
        python benchmarks/signal/benchmark_iir.py `
            --output (Join-Path $ResolvedOutputDir "iir_sos.csv")
    }

    Invoke-Checked "DPSS benchmark" {
        python benchmarks/signal/benchmark_dpss.py `
            --output (Join-Path $ResolvedOutputDir "dpss.csv")
    }

    Invoke-Checked "M8 offline provenance benchmark" {
        python benchmarks/experiments/benchmark_provenance.py `
            --output (Join-Path $ResolvedOutputDir "m8_provenance.jsonl")
    }

    $KnnDevice = if ($PythonInfo.CudaAvailable) { "both" } else { "cpu" }
    Invoke-Checked "KNN standard benchmark" {
        python benchmarks/models/benchmark_knn.py `
            --suite standard `
            --device $KnnDevice `
            --output (Join-Path $ResolvedOutputDir "knn_standard.csv")
    }

    if ($IncludeLargeKnn) {
        if ($PythonInfo.CudaAvailable) {
            Invoke-Checked "KNN large CUDA benchmark" {
                python benchmarks/models/benchmark_knn.py `
                    --suite large `
                    --device cuda `
                    --output (Join-Path $ResolvedOutputDir "knn_large.csv")
            }
        }
        else {
            Write-Warning (
                "请求了 -IncludeLargeKnn，但 CUDA 不可用；" +
                "跳过 large KNN，避免极慢的 CPU exact KNN。"
            )
        }
    }

    if ($PythonInfo.CudaAvailable) {
        Invoke-Checked "KDE CPU/CUDA benchmark" {
            python benchmarks/models/benchmark_kde.py `
                --output (Join-Path $ResolvedOutputDir "kde.csv")
        }
    }
    else {
        Write-Warning (
            "CUDA 不可用，跳过 KDE benchmark；" +
            "当前 KDE 入口固定同时运行 CPU 和 CUDA。"
        )
    }

    Invoke-ToUtf8File `
        -Description "Stateful FIR benchmark" `
        -Path (Join-Path $ResolvedOutputDir "stateful_fir.csv") `
        -Command {
            python benchmarks/signal/benchmark_fir_realtime.py
        }

    Invoke-ToUtf8File `
        -Description "Stateful IIR/SOS benchmark" `
        -Path (Join-Path $ResolvedOutputDir "stateful_iir_sos.csv") `
        -Command {
            python benchmarks/signal/benchmark_iir_realtime.py
        }
}

$Manifest = [ordered]@{
    script_version = $ScriptVersion
    generated_at = (Get-Date).ToString("o")
    config = $Config
    generator = $ResolvedGenerator
    cmake_version = $CMakeVersion.ToString()
    cuda_option = $Cuda
    mkl_option = $Mkl
    experiment_presentation_option = $ExperimentPresentation
    pyneurale_version = $ProjectVersionInfo.FullVersion
    python_build_type = if ($PythonInfo) { $PythonInfo.BuildType } else { $null }
    cuda_available = if ($PythonInfo) {
        $PythonInfo.CudaAvailable
    }
    else {
        $null
    }
    include_large_knn = [bool]$IncludeLargeKnn
    native_benchmark_targets = @($NativeBenchmarkTargets)
    native_benchmark_tests = @(
        $NativeBenchmarkTests | ForEach-Object { $_.name }
    )
}

# 无 BOM UTF8 写入（Write-Utf8NoBom 统一了脚本里所有结果文件落盘的编码）。
$ManifestJson = $Manifest | ConvertTo-Json -Depth 4
Write-Utf8NoBom `
    -Path (Join-Path $ResolvedOutputDir "run_manifest.json") `
    -Text ($ManifestJson + [Environment]::NewLine)

Write-Host ""
Write-Host "Benchmark 批量运行完成。" -ForegroundColor Green
Write-Host "结果目录：$ResolvedOutputDir"
Get-ChildItem -Path $ResolvedOutputDir -File |
    Sort-Object Name |
    ForEach-Object {
        Write-Host "  $($_.Name)"
    }
}
finally {
    Pop-Location
}
