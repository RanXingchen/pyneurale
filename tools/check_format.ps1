# SPDX-FileCopyrightText: 2026 PyNeurale contributors
# SPDX-License-Identifier: MIT

[CmdletBinding()]
param(
    [string] $Python = "python"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$PythonCommand = Get-Command $Python -ErrorAction SilentlyContinue

if ($null -eq $PythonCommand) {
    throw "Python command not found: $Python"
}

Push-Location $Root

try {
    & $PythonCommand.Source -m pre_commit --version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "pre-commit is not installed for $($PythonCommand.Source). Install the dev dependencies for that environment."
    }

    & $PythonCommand.Source -m pre_commit run --all-files --show-diff-on-failure

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Files were reformatted or repository checks failed."
        Write-Host "Review the changes, then run this script again."
        exit $LASTEXITCODE
    }

    Write-Host "All formatting and repository hygiene checks passed."
}
finally {
    Pop-Location
}
