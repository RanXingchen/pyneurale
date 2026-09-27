#!/usr/bin/env bash
# Unix entry point: delegates to the cross-platform run_benchmarks.ps1 (requires pwsh).
# This file contains no build logic; it only saves Linux users from typing the pwsh
# command by hand.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v pwsh >/dev/null 2>&1; then
    echo "PowerShell (pwsh) is required but was not found. Please install it and retry:" >&2
    echo "  Ubuntu/Debian:  see https://learn.microsoft.com/powershell/scripting/install/install-ubuntu" >&2
    echo "  macOS (brew):  brew install --cask powershell" >&2
    exit 1
fi

exec pwsh "$script_dir/run_benchmarks.ps1" "$@"
