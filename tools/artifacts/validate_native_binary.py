# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Validate native exports and shared-library boundaries for a wheel."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

if __package__:
    from .artifact_profiles import PROFILE_CAPABILITIES, PROFILE_MODULES
else:
    from artifact_profiles import PROFILE_CAPABILITIES, PROFILE_MODULES

_BUNDLED_PRESENTATION_DEPENDENCIES = ("freetype", "glfw", "harfbuzz", "qt5", "qt6")
_CUDA_DEPENDENCIES = ("cudart", "libcuda", "nvcuda", "nvrtc")
_MKL_DEPENDENCIES = ("iomp", "mkl")
_BINARY_SUFFIXES = (".pyd", ".so")
_ELF_HOUSEKEEPING_EXPORTS = frozenset({"__bss_start", "_edata", "_end"})

_WINDOWS_CORE_DEPENDENCIES = (
    r"(?:api|ext)-ms-win-[a-z0-9._-]+\.dll",
    r"concrt\d+(?:_[a-z0-9_]+)?\.dll",
    r"kernel32\.dll",
    r"msvcp\d+(?:_[a-z0-9_]+)?\.dll",
    r"python\d+(?:_d)?\.dll",
    r"ucrtbase\.dll",
    r"vcomp\d+\.dll",
    r"vcruntime\d+(?:_[a-z0-9_]+)?\.dll",
)
_WINDOWS_PRESENTATION_DEPENDENCIES = (
    r"advapi32\.dll",
    r"comdlg32\.dll",
    r"dwmapi\.dll",
    r"gdi32\.dll",
    r"imm32\.dll",
    r"ole32\.dll",
    r"opengl32\.dll",
    r"setupapi\.dll",
    r"shell32\.dll",
    r"shcore\.dll",
    r"user32\.dll",
    r"version\.dll",
    r"winmm\.dll",
    r"ws2_32\.dll",
)
_WINDOWS_CUDA_DEPENDENCIES = (r"cudart64_\d+\.dll",)
_WINDOWS_MKL_DEPENDENCIES = (r"libiomp5md\.dll",)

_LINUX_CORE_DEPENDENCIES = (
    r"ld-linux-[a-z0-9_.-]+\.so(?:\.\d+)*",
    r"libatomic\.so(?:\.\d+)*",
    r"libc\.so(?:\.\d+)*",
    r"libdl\.so(?:\.\d+)*",
    r"libgcc_s\.so(?:\.\d+)*",
    r"libgomp\.so(?:\.\d+)*",
    r"libm\.so(?:\.\d+)*",
    r"libpthread\.so(?:\.\d+)*",
    r"libpython\d+\.\d+(?:[a-z]+)?\.so(?:\.\d+)*",
    r"librt\.so(?:\.\d+)*",
    r"libstdc\+\+\.so(?:\.\d+)*",
)
_LINUX_PRESENTATION_DEPENDENCIES = (
    r"libegl\.so(?:\.\d+)*",
    r"libgl\.so(?:\.\d+)*",
    r"libgldispatch\.so(?:\.\d+)*",
    r"libglx\.so(?:\.\d+)*",
    r"libopengl\.so(?:\.\d+)*",
    r"libx11\.so(?:\.\d+)*",
    r"libxau\.so(?:\.\d+)*",
    r"libxcb\.so(?:\.\d+)*",
    r"libxcursor\.so(?:\.\d+)*",
    r"libxdmcp\.so(?:\.\d+)*",
    r"libxext\.so(?:\.\d+)*",
    r"libxfixes\.so(?:\.\d+)*",
    r"libxi\.so(?:\.\d+)*",
    r"libxinerama\.so(?:\.\d+)*",
    r"libxrandr\.so(?:\.\d+)*",
    r"libxrender\.so(?:\.\d+)*",
)
_LINUX_CUDA_DEPENDENCIES = (r"libcudart\.so(?:\.\d+)*",)
_LINUX_MKL_DEPENDENCIES = (r"libiomp5(?:-[a-f0-9]+)?\.so(?:\.\d+)*",)


class NativeBinaryValidationError(RuntimeError):
    """A native extension widens its published binary boundary."""


def validate_audit_report(report: str, profile: str) -> None:
    """Reject cross-profile dependencies in an auditwheel/delvewheel report."""

    capabilities = PROFILE_CAPABILITIES.get(profile)
    if capabilities is None:
        raise NativeBinaryValidationError(f"unknown artifact profile: {profile!r}")
    forbidden = list(_BUNDLED_PRESENTATION_DEPENDENCIES)
    if "cuda" not in capabilities:
        forbidden.extend(_CUDA_DEPENDENCIES)
    if "mkl" not in capabilities:
        forbidden.extend(_MKL_DEPENDENCIES)
    lowered = report.casefold()
    found = [token for token in forbidden if token in lowered]
    if found:
        raise NativeBinaryValidationError(
            f"{profile} shared-library report contains forbidden dependencies: " + ", ".join(found)
        )


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return any(re.fullmatch(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def validate_binary_contract(
    *,
    module: str,
    profile: str,
    platform: str,
    exports: list[str],
    dependencies: list[str],
) -> None:
    """Validate one already-inspected extension module."""

    expected_modules = PROFILE_MODULES.get(profile)
    if expected_modules is None:
        raise NativeBinaryValidationError(f"unknown artifact profile: {profile!r}")
    if module not in expected_modules:
        raise NativeBinaryValidationError(
            f"{module} does not belong to the {profile} artifact profile"
        )
    capabilities = PROFILE_CAPABILITIES[profile]

    expected_export = f"PyInit_{module}"
    allowed_exports = {expected_export}
    if platform == "linux":
        allowed_exports.update(_ELF_HOUSEKEEPING_EXPORTS)
    elif platform != "windows":
        raise NativeBinaryValidationError(f"unsupported binary platform: {platform!r}")
    unexpected_exports = sorted(set(exports) - allowed_exports)
    if expected_export not in exports or unexpected_exports:
        raise NativeBinaryValidationError(
            f"{module} must export only {expected_export}; found {sorted(set(exports))}"
        )

    if platform == "windows":
        allowed_dependencies = list(_WINDOWS_CORE_DEPENDENCIES)
        if "presentation" in capabilities and module == "_native":
            allowed_dependencies.extend(_WINDOWS_PRESENTATION_DEPENDENCIES)
        if "cuda" in capabilities:
            allowed_dependencies.extend(_WINDOWS_CUDA_DEPENDENCIES)
        if "mkl" in capabilities and module == "_native":
            allowed_dependencies.extend(_WINDOWS_MKL_DEPENDENCIES)
    else:
        allowed_dependencies = list(_LINUX_CORE_DEPENDENCIES)
        if "presentation" in capabilities and module == "_native":
            allowed_dependencies.extend(_LINUX_PRESENTATION_DEPENDENCIES)
        if "cuda" in capabilities:
            allowed_dependencies.extend(_LINUX_CUDA_DEPENDENCIES)
        if "mkl" in capabilities and module == "_native":
            allowed_dependencies.extend(_LINUX_MKL_DEPENDENCIES)

    unexpected_dependencies = sorted(
        dependency
        for dependency in set(dependencies)
        if not _matches_any(
            re.sub(r"-[a-f0-9]{8}(?=\.so)", "", Path(dependency).name.casefold())
            if "mkl" in capabilities and platform == "linux"
            else Path(dependency).name.casefold(),
            tuple(allowed_dependencies),
        )
    )
    if unexpected_dependencies:
        raise NativeBinaryValidationError(
            f"{module} has dependencies outside the {profile}/{platform} boundary: "
            + ", ".join(unexpected_dependencies)
        )


def _run(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise NativeBinaryValidationError(
            f"binary inspection command failed ({completed.returncode}): "
            f"{' '.join(command)}\n{completed.stderr.strip()}"
        )
    return completed.stdout


def _find_dumpbin() -> str:
    direct = shutil.which("dumpbin")
    if direct:
        return direct

    installer = Path(os.environ.get("ProgramFiles(x86)", "")) / (
        "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    if installer.is_file():
        output = _run(
            [
                str(installer),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-find",
                r"VC\Tools\MSVC\**\bin\Hostx64\x64\dumpbin.exe",
            ]
        )
        candidates = [line.strip() for line in output.splitlines() if line.strip()]
        if candidates:
            return candidates[-1]
    raise NativeBinaryValidationError("dumpbin was not found in the Visual Studio toolchain")


def _parse_dumpbin_exports(report: str) -> list[str]:
    exports: list[str] = []
    pattern = re.compile(r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)", re.IGNORECASE)
    for line in report.splitlines():
        match = pattern.match(line)
        if match:
            exports.append(match.group(1))
    return exports


def _parse_dumpbin_dependencies(report: str) -> list[str]:
    dependencies: list[str] = []
    active = False
    for raw_line in report.splitlines():
        line = raw_line.strip()
        if line == "Image has the following dependencies:":
            active = True
            continue
        if active and line == "Summary":
            break
        if active and re.fullmatch(r"[A-Za-z0-9._+-]+\.dll", line, flags=re.IGNORECASE):
            dependencies.append(line)
    return dependencies


def _parse_nm_exports(report: str) -> list[str]:
    exports: list[str] = []
    for line in report.splitlines():
        fields = line.split()
        if len(fields) >= 2 and len(fields[1]) == 1 and fields[1].isalpha():
            exports.append(fields[0].split("@", 1)[0])
    return exports


def _parse_readelf_dependencies(report: str) -> list[str]:
    return re.findall(r"\(NEEDED\).*?\[([^]]+)]", report)


def _inspect_binary(binary: Path, platform: str) -> tuple[str, list[str], list[str]]:
    if platform == "windows":
        tool = _find_dumpbin()
        exports = _parse_dumpbin_exports(_run([tool, "/NOLOGO", "/EXPORTS", str(binary)]))
        dependencies = _parse_dumpbin_dependencies(
            _run([tool, "/NOLOGO", "/DEPENDENTS", str(binary)])
        )
        return tool, exports, dependencies
    if platform == "linux":
        nm = shutil.which("nm")
        readelf = shutil.which("readelf")
        if not nm or not readelf:
            raise NativeBinaryValidationError("nm and readelf are required on Linux")
        exports = _parse_nm_exports(
            _run([nm, "-D", "--defined-only", "--format=posix", str(binary)])
        )
        dependencies = _parse_readelf_dependencies(_run([readelf, "-d", str(binary)]))
        return f"{nm};{readelf}", exports, dependencies
    raise NativeBinaryValidationError(f"unsupported host platform: {platform!r}")


def _select_wheel(wheel: Path | None, wheel_dir: Path | None) -> Path:
    if (wheel is None) == (wheel_dir is None):
        raise NativeBinaryValidationError("provide exactly one of --wheel or --wheel-dir")
    if wheel is not None:
        if not wheel.is_file():
            raise NativeBinaryValidationError(f"wheel does not exist: {wheel}")
        return wheel
    assert wheel_dir is not None
    candidates = sorted(wheel_dir.glob("*.whl"))
    if len(candidates) != 1:
        raise NativeBinaryValidationError(
            f"expected exactly one wheel in {wheel_dir}, found {len(candidates)}"
        )
    return candidates[0]


def inspect_wheel(wheel: Path, profile: str, platform: str) -> dict[str, object]:
    """Inspect all expected native modules in one wheel."""

    expected = PROFILE_MODULES.get(profile)
    if expected is None:
        raise NativeBinaryValidationError(f"unknown artifact profile: {profile!r}")

    with (
        zipfile.ZipFile(wheel) as archive,
        tempfile.TemporaryDirectory(prefix="pyneurale-native-binary-") as tmp,
    ):
        members: dict[str, str] = {}
        for name in archive.namelist():
            basename = Path(name).name
            if not name.startswith("neurale/") or not basename.endswith(_BINARY_SUFFIXES):
                continue
            module = basename.split(".", 1)[0]
            if Path(name).parent == Path("neurale") and module.startswith("_"):
                if module in members:
                    raise NativeBinaryValidationError(f"duplicate native module in wheel: {module}")
                members[module] = name
        if set(members) != set(expected):
            raise NativeBinaryValidationError(
                f"{profile} wheel native modules must be {sorted(expected)}, "
                f"found {sorted(members)}"
            )

        results: list[dict[str, object]] = []
        for module in sorted(members):
            member = members[module]
            destination = Path(tmp) / Path(member).name
            destination.write_bytes(archive.read(member))
            tool, exports, dependencies = _inspect_binary(destination, platform)
            validate_binary_contract(
                module=module,
                profile=profile,
                platform=platform,
                exports=exports,
                dependencies=dependencies,
            )
            results.append(
                {
                    "module": module,
                    "archive_path": member,
                    "inspection_tool": tool,
                    "exports": sorted(exports),
                    "dependencies": sorted(dependencies, key=str.casefold),
                }
            )
    return {
        "profile": profile,
        "platform": platform,
        "wheel": str(wheel.resolve()),
        "modules": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_MODULES), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wheel", type=Path)
    source.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    platform = (
        "windows"
        if sys.platform == "win32"
        else "linux"
        if sys.platform.startswith("linux")
        else sys.platform
    )
    try:
        wheel = _select_wheel(args.wheel, args.wheel_dir)
        result = inspect_wheel(wheel, args.profile, platform)
        if args.audit_report is not None:
            validate_audit_report(args.audit_report.read_text(encoding="utf-8"), args.profile)
            result["audit_report"] = str(args.audit_report.resolve())
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(rendered + "\n", encoding="utf-8")
    except (OSError, zipfile.BadZipFile, NativeBinaryValidationError) as exc:
        print(f"native binary validation failed: {exc}", file=sys.stderr)
        return 1
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
