# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Validate the native-module and dependency contract of a PyNeurale wheel."""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

if __package__:
    from .artifact_profiles import LICENSE_FILES, PROFILE_MODULES, RUNTIME_REQUIREMENTS
else:
    from artifact_profiles import LICENSE_FILES, PROFILE_MODULES, RUNTIME_REQUIREMENTS

_DECLARED_EXTRAS = frozenset({"dev", "docs", "lsl", "nrf", "test"})
_FORBIDDEN_CAPABILITY_EXTRAS = frozenset(
    {
        "cuda",
        "experiment-presentation",
        "presentation",
        "torch",
        "visualization",
        "visualization-live",
    }
)
_FORBIDDEN_RUNTIME_REQUIREMENTS = frozenset(
    {
        "freetype",
        "freetype-py",
        "glfw",
        "harfbuzz",
        "matplotlib",
        "pyglfw",
        "pyside6",
        "pytorch",
        "torch",
        "uharfbuzz",
    }
)


class WheelProfileError(ValueError):
    """A wheel does not implement its declared PyNeurale artifact profile."""


def _normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if match is None:
        raise WheelProfileError(f"invalid Requires-Dist value: {requirement!r}")
    return _normalize_name(match.group(1))


def _native_modules(names: list[str]) -> frozenset[str]:
    modules: set[str] = set()
    for name in names:
        path = PurePosixPath(name)
        if path.parent != PurePosixPath("neurale") or path.suffix not in {".pyd", ".so"}:
            continue
        module = path.name.split(".", 1)[0]
        if module.startswith("_"):
            modules.add(module)
    return frozenset(modules)


def validate_wheel_profile(wheel: Path, profile: str) -> None:
    """Raise ``WheelProfileError`` unless *wheel* matches *profile*."""

    if profile not in PROFILE_MODULES:
        raise WheelProfileError(f"unknown wheel profile: {profile!r}")
    if not wheel.is_file():
        raise WheelProfileError(f"wheel does not exist: {wheel}")

    try:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata_paths) != 1:
                raise WheelProfileError("wheel must contain exactly one .dist-info/METADATA file")
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
            license_root = metadata_paths[0].removesuffix("METADATA") + "licenses/"
            bundled_licenses = {
                name.removeprefix(license_root) for name in names if name.startswith(license_root)
            }
            if missing := LICENSE_FILES - bundled_licenses:
                raise WheelProfileError(f"wheel is missing license files: {sorted(missing)}")
    except zipfile.BadZipFile as exc:
        raise WheelProfileError(f"invalid wheel archive: {wheel}") from exc

    distribution = _normalize_name(metadata.get("Name", ""))
    if distribution != "pyneurale":
        raise WheelProfileError(f"unexpected distribution name: {distribution!r}")

    modules = _native_modules(names)
    expected_modules = PROFILE_MODULES[profile]
    if modules != expected_modules:
        raise WheelProfileError(
            f"{profile} profile requires native modules {sorted(expected_modules)}, "
            f"found {sorted(modules)}"
        )
    if profile == "release" and wheel.name.endswith("-win_amd64.whl"):
        openmp_copies = [
            name
            for name in names
            if re.fullmatch(r"libiomp5md(?:-[0-9a-f]+)?\.dll", Path(name).name.lower())
        ]
        if len(openmp_copies) != 1:
            raise WheelProfileError(
                f"release Windows wheel requires one Intel OpenMP runtime, found {len(openmp_copies)}"
            )
    if profile == "release" and "-manylinux_" in wheel.name:
        bundled = [Path(name).name.lower() for name in names if name.startswith("pyneurale.libs/")]
        for runtime in ("libiomp5", "libcudart"):
            copies = [
                name
                for name in bundled
                if re.fullmatch(rf"{runtime}-[0-9a-f]+\.so(?:\.\d+)*", name)
            ]
            if len(copies) != 1:
                raise WheelProfileError(
                    f"release Linux wheel requires one bundled {runtime} runtime, found {len(copies)}"
                )

    requirements = metadata.get_all("Requires-Dist", [])
    runtime_requirements = frozenset(
        _requirement_name(requirement)
        for requirement in requirements
        if "extra ==" not in requirement.lower()
    )
    forbidden_requirements = runtime_requirements & _FORBIDDEN_RUNTIME_REQUIREMENTS
    if forbidden_requirements:
        raise WheelProfileError(
            "forbidden core runtime requirements: " + ", ".join(sorted(forbidden_requirements))
        )
    if runtime_requirements != RUNTIME_REQUIREMENTS:
        raise WheelProfileError(
            f"runtime requirements must be {sorted(RUNTIME_REQUIREMENTS)}, "
            f"found {sorted(runtime_requirements)}"
        )

    extras = frozenset(_normalize_name(value) for value in metadata.get_all("Provides-Extra", []))
    forbidden_extras = {
        extra
        for extra in extras
        if extra in _FORBIDDEN_CAPABILITY_EXTRAS
        or extra.startswith("device-")
        or extra.startswith("vendor-")
    }
    if forbidden_extras:
        raise WheelProfileError(
            "native capability must not be represented by an inert Python extra: "
            + ", ".join(sorted(forbidden_extras))
        )
    if extras != _DECLARED_EXTRAS:
        raise WheelProfileError(
            f"declared extras must be {sorted(_DECLARED_EXTRAS)}, found {sorted(extras)}"
        )


def _select_wheel(wheel: Path | None, wheel_dir: Path | None) -> Path:
    if (wheel is None) == (wheel_dir is None):
        raise WheelProfileError("provide exactly one of WHEEL or --wheel-dir")
    if wheel is not None:
        return wheel
    candidates = sorted(wheel_dir.glob("*.whl")) if wheel_dir is not None else []
    if len(candidates) != 1:
        raise WheelProfileError(
            f"wheel directory must contain exactly one wheel, found {len(candidates)}"
        )
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", nargs="?", type=Path)
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--profile", choices=sorted(PROFILE_MODULES), required=True)
    args = parser.parse_args(argv)
    try:
        wheel = _select_wheel(args.wheel, args.wheel_dir)
        validate_wheel_profile(wheel, args.profile)
    except WheelProfileError as exc:
        print(f"wheel profile validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"validated {args.profile} wheel profile: {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
