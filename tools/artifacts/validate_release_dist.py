# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Check that a release directory contains the complete supported artifact matrix."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import Version
from validate_wheel_profile import validate_wheel_profile

_EXPECTED = {
    (python, platform) for python in ("cp311", "cp312") for platform in ("windows", "linux")
}
_FORBIDDEN_SDIST_ROOTS = {".venv", "build", "dist", "private", "temp"}


def _validate_sdist(sdist: Path, version: Version) -> None:
    root = f"pyneurale-{version}"
    required = {
        "CMakeLists.txt",
        "LICENSE",
        "NOTICE.md",
        "PKG-INFO",
        "pyproject.toml",
        "tools/artifacts/artifact_profiles.py",
    }
    found: set[str] = set()
    with tarfile.open(sdist, "r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != root:
                raise ValueError(f"unsafe sdist member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"sdist contains a link: {member.name}")
            relative = path.parts[1:]
            if not relative:
                continue
            if (
                relative[0] in _FORBIDDEN_SDIST_ROOTS
                or relative == ("AGENTS.md",)
                or relative[:2] == ("docs", "_build")
                or relative[:3] == ("docs", "api", "_generated")
            ):
                raise ValueError(f"sdist contains a local build file: {member.name}")
            found.add("/".join(relative))
    if missing := required - found:
        raise ValueError(f"sdist is missing required files: {sorted(missing)}")


def validate_release_dist(directory: Path, version: str | None = None) -> dict[str, object]:
    """Require four distinct release wheels and one matching source archive."""

    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 4 or len(sdists) != 1:
        raise ValueError(
            f"expected four wheels and one sdist, found {len(wheels)} and {len(sdists)}"
        )

    name, source_version = parse_sdist_filename(sdists[0].name)
    if str(name) != "pyneurale":
        raise ValueError(f"unexpected sdist distribution: {name}")
    if version is not None and source_version != Version(version):
        raise ValueError(f"sdist version {source_version} does not match {version}")
    _validate_sdist(sdists[0], source_version)

    matrix: dict[tuple[str, str], str] = {}
    for wheel in wheels:
        wheel_name, wheel_version, _, tags = parse_wheel_filename(wheel.name)
        if str(wheel_name) != "pyneurale" or wheel_version != source_version:
            raise ValueError(f"wheel name/version differs from sdist: {wheel.name}")
        platforms = {
            "windows"
            if tag.platform == "win_amd64"
            else "linux"
            if tag.platform == "manylinux_2_39_x86_64"
            else None
            for tag in tags
        }
        interpreters = {tag.interpreter for tag in tags}
        if len(platforms) != 1 or None in platforms or len(interpreters) != 1:
            raise ValueError(f"unsupported wheel tags: {wheel.name}")
        if any(tag.abi != tag.interpreter for tag in tags):
            raise ValueError(f"wheel must use a matching CPython ABI: {wheel.name}")
        key = (next(iter(interpreters)), next(iter(platforms)))
        if key not in _EXPECTED or key in matrix:
            raise ValueError(f"unexpected or duplicate release wheel: {wheel.name}")
        validate_wheel_profile(wheel, "release")
        matrix[key] = wheel.name

    if set(matrix) != _EXPECTED:
        raise ValueError(f"missing release wheels: {sorted(_EXPECTED - set(matrix))}")
    return {
        "version": str(source_version),
        "sdist": sdists[0].name,
        "wheels": {
            f"{python}-{platform}": matrix[python, platform] for python, platform in sorted(matrix)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--version")
    args = parser.parse_args()
    try:
        result = validate_release_dist(args.directory, args.version)
    except (OSError, ValueError, tarfile.TarError) as exc:
        print(f"release artifact validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
