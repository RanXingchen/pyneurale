# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Discovery for native C++ test harnesses driven from Python tests.

This module centralises the lookup with one priority order:

    1. an explicit executable path, via the per-harness env var -- errors loudly
    2. ``NEURALE_BUILD_DIR`` -- a single named build tree -- errors loudly
    3. known CMake layouts across every build tree, with the platform suffix
       appended
    4. ``PATH`` via :func:`shutil.which`

When several trees exist at once, a recursive search would pick whichever sorts
first -- and on a checkout that keeps an ASAN sibling beside a Release one, an
ordinary run could land on the ASAN harness. So trees whose name carries a
sanitizer marker (``asan``/``tsan``/``ubsan``/``lsan``) are tried after clean
trees, and ``NEURALE_BUILD_DIR`` remains the way to pin a specific tree when
even that is not enough.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

#: Multi-config generators (Visual Studio, Ninja Multi-Config) place binaries
#: under one of these subdirectories; single-config generators (Ninja, Unix
#: Makefiles) place them directly in ``cpp/tests``. Checked in this order so a
#: plain Release build wins over a debug sibling when both exist; ``Release``
#: is first because it is what a CI gate builds.
_CMAKE_CONFIGS = ("Release", "RelWithDebInfo", "Debug", "MinSizeRel")

#: The shared "which build tree" override. A CI job pins one tree so discovery
#: never silently picks up a neighbour built under different flags. When set,
#: only that tree is searched, and a missing binary there is an error.
BUILD_DIR_ENV = "NEURALE_BUILD_DIR"

#: Build-tree name fragments that mark a sanitizer build. A tree carrying one
#: of these is a different build for a different purpose; an ordinary run that
#: landed on it would run the wrong binary, so it is tried after clean trees.
_SANITIZER_MARKERS = ("asan", "tsan", "ubsan", "lsan")


class NativeHarnessConfigError(RuntimeError):
    """An explicitly configured harness path or build dir does not resolve.

    Raised, not returned as ``None``, so a typo in ``NEURALE_END_TO_END_TEST_BINARY``
    or ``NEURALE_BUILD_DIR`` surfaces as a test error rather than being masked
    as a "not built" skip.
    """


def _platform_name(binary_name: str) -> str:
    """The harness file name on the current platform."""
    return binary_name + (".exe" if sys.platform == "win32" else "")


def _candidates_under(test_dir: Path, name: str) -> list[Path]:
    """Every path that could be the binary inside one build tree's ``cpp/tests``.

    The single-config location is listed first so a plain Ninja/Makefile build
    matches before any multi-config sibling, matching the layout the project
    documents (``build/cpp/tests/<name>``).
    """
    return [test_dir / name, *(test_dir / config / name for config in _CMAKE_CONFIGS)]


def _enumerate_build_trees(root: Path) -> list[tuple[Path, Path]]:
    """Build trees as ``(tree, cpp/tests dir)`` pairs, ordered.

    Covers both on-disk layouts the project uses:

    * a single-tree checkout -- ``build/cpp/tests``, ``build-release/cpp/tests``;
    * a container checkout where one ``build/`` holds several subtrees --
      ``build/cpp-tests/cpp/tests``, ``build/asan-msvc/cpp/tests``.

    Trees whose name carries a sanitizer marker are returned after clean trees,
    so a checkout that keeps an ASAN sibling beside a Release one does not hand
    an ordinary run the ASAN harness. Within each group the order is
    alphabetical for determinism. ``NEURALE_BUILD_DIR`` is the escape hatch when
    even clean trees need disambiguating.
    """

    def sort_key(pair: tuple[Path, Path]) -> tuple[int, str]:
        tree, _test_dir = pair
        name = tree.name.lower()
        sanitizer = 1 if any(marker in name for marker in _SANITIZER_MARKERS) else 0
        return (sanitizer, tree.name)

    pairs: list[tuple[Path, Path]] = []
    for build_dir in sorted(root.glob("build*")):
        if not build_dir.is_dir():
            continue
        single = build_dir / "cpp" / "tests"
        if single.is_dir():
            # Single-tree: the build* dir owns cpp/tests directly.
            pairs.append((build_dir, single))
        for sub in sorted(build_dir.iterdir()):
            if not sub.is_dir():
                continue
            nested = sub / "cpp" / "tests"
            if nested.is_dir():
                # Container: a <subtree> under build/ owns cpp/tests.
                pairs.append((sub, nested))
    pairs.sort(key=sort_key)
    return pairs


def find_native_test_binary(
    binary_name: str,
    *,
    explicit_env: str,
    repo_root: Path,
) -> Path | None:
    """Locate a native C++ test harness by name, or return ``None``.

    Parameters
    ----------
    binary_name:
        The CMake target name without any platform suffix (e.g.
        ``"neurale_recording_end_to_end_test"``); ``.exe`` is appended on Windows.
    explicit_env:
        The per-harness override env var (e.g. ``NEURALE_END_TO_END_TEST_BINARY``).
        When set it must name an existing file or this raises.
    repo_root:
        The repository root, used to enumerate build trees for step 3 (both
        the single-tree and the ``build/<subtree>/`` container layouts).
        Callers pass ``Path(__file__).resolve().parents[N]``.

    Returns ``None`` only when no explicit configuration was given and
    best-effort discovery found nothing; the caller may then skip. Any
    explicit configuration that names something missing raises
    :class:`NativeHarnessConfigError`.
    """
    name = _platform_name(binary_name)

    override = os.environ.get(explicit_env)
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise NativeHarnessConfigError(
                f"{explicit_env} points to a missing executable: {candidate}"
            )
        return candidate.resolve()

    build_dir = os.environ.get(BUILD_DIR_ENV)
    if build_dir:
        test_dir = Path(build_dir).expanduser() / "cpp" / "tests"
        for candidate in _candidates_under(test_dir, name):
            if candidate.is_file():
                return candidate.resolve()
        raise NativeHarnessConfigError(
            f"{BUILD_DIR_ENV}={build_dir!r} has no {name} under {test_dir} "
            f"(checked single-config and {', '.join(_CMAKE_CONFIGS)} layouts)"
        )

    for _tree, test_dir in _enumerate_build_trees(repo_root):
        for candidate in _candidates_under(test_dir, name):
            if candidate.is_file():
                return candidate.resolve()

    found = shutil.which(binary_name)
    return Path(found).resolve() if found else None


def require_native_test_binary(
    binary_name: str,
    *,
    explicit_env: str,
    repo_root: Path,
) -> Path:
    """Locate a native harness, raising :class:`NativeHarnessConfigError` if absent.

    Like :func:`find_native_test_binary` but the "nothing configured, nothing
    found" case raises too. Use this from jobs that must not silently skip: a
    native CI gate that discovers no harness has lost its binary and should
    fail rather than report green on zero tests.
    """
    found = find_native_test_binary(binary_name, explicit_env=explicit_env, repo_root=repo_root)
    if found is None:
        raise NativeHarnessConfigError(
            f"no {binary_name} found: build the C++ tests, set {explicit_env} to "
            f"the executable, or set {BUILD_DIR_ENV} to the build tree"
        )
    return found
