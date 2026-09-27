#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Optional concrete native experiment presentation.

Importing this module does not initialize GLFW, create an OpenGL context, inspect
monitors, or load a font. Native dependencies are loaded only when a presentation
operation is requested. The native paradigm layer remains the owner of task
semantics; these concrete presenters only draw prepared snapshots/requests and
translate native input.

The formal Center-Out closed-loop session connects its presenter to the native
experiment trace bridge. Software submit/swap-return times and the frozen
presentation configuration are recorded as evidence; they are not physical
pixel or photon onset measurements.
"""

from __future__ import annotations

from typing import Final

from ._center_out import (
    CenterOutPresentationConfig,
    CenterOutPresentationTheme,
    CenterOutPresenter,
    CenterOutSessionPresentationBridge,
)
from ._ssvep import SSVEPDisplay, SSVEPDisplayConfig

_DEPENDENCY_NAMES: Final = ("glfw", "opengl_minimum", "freetype", "harfbuzz")
_NATIVE_EXPORTS: Final = {
    "AspectPolicy",
    "CenterOutPresentationControlEvent",
    "CenterOutPresentationControlKind",
    "Color",
    "LogicalRect",
    "Point2D",
    "PresentationResourceStats",
    "PresentationEnvironment",
    "PresentationRuntimeStatus",
    "SoftwarePresentationTimes",
    "SpeechPresentationConfig",
    "SpeechPresentationControlEvent",
    "SpeechPresentationControlKind",
    "SpeechPresentationEvidence",
    "SpeechPresentationResult",
    "SpeechPresentationStyle",
    "SpeechPresenter",
    "SpeechTextConfig",
    "WebGridPresentationConfig",
    "WebGridPresentationInput",
    "WebGridPresentationInputKind",
    "WebGridPresentationStyle",
    "WebGridPresenter",
    "WindowSize",
    "renderer_monotonic_now_ns",
}


def dependency_versions() -> dict[str, tuple[int, ...]]:
    """Return versions compiled into the optional native presentation module.

    Raises
    ------
    neurale.exceptions.DependencyError
        If this installation was built without experiment presentation support.
    neurale.exceptions.NativeExtensionError
        If the shared native extension or one of its dependencies cannot be loaded.
    """

    from ._native import load_presentation_extension

    values = load_presentation_extension().dependency_versions()
    return {name: tuple(int(part) for part in values[name]) for name in _DEPENDENCY_NAMES}


def __getattr__(name: str) -> object:
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from ._native import load_presentation_extension

    value = getattr(load_presentation_extension(), name)
    globals()[name] = value
    return value


def _native_available() -> bool:
    from importlib.util import find_spec

    return find_spec("neurale._native") is not None


def __dir__() -> list[str]:
    names = set(globals())
    if _native_available():
        names.update(_NATIVE_EXPORTS)
    return sorted(names)


__all__ = [
    "SSVEPDisplay",
    "SSVEPDisplayConfig",
    "CenterOutPresentationConfig",
    "CenterOutPresentationTheme",
    "CenterOutPresenter",
    "CenterOutSessionPresentationBridge",
    "dependency_versions",
    *sorted(_NATIVE_EXPORTS),
]
