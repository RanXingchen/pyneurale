#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Access optional presentation bindings in the shared native extension."""

from __future__ import annotations

from functools import cache

from neurale._native_loader import load_native_namespace
from neurale.exceptions import DependencyError, NativeUnavailableError


@cache
def load_presentation_extension() -> object:
    """Return the compiled presentation namespace, or explain how to enable it."""
    try:
        return load_native_namespace("experiments.presentation")
    except NativeUnavailableError as exc:
        raise DependencyError(
            "Experiment presentation support is not installed. Rebuild PyNeurale "
            "with NEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON and the GLFW, OpenGL, "
            "FreeType, and HarfBuzz development dependencies."
        ) from exc
