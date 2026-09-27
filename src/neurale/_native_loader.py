#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Shared helpers for importing optional native extensions."""

from __future__ import annotations

import importlib
from functools import cache
from types import ModuleType

from neurale.exceptions import NativeExtensionError, NativeUnavailableError


@cache
def load_native_extension() -> ModuleType:
    try:
        return importlib.import_module("neurale._native")
    except ModuleNotFoundError as exc:
        if exc.name == "neurale._native":
            raise NativeUnavailableError("The native extension is not installed.") from exc
        if exc.name is not None:
            raise NativeExtensionError("The native extension has a missing dependency.") from exc
        raise NativeExtensionError("The native extension could not be loaded.") from exc
    except (ImportError, OSError) as exc:
        raise NativeExtensionError("The native extension could not be loaded.") from exc


@cache
def load_native_namespace(namespace: str) -> object:
    current: object = load_native_extension()
    for part in namespace.split("."):
        current = getattr(current, part, None)
        if current is None:
            raise NativeUnavailableError(f"Native namespace {namespace!r} is unavailable.")
    return current
