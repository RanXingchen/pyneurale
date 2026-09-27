#!/usr/bin/env python3

from __future__ import annotations

import pytest

from neurale import _native_loader
from neurale.exceptions import NativeExtensionError
from neurale.runtime import _native_probe


def test_native_probe_suppresses_only_absent_module(
    monkeypatch,
) -> None:
    def missing(name: str):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(_native_loader.importlib, "import_module", missing)

    _native_loader.load_native_extension.cache_clear()
    assert _native_probe.probe_native_build() is None


@pytest.mark.parametrize("error", [ImportError("broken ABI"), OSError("bad DLL")])
def test_optional_native_probe_surfaces_broken_extension(monkeypatch, error) -> None:
    def broken(name: str):
        raise error

    monkeypatch.setattr(_native_loader.importlib, "import_module", broken)

    _native_loader.load_native_extension.cache_clear()
    with pytest.raises(NativeExtensionError, match="could not be loaded"):
        _native_probe.probe_native_build()
