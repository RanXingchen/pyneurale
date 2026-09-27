#!/usr/bin/env python3

from __future__ import annotations

from types import SimpleNamespace

import pytest

from neurale import _native_loader
from neurale.exceptions import NativeUnavailableError


def test_native_module_loader_caches_models_namespace(monkeypatch) -> None:
    models = SimpleNamespace()
    module = SimpleNamespace(models=models)
    calls = 0

    def import_module(name: str):
        nonlocal calls
        calls += 1
        assert name == "neurale._native"
        return module

    _native_loader.load_native_extension.cache_clear()
    _native_loader.load_native_namespace.cache_clear()
    monkeypatch.setattr(_native_loader.importlib, "import_module", import_module)
    try:
        assert _native_loader.load_native_namespace("models") is models
        assert _native_loader.load_native_namespace("models") is models
        assert calls == 1
    finally:
        _native_loader.load_native_extension.cache_clear()
        _native_loader.load_native_namespace.cache_clear()


def test_native_module_loader_retries_absence(monkeypatch) -> None:
    calls = 0

    def import_module(name: str):
        nonlocal calls
        calls += 1
        raise ModuleNotFoundError(name=name)

    _native_loader.load_native_extension.cache_clear()
    _native_loader.load_native_namespace.cache_clear()
    monkeypatch.setattr(_native_loader.importlib, "import_module", import_module)
    try:
        with pytest.raises(NativeUnavailableError):
            _native_loader.load_native_namespace("models")
        with pytest.raises(NativeUnavailableError):
            _native_loader.load_native_namespace("models")
        assert calls == 2
    finally:
        _native_loader.load_native_extension.cache_clear()
        _native_loader.load_native_namespace.cache_clear()
