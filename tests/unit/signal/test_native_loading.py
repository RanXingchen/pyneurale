from __future__ import annotations

from types import SimpleNamespace

import pytest

from neurale import _native_loader
from neurale.exceptions import NativeUnavailableError


def test_native_module_loader_caches_success(monkeypatch) -> None:
    signal = SimpleNamespace()
    module = SimpleNamespace(signal=signal)
    calls = 0

    def import_module(name: str):
        nonlocal calls
        calls += 1
        assert name == "neurale._native"
        return module

    _native_loader.load_native_extension.cache_clear()
    monkeypatch.setattr(_native_loader.importlib, "import_module", import_module)
    try:
        assert _native_loader.load_native_namespace("signal") is signal
        assert _native_loader.load_native_namespace("signal") is signal
        assert calls == 1
    finally:
        _native_loader.load_native_extension.cache_clear()
        _native_loader.load_native_namespace.cache_clear()


def test_native_module_loader_retries_absence(monkeypatch) -> None:
    calls = 0

    def import_module(name: str):
        nonlocal calls
        calls += 1
        error = ModuleNotFoundError(name=name)
        raise error

    _native_loader.load_native_extension.cache_clear()
    monkeypatch.setattr(_native_loader.importlib, "import_module", import_module)
    try:
        with pytest.raises(NativeUnavailableError):
            _native_loader.load_native_namespace("signal")
        with pytest.raises(NativeUnavailableError):
            _native_loader.load_native_namespace("signal")
        assert calls == 2
    finally:
        _native_loader.load_native_extension.cache_clear()
        _native_loader.load_native_namespace.cache_clear()


def test_native_operation_loader_returns_requested_namespace(monkeypatch) -> None:
    module = SimpleNamespace(
        signal=SimpleNamespace(
            windows=object(),
            filtering=object(),
            resampling=object(),
            spectral=object(),
            transforms=object(),
        )
    )
    _native_loader.load_native_extension.cache_clear()
    _native_loader.load_native_namespace.cache_clear()
    monkeypatch.setattr(_native_loader, "load_native_extension", lambda: module)
    try:
        assert _native_loader.load_native_namespace("signal.windows") is module.signal.windows
        assert _native_loader.load_native_namespace("signal.filtering") is module.signal.filtering
        assert _native_loader.load_native_namespace("signal.resampling") is module.signal.resampling
        assert _native_loader.load_native_namespace("signal.spectral") is module.signal.spectral
        assert _native_loader.load_native_namespace("signal.transforms") is module.signal.transforms
        with pytest.raises(NativeUnavailableError):
            _native_loader.load_native_namespace("signal.missing")
    finally:
        _native_loader.load_native_namespace.cache_clear()
