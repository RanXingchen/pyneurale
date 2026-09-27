#!/usr/bin/env python3

"""The optional Torch namespace, checked from both sides of the boundary.

Two properties are worth more than anything the namespace currently contains:
importing :mod:`neurale.decoding` neither imports nor *probes* Torch, and
importing :mod:`neurale.decoding.torch` does not import Torch either. The
probe half matters because ``importlib.util.find_spec`` is cheap but not free,
and because a probe at import time is how an optional dependency quietly
becomes a required one.

Torch is absent from the reference environment, so absence is simulated rather
than assumed: ``sys.modules["torch"] = None`` is the standard block, and it
makes ``find_spec`` raise where a missing distribution would make it return
``None``. Presence is simulated the same way with a stub. Where Torch really is
installed, one further test uses the real framework.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType

import pytest
from _subprocess_probe import probe_json

from neurale.decoding import torch as torch_boundary
from neurale.exceptions import DependencyError

TORCH_INSTALLED = importlib.util.find_spec("torch") is not None


@pytest.fixture
def absent_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every lookup of ``torch`` fail, installed or not."""

    monkeypatch.setitem(sys.modules, "torch", None)


@pytest.fixture
def stub_torch(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Make ``torch`` resolvable to a module that imports nothing."""

    stub = ModuleType("torch")
    stub.__spec__ = importlib.util.spec_from_loader("torch", loader=None)
    stub.__version__ = "0.0.0-stub"
    monkeypatch.setitem(sys.modules, "torch", stub)
    return stub


# The boundary as seen from neurale.decoding


def test_importing_decoding_neither_imports_nor_probes_torch() -> None:
    """A finder on ``sys.meta_path`` sees every probe, including one that imports nothing.

    Asserting only that ``torch`` is absent from ``sys.modules`` would pass for
    a package that calls ``find_spec("torch")`` at import time to decide what
    to export -- which is exactly the pattern this boundary exists to prevent.
    """

    code = """
import json
import sys

probes = []


class ProbeRecorder:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "torch":
            probes.append(name)
        return None


sys.meta_path.insert(0, ProbeRecorder())

import importlib

importlib.import_module("neurale.decoding")
print(json.dumps({
    "probes": probes,
    "imported_torch": "torch" in sys.modules,
    "imported_boundary": "neurale.decoding.torch" in sys.modules,
}))
"""
    result = probe_json(code)

    assert result["probes"] == []
    assert not result["imported_torch"]
    assert not result["imported_boundary"]


def test_boundary_import_does_not_import_torch() -> None:
    """Resolution happens at first use, as it does for SciPy in ``neurale.io``."""

    code = """
import json
import sys

probes = []


class ProbeRecorder:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "torch":
            probes.append(name)
        return None


sys.meta_path.insert(0, ProbeRecorder())

import importlib

boundary = importlib.import_module("neurale.decoding.torch")
print(json.dumps({
    "exports": sorted(boundary.__all__),
    "probes": probes,
    "imported_torch": "torch" in sys.modules,
}))
"""
    result = probe_json(code)

    assert not result["imported_torch"]
    assert result["probes"] == []
    assert result["exports"] == ["has_torch", "require_torch", "torch_status"]


def test_boundary_is_not_exported_beside_decoders() -> None:
    decoding = importlib.import_module("neurale.decoding")

    assert set(torch_boundary.__all__).isdisjoint(decoding.__all__)
    assert "torch" not in decoding.__all__


def test_namespace_defines_no_model_class() -> None:
    """Nothing here may look like an implemented decoder.

    A placeholder class would falsely advertise decoder support; an empty
    namespace accurately reports what is available.
    """

    defined_here = [
        name
        for name, value in vars(torch_boundary).items()
        if isinstance(value, type) and getattr(value, "__module__", "") == torch_boundary.__name__
    ]

    assert defined_here == []
    assert all(callable(getattr(torch_boundary, name)) for name in torch_boundary.__all__)


# Resolving Torch when it is absent


def test_status_reports_torch_unavailable_with_reason(absent_torch: None) -> None:
    status = torch_boundary.torch_status()

    assert not status.available
    assert status.import_name == "torch"
    assert status.reason


def test_has_torch_is_false_when_torch_cannot_be_located(absent_torch: None) -> None:
    assert torch_boundary.has_torch() is False


def test_absent_torch_raises_dependency_error(absent_torch: None) -> None:
    """A missing framework fails through the public hierarchy, with an install hint."""

    with pytest.raises(DependencyError, match="torch is required for this operation") as failure:
        torch_boundary.require_torch()

    assert isinstance(failure.value, ImportError)
    assert 'pip install "torch"' in str(failure.value)


# Resolving Torch when it is present


def test_status_and_require_agree_when_torch_can_be_located(stub_torch: ModuleType) -> None:
    status = torch_boundary.torch_status()

    assert status.available
    assert torch_boundary.has_torch() is True
    assert torch_boundary.require_torch() is stub_torch


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch is not installed in this environment")
def test_require_torch_returns_installed_framework() -> None:
    module = torch_boundary.require_torch()

    assert module.__name__ == "torch"
    assert torch_boundary.has_torch() is True
    assert torch_boundary.torch_status().available
