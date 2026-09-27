#!/usr/bin/env python3

from __future__ import annotations

import importlib
import importlib.util
import sys

from _subprocess_probe import probe_json


def test_top_level_import_is_light() -> None:
    for module_name in list(sys.modules):
        if module_name == "neurale" or module_name.startswith("neurale."):
            del sys.modules[module_name]

    neurale = importlib.import_module("neurale")

    assert isinstance(neurale.__version__, str)
    assert "data" not in vars(neurale)
    assert "runtime" not in vars(neurale)
    assert "models" not in vars(neurale)
    assert not any(name.startswith("neurale.data") for name in sys.modules)
    assert not any(name.startswith("neurale.runtime") for name in sys.modules)
    assert not any(name.startswith("neurale.models") for name in sys.modules)


def test_top_level_import_loads_no_process_or_native_deps() -> None:
    # What the contract forbids is ``import neurale`` *loading* these modules,
    # so the probe diffs sys.modules across the import instead of reading it
    # absolutely. Interpreter startup can already have loaded some of them
    # before any neurale code runs: in an editable install the scikit-build-core
    # path hook ``_editable_skbc_pyneurale.pth`` imports
    # ``_editable_skbc_pyneurale``, which imports ``importlib.util`` and so
    # ``threading``. That is site processing, not neurale, and an absolute check
    # reports it as a contract breach that only appears in editable checkouts.
    script = """
import json
import sys

before = frozenset(sys.modules)
import neurale
loaded_by_import = frozenset(sys.modules) - before

print(json.dumps({
    "version_is_string": isinstance(neurale.__version__, str),
    "loaded": sorted(
        name
        for name in loaded_by_import
        if name == "threading"
        or name == "numpy"
        or name.startswith("neurale._native")
        or name.startswith("neurale._cuda")
        or name.startswith("neurale.runtime")
    ),
}))
"""
    assert probe_json(script) == {
        "version_is_string": True,
        "loaded": [],
    }


def test_core_namespace_does_not_exist() -> None:
    """Verify the obsolete core package is no longer importable."""

    assert importlib.util.find_spec("neurale.core") is None
