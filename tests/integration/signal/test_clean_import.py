#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path

from _subprocess_probe import probe_json


def test_signal_import_loads_scipy_without_runtime() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    source_root = repository_root / "src"
    script = "import json, sys; import neurale.signal; print(json.dumps(sorted(sys.modules)))"
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((str(source_root), existing_pythonpath))
        if existing_pythonpath
        else str(source_root)
    )

    loaded_modules = set(probe_json(script, env=environment))

    assert "scipy" in loaded_modules
    assert "neurale._native" not in loaded_modules
    assert not any(name.startswith("neurale.runtime") for name in loaded_modules)
