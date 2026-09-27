# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Loading a repository file that no import path reaches.

``benchmarks/`` and ``tools/`` hold real code the suite has to check -- the
regression policy, the wheel validators, the benchmark profiles -- but neither
is a Python package and neither ships in a wheel, so ``import`` cannot find
them. The tests load those files by path instead, and this module owns the one
way of doing it so each test site is a single call.

The module is registered in :data:`sys.modules` before it is executed:
``@dataclass(slots=True)`` looks its own module up there while the class body is
still being built, and repository tooling is free to use one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_repository_module(name: str, path: Path) -> ModuleType:
    """Load the file at ``path`` under the module name ``name``."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load the repository module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
