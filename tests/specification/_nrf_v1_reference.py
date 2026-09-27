#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Access to the NRF v1 specification and its executable reference tooling.

The reference validator, the JCS helpers, and the vector generator live beside
the specification they define, in ``specifications/nrf/v1/tools/``. That
directory is deliberately not a Python package and is not installed into any
wheel, so this module loads each file by path.

That is more ceremony than an import, and it says something an import could
not: these tests are running the reference implementation *from this
repository*, against a specification tree in this repository. Nothing here
resolves through an installed distribution -- which is the whole point, since
what is under test elsewhere is the installed package.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = REPOSITORY_ROOT / "specifications" / "nrf" / "v1"
TOOLS_DIR = SPEC_DIR / "tools"

#: The normative, language-neutral vectors every layer of the suite is pinned to.
VECTORS: dict[str, Any] = json.loads((SPEC_DIR / "test-vectors.json").read_text(encoding="utf-8"))


def load_reference(name: str) -> ModuleType:
    """Load one reference tool by path, under a name of its own.

    The module is registered in :data:`sys.modules` before it is executed:
    ``@dataclass(slots=True)`` looks its own module up there while the class
    body is still being built, and a reference tool is free to use one.
    """
    path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"nrf_v1_reference_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load the NRF v1 reference tool at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


jcs = load_reference("jcs")
semantic_validation = load_reference("semantic_validation")

canonical_json = jcs.canonical_json
canonical_json_bytes = jcs.canonical_json_bytes

SEMANTIC_VALIDATION_VERSION: str = semantic_validation.SEMANTIC_VALIDATION_VERSION
NrfSemanticValidationError = semantic_validation.NrfSemanticValidationError
portable_path_key = semantic_validation.portable_path_key
validate_manifest_semantics = semantic_validation.validate_manifest_semantics


def load_generator() -> ModuleType:
    """Load the vector generator.

    It is loaded on demand rather than at import: it is only needed by the
    reproducibility test, and executing it costs more than the other two.
    """
    return load_reference("generate_test_vectors")
