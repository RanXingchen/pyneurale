#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Access to the private native spool v1 specification and its reference tools.

Loaded by path for the same reason the NRF reference is: the tools directory is
specification material that happens to be executable, not a Python package, and
nothing here may resolve through an installed distribution. These tests check a
specification tree in this repository against the reference implementation in
that same tree -- no part of ``neurale`` is involved, and none may be imported
from this module.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPOOL_SPEC_DIR = REPOSITORY_ROOT / "specifications" / "native-spool" / "v1"
SPOOL_TOOLS_DIR = SPOOL_SPEC_DIR / "tools"
SPOOL_VECTOR_DIR = SPOOL_SPEC_DIR / "vectors"
SPOOL_README = SPOOL_SPEC_DIR / "README.md"


def load_spool_tool(name: str) -> ModuleType:
    """Load one spool reference tool by path, under a name of its own."""
    path = SPOOL_TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"native_spool_v1_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load the native spool reference tool at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


spool_format = load_spool_tool("spool_format")
generate_vectors = load_spool_tool("generate_vectors")

#: The vector index, as committed.
VECTOR_IDX: dict[str, Any] = json.loads(
    (SPOOL_VECTOR_DIR / "index.json").read_text(encoding="utf-8")
)

#: Vector name -> its committed bytes.
VECTOR_BYTES: dict[str, bytes] = {
    entry["name"]: (SPOOL_VECTOR_DIR / entry["file"]).read_bytes()
    for entry in VECTOR_IDX["vectors"]
}

#: Vector name -> the verdict the specification says a reader must produce.
VECTOR_EXPECTATIONS: dict[str, dict[str, Any]] = {
    entry["name"]: entry["expect"] for entry in VECTOR_IDX["vectors"]
}


def specification_text() -> str:
    """Return the normative document, for the rules that are stated in prose."""
    return SPOOL_README.read_text(encoding="utf-8")
