#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Draft 2020-12 validation, the first of the two mandatory NRF v1 layers.

Specification reference: ``README.md`` section 4 and ``semantic-validation.md``
section 1. JSON Schema is authoritative for local structure, required members,
scalar domains, conditional stream layouts, and nullable-validity presence.
Cross-object rules live in :mod:`neurale.io.nrf._semantics`. Passing either
layer alone is insufficient.

The schema documents are read from the ``schemas/`` package, which the build
populates from the one normative location, ``specifications/nrf/v1/``. An
installed wheel does not carry the repository ``specifications/`` tree, but a
reader must still validate a manifest it did not write, so the files are
installed as a package resource rather than kept as a second copy in the source
tree that could drift. A plain source checkout therefore has no schemas: this
module needs a built or installed package.
``tests/unit/io/nrf/test_schema_layer.py`` checks the installed documents
against the normative originals byte-for-byte.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from importlib import resources
from typing import Any

from neurale.runtime.dependencies import require_dependency

from ._errors import NrfSchemaError

MANIFEST_SCHEMA = "manifest.schema.json"
JOURNAL_RECORD_SCHEMA = "journal-record.schema.json"
CHECKPOINT_SCHEMA = "checkpoint.schema.json"
RECOVERY_REPORT_SCHEMA = "recovery-report.schema.json"
HEAD_SCHEMA = "head.schema.json"

SCHEMA_NAMES = (
    MANIFEST_SCHEMA,
    JOURNAL_RECORD_SCHEMA,
    CHECKPOINT_SCHEMA,
    RECOVERY_REPORT_SCHEMA,
    HEAD_SCHEMA,
)

#: NRF format major version this implementation accepts. A reader MUST reject
#: any other major version rather than attempt a best-effort read.
SUPPORTED_MAJOR_VERSION = 1

#: Highest minor version this implementation understands in full.
SUPPORTED_MINOR_VERSION = 0


def schema_bytes(name: str) -> bytes:
    """Return the raw bytes of one installed NRF schema."""
    if name not in SCHEMA_NAMES:
        raise NrfSchemaError(f"unknown NRF schema {name!r}")
    try:
        return (resources.files(f"{__package__}.schemas") / name).read_bytes()
    except (FileNotFoundError, ModuleNotFoundError) as error:
        # The likely cause is running against the source tree, which holds no
        # schema: the build installs them from specifications/nrf/v1. Saying so
        # is worth more than a bare FileNotFoundError on a path nobody put there.
        raise NrfSchemaError(
            f"the NRF schema {name!r} is not installed; it is installed into the package "
            "from specifications/nrf/v1 at build time, so build or install PyNeurale rather "
            "than importing it from the source tree"
        ) from error


@cache
def schema(name: str) -> dict[str, Any]:
    """Return one parsed NRF schema document."""
    return json.loads(schema_bytes(name).decode("utf-8"))


@cache
def _validator(name: str):
    jsonschema = require_dependency("jsonschema", extra="nrf")
    document = schema(name)
    # format_checker enables the date-time annotation the manifest relies on;
    # the calendar-validity rule is additionally enforced semantically.
    return jsonschema.Draft202012Validator(document, format_checker=jsonschema.FormatChecker())


def validate_document(value: Any, name: str) -> None:
    """Validate *value* against the named NRF schema.

    Raises
    ------
    NrfSchemaError
        If the document does not conform. The message names the failing schema
        and JSON pointer so the caller can tell this layer apart from the
        semantic layer.
    """
    errors = sorted(_validator(name).iter_errors(value), key=lambda error: list(error.path))
    if not errors:
        return
    first = _most_specific(errors[0])
    pointer = "/".join(str(part) for part in first.path)
    location = f" at /{pointer}" if pointer else ""
    raise NrfSchemaError(f"{name} validation failed{location}: {first.message}")


#: Validators that report a *consequence* of a defect rather than the defect. A
#: record missing a required member also has "unevaluated" members, but only the
#: missing one is worth naming.
_CONSEQUENTIAL_VALIDATORS = frozenset({"unevaluatedProperties", "additionalProperties"})


def _most_specific(error: Any) -> Any:
    """Descend a ``oneOf``/``anyOf`` failure to the branch that explains it.

    A journal record is a ``oneOf`` over the four record kinds, so the raw
    top-level error says only "is not valid under any of the given schemas" and
    quotes the whole record back. The branch the record's own ``kind`` selects
    is the one with something useful to say.
    """
    while error.context:
        error = _closest_branch_error(error.context)
    return error


def _closest_branch_error(context: Any) -> Any:
    """Return the one error among a set of failed branches that names the defect."""
    branches: dict[Any, list[Any]] = {}
    for item in context:
        key = item.schema_path[0] if item.schema_path else 0
        branches.setdefault(key, []).append(item)
    # A branch that failed on a discriminating constant -- ``kind`` for a
    # journal record -- is simply the wrong branch and has nothing to say.
    discriminated = [
        errors
        for errors in branches.values()
        if not any(item.validator == "const" for item in errors)
    ]
    chosen = min(discriminated or list(branches.values()), key=len)
    ordered = sorted(chosen, key=lambda item: list(item.path))
    substantive = [item for item in ordered if item.validator not in _CONSEQUENTIAL_VALIDATORS]
    return (substantive or ordered)[0]


def require_supported_version(manifest: Mapping[str, Any]) -> None:
    """Reject a manifest this reader must not interpret.

    A version 1 reader MUST reject any other major version. A higher minor
    version is readable only when unknown required features are absent, which
    the schema layer establishes by rejecting unknown required members outside
    declared extension points.
    """
    version = manifest.get("version")
    if not isinstance(version, Mapping):
        raise NrfSchemaError("manifest is missing its version object")
    major = version.get("major")
    minor = version.get("minor")
    if major != SUPPORTED_MAJOR_VERSION:
        raise NrfSchemaError(
            f"unsupported major version {major!r}; this reader implements NRF "
            f"v{SUPPORTED_MAJOR_VERSION}"
        )
    if not isinstance(minor, int) or minor < 0:
        raise NrfSchemaError(f"invalid minor version {minor!r}")
