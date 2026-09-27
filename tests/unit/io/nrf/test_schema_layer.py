#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The Draft 2020-12 layer and the schemas the build installs into the package.

An installed wheel does not carry the repository ``specifications/`` tree, so
the build installs the normative schema files into ``neurale.io.nrf.schemas``
as package resources. The source tree keeps no second copy. These tests check
what a reader actually loads -- the installed resources -- against the
normative originals byte-for-byte, so a wheel built from a stale or incomplete
install rule fails here instead of silently validating against the wrong
contract.
"""

from __future__ import annotations

from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from neurale.io.nrf import _schemas
from neurale.io.nrf._errors import NrfSchemaError
from neurale.io.nrf._schemas import (
    CHECKPOINT_SCHEMA,
    HEAD_SCHEMA,
    JOURNAL_RECORD_SCHEMA,
    MANIFEST_SCHEMA,
    RECOVERY_REPORT_SCHEMA,
    SCHEMA_NAMES,
    SUPPORTED_MAJOR_VERSION,
    require_supported_version,
    schema_bytes,
    validate_document,
)

from .nrf_support import REPOSITORY_ROOT, SPEC_DIR, VECTORS


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_installed_schema_matches_normative_artifact(name: str) -> None:
    assert schema_bytes(name) == (SPEC_DIR / name).read_bytes()


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schemas_are_installed_as_package_resources(name: str) -> None:
    """A reader resolves schemas through the package, not the repository.

    The byte-identity test above would also pass if the loader had a fallback to
    the specification tree, which an installed wheel does not have. This asserts
    the resource itself is there.
    """
    assert (resources.files("neurale.io.nrf.schemas") / name).is_file()


def test_every_normative_schema_is_installed() -> None:
    on_disk = {path.name for path in SPEC_DIR.glob("*.schema.json")}
    assert on_disk == set(SCHEMA_NAMES)


def test_source_tree_holds_no_schema_copy() -> None:
    """The specification tree is the only place a schema is edited.

    A second copy under ``src`` is exactly the drift this layout rules out: it
    would be tracked, editable, and indistinguishable from the installed
    resource to anyone reading the package.
    """
    package = REPOSITORY_ROOT / "src" / "neurale" / "io" / "nrf" / "schemas"
    assert not list(package.glob("*.schema.json"))


def test_uninstalled_schema_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The likely cause is a source checkout, and the error should say that.

    Nothing puts a schema under ``src``, so someone running against the source
    tree gets a missing file at a path they never created. A bare
    ``FileNotFoundError`` sends them looking for a corrupt install.
    """
    monkeypatch.setattr(_schemas.resources, "files", lambda package: tmp_path)
    with pytest.raises(NrfSchemaError, match="is not installed"):
        _schemas.schema_bytes(MANIFEST_SCHEMA)


def test_unknown_schema_name_is_rejected() -> None:
    with pytest.raises(NrfSchemaError, match="unknown NRF schema"):
        schema_bytes("not-a.schema.json")


# --- the normative documents validate -------------------------------------


def test_normative_manifest_validates() -> None:
    validate_document(VECTORS["manifest"], MANIFEST_SCHEMA)


def test_normative_checkpoint_head_and_report_validate() -> None:
    validate_document(VECTORS["checkpoint"], CHECKPOINT_SCHEMA)
    validate_document(VECTORS["head"], HEAD_SCHEMA)
    validate_document(
        VECTORS["recoverable_uncommitted_tail"]["expected_report"], RECOVERY_REPORT_SCHEMA
    )


def test_every_vector_journal_record_validates() -> None:
    records = VECTORS["complete_sequence"]
    assert records
    for record in records:
        validate_document(record, JOURNAL_RECORD_SCHEMA)


# --- the layer actually rejects -------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["Streams/Neural/Data", "streams/cursor/../data", "/streams/cursor/data", "con/data"],
)
def test_schema_layer_rejects_non_canonical_paths(path: str) -> None:
    manifest = deepcopy(VECTORS["manifest"])
    manifest["streams"][0]["data"]["path"] = path
    with pytest.raises(NrfSchemaError, match=MANIFEST_SCHEMA):
        validate_document(manifest, MANIFEST_SCHEMA)


def test_schema_error_names_the_failing_location() -> None:
    manifest = deepcopy(VECTORS["manifest"])
    manifest["streams"][0]["data"]["path"] = "Streams/Neural/Data"
    with pytest.raises(NrfSchemaError) as error:
        validate_document(manifest, MANIFEST_SCHEMA)
    # The message must let a caller tell this layer apart from the semantic one.
    assert "/streams/0/data/path" in str(error.value)


@pytest.mark.parametrize("timestamp", ["2026-12-31T23:59:60Z", "2026-01-01T00:00:00+01:00"])
def test_schema_layer_rejects_unsupported_timestamps(timestamp: str) -> None:
    manifest = deepcopy(VECTORS["manifest"])
    manifest["session"]["created_at"] = timestamp
    with pytest.raises(NrfSchemaError):
        validate_document(manifest, MANIFEST_SCHEMA)


# --- version gate ---------------------------------------------------------


def test_supported_version_accepts_normative_manifest() -> None:
    require_supported_version(VECTORS["manifest"])
    assert SUPPORTED_MAJOR_VERSION == 1


@pytest.mark.parametrize("major", [0, 2, 99])
def test_unknown_major_version_is_rejected(major: int) -> None:
    manifest = deepcopy(VECTORS["manifest"])
    manifest["version"]["major"] = major
    with pytest.raises(NrfSchemaError, match="unsupported major version"):
        require_supported_version(manifest)


def test_higher_minor_version_passes_version_gate() -> None:
    # A higher minor is readable when unknown required features are absent; the
    # schema layer is what establishes their absence.
    manifest = deepcopy(VECTORS["manifest"])
    manifest["version"]["minor"] = 99
    require_supported_version(manifest)


@pytest.mark.parametrize("version", [None, {}, {"major": 1}, {"major": 1, "minor": -1}])
def test_malformed_version_object_is_rejected(version: Any) -> None:
    manifest = deepcopy(VECTORS["manifest"])
    if version is None:
        manifest.pop("version")
    else:
        manifest["version"] = version
    with pytest.raises(NrfSchemaError):
        require_supported_version(manifest)
