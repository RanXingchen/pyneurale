#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""What ``neurale.io.nrf._semantics`` does with a manifest, in its own right.

Whether this validator agrees with the specification's reference validator is a
conformance question, and it is asked in
``tests/specification/test_nrf_v1_package_conformance.py``. What is left here is
the package's own behaviour: which of its two error types a given defect
produces, and the wording a caller is entitled to match on. The specification
permits diagnostics to differ between implementations, so those are exactly the
things the differential suite cannot check.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from neurale.io.nrf._errors import NrfSchemaError, NrfSemanticError
from neurale.io.nrf._semantics import validate_manifest

from .nrf_support import VECTORS


def test_termination_schema_contract_is_enforced() -> None:
    manifest = deepcopy(VECTORS["manifest"])
    termination = next(
        schema for schema in manifest["record_schemas"] if schema["kind"] == "session_termination"
    )
    field = next(item for item in termination["fields"] if item["name"] == "fault_id")
    field["nullable"] = False
    with pytest.raises(NrfSemanticError, match="session_termination field contract"):
        validate_manifest(manifest)


def test_termination_extent_and_seal_state_must_agree() -> None:
    # The normative manifest describes a terminated session: the termination
    # target already has extent 1 and is sealed. Unsealing it while the row
    # stays committed is the state NRF-SEM-017 forbids.
    manifest = deepcopy(VECTORS["manifest"])
    termination = next(
        schema for schema in manifest["record_schemas"] if schema["kind"] == "session_termination"
    )
    assert termination["committed_extent"] == 1
    manifest["commit"]["sealed_targets"] = [
        path for path in manifest["commit"]["sealed_targets"] if path != termination["path"]
    ]
    with pytest.raises(NrfSemanticError, match="extent and seal state disagree"):
        validate_manifest(manifest)


def test_discontinuity_schema_cannot_be_shared_between_streams() -> None:
    manifest = deepcopy(VECTORS["manifest"])
    if len(manifest["streams"]) < 2:
        pytest.skip("normative manifest declares a single stream")
    shared = manifest["streams"][0]["segment_policy"]["discontinuity_record_schema_id"]
    manifest["streams"][1]["segment_policy"]["discontinuity_record_schema_id"] = shared
    with pytest.raises(NrfSemanticError):
        validate_manifest(manifest)


def test_required_record_kinds_must_all_exist() -> None:
    manifest = deepcopy(VECTORS["manifest"])
    manifest["record_schemas"] = [
        schema for schema in manifest["record_schemas"] if schema["kind"] != "commands"
    ]
    with pytest.raises(NrfSemanticError, match="required record schemas are missing"):
        validate_manifest(manifest)


def test_ijson_violation_is_schema_layer_error() -> None:
    # A canonicalization problem is not a relationship problem; the caller must
    # be able to tell them apart.
    manifest = deepcopy(VECTORS["manifest"])
    manifest.setdefault("metadata", {})["big"] = 9007199254740992
    with pytest.raises(NrfSchemaError):
        validate_manifest(manifest)


def test_validator_rejects_non_mapping_manifest() -> None:
    with pytest.raises(NrfSemanticError, match="must be a JSON object"):
        validate_manifest([])
