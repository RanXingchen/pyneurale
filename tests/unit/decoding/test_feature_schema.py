#!/usr/bin/env python3

from __future__ import annotations

import os

import numpy as np
import pytest
from _subprocess_probe import probe_json

from neurale.data import ChannelInfo, ChannelTable, Clock, FeatureMatrix
from neurale.decoding import ContinuousTargetSchema, FeatureSchema
from neurale.exceptions import ValidationError


def _matrix(**overrides: object) -> FeatureMatrix:
    settings: dict[str, object] = {
        "data": np.zeros((8, 3), dtype=np.float64),
        "fs": 20.0,
        "feature_names": ["lmp", "beta", "gamma"],
        "unit": ["uV", "uV^2", "uV^2"],
        "source_signal": "m1_lfp",
        "window_size": 0.256,
        "shift": 0.05,
        "attrs": {"timestamp_reference": "window_center"},
    }
    settings.update(overrides)
    return FeatureMatrix(**settings)


# --------------------------------------------------------------------------------------
# Derivation from a feature matrix
# --------------------------------------------------------------------------------------


def test_schema_captures_canonical_feature_contract() -> None:
    schema = FeatureSchema.from_feature_matrix(_matrix())

    assert schema.feature_names == ("lmp", "beta", "gamma")
    assert schema.units == ("uV", "uV^2", "uV^2")
    assert schema.dtype == np.dtype(np.float64)
    assert schema.n_features == 3
    assert schema.fs == 20.0
    assert schema.window_size == 0.256
    assert schema.shift == 0.05
    assert schema.source_signal == "m1_lfp"
    assert schema.timestamp_reference == "window_center"


def test_shared_unit_expands_per_feature() -> None:
    """The canonical form is per-feature, so a shared unit is not a second spelling."""

    shared = FeatureSchema.from_feature_matrix(_matrix(unit="uV"))
    listed = FeatureSchema.from_feature_matrix(_matrix(unit=["uV", "uV", "uV"]))

    assert shared.units == ("uV", "uV", "uV")
    assert shared.fingerprint == listed.fingerprint


def test_irregular_matrix_declares_no_observation_rate() -> None:
    matrix = _matrix(fs=None, time=np.array([0.0, 0.1, 0.3, 0.35, 0.9, 1.0, 1.4, 2.0]))
    schema = FeatureSchema.from_feature_matrix(matrix)

    assert schema.fs is None


def test_absent_timestamp_reference_stays_undeclared() -> None:
    assert FeatureSchema.from_feature_matrix(_matrix(attrs={})).timestamp_reference is None


def test_non_string_timestamp_reference_is_metadata_error() -> None:
    with pytest.raises(ValidationError, match="timestamp_reference"):
        FeatureSchema.from_feature_matrix(_matrix(attrs={"timestamp_reference": 3}))


def test_from_feature_matrix_requires_feature_matrix() -> None:
    with pytest.raises(ValidationError, match="must be a FeatureMatrix"):
        FeatureSchema.from_feature_matrix(np.zeros((4, 2)))


# --------------------------------------------------------------------------------------
# Fingerprint determinism
# --------------------------------------------------------------------------------------


def test_equal_schemas_share_fingerprint() -> None:
    assert (
        FeatureSchema.from_feature_matrix(_matrix()).fingerprint
        == FeatureSchema.from_feature_matrix(_matrix()).fingerprint
    )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"data": np.ones((8, 3))}, id="values"),
        pytest.param({"data": np.zeros((64, 3))}, id="frame-count"),
        pytest.param({"t0": 931.5}, id="start-time"),
        pytest.param({"attrs": {"timestamp_reference": "window_center", "run": 7}}, id="attrs"),
    ],
)
def test_fingerprint_ignores_non_schema_input(overrides: dict[str, object]) -> None:
    """One schema describes a family of recordings, not one of them."""

    reference = FeatureSchema.from_feature_matrix(_matrix()).fingerprint

    assert FeatureSchema.from_feature_matrix(_matrix(**overrides)).fingerprint == reference


def test_fingerprint_is_stable_across_processes() -> None:
    """It is a stored contract, so it may not depend on a process-local hash seed."""

    code = """
import json

import numpy as np

from neurale.data import FeatureMatrix
from neurale.decoding import FeatureSchema

matrix = FeatureMatrix(
    data=np.zeros((8, 3)),
    fs=20.0,
    feature_names=["lmp", "beta", "gamma"],
    unit=["uV", "uV^2", "uV^2"],
    source_signal="m1_lfp",
    window_size=0.256,
    shift=0.05,
    attrs={"timestamp_reference": "window_center"},
)
print(json.dumps(FeatureSchema.from_feature_matrix(matrix).fingerprint))
"""
    # Only the hash seed is under test. Wiping the rest of the environment
    # would also wipe whatever makes neurale importable, and turn an import
    # failure into a fingerprint failure.
    runs = [probe_json(code, env={**os.environ, "PYTHONHASHSEED": seed}) for seed in ("0", "12345")]

    assert runs[0] == runs[1] == FeatureSchema.from_feature_matrix(_matrix()).fingerprint


# --------------------------------------------------------------------------------------
# Fingerprint sensitivity
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"feature_names": ["lmp", "beta", "high_gamma"]}, id="renamed"),
        pytest.param({"feature_names": ["beta", "lmp", "gamma"]}, id="reordered"),
        pytest.param({"unit": ["uV", "uV^2", "uV"]}, id="units"),
        pytest.param({"data": np.zeros((8, 3), dtype=np.float32)}, id="dtype"),
        pytest.param({"fs": 25.0}, id="observation-rate"),
        pytest.param({"window_size": 0.512}, id="window"),
        pytest.param({"shift": 0.1}, id="shift"),
        pytest.param({"source_signal": "pmd_lfp"}, id="source"),
        pytest.param({"attrs": {"timestamp_reference": "window_end"}}, id="timestamp-reference"),
        pytest.param({"attrs": {}}, id="timestamp-reference-dropped"),
    ],
)
def test_fingerprint_changes_with_schema_change(
    overrides: dict[str, object],
) -> None:
    reference = FeatureSchema.from_feature_matrix(_matrix()).fingerprint

    assert FeatureSchema.from_feature_matrix(_matrix(**overrides)).fingerprint != reference


def test_name_permutation_is_different_input() -> None:
    """Order is part of the contract: column two is not interchangeable with column one."""

    reordered = _matrix(
        data=np.zeros((8, 3)),
        feature_names=["beta", "lmp", "gamma"],
        unit=["uV^2", "uV", "uV^2"],
    )
    schema = FeatureSchema.from_feature_matrix(_matrix())

    assert not schema.is_compatible_with(FeatureSchema.from_feature_matrix(reordered))


# --------------------------------------------------------------------------------------
# Compatibility
# --------------------------------------------------------------------------------------


def test_compatibility_agrees_with_fingerprint_equality() -> None:
    schema = FeatureSchema.from_feature_matrix(_matrix())
    same = FeatureSchema.from_feature_matrix(_matrix(data=np.ones((3, 3))))
    other = FeatureSchema.from_feature_matrix(_matrix(fs=25.0))

    assert schema.is_compatible_with(same) is (schema.fingerprint == same.fingerprint) is True
    assert schema.is_compatible_with(other) is (schema.fingerprint == other.fingerprint) is False


def test_require_compatible_names_differing_field() -> None:
    schema = FeatureSchema.from_feature_matrix(_matrix())
    other = FeatureSchema.from_feature_matrix(_matrix(window_size=0.512))

    with pytest.raises(ValidationError, match=r"window_size is 0.512, expected 0.256"):
        schema.require_compatible(other)


def test_one_sided_field_declaration_is_mismatch() -> None:
    """A decoder cannot show that an undeclared side agrees, so it does not assume it."""

    schema = FeatureSchema.from_feature_matrix(_matrix())
    undeclared = FeatureSchema.from_feature_matrix(_matrix(shift=None))

    with pytest.raises(ValidationError, match="shift is None"):
        schema.require_compatible(undeclared)


def test_two_undeclared_fields_match() -> None:
    schema = FeatureSchema.from_feature_matrix(_matrix(shift=None, window_size=None))
    other = FeatureSchema.from_feature_matrix(_matrix(shift=None, window_size=None))

    schema.require_compatible(other)


def test_compatibility_requires_schema() -> None:
    schema = FeatureSchema.from_feature_matrix(_matrix())

    with pytest.raises(ValidationError, match="must be a FeatureSchema"):
        schema.is_compatible_with("float64")
    with pytest.raises(ValidationError, match="must be a FeatureSchema"):
        schema.require_compatible(None)


# --------------------------------------------------------------------------------------
# Direct construction
# --------------------------------------------------------------------------------------


def test_schema_can_be_declared_without_matrix() -> None:
    schema = FeatureSchema(
        feature_names=["a", "b"],
        units=["uV", "uV"],
        dtype=np.float64,
        fs=10.0,
    )

    assert schema.n_features == 2
    assert schema.window_size is None
    assert isinstance(schema.fingerprint, str)


def test_schema_is_immutable() -> None:
    schema = FeatureSchema(feature_names=["a"], units=["uV"], dtype=np.float64)

    with pytest.raises(AttributeError):
        schema.fs = 10.0


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        pytest.param({"feature_names": []}, "at least one feature", id="empty"),
        pytest.param({"feature_names": ["a", "a"]}, "must be unique", id="duplicate"),
        pytest.param({"feature_names": ["a", ""]}, "non-empty strings", id="blank"),
        pytest.param({"feature_names": "ab"}, "sequence of strings", id="bare-string"),
        pytest.param({"units": ["uV"]}, "one unit per feature", id="unit-count"),
        pytest.param({"units": [1, 2]}, "non-empty strings", id="unit-type"),
        pytest.param({"units": ["", "uV"]}, "non-empty strings", id="unit-blank"),
        pytest.param({"dtype": np.dtype("U8")}, "numeric feature data", id="dtype"),
        pytest.param({"fs": 0.0}, "fs", id="rate"),
        pytest.param({"window_size": -1.0}, "window_size", id="window"),
        pytest.param({"shift": np.inf}, "shift", id="shift"),
        pytest.param({"source_signal": ""}, "source_signal", id="source"),
        pytest.param({"timestamp_reference": 7}, "timestamp_reference", id="reference"),
    ],
)
def test_schema_rejects_invalid_declaration(
    settings: dict[str, object],
    message: str,
) -> None:
    declaration: dict[str, object] = {
        "feature_names": ["a", "b"],
        "units": ["uV", "uV"],
        "dtype": np.float64,
    }
    declaration.update(settings)

    with pytest.raises(ValidationError, match=message):
        FeatureSchema(**declaration)


# --------------------------------------------------------------------------------------
# The continuous target schema
# --------------------------------------------------------------------------------------


def _channels(units: list[str]) -> ChannelTable:
    return ChannelTable(
        ChannelInfo(name=name, index=idx, type="behavior", unit=unit)
        for idx, (name, unit) in enumerate(zip(["vx", "vy"], units, strict=True))
    )


def _target_schema(**overrides: object) -> ContinuousTargetSchema:
    settings: dict[str, object] = {
        "channels": _channels(["m/s", "m/s"]),
        "unit": "m/s",
        "name": "hand_velocity",
        "fs": 20.0,
    }
    settings.update(overrides)
    return ContinuousTargetSchema(**settings)


def test_target_schema_describes_predicted_outputs() -> None:
    schema = _target_schema()

    assert schema.n_outputs == 2
    assert schema.output_names == ["vx", "vy"]
    assert schema.unit == "m/s"


def test_target_schema_accepts_unit_per_output() -> None:
    schema = _target_schema(channels=_channels(["m/s", "rad/s"]), unit=["m/s", "rad/s"])

    assert schema.unit == ("m/s", "rad/s")


@pytest.mark.parametrize(
    ("unit", "message"),
    [
        pytest.param([], "must contain 2 entries", id="empty"),
        pytest.param(["m/s"], "must contain 2 entries", id="too-few"),
        pytest.param([1, 2], "must be a non-empty string", id="not-text"),
        pytest.param(["m/s", ""], "must be a non-empty string", id="blank-entry"),
        pytest.param("", "must be a non-empty string", id="blank-shared"),
        pytest.param(["m/s", "rad"], "agree with the channel units", id="disagrees"),
        pytest.param("rad", "agree with the channel units", id="shared-disagrees"),
    ],
)
def test_target_schema_rejects_invalid_unit(unit: object, message: str) -> None:
    """A schema that only fails on the first prediction is a schema persistence can store."""

    with pytest.raises(ValidationError, match=message):
        _target_schema(unit=unit)


def test_target_schema_rejects_empty_channels() -> None:
    with pytest.raises(ValidationError, match="at least one output"):
        ContinuousTargetSchema(
            channels=ChannelTable(),
            unit="m/s",
            name="hand_velocity",
            fs=20.0,
        )


def test_target_schema_metadata_is_deep_frozen() -> None:
    """It is fitted-model metadata, so it may not be edited through the schema."""

    schema = _target_schema(attrs={"task": "center_out", "nested": {"block": 1}})

    with pytest.raises(TypeError):
        schema.attrs["task"] = "other"
    with pytest.raises(TypeError):
        schema.attrs["nested"]["block"] = 2


def test_target_schema_stamps_given_clock() -> None:
    fitted = Clock(name="task", type="host")
    other = Clock(name="neural", type="acquisition")
    schema = _target_schema(clock=fitted)
    time = np.arange(4, dtype=np.float64) * 0.05

    assert schema.build(np.zeros((4, 2)), time).clock == fitted
    assert schema.build(np.zeros((4, 2)), time, clock=other).clock == other


def test_building_rejects_non_clock() -> None:
    with pytest.raises(ValidationError, match="clock must be a Clock"):
        _target_schema().build(np.zeros((4, 2)), np.arange(4, dtype=np.float64), clock="task")
