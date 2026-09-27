#!/usr/bin/env python3

"""What the format refuses, and how clearly it says so.

A persistence format is only as trustworthy as its refusals. Every test here
damages a valid artifact in one specific way -- a version it cannot read, a
field that contradicts another, a file that no longer matches what the manifest
says about it -- and asserts that the load fails with the error that names the
problem, rather than succeeding with a decoder that is quietly not the one that
was saved.

The pickle tests are the reason the format exists. An artifact may make a load
fail; it may never make a load execute something.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neurale.data import FeatureMatrix, SignalArray
from neurale.decoding import LinearDecoder
from neurale.decoding.persistence import (
    DecoderArtifactCorruptionError,
    DecoderArtifactError,
    DecoderArtifactFormatError,
    DecoderArtifactVersionError,
    load_decoder,
    read_manifest,
    save_decoder,
    supported_types,
)
from neurale.exceptions import ValidationError
from neurale.models import StandardScaler

# --------------------------------------------------------------------------------------
# Helpers for damaging an artifact after it was written
# --------------------------------------------------------------------------------------


def _manifest(artifact: Path) -> dict[str, Any]:
    return json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))


def _rewrite_manifest(artifact: Path, manifest: Mapping[str, Any]) -> None:
    (artifact / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _edit_manifest(artifact: Path, edit) -> None:
    manifest = _manifest(artifact)
    edit(manifest)
    _rewrite_manifest(artifact, manifest)


def _array_path(artifact: Path, name: str) -> Path:
    return artifact / "arrays" / f"{name}.npy"


def _reseal(artifact: Path, name: str) -> None:
    """Update the recorded digest so a damaged file passes the integrity check.

    Several tests are about a check *behind* the digest -- the dtype, the shape,
    the refusal to unpickle. Leaving the digest stale would stop the load at the
    first gate and prove only that the digest works, which its own test already
    does.
    """

    digest = sha256(_array_path(artifact, name).read_bytes()).hexdigest()
    _edit_manifest(artifact, lambda manifest: manifest["arrays"][name].update({"sha256": digest}))


def _npy_header(path: Path) -> tuple[tuple[int, ...], bool, np.dtype]:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        assert version == (1, 0)
        return np.lib.format.read_array_header_1_0(handle)


class Detonator:
    """An object whose reconstruction is observable -- and must never happen."""

    detonated = False

    def __reduce__(self):  # pragma: no cover - the whole point is that it is not called
        return (_detonate, ())


def _detonate() -> Detonator:  # pragma: no cover - see Detonator.__reduce__
    Detonator.detonated = True
    return Detonator()


@pytest.fixture
def artifact(linear: LinearDecoder, tmp_path: Path) -> Path:
    """A valid, freshly published artifact for a fitted linear decoder."""

    return save_decoder(linear, tmp_path / "decoder")


# --------------------------------------------------------------------------------------
# The envelope
# --------------------------------------------------------------------------------------


def test_non_artifact_path_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DecoderArtifactFormatError, match="not a decoder artifact directory"):
        load_decoder(tmp_path / "nothing-here")

    (tmp_path / "plain.txt").write_text("not an artifact", encoding="utf-8")
    with pytest.raises(DecoderArtifactFormatError, match="not a decoder artifact directory"):
        load_decoder(tmp_path / "plain.txt")


def test_directory_without_manifest_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(DecoderArtifactFormatError, match=r"has no manifest\.json"):
        load_decoder(tmp_path / "empty")


def test_non_json_manifest_is_rejected(artifact: Path) -> None:
    (artifact / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DecoderArtifactFormatError, match="not valid JSON"):
        load_decoder(artifact)


def test_non_object_manifest_is_rejected(artifact: Path) -> None:
    (artifact / "manifest.json").write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(DecoderArtifactFormatError, match="must be a JSON object"):
        load_decoder(artifact)


def test_foreign_artifact_marker_is_rejected(artifact: Path) -> None:
    _edit_manifest(artifact, lambda manifest: manifest.update({"artifact": "some.other.format"}))
    with pytest.raises(DecoderArtifactFormatError, match="not a decoder artifact"):
        load_decoder(artifact)


def test_unknown_major_version_is_rejected(artifact: Path) -> None:
    recorded = _manifest(artifact)["format_version"]["major"]
    _edit_manifest(
        artifact,
        lambda manifest: manifest["format_version"].update({"major": recorded + 1}),
    )
    with pytest.raises(DecoderArtifactVersionError, match=f"format version {recorded + 1}"):
        load_decoder(artifact)


def test_newer_minor_version_loads(artifact: Path, linear: LinearDecoder, continuous) -> None:
    # A minor bump adds fields an older reader can ignore, so it must read --
    # otherwise every additive change breaks artifacts it was designed not to.
    _edit_manifest(
        artifact,
        lambda manifest: manifest["format_version"].update(
            {"minor": manifest["format_version"]["minor"] + 1}
        ),
    )
    restored = load_decoder(artifact)
    assert np.array_equal(
        restored.predict(continuous.features).data,
        linear.predict(continuous.features).data,
    )


def test_unknown_decoder_type_is_rejected(artifact: Path) -> None:
    _edit_manifest(artifact, lambda manifest: manifest["decoder"].update({"type": "transformer"}))
    with pytest.raises(DecoderArtifactVersionError, match="'transformer' decoder") as caught:
        load_decoder(artifact)
    # The refusal has to say what *can* be read, or the caller is left guessing.
    for name in supported_types():
        assert name in str(caught.value)


def test_unknown_decoder_type_version_is_rejected(artifact: Path) -> None:
    recorded = _manifest(artifact)["decoder"]["type_version"]
    _edit_manifest(
        artifact,
        lambda manifest: manifest["decoder"].update({"type_version": recorded + 1}),
    )
    with pytest.raises(DecoderArtifactVersionError, match=f"version {recorded + 1}"):
        load_decoder(artifact)


@pytest.mark.parametrize("field", ["artifact", "format_version", "decoder", "arrays", "payload"])
def test_missing_envelope_field_is_named(artifact: Path, field: str) -> None:
    _edit_manifest(artifact, lambda manifest: manifest.pop(field))
    with pytest.raises(DecoderArtifactFormatError, match=f"manifest.{field} is missing"):
        load_decoder(artifact)


def test_malformed_envelope_field_is_rejected(artifact: Path) -> None:
    _edit_manifest(artifact, lambda manifest: manifest["decoder"].update({"type_version": "1"}))
    with pytest.raises(DecoderArtifactFormatError, match="type_version must be int"):
        load_decoder(artifact)


def test_unloadable_artifact_is_inspectable(artifact: Path) -> None:
    _edit_manifest(artifact, lambda manifest: manifest["decoder"].update({"type": "transformer"}))

    manifest = read_manifest(artifact)
    assert manifest["decoder"]["type"] == "transformer"
    assert manifest["writer"]["library"] == "neurale"
    with pytest.raises(DecoderArtifactVersionError):
        load_decoder(artifact)


# --------------------------------------------------------------------------------------
# The payload
# --------------------------------------------------------------------------------------


def test_feature_schema_fingerprint_mismatch_is_rejected(artifact: Path) -> None:
    _edit_manifest(
        artifact,
        lambda manifest: manifest["payload"]["feature_schema"].update(
            {
                "feature_names": [
                    f"renamed_{idx}"
                    for idx in range(len(manifest["payload"]["feature_schema"]["units"]))
                ]
            }
        ),
    )
    with pytest.raises(DecoderArtifactFormatError, match="recorded fingerprint"):
        load_decoder(artifact)


def test_invalid_feature_schema_is_rejected(artifact: Path) -> None:
    _edit_manifest(
        artifact,
        lambda manifest: manifest["payload"]["feature_schema"].update({"fs": -1.0}),
    )
    with pytest.raises(DecoderArtifactFormatError, match="not a valid FeatureSchema"):
        load_decoder(artifact)


def test_unknown_device_is_rejected(artifact: Path) -> None:
    _edit_manifest(artifact, lambda manifest: manifest["payload"].update({"device": "tpu"}))
    with pytest.raises(DecoderArtifactFormatError, match=r"payload\.device is 'tpu'"):
        load_decoder(artifact)


def test_features_mismatching_fitted_schema_are_rejected(artifact: Path, continuous) -> None:
    # The compatibility boundary a decoder enforces has to survive the round
    # trip: a restored decoder that accepted anything would be worse than one
    # that failed to load at all.
    restored = load_decoder(artifact)
    other = FeatureMatrix(
        data=np.asarray(continuous.features.data)[:, :3],
        fs=continuous.features.fs,
        feature_names=["other_0", "other_1", "other_2"],
        unit="Hz",
        shift=continuous.features.shift,
    )
    with pytest.raises(ValidationError, match="fitted feature schema"):
        restored.predict(other)


# --------------------------------------------------------------------------------------
# The arrays
# --------------------------------------------------------------------------------------


def test_missing_array_file_is_rejected(artifact: Path) -> None:
    _array_path(artifact, "model.coef").unlink()
    with pytest.raises(DecoderArtifactCorruptionError, match=r"'model\.coef' is missing"):
        load_decoder(artifact)


def test_undeclared_array_is_rejected(artifact: Path) -> None:
    _edit_manifest(artifact, lambda manifest: manifest["arrays"].pop("model.coef"))
    with pytest.raises(DecoderArtifactFormatError, match="not declared in its 'arrays' section"):
        load_decoder(artifact)


def test_altered_array_is_rejected(artifact: Path) -> None:
    declared = _manifest(artifact)["arrays"]["model.coef"]
    tampered = np.zeros(tuple(declared["shape"]), dtype=np.dtype(declared["dtype"]))
    np.save(_array_path(artifact, "model.coef"), tampered, allow_pickle=False)

    with pytest.raises(DecoderArtifactCorruptionError, match="does not match its recorded digest"):
        load_decoder(artifact)


def test_truncated_array_is_rejected(artifact: Path) -> None:
    path = _array_path(artifact, "model.coef")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 8])

    with pytest.raises(DecoderArtifactCorruptionError, match="truncated or altered"):
        load_decoder(artifact)


def test_truncated_array_with_updated_digest_is_rejected(artifact: Path) -> None:
    # Damage plus a matching digest is what a re-written artifact looks like, so
    # the reader must not be relying on the checksum alone to notice.
    path = _array_path(artifact, "model.coef")
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 8])
    _reseal(artifact, "model.coef")

    with pytest.raises(DecoderArtifactCorruptionError, match=r"model\.coef") as caught:
        load_decoder(artifact)
    assert "truncated" in str(caught.value) or "plain array" in str(caught.value)


def test_wrong_dtype_array_is_rejected(artifact: Path) -> None:
    declared = _manifest(artifact)["arrays"]["model.coef"]
    np.save(
        _array_path(artifact, "model.coef"),
        np.zeros(tuple(declared["shape"]), dtype=np.float32),
        allow_pickle=False,
    )
    _reseal(artifact, "model.coef")

    with pytest.raises(DecoderArtifactCorruptionError, match="has dtype '<f4'"):
        load_decoder(artifact)


def test_wrong_shape_array_is_rejected(artifact: Path) -> None:
    declared = _manifest(artifact)["arrays"]["model.coef"]
    reshaped = np.zeros((*tuple(declared["shape"])[::-1], 1), dtype=np.dtype(declared["dtype"]))
    np.save(_array_path(artifact, "model.coef"), reshaped, allow_pickle=False)
    _reseal(artifact, "model.coef")

    with pytest.raises(DecoderArtifactCorruptionError, match="has shape"):
        load_decoder(artifact)


def test_invalid_declared_dtype_is_rejected(artifact: Path) -> None:
    # NumPy raises a bare TypeError for one it does not recognize; a malformed
    # artifact must not surface as an exception from a call the caller never
    # made.
    _edit_manifest(
        artifact,
        lambda manifest: manifest["arrays"]["model.coef"].update({"dtype": "not-a-dtype"}),
    )
    with pytest.raises(DecoderArtifactFormatError, match="is not a NumPy dtype"):
        load_decoder(artifact)


def test_declared_object_dtype_is_rejected_before_read(artifact: Path) -> None:
    _edit_manifest(
        artifact,
        lambda manifest: manifest["arrays"]["model.coef"].update({"dtype": "|O"}),
    )
    with pytest.raises(DecoderArtifactFormatError, match="never stores"):
        load_decoder(artifact)


@pytest.mark.parametrize("shape", [[-1, 6], ["2", 6], [True, 6]], ids=["negative", "text", "bool"])
def test_invalid_declared_shape_is_rejected(artifact: Path, shape: list) -> None:
    _edit_manifest(
        artifact,
        lambda manifest: manifest["arrays"]["model.coef"].update({"shape": shape}),
    )
    with pytest.raises(DecoderArtifactFormatError, match="must be a non-negative integer"):
        load_decoder(artifact)


def test_pickled_array_payload_is_rejected(artifact: Path) -> None:
    Detonator.detonated = False
    np.save(
        _array_path(artifact, "model.coef"),
        np.array([Detonator()], dtype=object),
        allow_pickle=True,
    )
    _reseal(artifact, "model.coef")

    with pytest.raises(DecoderArtifactCorruptionError, match="never unpickles"):
        load_decoder(artifact)
    assert not Detonator.detonated


def test_pickled_array_is_rejected_under_object_dtype(
    artifact: Path,
) -> None:
    # Declaring the object dtype in the manifest does not buy the artifact
    # anything: the refusal happens while reading the file, before the declared
    # dtype is consulted at all.
    Detonator.detonated = False
    np.save(
        _array_path(artifact, "model.coef"),
        np.array([Detonator()], dtype=object),
        allow_pickle=True,
    )
    _reseal(artifact, "model.coef")
    _edit_manifest(
        artifact,
        lambda manifest: manifest["arrays"]["model.coef"].update({"dtype": "|O", "shape": [1]}),
    )

    with pytest.raises(DecoderArtifactCorruptionError, match="never unpickles"):
        load_decoder(artifact)
    assert not Detonator.detonated


# --------------------------------------------------------------------------------------
# Fitted state that contradicts itself
# --------------------------------------------------------------------------------------


def test_language_model_counts_must_match_vocabulary(beam, tmp_path: Path) -> None:
    # An eos row of transition counts would describe a corpus that continued
    # after a sequence ended. The artifact stores counts and recomputes the
    # distributions through the same normalization a fit uses, so this is the
    # one place the contradiction can be caught.
    published = save_decoder(beam, tmp_path / "beam")
    declared = _manifest(published)["arrays"]["model.transition_counts"]
    counts = np.load(_array_path(published, "model.transition_counts"))
    counts[beam.language_model.fitted_vocabulary_.eos_index, 0] = 3.0
    np.save(_array_path(published, "model.transition_counts"), counts, allow_pickle=False)
    _reseal(published, "model.transition_counts")
    assert list(counts.shape) == declared["shape"]

    with pytest.raises(DecoderArtifactFormatError, match="does not describe a valid fit"):
        load_decoder(published)


def test_vocabulary_smaller_than_counts_is_rejected(beam, tmp_path: Path) -> None:
    published = save_decoder(beam, tmp_path / "beam")
    _edit_manifest(
        artifact=published,
        edit=lambda manifest: manifest["payload"]["vocabulary"]["tokens"].append("extra"),
    )
    with pytest.raises((DecoderArtifactFormatError, DecoderArtifactCorruptionError)):
        load_decoder(published)


def test_class_permutation_with_repeated_column_is_rejected(lda, tmp_path: Path) -> None:
    # The stored permutation is what maps the model's own class order onto the
    # declared one. A value that is not a permutation would silently select the
    # wrong probability columns, which is worse than failing to load.
    published = save_decoder(lda, tmp_path / "lda")
    order = np.load(_array_path(published, "fitted.order"))
    np.save(_array_path(published, "fitted.order"), np.zeros_like(order), allow_pickle=False)
    _reseal(published, "fitted.order")

    with pytest.raises(DecoderArtifactFormatError, match="repeats an index"):
        load_decoder(published)


def test_class_permutation_outside_declared_classes_is_rejected(lda, tmp_path: Path) -> None:
    published = save_decoder(lda, tmp_path / "lda")
    order = np.load(_array_path(published, "fitted.order"))
    np.save(_array_path(published, "fitted.order"), order + 1, allow_pickle=False)
    _reseal(published, "fitted.order")

    with pytest.raises(DecoderArtifactFormatError, match="not a permutation"):
        load_decoder(published)


def test_wrong_kind_index_vector_is_rejected(lda, tmp_path: Path) -> None:
    # Casting a stored float to an index type truncates, so a value that was
    # never an index would silently become one.
    published = save_decoder(lda, tmp_path / "lda")
    order = np.load(_array_path(published, "fitted.order")).astype(np.float64)
    np.save(_array_path(published, "fitted.order"), order, allow_pickle=False)
    _reseal(published, "fitted.order")
    _edit_manifest(
        published,
        lambda manifest: manifest["arrays"]["fitted.order"].update({"dtype": order.dtype.str}),
    )

    with pytest.raises(DecoderArtifactFormatError, match="must hold integers"):
        load_decoder(published)


# --------------------------------------------------------------------------------------
# The model contract a dimension check cannot see
# --------------------------------------------------------------------------------------


def test_selection_naming_other_features_is_rejected(kalman_selected, tmp_path: Path) -> None:
    # Every matrix still multiplies: the selection keeps its length, so nothing
    # downstream has a shape to complain about. What changes is *which* neural
    # features reach the model, under names the decoder goes on reporting.
    published = save_decoder(kalman_selected, tmp_path / "kalman")
    selection = np.load(_array_path(published, "fitted.selection"))
    selection[0] = 3
    np.save(_array_path(published, "fitted.selection"), selection, allow_pickle=False)
    _reseal(published, "fitted.selection")

    with pytest.raises(DecoderArtifactFormatError, match="selected_feature_names"):
        load_decoder(published)


def test_selection_outside_stored_schema_is_rejected(kalman_selected, tmp_path: Path) -> None:
    published = save_decoder(kalman_selected, tmp_path / "kalman")
    selection = np.load(_array_path(published, "fitted.selection"))
    selection[0] = 99
    np.save(_array_path(published, "fitted.selection"), selection, allow_pickle=False)
    _reseal(published, "fitted.selection")

    with pytest.raises(DecoderArtifactFormatError, match="indexes column 99"):
        load_decoder(published)


def test_selection_with_repeated_feature_is_rejected(kalman_selected, tmp_path: Path) -> None:
    published = save_decoder(kalman_selected, tmp_path / "kalman")
    selection = np.load(_array_path(published, "fitted.selection"))
    selection[0] = selection[1]
    np.save(_array_path(published, "fitted.selection"), selection, allow_pickle=False)
    _reseal(published, "fitted.selection")

    with pytest.raises(DecoderArtifactFormatError, match="repeats an index"):
        load_decoder(published)


def test_unobserved_selection_is_rejected(kalman_selected, tmp_path: Path) -> None:
    # Dropping the selection entirely, and the names with it, leaves an
    # artifact that is internally consistent right up to the model: the decoder
    # would feed six features to a model estimated on three.
    published = save_decoder(kalman_selected, tmp_path / "kalman")

    def unselect(manifest: dict[str, Any]) -> None:
        manifest["payload"]["fitted"]["has_selection"] = False
        manifest["payload"]["fitted"]["selected_feature_names"] = list(
            manifest["payload"]["feature_schema"]["feature_names"]
        )

    _edit_manifest(published, unselect)
    with pytest.raises(DecoderArtifactFormatError, match="selects 6 features"):
        load_decoder(published)


def test_inconsistent_segment_lengths_are_rejected(kalman, tmp_path: Path) -> None:
    published = save_decoder(kalman, tmp_path / "kalman")
    lengths = np.load(_array_path(published, "fitted.segment_lengths"))
    np.save(_array_path(published, "fitted.segment_lengths"), lengths + 1, allow_pickle=False)
    _reseal(published, "fitted.segment_lengths")

    with pytest.raises(DecoderArtifactFormatError, match="segment lengths sum to"):
        load_decoder(published)


def test_wrong_width_target_schema_is_rejected(kalman, tmp_path: Path) -> None:
    published = save_decoder(kalman, tmp_path / "kalman")
    _edit_manifest(
        published,
        lambda manifest: manifest["payload"]["target_schema"]["channels"].pop(),
    )
    with pytest.raises(DecoderArtifactFormatError, match="dimensional state"):
        load_decoder(published)


@pytest.mark.parametrize("name", ["linear", "ridge"])
def test_context_not_matching_model_width_is_rejected(
    name: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    decoder = request.getfixturevalue(name)
    published = save_decoder(decoder, tmp_path / name)
    _edit_manifest(
        published,
        lambda manifest: manifest["payload"]["configuration"].update(
            {"context": {"left": 1, "right": 1}}
        ),
    )
    with pytest.raises(DecoderArtifactFormatError, match="columns, but the stored feature schema"):
        load_decoder(published)


def _narrow_fitted_scaler(artifact: Path, statistics: tuple[str, ...]) -> None:
    """Drop one column from every fitted statistic, and keep the artifact tidy.

    Files, declared shapes, digests and the declared ``n_features_in`` are all
    brought back into agreement, so nothing but the stage contract itself can
    notice that the scaler no longer fits the schema it is stored beside.
    """

    width = 0
    for statistic in statistics:
        name = f"fitted.scaler.{statistic}"
        narrowed = np.load(_array_path(artifact, name))[:-1]
        width = int(narrowed.shape[0])
        np.save(_array_path(artifact, name), narrowed, allow_pickle=False)
        _reseal(artifact, name)

        def declare(manifest: dict[str, Any], name: str = name, width: int = width) -> None:
            manifest["arrays"][name].update({"shape": [width]})

        _edit_manifest(artifact, declare)

    _edit_manifest(
        artifact,
        lambda manifest: manifest["payload"]["configuration"]["scaler"].update(
            {"n_features_in": width}
        ),
    )


@pytest.mark.parametrize(
    ("name", "statistics"),
    [
        ("linear", ("data_min", "data_max", "data_range", "scale", "min")),
        ("lda", ("mean", "var", "scale")),
    ],
)
def test_scaler_fitted_on_other_features_is_rejected(
    name: str, statistics: tuple[str, ...], tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    # The scaler runs before the temporal context, so its width answers to the
    # feature schema rather than to the model. A model-width check passes right
    # over this: every stored shape downstream stays consistent, and the
    # artifact would restore a decoder that looks fitted and fails on the first
    # prediction instead of failing to load.
    decoder = request.getfixturevalue(name)
    published = save_decoder(decoder, tmp_path / name)
    _narrow_fitted_scaler(published, statistics)

    with pytest.raises(DecoderArtifactFormatError, match="scaler was fitted on"):
        load_decoder(published)


def test_regression_rank_above_stored_design_is_rejected(linear, tmp_path: Path) -> None:
    published = save_decoder(linear, tmp_path / "linear")
    _edit_manifest(
        published,
        lambda manifest: manifest["payload"]["fitted"].update({"rank": 10_000}),
    )
    with pytest.raises(
        DecoderArtifactFormatError, match=r"above the \d+ the stored design can have"
    ):
        load_decoder(published)


def test_lda_stage_width_mismatch_is_rejected(lda, tmp_path: Path) -> None:
    published = save_decoder(lda, tmp_path / "lda")
    _edit_manifest(
        published,
        lambda manifest: manifest["payload"]["configuration"].update(
            {"context": {"left": 2, "right": 0}}
        ),
    )
    with pytest.raises(DecoderArtifactFormatError, match="columns, but the stored feature schema"):
        load_decoder(published)


# --------------------------------------------------------------------------------------
# The configuration a load rebuilds
# --------------------------------------------------------------------------------------


def _set_feature_range(artifact: Path, values: list[Any]) -> None:
    _edit_manifest(
        artifact,
        lambda manifest: manifest["payload"]["configuration"]["scaler"]["options"].update(
            {"feature_range": values}
        ),
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (["bad", 1.0], "must be a number"),
        ([False, 1.0], "must be a number"),
        ([float("nan"), 1.0], "range without ends"),
        ([0.0, float("inf")], "range without ends"),
        ([1.0], "must hold two numbers"),
        ([0.0, 1.0, 2.0], "must hold two numbers"),
    ],
)
def test_malformed_feature_range_is_format_error(
    artifact: Path, values: list[Any], message: str
) -> None:
    # The configuration is rebuilt by calling a constructor, and a constructor
    # given raw JSON will raise whatever it raises: float("bad") is a bare
    # ValueError, and float(False) is 0.0 -- a range end nobody stored, accepted
    # silently. Callers of this module are told to expect three error types, so
    # the field is validated here rather than converted and hoped for.
    _set_feature_range(artifact, values)
    with pytest.raises(DecoderArtifactFormatError, match=message):
        load_decoder(artifact)


@pytest.mark.parametrize("values", [[1.0, 0.0], [1.0, 1.0]])
def test_unacceptable_feature_range_is_rejected(artifact: Path, values: list[float]) -> None:
    # Two numbers that are not an interval. The scaler is what knows that, so
    # the check stays there; what this module owns is the error type the
    # refusal arrives as.
    _set_feature_range(artifact, values)
    with pytest.raises(DecoderArtifactFormatError, match="not a valid scaler configuration"):
        load_decoder(artifact)


# --------------------------------------------------------------------------------------
# The device an artifact may claim
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["kalman", "linear", "ridge", "lda"])
def test_unrestorable_device_is_rejected(
    name: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    # Every current codec rebuilds a CPU-only model. Accepting "cuda" would set
    # device_ to a device the restored parameters are demonstrably not on, and
    # "the fitted actual device" is exactly what the field is for.
    decoder = request.getfixturevalue(name)
    published = save_decoder(decoder, tmp_path / name)
    _edit_manifest(published, lambda manifest: manifest["payload"].update({"device": "cuda"}))

    with pytest.raises(DecoderArtifactFormatError, match="restores a fit on"):
        load_decoder(published)


# --------------------------------------------------------------------------------------
# The runtime_state flag has to mean something
# --------------------------------------------------------------------------------------


def test_runtime_flag_contradicting_payload_is_rejected(kalman, tmp_path: Path) -> None:
    published = save_decoder(kalman, tmp_path / "kalman", runtime_state=True)
    _edit_manifest(published, lambda manifest: manifest.update({"runtime_state": False}))

    with pytest.raises(DecoderArtifactFormatError, match="envelope and the payload disagree"):
        load_decoder(published)


def test_runtime_flag_without_payload_is_rejected(kalman, tmp_path: Path) -> None:
    published = save_decoder(kalman, tmp_path / "kalman")
    _edit_manifest(published, lambda manifest: manifest.update({"runtime_state": True}))

    with pytest.raises(DecoderArtifactFormatError, match="envelope and the payload disagree"):
        load_decoder(published)


def test_stateless_decoder_rejects_runtime_state(linear, tmp_path: Path) -> None:
    published = save_decoder(linear, tmp_path / "linear")
    _edit_manifest(published, lambda manifest: manifest.update({"runtime_state": True}))

    with pytest.raises(DecoderArtifactFormatError, match="carries no runtime state"):
        load_decoder(published)


def test_runtime_arrays_without_flag_are_rejected(kalman, tmp_path: Path) -> None:
    # The flag is what a caller inspects to decide whether an artifact can
    # resume a session, so an artifact that carries runtime arrays while
    # declaring none is refused rather than silently read either way.
    published = save_decoder(kalman, tmp_path / "kalman", runtime_state=True)

    def strip(manifest: dict[str, Any]) -> None:
        manifest["runtime_state"] = False
        manifest["payload"]["runtime"] = None

    _edit_manifest(published, strip)
    with pytest.raises(DecoderArtifactFormatError, match="declares runtime arrays"):
        load_decoder(published)


@pytest.mark.parametrize("field", ["runtime_state", "writer"])
def test_missing_stamp_is_rejected(artifact: Path, field: str) -> None:
    _edit_manifest(artifact, lambda manifest: manifest.pop(field))
    with pytest.raises(DecoderArtifactFormatError, match=f"manifest.{field} is missing"):
        load_decoder(artifact)


def test_malformed_writer_stamp_is_rejected(artifact: Path) -> None:
    _edit_manifest(artifact, lambda manifest: manifest["writer"].update({"version": 1}))
    with pytest.raises(DecoderArtifactFormatError, match=r"manifest\.writer\.version must be str"):
        load_decoder(artifact)


# --------------------------------------------------------------------------------------
# A resumed beam has to be a beam the search could have produced
# --------------------------------------------------------------------------------------


@pytest.fixture
def running(beam, steps: np.ndarray, tmp_path: Path) -> Path:
    """A beam decoder partway through a sequence, saved with its runtime state."""

    beam.update(steps[:2])
    return save_decoder(beam, tmp_path / "beam", runtime_state=True)


def test_ended_active_hypothesis_is_rejected(running: Path, beam) -> None:
    # Restoring one would extend a sequence past its own end, which the search
    # itself can never do.
    indices = np.load(_array_path(running, "runtime.active.indices"))
    indices[1] = beam.vocabulary.eos_index
    np.save(_array_path(running, "runtime.active.indices"), indices, allow_pickle=False)
    _reseal(running, "runtime.active.indices")

    with pytest.raises(DecoderArtifactFormatError, match="already emits"):
        load_decoder(running)


def test_wrong_length_active_hypothesis_is_rejected(running: Path) -> None:
    _edit_manifest(running, lambda manifest: manifest["payload"]["runtime"].update({"n_steps": 3}))
    with pytest.raises(DecoderArtifactFormatError, match="tokens after 3 steps"):
        load_decoder(running)


def test_beam_wider_than_bound_is_rejected(running: Path) -> None:
    _edit_manifest(
        running,
        lambda manifest: manifest["payload"]["configuration"].update({"beam_width": 1}),
    )
    with pytest.raises(DecoderArtifactFormatError, match="above the 1 this decoder"):
        load_decoder(running)


def test_beam_index_outside_vocabulary_is_rejected(running: Path) -> None:
    indices = np.load(_array_path(running, "runtime.active.indices"))
    indices[0] = 99
    np.save(_array_path(running, "runtime.active.indices"), indices, allow_pickle=False)
    _reseal(running, "runtime.active.indices")

    with pytest.raises(DecoderArtifactFormatError, match="names column 99"):
        load_decoder(running)


def test_non_numeric_beam_score_is_rejected(running: Path) -> None:
    scores = np.load(_array_path(running, "runtime.active.scores"))
    scores[0, 0] = np.nan
    np.save(_array_path(running, "runtime.active.scores"), scores, allow_pickle=False)
    _reseal(running, "runtime.active.scores")

    with pytest.raises(DecoderArtifactFormatError, match="holds a nan"):
        load_decoder(running)


def test_non_vector_beam_lengths_is_rejected(running: Path) -> None:
    lengths = np.load(_array_path(running, "runtime.active.lengths"))
    np.save(_array_path(running, "runtime.active.lengths"), lengths[0], allow_pickle=False)
    _reseal(running, "runtime.active.lengths")
    _edit_manifest(
        running,
        lambda manifest: manifest["arrays"]["runtime.active.lengths"].update({"shape": []}),
    )

    with pytest.raises(DecoderArtifactFormatError, match="must be 1D"):
        load_decoder(running)


def test_non_log_probability_prior_is_rejected(running: Path) -> None:
    prior = np.load(_array_path(running, "runtime.prior.transitions"))
    prior[0, 0] = 1.0
    np.save(_array_path(running, "runtime.prior.transitions"), prior, allow_pickle=False)
    _reseal(running, "runtime.prior.transitions")

    with pytest.raises(DecoderArtifactFormatError, match="must hold log probabilities"):
        load_decoder(running)


def test_wrong_shape_prior_is_rejected(running: Path) -> None:
    _edit_manifest(
        running,
        lambda manifest: manifest["payload"]["runtime"]["prior_vocabulary"]["tokens"].pop(0),
    )
    with pytest.raises(DecoderArtifactFormatError, match="prior vocabulary needs"):
        load_decoder(running)


# --------------------------------------------------------------------------------------
# What a save refuses
# --------------------------------------------------------------------------------------


def test_decoder_without_codec_cannot_be_saved(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="no decoder artifact codec"):
        save_decoder(StandardScaler(), tmp_path / "scaler")
    assert not (tmp_path / "scaler").exists()


def test_unfitted_decoder_cannot_be_saved(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="is not fitted"):
        save_decoder(LinearDecoder(), tmp_path / "unfitted")
    assert not (tmp_path / "unfitted").exists()


@pytest.mark.parametrize("keywords", [{"overwrite": "yes"}, {"runtime_state": 1}])
def test_save_flags_must_be_bools(
    linear: LinearDecoder, tmp_path: Path, keywords: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError, match="must be a bool"):
        save_decoder(linear, tmp_path / "decoder", **keywords)


@pytest.mark.parametrize(
    "value",
    [np.array([1.0, 2.0]), b"raw bytes", {"a", "b"}],
    ids=["ndarray", "bytes", "set"],
)
def test_metadata_requiring_pickle_is_rejected(continuous, tmp_path: Path, value: object) -> None:
    # The typed layer admits a wider range of attrs than JSON can carry. The
    # artifact refuses the difference at save time instead of dropping it, so a
    # round trip never returns metadata the caller did not store.
    target = SignalArray.from_array(
        np.asarray(continuous.target.data),
        fs=continuous.target.fs,
        time=continuous.features.time.copy(),
        channel_names=["x", "y"],
        channel_types="behavior",
        units="m",
        name="cursor",
        attrs={"payload": value},
    )
    decoder = LinearDecoder().fit(continuous.features, target)

    with pytest.raises(DecoderArtifactFormatError, match="this format does not store"):
        save_decoder(decoder, tmp_path / "decoder")
    assert sorted(tmp_path.iterdir()) == []


def test_saving_over_existing_artifact_requires_overwrite(
    artifact: Path, linear: LinearDecoder, continuous
) -> None:
    with pytest.raises(FileExistsError, match="overwrite=True"):
        save_decoder(linear, artifact)

    restored = load_decoder(artifact)
    assert np.array_equal(
        restored.predict(continuous.features).data,
        linear.predict(continuous.features).data,
    )


def test_overwrite_publishes_artifact_in_place(artifact: Path, ridge, continuous) -> None:
    save_decoder(ridge, artifact, overwrite=True)

    restored = load_decoder(artifact)
    assert type(restored) is type(ridge)
    assert np.array_equal(
        restored.predict(continuous.features).data,
        ridge.predict(continuous.features).data,
    )
    # The publish is a rename, not a merge: nothing of the previous artifact,
    # and no staging directory, survives beside it.
    assert sorted(path.name for path in artifact.parent.iterdir()) == [artifact.name]


def test_failed_save_leaves_previous_artifact(
    artifact: Path, continuous, linear: LinearDecoder
) -> None:
    target = SignalArray.from_array(
        np.asarray(continuous.target.data),
        fs=continuous.target.fs,
        time=continuous.features.time.copy(),
        channel_names=["x", "y"],
        channel_types="behavior",
        units="m",
        name="cursor",
        attrs={"payload": b"raw bytes"},
    )
    doomed = LinearDecoder().fit(continuous.features, target)

    with pytest.raises(DecoderArtifactFormatError):
        save_decoder(doomed, artifact, overwrite=True)

    restored = load_decoder(artifact)
    assert np.array_equal(
        restored.predict(continuous.features).data,
        linear.predict(continuous.features).data,
    )
    assert sorted(path.name for path in artifact.parent.iterdir()) == [artifact.name]


# --------------------------------------------------------------------------------------
# What the published files actually contain
# --------------------------------------------------------------------------------------


def test_artifact_holds_only_manifest_and_arrays(artifact: Path) -> None:
    files = sorted(path.relative_to(artifact).as_posix() for path in artifact.rglob("*"))
    assert files[0] == "arrays"
    assert "manifest.json" in files
    for name in files:
        assert name in ("arrays", "manifest.json") or (
            name.startswith("arrays/") and name.endswith(".npy")
        ), name


@pytest.mark.parametrize("name", ["kalman", "linear", "ridge", "lda", "beam"])
def test_stored_arrays_hold_no_executable_payload(
    name: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    decoder = request.getfixturevalue(name)
    published = save_decoder(decoder, tmp_path / name)

    stored = sorted((published / "arrays").glob("*.npy"))
    assert stored, "an artifact with no arrays would make this test vacuous"
    for path in stored:
        _shape, _fortran, dtype = _npy_header(path)
        # An object dtype is the only way a .npy file can carry a pickle, and
        # the header is where it would have to be declared.
        assert not dtype.hasobject
        assert dtype.kind in "biufU"
        with path.open("rb") as handle:
            np.load(handle, allow_pickle=False)


@pytest.mark.parametrize("name", ["kalman", "linear", "ridge", "lda", "beam"])
def test_manifest_names_no_import_path(
    name: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    decoder = request.getfixturevalue(name)
    published = save_decoder(decoder, tmp_path / name)
    text = (published / "manifest.json").read_text(encoding="utf-8")

    # A load selects its reader from a fixed table of type identifiers. If the
    # class or its module appeared here, something in the artifact could steer
    # what a load constructs -- which is exactly what this format does not do.
    assert type(decoder).__name__ not in text
    assert type(decoder).__module__ not in text
    for token in ("__reduce__", "__class__", "!!python", "copy_reg", "pickle"):
        assert token not in text


def test_manifest_is_canonical_and_reproducible(linear: LinearDecoder, tmp_path: Path) -> None:
    first = (save_decoder(linear, tmp_path / "a") / "manifest.json").read_bytes()
    second = (save_decoder(linear, tmp_path / "b") / "manifest.json").read_bytes()

    assert first == second
    assert first.endswith(b"\n")
    assert b", " not in first and b'": ' not in first


def test_declared_arrays_match_declaration(artifact: Path) -> None:
    manifest = read_manifest(artifact)
    assert manifest["arrays"], "a linear decoder stores model parameters"
    for name, entry in manifest["arrays"].items():
        path = _array_path(artifact, name)
        assert path.is_file(), name
        shape, _fortran, dtype = _npy_header(path)
        assert list(shape) == entry["shape"]
        assert dtype.str == entry["dtype"]
        assert sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_npy_headers_are_literal_dicts(artifact: Path) -> None:
    # A .npy header is a Python literal by specification; ast.literal_eval
    # parsing it is the check that nothing has smuggled a call into one.
    for path in sorted((artifact / "arrays").glob("*.npy")):
        with path.open("rb") as handle:
            np.lib.format.read_magic(handle)
            length = int.from_bytes(handle.read(2), "little")
            header = ast.literal_eval(handle.read(length).decode("latin1"))
        assert set(header) == {"descr", "fortran_order", "shape"}
        assert "O" not in str(header["descr"])


def test_artifact_errors_share_one_family() -> None:
    # Callers that only want "the artifact is unusable" should be able to catch
    # one thing, and callers that distinguish a version from corruption should
    # not have to catch a bare ValueError to do it.
    for error in (
        DecoderArtifactVersionError,
        DecoderArtifactFormatError,
        DecoderArtifactCorruptionError,
    ):
        assert issubclass(error, DecoderArtifactError)
        assert issubclass(error, ValueError)
