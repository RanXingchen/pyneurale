#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Codecs for the two preprocessing stages a decoder may own.

Both are part of the fitted decoder rather than of its configuration: the
scaler is the decoder's own frozen copy, whose statistics decide what the model
sees, and the context decides how many frames a prediction row spans. An
artifact that stored the configuration and re-derived the statistics would not
be a saved decoder at all.

A restored scaler is frozen again on the way in, because that is the state a
fitted decoder publishes: it can be read and applied, but not refitted behind
the decoder that runs it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from neurale.exceptions import ValidationError
from neurale.models.preprocessing import MinMaxScaler, StandardScaler, TemporalContext

from .._stages import FeatureScaler, FeatureStages
from . import _canonical as canonical
from ._artifact import ArrayReader, ArrayWriter
from ._errors import DecoderArtifactFormatError

#: Stable identifiers for the scalers a decoder can own. These name the *kind*
#: of transform, not the class that implements it: an artifact never records an
#: import path, so nothing in it can select what a load will import.
_STANDARD = "standard"
_MINMAX = "minmax"

#: The fitted statistics each scaler kind stores, in manifest order.
_STATISTICS = {
    _STANDARD: ("mean", "var", "scale"),
    _MINMAX: ("data_min", "data_max", "data_range", "scale", "min"),
}


def encode_scaler(scaler: FeatureScaler | None, writer: ArrayWriter, prefix: str) -> Any:
    """Write a fitted scaler's statistics and return its manifest entry."""

    if scaler is None:
        return None
    if isinstance(scaler, StandardScaler):
        kind = _STANDARD
        options: dict[str, Any] = {"with_mean": scaler.with_mean, "with_std": scaler.with_std}
        values = {"mean": scaler.mean_, "var": scaler.var_, "scale": scaler.scale_}
    elif isinstance(scaler, MinMaxScaler):
        kind = _MINMAX
        low, high = scaler.feature_range
        options = {"feature_range": [float(low), float(high)]}
        values = {
            "data_min": scaler.data_min_,
            "data_max": scaler.data_max_,
            "data_range": scaler.data_range_,
            "scale": scaler.scale_,
            "min": scaler.min_,
        }
    else:  # pragma: no cover - validate_scaler admits no third kind
        raise DecoderArtifactFormatError(
            f"a decoder owning a {type(scaler).__name__} cannot be saved by this format."
        )

    for name in _STATISTICS[kind]:
        writer.add(f"{prefix}.{name}", values[name])
    return {
        "kind": kind,
        "options": options,
        "n_samples": int(scaler.n_samples_),
        "n_features_in": int(scaler.n_features_in_),
    }


def decode_scaler(
    payload: object, reader: ArrayReader, prefix: str, *, path: str
) -> FeatureScaler | None:
    """Rebuild a fitted, frozen scaler from the manifest and its arrays."""

    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise DecoderArtifactFormatError(f"{path} must be a JSON object or null.")

    kind = canonical.require_typed(payload, "kind", str, path=path)
    if kind not in _STATISTICS:
        raise DecoderArtifactFormatError(
            f"{path}.kind is {kind!r}; this format knows {sorted(_STATISTICS)}."
        )
    options = canonical.require_typed(payload, "options", dict, path=path)
    n_samples = canonical.require_typed(payload, "n_samples", int, path=path)
    arrays = {name: reader.get(f"{prefix}.{name}") for name in _STATISTICS[kind]}

    try:
        if kind == _STANDARD:
            scaler: FeatureScaler = StandardScaler._restore_fitted(
                with_mean=canonical.require_typed(
                    options, "with_mean", bool, path=f"{path}.options"
                ),
                with_std=canonical.require_typed(options, "with_std", bool, path=f"{path}.options"),
                n_samples=n_samples,
                **arrays,
            )
        else:
            scaler = MinMaxScaler._restore_fitted(
                feature_range=decode_feature_range(options, path=f"{path}.options"),
                n_samples=n_samples,
                **arrays,
            )
    except ValidationError as exc:
        raise DecoderArtifactFormatError(f"{path} is not a valid fitted scaler: {exc}") from exc

    declared = canonical.require_typed(payload, "n_features_in", int, path=path)
    if scaler.n_features_in_ != declared:
        raise DecoderArtifactFormatError(
            f"{path}.n_features_in is {declared}, but the stored statistics describe "
            f"{scaler.n_features_in_} features."
        )
    # A published scaler is frozen; restoring one unfrozen would hand a caller
    # the ability to refit what a fitted decoder runs.
    scaler.freeze()
    return scaler


def encode_context(context: TemporalContext | None) -> Any:
    """Return the manifest form of a temporal context."""

    if context is None:
        return None
    return {"left": int(context.left), "right": int(context.right)}


def decode_context(payload: object, *, path: str) -> TemporalContext | None:
    """Rebuild a temporal context from the manifest."""

    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise DecoderArtifactFormatError(f"{path} must be a JSON object or null.")
    try:
        return TemporalContext(
            left=canonical.require_typed(payload, "left", int, path=path),
            right=canonical.require_typed(payload, "right", int, path=path),
        )
    except ValidationError as exc:
        raise DecoderArtifactFormatError(f"{path} is not a valid TemporalContext: {exc}") from exc


def encode_stages(stages: FeatureStages, writer: ArrayWriter, prefix: str) -> dict[str, Any]:
    """Return the manifest form of a fitted decoder's owned stages."""

    return {
        "scaler": encode_scaler(stages.scaler, writer, f"{prefix}.scaler"),
        "context": encode_context(stages.context),
    }


def decode_stages(payload: object, reader: ArrayReader, prefix: str, *, path: str) -> FeatureStages:
    """Rebuild a fitted decoder's owned stages from the manifest."""

    if not isinstance(payload, Mapping):
        raise DecoderArtifactFormatError(f"{path} must be a JSON object.")
    return FeatureStages(
        decode_scaler(
            canonical.require(payload, "scaler", path=path),
            reader,
            f"{prefix}.scaler",
            path=f"{path}.scaler",
        ),
        decode_context(
            canonical.require(payload, "context", path=path),
            path=f"{path}.context",
        ),
    )


def decode_feature_range(options: Mapping[str, Any], *, path: str) -> tuple[float, float]:
    """Return a stored ``feature_range`` as two finite floats.

    Shared by the fitted scaler and by the configuration recipe, because both
    read the same field out of the same artifact and neither may reach for a
    bare ``float()``: that turns ``"bad"`` into a ``ValueError`` no caller of
    this module is told to expect, and ``False`` into ``0.0``, which is a range
    end nobody stored. The ends must also be finite -- a scaler cannot map
    features onto an interval without them, and a manifest is written with
    ``allow_nan=False``, so a non-finite one came from somewhere else.
    """

    values = canonical.require_typed(options, "feature_range", list, path=path)
    if len(values) != 2:
        raise DecoderArtifactFormatError(
            f"{path}.feature_range must hold two numbers; the artifact stores {len(values)}."
        )
    for i, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise DecoderArtifactFormatError(
                f"{path}.feature_range[{i}] must be a number; got {type(item).__name__}."
            )
        if not math.isfinite(item):
            raise DecoderArtifactFormatError(
                f"{path}.feature_range[{i}] is {item!r}, and a scaler cannot map features "
                "onto a range without ends."
            )
    return (float(values[0]), float(values[1]))


__all__ = [
    "decode_context",
    "decode_feature_range",
    "decode_scaler",
    "decode_stages",
    "encode_context",
    "encode_scaler",
    "encode_stages",
]
