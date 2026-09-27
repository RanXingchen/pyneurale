#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Codecs for the typed metadata a fitted decoder carries.

Everything here round-trips through the manifest rather than through an array:
a feature schema, a timeline, a channel table, and the target metadata a
continuous prediction is built from are all descriptions, and a description
that a person cannot read out of the manifest is a description no one can
check.

The feature schema is the one that matters most. Its fingerprint is the
compatibility boundary a decoder enforces before predicting, so it is stored
*and* recomputed on load: an artifact whose recorded fingerprint disagrees with
the fields beside it describes two different inputs at once, and is refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from neurale.data import ChannelTable, Clock
from neurale.data.channels import ChannelInfo
from neurale.exceptions import ValidationError

from ..schema import ContinuousTargetSchema, FeatureSchema
from . import _canonical as canonical
from ._errors import DecoderArtifactFormatError

# ------------------------------------------------------------------- clocks


def encode_clock(clock: Clock | None) -> dict[str, Any] | None:
    """Return the manifest form of a timeline declaration."""

    if clock is None:
        return None
    return {
        "name": clock.name,
        "type": clock.type,
        "rate": canonical.portable(clock.rate, path="clock.rate"),
        "epoch": clock.epoch,
        "offset": canonical.portable(clock.offset, path="clock.offset"),
        "drift": canonical.portable(clock.drift, path="clock.drift"),
        "synchronization_domain": clock.synchronization_domain,
        "attrs": canonical.portable(dict(clock.attrs), path="clock.attrs"),
    }


def decode_clock(payload: object, *, path: str) -> Clock | None:
    """Rebuild a timeline declaration from the manifest."""

    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise DecoderArtifactFormatError(f"{path} must be a JSON object or null.")
    return _build(
        Clock,
        path,
        name=canonical.require_typed(payload, "name", str, path=path),
        type=canonical.require_typed(payload, "type", str, path=path),
        rate=canonical.optional_number(payload, "rate", path=path),
        epoch=canonical.optional_text(payload, "epoch", path=path),
        offset=canonical.optional_number(payload, "offset", path=path) or 0.0,
        drift=canonical.optional_number(payload, "drift", path=path) or 0.0,
        synchronization_domain=canonical.optional_text(
            payload, "synchronization_domain", path=path
        ),
        attrs=canonical.require_typed(payload, "attrs", dict, path=path),
    )


# ----------------------------------------------------------------- channels


def encode_channels(channels: ChannelTable) -> list[dict[str, Any]]:
    """Return the manifest form of a channel table, in column order."""

    return [_encode_channel(channel, idx) for idx, channel in enumerate(channels.channels)]


def _encode_channel(channel: ChannelInfo, pos: int) -> dict[str, Any]:
    path = f"channels[{pos}]"
    return {
        "name": channel.name,
        "index": int(channel.index),
        "type": channel.type,
        "unit": channel.unit,
        "valid": bool(channel.valid),
        "bad": bool(channel.bad),
        "impedance": _encode_impedance(channel.impedance, path=path),
        "electrode": channel.electrode,
        "contact": None if channel.contact is None else int(channel.contact),
        "position": None if channel.pos is None else [float(v) for v in channel.pos],
        "attrs": canonical.portable(dict(channel.attrs), path=f"{path}.attrs"),
    }


def _encode_impedance(value: object, *, path: str) -> Any:
    # Channel impedance is genuinely complex-valued, which JSON has no notion
    # of, so a complex one is stored as its two components under a tagged
    # object rather than flattened to a magnitude that could not be undone.
    if value is None:
        return None
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return canonical.portable(float(value), path=f"{path}.impedance")


def _decode_impedance(value: object, *, path: str) -> complex | float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        real = canonical.optional_number(value, "real", path=path)
        imag = canonical.optional_number(value, "imag", path=path)
        if real is None or imag is None:
            raise DecoderArtifactFormatError(f"{path}.impedance must declare real and imag.")
        return complex(real, imag)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DecoderArtifactFormatError(f"{path}.impedance must be a number, an object, or null.")
    return float(value)


def decode_channels(payload: object, *, path: str) -> ChannelTable:
    """Rebuild a channel table from the manifest."""

    if not isinstance(payload, list):
        raise DecoderArtifactFormatError(f"{path} must be a JSON array.")
    infos: list[ChannelInfo] = []
    for pos, entry in enumerate(payload):
        where = f"{path}[{pos}]"
        if not isinstance(entry, Mapping):
            raise DecoderArtifactFormatError(f"{where} must be a JSON object.")
        raw_position = canonical.require(entry, "position", path=where)
        infos.append(
            _build(
                ChannelInfo,
                where,
                name=canonical.require_typed(entry, "name", str, path=where),
                index=canonical.require_typed(entry, "index", int, path=where),
                type=canonical.require_typed(entry, "type", str, path=where),
                unit=canonical.require_typed(entry, "unit", str, path=where),
                valid=canonical.require_typed(entry, "valid", bool, path=where),
                bad=canonical.require_typed(entry, "bad", bool, path=where),
                impedance=_decode_impedance(
                    canonical.require(entry, "impedance", path=where), path=where
                ),
                electrode=canonical.optional_text(entry, "electrode", path=where),
                contact=_optional_int(entry, "contact", path=where),
                pos=None if raw_position is None else _triple(raw_position, path=where),
                attrs=canonical.require_typed(entry, "attrs", dict, path=where),
            )
        )
    return _build(ChannelTable, path, channels=infos)


# ----------------------------------------------------------- feature schema


def encode_feature_schema(schema: FeatureSchema) -> dict[str, Any]:
    """Return the manifest form of a fitted feature schema."""

    return {
        "feature_names": list(schema.feature_names),
        "units": list(schema.units),
        "dtype": str(schema.dtype),
        "fs": canonical.portable(schema.fs, path="feature_schema.rate"),
        "window_size": canonical.portable(schema.window_size, path="feature_schema.window_size"),
        "shift": canonical.portable(schema.shift, path="feature_schema.shift"),
        "timestamp_reference": schema.timestamp_reference,
        "source_signal": schema.source_signal,
        "fingerprint": schema.fingerprint,
    }


def decode_feature_schema(payload: object, *, path: str) -> FeatureSchema:
    """Rebuild a feature schema and verify it against its recorded fingerprint."""

    if not isinstance(payload, Mapping):
        raise DecoderArtifactFormatError(f"{path} must be a JSON object.")
    dtype_name = canonical.require_typed(payload, "dtype", str, path=path)
    try:
        dtype = np.dtype(dtype_name)
    except TypeError as exc:
        raise DecoderArtifactFormatError(f"{path}.dtype is not a NumPy dtype: {exc}") from exc

    schema = _build(
        FeatureSchema,
        path,
        feature_names=canonical.text_sequence(payload, "feature_names", path=path),
        units=canonical.text_sequence(payload, "units", path=path),
        dtype=dtype,
        fs=canonical.optional_number(payload, "fs", path=path),
        window_size=canonical.optional_number(payload, "window_size", path=path),
        shift=canonical.optional_number(payload, "shift", path=path),
        timestamp_reference=canonical.optional_text(payload, "timestamp_reference", path=path),
        source_signal=canonical.optional_text(payload, "source_signal", path=path),
    )

    recorded = canonical.require_typed(payload, "fingerprint", str, path=path)
    if schema.fingerprint != recorded:
        raise DecoderArtifactFormatError(
            f"{path} does not match its recorded fingerprint: the stored fields hash to "
            f"{schema.fingerprint!r}, the manifest records {recorded!r}. The artifact describes "
            "two different inputs, so which one the decoder was fitted on is unknowable."
        )
    return schema


# ------------------------------------------------------------ target schema


def encode_target_schema(schema: ContinuousTargetSchema) -> dict[str, Any]:
    """Return the manifest form of the fitted continuous target metadata."""

    unit = schema.unit
    return {
        "channels": encode_channels(schema.channels),
        "unit": unit if isinstance(unit, str) else list(unit),
        "name": schema.name,
        "fs": float(schema.fs),
        "clock": encode_clock(schema.clock),
        "attrs": canonical.portable(dict(schema.attrs), path="target_schema.attrs"),
    }


def decode_target_schema(payload: object, *, path: str) -> ContinuousTargetSchema:
    """Rebuild the fitted continuous target metadata from the manifest."""

    if not isinstance(payload, Mapping):
        raise DecoderArtifactFormatError(f"{path} must be a JSON object.")
    unit = canonical.require(payload, "unit", path=path)
    if isinstance(unit, list):
        unit = canonical.text_sequence(payload, "unit", path=path)
    elif not isinstance(unit, str):
        raise DecoderArtifactFormatError(f"{path}.unit must be a string or a list of strings.")

    rate = canonical.optional_number(payload, "fs", path=path)
    if rate is None:
        raise DecoderArtifactFormatError(f"{path}.fs must be a number.")

    return _build(
        ContinuousTargetSchema,
        path,
        channels=decode_channels(canonical.require(payload, "channels", path=path), path=path),
        unit=unit,
        name=canonical.require_typed(payload, "name", str, path=path),
        fs=rate,
        clock=decode_clock(canonical.require(payload, "clock", path=path), path=f"{path}.clock"),
        attrs=canonical.require_typed(payload, "attrs", dict, path=path),
    )


# ----------------------------------------------------------------- helpers


def _build(factory: Any, path: str, **arguments: Any) -> Any:
    # A stored field can be individually well-formed and still describe
    # something the typed layer refuses -- a negative sampling rate, units that
    # disagree with the channels. Letting the constructor decide keeps one
    # definition of valid, and reports it as a malformed artifact rather than
    # as an argument error from a call the caller never made.
    try:
        return factory(**arguments)
    except ValidationError as exc:
        raise DecoderArtifactFormatError(
            f"{path} is not a valid {factory.__name__}: {exc}"
        ) from exc


def _optional_int(payload: Mapping[str, Any], key: str, *, path: str) -> int | None:
    value = canonical.require(payload, key, path=path)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise DecoderArtifactFormatError(f"{path}.{key} must be an integer or null.")
    return int(value)


def _triple(value: object, *, path: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise DecoderArtifactFormatError(f"{path}.position must be three numbers or null.")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise DecoderArtifactFormatError(f"{path}.position must contain numbers.")
    return (float(value[0]), float(value[1]), float(value[2]))


__all__ = [
    "decode_channels",
    "decode_clock",
    "decode_feature_schema",
    "decode_target_schema",
    "encode_channels",
    "encode_clock",
    "encode_feature_schema",
    "encode_target_schema",
]
