#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Canonical input and target schemas for classical decoders."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from neurale.data import ChannelTable, Clock, FeatureMatrix, SignalArray

# Metadata immutability and the shared/per-item unit rule are contracts of the
# typed data layer. Decoding reuses them rather than growing a second spelling
# that could drift out of agreement with SignalArray.
from neurale.data._helpers import freeze_metadata, validate_unit
from neurale.exceptions import ValidationError

from ._validation import optional_positive_float, optional_text

# The fingerprint is a stored contract, not a process-local hash. Every future
# change to the covered field set must bump this so that a schema recorded by
# an older release can never collide with a differently-meant newer one.
_FINGERPRINT_VERSION = 1
_FINGERPRINT_DIGEST_BYTES = 16

# Order matters: it is both the fingerprint payload order and the order in
# which require_compatible() reports the first difference it finds.
_COMPATIBILITY_FIELDS = (
    "feature_names",
    "units",
    "dtype",
    "fs",
    "window_size",
    "shift",
    "timestamp_reference",
    "source_signal",
)


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """Describe the feature frames a decoder consumes.

    The schema is the compatibility boundary between feature extraction and a
    decoder. It carries what makes two feature matrices interchangeable and
    nothing that merely varies between two runs of the same extraction: the
    frame count, the frame timestamps, the data, and ``t0`` are deliberately
    absent, so one schema describes a whole family of recordings.

    Parameters
    ----------
    feature_names : sequence of str
        Unique feature names in column order. Order participates in
        compatibility: the same names permuted describe a different input.
    units : sequence of str
        One unit per feature, in the same column order. A
        :class:`~neurale.data.arrays.FeatureMatrix` that shares one unit across
        every feature is expanded here, so the canonical form is always
        per-feature.
    dtype : numpy.dtype or dtype-like
        Element type of the feature data. A decoder converts nothing
        implicitly, so a different dtype is a different input.
    fs : float or None, optional
        Observation rate in hertz, or ``None`` for irregularly timed frames.
    window_size : float or None, optional
        Analysis-window duration in seconds.
    shift : float or None, optional
        Time shift between adjacent frames in seconds.
    timestamp_reference : str or None, optional
        What a frame timestamp refers to, such as ``"window_center"``. Two
        otherwise identical extractions that timestamp differently are not
        interchangeable, which is why this participates in compatibility.
    source_signal : str or None, optional
        Name of the signal the features were derived from.

    Raises
    ------
    neurale.exceptions.ValidationError
        If names, units, dtype, or timing metadata are invalid.

    Notes
    -----
    Every declared field participates in compatibility. A field that is
    ``None`` on both sides is simply two matching declarations of "unknown"; a
    field declared on one side only is a mismatch, because the decoder cannot
    show that the undeclared side agrees.
    """

    feature_names: Sequence[str]
    units: Sequence[str]
    dtype: Any
    fs: float | None = None
    window_size: float | None = None
    shift: float | None = None
    timestamp_reference: str | None = None
    source_signal: str | None = None
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        names = _validate_names(self.feature_names)
        units = _validate_units(self.units, len(names))
        dtype = _validate_dtype(self.dtype)

        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(
            self,
            "fs",
            optional_positive_float(self.fs, "fs"),
        )
        object.__setattr__(
            self,
            "window_size",
            optional_positive_float(self.window_size, "window_size"),
        )
        object.__setattr__(self, "shift", optional_positive_float(self.shift, "shift"))
        object.__setattr__(
            self,
            "timestamp_reference",
            optional_text(self.timestamp_reference, "timestamp_reference"),
        )
        object.__setattr__(
            self,
            "source_signal",
            optional_text(self.source_signal, "source_signal"),
        )
        object.__setattr__(self, "fingerprint", _fingerprint(self))

    @classmethod
    def from_feature_matrix(cls, X: FeatureMatrix) -> Self:
        """Derive the canonical schema of one prepared feature matrix.

        ``timestamp_reference`` is read from ``X.attrs`` when the
        extraction recorded one; the frame count, timestamps, and data are not
        part of a schema and are ignored here.
        """

        if not isinstance(X, FeatureMatrix):
            raise ValidationError("X must be a FeatureMatrix.")
        unit = X.unit
        units = [unit] * X.n_features if isinstance(unit, str) else list(unit)
        return cls(
            feature_names=list(X.feature_names),
            units=units,
            dtype=X.data.dtype,
            fs=X.fs,
            window_size=X.window_size,
            shift=X.shift,
            timestamp_reference=_attr_text(X.attrs, "timestamp_reference"),
            source_signal=X.source_signal,
        )

    @property
    def n_features(self) -> int:
        """Number of features, which is the length of ``feature_names``."""

        return len(self.feature_names)

    def is_compatible_with(self, other: FeatureSchema) -> bool:
        """Report whether another schema describes an interchangeable input."""

        if not isinstance(other, FeatureSchema):
            raise ValidationError("other must be a FeatureSchema.")
        return self.fingerprint == other.fingerprint

    def require_compatible(self, other: FeatureSchema, *, name: str = "X") -> None:
        """Raise unless another schema describes an interchangeable input.

        The message names the first field that differs, in the order the
        fingerprint covers them, because a fingerprint mismatch on its own does
        not tell a caller what to fix.
        """

        if not isinstance(other, FeatureSchema):
            raise ValidationError("other must be a FeatureSchema.")
        for attribute in _COMPATIBILITY_FIELDS:
            expected = getattr(self, attribute)
            found = getattr(other, attribute)
            if expected != found:
                raise ValidationError(
                    f"{name} do not match the fitted feature schema: {attribute} is {found!r}, "
                    f"expected {expected!r}."
                )
        if self.fingerprint != other.fingerprint:  # pragma: no cover - defensive
            raise ValidationError(
                f"{name} do not match the fitted feature schema: fingerprints differ."
            )


@dataclass(frozen=True, slots=True)
class ContinuousTargetSchema:
    """Describe the continuous target a decoder was fitted on.

    A continuous prediction is a :class:`~neurale.data.arrays.SignalArray`, and
    this is where its metadata comes from: the channels, units, name, rate, and
    clock of the fit target. The decoder therefore never invents output
    metadata, and a prediction is directly comparable with the target it was
    trained to reproduce.

    Parameters
    ----------
    channels : neurale.data.channels.ChannelTable
        One entry per predicted output, in column order.
    unit : str or sequence of str
        Shared unit or one unit per output; must agree with the channel units.
    name : str
        Non-empty target name, reused as the predicted signal name.
    fs : float
        Target sampling rate in hertz.
    clock : neurale.data.time.Clock or None, optional
        Clock the target timestamps belong to.
    attrs : mapping, optional
        Target metadata, carried onto the prediction unchanged. Deep-frozen
        using the closed value domain documented by :mod:`neurale.data`: this
        is fitted-model metadata, so it may not be edited in place through a
        decoder that has already been fitted.

    Raises
    ------
    neurale.exceptions.ValidationError
        If the channels, name, rate, clock, units, or metadata are invalid, or
        if the units disagree with the channel units.
    """

    channels: ChannelTable
    unit: str | Sequence[str]
    name: str
    fs: float
    clock: Clock | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.channels, ChannelTable):
            raise ValidationError("channels must be a ChannelTable.")
        if len(self.channels) == 0:
            raise ValidationError("channels must contain at least one output.")
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError("name must be a non-empty string.")
        rate = optional_positive_float(self.fs, "fs")
        if rate is None:
            raise ValidationError("fs must be provided.")
        if self.clock is not None and not isinstance(self.clock, Clock):
            raise ValidationError("clock must be a Clock or None.")
        # The unit contract is checked here rather than deferred to build():
        # a schema that looks valid until the first prediction is a schema that
        # persistence can store and reload before anyone finds out.
        unit = validate_unit(self.unit, len(self.channels), "unit")
        declared = [unit] * len(self.channels) if isinstance(unit, str) else list(unit)
        if self.channels.units != declared:
            raise ValidationError("unit must agree with the channel units.")
        object.__setattr__(self, "unit", unit if isinstance(unit, str) else tuple(unit))
        object.__setattr__(self, "fs", rate)
        object.__setattr__(self, "attrs", freeze_metadata(self.attrs))

    @classmethod
    def from_signal(cls, y: SignalArray) -> Self:
        """Capture the metadata of one continuous target."""

        if not isinstance(y, SignalArray):
            raise ValidationError("y must be a SignalArray.")
        return cls(
            channels=y.channels.copy(),
            unit=y.unit,
            name=y.name,
            fs=y.fs,
            clock=y.clock,
            attrs=dict(y.attrs),
        )

    @property
    def n_outputs(self) -> int:
        """Number of predicted outputs."""

        return len(self.channels)

    @property
    def output_names(self) -> list[str]:
        """Predicted output names in column order."""

        return self.channels.names

    def build(
        self,
        data: np.ndarray,
        time: np.ndarray,
        *,
        clock: Clock | None = None,
    ) -> SignalArray:
        """Wrap predicted values in a metadata-complete signal.

        ``time`` comes from the input frames rather than from the fit target:
        the metadata says what the outputs *are*, the input says *when* they
        were predicted. ``clock`` follows the timestamps for the same reason --
        a clock describes a time axis, so stamping the fit target's clock onto
        another recording's timestamps would describe neither. It defaults to
        the clock of the fit target.
        """

        if clock is not None and not isinstance(clock, Clock):
            raise ValidationError("clock must be a Clock or None.")
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValidationError("predicted values must be a 2D numpy.ndarray.")
        if data.shape[1] != self.n_outputs:
            raise ValidationError(
                f"predicted values must have {self.n_outputs} outputs; got {data.shape[1]}."
            )
        if data.shape[0] != time.shape[0]:
            raise ValidationError("predicted values must have one row per input frame.")
        return SignalArray(
            data=data,
            fs=self.fs,
            time=np.array(time, dtype=np.float64, copy=True),
            t0=float(time[0]) if time.size else 0.0,
            clock=self.clock if clock is None else clock,
            channels=self.channels.copy(),
            unit=self.unit,
            name=self.name,
            attrs=dict(self.attrs),
        )


def _validate_names(feature_names: object) -> tuple[str, ...]:
    if isinstance(feature_names, str) or not isinstance(feature_names, Sequence | np.ndarray):
        raise ValidationError("feature_names must be a sequence of strings.")
    names = tuple(str(name) if isinstance(name, np.str_) else name for name in feature_names)
    if not names:
        raise ValidationError("feature_names must contain at least one feature.")
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValidationError("feature_names must contain non-empty strings.")
    if len(set(names)) != len(names):
        raise ValidationError("feature_names must be unique.")
    return names


def _validate_units(units: object, n_features: int) -> tuple[str, ...]:
    if isinstance(units, str) or not isinstance(units, Sequence | np.ndarray):
        raise ValidationError("units must be a sequence of strings.")
    values = tuple(str(unit) if isinstance(unit, np.str_) else unit for unit in units)
    if len(values) != n_features:
        raise ValidationError("units must contain one unit per feature.")
    for unit in values:
        if not isinstance(unit, str) or not unit:
            raise ValidationError("units must contain non-empty strings.")
    return values


def _validate_dtype(dtype: object) -> np.dtype:
    try:
        resolved = np.dtype(dtype)
    except TypeError as exc:
        raise ValidationError(f"dtype must be a NumPy dtype: {exc}") from exc
    if not np.issubdtype(resolved, np.number) and not np.issubdtype(resolved, np.bool_):
        raise ValidationError("dtype must describe numeric feature data.")
    return resolved


def _attr_text(attrs: Mapping[str, Any], key: str) -> str | None:
    if key not in attrs:
        return None
    return optional_text(attrs[key], f"attrs[{key!r}]")


def _fingerprint(schema: FeatureSchema) -> str:
    # A canonical JSON payload keyed by field name, hashed with a fixed digest
    # size. JSON round-trips float64 exactly through repr(), and allow_nan is
    # off so a non-finite rate could never be encoded as a bare token.
    payload = {
        "version": _FINGERPRINT_VERSION,
        "feature_names": list(schema.feature_names),
        "units": list(schema.units),
        # str() rather than dtype.str: it distinguishes exactly what dtype
        # equality distinguishes, without writing the platform byte order into
        # the fingerprint of a native dtype.
        "dtype": str(schema.dtype),
        "fs": schema.fs,
        "window_size": schema.window_size,
        "shift": schema.shift,
        "timestamp_reference": schema.timestamp_reference,
        "source_signal": schema.source_signal,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.blake2b(encoded.encode("utf-8"), digest_size=_FINGERPRINT_DIGEST_BYTES)
    return digest.hexdigest()
