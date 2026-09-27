#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Immutable built-in stage specifications for native realtime pipelines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from hashlib import sha256
from math import isfinite
from typing import Any, ClassVar, Literal, TypeAlias

from neurale.exceptions import ValidationError


def _positive(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValidationError(
            f"{name} must be a {'non-negative' if allow_zero else 'positive'} integer."
        )
    return value


def _positive_float(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0
    ):
        raise ValidationError(f"{name} must be positive and finite.")
    return float(value)


def _duration_seconds(value: object, name: str) -> float:
    result = _positive_float(value, name)
    nanoseconds = Decimal(str(result)) * 1_000_000_000
    if nanoseconds != nanoseconds.to_integral_value() or nanoseconds > (1 << 64) - 1:
        raise ValidationError(f"{name} must resolve to a positive integer number of nanoseconds.")
    return result


def _duration_ns(value: float) -> int:
    return int(Decimal(str(value)) * 1_000_000_000)


def _nonnegative(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{name} must be a non-negative integer.")
    return value


def _identifier(value: int, name: str) -> int:
    return _positive(value, name)


def _floats(values: object, name: str, *, nonempty: bool = True) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValidationError(f"{name} must be a sequence of finite numbers.")
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a sequence of finite numbers.") from exc
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw):
        raise ValidationError(f"{name} must be a sequence of finite numbers.")
    result = tuple(float(value) for value in raw)
    if (nonempty and not result) or not all(isfinite(value) for value in result):
        raise ValidationError(f"{name} must contain finite numbers.")
    return result


def _strings(values: object, name: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValidationError(f"{name} must be a sequence of strings.")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError(f"{name} must be a sequence of strings.") from exc
    if (nonempty and not result) or any(
        not isinstance(value, str) or not value for value in result
    ):
        raise ValidationError(f"{name} must contain non-empty strings.")
    return result


def _indices(values: object, name: str, *, nonempty: bool = False) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValidationError(f"{name} must be a sequence of non-negative integers.")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError(f"{name} must be a sequence of non-negative integers.") from exc
    if (nonempty and not result) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in result
    ):
        raise ValidationError(f"{name} must contain non-negative integers.")
    return result


def _channel_selectors(
    values: object, name: str, *, nonempty: bool = True
) -> tuple[int | str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValidationError(f"{name} must be a sequence of channel names or indices.")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError(f"{name} must be a sequence of channel names or indices.") from exc
    if nonempty and not result:
        raise ValidationError(f"{name} must not be empty.")
    if not result:
        return ()
    if all(type(value) is int and value >= 0 for value in result):
        normalized: tuple[int | str, ...] = result
    elif all(isinstance(value, str) and value for value in result):
        normalized = result
    else:
        raise ValidationError(
            f"{name} must contain either non-negative indices or non-empty names, not both."
        )
    if len(set(normalized)) != len(normalized):
        raise ValidationError(f"{name} must not contain duplicates.")
    return normalized


def _choice(value: str, choices: tuple[str, ...], name: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValidationError(f"{name} must be one of: {', '.join(choices)}.")
    return value


def _filter_design(
    filter_type: str,
    cutoff_hz: object,
    filter_order: object,
    filter_kind: str,
    passband_ripple_db: object,
    stopband_attenuation_db: object,
) -> tuple[str, float | tuple[float, float], int, str, float | None, float | None]:
    filter_type = _choice(
        filter_type, ("lowpass", "highpass", "bandpass", "bandstop"), "filter_type"
    )
    filter_order = _positive(filter_order, "filter_order")
    filter_kind = _choice(filter_kind, ("butterworth", "bessel", "elliptic"), "filter_kind")
    if filter_kind == "bessel" and filter_order > 16:
        raise ValidationError("Bessel filter_order must not exceed 16.")

    expected = 2 if filter_type in ("bandpass", "bandstop") else 1
    if isinstance(cutoff_hz, bool):
        raise ValidationError("cutoff_hz must contain positive finite frequencies.")
    raw = (cutoff_hz,) if isinstance(cutoff_hz, int | float) else cutoff_hz
    values = _floats(raw, "cutoff_hz")
    if len(values) != expected or any(value <= 0.0 for value in values):
        label = "two frequencies" if expected == 2 else "one frequency"
        raise ValidationError(f"{filter_type} cutoff_hz must contain {label}.")
    if expected == 2 and values[0] >= values[1]:
        raise ValidationError("band filter cutoff_hz must be strictly increasing.")
    normalized_cutoff: float | tuple[float, float] = values if expected == 2 else values[0]

    if filter_kind == "elliptic":
        if passband_ripple_db is None or stopband_attenuation_db is None:
            raise ValidationError(
                "elliptic filters require passband_ripple_db and stopband_attenuation_db."
            )
        passband_ripple_db = _positive_float(passband_ripple_db, "passband_ripple_db")
        stopband_attenuation_db = _positive_float(
            stopband_attenuation_db, "stopband_attenuation_db"
        )
        if passband_ripple_db >= stopband_attenuation_db:
            raise ValidationError("passband_ripple_db must be less than stopband_attenuation_db.")
    elif passband_ripple_db is not None or stopband_attenuation_db is not None:
        raise ValidationError(
            "passband_ripple_db and stopband_attenuation_db are only valid for elliptic filters."
        )
    return (
        filter_type,
        normalized_cutoff,
        filter_order,
        filter_kind,
        passband_ripple_db,
        stopband_attenuation_db,
    )


class _Stage:
    kind: ClassVar[str]

    def to_document(self) -> dict[str, object]:
        return {"kind": self.kind, **asdict(self)}

    def _add_to(self, builder: Any, native: Any, streaming: Any) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SpatialReferenceStage(_Stage):
    """Subtract a mean or median reference from every sampled channel."""

    kind: ClassVar[str] = "spatial_reference"
    reference_channels: tuple[int, ...] = ()
    statistic: Literal["mean", "median"] = "mean"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference_channels", _indices(self.reference_channels, "reference_channels")
        )
        object.__setattr__(
            self, "statistic", _choice(self.statistic, ("mean", "median"), "statistic")
        )

    def _add_to(self, builder: Any, native: Any, streaming: Any) -> None:
        builder.add_spatial_reference(
            self.reference_channels,
            getattr(native._SpatialReferenceStatistic, self.statistic.upper()),
        )


@dataclass(frozen=True, slots=True)
class BadChannelRemovalStage(_Stage):
    """Remove explicit bad channels and channels outside an impedance range."""

    kind: ClassVar[str] = "bad_channel_removal"
    bad_channels: tuple[int | str, ...] = ()
    min_impedance_ohm: float | None = None
    max_impedance_ohm: float | None = None

    def __post_init__(self) -> None:
        bad_channels = _channel_selectors(self.bad_channels, "bad_channels", nonempty=False)
        minimum = self._impedance_limit(self.min_impedance_ohm, "min_impedance_ohm")
        maximum = self._impedance_limit(self.max_impedance_ohm, "max_impedance_ohm")
        if not bad_channels and minimum is None and maximum is None:
            raise ValidationError(
                "BadChannelRemovalStage requires bad_channels or an impedance limit."
            )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValidationError("min_impedance_ohm must not exceed max_impedance_ohm.")
        object.__setattr__(self, "bad_channels", bad_channels)
        object.__setattr__(self, "min_impedance_ohm", minimum)
        object.__setattr__(self, "max_impedance_ohm", maximum)

    @staticmethod
    def _impedance_limit(value: object, name: str) -> float | None:
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(value)
            or value < 0
        ):
            raise ValidationError(f"{name} must be finite and non-negative, or None.")
        return float(value)

    def _add_to(self, builder: Any, native: Any, streaming: Any) -> None:
        if self.bad_channels and isinstance(self.bad_channels[0], int):
            indices, names = self.bad_channels, ()
        else:
            indices, names = (), self.bad_channels
        builder.add_bad_channel_removal(
            indices,
            names,
            self.min_impedance_ohm,
            self.max_impedance_ohm,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FilterStage(_Stage):
    """Design and apply one causal IIR filter to a sampled stream."""

    kind: ClassVar[str] = "filter"
    filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"]
    cutoff_hz: float | tuple[float, float]
    filter_order: int = 4
    filter_kind: Literal["butterworth", "bessel", "elliptic"] = "butterworth"
    passband_ripple_db: float | None = None
    stopband_attenuation_db: float | None = None

    def __post_init__(self) -> None:
        values = _filter_design(
            self.filter_type,
            self.cutoff_hz,
            self.filter_order,
            self.filter_kind,
            self.passband_ripple_db,
            self.stopband_attenuation_db,
        )
        for name, value in zip(
            (
                "filter_type",
                "cutoff_hz",
                "filter_order",
                "filter_kind",
                "passband_ripple_db",
                "stopband_attenuation_db",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)

    def to_document(self) -> dict[str, object]:
        document = _Stage.to_document(self)
        if self.filter_kind != "elliptic":
            document.pop("passband_ripple_db")
            document.pop("stopband_attenuation_db")
        return document

    def _add_to(self, builder: Any, native: Any, streaming: Any) -> None:
        cutoff = (self.cutoff_hz,) if isinstance(self.cutoff_hz, float) else self.cutoff_hz
        builder.add_filter(
            self.filter_type,
            cutoff,
            self.filter_order,
            self.filter_kind,
            self.passband_ripple_db or 0.0,
            self.stopband_attenuation_db or 0.0,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LineNoiseFilterStage(_Stage):
    """Suppress mains interference and its harmonics with Butterworth notches."""

    kind: ClassVar[str] = "line_noise_filter"
    frequency_hz: float = 50.0
    bandwidth_hz: float = 2.0
    harmonics: int = 3
    filter_order: int = 2

    def __post_init__(self) -> None:
        frequency = _positive_float(self.frequency_hz, "frequency_hz")
        bandwidth = _positive_float(self.bandwidth_hz, "bandwidth_hz")
        harmonics = _positive(self.harmonics, "harmonics")
        order = _positive(self.filter_order, "filter_order")
        if bandwidth >= 2.0 * frequency:
            raise ValidationError("bandwidth_hz must be less than twice frequency_hz.")
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "bandwidth_hz", bandwidth)
        object.__setattr__(self, "harmonics", harmonics)
        object.__setattr__(self, "filter_order", order)

    def _add_to(self, builder: Any, native: Any, streaming: Any) -> None:
        builder.add_line_noise_filter(
            self.frequency_hz,
            self.bandwidth_hz,
            self.harmonics,
            self.filter_order,
        )


@dataclass(frozen=True, slots=True)
class ResampleStage(_Stage):
    """Rationally resample a stream with a prepared native anti-alias filter."""

    kind: ClassVar[str] = "resample"
    output_schema_id: int
    up: int
    down: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_schema_id", _identifier(self.output_schema_id, "output_schema_id")
        )
        object.__setattr__(self, "up", _positive(self.up, "up"))
        object.__setattr__(self, "down", _positive(self.down, "down"))

    def _add_to(self, builder: Any, native: Any, streaming: Any) -> None:
        builder.add_resampler(self.output_schema_id, self.up, self.down)


@dataclass(frozen=True, slots=True)
class Band:
    """One named half-open frequency interval in hertz."""

    name: str
    low_hz: float
    high_hz: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValidationError("a band name must be a string.")
        if isinstance(self.low_hz, bool) or not isinstance(self.low_hz, (int, float)):
            raise ValidationError("band frequencies must be finite numbers.")
        if isinstance(self.high_hz, bool) or not isinstance(self.high_hz, (int, float)):
            raise ValidationError("band frequencies must be finite numbers.")
        object.__setattr__(self, "low_hz", float(self.low_hz))
        object.__setattr__(self, "high_hz", float(self.high_hz))
        if (
            not self.name
            or not isfinite(self.low_hz)
            or not isfinite(self.high_hz)
            or self.low_hz < 0
            or self.high_hz <= self.low_hz
        ):
            raise ValidationError("a band requires a name and finite 0 <= low_hz < high_hz.")


def _bands(values: object) -> tuple[Band, ...]:
    if isinstance(values, (str, bytes)):
        raise ValidationError("bands must be a sequence of Band configurations.")
    try:
        raw = tuple(values)  # type: ignore[arg-type]
        result = tuple(item if isinstance(item, Band) else Band(**item) for item in raw)
    except TypeError as exc:
        raise ValidationError("bands must be a sequence of Band configurations.") from exc
    if not result:
        raise ValidationError("bands must not be empty.")
    return result


class _Feature:
    kind: ClassVar[str]

    def to_document(self) -> dict[str, object]:
        return {"kind": self.kind, **asdict(self)}

    def _native(self, native: Any, streaming: Any) -> object:
        raise NotImplementedError


_MULTITAPER_BANDPOWER_ALGORITHM_VERSION = "multitaper-bandpower-v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class MultitaperBandpowerFeature(_Feature):
    """One PMTM bandpower definition in a synchronized feature stage."""

    kind: ClassVar[str] = "multitaper_bandpower"
    bands: tuple[Band, ...]
    time_bandwidth: float = 2.5
    n_tapers: int | None = None
    weighting: Literal["unity", "eigen", "adaptive"] = "adaptive"
    detrend: Literal["none", "mean", "linear"] = "none"
    backend: Literal["auto", "builtin"] = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "bands", _bands(self.bands))
        if (
            isinstance(self.time_bandwidth, bool)
            or not isinstance(self.time_bandwidth, (int, float))
            or not isfinite(self.time_bandwidth)
            or self.time_bandwidth <= 0
        ):
            raise ValidationError("time_bandwidth must be positive and finite.")
        object.__setattr__(self, "time_bandwidth", float(self.time_bandwidth))
        object.__setattr__(
            self, "weighting", _choice(self.weighting, ("unity", "eigen", "adaptive"), "weighting")
        )
        maximum_tapers = int(2.0 * self.time_bandwidth - 1.0)
        object.__setattr__(
            self, "detrend", _choice(self.detrend, ("none", "mean", "linear"), "detrend")
        )
        object.__setattr__(self, "backend", _choice(self.backend, ("auto", "builtin"), "backend"))
        if maximum_tapers < 2:
            raise ValidationError("time_bandwidth must support at least two tapers.")
        if self.n_tapers is not None:
            object.__setattr__(self, "n_tapers", _positive(self.n_tapers, "n_tapers"))
            if self.n_tapers < 2 or self.n_tapers > maximum_tapers:
                raise ValidationError("time_bandwidth and n_tapers do not define a valid DPSS set.")
        if len({item.name for item in self.bands}) != len(self.bands):
            raise ValidationError("band names must be unique.")

    def to_document(self) -> dict[str, object]:
        document = _Feature.to_document(self)
        document["algorithm_version"] = _MULTITAPER_BANDPOWER_ALGORITHM_VERSION
        if self.detrend == "none":
            document.pop("detrend")
        if self.backend == "auto":
            document.pop("backend")
        return document

    def _native(self, native: Any, streaming: Any) -> object:
        value = native._MultitaperBandpowerBranchConfig()
        bands = []
        for item in self.bands:
            band = native._MultitaperBand()
            band.name, band.low_hz, band.high_hz = item.name, item.low_hz, item.high_hz
            bands.append(band)
        value.bands = bands
        value.time_bandwidth = self.time_bandwidth
        value.n_tapers = self.n_tapers or 0
        value.weighting = getattr(native._MultitaperWeighting, self.weighting.upper())
        value.detrend = getattr(native._FeatureDetrend, self.detrend.upper())
        value.backend = getattr(native._FeatureSpectralBackend, self.backend.upper())
        return value


_HILBERT_ENVELOPE_ALGORITHM_VERSION = "hilbert-envelope-v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class HilbertEnvelopeFeature(_Feature):
    """Band-pass and extract Hilbert envelopes in a synchronized feature stage."""

    kind: ClassVar[str] = "hilbert_envelope"
    bands: tuple[Band, ...]
    filter_order: int = 4
    filter_kind: Literal["butterworth", "bessel", "elliptic"] = "butterworth"
    passband_ripple_db: float | None = None
    stopband_attenuation_db: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bands", _bands(self.bands))
        if any(band.low_hz <= 0 for band in self.bands):
            raise ValidationError("Hilbert band low_hz must be positive.")
        _, _, order, kind, ripple, attenuation = _filter_design(
            "bandpass",
            (self.bands[0].low_hz, self.bands[0].high_hz),
            self.filter_order,
            self.filter_kind,
            self.passband_ripple_db,
            self.stopband_attenuation_db,
        )
        object.__setattr__(self, "filter_order", order)
        object.__setattr__(self, "filter_kind", kind)
        object.__setattr__(self, "passband_ripple_db", ripple)
        object.__setattr__(self, "stopband_attenuation_db", attenuation)
        if len({item.name for item in self.bands}) != len(self.bands):
            raise ValidationError("band names must be unique.")

    def to_document(self) -> dict[str, object]:
        document = _Feature.to_document(self)
        document["algorithm_version"] = _HILBERT_ENVELOPE_ALGORITHM_VERSION
        if self.filter_kind != "elliptic":
            document.pop("passband_ripple_db")
            document.pop("stopband_attenuation_db")
        return document

    def _native(self, native: Any, streaming: Any) -> object:
        value = native._HilbertEnvelopeBranchConfig()
        bands = []
        for item in self.bands:
            band = native._MultitaperBand()
            band.name, band.low_hz, band.high_hz = item.name, item.low_hz, item.high_hz
            bands.append(band)
        value.bands = bands
        for name in (
            "filter_order",
            "filter_kind",
        ):
            setattr(value, name, getattr(self, name))
        value.passband_ripple_db = self.passband_ripple_db or 0.0
        value.stopband_attenuation_db = self.stopband_attenuation_db or 0.0
        return value


_LMP_ALGORITHM_VERSION = "lmp-v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class LmpFeature(_Feature):
    """Low-pass and extract LMP values in a synchronized feature stage."""

    kind: ClassVar[str] = "lmp"
    cutoff_hz: float
    filter_order: int = 4
    filter_kind: Literal["butterworth", "bessel", "elliptic"] = "butterworth"
    passband_ripple_db: float | None = None
    stopband_attenuation_db: float | None = None

    def __post_init__(self) -> None:
        _, cutoff, order, kind, ripple, attenuation = _filter_design(
            "lowpass",
            self.cutoff_hz,
            self.filter_order,
            self.filter_kind,
            self.passband_ripple_db,
            self.stopband_attenuation_db,
        )
        object.__setattr__(self, "cutoff_hz", cutoff)
        object.__setattr__(self, "filter_order", order)
        object.__setattr__(self, "filter_kind", kind)
        object.__setattr__(self, "passband_ripple_db", ripple)
        object.__setattr__(self, "stopband_attenuation_db", attenuation)

    def to_document(self) -> dict[str, object]:
        document = _Feature.to_document(self)
        document["algorithm_version"] = _LMP_ALGORITHM_VERSION
        if self.filter_kind != "elliptic":
            document.pop("passband_ripple_db")
            document.pop("stopband_attenuation_db")
        return document

    def _native(self, native: Any, streaming: Any) -> object:
        value = native._LmpBranchConfig()
        for name in (
            "cutoff_hz",
            "filter_order",
            "filter_kind",
        ):
            setattr(value, name, getattr(self, name))
        value.passband_ripple_db = self.passband_ripple_db or 0.0
        value.stopband_attenuation_db = self.stopband_attenuation_db or 0.0
        return value


FeatureSpec: TypeAlias = MultitaperBandpowerFeature | HilbertEnvelopeFeature | LmpFeature


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureStage(_Stage):
    """Synchronously concatenate an ordered subset of the built-in feature extractors."""

    kind: ClassVar[str] = "feature"
    window_seconds: float = 0.2
    update_interval_seconds: float = 0.05
    features: tuple[FeatureSpec, ...] = (LmpFeature(cutoff_hz=4.0),)

    def __post_init__(self) -> None:
        for name in ("window_seconds", "update_interval_seconds"):
            object.__setattr__(self, name, _duration_seconds(getattr(self, name), name))
        if self.update_interval_seconds > self.window_seconds:
            raise ValidationError("update_interval_seconds must not exceed window_seconds.")
        if isinstance(self.features, (str, bytes)):
            raise ValidationError("features must be a sequence of feature configurations.")
        try:
            features = tuple(self.features)
        except TypeError as exc:
            raise ValidationError("features must be a sequence of feature configurations.") from exc
        allowed = (MultitaperBandpowerFeature, HilbertEnvelopeFeature, LmpFeature)
        if not 1 <= len(features) <= len(allowed) or any(
            not isinstance(feature, allowed) for feature in features
        ):
            raise ValidationError("features must contain one to three supported definitions.")
        if len({type(feature) for feature in features}) != len(features):
            raise ValidationError("each feature type may appear at most once.")
        object.__setattr__(self, "features", features)

    @property
    def algorithm_version(self) -> str:
        document = {
            "version": 1,
            "window_seconds": self.window_seconds,
            "update_interval_seconds": self.update_interval_seconds,
            "features": [feature.to_document() for feature in self.features],
        }
        payload = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"1:{sha256(payload).hexdigest()}"

    def to_document(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "window_seconds": self.window_seconds,
            "update_interval_seconds": self.update_interval_seconds,
            "features": [feature.to_document() for feature in self.features],
        }

    def _add_to(self, builder: Any, native: Any, streaming: Any) -> None:
        value = native._FeatureStackConfig()
        value.window_ns = _duration_ns(self.window_seconds)
        value.update_interval_ns = _duration_ns(self.update_interval_seconds)
        value.algorithm_version = self.algorithm_version
        value.branches = [feature._native(native, streaming) for feature in self.features]
        builder.add_feature(value)


@dataclass(frozen=True, slots=True)
class SpikeDetectorStage(_Stage):
    """Run bounded threshold detection and publish sparse spike blocks."""

    kind: ClassVar[str] = "spike_detector"
    output_schema_id: int
    output_signal_id: int
    block_capacity: int
    refractory_samples: int
    alignment_search_radius: int
    pre_samples: int
    post_samples: int
    channel_centers: tuple[float, ...]
    channel_thresholds: tuple[float, ...]
    electrode_groups: tuple[tuple[int, ...], ...]
    polarity: Literal["negative", "positive", "both"] = "negative"
    boundary_behavior: Literal["drop", "raise"] = "drop"
    overflow_policy: Literal["fault", "drop_newest"] = "fault"

    def __post_init__(self) -> None:
        for name in ("output_schema_id", "output_signal_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "block_capacity", _positive(self.block_capacity, "block_capacity"))
        for name in (
            "refractory_samples",
            "alignment_search_radius",
            "pre_samples",
            "post_samples",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(
            self, "channel_centers", _floats(self.channel_centers, "channel_centers")
        )
        object.__setattr__(
            self, "channel_thresholds", _floats(self.channel_thresholds, "channel_thresholds")
        )
        if len(self.channel_centers) != len(self.channel_thresholds):
            raise ValidationError("channel_centers must match channel_thresholds.")
        groups = tuple(
            _indices(group, "electrode_groups", nonempty=True) for group in self.electrode_groups
        )
        if not groups:
            raise ValidationError("electrode_groups must not be empty.")
        object.__setattr__(self, "electrode_groups", groups)
        object.__setattr__(
            self, "polarity", _choice(self.polarity, ("negative", "positive", "both"), "polarity")
        )
        object.__setattr__(
            self,
            "boundary_behavior",
            _choice(self.boundary_behavior, ("drop", "raise"), "boundary_behavior"),
        )
        object.__setattr__(
            self,
            "overflow_policy",
            _choice(self.overflow_policy, ("fault", "drop_newest"), "overflow_policy"),
        )

    def _add_to(self, builder: Any, native: Any, streaming: Any) -> None:
        value = native._SpikeDetectorConfig()
        for name in (
            "output_schema_id",
            "output_signal_id",
            "block_capacity",
            "refractory_samples",
            "alignment_search_radius",
            "pre_samples",
            "post_samples",
            "channel_centers",
            "channel_thresholds",
            "electrode_groups",
        ):
            setattr(value, name, getattr(self, name))
        value.polarity = getattr(native._DetectionPolarity, self.polarity.upper())
        value.boundary_behavior = getattr(native._BoundaryBehavior, self.boundary_behavior.upper())
        value.overflow_policy = getattr(native._SpikeOverflowPolicy, self.overflow_policy.upper())
        builder.add_spike_detector(value)


@dataclass(frozen=True, slots=True)
class FittedFeatureContract:
    """Feature identity and timing contract captured when a decoder was fitted."""

    feature_names: tuple[str, ...]
    feature_unit_symbols: tuple[str, ...]
    observation_rate_numerator: int
    observation_rate_denominator: int
    window_length_ns: int
    shift_ns: int
    algorithm_name: str
    algorithm_version: str
    source_stream: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", _strings(self.feature_names, "feature_names"))
        object.__setattr__(
            self,
            "feature_unit_symbols",
            _strings(self.feature_unit_symbols, "feature_unit_symbols", nonempty=False),
        )
        if len(self.feature_names) != len(self.feature_unit_symbols):
            raise ValidationError("feature_unit_symbols must match feature_names.")
        object.__setattr__(
            self,
            "observation_rate_numerator",
            _positive(
                self.observation_rate_numerator, "observation_rate_numerator", allow_zero=True
            ),
        )
        object.__setattr__(
            self,
            "observation_rate_denominator",
            _positive(self.observation_rate_denominator, "observation_rate_denominator"),
        )
        object.__setattr__(
            self, "window_length_ns", _positive(self.window_length_ns, "window_length_ns")
        )
        object.__setattr__(self, "shift_ns", _positive(self.shift_ns, "shift_ns", allow_zero=True))
        if (self.observation_rate_numerator == 0) != (self.shift_ns == 0):
            raise ValidationError("irregular features require both zero rate and zero shift.")
        if any(
            not isinstance(value, str) or not value
            for value in (self.algorithm_name, self.algorithm_version, self.source_stream)
        ):
            raise ValidationError("algorithm and source stream identities must not be empty.")

    def native(self, native: Any, streaming: Any) -> object:
        value = native._FittedFeatureContract()
        value.feature_names = self.feature_names
        value.feature_unit_symbols = self.feature_unit_symbols
        value.observation_rate = streaming.RationalRate(
            self.observation_rate_numerator, self.observation_rate_denominator
        )
        value.window_length_ns = self.window_length_ns
        value.shift_ns = self.shift_ns
        value.algorithm_name = self.algorithm_name
        value.algorithm_version = self.algorithm_version
        value.source_stream = self.source_stream
        value.timestamp_reference = streaming.FeatureTimestampReference.WINDOW_CENTER
        return value


@dataclass(frozen=True, slots=True)
class LinearDecoderStage(_Stage):
    """Apply a fitted affine model to an explicitly selected feature set."""

    kind: ClassVar[str] = "linear_decoder"
    output_schema_id: int
    output_signal_id: int
    output_channel_set_id: int
    output_physical_unit: str
    feature_set_id: int
    selection: tuple[int, ...]
    selected_feature_names: tuple[str, ...]
    fitted_feature_contract: FittedFeatureContract
    n_features: int
    n_outputs: int
    coefficients: tuple[float, ...]
    intercept: tuple[float, ...]
    scaling: Literal["none", "standard", "minmax"] = "none"
    scaler_center: tuple[float, ...] = ()
    scaler_scale: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _normalize_decoder(self)
        object.__setattr__(self, "coefficients", _floats(self.coefficients, "coefficients"))
        object.__setattr__(self, "intercept", _floats(self.intercept, "intercept"))
        if (
            len(self.coefficients) != self.n_features * self.n_outputs
            or len(self.intercept) != self.n_outputs
        ):
            raise ValidationError(
                "linear decoder parameter shapes do not match n_features/n_outputs."
            )
        if self.n_features != len(self.selection):
            raise ValidationError("n_features must match selection.")

    def _add_to(self, builder: Any, native: Any, streaming: Any) -> None:
        value = native._LinearDecoderConfig()
        _set_decoder_common(value, self, native, streaming)
        model = native._LinearModelState()
        model.n_features, model.n_outputs = self.n_features, self.n_outputs
        model.coef, model.intercept = self.coefficients, self.intercept
        value.model = model
        if isinstance(self, LdaDecoderStage):
            value.classes = self.classes
        builder.add_linear_decoder(value)


@dataclass(frozen=True, slots=True)
class LdaDecoderStage(LinearDecoderStage):
    """Native LDA inference yielding one numeric class label per observation.

    ``classes`` follows model score order. Binary zero scores choose the first
    class; multiclass ties choose the first maximum. Labels are exact integers
    carried in float64 frames. Prefer ``decoder_stage`` to export a fitted decoder.
    """

    kind: ClassVar[str] = "lda_decoder"
    classes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        LinearDecoderStage.__post_init__(self)
        labels = tuple(self.classes)
        if any(isinstance(v, bool) or not isinstance(v, int) or abs(v) > 2**53 - 1 for v in labels):
            raise ValidationError("native classes must be exact float64 integer labels.")
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValidationError("classes must contain at least two unique labels.")
        if self.n_outputs != (1 if len(labels) == 2 else len(labels)):
            raise ValidationError("LDA model outputs must match classes.")
        object.__setattr__(self, "classes", labels)


@dataclass(frozen=True, slots=True)
class KalmanDecoderStage(_Stage):
    """Apply a fitted linear-Gaussian recursive decoder."""

    kind: ClassVar[str] = "kalman_decoder"
    output_schema_id: int
    output_signal_id: int
    output_channel_set_id: int
    output_physical_unit: str
    feature_set_id: int
    selection: tuple[int, ...]
    selected_feature_names: tuple[str, ...]
    fitted_feature_contract: FittedFeatureContract
    state_dim: int
    observation_dim: int
    transition: tuple[float, ...]
    transition_offset: tuple[float, ...]
    observation: tuple[float, ...]
    observation_offset: tuple[float, ...]
    process_covariance: tuple[float, ...]
    observation_covariance: tuple[float, ...]
    initial_state: tuple[float, ...]
    initial_covariance: tuple[float, ...]
    innovation_jitter: float = 0.0
    missing: Literal["error", "predict"] = "error"
    scaling: Literal["none", "standard", "minmax"] = "none"
    scaler_center: tuple[float, ...] = ()
    scaler_scale: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _normalize_decoder(self)
        object.__setattr__(self, "state_dim", _positive(self.state_dim, "state_dim"))
        object.__setattr__(
            self, "observation_dim", _positive(self.observation_dim, "observation_dim")
        )
        for name in (
            "transition",
            "transition_offset",
            "observation",
            "observation_offset",
            "process_covariance",
            "observation_covariance",
            "initial_state",
            "initial_covariance",
        ):
            object.__setattr__(self, name, _floats(getattr(self, name), name))
        expected = {
            "transition": self.state_dim * self.state_dim,
            "transition_offset": self.state_dim,
            "observation": self.observation_dim * self.state_dim,
            "observation_offset": self.observation_dim,
            "process_covariance": self.state_dim * self.state_dim,
            "observation_covariance": self.observation_dim * self.observation_dim,
            "initial_state": self.state_dim,
            "initial_covariance": self.state_dim * self.state_dim,
        }
        if any(len(getattr(self, name)) != size for name, size in expected.items()):
            raise ValidationError("Kalman decoder parameter shapes do not match their dimensions.")
        if self.observation_dim != len(self.selection):
            raise ValidationError("observation_dim must match selection.")
        if (
            isinstance(self.innovation_jitter, bool)
            or not isinstance(self.innovation_jitter, (int, float))
            or not isfinite(self.innovation_jitter)
            or self.innovation_jitter < 0
        ):
            raise ValidationError("innovation_jitter must be finite and non-negative.")
        object.__setattr__(self, "innovation_jitter", float(self.innovation_jitter))
        object.__setattr__(self, "missing", _choice(self.missing, ("error", "predict"), "missing"))

    def _add_to(self, builder: Any, native: Any, streaming: Any) -> None:
        value = native._KalmanDecoderConfig()
        _set_decoder_common(value, self, native, streaming)
        model = native._LinearGaussianModelState()
        for name in (
            "state_dim",
            "observation_dim",
            "transition",
            "transition_offset",
            "observation",
            "observation_offset",
            "process_covariance",
            "observation_covariance",
            "initial_state",
            "initial_covariance",
        ):
            setattr(model, name, getattr(self, name))
        value.model = model
        value.innovation_jitter = self.innovation_jitter
        value.missing = getattr(native._KalmanMissingPolicy, self.missing.upper())
        builder.add_kalman_decoder(value)


def _normalize_decoder(value: Any) -> None:
    for name in ("output_schema_id", "output_signal_id", "output_channel_set_id", "feature_set_id"):
        object.__setattr__(value, name, _identifier(getattr(value, name), name))
    object.__setattr__(value, "selection", _indices(value.selection, "selection", nonempty=True))
    object.__setattr__(
        value,
        "selected_feature_names",
        _strings(value.selected_feature_names, "selected_feature_names"),
    )
    if len(value.selection) != len(value.selected_feature_names):
        raise ValidationError("selection must match selected_feature_names.")
    if len(set(value.selection)) != len(value.selection):
        raise ValidationError("selection must not repeat a feature column.")
    if not isinstance(value.fitted_feature_contract, FittedFeatureContract):
        object.__setattr__(
            value, "fitted_feature_contract", FittedFeatureContract(**value.fitted_feature_contract)
        )
    if hasattr(value, "n_features"):
        object.__setattr__(value, "n_features", _positive(value.n_features, "n_features"))
    if hasattr(value, "n_outputs"):
        object.__setattr__(value, "n_outputs", _positive(value.n_outputs, "n_outputs"))
    object.__setattr__(
        value,
        "output_physical_unit",
        _choice(
            value.output_physical_unit,
            ("unspecified", "volts", "amperes", "dimensionless"),
            "output_physical_unit",
        ),
    )
    object.__setattr__(
        value, "scaling", _choice(value.scaling, ("none", "standard", "minmax"), "scaling")
    )
    object.__setattr__(
        value, "scaler_center", _floats(value.scaler_center, "scaler_center", nonempty=False)
    )
    object.__setattr__(
        value, "scaler_scale", _floats(value.scaler_scale, "scaler_scale", nonempty=False)
    )
    width = 0 if value.scaling == "none" else len(value.selection)
    if len(value.scaler_center) != width or len(value.scaler_scale) != width:
        raise ValidationError("scaler parameters must match the selected feature count.")
    if any(scale == 0.0 for scale in value.scaler_scale):
        raise ValidationError("scaler_scale values must be nonzero.")


def _set_decoder_common(target: Any, source: Any, native: Any, streaming: Any) -> None:
    for name in (
        "output_schema_id",
        "output_signal_id",
        "output_channel_set_id",
        "feature_set_id",
        "selection",
        "selected_feature_names",
        "scaler_center",
        "scaler_scale",
    ):
        setattr(target, name, getattr(source, name))
    target.output_physical_unit = getattr(
        streaming.PhysicalUnit, source.output_physical_unit.upper()
    )
    target.fitted_feature_contract = source.fitted_feature_contract.native(native, streaming)
    target.scaling = getattr(native._FeatureScaling, source.scaling.upper())


StageSpec: TypeAlias = (
    BadChannelRemovalStage
    | SpatialReferenceStage
    | FilterStage
    | LineNoiseFilterStage
    | ResampleStage
    | FeatureStage
    | SpikeDetectorStage
    | LinearDecoderStage
    | LdaDecoderStage
    | KalmanDecoderStage
)


_STAGE_TYPES = {
    value.kind: value
    for value in (
        BadChannelRemovalStage,
        SpatialReferenceStage,
        FilterStage,
        LineNoiseFilterStage,
        ResampleStage,
        FeatureStage,
        SpikeDetectorStage,
        LinearDecoderStage,
        LdaDecoderStage,
        KalmanDecoderStage,
    )
}


def supported_stage_kinds() -> tuple[str, ...]:
    """Return the stable identifiers accepted by version-1 pipeline plans."""

    return tuple(_STAGE_TYPES)


def stage_from_document(document: object) -> StageSpec:
    if not isinstance(document, dict):
        raise ValidationError("each stage document must be an object.")
    values = dict(document)
    kind = values.pop("kind", None)
    if not isinstance(kind, str):
        raise ValidationError(f"unsupported built-in pipeline stage kind: {kind!r}.")
    stage_type = _STAGE_TYPES.get(kind)
    if stage_type is None:
        raise ValidationError(f"unsupported built-in pipeline stage kind: {kind!r}.")
    try:
        for name, nested in (("fitted_feature_contract", FittedFeatureContract),):
            if name in values and isinstance(values[name], dict):
                values[name] = nested(**values[name])
        if kind == "feature":
            features = values.get("features")
            if not isinstance(features, list):
                raise ValidationError("feature stage features must be an array.")
            values["features"] = tuple(feature_from_document(value) for value in features)
        return stage_type(**values)
    except TypeError as exc:
        raise ValidationError(f"invalid {kind!r} stage document: {exc}") from exc


_FEATURE_TYPES = {
    value.kind: value for value in (MultitaperBandpowerFeature, HilbertEnvelopeFeature, LmpFeature)
}


def feature_from_document(document: object) -> FeatureSpec:
    if not isinstance(document, dict):
        raise ValidationError("each feature definition must be an object.")
    values = dict(document)
    kind = values.pop("kind", None)
    if not isinstance(kind, str):
        raise ValidationError(f"unsupported feature kind: {kind!r}.")
    feature_type = _FEATURE_TYPES.get(kind)
    if feature_type is None:
        raise ValidationError(f"unsupported feature kind: {kind!r}.")
    try:
        algorithms = {
            "multitaper_bandpower": (_MULTITAPER_BANDPOWER_ALGORITHM_VERSION, "PMTM"),
            "hilbert_envelope": (_HILBERT_ENVELOPE_ALGORITHM_VERSION, "Hilbert"),
            "lmp": (_LMP_ALGORITHM_VERSION, "LMP"),
        }
        if kind in algorithms:
            expected_version, algorithm_name = algorithms[kind]
            algorithm_version = values.pop("algorithm_version", None)
            if algorithm_version != expected_version:
                raise ValidationError(
                    f"unsupported {algorithm_name} algorithm_version: {algorithm_version!r}."
                )
        if kind in ("multitaper_bandpower", "hilbert_envelope") and "bands" in values:
            bands = values["bands"]
            if isinstance(bands, (str, bytes)):
                raise ValidationError("bands must be a sequence of band objects.")
            values["bands"] = tuple(
                Band(**item) if isinstance(item, dict) else item for item in bands
            )
        return feature_type(**values)
    except TypeError as exc:
        raise ValidationError(f"invalid {kind!r} feature document: {exc}") from exc


__all__ = [
    "BadChannelRemovalStage",
    "Band",
    "FeatureSpec",
    "FeatureStage",
    "FilterStage",
    "FittedFeatureContract",
    "HilbertEnvelopeFeature",
    "KalmanDecoderStage",
    "LdaDecoderStage",
    "LineNoiseFilterStage",
    "LinearDecoderStage",
    "LmpFeature",
    "MultitaperBandpowerFeature",
    "ResampleStage",
    "SpatialReferenceStage",
    "SpikeDetectorStage",
    "StageSpec",
    "supported_stage_kinds",
]
