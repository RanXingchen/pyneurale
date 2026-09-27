#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Typed offline spike-waveform feature projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from neurale.data import ChannelInfo, FeatureMatrix, SpikeWaveformBatch
from neurale.exceptions import ValidationError
from neurale.models import LPP, PCA

_ProjectionEstimator: TypeAlias = PCA | LPP
WaveformMeasurement: TypeAlias = Literal["amplitude", "width", "energy"]

_MEASUREMENTS = ("amplitude", "width", "energy")


@dataclass(frozen=True, slots=True)
class _WaveformSchema:
    n_samples: int
    pre_samples: int
    post_samples: int
    fs: float
    channels: tuple[object, ...]
    flatten_order: Literal["C"] = "C"


class WaveformProjector:
    """Own a PCA or LPP estimator together with its fitted waveform schema.

    The wrapped estimator remains available through the read-only
    :attr:`estimator` property. If it is fitted outside this wrapper after a
    waveform fit, the stored schema is invalidated and later transformation is
    rejected instead of silently applying unrelated model state.
    """

    def __init__(self, estimator: _ProjectionEstimator) -> None:
        if not isinstance(estimator, (PCA, LPP)):
            raise ValidationError("estimator must be a neurale.models.PCA or LPP.")
        self._estimator = estimator
        self._waveform_schema: _WaveformSchema | None = None
        self._fitted_components: np.ndarray | None = None

    @property
    def estimator(self) -> _ProjectionEstimator:
        """Return the wrapped estimator without transferring ownership."""

        return self._estimator

    @property
    def device_(self) -> str:
        """Return the actual fitted device reported by the estimator."""

        return self._estimator.device_

    def _fit_transform(
        self,
        values: np.ndarray,
        schema: _WaveformSchema,
        n_spikes: int,
    ) -> np.ndarray:
        # A fit attempt invalidates the previous sorting contract even if the
        # estimator later rejects the new data. This avoids retaining a schema
        # that may no longer describe partially changed estimator state.
        self._waveform_schema = None
        self._fitted_components = None
        projected = _validated_projection(self._estimator.fit_transform(values), n_spikes)
        self._waveform_schema = schema
        self._fitted_components = self._estimator.components_
        return projected

    def _transform(
        self,
        values: np.ndarray,
        schema: _WaveformSchema,
        n_spikes: int,
    ) -> np.ndarray:
        current_components = self._estimator.components_
        if self._waveform_schema is None or self._fitted_components is None:
            raise ValidationError(
                "WaveformProjector has no waveform schema; fit it first with "
                "project_waveform_features(..., fit=True)."
            )
        if current_components is not self._fitted_components:
            self._waveform_schema = None
            self._fitted_components = None
            raise ValidationError(
                "WaveformProjector estimator was refitted outside the wrapper; "
                "fit the waveform projector again before transform."
            )
        if self._waveform_schema != schema:
            raise ValidationError(
                "waveform schema does not match the schema used to fit the projection."
            )
        return _validated_projection(self._estimator.transform(values), n_spikes)

    def __repr__(self) -> str:
        fitted = self._waveform_schema is not None
        return f"WaveformProjector(estimator={self._estimator!r}, fitted={fitted})"


def project_waveform_features(
    batch: SpikeWaveformBatch,
    projection: WaveformProjector | None = None,
    *,
    fit: bool = True,
    measurements: Sequence[WaveformMeasurement] = ("amplitude", "width"),
) -> FeatureMatrix:
    """Project aligned spike waveforms and append simple measurements.

    Rows preserve the input batch order and correspond one-to-one with source
    spikes. Projection input flattens ``(sample, channel)`` in C order. When a
    :class:`WaveformProjector` wrapping :class:`neurale.models.PCA` or
    :class:`neurale.models.LPP` is supplied, ``fit=True`` calls the estimator's
    existing ``fit_transform`` implementation and records the waveform schema
    in the wrapper. ``fit=False`` first
    requires an exact schema match, then calls ``transform`` on its fitted
    state. Sorting does not implement either projection algorithm.

    Projection columns precede measurement columns. Measurements are emitted
    in the requested order:

    - ``amplitude`` is the signed, centered detection amplitude already stored
      by :class:`SpikeWaveformBatch`;
    - ``width`` is the interval in seconds from the aligned dominant extremum
      on the peak channel to the earliest following opposite extremum;
    - ``energy`` is the discrete integral ``sum(waveform**2) / fs``
      over every waveform sample and channel.

    Projection, amplitude, and energy require one shared source amplitude unit
    across channels. Width-only extraction remains valid for mixed-unit channel
    tables. The result uses ``(spike, feature)`` axis order, aligned spike times,
    and explicit source-spike linkage in ``FeatureMatrix.attrs``.

    Parameters
    ----------
    batch : neurale.data.SpikeWaveformBatch
        One continuous-segment waveform batch.
    projection : neurale.sorting.WaveformProjector or None, optional
        Sorting-owned wrapper around an existing PCA or LPP estimator. The
        estimator is fitted in place when ``fit=True`` and reused without
        refitting when ``fit=False``. A bare estimator is rejected because it
        cannot own the waveform schema lifecycle.
    fit : bool, default=True
        Select ``fit_transform`` versus ``transform`` for ``projection``.
    measurements : sequence of {"amplitude", "width", "energy"}
        Unique simple measurements to append. Pass an empty sequence for
        projection-only output.

    Returns
    -------
    neurale.data.FeatureMatrix
        Spike-major typed features linked to the source batch.

    Raises
    ------
    neurale.exceptions.ValidationError
        If inputs, measurement names, units, projector type/state, or numerical
        results violate the contract.
    """

    if not isinstance(batch, SpikeWaveformBatch):
        raise ValidationError("batch must be a SpikeWaveformBatch.")
    if not isinstance(fit, bool):
        raise ValidationError("fit must be a bool.")
    selected = _validate_measurements(measurements)
    projector_kind = _projector_kind(projection)
    if projection is None and not selected:
        raise ValidationError("at least one projection or measurement must be requested.")

    needs_amp_unit = projection is not None or bool({"amplitude", "energy"} & set(selected))
    amp_unit = _shared_amp_unit(batch) if needs_amp_unit else None

    columns: list[np.ndarray] = []
    names: list[str] = []
    units: list[str] = []
    if projection is not None:
        schema = _waveform_schema(batch)
        flattened = batch.waveforms.reshape(batch.n_spikes, batch.n_samples * batch.n_channels)
        projected = (
            projection._fit_transform(flattened, schema, batch.n_spikes)
            if fit
            else projection._transform(flattened, schema, batch.n_spikes)
        )
        columns.append(projected)
        names.extend(f"{projector_kind}_{idx:03d}" for idx in range(projected.shape[1]))
        projection_unit = amp_unit if projector_kind == "pca" else "a.u."
        units.extend([projection_unit] * projected.shape[1])

    for measurement in selected:
        values, unit = _measurement(batch, measurement, amp_unit)
        columns.append(values[:, np.newaxis])
        names.append(measurement)
        units.append(unit)

    data = columns[0] if len(columns) == 1 else np.concatenate(columns, axis=1)
    attrs: dict[str, object] = {
        "observation_axis": "spike",
        "time_reference": "aligned_spike",
        "timing": "irregular",
        "source_fs": batch.fs,
        "source_batch_indices": np.arange(batch.n_spikes, dtype=np.int64),
        "source_sample_indices": np.array(batch.sample_indices, copy=True),
        "source_peak_channel_indices": np.array(batch.peak_channel_indices, copy=True),
        "source_electrode_group_ids": np.array(batch.electrode_group_ids, copy=True),
        "source_segment_id": batch.segment_id,
        "source_clock": batch.clock,
        "source_channels": batch.channels,
        "source_waveform_axis_order": batch.axis_order,
        "source_pre_samples": batch.pre_samples,
        "source_post_samples": batch.post_samples,
        "source_polarity": batch.polarity,
        "source_spike_polarities": (
            None if batch.spike_polarities is None else np.array(batch.spike_polarities, copy=True)
        ),
        "projection_method": projector_kind,
        "projection_fit": fit if projection is not None else None,
        "projection_device": projection.device_ if projection is not None else None,
        "measurements": selected,
    }
    return FeatureMatrix(
        data=data,
        fs=None,
        time=np.array(batch.times, copy=True),
        feature_names=names,
        source_signal=batch.source_stream,
        window_size=batch.n_samples / batch.fs,
        shift=None,
        unit=units,
        attrs=attrs,
    )


def _waveform_schema(batch: SpikeWaveformBatch) -> _WaveformSchema:
    return _WaveformSchema(
        n_samples=batch.n_samples,
        pre_samples=batch.pre_samples,
        post_samples=batch.post_samples,
        fs=batch.fs,
        channels=tuple(_channel_fingerprint(channel) for channel in batch.channels),
    )


def _channel_fingerprint(channel: ChannelInfo) -> tuple[object, ...]:
    return (
        channel.name,
        channel.index,
        channel.type,
        channel.unit,
        channel.valid,
        channel.bad,
        channel.impedance,
        channel.electrode,
        channel.contact,
        channel.pos,
        _metadata_fingerprint(channel.attrs),
    )


def _metadata_fingerprint(value: object) -> object:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return ("array", contiguous.dtype.str, contiguous.shape, contiguous.tobytes())
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (
                    str(key),
                    _metadata_fingerprint(item),
                )
                for key, item in value.items()
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(_metadata_fingerprint(item) for item in value)
    return value


def _validate_measurements(
    measurements: Sequence[WaveformMeasurement],
) -> tuple[WaveformMeasurement, ...]:
    if isinstance(measurements, (str, bytes)):
        raise ValidationError("measurements must be a sequence of measurement names.")
    try:
        selected = tuple(measurements)
    except TypeError as exc:
        raise ValidationError("measurements must be a sequence of measurement names.") from exc
    for measurement in selected:
        if measurement not in _MEASUREMENTS:
            raise ValidationError("measurements entries must be 'amplitude', 'width', or 'energy'.")
    if len(selected) != len(set(selected)):
        raise ValidationError("measurements must not contain duplicates.")
    return selected


def _projector_kind(projection: WaveformProjector | None) -> str | None:
    if projection is None:
        return None
    if not isinstance(projection, WaveformProjector):
        raise ValidationError("projection must be a neurale.sorting.WaveformProjector or None.")
    estimator = projection.estimator
    if isinstance(estimator, PCA):
        return "pca"
    assert isinstance(estimator, LPP)
    return estimator.proj_method.lower()


def _shared_amp_unit(batch: SpikeWaveformBatch) -> str:
    units = tuple(batch.channels.units)
    if not units or any(unit != units[0] for unit in units[1:]):
        raise ValidationError(
            "waveform projection, amplitude, and energy require one shared channel unit."
        )
    return units[0]


def _validated_projection(values: np.ndarray, n_spikes: int) -> np.ndarray:
    projected = np.asarray(values)
    if projected.ndim != 2 or projected.shape[0] != n_spikes or projected.shape[1] == 0:
        raise ValidationError("projection must return shape (n_spikes, n_components).")
    if np.issubdtype(projected.dtype, np.complexfloating) or not np.all(np.isfinite(projected)):
        raise ValidationError("projection must return finite real values.")
    return projected


def _measurement(
    batch: SpikeWaveformBatch,
    measurement: WaveformMeasurement,
    amp_unit: str | None,
) -> tuple[np.ndarray, str]:
    if measurement == "amplitude":
        assert amp_unit is not None
        return np.asarray(batch.amps), amp_unit
    if measurement == "width":
        return _waveform_widths(batch), "s"
    assert measurement == "energy"
    assert amp_unit is not None
    with np.errstate(over="ignore", invalid="ignore"):
        energy = (
            np.einsum(
                "ijk,ijk->i",
                batch.waveforms,
                batch.waveforms,
                dtype=np.float64,
                optimize=False,
            )
            / batch.fs
        )
    if not np.all(np.isfinite(energy)):
        raise ValidationError("waveform energy is not finite in float64.")
    return energy, f"{amp_unit}^2*s"


def _waveform_widths(batch: SpikeWaveformBatch) -> np.ndarray:
    widths = np.empty(batch.n_spikes, dtype=np.float64)
    if batch.polarity == "both":
        assert batch.spike_polarities is not None
        polarities = batch.spike_polarities
    else:
        value = -1 if batch.polarity == "negative" else 1
        polarities = np.full(batch.n_spikes, value, dtype=np.int8)

    for spike in range(batch.n_spikes):
        channel = int(batch.peak_channel_indices[spike])
        tail = batch.waveforms[spike, batch.pre_samples :, channel]
        opposite = int(np.argmax(tail) if polarities[spike] < 0 else np.argmin(tail))
        widths[spike] = opposite / batch.fs
    return widths


__all__ = ["WaveformProjector", "project_waveform_features"]
