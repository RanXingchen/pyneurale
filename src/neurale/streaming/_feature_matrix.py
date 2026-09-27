#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Explicit control-plane conversion for native feature frames."""

from __future__ import annotations

import math

import numpy as np

from neurale.data import FeatureMatrix
from neurale.exceptions import ValidationError

from . import _native

_DTYPES = {
    _native.SignalDType.INT16: np.dtype(np.int16),
    _native.SignalDType.INT32: np.dtype(np.int32),
    _native.SignalDType.FLOAT32: np.dtype(np.float32),
    _native.SignalDType.FLOAT64: np.dtype(np.float64),
}


def _find_by_id(items, item_id, message: str):
    try:
        return next(item for item in items if item.id == item_id)
    except StopIteration as exc:
        raise ValidationError(message) from exc


def _feature_contract(schema, signal_id: int):
    signal = _find_by_id(schema.signals, signal_id, f"unknown feature signal id: {signal_id}.")
    if signal.kind != _native.SignalKind.FEATURE:
        raise ValidationError(f"signal {signal_id} is not a feature stream.")
    descriptor = _find_by_id(
        schema.feature_sets,
        signal.feature_set_id,
        f"missing feature-set descriptor id: {signal.feature_set_id}.",
    )
    unit_symbols = {unit.id: unit.symbol for unit in schema.units}
    try:
        units = [unit_symbols[unit_id] for unit_id in descriptor.unit_ids]
    except KeyError as exc:
        raise ValidationError(f"missing unit descriptor id: {exc.args[0]}.") from exc
    return signal, descriptor, units


def _rate_hz(signal) -> float:
    return signal.fs.numerator / signal.fs.denominator


def feature_matrix_from_frame(frame, schema, signal_id: int) -> FeatureMatrix:
    """Copy one native feature block into a metadata-complete FeatureMatrix."""
    if frame.schema_id != schema.id:
        raise ValidationError("frame schema_id does not match StreamSchema.id.")
    signal, descriptor, units = _feature_contract(schema, signal_id)
    blocks = [block for block in frame.blocks if block.signal_id == signal_id]
    if len(blocks) != 1:
        raise ValidationError("feature frame must contain exactly one requested block.")
    block = blocks[0]
    dtype = _DTYPES[signal.dtype]
    expected_bytes = block.n_samples * signal.n_channels * dtype.itemsize
    if block.payload_byte_count != expected_bytes:
        raise ValidationError("feature payload does not match (observations, features).")
    start = block.payload_offset
    stop = start + block.payload_byte_count
    payload = np.asarray(frame.payload, dtype=np.uint8)
    if stop > payload.size:
        raise ValidationError("feature payload range exceeds the frame payload.")
    data = np.frombuffer(payload[start:stop], dtype=dtype).copy()
    data = data.reshape(block.n_samples, signal.n_channels)
    rate = _rate_hz(signal)
    irregular = signal.observation_timing == _native.ObservationTiming.IRREGULAR
    if irregular and block.n_samples != 1:
        raise ValidationError("irregular feature frames contain exactly one observation.")
    return FeatureMatrix(
        data=data,
        fs=None if irregular else rate,
        time=np.array([block.observation_time_start_ns / 1_000_000_000]) if irregular else None,
        t0=block.observation_time_start_ns / 1_000_000_000,
        feature_names=list(descriptor.feature_names),
        source_signal=descriptor.source_stream,
        window_size=descriptor.window_length_ns / 1_000_000_000,
        shift=None if irregular else descriptor.shift_ns / 1_000_000_000,
        unit=units,
        attrs={
            "feature_set_id": descriptor.id,
            "source_stream_id": descriptor.source_stream_id,
            "timestamp_reference": "window_center",
            "algorithm_name": descriptor.algorithm_name,
            "algorithm_version": descriptor.algorithm_version,
        },
    )


def feature_matrix_to_frame(
    matrix: FeatureMatrix,
    schema,
    signal_id: int,
    *,
    session_id: int = 0,
    sequence: int = 0,
    sample_idx_start: int = 0,
    host_received_ns: int = 0,
):
    """Copy a compatible FeatureMatrix into a native observer-frame value."""
    if not isinstance(matrix, FeatureMatrix):
        raise TypeError("matrix must be a FeatureMatrix.")
    signal, descriptor, units = _feature_contract(schema, signal_id)
    expected_dtype = _DTYPES[signal.dtype]
    if matrix.data.dtype != expected_dtype:
        raise ValidationError(
            f"FeatureMatrix dtype must be {expected_dtype}; got {matrix.data.dtype}."
        )
    if matrix.data.shape[1] != signal.n_channels:
        raise ValidationError("FeatureMatrix feature count does not match the schema.")
    if matrix.n_frames == 0:
        raise ValidationError("FeatureMatrix must contain at least one observation.")
    if matrix.n_frames > signal.max_block_samples:
        raise ValidationError("FeatureMatrix observation count exceeds the schema bound.")
    if list(matrix.feature_names) != list(descriptor.feature_names):
        raise ValidationError("FeatureMatrix feature names do not match the descriptor.")
    matrix_units = (
        [matrix.unit] * matrix.n_features if isinstance(matrix.unit, str) else list(matrix.unit)
    )
    if matrix_units != units:
        raise ValidationError("FeatureMatrix units do not match the unit registry.")
    if matrix.source_signal != descriptor.source_stream:
        raise ValidationError("FeatureMatrix source signal does not match the descriptor.")
    window_seconds = descriptor.window_length_ns / 1_000_000_000
    shift_seconds = descriptor.shift_ns / 1_000_000_000
    if matrix.window_size is None or not math.isclose(
        matrix.window_size, window_seconds, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValidationError("FeatureMatrix window size does not match the descriptor.")
    if signal.observation_timing == _native.ObservationTiming.IRREGULAR:
        if matrix.fs is not None or matrix.shift is not None or matrix.n_frames != 1:
            raise ValidationError(
                "irregular features require one observation with no fixed rate or shift."
            )
    else:
        if matrix.shift is None or not math.isclose(
            matrix.shift, shift_seconds, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValidationError("FeatureMatrix shift does not match the descriptor.")
        rate = _rate_hz(signal)
        if matrix.fs is None:
            raise ValidationError(
                "native feature frames require a regular FeatureMatrix sampling rate."
            )
        if not math.isclose(matrix.fs, rate, rel_tol=0.0, abs_tol=1e-12):
            raise ValidationError("FeatureMatrix sampling rate does not match the schema.")
        expected_time = matrix.time[0] + np.arange(matrix.n_frames) / rate
        if not np.allclose(matrix.time, expected_time, rtol=0.0, atol=1e-12):
            raise ValidationError("FeatureMatrix timestamps are not regular.")
    observation_time_start_ns = round(float(matrix.time[0]) * 1_000_000_000)
    if observation_time_start_ns < 0:
        raise ValidationError("feature observation timestamps must be non-negative.")
    values = np.ascontiguousarray(matrix.data)
    payload = values.view(np.uint8).reshape(-1).copy()
    block = _native.PythonObserverSignalBlock(
        sample_idx_start,
        0,
        0,
        payload.size,
        signal_id,
        matrix.n_frames,
        observation_time_start_ns=observation_time_start_ns,
    )
    return _native.PythonObserverFrame(
        session_id,
        sequence,
        host_received_ns,
        schema_id=schema.id,
        blocks=[block],
        payload=payload,
    )


__all__ = ["feature_matrix_from_frame", "feature_matrix_to_frame"]
