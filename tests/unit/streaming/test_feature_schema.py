#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

import neurale.streaming as streaming
from neurale.data import FeatureMatrix
from neurale.exceptions import ValidationError


def _descriptor() -> streaming.FeatureSetDescriptor:
    return streaming.FeatureSetDescriptor(
        71,
        ["lmp", "beta_power"],
        [1, 2],
        9,
        "motor-cortex",
        100_000_000,
        50_000_000,
        algorithm_name="test-features",
        algorithm_version="1",
    )


def _schema() -> streaming.StreamSchema:
    feature = streaming.SignalSchema(
        4,
        streaming.SignalDType.FLOAT32,
        2,
        2,
        4,
        streaming.RationalRate(20, 1),
        3,
        kind=streaming.SignalKind.FEATURE,
        feature_set_id=71,
        observation_timing=streaming.ObservationTiming.REGULAR,
    )
    return streaming.StreamSchema(
        31,
        [feature],
        [_descriptor()],
        [
            streaming.UnitDescriptor(1, "V^2", "squared volts"),
            streaming.UnitDescriptor(2, "V^2/Hz", "power density"),
        ],
    )


def test_irregular_feature_frame_preserves_explicit_time():
    signal = streaming.SignalSchema(
        4,
        streaming.SignalDType.FLOAT64,
        1,
        1,
        1,
        streaming.RationalRate(0, 1),
        3,
        kind=streaming.SignalKind.FEATURE,
        feature_set_id=71,
        observation_timing=streaming.ObservationTiming.IRREGULAR,
    )
    descriptor = streaming.FeatureSetDescriptor(
        71,
        ["mean"],
        [1],
        9,
        "test",
        100_000_000,
        0,
        algorithm_name="interval_mean",
        algorithm_version="1",
    )
    schema = streaming.StreamSchema(31, [signal], [descriptor], [streaming.UnitDescriptor(1, "1")])
    matrix = FeatureMatrix(
        np.array([[3.0]]),
        fs=None,
        time=np.array([1.25]),
        feature_names=["mean"],
        source_signal="test",
        window_size=0.1,
        unit="1",
    )
    frame = streaming.feature_matrix_to_frame(matrix, schema, 4, sequence=17)
    restored = streaming.feature_matrix_from_frame(frame, schema, 4)
    assert restored.fs is None and restored.shift is None
    np.testing.assert_array_equal(restored.time, [1.25])
    np.testing.assert_array_equal(restored.data, matrix.data)
    assert frame.sequence == 17


def test_feature_descriptor_round_trips_as_immutable_schema() -> None:
    schema = _schema()
    signal = schema.signals[0]
    descriptor = schema.feature_sets[0]

    assert signal.kind == streaming.SignalKind.FEATURE
    assert signal.feature_set_id == 71
    assert signal.observation_timing == streaming.ObservationTiming.REGULAR
    assert descriptor.feature_names == ["lmp", "beta_power"]
    assert descriptor.unit_ids == [1, 2]
    assert descriptor.source_stream_id == 9
    assert descriptor.source_stream == "motor-cortex"
    assert descriptor.window_length_ns == 100_000_000
    assert descriptor.shift_ns == 50_000_000
    assert descriptor.timestamp_reference == streaming.FeatureTimestampReference.WINDOW_CENTER
    assert [unit.symbol for unit in schema.units] == ["V^2", "V^2/Hz"]
    with pytest.raises(AttributeError):
        descriptor.id = 99


def test_missing_and_duplicate_descriptor_ids_are_rejected() -> None:
    feature = streaming.SignalSchema(
        4,
        streaming.SignalDType.FLOAT32,
        2,
        1,
        4,
        streaming.RationalRate(20, 1),
        3,
        kind=streaming.SignalKind.FEATURE,
        feature_set_id=71,
        observation_timing=streaming.ObservationTiming.REGULAR,
    )
    with pytest.raises(ValueError, match="missing descriptor"):
        streaming.StreamSchema(31, [feature])
    descriptor = _descriptor()
    with pytest.raises(ValueError, match="duplicate feature-set descriptor"):
        streaming.FeatureSetDescriptorRegistry([descriptor, descriptor])


def test_missing_and_duplicate_unit_ids_are_rejected() -> None:
    descriptor = _descriptor()
    with pytest.raises(ValueError, match="duplicate unit id"):
        streaming.UnitRegistry(
            [
                streaming.UnitDescriptor(1, "V", "volts"),
                streaming.UnitDescriptor(1, "mV", "millivolts"),
            ]
        )
    feature = streaming.SignalSchema(
        4,
        streaming.SignalDType.FLOAT32,
        2,
        1,
        4,
        streaming.RationalRate(20, 1),
        3,
        kind=streaming.SignalKind.FEATURE,
        feature_set_id=71,
        observation_timing=streaming.ObservationTiming.REGULAR,
    )
    with pytest.raises(ValueError, match="missing unit"):
        streaming.StreamSchema(
            31,
            [feature],
            [descriptor],
            [streaming.UnitDescriptor(1, "V^2")],
        )


def test_feature_matrix_frame_round_trip_preserves_metadata() -> None:
    schema = _schema()
    matrix = FeatureMatrix(
        data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        fs=20.0,
        t0=1.25,
        feature_names=["lmp", "beta_power"],
        source_signal="motor-cortex",
        window_size=0.1,
        shift=0.05,
        unit=["V^2", "V^2/Hz"],
    )

    frame = streaming.feature_matrix_to_frame(matrix, schema, 4, session_id=5, sequence=6)
    assert frame.blocks[0].observation_time_start_ns == 1_250_000_000
    restored = streaming.feature_matrix_from_frame(frame, schema, 4)

    np.testing.assert_array_equal(restored.data, matrix.data)
    np.testing.assert_allclose(restored.time, [1.25, 1.30])
    assert restored.feature_names == matrix.feature_names
    assert restored.unit == matrix.unit
    assert restored.source_signal == "motor-cortex"
    assert restored.window_size == 0.1
    assert restored.shift == 0.05
    assert restored.attrs["feature_set_id"] == 71
    assert restored.attrs["source_stream_id"] == 9
    assert restored.attrs["timestamp_reference"] == "window_center"


def test_feature_matrix_rejects_irregular_timing_and_shape() -> None:
    schema = _schema()
    unspecified_rate = FeatureMatrix(
        data=np.ones((2, 2), dtype=np.float32),
        fs=None,
        time=np.array([1.0, 1.06]),
        feature_names=["lmp", "beta_power"],
        source_signal="motor-cortex",
        window_size=0.1,
        shift=0.05,
        unit=["V^2", "V^2/Hz"],
    )
    with pytest.raises(ValidationError, match="regular FeatureMatrix sampling rate"):
        streaming.feature_matrix_to_frame(unspecified_rate, schema, 4)

    with pytest.raises(ValidationError, match="use fs=None"):
        FeatureMatrix(
            data=np.ones((2, 2), dtype=np.float32),
            fs=20.0,
            time=np.array([1.0, 1.06]),
            feature_names=["lmp", "beta_power"],
            source_signal="motor-cortex",
            window_size=0.1,
            shift=0.05,
            unit=["V^2", "V^2/Hz"],
        )

    frame = streaming.feature_matrix_to_frame(
        FeatureMatrix(
            data=np.ones((1, 2), dtype=np.float32),
            fs=20.0,
            feature_names=["lmp", "beta_power"],
            source_signal="motor-cortex",
            window_size=0.1,
            shift=0.05,
            unit=["V^2", "V^2/Hz"],
        ),
        schema,
        4,
    )
    malformed = streaming.PythonObserverSignalBlock(0, 0, 0, 4, 4, 1, observation_time_start_ns=0)
    bad_frame = streaming.PythonObserverFrame(
        0,
        0,
        0,
        schema_id=schema.id,
        blocks=[malformed],
        payload=frame.payload,
    )
    with pytest.raises(ValidationError, match="does not match"):
        streaming.feature_matrix_from_frame(bad_frame, schema, 4)
