#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import asdict

import pytest

import neurale.pipeline as realtime
from neurale.exceptions import ValidationError


def _contract() -> realtime.FittedFeatureContract:
    return realtime.FittedFeatureContract(
        feature_names=("x",),
        feature_unit_symbols=("1",),
        observation_rate_numerator=100,
        observation_rate_denominator=1,
        window_length_ns=10_000_000,
        shift_ns=10_000_000,
        algorithm_name="test",
        algorithm_version="1",
        source_stream="features",
    )


def _linear() -> realtime.LinearDecoderStage:
    return realtime.LinearDecoderStage(
        output_schema_id=2,
        output_signal_id=2,
        output_channel_set_id=2,
        output_physical_unit="dimensionless",
        feature_set_id=2,
        selection=(0,),
        selected_feature_names=("x",),
        fitted_feature_contract=_contract(),
        n_features=1,
        n_outputs=1,
        coefficients=(1.0,),
        intercept=(0.0,),
    )


def _kalman() -> realtime.KalmanDecoderStage:
    return realtime.KalmanDecoderStage(
        output_schema_id=2,
        output_signal_id=2,
        output_channel_set_id=2,
        output_physical_unit="dimensionless",
        feature_set_id=2,
        selection=(0,),
        selected_feature_names=("x",),
        fitted_feature_contract=_contract(),
        state_dim=1,
        observation_dim=1,
        transition=(1.0,),
        transition_offset=(0.0,),
        observation=(1.0,),
        observation_offset=(0.0,),
        process_covariance=(1.0,),
        observation_covariance=(1.0,),
        initial_state=(0.0,),
        initial_covariance=(1.0,),
    )


def _lda():
    values = asdict(_linear())
    values["fitted_feature_contract"] = _contract()
    return realtime.LdaDecoderStage(**values, classes=(17, 42))


@pytest.mark.parametrize("stage", [_linear(), _kalman(), _lda()])
def test_decoder_stage_round_trips_and_compiles(stage) -> None:
    plan = realtime.PipelinePlan((stage,))

    restored = realtime.PipelinePlan.from_document(plan.to_document())

    assert restored == plan
    assert realtime.compile_pipeline(restored).stage_count == 1


def test_decoder_rejects_unknown_physical_unit() -> None:
    document = _linear().to_document()
    document["output_physical_unit"] = "pixels"

    with pytest.raises(ValidationError, match="output_physical_unit"):
        realtime.PipelinePlan.from_document(
            {"kind": "pipeline", "version": 1, "stages": [document]}
        )


@pytest.mark.parametrize("classes", [(1,), (1, 1), ("a", "b"), (True, 2), (1, 2**53)])
def test_native_class_labels_are_explicit(classes):
    from dataclasses import replace

    with pytest.raises(ValidationError):
        replace(_lda(), classes=classes)


def test_lda_pipeline_runs_without_an_experiment():
    import neurale.streaming as streaming

    signal = streaming.SignalSchema(
        1,
        streaming.SignalDType.FLOAT64,
        1,
        4,
        4,
        streaming.RationalRate(100, 1),
        1,
        kind=streaming.SignalKind.FEATURE,
        feature_set_id=2,
        observation_timing=streaming.ObservationTiming.REGULAR,
    )
    descriptor = streaming.FeatureSetDescriptor(
        2,
        ["x"],
        [1],
        1,
        "features",
        10_000_000,
        10_000_000,
        algorithm_name="test",
        algorithm_version="1",
    )
    schema = streaming.StreamSchema(
        1, [signal], [descriptor], [streaming.UnitDescriptor(1, "1", "dimensionless")]
    )
    compiled = realtime.compile_pipeline(realtime.PipelinePlan((_lda(),)))
    consumer = streaming.CountingNativeConsumer()
    runner = compiled.create_runner(
        schema,
        streaming.RealtimeConfig(),
        streaming.SyntheticNativeSource(schema, 3),
        consumer,
        profile=streaming.ExecutionProfile.REALTIME,
        safety_controller=streaming.RecordingNativeSafetyController(),
    )
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == streaming.StreamStatus.OK
    assert consumer.frame_count == 3
    assert runner.outstanding_frames == 0
