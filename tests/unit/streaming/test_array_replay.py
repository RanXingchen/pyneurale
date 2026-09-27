"""Public native replay contracts through the installed extension."""

import numpy as np
import pytest

from neurale import pipeline as p
from neurale import streaming as s
from neurale.features.online import BandpowerProcessor


def schema(channels=2):
    return s.StreamSchema(
        1,
        [
            s.SignalSchema(
                1,
                s.SignalDType.FLOAT64,
                channels,
                10,
                10,
                s.RationalRate(1000, 1),
                1,
                device_tick_tracking=s.DeviceTickTracking.SAMPLE_COUNTER,
                physical_unit=s.PhysicalUnit.DIMENSIONLESS,
            )
        ],
    )


def test_owned_payload_reset_and_timing():
    layout = schema()
    data = np.arange(200, dtype=np.float64).reshape(100, 2)
    source = s.ArrayReplaySource(layout, data)
    expected = data.copy()
    data[:] = -1
    sink = s.NativeResultSink(layout, 100)
    runner = s.StreamRunner(
        layout,
        s.RealtimeConfig(),
        source,
        s.IdentityNativeProcessor(),
        sink,
        safety_controller=s.RecordingNativeSafetyController(),
    )
    for attempt in range(2):
        if attempt:
            assert runner.reset() == s.StreamStatus.OK
            assert len(source.timings) == 0
            assert sink.snapshot()[0].shape == (0, 2)
        assert runner.prepare() == s.StreamStatus.OK
        assert runner.arm() == s.StreamStatus.OK
        assert runner.run() == s.StreamStatus.OK
        values, stamps = sink.snapshot()
        np.testing.assert_array_equal(values, expected)
        np.testing.assert_array_equal(stamps[:, 0], np.arange(100))
        timing = source.timings
        np.testing.assert_array_equal(np.diff(timing[:, 1]), 10_000_000)
        assert np.all(timing[:, 2] >= timing[:, 1])
        assert np.all(stamps[:, 3] >= timing[stamps[:, 1] // 10, 2])
        assert runner.outstanding_frames == 0


@pytest.mark.parametrize("data", [np.ones((10, 2), dtype=np.float32), np.ones((10, 4))[:, ::2]])
def test_no_implicit_conversion(data):
    with pytest.raises(TypeError):
        s.ArrayReplaySource(schema(), data)


def test_partial_block_rejected():
    with pytest.raises(ValueError):
        s.ArrayReplaySource(schema(), np.ones((11, 2)))


def test_capacity_fault_preserves_prefix():
    layout = schema()
    source = s.ArrayReplaySource(layout, np.ones((30, 2)))
    sink = s.NativeResultSink(layout, 10)
    runner = s.StreamRunner(
        layout,
        s.RealtimeConfig(),
        source,
        s.IdentityNativeProcessor(),
        sink,
        safety_controller=s.RecordingNativeSafetyController(),
    )
    assert runner.prepare() == s.StreamStatus.OK
    assert runner.arm() == s.StreamStatus.OK
    assert runner.run() != s.StreamStatus.OK
    assert sink.snapshot()[0].shape == (10, 2)
    assert runner.outstanding_frames == 0


def test_windowed_null_capture_preserves_cadence_and_reset():
    layout = schema()
    source = s.ArrayReplaySource(layout, np.ones((650, 2)))
    sink = s.NativeResultSink(layout, 10, window_samples=256, hop_samples=40)
    runner = s.StreamRunner(
        layout,
        s.RealtimeConfig(),
        source,
        s.IdentityNativeProcessor(),
        sink,
        safety_controller=s.RecordingNativeSafetyController(),
    )
    for attempt in range(2):
        if attempt:
            assert runner.reset() == s.StreamStatus.OK
        assert runner.prepare() == s.StreamStatus.OK
        assert runner.arm() == s.StreamStatus.OK
        assert runner.run() == s.StreamStatus.OK
        values, stamps = sink.snapshot()
        np.testing.assert_array_equal(values, np.zeros((10, 5)))
        np.testing.assert_array_equal(stamps[:, 0], np.arange(10))
        np.testing.assert_array_equal(stamps[:, 1], np.arange(250, 650, 40))
        assert runner.outstanding_frames == 0


@pytest.mark.parametrize("window,hop", [(256, 0), (0, 40), (256, 1)])
def test_null_capture_rejects_ambiguous_cadence(window, hop):
    with pytest.raises(ValueError):
        s.NativeResultSink(schema(), 10, window_samples=window, hop_samples=hop)


@pytest.mark.parametrize("detrend", ["none", "mean", "linear"])
def test_native_pmtm_detrend_matches_public_spectral_processor(detrend):
    layout = schema()
    data = np.random.default_rng(73).normal(size=(500, 2)) + np.arange(500)[:, None] / 3
    feature = p.MultitaperBandpowerFeature(
        bands=(p.Band("alpha", 8, 32),),
        weighting="unity",
        n_tapers=4,
        detrend=detrend,
    )
    plan = p.PipelinePlan(
        (p.FeatureStage(window_seconds=0.256, update_interval_seconds=0.04, features=(feature,)),)
    )
    assert p.PipelinePlan.from_document(plan.to_document()) == plan
    if detrend == "none":
        assert "detrend" not in feature.to_document()
    compiled = p.compile_pipeline(plan)
    sink = s.NativeResultSink(compiled.output_schema(layout), 7)
    runner = compiled.create_runner(
        layout,
        s.RealtimeConfig(),
        s.ArrayReplaySource(layout, data),
        sink,
        profile="realtime",
        safety_controller=s.RecordingNativeSafetyController(),
    )
    assert runner.prepare() == s.StreamStatus.OK
    assert runner.arm() == s.StreamStatus.OK
    assert runner.run() == s.StreamStatus.OK
    reference = BandpowerProcessor(
        fs=1000,
        channels=2,
        bands={"alpha": (8, 32)},
        window_size=0.256,
        shift=0.04,
        nw=2.5,
        n_tapers=4,
        weighting="unity",
        detrend=detrend,
        backend="builtin",
    )
    expected = np.asarray(reference.process(data).data)
    values, stamps = sink.snapshot()
    np.testing.assert_allclose(values, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(stamps[:, 0], np.arange(7))
    np.testing.assert_array_equal(stamps[:, 1], np.arange(250, 500, 40))
