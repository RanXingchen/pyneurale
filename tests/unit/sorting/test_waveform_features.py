#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from neurale.data import ChannelTable, Clock, FeatureMatrix, SpikeWaveformBatch
from neurale.exceptions import DeviceUnavailableError, ValidationError
from neurale.models import LPP, PCA
from neurale.runtime import runtime_context
from neurale.sorting import WaveformProjector, project_waveform_features

pytestmark = pytest.mark.native


def _batch(
    waveforms: np.ndarray,
    *,
    polarities: np.ndarray | None = None,
    units: str | list[str] = "uV",
    fs: float = 1_000.0,
    channel_names: list[str] | None = None,
) -> SpikeWaveformBatch:
    values = np.asarray(waveforms)
    n_spikes, n_samples, n_channels = values.shape
    sample_indices = 100 + 10 * np.arange(n_spikes, dtype=np.int64)
    if polarities is None:
        polarity = "negative"
        spike_polarities = None
        amps = -np.arange(1, n_spikes + 1, dtype=np.float64)
    else:
        polarity = "both"
        spike_polarities = np.asarray(polarities, dtype=np.int8)
        amps = spike_polarities * np.arange(1, n_spikes + 1, dtype=np.float64)
    return SpikeWaveformBatch(
        waveforms=values,
        sample_indices=sample_indices,
        times=2.0 + (sample_indices - 100) / fs,
        peak_channel_indices=np.arange(n_spikes, dtype=np.int64) % n_channels,
        electrode_group_ids=np.arange(n_spikes, dtype=np.int64) % 2,
        amps=amps,
        channels=ChannelTable.from_names(
            channel_names
            if channel_names is not None
            else [f"contact-{idx}" for idx in range(n_channels)],
            type="spike",
            unit=units,
        ),
        fs=fs,
        clock=Clock("acquisition", "device", synchronization_domain="session"),
        source_stream="wideband",
        segment_id="segment-a",
        pre_samples=n_samples // 2,
        post_samples=n_samples - n_samples // 2 - 1,
        polarity=polarity,
        spike_polarities=spike_polarities,
        attrs={"detector": "threshold"},
    )


def _projection_batch() -> SpikeWaveformBatch:
    rng = np.random.default_rng(941)
    waveforms = rng.normal(size=(8, 5, 2))
    waveforms[:, 2, 0] -= np.linspace(2.0, 5.0, 8)
    waveforms[:, 2, 1] -= np.linspace(1.0, 3.0, 8) ** 2
    return _batch(waveforms)


def test_measurements_match_direct_calculations() -> None:
    waveforms = np.array(
        [
            [[0.0, 1.0], [-1.0, 0.0], [-5.0, 2.0], [-2.0, 0.0], [3.0, -1.0]],
            [[0.0, -1.0], [1.0, 0.0], [2.0, 4.0], [0.0, 1.0], [-1.0, -3.0]],
        ],
        dtype=np.float64,
    )
    batch = _batch(waveforms, polarities=np.array([-1, 1], dtype=np.int8))

    result = project_waveform_features(
        batch,
        measurements=("amplitude", "width", "energy"),
    )

    assert isinstance(result, FeatureMatrix)
    assert result.data.shape == (2, 3)
    np.testing.assert_array_equal(result.data[:, 0], batch.amps)
    np.testing.assert_array_equal(result.data[:, 1], [0.002, 0.002])
    np.testing.assert_allclose(result.data[:, 2], np.sum(waveforms**2, axis=(1, 2)) / 1_000.0)
    assert result.feature_names == ["amplitude", "width", "energy"]
    assert result.unit == ["uV", "s", "uV^2*s"]
    np.testing.assert_array_equal(result.time, batch.times)
    assert result.fs is None
    assert result.source_signal == batch.source_stream
    assert result.window_size == batch.n_samples / batch.fs
    assert result.shift is None
    assert result.attrs["observation_axis"] == "spike"
    assert result.attrs["time_reference"] == "aligned_spike"
    assert result.attrs["timing"] == "irregular"
    assert result.attrs["source_fs"] == batch.fs
    np.testing.assert_array_equal(result.attrs["source_batch_indices"], [0, 1])
    np.testing.assert_array_equal(result.attrs["source_sample_indices"], batch.sample_indices)
    np.testing.assert_array_equal(
        result.attrs["source_peak_channel_indices"], batch.peak_channel_indices
    )
    np.testing.assert_array_equal(
        result.attrs["source_electrode_group_ids"], batch.electrode_group_ids
    )
    assert result.attrs["source_segment_id"] == batch.segment_id
    assert result.attrs["source_clock"] is batch.clock
    assert result.attrs["source_channels"] is batch.channels
    assert result.attrs["source_waveform_axis_order"] == ("spike", "sample", "channel")
    assert result.attrs["source_polarity"] == "both"
    np.testing.assert_array_equal(result.attrs["source_spike_polarities"], [-1, 1])
    assert result.attrs["projection_method"] is None


def test_pca_projection_delegates_with_stable_columns() -> None:
    batch = _projection_batch()
    flattened = batch.waveforms.reshape(batch.n_spikes, -1)
    expected_model = PCA(3)
    with runtime_context(device="cpu"):
        expected = expected_model.fit_transform(flattened)
        model = WaveformProjector(PCA(3))
        result = project_waveform_features(
            batch,
            model,
            measurements=("width", "amplitude"),
        )

    np.testing.assert_allclose(result.data[:, :3], expected)
    np.testing.assert_array_equal(result.data[:, 4], batch.amps)
    assert result.feature_names == ["pca_000", "pca_001", "pca_002", "width", "amplitude"]
    assert result.unit == ["uV", "uV", "uV", "s", "uV"]
    assert model.device_ == "cpu"
    assert result.attrs["projection_method"] == "pca"
    assert result.attrs["projection_fit"] is True
    assert result.attrs["projection_device"] == "cpu"


@pytest.mark.parametrize("proj_method", ["OPP", "LPP"])
def test_lpp_projection_matches_existing_model(proj_method: str) -> None:
    batch = _projection_batch()
    flattened = batch.waveforms.reshape(batch.n_spikes, -1)
    with runtime_context(device="cpu"):
        expected = LPP(2, n_neighbors=3, proj_method=proj_method).fit_transform(flattened)
        model = WaveformProjector(LPP(2, n_neighbors=3, proj_method=proj_method))
        result = project_waveform_features(batch, model, measurements=())

    np.testing.assert_allclose(result.data, expected)
    prefix = proj_method.lower()
    assert result.feature_names == [f"{prefix}_000", f"{prefix}_001"]
    assert result.unit == ["a.u.", "a.u."]
    assert result.attrs["projection_method"] == prefix
    assert model.device_ == "cpu"


def test_fitted_projection_ignores_ambient_context() -> None:
    batch = _projection_batch()
    with runtime_context(device="cpu"):
        model = WaveformProjector(PCA(2))
        fitted = project_waveform_features(batch, model, measurements=())

    with runtime_context(device="cuda"):
        result = project_waveform_features(batch, model, fit=False, measurements=())

    np.testing.assert_allclose(result.data, fitted.data)
    assert model.device_ == "cpu"
    assert result.attrs["projection_device"] == "cpu"
    assert result.attrs["projection_fit"] is False


def test_pca_fit_preserves_cpu_only_device_rejection() -> None:
    model = WaveformProjector(PCA(2))
    with runtime_context(device="cuda"):
        with pytest.raises(DeviceUnavailableError, match=r"models\.decomposition"):
            project_waveform_features(_projection_batch(), model, measurements=())
    with pytest.raises(ValidationError, match="not fitted"):
        _ = model.device_


def test_empty_batch_and_mixed_unit_rules_are_explicit() -> None:
    empty = _batch(np.empty((0, 5, 2), dtype=np.float64))
    result = project_waveform_features(empty, measurements=("width",))
    assert result.data.shape == (0, 1)
    assert result.feature_names == ["width"]

    mixed = _batch(np.ones((2, 5, 2)), units=["uV", "mV"])
    width = project_waveform_features(mixed, measurements=("width",))
    assert width.unit == ["s"]
    with pytest.raises(ValidationError, match="shared channel unit"):
        project_waveform_features(mixed, measurements=("amplitude",))
    with pytest.raises(ValidationError, match="shared channel unit"):
        project_waveform_features(mixed, WaveformProjector(PCA(1)), measurements=())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"projection": object()}, "projection"),
        ({"measurements": ()}, "at least one"),
        ({"measurements": "width"}, "sequence"),
        ({"measurements": ("width", "width")}, "duplicates"),
        ({"measurements": ("area",)}, "entries"),
        ({"fit": 1}, "fit"),
    ],
)
def test_invalid_projection_and_measurement_are_rejected(kwargs, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        project_waveform_features(_projection_batch(), **kwargs)


def test_projection_state_errors_stay_owned_by_models() -> None:
    batch = _projection_batch()
    with pytest.raises(ValidationError, match="not fitted"):
        project_waveform_features(
            batch,
            WaveformProjector(PCA(2)),
            fit=False,
            measurements=(),
        )

    empty = _batch(np.empty((0, 5, 2), dtype=np.float64))
    with pytest.raises(ValidationError, match="must not be empty"):
        project_waveform_features(empty, WaveformProjector(PCA(1)), measurements=())


def test_fitted_projection_rejects_different_waveform_geometry() -> None:
    fitted_batch = _projection_batch()
    incompatible = _batch(np.ones((8, 2, 5), dtype=np.float64))
    with runtime_context(device="cpu"):
        model = WaveformProjector(PCA(2))
        project_waveform_features(fitted_batch, model, measurements=())
        with pytest.raises(ValidationError, match="waveform schema"):
            project_waveform_features(incompatible, model, fit=False, measurements=())


def test_fitted_projection_rejects_reordered_channels() -> None:
    fitted_batch = _projection_batch()
    reordered = _batch(
        np.array(fitted_batch.waveforms, copy=True),
        channel_names=["contact-1", "contact-0"],
    )
    with runtime_context(device="cpu"):
        model = WaveformProjector(PCA(2))
        project_waveform_features(fitted_batch, model, measurements=())
        with pytest.raises(ValidationError, match="waveform schema"):
            project_waveform_features(reordered, model, fit=False, measurements=())


def test_fitted_projection_rejects_different_fs() -> None:
    fitted_batch = _projection_batch()
    different_rate = _batch(
        np.array(fitted_batch.waveforms, copy=True),
        fs=2_000.0,
    )
    with runtime_context(device="cpu"):
        model = WaveformProjector(PCA(2))
        project_waveform_features(fitted_batch, model, measurements=())
        with pytest.raises(ValidationError, match="waveform schema"):
            project_waveform_features(different_rate, model, fit=False, measurements=())


def test_fitted_projection_rejects_changed_alignment() -> None:
    fitted_batch = _projection_batch()
    changed_alignment = SpikeWaveformBatch(
        waveforms=fitted_batch.waveforms,
        sample_indices=fitted_batch.sample_indices,
        times=fitted_batch.times,
        peak_channel_indices=fitted_batch.peak_channel_indices,
        electrode_group_ids=fitted_batch.electrode_group_ids,
        amps=fitted_batch.amps,
        channels=ChannelTable.from_names(
            fitted_batch.channels.names,
            type="spike",
            unit="mV",
        ),
        fs=fitted_batch.fs,
        clock=fitted_batch.clock,
        source_stream=fitted_batch.source_stream,
        segment_id=fitted_batch.segment_id,
        pre_samples=1,
        post_samples=3,
        polarity=fitted_batch.polarity,
        attrs=dict(fitted_batch.attrs),
    )
    with runtime_context(device="cpu"):
        model = WaveformProjector(PCA(2))
        project_waveform_features(fitted_batch, model, measurements=())
        with pytest.raises(ValidationError, match="waveform schema"):
            project_waveform_features(changed_alignment, model, fit=False, measurements=())


def test_fitted_projection_accepts_new_batch_with_same_schema() -> None:
    fitted_batch = _projection_batch()
    compatible = _batch(np.array(fitted_batch.waveforms + 0.25, copy=True))
    with runtime_context(device="cpu"):
        model = WaveformProjector(PCA(2))
        project_waveform_features(fitted_batch, model, measurements=())
        expected = model.estimator.transform(compatible.waveforms.reshape(compatible.n_spikes, -1))
        result = project_waveform_features(compatible, model, fit=False, measurements=())

    np.testing.assert_allclose(result.data, expected)


def test_projection_without_waveform_schema_is_rejected() -> None:
    batch = _projection_batch()
    with runtime_context(device="cpu"):
        model = WaveformProjector(PCA(2).fit(batch.waveforms.reshape(batch.n_spikes, -1)))
        with pytest.raises(ValidationError, match="fit it first"):
            project_waveform_features(batch, model, fit=False, measurements=())


@pytest.mark.parametrize("estimator_kind", ["pca", "lpp"])
def test_external_estimator_refit_invalidates_waveform_schema(estimator_kind: str) -> None:
    batch = _projection_batch()
    estimator = PCA(2) if estimator_kind == "pca" else LPP(2, n_neighbors=3, proj_method="LPP")
    projector = WaveformProjector(estimator)
    with runtime_context(device="cpu"):
        project_waveform_features(batch, projector, measurements=())
        refit_values = np.random.default_rng(771).normal(size=(12, 10))
        projector.estimator.fit(refit_values)
        with pytest.raises(ValidationError, match="refitted outside"):
            project_waveform_features(batch, projector, fit=False, measurements=())
        with pytest.raises(ValidationError, match="no waveform schema"):
            project_waveform_features(batch, projector, fit=False, measurements=())
        restored = project_waveform_features(batch, projector, fit=True, measurements=())
        reused = project_waveform_features(batch, projector, fit=False, measurements=())

    np.testing.assert_allclose(reused.data, restored.data)


def test_bare_projection_estimator_is_rejected() -> None:
    with pytest.raises(ValidationError, match="estimator"):
        WaveformProjector(object())
    with pytest.raises(ValidationError, match="WaveformProjector"):
        project_waveform_features(_projection_batch(), PCA(2), measurements=())
