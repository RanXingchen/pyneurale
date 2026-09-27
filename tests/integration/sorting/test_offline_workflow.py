#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from neurale.data import Clock, FeatureMatrix, SignalArray
from neurale.exceptions import DeviceUnavailableError, ValidationError
from neurale.models import PCA
from neurale.runtime import runtime_context
from neurale.sorting import (
    DetectionConfig,
    OfflineSortingResult,
    ValleySeekingResult,
    WaveformProjector,
    run_offline_sorting,
    valley_seeking,
)

pytestmark = pytest.mark.native


def _feature_matrix(data: np.ndarray) -> FeatureMatrix:
    return FeatureMatrix(
        data=data,
        fs=None,
        time=np.arange(data.shape[0], dtype=np.float64),
        feature_names=[f"f{idx}" for idx in range(data.shape[1])],
        source_signal="waveforms",
        unit="a.u.",
    )


def _signal() -> SignalArray:
    base = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 28)
    data = base[:, np.newaxis].astype(np.float64)
    data[20] = -12.0
    data[55] = -10.0
    data[90] = -11.0
    return SignalArray.from_array(
        data,
        fs=1_000.0,
        t0=2.0,
        channel_names=["c0"],
        channel_types="spike",
        units="uV",
        name="wideband",
        clock=Clock("acquisition", "device", synchronization_domain="session"),
    )


def test_valley_seeking_returns_typed_immutable_diagnostics() -> None:
    features = _feature_matrix(np.array([[0.0], [0.1], [5.0]], dtype=np.float64))
    initial = np.array([9, 9, -4], dtype=np.int64)
    initial_before = initial.copy()

    result = valley_seeking(features, initial, radius=0.2, max_iterations=10)

    assert isinstance(result, ValleySeekingResult)
    np.testing.assert_array_equal(result.labels, [9, 9, -4])
    np.testing.assert_array_equal(result.neighbor_counts, [1, 1, 0])
    assert result.label_order == (-4, 9)
    assert result.converged
    assert result.termination == "converged"
    assert result.neighbor_pairs == 1
    assert result.workspace_bytes >= 0
    np.testing.assert_array_equal(initial, initial_before)
    assert not result.labels.flags.writeable
    with pytest.raises(ValueError):
        result.labels.flags.writeable = True


def test_public_valley_seeking_default_seed_is_observation_index() -> None:
    features = _feature_matrix(np.array([[0.0], [3.0]], dtype=np.float64))

    result = valley_seeking(features, radius=0.1)

    np.testing.assert_array_equal(result.labels, [0, 1])
    assert result.label_order == (0, 1)


def test_valley_seeking_preserves_initial_label_vocabulary() -> None:
    features = _feature_matrix(np.array([[0.0], [0.1], [0.2]], dtype=np.float64))

    result = valley_seeking(
        features,
        np.array([0, 1, 1], dtype=np.int64),
        radius=1.0,
    )

    np.testing.assert_array_equal(result.labels, [1, 1, 1])
    assert result.label_order == (0, 1)


def test_valley_seeking_rejects_cuda_without_fallback() -> None:
    features = _feature_matrix(np.array([[0.0], [1.0]], dtype=np.float64))

    with runtime_context(device="cuda"):
        with pytest.raises(DeviceUnavailableError, match=r"sorting\.valley_seeking"):
            valley_seeking(features, radius=1.0)


def test_minimal_public_workflow_returns_every_typed_intermediate() -> None:
    result = run_offline_sorting(
        _signal(),
        DetectionConfig(
            alignment_search_radius=1,
            pre_samples=2,
            post_samples=2,
            threshold_multiplier=3.0,
            refractory_interval=0.004,
        ),
        radius=1e-12,
        min_cluster_size=1,
        noise_label=-1,
        measurements=("amplitude", "width"),
        retain_waveforms=True,
    )

    assert isinstance(result, OfflineSortingResult)
    assert result.waveforms.n_spikes == 3
    assert result.features.data.shape == (3, 2)
    np.testing.assert_array_equal(result.clustering.labels, [0, 1, 2])
    np.testing.assert_array_equal(result.curation.updated_labels, [0, 1, 2])
    assert result.spike_train.units == ["0", "1", "2"]
    assert result.spike_train.waveforms is not None
    np.testing.assert_array_equal(
        np.concatenate(result.spike_train.times),
        result.waveforms.times,
    )
    assert result.spike_train.attrs["source_clock"] is result.waveforms.clock
    assert result.spike_train.attrs["source_stream"] == "wideband"


def _two_identical_spike_signal() -> SignalArray:
    # Two spikes of identical amplitude on the repeating [-1, 0, 1, 0] baseline,
    # placed at sample indices that are congruent mod 4 (20 and 40) so the
    # extracted pre/post windows -- and therefore the (amplitude, width)
    # features -- are exactly identical. Two identical features with distinct
    # seed labels form a synchronous-update 2-cycle that never converges.
    base = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 28)
    data = base[:, np.newaxis].astype(np.float64)
    data[20] = -12.0
    data[40] = -12.0
    return SignalArray.from_array(
        data,
        fs=1_000.0,
        t0=2.0,
        channel_names=["c0"],
        channel_types="spike",
        units="uV",
        name="wideband",
        clock=Clock("acquisition", "device", synchronization_domain="session"),
    )


_DETECTION = DetectionConfig(
    alignment_search_radius=1,
    pre_samples=2,
    post_samples=2,
    threshold_multiplier=3.0,
    refractory_interval=0.004,
)


def test_workflow_does_not_finalize_nonconverged_clustering() -> None:
    # Two identical-feature spikes with distinct default seeds form a 2-cycle
    # whose latest labels depend on max_iterations parity and never stabilize.
    # The high-level workflow must refuse to wrap that into a final SpikeTrain.
    with pytest.raises(ValidationError, match="did not converge"):
        run_offline_sorting(
            _two_identical_spike_signal(),
            _DETECTION,
            radius=1.0,
            min_cluster_size=1,
            noise_label=-1,
            max_iterations=2,
        )


def test_workflow_default_seeds_never_collide_with_noise_label() -> None:
    # With default seeds the per-event labels would be ``[0, 1, 2]``; if the
    # caller's noise_label is 0, cluster 0 would be silently treated as noise
    # and vanish from the SpikeTrain with no curation record. The default seeds
    # must shift past noise_label so every detected spike survives unless an
    # explicit, audited curation operation relabels it.
    result = run_offline_sorting(
        _signal(),
        DetectionConfig(
            alignment_search_radius=1,
            pre_samples=2,
            post_samples=2,
            threshold_multiplier=3.0,
            refractory_interval=0.004,
        ),
        radius=1e-12,
        min_cluster_size=1,
        noise_label=0,
        measurements=("amplitude", "width"),
    )

    assert 0 not in result.clustering.label_order
    assert result.spike_train.attrs["noise_event_indices"].size == 0
    assert sum(len(times) for times in result.spike_train.times) == result.waveforms.n_spikes


def _no_spike_signal() -> SignalArray:
    # The repeating [-1, 0, 1, 0] baseline stays within the threshold band, so
    # nothing crosses and detection returns zero batches.
    base = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 28)
    data = base[:, np.newaxis].astype(np.float64)
    return SignalArray.from_array(
        data,
        fs=1_000.0,
        t0=2.0,
        channel_names=["c0"],
        channel_types="spike",
        units="uV",
        name="wideband",
        clock=Clock("acquisition", "device", synchronization_domain="session"),
    )


def test_workflow_empty_detection_yields_empty_result() -> None:
    # A signal with no threshold crossings yields one batch with zero spikes;
    # the workflow finalizes that to an empty, converged result rather than
    # raising or emitting a degenerate non-empty SpikeTrain.
    result = run_offline_sorting(
        _no_spike_signal(),
        _DETECTION,
        radius=1.0,
        min_cluster_size=1,
        noise_label=-1,
    )

    assert isinstance(result, OfflineSortingResult)
    assert result.waveforms.n_spikes == 0
    assert result.clustering.converged
    assert result.clustering.termination == "empty"
    assert result.spike_train.units == []
    assert result.spike_train.times == []
    assert result.features.data.shape == (0, 2)


def test_workflow_chunked_detection_matches_whole_signal() -> None:
    # Chunked detection must reproduce the whole-signal outcome through every
    # downstream stage: clustering labels, curation, and final spike times.
    whole = run_offline_sorting(
        _signal(),
        _DETECTION,
        radius=1e-12,
        min_cluster_size=1,
        noise_label=-1,
        chunk_size=None,
    )
    chunked = run_offline_sorting(
        _signal(),
        _DETECTION,
        radius=1e-12,
        min_cluster_size=1,
        noise_label=-1,
        chunk_size=32,
    )

    np.testing.assert_array_equal(chunked.clustering.labels, whole.clustering.labels)
    np.testing.assert_array_equal(chunked.curation.updated_labels, whole.curation.updated_labels)
    assert chunked.spike_train.units == whole.spike_train.units
    np.testing.assert_array_equal(
        np.concatenate(chunked.spike_train.times),
        np.concatenate(whole.spike_train.times),
    )


def test_workflow_rejects_explicit_cuda_without_cpu_fallback() -> None:
    with runtime_context(device="cuda"):
        with pytest.raises(DeviceUnavailableError, match=r"sorting\.detection"):
            run_offline_sorting(
                _signal(),
                _DETECTION,
                radius=1.0,
                min_cluster_size=1,
                noise_label=-1,
            )


def test_workflow_fits_explicit_projector_and_finalizes() -> None:
    # The default workflow is measurement-only; an explicit WaveformProjector
    # must be fitted and applied inside the workflow and still converge through
    # curation into a typed SpikeTrain.
    result = run_offline_sorting(
        _signal(),
        _DETECTION,
        radius=1.0,
        min_cluster_size=1,
        noise_label=-1,
        projection=WaveformProjector(PCA(2)),
        measurements=("amplitude", "width"),
    )

    assert isinstance(result, OfflineSortingResult)
    assert result.clustering.converged
    # Two PCA columns plus amplitude and width.
    assert result.features.data.shape == (result.waveforms.n_spikes, 4)
    assert result.features.feature_names[:2] == ["pca_000", "pca_001"]
    assert sum(len(times) for times in result.spike_train.times) == result.waveforms.n_spikes


def test_sorting_namespace_exposes_workflow_not_binding() -> None:
    import neurale.sorting as sorting

    assert sorting.valley_seeking is valley_seeking
    assert sorting.run_offline_sorting is run_offline_sorting
    assert "ValleySeekingResult" in sorting.__all__
    assert "OfflineSortingResult" in sorting.__all__
    assert not hasattr(sorting, "_valley_seeking")
