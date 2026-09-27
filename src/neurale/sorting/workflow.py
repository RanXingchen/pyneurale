#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Small, explicit offline spike-sorting workflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from neurale.data import FeatureMatrix, SignalArray, SpikeTrain, SpikeWaveformBatch
from neurale.exceptions import ValidationError

from .clustering import ValleySeekingResult, valley_seeking
from .curation import CurationResult, remove_tiny_clusters
from .detection import DetectionConfig
from .features import WaveformMeasurement, WaveformProjector, project_waveform_features
from .offline import detect_threshold_spikes


@dataclass(frozen=True, slots=True)
class OfflineSortingResult:
    """Diagnostic aggregate retaining every typed workflow stage."""

    waveforms: SpikeWaveformBatch
    features: FeatureMatrix
    clustering: ValleySeekingResult
    curation: CurationResult
    spike_train: SpikeTrain

    def __post_init__(self) -> None:
        if not isinstance(self.waveforms, SpikeWaveformBatch):
            raise ValidationError("workflow waveforms must be a SpikeWaveformBatch.")
        if not isinstance(self.features, FeatureMatrix):
            raise ValidationError("workflow features must be a FeatureMatrix.")
        if not isinstance(self.clustering, ValleySeekingResult):
            raise ValidationError("workflow clustering must be a ValleySeekingResult.")
        if not isinstance(self.curation, CurationResult):
            raise ValidationError("workflow curation must be a CurationResult.")
        if not isinstance(self.spike_train, SpikeTrain):
            raise ValidationError("workflow spike_train must be a SpikeTrain.")
        n_events = self.waveforms.n_spikes
        if self.features.data.shape[0] != n_events:
            raise ValidationError("workflow features must remain aligned to detected spikes.")
        if self.clustering.labels.size != n_events:
            raise ValidationError("workflow clustering labels must remain event-aligned.")
        if self.curation.updated_labels.size != n_events:
            raise ValidationError("workflow curation labels must remain event-aligned.")


def waveform_batch_to_spike_train(
    batch: SpikeWaveformBatch,
    labels: np.ndarray,
    *,
    noise_label: int,
    retain_waveforms: bool = False,
) -> SpikeTrain:
    """Group one continuous waveform batch into unit-oriented spike trains.

    Unit order is ascending numeric cluster-label order. The explicit noise
    label is excluded but never renumbered. Exact numeric labels and all
    event-level source linkage remain in ``SpikeTrain.attrs``. Waveform payload
    retention is opt-in because it can be large.
    """

    if not isinstance(batch, SpikeWaveformBatch):
        raise ValidationError("batch must be a SpikeWaveformBatch.")
    values = _event_labels(labels, batch.n_spikes)
    noise = _integer_label(noise_label, "noise_label")
    if not isinstance(retain_waveforms, bool):
        raise ValidationError("retain_waveforms must be a bool.")

    unit_labels = tuple(int(label) for label in np.unique(values) if int(label) != noise)
    event_indices = tuple(
        np.flatnonzero(values == label).astype(np.int64, copy=False) for label in unit_labels
    )
    times = [np.array(batch.times[indices], copy=True) for indices in event_indices]
    waveforms = (
        [np.array(batch.waveforms[indices], copy=True) for indices in event_indices]
        if retain_waveforms
        else None
    )
    noise_indices = np.flatnonzero(values == noise).astype(np.int64, copy=False)
    attrs = {
        "cluster_labels": np.asarray(unit_labels, dtype=np.int64),
        "noise_label": noise,
        "source_event_indices": event_indices,
        "source_sample_indices": tuple(
            np.array(batch.sample_indices[indices], copy=True) for indices in event_indices
        ),
        "source_peak_channel_indices": tuple(
            np.array(batch.peak_channel_indices[indices], copy=True) for indices in event_indices
        ),
        "noise_event_indices": np.array(noise_indices, copy=True),
        "source_clock": batch.clock,
        "source_channels": batch.channels,
        "source_stream": batch.source_stream,
        "source_segment_id": batch.segment_id,
        "source_fs": batch.fs,
        "source_pre_samples": batch.pre_samples,
        "source_post_samples": batch.post_samples,
        "source_polarity": batch.polarity,
        "waveforms_retained": retain_waveforms,
    }
    return SpikeTrain(
        times=times,
        units=[str(label) for label in unit_labels],
        channels=None,
        waveforms=waveforms,
        fs=batch.fs if retain_waveforms else None,
        attrs=attrs,
    )


def run_offline_sorting(
    signal: SignalArray,
    detection: DetectionConfig,
    *,
    radius: float,
    min_cluster_size: int,
    noise_label: int,
    projection: WaveformProjector | None = None,
    fit_projection: bool = True,
    measurements: Sequence[WaveformMeasurement] = ("amplitude", "width"),
    initial_labels: np.ndarray | None = None,
    max_iterations: int = 1000,
    chunk_size: int | None = None,
    sample_idx_offset: int = 0,
    retain_waveforms: bool = False,
) -> OfflineSortingResult:
    """Run the minimal single-segment offline sorting workflow.

    The stages are detection, typed waveform projection, native CPU Valley
    Seeking, explicit tiny-cluster curation, and typed ``SpikeTrain``
    conversion. Every intermediate result is returned for diagnostics.

    This convenience entry point intentionally handles one continuous signal
    only. Gap-aware or multi-segment callers compose ``detect_threshold_spikes``
    and the remaining public functions once per returned batch. If
    ``initial_labels`` is omitted, Valley Seeking uses one seed label per
    detected event. The default seeds are shifted to avoid ``noise_label`` so
    that no cluster is silently treated as noise before curation runs.
    ``fit_projection`` selects the existing ``WaveformProjector`` fit/transform
    contract. ``noise_label`` becomes special only during curation and final
    conversion; clustering treats all labels as nominal.

    The workflow refuses to finalize a clustering that did not converge
    (``converged is False``): the latest labels are a bounded but not-yet-stable
    iterate whose value can depend on ``max_iterations``. Call ``valley_seeking``
    directly to inspect such iterates.
    """

    noise = _integer_label(noise_label, "noise_label")
    batches = detect_threshold_spikes(
        signal,
        detection,
        chunk_size=chunk_size,
        sample_idx_offset=sample_idx_offset,
    )
    if len(batches) != 1:
        raise ValidationError("run_offline_sorting requires exactly one continuous segment.")
    batch = batches[0]
    features = project_waveform_features(
        batch,
        projection,
        fit=fit_projection,
        measurements=measurements,
    )
    resolved_initial_labels = (
        _default_seed_labels(batch.n_spikes, noise) if initial_labels is None else initial_labels
    )
    clustered = valley_seeking(
        features,
        resolved_initial_labels,
        radius=radius,
        max_iterations=max_iterations,
    )
    if not clustered.converged:
        raise ValidationError(
            "Valley Seeking did not converge within max_iterations; "
            "call valley_seeking directly to inspect the bounded latest labels."
        )
    curated = remove_tiny_clusters(
        clustered.labels,
        min_cluster_size=min_cluster_size,
        noise_label=noise,
    )
    spike_train = waveform_batch_to_spike_train(
        batch,
        curated.updated_labels,
        noise_label=noise,
        retain_waveforms=retain_waveforms,
    )
    return OfflineSortingResult(
        waveforms=batch,
        features=features,
        clustering=clustered,
        curation=curated,
        spike_train=spike_train,
    )


def _event_labels(value: object, n_events: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValidationError("labels must be a numpy.ndarray with dtype int64.")
    labels = np.asarray(value)
    if labels.ndim != 1 or labels.dtype != np.dtype(np.int64):
        raise ValidationError("labels must be 1D int64.")
    if labels.size != n_events:
        raise ValidationError("labels length must match batch.n_spikes.")
    return labels


def _default_seed_labels(n_events: int, noise_label: int) -> np.ndarray:
    # One seed label per event (``[0, 1, ..., n_events - 1]``) but shifted past
    # ``noise_label`` when it would otherwise collide, so curation never sees a
    # cluster that is silently equal to the explicit noise label.
    labels = np.arange(n_events, dtype=np.int64)
    if 0 <= noise_label < n_events:
        labels[noise_label:] += 1
    return labels


def _integer_label(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValidationError(f"{name} must be an integer label.")
    resolved = int(value)
    if resolved < np.iinfo(np.int64).min or resolved > np.iinfo(np.int64).max:
        raise ValidationError(f"{name} must fit int64.")
    return resolved
