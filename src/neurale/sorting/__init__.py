#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Typed offline spike detection and sorting contracts."""

from .clustering import ValleySeekingResult, valley_seeking
from .curation import (
    CurationOperationRecord,
    CurationResult,
    WaveformOutlierResult,
    merge_clusters,
    reject_waveform_outliers,
    remove_tiny_clusters,
    split_cluster,
)
from .detection import DetectionConfig, OnlineDetectorDescriptor
from .features import WaveformProjector, project_waveform_features
from .metrics import ClusterIsiMetrics, MatchingEventsResult, isi_metrics, matching_events
from .offline import detect_threshold_spikes
from .online import spike_block_to_waveform_batch
from .workflow import OfflineSortingResult, run_offline_sorting, waveform_batch_to_spike_train

__all__ = [
    "ClusterIsiMetrics",
    "CurationOperationRecord",
    "CurationResult",
    "DetectionConfig",
    "MatchingEventsResult",
    "OfflineSortingResult",
    "OnlineDetectorDescriptor",
    "ValleySeekingResult",
    "WaveformOutlierResult",
    "WaveformProjector",
    "detect_threshold_spikes",
    "isi_metrics",
    "matching_events",
    "merge_clusters",
    "project_waveform_features",
    "reject_waveform_outliers",
    "remove_tiny_clusters",
    "run_offline_sorting",
    "spike_block_to_waveform_batch",
    "split_cluster",
    "valley_seeking",
    "waveform_batch_to_spike_train",
]
