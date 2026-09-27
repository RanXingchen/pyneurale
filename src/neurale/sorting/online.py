#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Control-plane conversion for native fixed-capacity spike blocks."""

from __future__ import annotations

from typing import Any

import numpy as np

from neurale._native_loader import load_native_namespace
from neurale.data import ChannelTable, Clock, SpikeWaveformBatch
from neurale.exceptions import ValidationError
from neurale.sorting.detection import OnlineDetectorDescriptor


def _decode_native_block(payload: object) -> dict[str, Any]:
    native = load_native_namespace("sorting.online_detection")
    return native.decode_spike_block(payload)


def _validate_electrode_groups_partition(
    groups: tuple[tuple[int, ...], ...],
    n_channels: int,
) -> None:
    # ``_normalize_electrode_groups`` already rejects empty groups, duplicate
    # channels, and negative positions, so only the complete-partition and
    # upper-bound checks remain here: every channel position in the
    # ``ChannelTable`` must be covered exactly once.
    flattened = [channel for group in groups for channel in group]
    if any(channel < 0 or channel >= n_channels for channel in flattened):
        raise ValidationError(
            "electrode_groups contains a channel position outside the ChannelTable."
        )
    if sorted(flattened) != list(range(n_channels)):
        raise ValidationError("electrode_groups must cover every channel exactly once.")


def _validate_group_ids_defined(group_ids: np.ndarray, n_groups: int) -> None:
    if group_ids.size == 0:
        return
    if int(group_ids.min()) < 0 or int(group_ids.max()) >= n_groups:
        raise ValidationError("block contains an electrode_group_id with no defined group.")


def spike_block_to_waveform_batch(
    payload: object,
    *,
    channels: ChannelTable,
    fs: float,
    clock: Clock | None,
    source_stream: str,
    descriptor: OnlineDetectorDescriptor,
) -> SpikeWaveformBatch:
    """Copy a native online block into an immutable offline batch.

    This conversion is a control-plane operation: it loads the native decoder,
    allocates owning NumPy arrays, and constructs :class:`SpikeWaveformBatch`.
    It must not be called from a native processor callback or other realtime
    critical path. Only the block's valid prefix is copied; overflow counters
    remain available in the returned immutable ``attrs`` metadata.

    ``descriptor`` is required: the native payload carries per-spike
    ``electrode_group_ids`` but not the group definition, so the resolved
    electrode-group partition (and the rest of the detection provenance) must
    be supplied by the caller and travels with the batch as ``attrs``.
    """

    if not isinstance(channels, ChannelTable):
        raise ValidationError("channels must be a ChannelTable.")
    if not isinstance(descriptor, OnlineDetectorDescriptor):
        raise ValidationError("descriptor must be an OnlineDetectorDescriptor.")
    decoded = _decode_native_block(payload)

    n_channels = len(channels)
    groups = descriptor.electrode_groups
    _validate_electrode_groups_partition(groups, n_channels)
    group_ids = np.asarray(decoded["electrode_group_ids"], dtype=np.int64)
    _validate_group_ids_defined(group_ids, len(groups))
    if descriptor.channel_centers is not None and len(descriptor.channel_centers) != n_channels:
        raise ValidationError("channel_centers must match the channel count.")
    if (
        descriptor.channel_thresholds is not None
        and len(descriptor.channel_thresholds) != n_channels
    ):
        raise ValidationError("channel_thresholds must match the channel count.")

    polarity = str(decoded["polarity"])
    per_spike = np.asarray(decoded["spike_polarities"], dtype=np.int8)
    attrs: dict[str, Any] = {
        "detection_method": "native_online_threshold",
        "scores": tuple(float(value) for value in np.asarray(decoded["scores"])),
        "block_capacity": int(decoded["capacity"]),
        "overflow_count": int(decoded["overflow_count"]),
        "overflowed": bool(decoded["overflowed"]),
        "electrode_groups": groups,
    }
    if descriptor.channel_centers is not None:
        attrs["channel_centers"] = np.asarray(descriptor.channel_centers, dtype=np.float64)
    if descriptor.channel_thresholds is not None:
        attrs["channel_thresholds"] = np.asarray(descriptor.channel_thresholds, dtype=np.float64)
    if descriptor.refractory_samples is not None:
        attrs["refractory_samples"] = descriptor.refractory_samples
    if descriptor.alignment_search_radius is not None:
        attrs["alignment_search_radius"] = descriptor.alignment_search_radius
    if descriptor.boundary_behavior is not None:
        attrs["boundary_behavior"] = descriptor.boundary_behavior
    if descriptor.overflow_policy is not None:
        attrs["overflow_policy"] = descriptor.overflow_policy

    return SpikeWaveformBatch(
        waveforms=np.asarray(decoded["waveforms"]),
        sample_indices=np.asarray(decoded["sample_indices"], dtype=np.int64),
        times=np.asarray(decoded["times"], dtype=np.float64),
        peak_channel_indices=np.asarray(decoded["peak_channel_indices"], dtype=np.int64),
        electrode_group_ids=group_ids,
        amps=np.asarray(decoded["amps"]),
        channels=channels,
        fs=fs,
        clock=clock,
        source_stream=source_stream,
        segment_id=int(decoded["segment_id"]),
        pre_samples=int(decoded["pre_samples"]),
        post_samples=int(decoded["post_samples"]),
        polarity=polarity,
        spike_polarities=per_spike if polarity == "both" else None,
        attrs=attrs,
    )
