#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Typed configuration for threshold spike detection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from neurale._validation import validate_number
from neurale.exceptions import ValidationError

NoiseEstimator = Literal["mad"]
DetectionPolarity = Literal["negative", "positive", "both"]
BoundaryBehavior = Literal["drop", "raise"]
OverflowPolicy = Literal["fault", "drop_newest"]
ElectrodeGroups = Sequence[Sequence[int]] | None
NormalizedElectrodeGroups = tuple[tuple[int, ...], ...] | None


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Describe the mathematical threshold-detection contract.

    ``alignment_search_radius``, ``pre_samples``, and ``post_samples`` are
    sample counts. ``refractory_interval`` is measured in seconds and becomes
    ``ceil(refractory_interval * fs)`` samples for a particular
    input signal.

    The only currently approved noise estimator is ``"mad"``. For each
    channel, the reference oracle computes the channel median ``m`` and
    ``noise = median(abs(x - m)) / 0.6744897501960817``. A channel with zero
    noise produces no detections. The positive detection threshold is
    ``threshold_multiplier * noise``; ``polarity`` determines which signed
    excursions are eligible.

    ``electrode_groups`` is either ``None`` (one independent group per input
    channel) or an ordered partition of zero-based positions in the input
    ``ChannelTable``. Complete coverage and bounds are validated when a signal
    is supplied. Group order defines the stable integer group identifier.

    Waveforms always include the aligned sample, so their length is
    ``pre_samples + 1 + post_samples``. If an aligned waveform crosses an input
    boundary, ``"drop"`` discards that event and ``"raise"`` raises
    ``ValidationError``. The reference contract never pads or clips waveforms.
    """

    alignment_search_radius: int
    pre_samples: int
    post_samples: int
    noise_estimator: NoiseEstimator = "mad"
    threshold_multiplier: float = 3.5
    polarity: DetectionPolarity = "negative"
    refractory_interval: float = 1.5e-3
    electrode_groups: ElectrodeGroups = None
    boundary_behavior: BoundaryBehavior = "drop"

    def __post_init__(self) -> None:
        alignment_radius = _non_negative_integer(
            self.alignment_search_radius,
            "alignment_search_radius",
        )
        pre_samples = _non_negative_integer(self.pre_samples, "pre_samples")
        post_samples = _non_negative_integer(self.post_samples, "post_samples")
        if self.noise_estimator != "mad":
            raise ValidationError("noise_estimator must be 'mad'.")
        threshold_multiplier = float(
            validate_number(
                self.threshold_multiplier,
                "threshold_multiplier",
                kind="real",
                minimum=0.0,
                minimum_inclusive=False,
            )
        )
        if self.polarity not in ("negative", "positive", "both"):
            raise ValidationError("polarity must be one of: negative, positive, both.")
        refractory_interval = float(
            validate_number(
                self.refractory_interval,
                "refractory_interval",
                kind="real",
                minimum=0.0,
            )
        )
        groups = _normalize_electrode_groups(self.electrode_groups)
        if self.boundary_behavior not in ("drop", "raise"):
            raise ValidationError("boundary_behavior must be 'drop' or 'raise'.")

        object.__setattr__(self, "alignment_search_radius", alignment_radius)
        object.__setattr__(self, "pre_samples", pre_samples)
        object.__setattr__(self, "post_samples", post_samples)
        object.__setattr__(self, "threshold_multiplier", threshold_multiplier)
        object.__setattr__(self, "refractory_interval", refractory_interval)
        object.__setattr__(self, "electrode_groups", groups)


def _non_negative_integer(value: object, name: str) -> int:
    return int(validate_number(value, name, kind="integer", minimum=0))


def _normalize_electrode_groups(value: ElectrodeGroups) -> NormalizedElectrodeGroups:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValidationError("electrode_groups must be an ordered sequence of channel groups.")
    try:
        raw_groups = tuple(value)
    except TypeError as exc:
        raise ValidationError(
            "electrode_groups must be an ordered sequence of channel groups."
        ) from exc
    if not raw_groups:
        raise ValidationError("electrode_groups must not be empty.")

    groups: list[tuple[int, ...]] = []
    claimed: set[int] = set()
    for group_idx, raw_group in enumerate(raw_groups):
        if isinstance(raw_group, (str, bytes)):
            raise ValidationError(f"electrode_groups[{group_idx}] must contain channel positions.")
        try:
            values = tuple(raw_group)
        except TypeError as exc:
            raise ValidationError(
                f"electrode_groups[{group_idx}] must contain channel positions."
            ) from exc
        if not values:
            raise ValidationError("electrode groups must not be empty.")
        group = tuple(
            _non_negative_integer(channel, f"electrode_groups[{group_idx}] channel")
            for channel in values
        )
        if len(set(group)) != len(group):
            raise ValidationError("an electrode group cannot contain duplicate channels.")
        duplicate = claimed.intersection(group)
        if duplicate:
            raise ValidationError("a channel cannot belong to multiple electrode groups.")
        claimed.update(group)
        groups.append(group)
    return tuple(groups)


@dataclass(frozen=True, slots=True)
class OnlineDetectorDescriptor:
    """Immutable provenance for a native online threshold detector.

    A native fixed-capacity spike block carries per-spike ``electrode_group_ids``
    but not the group *definition*: group 0 alone cannot tell a consumer which
    channels belong to group 0. The offline :class:`SpikeWaveformBatch` contract
    requires the resolved electrode-group partition to travel with the batch as
    self-contained metadata, so ``spike_block_to_waveform_batch`` requires this
    descriptor. It also carries the detection parameters the payload omits
    (channel centers/thresholds, refractory window, alignment radius, boundary
    and overflow policy) so the converted offline object retains detection
    provenance without bundling variable-length state into the realtime payload.

    ``electrode_groups`` is required and normalized to a tuple of tuples of
    non-negative integers with no duplicate channels; the complete-partition
    check against a :class:`ChannelTable` runs at conversion time, mirroring
    :class:`DetectionConfig`.
    """

    electrode_groups: Sequence[Sequence[int]]
    channel_centers: Sequence[float] | None = None
    channel_thresholds: Sequence[float] | None = None
    refractory_samples: int | None = None
    alignment_search_radius: int | None = None
    boundary_behavior: BoundaryBehavior | None = None
    overflow_policy: OverflowPolicy | None = None

    def __post_init__(self) -> None:
        groups = _normalize_electrode_groups(self.electrode_groups)
        if groups is None:
            raise ValidationError("OnlineDetectorDescriptor.electrode_groups must be provided.")
        object.__setattr__(self, "electrode_groups", groups)
        if self.channel_centers is not None:
            centers = tuple(
                float(validate_number(value, "channel_centers", kind="real"))
                for value in self.channel_centers
            )
            object.__setattr__(self, "channel_centers", centers)
        if self.channel_thresholds is not None:
            thresholds = tuple(
                float(validate_number(value, "channel_thresholds", kind="real", minimum=0.0))
                for value in self.channel_thresholds
            )
            object.__setattr__(self, "channel_thresholds", thresholds)
        if (
            self.channel_centers is not None
            and self.channel_thresholds is not None
            and len(self.channel_centers) != len(self.channel_thresholds)
        ):
            raise ValidationError("channel_centers and channel_thresholds must have equal length.")
        if self.refractory_samples is not None:
            object.__setattr__(
                self,
                "refractory_samples",
                _non_negative_integer(self.refractory_samples, "refractory_samples"),
            )
        if self.alignment_search_radius is not None:
            object.__setattr__(
                self,
                "alignment_search_radius",
                _non_negative_integer(self.alignment_search_radius, "alignment_search_radius"),
            )
        if self.boundary_behavior is not None and self.boundary_behavior not in (
            "drop",
            "raise",
        ):
            raise ValidationError("boundary_behavior must be 'drop' or 'raise'.")
        if self.overflow_policy is not None and self.overflow_policy not in (
            "fault",
            "drop_newest",
        ):
            raise ValidationError("overflow_policy must be 'fault' or 'drop_newest'.")
