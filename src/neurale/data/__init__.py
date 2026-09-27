#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Public data model API for PyNeurale.

Deep-frozen metadata fields use a closed value domain: ``None``, Python
``bool``/``int``/``float``/``str``/``bytes``, NumPy scalars, recursively frozen
mappings/sequences/sets, ndarrays, and the explicitly admitted immutable
``ChannelInfo``, ``ChannelTable``, and ``Clock`` values. Non-object ndarrays
become immutable bytes-backed copies; object ndarrays are converted through
``tolist()`` and recursively frozen as containers. Unsupported leaves raise
``ValidationError``. See the user-guide metadata contract for details.

This contract applies to fields documented as deep-frozen. Mutable aggregate
containers such as ``Recording.metadata`` retain their separately documented
copy semantics.
"""

from .alignment import (
    AlignmentIndex,
    TrialEpoch,
    align_neural_behavior_nearest,
    convert_time,
    events_in_interval,
    extract_signal_epoch,
    extract_trial_epoch,
    from_reference_time,
    split_recording_trials,
    to_reference_time,
)
from .arrays import FeatureMatrix, SignalArray, SpikeTrain, SpikeWaveformBatch
from .channels import ChannelInfo, ChannelTable, ElectrodeArray
from .events import Event, EventSeries, Trial, TrialTable
from .recording import Recording
from .streaming import Frame
from .time import Clock

__all__ = [
    "AlignmentIndex",
    "ChannelInfo",
    "ChannelTable",
    "Clock",
    "ElectrodeArray",
    "Event",
    "EventSeries",
    "FeatureMatrix",
    "Frame",
    "Recording",
    "SignalArray",
    "SpikeTrain",
    "SpikeWaveformBatch",
    "Trial",
    "TrialEpoch",
    "TrialTable",
    "align_neural_behavior_nearest",
    "convert_time",
    "events_in_interval",
    "extract_signal_epoch",
    "extract_trial_epoch",
    "from_reference_time",
    "split_recording_trials",
    "to_reference_time",
]
