#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Feature extraction and preprocessing APIs."""

from .bandpower import (
    bandpower_features,
    hilbert_envelope_features,
    lmp_features,
)
from .erp import ERPEpochs, ERPWaveform, average_erp, erp_epochs
from .kinematics import (
    KinematicSeries,
    acceleration,
    as_kinematic_series,
    direction,
    kinematic_derivative,
    kinematic_trials,
    speed,
    velocity,
)
from .online import (
    BandpowerProcessor,
    HilbertEnvelopeProcessor,
    LmpProcessor,
)
from .oscillations import OscillationDetectionResult, OscillationScore, detect_oscillations
from .preprocessing import (
    NeuralPreprocessor,
    offline_preprocess,
    select_good_channels,
)
from .tuning import DirectionalTuningResult, TuningResult, binned_tuning, directional_tuning

__all__ = [
    "BandpowerProcessor",
    "DirectionalTuningResult",
    "ERPEpochs",
    "ERPWaveform",
    "HilbertEnvelopeProcessor",
    "KinematicSeries",
    "LmpProcessor",
    "NeuralPreprocessor",
    "OscillationDetectionResult",
    "OscillationScore",
    "TuningResult",
    "acceleration",
    "as_kinematic_series",
    "average_erp",
    "bandpower_features",
    "binned_tuning",
    "detect_oscillations",
    "direction",
    "directional_tuning",
    "erp_epochs",
    "hilbert_envelope_features",
    "kinematic_derivative",
    "kinematic_trials",
    "lmp_features",
    "offline_preprocess",
    "select_good_channels",
    "speed",
    "velocity",
]
