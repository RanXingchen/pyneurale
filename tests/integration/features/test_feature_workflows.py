#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neurale.data import Clock, FeatureMatrix, Recording, SignalArray, Trial, TrialTable
from neurale.features import binned_tuning, erp_epochs, kinematic_trials

pytestmark = pytest.mark.python_only


def test_typed_trial_workflow_preserves_public_metadata() -> None:
    fs = 10.0
    neural_time = np.arange(20, dtype=float) / fs
    # Extend the half-sample-shifted behavior grid on both sides so both strict
    # trial intervals remain fully covered without clipping.
    behavior_time = np.arange(21, dtype=float) / fs - 0.05
    clock = Clock(
        "acquisition",
        "device",
        rate=fs,
        synchronization_domain="session-7",
        attrs={"device_id": "synthetic-1"},
    )
    trials = TrialTable(
        (
            Trial(10, 0.0, 1.0, label="outbound", attrs={"condition": "left"}),
            Trial(20, 1.0, 2.0, label="return", attrs={"condition": "right"}),
        ),
        attrs={"block_set": "integration"},
    )

    neural = SignalArray.from_array(
        np.column_stack((neural_time, -neural_time)),
        fs=fs,
        time=neural_time,
        channel_names=("motor-left", "motor-right"),
        channel_types="ecog",
        units="uV",
        name="neural",
        clock=clock,
        attrs={"reference": "common-average"},
        copy_data=True,
    )
    pos = SignalArray.from_array(
        np.column_stack((neural_time, 2.0 * neural_time)),
        fs=fs,
        time=neural_time,
        channel_names=("x", "y"),
        channel_types="behavior",
        units="m",
        name="position",
        clock=clock,
        attrs={"calibration": "camera-3"},
        copy_data=True,
    )
    behavior = SignalArray.from_array(
        np.where(behavior_time < 1.0, 0.25, 0.75),
        fs=fs,
        time=behavior_time,
        channel_names=("phase",),
        channel_types="behavior",
        units="cycle",
        name="behavior",
        clock=clock,
        attrs={"task": "reach"},
        copy_data=True,
    )
    features = FeatureMatrix(
        data=np.column_stack((1.0 + neural_time, 2.0 + 2.0 * neural_time)),
        fs=fs,
        time=neural_time,
        feature_names=("unit-a", "unit-b"),
        source_signal="neural",
        window_size=None,
        shift=None,
        unit="Hz",
        attrs={"extractor": "point-observation"},
    )
    recording = Recording(
        signals={"neural": neural, "position": pos, "behavior": behavior},
        features={"rates": features},
        trials=trials,
        metadata={"session": "S7"},
    )

    epochs = erp_epochs(
        recording,
        "trials",
        tmin=0.0,
        tmax=0.3,
        reference_clock=clock,
    )
    assert epochs.data.shape == (2, 3, 2)
    assert epochs.selections == tuple(trials)
    assert epochs.source_clock == clock
    assert tuple(epochs.channels.names) == ("motor-left", "motor-right")
    assert tuple(epochs.channels.units) == ("uV", "uV")
    assert all(attrs["reference"] == "common-average" for attrs in epochs.signal_attrs)
    assert all(metadata["session"] == "S7" for metadata in epochs.recording_metadata)

    trajectories = kinematic_trials(
        recording,
        signal_name="position",
        reference_clock=clock,
    )
    assert trajectories.data.shape == (2, 10, 2)
    assert trajectories.trials == tuple(trials)
    assert trajectories.trial_ids == (10, 20)
    assert trajectories.clock == clock
    assert trajectories.dimension_names == ("x", "y")
    assert trajectories.units == ("m", "m")
    assert trajectories.source_signal == "position"
    assert trajectories.source_attrs["calibration"] == "camera-3"
    assert trajectories.attrs["recording_metadata"]["session"] == "S7"

    tuning = binned_tuning(
        recording.features["rates"],
        recording.signals["behavior"],
        behavior_dim="phase",
        bin_edges=np.array([0.0, 0.5, 1.0]),
        alignment_tol=0.051,
        neural_clock=clock,
        trials=trials,
        reference_clock=clock,
    )
    # The half-sample offset is ambiguous at each trial boundary under one
    # global nearest match. Trial-local common coverage excludes one leading
    # neural row per trial and keeps the two occupancies balanced.
    np.testing.assert_array_equal(tuning.occupancy, [9, 9])
    assert tuning.trials == tuple(trials)
    assert tuning.trial_table_attrs["block_set"] == "integration"
    assert tuning.feature_names == ("unit-a", "unit-b")
    assert tuning.feature_units == ("Hz", "Hz")
    assert tuning.behavior_names == ("phase",)
    assert tuning.behavior_units == ("cycle",)
    assert tuning.source_name == "neural"
    assert tuning.source_attrs["extractor"] == "point-observation"
    assert tuning.behavior_attrs["task"] == "reach"
    assert tuning.neural_clock == clock
    assert tuning.behavior_clock == clock


def test_namespace_boundary_uses_installed_public_api(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    script = """
import importlib.util
import sys

assert importlib.util.find_spec("neurale.analysis") is None
assert importlib.util.find_spec("neurale.metrics") is None

from neurale.features import binned_tuning, detect_oscillations, kinematic_trials
from neurale.models import accuracy_score

assert callable(detect_oscillations)
assert callable(kinematic_trials)
assert callable(binned_tuning)
assert callable(accuracy_score)
assert "neurale._native" not in sys.modules
assert "neurale._native_cuda" not in sys.modules
assert not any(name.startswith("_neurale") for name in sys.modules)
"""

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=tmp_path,
        env=environment,
    )
