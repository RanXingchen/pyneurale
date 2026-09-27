#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Run a native Center-Out closed loop starting with 100% assistance.

The first trial is fully computer guided. Its aligned native feature/intent
pairs fit a linear decoder off the realtime data plane. The fitted decoder is
then activated at the next trial boundary, where assistance drops to 90%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import neurale.pipeline as pipeline
import neurale.streaming as streaming
from neurale.decoding import KalmanDecoder
from neurale.devices.simulation import SimulatedNeuralDevice
from neurale.experiments import center_out as co
from neurale.signal.simulation import SignalGenerator

ASSISTANCE_SCHEDULE = (
    co.AssistanceBlock(1.0, 5),
    co.AssistanceBlock(0.5, 5),
    co.AssistanceBlock(0.0, 5),
)

SEED = 42


def build_task() -> object:
    """Build one repeatable outward-target Center-Out task."""

    return co.CenterOutTask(
        geometry_unit=co.GeometryUnit.NORMALIZED,
        layout=co.build_radial_layout(co.RadialLayoutRequest(radius=0.8)),
        acceptance=0.08,
        cursor_extent=0.05,
        movement_timeout_seconds=3.0,
        selection=co.TargetSelectionPolicy.BALANCED_SHUFFLED_CYCLES,
        seed=SEED,
    )


def build_feature_plan() -> pipeline.PipelinePlan:
    """Describe the native feature chain compiled by the session."""

    chs_remove = pipeline.BadChannelRemovalStage([1, 5, 10])
    car = pipeline.SpatialReferenceStage(statistic="mean")
    notch = pipeline.LineNoiseFilterStage()
    filt = pipeline.FilterStage(
        filter_type="bandpass",
        cutoff_hz=(1.0, 500.0),
    )
    feat = pipeline.FeatureStage(
        window_seconds=0.10,
        update_interval_seconds=0.05,
        features=(
            pipeline.MultitaperBandpowerFeature(
                bands=(pipeline.Band("low-freq", 1.0, 30.0),),
                time_bandwidth=2.5,
            ),
            pipeline.HilbertEnvelopeFeature(
                bands=(pipeline.Band("high-gamma", 170.0, 200.0),),
                filter_order=2,
            ),
            pipeline.LmpFeature(cutoff_hz=4.0, filter_order=2),
        ),
    )

    return pipeline.PipelinePlan(stages=(chs_remove, car, notch, filt, feat))


def build_presentation_config() -> object:
    """Build the optional concrete native Center-Out presentation."""

    from neurale.experiments.presentation import CenterOutPresentationConfig

    return CenterOutPresentationConfig(
        title="PyNeurale Center-Out closed loop", monitor=1, fullscreen=True
    )


def build_recorder(output: Path, device: SimulatedNeuralDevice) -> object:
    """Create the optional unified native session recorder."""

    from neurale.recording import RecorderConfig, SessionRecorder

    return SessionRecorder.create(RecorderConfig(path=output), device)


def run(*, output: Path | None, show: bool) -> None:
    """Run acquisition, native decoding/control, online fitting, and return."""

    np.random.seed(SEED)

    # Create a sine generator
    nchs, fs = 256, 4000.0
    freqs = np.random.randint(1, int(fs / 2), size=(nchs,))
    amps = np.sort(np.abs(np.random.randn(nchs)), descending=True)
    phases = np.random.rand(len(freqs))

    device = SimulatedNeuralDevice(
        SignalGenerator.sine(nchs, fs, freqs, amp=amps, phase=phases),
        samples_per_frame=4,  # 1 ms per frame
        physical_unit="dimensionless",
        paced=True,
    )

    recorder = None if output is None else build_recorder(output, device)

    session = co.CenterOutSession(
        co.CenterOutProtocol(
            task=build_task(),
            assistance_blocks=ASSISTANCE_SCHEDULE,
        ),
        build_feature_plan(),
        device,
        KalmanDecoder(jitter=1e-6),
        presentation_config=build_presentation_config() if show else None,
        recording=recorder,
    )
    outcome = session.run()

    if outcome.runtime_status != streaming.StreamStatus.OK:
        raise RuntimeError(f"runtime failed: {outcome.runtime_status}")
    if not outcome.experiment_trace_complete or outcome.trace_losses:
        raise RuntimeError("Center-Out experiment trace is incomplete")
    if outcome.completed_trials != sum(block.trials for block in ASSISTANCE_SCHEDULE):
        raise RuntimeError("the configured assistance trials did not complete")
    if not outcome.publications or outcome.active_decoder_version < 2:
        raise RuntimeError("the fitted decoder was not activated online")

    print(
        json.dumps(
            {
                "runtime_status": str(outcome.runtime_status),
                "completed_trials": outcome.completed_trials,
                "assistance": [block.assistance for block in ASSISTANCE_SCHEDULE],
                "published_versions": [item.version for item in outcome.publications],
                "active_decoder_version": outcome.active_decoder_version,
                "training_capture_drops": outcome.training_capture_drops,
                "presented_frames": outcome.presented_frames,
                "recording_complete": (
                    None if outcome.recording is None else outcome.recording.complete
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional output NRF path")
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the optional native OpenGL Center-Out presentation window",
    )
    arguments = parser.parse_args()
    run(output=arguments.output, show=arguments.show)
