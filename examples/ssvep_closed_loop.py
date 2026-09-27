#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT
"""Four-target SSVEP: simulated EEG, calibration, native LDA feedback and NRF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurale import pipeline
from neurale.decoding import LDADecoder
from neurale.devices.simulation import SimulatedNeuralDevice
from neurale.experiments import ssvep
from neurale.experiments.presentation import SSVEPDisplayConfig
from neurale.models.preprocessing import StandardScaler
from neurale.recording import RecorderConfig, SessionRecorder
from neurale.signal.simulation import SignalGenerator

FREQUENCIES = (8.0, 10.0, 12.0, 15.0)


def build_task() -> ssvep.SSVEPTask:
    return ssvep.SSVEPTask(
        targets=[ssvep.SSVEPTarget(i + 1, f) for i, f in enumerate(FREQUENCIES)],
        cue_duration=0.75,
        stimulation_duration=2.5,
        decision_timeout=1.0,
        feedback_duration=0.5,
        inter_trial=0.5,
        seed=42,
    )


def build_feature_plan() -> pipeline.PipelinePlan:
    bands = tuple(
        pipeline.Band(
            f"target-{i + 1}-harmonic-{harmonic}", f * harmonic - 0.75, f * harmonic + 0.75
        )
        for i, f in enumerate(FREQUENCIES)
        for harmonic in (1, 2)
    )
    return pipeline.PipelinePlan(
        stages=(
            pipeline.LineNoiseFilterStage(harmonics=1),
            pipeline.FilterStage(filter_type="bandpass", cutoff_hz=(5.0, 40.0)),
            pipeline.FeatureStage(
                window_seconds=1.0,
                update_interval_seconds=0.25,
                features=(pipeline.MultitaperBandpowerFeature(bands=bands, time_bandwidth=1.5),),
            ),
        )
    )


def run(*, output: Path | None = None, show: bool = False, monitor: int | None = None) -> dict:
    # 256 Hz makes the 0.25-second update interval exactly 64 samples.
    # Fixed tones do not encode the cued target. Accuracy is not an acceptance criterion.
    device = SimulatedNeuralDevice(
        SignalGenerator.tones(4, 256.0, (8.3, 10.2, 12.4, 15.1), amps=(4e-6, 3e-6, 2e-6, 1e-6)),
        samples_per_frame=4,
        channel_names=("O1", "Oz", "O2", "POz"),
        paced=True,
    )
    recorder = (
        None if output is None else SessionRecorder.create(RecorderConfig(path=output), device)
    )
    try:
        session = ssvep.SSVEPSession(
            ssvep.SSVEPProtocol(build_task()),
            build_feature_plan(),
            device,
            LDADecoder(scaler=StandardScaler(), shrinkage="auto"),
            presentation_config=SSVEPDisplayConfig(fullscreen=True, monitor=monitor)
            if show
            else None,
            recording=recorder,
        )
        outcome = session.run()

        result = {
            "calibration_trials": len(outcome.calibration_trials),
            "evaluation_trials": len(outcome.evaluation_trials),
            "correct_count": outcome.correct_count,
            "decoder_version": outcome.decoder_version,
            "decoder_fingerprint": outcome.decoder_fingerprint,
            "experiment_trace_complete": outcome.experiment_trace_complete,
            "trace_drops": outcome.trace_drops,
            "runtime_status": outcome.runtime_status.name,
            "stopped": outcome.stopped,
            "frames": outcome.frames,
            "max_software_swap_gap_ms": outcome.max_frame_interval * 1000,
            "expired_presentation_frames": outcome.expired_presentation_frames,
            "recording": str(output) if output is not None else None,
        }
        print(json.dumps(result, indent=2))
        return result
    finally:
        device.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--monitor", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(output=args.output, show=args.show, monitor=args.monitor)
